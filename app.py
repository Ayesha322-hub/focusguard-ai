import streamlit as st
import cv2
import numpy as np
import time
import json
import os
import sys
import glob
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from focusguard.core.vision_engine import VisionEngine
from focusguard.models.eye_state import EyeStateDetector
from focusguard.models.yawn_detector import YawnDetector, SoundAlarm
from focusguard.models.phone_detector import PhoneDetector
from focusguard.models.head_pose import HeadPoseDetector
from focusguard.models.object_distraction import DistractionObjectDetector

st.set_page_config(
    page_title='FocusGuard AI',
    page_icon='🛡️',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.markdown('''
<style>
    .main-title {
        font-size: 2.5rem;
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
        text-align: center;
    }
    .subtitle { text-align: center; color: #888; margin-bottom: 2rem; }
    .metric-card {
        background: #1e1e1e; padding: 1rem; border-radius: 10px;
        border-left: 4px solid #00c6ff; margin: 0.5rem 0;
    }
    .alert-box {
        background: #ff4b4b; color: white; padding: 0.8rem;
        border-radius: 8px; font-weight: bold; margin: 0.3rem 0;
        text-align: center; animation: pulse 1s infinite;
    }
    .ok-box {
        background: #00b050; color: white; padding: 0.8rem;
        border-radius: 8px; font-weight: bold; margin: 0.3rem 0;
        text-align: center;
    }
    @keyframes pulse {
        0% { opacity: 1; } 50% { opacity: 0.7; } 100% { opacity: 1; }
    }
    .stButton button { border-radius: 8px; }
</style>
''', unsafe_allow_html=True)

# ── RTC config for cloud (STUN servers for NAT traversal) ──
RTC_CONFIG = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {
            "urls": ["turn:global.turn.metered.ca:80"],
            "username": "af67f77bf2be46069c43713c",
            "credential": "2M4Pk0R+f3aFgDK8",
        },
        {
            "urls": ["turn:global.turn.metered.ca:80?transport=tcp"],
            "username": "af67f77bf2be46069c43713c",
            "credential": "2M4Pk0R+f3aFgDK8",
        },
        {
            "urls": ["turn:global.turn.metered.ca:443"],
            "username": "af67f77bf2be46069c43713c",
            "credential": "2M4Pk0R+f3aFgDK8",
        },
        {
            "urls": ["turn:global.turn.metered.ca:443?transport=tcp"],
            "username": "af67f77bf2be46069c43713c",
            "credential": "2M4Pk0R+f3aFgDK8",
        },
    ]
})

# ── SESSION STATE ──
for key, val in [
    ('page', 'home'), ('running', False),
    ('mode', 'student'), ('state', {}),
    ('score', 100.0), ('event_log', []),
    ('focused_seconds', 0.0), ('distracted_seconds', 0.0),
    ('absent_seconds', 0.0), ('session_start', None),
    ('total_absences', 0), ('total_multi_face', 0),
]:
    if key not in st.session_state:
        st.session_state[key] = val


# ── VIDEO PROCESSOR (runs in background thread) ──
class FocusGuardProcessor(VideoProcessorBase):
    def __init__(self, mode: str):
        self.mode = mode
        self.lock = threading.Lock()

        # Detectors
        self.vision = VisionEngine(max_faces=3 if mode == 'student' else 1)
        self.eye = EyeStateDetector(ear_threshold=0.21, drowsy_frames=15)
        self.yawn = YawnDetector(mar_threshold=0.40, yawn_min_frames=8)
        self.phone = PhoneDetector(confidence=0.45, alert_frames=6)
        self.head = HeadPoseDetector(yaw_threshold=20, pitch_threshold=15,
                                      distract_frames=15)
        self.obj = DistractionObjectDetector(confidence=0.40, alert_frames=8,
                                              mode=mode)
        # State
        self.score = 100.0
        self.event_log = []
        self.alerts = []
        self.state = {}
        self.session_start = datetime.now()
        self.focused_seconds = 0.0
        self.distracted_seconds = 0.0
        self.absent_seconds = 0.0
        self.last_tick = time.time()
        self.prev_drowsy = False
        self.prev_yawn = 0
        self.prev_phone = 0
        self.prev_head = 0
        self.prev_obj = 0
        self.no_face_counter = 0
        self.absent_alert = False
        self.absent_logged = False
        self.total_absences = 0
        self.multi_face_counter = 0
        self.multi_face_alert = False
        self.multi_face_logged = False
        self.total_multi_face = 0

    def log_event(self, event_type, details=''):
        self.event_log.append({
            'time': datetime.now().strftime('%H:%M:%S'),
            'type': event_type,
            'details': details
        })
        if self.mode == 'driver':
            pen = {'DROWSINESS': 5, 'PHONE': 4, 'LOOKING_AWAY': 3,
                   'YAWN': 2, 'DISTRACT_OBJECT': 2}
        else:
            pen = {'DROWSINESS': 4, 'PHONE': 5, 'LOOKING_AWAY': 3,
                   'YAWN': 1, 'DISTRACT_OBJECT': 2,
                   'ABSENT': 6, 'MULTI_FACE': 8}
        self.score = max(0.0, self.score - pen.get(event_type, 0))

    def get_grade(self):
        s = self.score
        if self.mode == 'driver':
            if s >= 90: return 'A — Excellent'
            if s >= 75: return 'B — Good'
            if s >= 60: return 'C — Fair'
            if s >= 40: return 'D — Poor'
            return 'F — Dangerous'
        else:
            if s >= 90: return 'A+ — Highly Focused'
            if s >= 75: return 'A — Focused'
            if s >= 60: return 'B — Average'
            if s >= 40: return 'C — Distracted'
            return 'D — Very Distracted'

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format='bgr24')
        img = cv2.flip(img, 1)
        h, w = img.shape[:2]

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.vision.face_mesh.process(rgb)
        num_faces = len(results.multi_face_landmarks) if results.multi_face_landmarks else 0
        landmarks = results.multi_face_landmarks[0] if num_faces > 0 else None
        face_found = num_faces > 0

        eye_r   = self.eye.analyze(landmarks, w, h)
        yawn_r  = self.yawn.analyze(landmarks, w, h)
        phone_r = self.phone.analyze(img)
        head_r  = self.head.analyze(landmarks, w, h)
        obj_r   = self.obj.analyze(img)

        img = self.eye.draw_eye_status(img, eye_r)
        img = self.yawn.draw_mouth_status(img, yawn_r)
        img = self.phone.draw_boxes(img, phone_r)
        img = self.obj.draw_boxes(img, obj_r)

        # Student extras
        if self.mode == 'student':
            if not face_found:
                self.no_face_counter += 1
                if self.no_face_counter >= 30:
                    self.absent_alert = True
                    if not self.absent_logged:
                        self.total_absences += 1
                        self.log_event('ABSENT', 'Left seat')
                        self.absent_logged = True
            else:
                self.no_face_counter = 0
                self.absent_alert = False
                self.absent_logged = False

            if num_faces >= 2:
                self.multi_face_counter += 1
                if self.multi_face_counter >= 15:
                    self.multi_face_alert = True
                    if not self.multi_face_logged:
                        self.total_multi_face += 1
                        self.log_event('MULTI_FACE', '%d faces' % num_faces)
                        self.multi_face_logged = True
            else:
                self.multi_face_counter = 0
                self.multi_face_alert = False
                self.multi_face_logged = False

        # Time tracking
        now = time.time()
        dt = min(now - self.last_tick, 0.5)
        self.last_tick = now
        is_distracted = (eye_r['is_drowsy'] or yawn_r['is_yawning'] or
                         phone_r['phone_detected'] or head_r['is_distracted'] or
                         obj_r['is_distracted'] or self.multi_face_alert)
        if self.mode == 'student' and self.absent_alert:
            self.absent_seconds += dt
        elif is_distracted:
            self.distracted_seconds += dt
        elif face_found:
            self.focused_seconds += dt

        # Log events
        if eye_r['is_drowsy'] and not self.prev_drowsy:
            self.log_event('DROWSINESS')
        self.prev_drowsy = eye_r['is_drowsy']
        if yawn_r['total_yawns'] > self.prev_yawn:
            self.log_event('YAWN')
            self.prev_yawn = yawn_r['total_yawns']
        if phone_r['total_events'] > self.prev_phone:
            self.log_event('PHONE')
            self.prev_phone = phone_r['total_events']
        if head_r['total_distractions'] > self.prev_head:
            self.log_event('LOOKING_AWAY', head_r['direction'])
            self.prev_head = head_r['total_distractions']
        if obj_r['total_events'] > self.prev_obj:
            self.log_event('DISTRACT_OBJECT', ', '.join(obj_r['object_names']))
            self.prev_obj = obj_r['total_events']

        # Build alerts list
        alerts = []
        if self.mode == 'student':
            if self.absent_alert:     alerts.append('STUDENT ABSENT')
            if self.multi_face_alert: alerts.append('MULTIPLE FACES DETECTED')
        if eye_r['is_drowsy']:
            alerts.append('SLEEPING' if self.mode == 'student' else 'DROWSINESS DETECTED')
        if phone_r['phone_detected']:  alerts.append('📱 PHONE USAGE')
        if head_r['is_distracted']:
            alerts.append('LOOKING AWAY' if self.mode == 'student' else 'EYES OFF ROAD')
        if yawn_r['is_yawning']:       alerts.append('YAWNING')
        if obj_r['is_distracted']:     alerts.append('OBJECT DISTRACTION')

        # Draw alert overlay on frame
        if alerts:
            cv2.rectangle(img, (0, 0), (w, h), (0, 0, 255), 6)
            y_pos = h // 2 - (len(alerts) * 28)
            for a in alerts:
                cv2.rectangle(img, (w//2 - 210, y_pos),
                              (w//2 + 210, y_pos + 46), (0, 0, 200), -1)
                cv2.putText(img, '!! ' + a + ' !!',
                            (w//2 - 195, y_pos + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                            (255, 255, 255), 2)
                y_pos += 56

        # Store state for UI thread (thread-safe)
        with self.lock:
            self.alerts = alerts
            self.state = {
                'eyes_closed':   eye_r['eyes_closed'],
                'blinks':        eye_r['blinks'],
                'yawns':         yawn_r['total_yawns'],
                'phone_events':  phone_r['total_events'],
                'head_dir':      head_r['direction'],
                'object_names':  obj_r['object_names'],
                'face_count':    num_faces,
                'absences':      self.total_absences,
                'multi_faces':   self.total_multi_face,
                'away_events':   head_r['total_distractions'],
                'obj_events':    obj_r['total_events'],
                'alerts':        alerts,
                'score':         self.score,
                'grade':         self.get_grade(),
                'focused_sec':   self.focused_seconds,
                'distracted_sec':self.distracted_seconds,
                'absent_sec':    self.absent_seconds,
            }

        return av.VideoFrame.from_ndarray(img, format='bgr24')

    def get_state(self):
        with self.lock:
            return dict(self.state)

    def save_report(self):
        os.makedirs('reports', exist_ok=True)
        end = datetime.now()
        duration = (end - self.session_start).total_seconds()
        total_t = self.focused_seconds + self.distracted_seconds + self.absent_seconds
        fp = (self.focused_seconds / total_t * 100) if total_t > 0 else 0
        dp = (self.distracted_seconds / total_t * 100) if total_t > 0 else 0
        ap = (self.absent_seconds / total_t * 100) if total_t > 0 else 0
        score_key = 'safety_score' if self.mode == 'driver' else 'focus_score'
        grade_key = 'safety_grade' if self.mode == 'driver' else 'focus_grade'
        report = {
            'mode': self.mode.upper(),
            'session_start': self.session_start.isoformat(),
            'session_end': end.isoformat(),
            'duration_seconds': round(duration, 2),
            score_key: round(self.score, 1),
            grade_key: self.get_grade(),
            'time_breakdown': {
                'focused_seconds':   round(self.focused_seconds, 1),
                'distracted_seconds':round(self.distracted_seconds, 1),
                'absent_seconds':    round(self.absent_seconds, 1),
                'focused_pct':       round(fp, 1),
                'distracted_pct':    round(dp, 1),
                'absent_pct':        round(ap, 1),
            },
            'stats': {
                'total_blinks':              self.eye.total_blinks,
                'total_yawns':               self.yawn.total_yawns,
                'phone_events':              self.phone.total_phone_events,
                'looking_away_events':       self.head.total_distractions,
                'object_distraction_events': self.obj.total_events,
                'absence_events':            self.total_absences,
                'multi_face_events':         self.total_multi_face,
            },
            'event_log': self.event_log,
        }
        fname = ('reports/' + self.mode + '_session_' +
                 end.strftime('%Y%m%d_%H%M%S') + '.json')
        with open(fname, 'w') as f:
            json.dump(report, f, indent=2)
        return fname, report


# ── PAGES ──
def page_home():
    st.markdown('<div class="main-title">🛡️ FocusGuard AI</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Unified Attention Monitoring for Students & Drivers</div>',
                unsafe_allow_html=True)
    st.markdown('---')

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('### 🚗 Driver Mode')
        st.write('Monitor driver fatigue, drowsiness, phone usage and looking away.')
        for f in ['Drowsiness (EAR)', 'Yawn (MAR)', 'Phone usage (YOLO)',
                  'Eyes off road', 'Object distraction']:
            st.write('✅ ' + f)
        if st.button('🚗 Launch Driver Mode', use_container_width=True, type='primary'):
            st.session_state.page = 'driver'
            st.rerun()

    with col2:
        st.markdown('### 📚 Student Mode')
        st.write('Track focus during online study or exam sessions.')
        for f in ['All driver features', 'Absence detection',
                  'Multi-face detection (cheating)', 'Focus % time tracking',
                  'Focus grade A+ to D']:
            st.write('✅ ' + f)
        if st.button('📚 Launch Student Mode', use_container_width=True, type='primary'):
            st.session_state.page = 'student'
            st.rerun()

    st.markdown('---')
    if st.button('📊 View Past Reports', use_container_width=True):
        st.session_state.page = 'reports'
        st.rerun()

    st.markdown('---')
    st.caption('Built with MediaPipe · YOLOv8 · streamlit-webrtc · OpenCV')


def page_monitor(mode):
    emoji  = '🚗' if mode == 'driver' else '📚'
    title  = emoji + (' Driver Mode' if mode == 'driver' else ' Student Mode')
    st.markdown('<div class="main-title">' + title + '</div>', unsafe_allow_html=True)

    col_back, _ = st.columns([1, 3])
    with col_back:
        if st.button('⬅️ Back'):
            st.session_state.page = 'home'
            st.rerun()

    st.markdown('---')
    st.info('📷 Allow camera access when your browser asks. The video stays in your browser — nothing is uploaded.')

    # ── webrtc streamer (replaces cv2.VideoCapture) ──
    ctx = webrtc_streamer(
        key='focusguard-' + mode,
        video_processor_factory=lambda: FocusGuardProcessor(mode),
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={'video': True, 'audio': False},
        async_processing=True,
    )

    st.markdown('---')

    # ── Live stats panel (auto-refreshes) ──
    if ctx.state.playing and ctx.video_processor:
        proc = ctx.video_processor
        state = proc.get_state()

        if not state:
            st.info('Starting detectors…')
            time.sleep(0.5)
            st.rerun()
            return

        # Score
        score = state.get('score', 100)
        grade = state.get('grade', '—')
        color = '#00c853' if score >= 75 else '#ff9800' if score >= 50 else '#f44336'
        st.markdown(
            '<div class="metric-card"><h2 style="color:' + color +
            ';margin:0;">Score: ' + str(int(score)) + '/100</h2>'
            '<p style="margin:0;color:#aaa;">' + grade + '</p></div>',
            unsafe_allow_html=True)

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric('Faces',  state.get('face_count', 0))
        col2.metric('Eyes',   'CLOSED' if state.get('eyes_closed') else 'OPEN')
        col3.metric('Blinks', state.get('blinks', 0))
        col4.metric('Yawns',  state.get('yawns', 0))

        col1, col2, col3, col4 = st.columns(4)
        col1.metric('📱 Phone',  state.get('phone_events', 0))
        col2.metric('↩️ Away',   state.get('away_events', 0))
        col3.metric('Head',      state.get('head_dir', '—'))
        col4.metric('Objects',   len(state.get('object_names', [])))

        if mode == 'student':
            col1, col2 = st.columns(2)
            col1.metric('Absences',   state.get('absences', 0))
            col2.metric('Multi-face', state.get('multi_faces', 0))

        # Time breakdown
        fs = state.get('focused_sec', 0)
        ds = state.get('distracted_sec', 0)
        total = max(fs + ds, 1)
        st.progress(int(fs / total * 100),
                    text=f'Focused: {fs:.0f}s ({fs/total*100:.0f}%)')
        st.progress(int(ds / total * 100),
                    text=f'Distracted: {ds:.0f}s ({ds/total*100:.0f}%)')

        # Alerts
        alerts = state.get('alerts', [])
        if alerts:
            for a in alerts:
                st.markdown('<div class="alert-box">⚠️ ' + a + '</div>',
                            unsafe_allow_html=True)
        else:
            st.markdown('<div class="ok-box">✅ All Good — Stay Focused!</div>',
                        unsafe_allow_html=True)

        # Save report button
        st.markdown('---')
        if st.button('💾 Save Report', type='secondary'):
            fname, _ = proc.save_report()
            st.success('Report saved: ' + fname)

        # Auto-refresh every 1 second
        time.sleep(1)
        st.rerun()

    elif not ctx.state.playing:
        st.warning('Click **START** above to begin monitoring.')


def page_reports():
    st.markdown('<div class="main-title">📊 Session Reports</div>',
                unsafe_allow_html=True)
    if st.button('⬅️ Back to Home'):
        st.session_state.page = 'home'
        st.rerun()
    st.markdown('---')

    if not os.path.exists('reports'):
        st.warning('No reports saved yet.')
        return
    files = sorted(glob.glob('reports/*.json'), reverse=True)
    if not files:
        st.warning('No reports saved yet.')
        return

    st.write(f'**{len(files)} session(s) found.**')
    for f in files[:20]:
        try:
            with open(f) as fp:
                data = json.load(fp)
        except Exception:
            continue
        mode      = data.get('mode', 'UNKNOWN')
        score_key = 'safety_score' if mode == 'DRIVER' else 'focus_score'
        grade_key = 'safety_grade' if mode == 'DRIVER' else 'focus_grade'
        emoji     = '🚗' if mode == 'DRIVER' else '📚'
        with st.expander(emoji + ' ' + mode + ' — ' + data['session_start'][:19]):
            c1, c2, c3 = st.columns(3)
            c1.metric('Score',    '%.0f/100' % data.get(score_key, 0))
            c2.metric('Grade',    data.get(grade_key, '—'))
            c3.metric('Duration', '%.0f s'   % data.get('duration_seconds', 0))
            st.json(data.get('stats', {}))
            if 'time_breakdown' in data:
                tb = data['time_breakdown']
                st.progress(int(tb['focused_pct']),
                            text='Focused: %.0f%%'     % tb['focused_pct'])
                st.progress(int(tb['distracted_pct']),
                            text='Distracted: %.0f%%'  % tb['distracted_pct'])
                st.progress(int(tb['absent_pct']),
                            text='Absent: %.0f%%'      % tb['absent_pct'])
            if data.get('event_log'):
                st.dataframe(data['event_log'], use_container_width=True)


def main():
    with st.sidebar:
        st.markdown('# 🛡️ FocusGuard AI')
        st.markdown('---')
        for label, page in [('🏠 Home','home'), ('🚗 Driver Mode','driver'),
                             ('📚 Student Mode','student'), ('📊 Reports','reports')]:
            if st.button(label, use_container_width=True):
                st.session_state.page = page
                st.rerun()
        st.markdown('---')
        st.caption('MediaPipe · YOLOv8 · streamlit-webrtc')

    page = st.session_state.page
    if page == 'home':        page_home()
    elif page == 'driver':    page_monitor('driver')
    elif page == 'student':   page_monitor('student')
    elif page == 'reports':   page_reports()


if __name__ == '__main__':
    main()
