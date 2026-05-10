import streamlit as st
import cv2
import numpy as np
import time
import json
import os
import sys
import glob
from datetime import datetime

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
        text-align: center;
    }
    .ok-box {
        background: #00b050; color: white; padding: 0.8rem;
        border-radius: 8px; font-weight: bold; margin: 0.3rem 0;
        text-align: center;
    }
</style>
''', unsafe_allow_html=True)


# ========= SESSION STATE INIT =========
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'running' not in st.session_state:
    st.session_state.running = False
if 'detectors' not in st.session_state:
    st.session_state.detectors = None


def init_detectors(mode):
    return {
        'mode': mode,
        'vision': VisionEngine(max_faces=3 if mode == 'student' else 1),
        'eye': EyeStateDetector(ear_threshold=0.21, drowsy_frames=15),
        'yawn': YawnDetector(mar_threshold=0.40, yawn_min_frames=8),
        'phone': PhoneDetector(confidence=0.45, alert_frames=6),
        'head': HeadPoseDetector(yaw_threshold=20, pitch_threshold=15,
                                 distract_frames=15),
        'obj': DistractionObjectDetector(confidence=0.40, alert_frames=8,
                                         mode=mode),
        'alarm': SoundAlarm(),
        'session_start': datetime.now(),
        'event_log': [],
        'score': 100.0,
        'prev_drowsy': False,
        'prev_yawn': 0, 'prev_phone': 0, 'prev_head': 0, 'prev_obj': 0,
        'no_face_counter': 0, 'absent_alert': False, 'absent_logged': False,
        'total_absences': 0,
        'multi_face_counter': 0, 'multi_face_alert': False,
        'multi_face_logged': False, 'total_multi_face': 0,
        'focused_seconds': 0.0, 'distracted_seconds': 0.0,
        'absent_seconds': 0.0, 'last_tick': time.time(),
        'last_alarm_state': False
    }


def log_event(d, event_type, details=''):
    d['event_log'].append({
        'time': datetime.now().strftime('%H:%M:%S'),
        'type': event_type, 'details': details
    })
    if d['mode'] == 'driver':
        pen = {'DROWSINESS': 5, 'PHONE': 4, 'LOOKING_AWAY': 3,
               'YAWN': 2, 'DISTRACT_OBJECT': 2}
    else:
        pen = {'DROWSINESS': 4, 'PHONE': 5, 'LOOKING_AWAY': 3,
               'YAWN': 1, 'DISTRACT_OBJECT': 2,
               'ABSENT': 6, 'MULTI_FACE': 8}
    d['score'] = max(0.0, d['score'] - pen.get(event_type, 0))


def get_grade(d):
    s = d['score']
    if d['mode'] == 'driver':
        if s >= 90: return 'A (Excellent)'
        if s >= 75: return 'B (Good)'
        if s >= 60: return 'C (Fair)'
        if s >= 40: return 'D (Poor)'
        return 'F (Dangerous)'
    else:
        if s >= 90: return 'A+ (Highly Focused)'
        if s >= 75: return 'A (Focused)'
        if s >= 60: return 'B (Average)'
        if s >= 40: return 'C (Distracted)'
        return 'D (Very Distracted)'


def save_report(d):
    os.makedirs('reports', exist_ok=True)
    end = datetime.now()
    duration = (end - d['session_start']).total_seconds()
    total_t = d['focused_seconds'] + d['distracted_seconds'] + d['absent_seconds']
    fp = (d['focused_seconds'] / total_t * 100) if total_t > 0 else 0
    dp = (d['distracted_seconds'] / total_t * 100) if total_t > 0 else 0
    ap = (d['absent_seconds'] / total_t * 100) if total_t > 0 else 0

    score_key = 'safety_score' if d['mode'] == 'driver' else 'focus_score'
    grade_key = 'safety_grade' if d['mode'] == 'driver' else 'focus_grade'

    report = {
        'mode': d['mode'].upper(),
        'session_start': d['session_start'].isoformat(),
        'session_end': end.isoformat(),
        'duration_seconds': round(duration, 2),
        score_key: round(d['score'], 1),
        grade_key: get_grade(d),
        'time_breakdown': {
            'focused_seconds': round(d['focused_seconds'], 1),
            'distracted_seconds': round(d['distracted_seconds'], 1),
            'absent_seconds': round(d['absent_seconds'], 1),
            'focused_pct': round(fp, 1),
            'distracted_pct': round(dp, 1),
            'absent_pct': round(ap, 1)
        },
        'stats': {
            'total_blinks': d['eye'].total_blinks,
            'total_yawns': d['yawn'].total_yawns,
            'phone_events': d['phone'].total_phone_events,
            'looking_away_events': d['head'].total_distractions,
            'object_distraction_events': d['obj'].total_events,
            'absence_events': d['total_absences'],
            'multi_face_events': d['total_multi_face']
        },
        'event_log': d['event_log']
    }
    fname = 'reports/' + d['mode'] + '_session_' + end.strftime('%Y%m%d_%H%M%S') + '.json'
    with open(fname, 'w') as f:
        json.dump(report, f, indent=2)
    return fname, report


def process_frame(d, frame):
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = d['vision'].face_mesh.process(rgb)
    num_faces = len(results.multi_face_landmarks) if results.multi_face_landmarks else 0
    landmarks = results.multi_face_landmarks[0] if num_faces > 0 else None
    face_found = num_faces > 0

    eye_r = d['eye'].analyze(landmarks, w, h)
    yawn_r = d['yawn'].analyze(landmarks, w, h)
    phone_r = d['phone'].analyze(frame)
    head_r = d['head'].analyze(landmarks, w, h)
    obj_r = d['obj'].analyze(frame)

    frame = d['eye'].draw_eye_status(frame, eye_r)
    frame = d['yawn'].draw_mouth_status(frame, yawn_r)
    frame = d['phone'].draw_boxes(frame, phone_r)
    frame = d['obj'].draw_boxes(frame, obj_r)

    # Student extras
    if d['mode'] == 'student':
        if not face_found:
            d['no_face_counter'] += 1
            if d['no_face_counter'] >= 30:
                d['absent_alert'] = True
                if not d['absent_logged']:
                    d['total_absences'] += 1
                    log_event(d, 'ABSENT', 'Left seat')
                    d['absent_logged'] = True
        else:
            d['no_face_counter'] = 0
            d['absent_alert'] = False
            d['absent_logged'] = False

        if num_faces >= 2:
            d['multi_face_counter'] += 1
            if d['multi_face_counter'] >= 15:
                d['multi_face_alert'] = True
                if not d['multi_face_logged']:
                    d['total_multi_face'] += 1
                    log_event(d, 'MULTI_FACE', '%d faces' % num_faces)
                    d['multi_face_logged'] = True
        else:
            d['multi_face_counter'] = 0
            d['multi_face_alert'] = False
            d['multi_face_logged'] = False

    # Time tracking
    now = time.time()
    dt = now - d['last_tick']
    d['last_tick'] = now
    is_distracted = (eye_r['is_drowsy'] or yawn_r['is_yawning'] or
                     phone_r['phone_detected'] or head_r['is_distracted'] or
                     obj_r['is_distracted'] or d['multi_face_alert'])
    if d['mode'] == 'student' and d['absent_alert']:
        d['absent_seconds'] += dt
    elif is_distracted:
        d['distracted_seconds'] += dt
    elif face_found:
        d['focused_seconds'] += dt

    # Events
    if eye_r['is_drowsy'] and not d['prev_drowsy']:
        log_event(d, 'DROWSINESS')
    d['prev_drowsy'] = eye_r['is_drowsy']
    if yawn_r['total_yawns'] > d['prev_yawn']:
        log_event(d, 'YAWN')
        d['prev_yawn'] = yawn_r['total_yawns']
    if phone_r['total_events'] > d['prev_phone']:
        log_event(d, 'PHONE')
        d['prev_phone'] = phone_r['total_events']
    if head_r['total_distractions'] > d['prev_head']:
        log_event(d, 'LOOKING_AWAY', head_r['direction'])
        d['prev_head'] = head_r['total_distractions']
    if obj_r['total_events'] > d['prev_obj']:
        log_event(d, 'DISTRACT_OBJECT', ', '.join(obj_r['object_names']))
        d['prev_obj'] = obj_r['total_events']

    # Build alerts
    alerts = []
    if d['mode'] == 'student':
        if d['absent_alert']: alerts.append('STUDENT ABSENT')
        if d['multi_face_alert']: alerts.append('MULTIPLE FACES')
    if eye_r['is_drowsy']:
        alerts.append('SLEEPING' if d['mode'] == 'student' else 'DROWSINESS')
    if phone_r['phone_detected']: alerts.append('PHONE USAGE')
    if head_r['is_distracted']:
        alerts.append('LOOKING AWAY' if d['mode'] == 'student' else 'EYES OFF ROAD')
    if yawn_r['is_yawning']: alerts.append('YAWNING')
    if obj_r['is_distracted']: alerts.append('OBJECT DISTRACTION')

    # Sound alarm
    alarm_should = len(alerts) > 0
    if alarm_should and not d['last_alarm_state']:
        d['alarm'].play()
    elif not alarm_should and d['last_alarm_state']:
        d['alarm'].stop()
    d['last_alarm_state'] = alarm_should

    # Draw alerts overlay
    if alerts:
        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 6)
        y_pos = h // 2 - (len(alerts) * 25)
        for a in alerts:
            cv2.rectangle(frame, (w // 2 - 200, y_pos),
                          (w // 2 + 200, y_pos + 45), (0, 0, 255), -1)
            cv2.putText(frame, '!! ' + a + ' !!', (w // 2 - 180, y_pos + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            y_pos += 55

    state = {
        'eyes_closed': eye_r['eyes_closed'], 'blinks': eye_r['blinks'],
        'yawns': yawn_r['total_yawns'], 'phone_events': phone_r['total_events'],
        'head_dir': head_r['direction'],
        'object_names': obj_r['object_names'], 'face_count': num_faces,
        'absences': d['total_absences'], 'multi_faces': d['total_multi_face'],
        'away_events': head_r['total_distractions'],
        'obj_events': obj_r['total_events'],
        'alerts': alerts, 'score': d['score'], 'grade': get_grade(d)
    }
    return frame, state


# ============= PAGES =============
def page_home():
    st.markdown('<div class="main-title">🛡️ FocusGuard AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Unified Attention Monitoring System</div>',
                unsafe_allow_html=True)
    st.markdown('---')

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('### 🚗 Driver Mode')
        st.write('Monitor driver fatigue, drowsiness, phone usage, looking away.')
        st.write('- Drowsiness (EAR)')
        st.write('- Yawn (MAR)')
        st.write('- Phone usage')
        st.write('- Eyes off road')
        st.write('- Object distraction')
        if st.button('🚗 Launch Driver Mode', use_container_width=True, type='primary'):
            st.session_state.page = 'driver'
            st.rerun()

    with col2:
        st.markdown('### 📚 Student Mode')
        st.write('Track focus during online study/exam sessions.')
        st.write('- All driver features (study tools allowed)')
        st.write('- Absence detection')
        st.write('- Multi-face detection (cheating)')
        st.write('- Focus % time tracking')
        if st.button('📚 Launch Student Mode', use_container_width=True, type='primary'):
            st.session_state.page = 'student'
            st.rerun()

    st.markdown('---')
    if st.button('📊 View Past Reports', use_container_width=True):
        st.session_state.page = 'reports'
        st.rerun()


def page_monitor(mode):
    title = '🚗 Driver Mode' if mode == 'driver' else '📚 Student Mode'
    st.markdown('<div class="main-title">' + title + '</div>', unsafe_allow_html=True)

    col_back, col_btn = st.columns([1, 1])
    with col_back:
        if st.button('⬅️ Back to Home'):
            st.session_state.running = False
            if st.session_state.detectors:
                st.session_state.detectors['alarm'].cleanup()
                st.session_state.detectors = None
            st.session_state.page = 'home'
            st.rerun()

    with col_btn:
        if not st.session_state.running:
            if st.button('▶️ START', type='primary', use_container_width=True):
                st.session_state.detectors = init_detectors(mode)
                st.session_state.running = True
                st.rerun()
        else:
            if st.button('⏹️ STOP & SAVE', type='secondary', use_container_width=True):
                if st.session_state.detectors:
                    fname, _ = save_report(st.session_state.detectors)
                    st.success('Report saved: ' + fname)
                    st.session_state.detectors['alarm'].cleanup()
                    st.session_state.detectors = None
                st.session_state.running = False
                time.sleep(2)
                st.rerun()

    st.markdown('---')

    if not st.session_state.running:
        st.info('Click ▶️ START to begin monitoring.')
        return

    col_video, col_stats = st.columns([2, 1])
    video_placeholder = col_video.empty()
    stats_placeholder = col_stats.empty()
    alert_placeholder = col_stats.empty()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    if not cap.isOpened():
        st.error('Could not open webcam!')
        st.session_state.running = False
        return

    d = st.session_state.detectors
    try:
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                break

            frame, state = process_frame(d, frame)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb, channels='RGB', use_container_width=True)

            with stats_placeholder.container():
                score = state['score']
                if score >= 75: color = '#00c853'
                elif score >= 50: color = '#ff9800'
                else: color = '#f44336'
                st.markdown(
                    '<div class="metric-card"><h2 style="color:' + color +
                    ';margin:0;">Score: ' + str(int(score)) +
                    '/100</h2><p style="margin:0;color:#aaa;">' + state['grade'] +
                    '</p></div>', unsafe_allow_html=True)

                c1, c2 = st.columns(2)
                c1.metric('Faces', state['face_count'])
                c2.metric('Eyes', 'CLOSE' if state['eyes_closed'] else 'OPEN')

                c1, c2 = st.columns(2)
                c1.metric('Blinks', state['blinks'])
                c2.metric('Yawns', state['yawns'])

                c1, c2 = st.columns(2)
                c1.metric('Phone', state['phone_events'])
                c2.metric('Away', state['away_events'])

                if mode == 'student':
                    c1, c2 = st.columns(2)
                    c1.metric('Absences', state['absences'])
                    c2.metric('Multi-Face', state['multi_faces'])

                st.markdown('**Head:** ' + state['head_dir'])
                if state['object_names']:
                    st.markdown('**Objects:** ' + ', '.join(state['object_names']))

            with alert_placeholder.container():
                if state['alerts']:
                    for a in state['alerts']:
                        st.markdown('<div class="alert-box">⚠️ ' + a + '</div>',
                                    unsafe_allow_html=True)
                else:
                    st.markdown('<div class="ok-box">✅ All Good</div>',
                                unsafe_allow_html=True)
    finally:
        cap.release()
        if d:
            d['alarm'].stop()


def page_reports():
    st.markdown('<div class="main-title">📊 Session Reports</div>', unsafe_allow_html=True)
    if st.button('⬅️ Back to Home'):
        st.session_state.page = 'home'
        st.rerun()
    st.markdown('---')

    if not os.path.exists('reports'):
        st.warning('No reports yet.')
        return
    files = sorted(glob.glob('reports/*.json'), reverse=True)
    if not files:
        st.warning('No reports yet.')
        return

    st.write('**' + str(len(files)) + ' sessions found.**')
    for f in files[:20]:
        try:
            with open(f) as fp:
                data = json.load(fp)
        except Exception:
            continue
        mode = data.get('mode', 'UNKNOWN')
        score_key = 'safety_score' if mode == 'DRIVER' else 'focus_score'
        grade_key = 'safety_grade' if mode == 'DRIVER' else 'focus_grade'
        emoji = '🚗' if mode == 'DRIVER' else '📚'
        title = emoji + ' ' + mode + ' - ' + data['session_start'][:19]
        with st.expander(title):
            c1, c2, c3 = st.columns(3)
            c1.metric('Score', '%.0f/100' % data[score_key])
            c2.metric('Grade', data[grade_key])
            c3.metric('Duration', '%.0f sec' % data['duration_seconds'])
            st.json(data.get('stats', {}))
            if 'time_breakdown' in data:
                tb = data['time_breakdown']
                st.progress(int(tb['focused_pct']),
                            text='Focused: %.0f%%' % tb['focused_pct'])
                st.progress(int(tb['distracted_pct']),
                            text='Distracted: %.0f%%' % tb['distracted_pct'])
                st.progress(int(tb['absent_pct']),
                            text='Absent: %.0f%%' % tb['absent_pct'])
            if data.get('event_log'):
                st.dataframe(data['event_log'], use_container_width=True)


def main():
    with st.sidebar:
        st.markdown('# 🛡️ FocusGuard AI')
        st.markdown('---')
        if st.button('🏠 Home', use_container_width=True):
            st.session_state.running = False
            st.session_state.page = 'home'
            st.rerun()
        if st.button('🚗 Driver Mode', use_container_width=True):
            st.session_state.running = False
            st.session_state.page = 'driver'
            st.rerun()
        if st.button('📚 Student Mode', use_container_width=True):
            st.session_state.running = False
            st.session_state.page = 'student'
            st.rerun()
        if st.button('📊 Reports', use_container_width=True):
            st.session_state.running = False
            st.session_state.page = 'reports'
            st.rerun()
        st.markdown('---')
        st.write('Built with MediaPipe + YOLOv8 + Streamlit')

    page = st.session_state.page
    if page == 'home':
        page_home()
    elif page == 'driver':
        page_monitor('driver')
    elif page == 'student':
        page_monitor('student')
    elif page == 'reports':
        page_reports()


if __name__ == '__main__':
    main()
