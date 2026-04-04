"""
ChildTrack Surveillance System — Streamlit UI
Run: streamlit run app/streamlit_app.py
"""

import streamlit as st
import cv2
import numpy as np
import time
import sys
import os
from datetime import datetime
from pathlib import Path

# ── Path setup ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

st.set_page_config(
    page_title="ChildTrack Surveillance System",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════
# CSS
# ══════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    background-color: #030508 !important;
    color: #c8d8e8 !important;
    font-family: 'Rajdhani', sans-serif !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0.5rem !important; }
[data-testid="stSidebar"] {
    background: #060c14 !important;
    border-right: 1px solid #0d2137 !important;
}

/* ── Header ── */
.sys-header {
    background: linear-gradient(90deg, #020609 0%, #071525 50%, #020609 100%);
    border-bottom: 1px solid #0d4a7a;
    border-top: 1px solid #0d4a7a;
    padding: 10px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1rem;
    font-family: 'Share Tech Mono', monospace;
}
.sys-logo {
    font-size: 1rem;
    color: #00aaff;
    letter-spacing: 4px;
    text-transform: uppercase;
    display: flex;
    align-items: center;
    gap: 10px;
}
.sys-logo-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #00ff88;
    box-shadow: 0 0 8px #00ff88;
    display: inline-block;
    animation: pulse 2s infinite;
}
.sys-center {
    font-size: 0.7rem;
    color: #2a4a65;
    letter-spacing: 3px;
    display: flex;
    gap: 20px;
}
.sys-center span { color: #00aaff; }
.sys-right {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    color: #2a4a65;
    display: flex;
    align-items: center;
    gap: 14px;
}
.rec-badge {
    color: #ff3333;
    display: flex;
    align-items: center;
    gap: 5px;
    letter-spacing: 2px;
}
.rec-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #ff3333;
    animation: pulse 1.2s infinite;
    display: inline-block;
}
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.4;transform:scale(.8)} }

/* ── Cards ── */
.info-card {
    background: #060c14;
    border: 1px solid #0d2137;
    border-radius: 6px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.info-card-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.62rem;
    color: #007acc;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 10px;
    border-bottom: 1px solid #0d2137;
    padding-bottom: 6px;
}
.info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0;
    font-size: 0.82rem;
    border-bottom: 1px solid #060e18;
}
.info-row:last-child { border-bottom: none; }
.info-key { color: #2a4a65; font-size: 0.75rem; }
.info-val { font-family: 'Share Tech Mono', monospace; color: #00aaff; font-size: 0.75rem; }
.info-val.green { color: #00e676; }
.info-val.red   { color: #ff1744; }
.info-val.amber { color: #ffab00; }

/* ── Alerts ── */
.alert-box {
    border: 1px solid;
    border-radius: 6px;
    padding: 12px 14px;
    margin-bottom: 10px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 1px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.alert-safe    { border-color: #00e676; background: rgba(0,230,118,.04); color: #00e676; }
.alert-danger  { border-color: #ff1744; background: rgba(255,23,68,.06);  color: #ff5555;
                 animation: glow 2s infinite; }
@keyframes glow { 0%,100%{box-shadow:0 0 0} 50%{box-shadow:0 0 14px rgba(255,23,68,.3)} }

/* ── Camera bar ── */
.cam-bar {
    background: #000;
    border: 1px solid #0d4a7a;
    border-radius: 6px;
    padding: 7px 14px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.68rem;
    margin-bottom: 6px;
}

/* ── Timeline ── */
.tl-item {
    display: flex;
    gap: 10px;
    align-items: flex-start;
    padding: 6px 0;
    border-bottom: 1px solid #060e18;
    font-size: 0.78rem;
}
.tl-time { font-family:'Share Tech Mono',monospace; color:#2a4a65; font-size:.65rem; min-width:68px; margin-top:2px; }
.tl-dot-safe  { width:7px;height:7px;border-radius:50%;background:#00e676;margin-top:4px;flex-shrink:0; }
.tl-dot-alert { width:7px;height:7px;border-radius:50%;background:#ff1744;margin-top:4px;flex-shrink:0;animation:pulse 1s infinite; }
.tl-msg       { color:#c8d8e8; line-height:1.4; }
.tl-msg.alert { color:#ff5555; }

/* ── Stats ── */
.stat-block {
    background: #060c14;
    border: 1px solid #0d2137;
    border-radius: 8px;
    padding: 18px 14px;
    text-align: center;
}
.stat-num {
    font-family: 'Share Tech Mono', monospace;
    font-size: 2.6rem; font-weight: 700;
    line-height: 1; margin-bottom: 6px;
}
.stat-lbl { font-size: .62rem; letter-spacing: 3px; text-transform: uppercase; color: #2a4a65; }
.stat-num.green { color:#00e676; text-shadow:0 0 20px rgba(0,230,118,.3); }
.stat-num.red   { color:#ff1744; text-shadow:0 0 20px rgba(255,23,68,.3); }
.stat-num.blue  { color:#00aaff; text-shadow:0 0 20px rgba(0,180,255,.3); }
.stat-num.amber { color:#ffab00; text-shadow:0 0 20px rgba(255,171,0,.3); }

/* ── Store map ── */
.store-map-wrap {
    background: #030810;
    border: 1px solid #0d2137;
    border-radius: 8px;
    padding: 16px;
    font-family: 'Share Tech Mono', monospace;
}
.store-map-title {
    font-size: .62rem; color: #007acc;
    letter-spacing: 3px; margin-bottom: 12px;
}
.store-inner {
    border: 1px solid #0d4a7a;
    border-radius: 4px;
    height: 140px;
    position: relative;
    background: #020609;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #2a4a65;
    font-size: .62rem;
    letter-spacing: 2px;
}
.store-cam-pin {
    position: absolute;
    top: -8px; left: 50%;
    transform: translateX(-50%);
    display: flex; align-items: center; gap: 5px;
    background: #020609;
    padding: 0 8px;
}
.cam-live-dot {
    width: 9px; height: 9px; border-radius: 50%;
    background: #00e676;
    box-shadow: 0 0 8px #00e676;
    animation: pulse 2s infinite;
    display: inline-block;
}
.store-exit {
    position: absolute;
    bottom: 8px; right: 10px;
    font-size: .58rem; color: #2a4a65;
}
.store-footer {
    margin-top: 10px;
    font-size: .62rem; color: #2a4a65;
    display: flex; align-items: center; gap: 6px;
}

/* ── Scenarios ── */
.sc-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 2px 0 10px;
}
.sc-num   { font-family:'Share Tech Mono',monospace; font-size:.62rem; color:#2a4a65; letter-spacing:3px; }
.sc-title { font-size:1rem; font-weight:700; color:#e0eeff; margin-top:2px; }

.badge {
    font-family: 'Share Tech Mono', monospace;
    font-size: .6rem; padding: 3px 10px;
    border-radius: 3px; letter-spacing: 1px;
}
.badge-green { background:rgba(0,230,118,.1); color:#00e676; border:1px solid rgba(0,230,118,.3); }
.badge-red   { background:rgba(255,23,68,.1);  color:#ff1744; border:1px solid rgba(255,23,68,.3); animation:pulse 1s infinite; }

/* ── Tabs ── */
[data-baseweb="tab"] {
    font-family: 'Share Tech Mono', monospace !important;
    letter-spacing: 2px !important;
    font-size: .72rem !important;
}
[aria-selected="true"] {
    color: #00aaff !important;
    border-bottom: 2px solid #00aaff !important;
}

::-webkit-scrollbar{width:3px;}
::-webkit-scrollbar-thumb{background:#0d2137;border-radius:2px;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════
# Helpers
# ══════════════════════════════════════
def now_str():
    return datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

BASE_DIR = Path(os.path.dirname(__file__))

def video_path(filename):
    return BASE_DIR / filename

def render_header():
    st.markdown(f"""
    <div class="sys-header">
        <div class="sys-logo">
            <div class="sys-logo-dot"></div>
            CHILDTRACK &nbsp;/&nbsp; SURVEILLANCE SYSTEM &nbsp; v1.0
        </div>
        <div class="sys-center">
            <span>CAM-01 <span>ONLINE</span></span>
            <span>MODEL <span>LOADED</span></span>
            <span>DB <span>ACTIVE</span></span>
        </div>
        <div class="sys-right">
            <div class="rec-badge"><div class="rec-dot"></div> REC</div>
            <div>{now_str()}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def info_card(title, rows):
    rows_html = "".join(
        f'<div class="info-row"><span class="info-key">{k}</span>'
        f'<span class="info-val {v[1] if isinstance(v,tuple) else ""}">'
        f'{v[0] if isinstance(v,tuple) else v}</span></div>'
        for k, v in rows.items()
    )
    st.markdown(f"""
    <div class="info-card">
        <div class="info-card-title">{title}</div>
        {rows_html}
    </div>
    """, unsafe_allow_html=True)

def alert_box(kind, msg):
    cls  = "alert-safe" if kind == "safe" else "alert-danger"
    icon = "✓" if kind == "safe" else "🚨"
    st.markdown(f'<div class="alert-box {cls}">{icon} &nbsp; {msg}</div>',
                unsafe_allow_html=True)

def stat_block(num, lbl, color):
    st.markdown(f"""
    <div class="stat-block">
        <div class="stat-num {color}">{num}</div>
        <div class="stat-lbl">{lbl}</div>
    </div>
    """, unsafe_allow_html=True)

def tl_item(t, typ, msg):
    dot = "tl-dot-safe" if typ == "safe" else "tl-dot-alert"
    cls = "alert" if typ == "alert" else ""
    st.markdown(f"""
    <div class="tl-item">
        <span class="tl-time">{t}</span>
        <div class="{dot}"></div>
        <span class="tl-msg {cls}">{msg}</span>
    </div>
    """, unsafe_allow_html=True)

def store_map():
    st.markdown("""
    <div class="store-map-wrap">
        <div class="store-map-title">STORE LAYOUT — CAM COVERAGE</div>
        <div class="store-inner">
            <div class="store-cam-pin">
                <div class="cam-live-dot"></div>
                <span style="color:#00e676;font-size:.62rem;">CAM-01</span>
            </div>
            SHOPPING FLOOR
            <div class="store-exit">EXIT ZONE</div>
        </div>
        <div class="store-footer">
            <div class="cam-live-dot" style="width:7px;height:7px;"></div>
            CAM-01 ACTIVE &nbsp;|&nbsp; Entry + Exit checkpoint only &nbsp;|&nbsp; Interior: NOT monitored
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════
# Scenarios data
# ══════════════════════════════════════
SCENARIOS = [
    {
        "num": "CASE 01", "title": "Normal Entry & Exit", "icon": "✅",
        "status": "safe", "video": "scenario_01_normal_entry_exit.mp4",
        "verdict": "✓ SAFE — Group exited together",
        "details": {
            "Adult ID": "ID:3", "Child ID": "ID:7",
            "Entry time": "10:02:14", "Exit time": "10:18:44",
            "Distance": ("34px  ✓", "green"), "Time window": ("2.1s  ✓", "green"),
            "Group": "G1", "Exit status": ("COMPLETE", "green"),
        },
        "timeline": [
            ("10:02:14","safe","Adult ID:3 entered"),
            ("10:02:16","safe","Child ID:7 linked → G1"),
            ("10:02:18","safe","Group G1 window closed"),
            ("10:18:44","safe","Adult ID:3 exited"),
            ("10:18:46","safe","Child ID:7 exited with companion"),
            ("10:18:46","safe","Group G1 → COMPLETE ✓"),
        ]
    },
    {
        "num": "CASE 02", "title": "Child Exited Alone", "icon": "🚨",
        "status": "alert", "video": "scenario_02_child_alone_exit.mp4",
        "verdict": "🚨 ALERT — Child exited without companion",
        "details": {
            "Adult ID": "ID:3", "Child ID": "ID:7",
            "Entry time": "10:05:22", "Child exit": ("10:21:09  ⚠", "red"),
            "Adult status": ("STILL INSIDE", "red"),
            "Group": "G2", "Exit status": ("⚠ ALERT", "red"),
        },
        "timeline": [
            ("10:05:22","safe","Adult ID:3 entered"),
            ("10:05:24","safe","Child ID:7 linked → G2"),
            ("10:05:26","safe","Group G2 window closed"),
            ("10:21:09","alert","Child ID:7 detected at exit"),
            ("10:21:09","alert","Adult ID:3 — NOT at exit"),
            ("10:21:09","alert","🚨 child_alone triggered"),
        ]
    },
    {
        "num": "CASE 03", "title": "Adult Left — Child Inside", "icon": "🚨",
        "status": "alert", "video": "scenario_03_adult_left_child.mp4",
        "verdict": "🚨 ALERT — Adult left, child may be inside",
        "details": {
            "Adult ID": "ID:5", "Child ID": "ID:9",
            "Entry time": "10:11:33", "Adult exit": ("10:29:05  ⚠", "red"),
            "Child status": ("STILL INSIDE", "red"),
            "Group": "G3", "Exit status": ("⚠ ALERT", "red"),
        },
        "timeline": [
            ("10:11:33","safe","Adult ID:5 entered"),
            ("10:11:35","safe","Child ID:9 linked → G3"),
            ("10:11:37","safe","Group G3 window closed"),
            ("10:29:05","alert","Adult ID:5 detected at exit"),
            ("10:29:05","alert","Child ID:9 — NOT at exit"),
            ("10:29:05","alert","🚨 adult_left_child triggered"),
        ]
    },
    {
        "num": "CASE 04", "title": "Suspected Child Abduction", "icon": "🚨",
        "status": "alert", "video": "scenario_04_child_abduction.mp4",
        "verdict": "🚨 CRITICAL — Child exiting with unknown adult",
        "details": {
            "Original Adult": "ID:2  (Group A)", "Child ID": "ID:8  (Group A)",
            "Unknown Adult": ("ID:11 (Group B)", "red"),
            "Entry time": "10:07:50", "Exit event": ("10:24:18  ⚠", "red"),
            "Group match": ("MISMATCH ⚠", "red"), "Exit status": ("⚠ CRITICAL", "red"),
        },
        "timeline": [
            ("10:07:50","safe","Adult ID:2 + Child ID:8 → Group A"),
            ("10:08:10","safe","Adult ID:11 entered → Group B"),
            ("10:24:18","alert","Child ID:8 detected at exit"),
            ("10:24:18","alert","Companion: ID:11 (Group B) ≠ Group A"),
            ("10:24:18","alert","🚨 CRITICAL: wrong_group / abduction risk"),
        ]
    },
]


# ══════════════════════════════════════
# Session state
# ══════════════════════════════════════
if "cam_running" not in st.session_state:
    st.session_state.cam_running = False
if "model" not in st.session_state:
    st.session_state.model = None
if "alert_count" not in st.session_state:
    st.session_state.alert_count = 0


# ══════════════════════════════════════
# Sidebar
# ══════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:14px 0 8px;">
        <div style="font-family:'Share Tech Mono',monospace;font-size:.95rem;
                    color:#00aaff;letter-spacing:3px;">CHILDTRACK</div>
        <div style="font-size:.62rem;color:#2a4a65;letter-spacing:2px;margin-top:4px;">
            SURVEILLANCE SYSTEM
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio("NAV", [
        "📹  Live Monitor",
        "🚨  Alert Cases",
        "📊  Dashboard"
    ], label_visibility="collapsed")

    st.divider()

    st.markdown('<div style="font-family:\'Share Tech Mono\',monospace;font-size:.62rem;color:#2a4a65;letter-spacing:2px;padding:6px 0;">SYSTEM STATUS</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:.8rem;color:#00e676;font-family:\'Share Tech Mono\',monospace;">● CAM-01 &nbsp; ONLINE</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:.8rem;color:#00e676;font-family:\'Share Tech Mono\',monospace;">● MODEL &nbsp;&nbsp; LOADED</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:.8rem;color:#00e676;font-family:\'Share Tech Mono\',monospace;">● DB &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ACTIVE</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown(f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:.62rem;color:#2a4a65;">{now_str()}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════
# Header
# ══════════════════════════════════════
render_header()


# ══════════════════════════════════════
# PAGE 1 — LIVE MONITOR
# ══════════════════════════════════════
if "Live Monitor" in page:

    col_cam, col_info = st.columns([2.2, 1], gap="medium")

    with col_cam:

        st.markdown(f"""
        <div class="cam-bar">
            <span style="color:#00aaff;">CAM-01 &nbsp;/&nbsp; ENTRANCE</span>
            <span style="color:#2a4a65;">1920×1080 &nbsp;|&nbsp; 30FPS</span>
            <span style="color:#ff3333;">
                <span style="display:inline-block;width:6px;height:6px;border-radius:50%;
                             background:#ff3333;"></span>
                &nbsp; REC &nbsp; {now_str().split()[1]}
            </span>
        </div>
        """, unsafe_allow_html=True)

        # Controls
        c1, c2 = st.columns(2)
        with c1:
            if st.button("▶ Start Camera", use_container_width=True):
                st.session_state.cam_running = True
                # Load model once
                if st.session_state.model is None:
                    try:
                        from detect import load_model
                        st.session_state.model = load_model()
                    except Exception as e:
                        st.error(f"Model load error: {e}")
        with c2:
            if st.button("⏹ Stop Camera", use_container_width=True):
                st.session_state.cam_running = False

        frame_placeholder = st.empty()
        status_placeholder = st.empty()

        if st.session_state.cam_running:
            try:
                from track import track_persons, cleanup_old_tracks, draw_tracks
                model = st.session_state.model

                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    status_placeholder.error("❌ Cannot open camera")
                    st.session_state.cam_running = False
                else:
                    status_placeholder.success("● Camera running — model active")

                    while st.session_state.cam_running:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        frame = cv2.resize(frame, (960, 540))

                        if model is not None:
                            tracks = track_persons(model, frame)
                            active_ids = {t['track_id'] for t in tracks}
                            cleanup_old_tracks(active_ids)
                            frame = draw_tracks(frame, tracks)

                            # Overlay
                            ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
                            cv2.putText(frame, "CAM-01 / ENTRANCE", (12, 24),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,170,255), 1)
                            cv2.putText(frame, ts, (12, frame.shape[0]-12),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,170,255), 1)
                            cv2.putText(frame, "REC", (frame.shape[1]-70, 24),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 1)
                            cv2.circle(frame, (frame.shape[1]-85, 19), 5, (0,0,255), -1)

                        frame_placeholder.image(frame, channels="BGR",
                                                use_container_width=True)
                        time.sleep(0.03)

                    cap.release()

            except ImportError as e:
                status_placeholder.warning(f"⚠ Import error: {e}")
                # Fallback: plain camera
                cap = cv2.VideoCapture(0)
                while st.session_state.cam_running:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.resize(frame, (960, 540))
                    frame_placeholder.image(frame, channels="BGR",
                                            use_container_width=True)
                    time.sleep(0.03)
                cap.release()
        else:
            blank = np.zeros((540, 960, 3), dtype=np.uint8)
            cv2.putText(blank, "CHILDTRACK SURVEILLANCE FEED", (180, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,170,255), 2)
            cv2.putText(blank, "Press  Start Camera  to begin", (240, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60,90,110), 1)
            frame_placeholder.image(blank, channels="BGR", use_container_width=True)

        st.markdown("""
        <div style="display:flex;gap:12px;margin-top:6px;
                    font-family:'Share Tech Mono',monospace;font-size:.62rem;">
            <span style="color:#00e676;">● BYTETRACK RUNNING</span>
            <span style="color:#2a4a65;">|</span>
            <span style="color:#00aaff;">● YOLO ACTIVE</span>
            <span style="color:#2a4a65;">|</span>
            <span style="color:#00e676;">● RE-ID ACTIVE</span>
        </div>
        """, unsafe_allow_html=True)

    with col_info:

        alert_box("safe", "NO ACTIVE ALERTS")

        info_card("ACTIVE GROUPS", {
            "Groups inside"  : "—",
            "Total adults"   : "—",
            "Total children" : "—",
            "Monitored"      : "—",
            "Alerts today"   : ("0", "red"),
        })

        info_card("LAST DETECTED", {
            "Track ID"   : "—",
            "Label"      : "—",
            "Direction"  : "—",
            "Group"      : "—",
            "Confidence" : "—",
        })

        info_card("SYSTEM STATUS", {
            "● Camera"      : ("ONLINE", "green"),
            "● YOLO Model"  : ("LOADED", "green"),
            "● ByteTrack"   : ("RUNNING", "green"),
            "● Re-ID"       : ("ACTIVE", "green"),
            "● Database"    : ("READY", "green"),
        })


# ══════════════════════════════════════
# PAGE 2 — ALERT CASES
# ══════════════════════════════════════
elif "Alert Cases" in page:

    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace;font-size:.68rem;
                color:#2a4a65;letter-spacing:3px;margin-bottom:1rem;">
        ALERT CASES REVIEW — 4 DOCUMENTED INCIDENTS
    </div>
    """, unsafe_allow_html=True)

    for sc in SCENARIOS:
        badge_cls = "badge-green" if sc["status"] == "safe" else "badge-red"
        badge_txt = "SAFE" if sc["status"] == "safe" else "ALERT"

        with st.expander(f"{sc['icon']}  {sc['num']} — {sc['title']}", expanded=False):

            col_v, col_d = st.columns([1.6, 1], gap="medium")

            with col_v:
                hdr_color = "#00e676" if sc["status"] == "safe" else "#ff1744"
                hdr_label = "● REC" if sc["status"] == "safe" else "⚠ ALERT TRIGGERED"

                st.markdown(f"""
                <div style="background:#000;border:1px solid #0d4a7a;
                            border-radius:6px 6px 0 0;padding:7px 12px;
                            display:flex;justify-content:space-between;
                            font-family:'Share Tech Mono',monospace;font-size:.65rem;">
                    <span style="color:#00aaff;">CAM-01 &nbsp;/&nbsp; {sc['num']}</span>
                    <span style="color:{hdr_color};">{hdr_label}</span>
                </div>
                """, unsafe_allow_html=True)

                vfile = video_path(sc["video"])
                if vfile.exists():
                    st.video(str(vfile))
                else:
                    st.markdown(f"""
                    <div style="background:#000;border:1px solid #0d2137;
                                border-radius:0 0 6px 6px;height:220px;
                                display:flex;flex-direction:column;
                                align-items:center;justify-content:center;
                                font-family:'Share Tech Mono',monospace;color:#0d4a7a;">
                        <div style="font-size:1.4rem;margin-bottom:8px;">📷</div>
                        <div style="letter-spacing:2px;font-size:.7rem;">VIDEO NOT FOUND</div>
                        <div style="font-size:.58rem;margin-top:6px;color:#1a3050;">{sc['video']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                vtype = "safe" if sc["status"] == "safe" else "danger"
                alert_box(vtype, sc["verdict"])

            with col_d:
                info_card("DETECTION DETAILS", sc["details"])

                st.markdown('<div class="info-card-title" style="margin-top:10px;">EVENT TIMELINE</div>',
                            unsafe_allow_html=True)
                for t, typ, msg in sc["timeline"]:
                    tl_item(t, typ, msg)


# ══════════════════════════════════════
# PAGE 3 — DASHBOARD
# ══════════════════════════════════════
elif "Dashboard" in page:

    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace;font-size:.68rem;
                color:#2a4a65;letter-spacing:3px;margin-bottom:1rem;">
        OPERATIONS DASHBOARD — SESSION SUMMARY
    </div>
    """, unsafe_allow_html=True)

    # Stats
    c1, c2, c3, c4 = st.columns(4)
    with c1: stat_block("4",   "CASES TESTED",    "blue")
    with c2: stat_block("3",   "ALERTS TRIGGERED","red")
    with c3: stat_block("1",   "SAFE EXITS",       "green")
    with c4: stat_block("75%", "ALERT RATE",       "amber")

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.3, 1], gap="medium")

    with col_left:

        store_map()

        st.markdown("<br>", unsafe_allow_html=True)

        info_card("ALERT BREAKDOWN", {
            "🚨 child_alone"      : ("1 event", "red"),
            "🚨 adult_left_child" : ("1 event", "red"),
            "🚨 wrong_group"      : ("1 event", "red"),
            "✅ clean_exit"       : ("1 event", "green"),
            "⏱ timeout"          : "0 events",
        })

        info_card("GROUP SUMMARY", {
            "G1 — Adults:1 Children:1" : ("✅ COMPLETE",   "green"),
            "G2 — Adults:1 Children:1" : ("🚨 child_alone","red"),
            "G3 — Adults:1 Children:1" : ("🚨 adult_left", "red"),
            "G4A— Adults:1 Children:1" : ("🚨 wrong_group","red"),
            "G4B— Adults:1 Children:0" : ("✅ COMPLETE",   "green"),
        })

    with col_right:

        st.markdown('<div class="info-card-title">FULL EVENT LOG</div>',
                    unsafe_allow_html=True)

        events = [
            ("10:02:14","safe", "G1 — Normal entry detected"),
            ("10:02:16","safe", "Child ID:7 linked to Group G1"),
            ("10:05:22","safe", "G2 — Entry detected"),
            ("10:05:24","safe", "Child ID:7 linked to Group G2"),
            ("10:07:50","safe", "G3-A entry — Adult ID:2 + Child ID:8"),
            ("10:08:10","safe", "G3-B entry — Adult ID:11 (separate)"),
            ("10:11:33","safe", "G4 — Entry detected"),
            ("10:18:46","safe", "G1 → COMPLETE — Safe exit ✓"),
            ("10:21:09","alert","🚨 child_alone — Case 02"),
            ("10:24:18","alert","🚨 wrong_group — Case 04"),
            ("10:29:05","alert","🚨 adult_left_child — Case 03"),
        ]
        for t, typ, msg in events:
            tl_item(t, typ, msg)












# # واجهة بسيطة لعرض النتائج
# # يعرض الفيديو مع الـ IDs والمجموعات
# # يعرض التنبيهات بشكل واضح
# # بناء سستم كامل يبين لي ايش التنبيهات  وايش الحاصل والقرارات الي اتخذها المودل بناء على السيناريوهات السابقة 



# """
# ╔══════════════════════════════════════════════════════════╗
# ║         CHILDTRACK SURVEILLANCE SYSTEM v1.0              ║
# ║         app/streamlit_app.py                             ║
# ╚══════════════════════════════════════════════════════════╝

# How to run:
#     streamlit run app/streamlit_app.py

# Requirements:
#     pip install streamlit

# Video files expected at:
#     data/videos/scenario_01_normal_entry_exit.mp4
#     data/videos/scenario_02_child_alone_exit.mp4
#     data/videos/scenario_03_adult_left_child.mp4
#     data/videos/scenario_04_child_abduction.mp4
# """

# import streamlit as st
# import time
# import random
# from datetime import datetime
# from pathlib import Path

# # ──────────────────────────────────────
# # Page config — must be first st command
# # ──────────────────────────────────────
# st.set_page_config(
#     page_title  = "ChildTrack Surveillance System",
#     page_icon   = "🔒",
#     layout      = "wide",
#     initial_sidebar_state = "expanded"
# )

# # ──────────────────────────────────────
# # Global CSS — surveillance dark theme
# # ──────────────────────────────────────
# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

# /* ── Base ── */
# html, body, [class*="css"] {
#     background-color: #030508 !important;
#     color: #c8d8e8 !important;
#     font-family: 'Rajdhani', sans-serif !important;
# }

# /* ── Hide Streamlit chrome ── */
# #MainMenu, footer, header { visibility: hidden; }
# .block-container { padding-top: 1rem !important; }

# /* ── Sidebar ── */
# [data-testid="stSidebar"] {
#     background: #060c14 !important;
#     border-right: 1px solid #0d2137 !important;
# }

# /* ── System header bar ── */
# .sys-header {
#     background: linear-gradient(90deg, #030508 0%, #071525 50%, #030508 100%);
#     border-bottom: 1px solid #0d4a7a;
#     border-top: 1px solid #0d4a7a;
#     padding: 10px 24px;
#     display: flex;
#     align-items: center;
#     justify-content: space-between;
#     margin-bottom: 1.2rem;
#     font-family: 'Share Tech Mono', monospace;
# }
# .sys-title {
#     font-size: 1.1rem;
#     color: #00aaff;
#     letter-spacing: 4px;
#     text-transform: uppercase;
# }
# .sys-status {
#     font-size: 0.75rem;
#     color: #00ff88;
#     letter-spacing: 2px;
# }
# .sys-time {
#     font-size: 0.75rem;
#     color: #607080;
#     letter-spacing: 1px;
#     font-family: 'Share Tech Mono', monospace;
# }

# /* ── Threat bar ── */
# .threat-bar {
#     display: flex;
#     align-items: center;
#     gap: 12px;
#     padding: 8px 16px;
#     border: 1px solid #0d2137;
#     border-radius: 4px;
#     background: #060c14;
#     margin-bottom: 1rem;
#     font-family: 'Share Tech Mono', monospace;
#     font-size: 0.8rem;
#     letter-spacing: 2px;
# }
# .threat-low   { color: #00ff88; }
# .threat-med   { color: #ffaa00; }
# .threat-high  { color: #ff3333; animation: blink 1s infinite; }
# @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

# /* ── Camera feed container ── */
# .cam-container {
#     position: relative;
#     border: 1px solid #0d4a7a;
#     border-radius: 4px;
#     background: #000;
#     overflow: hidden;
# }
# .cam-overlay-tl {
#     position: absolute; top: 10px; left: 12px;
#     font-family: 'Share Tech Mono', monospace;
#     font-size: 0.72rem; color: #00aaff;
#     letter-spacing: 1px; text-shadow: 0 0 8px #00aaff88;
#     pointer-events: none;
# }
# .cam-overlay-tr {
#     position: absolute; top: 10px; right: 12px;
#     font-family: 'Share Tech Mono', monospace;
#     font-size: 0.72rem; color: #ff3333;
#     letter-spacing: 1px;
#     display: flex; align-items: center; gap: 6px;
# }
# .rec-dot {
#     width: 8px; height: 8px; border-radius: 50%;
#     background: #ff3333;
#     animation: blink 1.2s infinite;
#     display: inline-block;
# }

# /* ── Info panel cards ── */
# .info-card {
#     background: #060c14;
#     border: 1px solid #0d2137;
#     border-radius: 6px;
#     padding: 14px 16px;
#     margin-bottom: 10px;
# }
# .info-card-title {
#     font-family: 'Share Tech Mono', monospace;
#     font-size: 0.65rem;
#     color: #007acc;
#     letter-spacing: 3px;
#     text-transform: uppercase;
#     margin-bottom: 10px;
#     border-bottom: 1px solid #0d2137;
#     padding-bottom: 6px;
# }
# .info-row {
#     display: flex;
#     justify-content: space-between;
#     align-items: center;
#     padding: 4px 0;
#     font-size: 0.85rem;
#     border-bottom: 1px solid #060e18;
# }
# .info-val {
#     font-family: 'Share Tech Mono', monospace;
#     color: #00aaff;
# }

# /* ── Alert box ── */
# .alert-box {
#     border: 1px solid;
#     border-radius: 6px;
#     padding: 14px 16px;
#     margin-bottom: 10px;
#     font-family: 'Share Tech Mono', monospace;
#     font-size: 0.8rem;
#     letter-spacing: 1px;
# }
# .alert-safe    { border-color: #00ff88; background: rgba(0,255,136,0.04); color: #00ff88; }
# .alert-warning { border-color: #ff3333; background: rgba(255,51,51,0.06); color: #ff5555; animation: border-pulse 2s infinite; }
# @keyframes border-pulse {
#     0%,100% { box-shadow: 0 0 0 rgba(255,51,51,0); }
#     50%      { box-shadow: 0 0 12px rgba(255,51,51,0.4); }
# }

# /* ── Scenario card ── */
# .scenario-card {
#     background: #060c14;
#     border: 1px solid #0d2137;
#     border-radius: 8px;
#     padding: 20px;
#     margin-bottom: 16px;
# }
# .scenario-num {
#     font-family: 'Share Tech Mono', monospace;
#     font-size: 0.65rem;
#     color: #007acc;
#     letter-spacing: 3px;
#     margin-bottom: 4px;
# }
# .scenario-title {
#     font-size: 1.1rem;
#     font-weight: 700;
#     color: #e0eeff;
#     margin-bottom: 12px;
# }

# /* ── Timeline event ── */
# .timeline-event {
#     display: flex;
#     gap: 12px;
#     align-items: flex-start;
#     padding: 8px 0;
#     border-bottom: 1px solid #060e18;
#     font-size: 0.82rem;
# }
# .t-time {
#     font-family: 'Share Tech Mono', monospace;
#     color: #607080;
#     min-width: 70px;
#     font-size: 0.72rem;
# }
# .t-dot-safe   { width:8px;height:8px;border-radius:50%;background:#00ff88;margin-top:4px;flex-shrink:0; }
# .t-dot-alert  { width:8px;height:8px;border-radius:50%;background:#ff3333;margin-top:4px;flex-shrink:0;animation:blink 1s infinite; }

# /* ── Dashboard stat ── */
# .dash-stat {
#     background: #060c14;
#     border: 1px solid #0d2137;
#     border-radius: 8px;
#     padding: 18px;
#     text-align: center;
# }
# .dash-stat-num {
#     font-family: 'Share Tech Mono', monospace;
#     font-size: 2.4rem;
#     font-weight: 700;
#     line-height: 1;
#     margin-bottom: 6px;
# }
# .dash-stat-label {
#     font-size: 0.72rem;
#     letter-spacing: 2px;
#     text-transform: uppercase;
#     color: #607080;
# }
# .stat-green { color: #00ff88; }
# .stat-red   { color: #ff3333; }
# .stat-blue  { color: #00aaff; }
# .stat-amber { color: #ffaa00; }

# /* ── Store map ── */
# .store-map {
#     background: #060c14;
#     border: 1px solid #0d2137;
#     border-radius: 8px;
#     padding: 20px;
#     position: relative;
#     min-height: 200px;
#     font-family: 'Share Tech Mono', monospace;
#     font-size: 0.75rem;
# }
# .cam-dot {
#     display: inline-block;
#     width: 10px; height: 10px;
#     border-radius: 50%;
#     background: #00ff88;
#     animation: blink 2s infinite;
#     margin-right: 6px;
# }

# /* ── Tabs ── */
# [data-baseweb="tab"] {
#     font-family: 'Share Tech Mono', monospace !important;
#     letter-spacing: 2px !important;
#     font-size: 0.75rem !important;
#     color: #607080 !important;
# }
# [aria-selected="true"] {
#     color: #00aaff !important;
#     border-bottom: 2px solid #00aaff !important;
# }

# /* ── Metric ── */
# [data-testid="stMetric"] {
#     background: #060c14;
#     border: 1px solid #0d2137;
#     border-radius: 6px;
#     padding: 10px 14px;
# }

# /* ── Video ── */
# video { border-radius: 4px; }
# </style>
# """, unsafe_allow_html=True)


# # ──────────────────────────────────────
# # Helpers
# # ──────────────────────────────────────
# def now_str():
#     return datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

# # def video_path(filename):
# #     return Path("data/videos") / filename

# BASE_DIR = Path(r"C:\Users\hamee\OneDrive\سطح المكتب\Alaa_Tuwaiq\Projects\ComputerVision_Project\computer-vision-project-childtrack\app")
# def video_path(filename):
#     return BASE_DIR / filename


# def video_exists(filename):
#     return video_path(filename).exists()

# def render_header():
#     st.markdown(f"""
#     <div class="sys-header">
#         <div class="sys-title">🔒 &nbsp; ChildTrack &nbsp;—&nbsp; Surveillance System &nbsp; v1.0</div>
#         <div class="sys-status">● SYSTEM ONLINE</div>
#         <div class="sys-time" id="clock">{now_str()}</div>
#     </div>
#     """, unsafe_allow_html=True)

# def render_threat(level="LOW"):
#     colors = {"LOW": "threat-low", "MEDIUM": "threat-med", "HIGH": "threat-high"}
#     icons  = {"LOW": "▲", "MEDIUM": "▲▲", "HIGH": "▲▲▲"}
#     st.markdown(f"""
#     <div class="threat-bar">
#         THREAT LEVEL :
#         <span class="{colors[level]}"> {icons[level]} &nbsp; {level} </span>
#         &nbsp;|&nbsp; CAM-01: ENTRANCE &nbsp;|&nbsp; MONITORING ACTIVE
#     </div>
#     """, unsafe_allow_html=True)

# def alert_box(alert_type, message):
#     cls = "alert-safe" if alert_type == "safe" else "alert-warning"
#     icon = "✓" if alert_type == "safe" else "⚠"
#     st.markdown(f'<div class="alert-box {cls}">{icon} &nbsp; {message}</div>',
#                 unsafe_allow_html=True)

# def info_card(title, rows):
#     rows_html = "".join(
#         f'<div class="info-row"><span>{k}</span><span class="info-val">{v}</span></div>'
#         for k, v in rows.items()
#     )
#     st.markdown(f"""
#     <div class="info-card">
#         <div class="info-card-title">{title}</div>
#         {rows_html}
#     </div>
#     """, unsafe_allow_html=True)

# def dash_stat(number, label, color_class):
#     st.markdown(f"""
#     <div class="dash-stat">
#         <div class="dash-stat-num {color_class}">{number}</div>
#         <div class="dash-stat-label">{label}</div>
#     </div>
#     """, unsafe_allow_html=True)


# # ──────────────────────────────────────
# # Scenarios data
# # ──────────────────────────────────────
# SCENARIOS = [
#     {
#         "num"      : "CASE 01",
#         "title"    : "Normal Entry & Exit",
#         "icon"     : "✅",
#         "status"   : "safe",
#         "video"    : "scenario_01_normal_entry_exit.mp4",
#         "details"  : {
#             "Adult ID"       : "ID:3",
#             "Child ID"       : "ID:7",
#             "Entry time"     : "10:02:14",
#             "Exit time"      : "10:18:44",
#             "Distance"       : "34px  ✓",
#             "Time window"    : "2.1s   ✓",
#             "Group"          : "G1",
#             "Exit status"    : "COMPLETE",
#         },
#         "verdict"  : "✓ SAFE — Group exited together",
#         "timeline" : [
#             ("10:02:14", "safe",  "Adult ID:3 entered"),
#             ("10:02:16", "safe",  "Child ID:7 entered — linked to G1"),
#             ("10:02:18", "safe",  "Group G1 window closed"),
#             ("10:18:44", "safe",  "Adult ID:3 exited"),
#             ("10:18:46", "safe",  "Child ID:7 exited with companion"),
#             ("10:18:46", "safe",  "Group G1 → COMPLETE ✓"),
#         ]
#     },
#     {
#         "num"      : "CASE 02",
#         "title"    : "Child Exited Alone",
#         "icon"     : "🚨",
#         "status"   : "alert",
#         "video"    : "scenario_02_child_alone_exit.mp4",
#         "details"  : {
#             "Adult ID"       : "ID:3",
#             "Child ID"       : "ID:7",
#             "Entry time"     : "10:05:22",
#             "Exit time"      : "10:21:09",
#             "Child exit"     : "10:21:09  ⚠",
#             "Adult status"   : "STILL INSIDE",
#             "Group"          : "G2",
#             "Exit status"    : "⚠ ALERT",
#         },
#         "verdict"  : "🚨 ALERT — Child exited without companion",
#         "timeline" : [
#             ("10:05:22", "safe",  "Adult ID:3 entered"),
#             ("10:05:24", "safe",  "Child ID:7 entered — linked to G2"),
#             ("10:05:26", "safe",  "Group G2 window closed"),
#             ("10:21:09", "alert", "Child ID:7 detected at exit"),
#             ("10:21:09", "alert", "Adult ID:3 — NOT at exit zone"),
#             ("10:21:09", "alert", "🚨 ALERT: child_alone triggered"),
#         ]
#     },
#     {
#         "num"      : "CASE 03",
#         "title"    : "Adult Left — Child Inside",
#         "icon"     : "🚨",
#         "status"   : "alert",
#         "video"    : "scenario_03_adult_left_child.mp4",
#         "details"  : {
#             "Adult ID"       : "ID:5",
#             "Child ID"       : "ID:9",
#             "Entry time"     : "10:11:33",
#             "Adult exit"     : "10:29:05  ⚠",
#             "Child status"   : "STILL INSIDE",
#             "Group"          : "G3",
#             "Exit status"    : "⚠ ALERT",
#         },
#         "verdict"  : "🚨 ALERT — Adult left, child may be inside",
#         "timeline" : [
#             ("10:11:33", "safe",  "Adult ID:5 entered"),
#             ("10:11:35", "safe",  "Child ID:9 entered — linked to G3"),
#             ("10:11:37", "safe",  "Group G3 window closed"),
#             ("10:29:05", "alert", "Adult ID:5 detected at exit"),
#             ("10:29:05", "alert", "Child ID:9 — NOT at exit zone"),
#             ("10:29:05", "alert", "🚨 ALERT: adult_left_child triggered"),
#         ]
#     },
#     {
#         "num"      : "CASE 04",
#         "title"    : "Suspected Child Abduction",
#         "icon"     : "🚨",
#         "status"   : "alert",
#         "video"    : "scenario_04_child_abduction.mp4",
#         "details"  : {
#             "Original Adult" : "ID:2  (Group A)",
#             "Child ID"       : "ID:8  (Group A)",
#             "Unknown Adult"  : "ID:11 (Group B)",
#             "Entry time"     : "10:07:50",
#             "Exit event"     : "10:24:18  ⚠",
#             "Group match"    : "MISMATCH ⚠",
#             "Exit status"    : "⚠ CRITICAL ALERT",
#         },
#         "verdict"  : "🚨 CRITICAL — Child exiting with unknown adult",
#         "timeline" : [
#             ("10:07:50", "safe",  "Adult ID:2 + Child ID:8 entered → Group A"),
#             ("10:08:10", "safe",  "Adult ID:11 entered separately → Group B"),
#             ("10:24:18", "alert", "Child ID:8 detected at exit"),
#             ("10:24:18", "alert", "Companion: Adult ID:11 (Group B) ≠ Group A"),
#             ("10:24:18", "alert", "🚨 CRITICAL ALERT: wrong_group / abduction risk"),
#         ]
#     },
# ]


# # ──────────────────────────────────────
# # Sidebar
# # ──────────────────────────────────────
# with st.sidebar:
#     st.markdown("""
#     <div style="text-align:center;padding:16px 0 8px;">
#         <div style="font-family:'Share Tech Mono',monospace;font-size:1rem;
#                     color:#00aaff;letter-spacing:3px;">CHILDTRACK</div>
#         <div style="font-size:0.65rem;color:#607080;letter-spacing:2px;margin-top:4px;">
#             SURVEILLANCE SYSTEM
#         </div>
#     </div>
#     """, unsafe_allow_html=True)

#     st.divider()

#     page = st.radio(
#         "NAVIGATION",
#         ["📹  Live Monitor", "🎬  Scenarios", "📊  Dashboard"],
#         label_visibility="collapsed"
#     )

#     st.divider()

#     st.markdown("""
#     <div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;
#                 color:#607080;letter-spacing:2px;padding:8px 0;">
#     SYSTEM STATUS
#     </div>
#     """, unsafe_allow_html=True)

#     st.markdown('<div style="font-size:0.8rem;color:#00ff88;">● CAM-01  ONLINE</div>', unsafe_allow_html=True)
#     st.markdown('<div style="font-size:0.8rem;color:#00ff88;">● MODEL   LOADED</div>', unsafe_allow_html=True)
#     st.markdown('<div style="font-size:0.8rem;color:#00ff88;">● DB      ACTIVE</div>', unsafe_allow_html=True)
#     st.markdown('<div style="font-size:0.8rem;color:#607080;">● RE-ID   STANDBY</div>', unsafe_allow_html=True)

#     st.divider()
#     st.markdown(f"""
#     <div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;
#                 color:#607080;letter-spacing:1px;">{now_str()}</div>
#     """, unsafe_allow_html=True)


# # ──────────────────────────────────────
# # HEADER (all pages)
# # ──────────────────────────────────────
# render_header()


# # ══════════════════════════════════════
# # PAGE 1 — LIVE MONITOR
# # ══════════════════════════════════════
# if "Live Monitor" in page:

#     render_threat("LOW")

#     col_video, col_info = st.columns([2.2, 1], gap="medium")

#     with col_video:
#         # Camera overlay header
#         st.markdown(f"""
#         <div style="background:#000;border:1px solid #0d4a7a;border-radius:6px;padding:8px 14px;
#                     display:flex;justify-content:space-between;align-items:center;
#                     font-family:'Share Tech Mono',monospace;font-size:0.72rem;margin-bottom:6px;">
#             <span style="color:#00aaff;">CAM-01 &nbsp;/&nbsp; ENTRANCE</span>
#             <span style="color:#607080;">1920×1080 &nbsp;|&nbsp; 30FPS</span>
#             <span style="color:#ff3333;">
#                 <span style="display:inline-block;width:7px;height:7px;border-radius:50%;
#                              background:#ff3333;animation:blink 1.2s infinite;"></span>
#                 &nbsp; REC &nbsp; {now_str().split()[1]}
#             </span>
#         </div>
#         """, unsafe_allow_html=True)

#         vfile = video_path("scenario_01_normal_entry_exit.mp4")
#         if vfile.exists():
#             st.video(str(vfile))
#         else:
#             st.markdown("""
#             <div style="background:#000;border:1px solid #0d4a7a;border-radius:6px;
#                         height:340px;display:flex;flex-direction:column;
#                         align-items:center;justify-content:center;
#                         font-family:'Share Tech Mono',monospace;color:#0d4a7a;">
#                 <div style="font-size:2rem;margin-bottom:12px;">📷</div>
#                 <div style="letter-spacing:3px;font-size:0.8rem;">NO SIGNAL</div>
#                 <div style="font-size:0.65rem;margin-top:8px;color:#334455;">
#                     Place video at: data/videos/scenario_01_normal_entry_exit.mp4
#                 </div>
#             </div>
#             """, unsafe_allow_html=True)

#         st.markdown("""
#         <div style="display:flex;gap:10px;margin-top:8px;
#                     font-family:'Share Tech Mono',monospace;font-size:0.65rem;">
#             <span style="color:#00ff88;">● TRACKING ACTIVE</span>
#             <span style="color:#607080;">&nbsp;|&nbsp;</span>
#             <span style="color:#00aaff;">● BYTETRACK RUNNING</span>
#             <span style="color:#607080;">&nbsp;|&nbsp;</span>
#             <span style="color:#607080;">● RE-ID STANDBY</span>
#         </div>
#         """, unsafe_allow_html=True)

#     with col_info:

#         info_card("ACTIVE GROUPS", {
#             "Groups inside"  : "2",
#             "Total adults"   : "3",
#             "Total children" : "2",
#             "Monitored"      : "2 groups",
#         })

#         info_card("LAST DETECTED", {
#             "Track ID"   : "ID:7",
#             "Label"      : "child",
#             "Direction"  : "entry",
#             "Group"      : "G1",
#             "Conf"       : "0.91",
#         })

#         info_card("GROUP G1", {
#             "Adults"     : "ID:3",
#             "Children"   : "ID:7",
#             "Entry"      : "10:02:14",
#             "Window"     : "CLOSED",
#             "Status"     : "INSIDE ✓",
#         })

#         alert_box("safe", "NO ACTIVE ALERTS")


# # ══════════════════════════════════════
# # PAGE 2 — SCENARIOS
# # ══════════════════════════════════════
# elif "Scenarios" in page:

#     st.markdown("""
#     <div style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;
#                 color:#607080;letter-spacing:3px;margin-bottom:1rem;">
#         SCENARIO REVIEW — 4 TEST CASES
#     </div>
#     """, unsafe_allow_html=True)

#     for sc in SCENARIOS:
#         with st.expander(f"{sc['icon']}  {sc['num']} — {sc['title']}", expanded=False):

#             col_v, col_d = st.columns([1.6, 1], gap="medium")

#             with col_v:
#                 # Camera-style header over video
#                 color = "#00ff88" if sc["status"] == "safe" else "#ff3333"
#                 st.markdown(f"""
#                 <div style="background:#000;border:1px solid #0d4a7a;border-radius:6px 6px 0 0;
#                             padding:7px 12px;display:flex;justify-content:space-between;
#                             font-family:'Share Tech Mono',monospace;font-size:0.68rem;">
#                     <span style="color:#00aaff;">CAM-01 &nbsp;/&nbsp; {sc['num']}</span>
#                     <span style="color:{color};">
#                         {'● REC' if sc['status']=='safe' else '⚠ ALERT TRIGGERED'}
#                     </span>
#                 </div>
#                 """, unsafe_allow_html=True)

#                 vfile = video_path(sc["video"])
#                 if vfile.exists():
#                     st.video(str(vfile))
#                 else:
#                     st.markdown(f"""
#                     <div style="background:#000;border:1px solid #0d2137;border-radius:0 0 6px 6px;
#                                 height:240px;display:flex;flex-direction:column;
#                                 align-items:center;justify-content:center;
#                                 font-family:'Share Tech Mono',monospace;color:#0d4a7a;">
#                         <div style="font-size:1.5rem;margin-bottom:8px;">📷</div>
#                         <div style="letter-spacing:2px;font-size:0.72rem;">VIDEO NOT FOUND</div>
#                         <div style="font-size:0.6rem;margin-top:6px;color:#334455;">
#                             {sc['video']}
#                         </div>
#                     </div>
#                     """, unsafe_allow_html=True)

#                 # Verdict banner
#                 cls = "alert-safe" if sc["status"] == "safe" else "alert-warning"
#                 st.markdown(f'<div class="alert-box {cls}" style="margin-top:8px;">{sc["verdict"]}</div>',
#                             unsafe_allow_html=True)

#             with col_d:
#                 # Detection details
#                 info_card("DETECTION DETAILS", sc["details"])

#                 # Timeline
#                 st.markdown("""
#                 <div class="info-card-title" style="margin-top:10px;">
#                     EVENT TIMELINE
#                 </div>
#                 """, unsafe_allow_html=True)

#                 for t_time, t_type, t_msg in sc["timeline"]:
#                     dot = "t-dot-safe" if t_type == "safe" else "t-dot-alert"
#                     st.markdown(f"""
#                     <div class="timeline-event">
#                         <span class="t-time">{t_time}</span>
#                         <span class="{dot}"></span>
#                         <span style="color:{'#c8d8e8' if t_type=='safe' else '#ff7777'};">{t_msg}</span>
#                     </div>
#                     """, unsafe_allow_html=True)


# # ══════════════════════════════════════
# # PAGE 3 — DASHBOARD
# # ══════════════════════════════════════
# elif "Dashboard" in page:

#     st.markdown("""
#     <div style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;
#                 color:#607080;letter-spacing:3px;margin-bottom:1rem;">
#         OPERATIONS DASHBOARD — TODAY'S SUMMARY
#     </div>
#     """, unsafe_allow_html=True)

#     # ── Top stats row ──
#     c1, c2, c3, c4, c5 = st.columns(5)
#     with c1: dash_stat("4",  "SCENARIOS TESTED",  "stat-blue")
#     with c2: dash_stat("3",  "ALERTS TRIGGERED",  "stat-red")
#     with c3: dash_stat("1",  "SAFE EXITS",         "stat-green")
#     with c4: dash_stat("2",  "CHILDREN FLAGGED",   "stat-amber")
#     with c5: dash_stat("75%","ALERT RATE",          "stat-red")

#     st.markdown("<br>", unsafe_allow_html=True)

#     col_map, col_alerts = st.columns([1.3, 1], gap="medium")

#     with col_map:
#         # Store map
#         st.markdown("""
#         <div class="store-map">
#             <div style="color:#007acc;letter-spacing:3px;font-size:0.65rem;margin-bottom:16px;">
#                 STORE LAYOUT — CAM COVERAGE
#             </div>

#             <div style="border:1px solid #0d4a7a;border-radius:4px;padding:20px;
#                         min-height:160px;position:relative;background:#030810;">

#                 <div style="position:absolute;top:10px;left:50%;transform:translateX(-50%);
#                             border:1px solid #0d4a7a;border-radius:3px;padding:4px 20px;
#                             font-size:0.65rem;color:#607080;letter-spacing:2px;">
#                     STORE ENTRANCE
#                 </div>

#                 <div style="position:absolute;top:10px;left:14px;">
#                     <span class="cam-dot"></span>
#                     <span style="color:#00ff88;font-size:0.65rem;">CAM-01</span>
#                 </div>

#                 <div style="position:absolute;bottom:14px;left:50%;transform:translateX(-50%);
#                             font-size:0.65rem;color:#607080;letter-spacing:2px;">
#                     SHOPPING FLOOR (NOT MONITORED)
#                 </div>

#                 <div style="position:absolute;bottom:14px;right:14px;
#                             font-size:0.65rem;color:#334455;">
#                     EXIT ZONE
#                 </div>
#             </div>

#             <div style="margin-top:12px;font-size:0.7rem;color:#607080;">
#                 <span class="cam-dot"></span> Active camera &nbsp;|&nbsp;
#                 Monitoring: Entry + Exit checkpoint only
#             </div>
#         </div>
#         """, unsafe_allow_html=True)

#         st.markdown("<br>", unsafe_allow_html=True)

#         # Alert breakdown
#         info_card("ALERT BREAKDOWN TODAY", {
#             "🚨 child_alone"       : "1 event",
#             "🚨 adult_left_child"  : "1 event",
#             "🚨 wrong_group"       : "1 event",
#             "✅ clean_exit"        : "1 event",
#             "⏱ timeout"           : "0 events",
#         })

#     with col_alerts:
#         # Full alert log
#         st.markdown("""
#         <div class="info-card-title">ALERT LOG — ALL EVENTS</div>
#         """, unsafe_allow_html=True)

#         log_events = [
#             ("10:02:14", "safe",  "G1 — Normal entry detected"),
#             ("10:02:16", "safe",  "Child ID:7 linked to Group G1"),
#             ("10:05:22", "safe",  "G2 — Entry detected"),
#             ("10:05:24", "safe",  "Child ID:7 linked to Group G2"),
#             ("10:07:50", "safe",  "G3 (A) — Entry detected"),
#             ("10:08:10", "safe",  "G4 (B) — Separate entry"),
#             ("10:11:33", "safe",  "G5 — Entry detected"),
#             ("10:18:46", "safe",  "G1 → COMPLETE — Safe exit ✓"),
#             ("10:21:09", "alert", "🚨 child_alone — Case 02"),
#             ("10:24:18", "alert", "🚨 wrong_group — Case 04"),
#             ("10:29:05", "alert", "🚨 adult_left_child — Case 03"),
#         ]

#         for t, typ, msg in log_events:
#             dot = "t-dot-safe" if typ == "safe" else "t-dot-alert"
#             color = "#c8d8e8" if typ == "safe" else "#ff7777"
#             st.markdown(f"""
#             <div class="timeline-event">
#                 <span class="t-time">{t}</span>
#                 <span class="{dot}"></span>
#                 <span style="color:{color};font-size:0.8rem;">{msg}</span>
#             </div>
#             """, unsafe_allow_html=True)

#         st.markdown("<br>", unsafe_allow_html=True)

#         # Group summary table
#         info_card("GROUP SUMMARY", {
#             "G1 — Adults:1 Children:1" : "✅ COMPLETE",
#             "G2 — Adults:1 Children:1" : "🚨 child_alone",
#             "G3 — Adults:1 Children:1" : "🚨 adult_left",
#             "G4 — Adults:1 Children:0" : "✅ COMPLETE",
#             "G5 — Adults:1 Children:1" : "🚨 wrong_group",
#         })