import cv2
import time
import threading
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

from detector import ObjectDetector, get_decision

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ObstacleIQ",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');
:root {
  --bg:#0a0c10; --surface:#111318; --surface2:#161b24;
  --border:#1e2330; --accent:#00e5ff; --text:#e8ecf5; --muted:#5a6478;
}
*,*::before,*::after{box-sizing:border-box}
*{font-family:'Syne',sans-serif}
html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"],.main{
  background-color:var(--bg)!important;color:var(--text)}
#MainMenu,footer,header,[data-testid="stSidebar"],[data-testid="collapsedControl"]{display:none!important}
.block-container{padding:0 1.5rem 2rem!important;max-width:100%!important}
.topbar{display:flex;align-items:center;justify-content:space-between;
  padding:.9rem 0 .75rem;border-bottom:1px solid var(--border);margin-bottom:1rem}
.topbar-brand{display:flex;align-items:center;gap:12px}
.topbar-icon{width:36px;height:36px;background:linear-gradient(135deg,#00e5ff,#0055aa);
  border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:1.1rem}
.topbar-title{font-size:1.3rem;font-weight:800;letter-spacing:-.02em;line-height:1}
.topbar-sub{font-size:.52rem;color:#3a4860;font-family:'Space Mono',monospace;
  letter-spacing:.12em;text-transform:uppercase;margin-top:2px}
.topbar-badge{font-family:'Space Mono',monospace;font-size:.62rem;
  background:rgba(0,229,255,.08);border:1px solid rgba(0,229,255,.2);
  color:var(--muted);border-radius:6px;padding:.3rem .7rem}
[data-testid="stExpander"]{background:var(--surface)!important;
  border:1px solid var(--border)!important;border-radius:12px!important;margin-bottom:1rem!important}
[data-testid="stExpander"] summary{font-family:'Space Mono',monospace!important;font-size:.7rem!important;
  font-weight:700!important;letter-spacing:.12em!important;text-transform:uppercase!important;
  color:var(--muted)!important;padding:.75rem 1rem!important}
[data-testid="stExpander"] summary:hover{color:var(--accent)!important}
[data-testid="stExpanderDetails"]{padding:.5rem 1rem 1rem!important}
.stSlider label,.stSelectbox label,.stNumberInput label{
  font-family:'Space Mono',monospace!important;font-size:.68rem!important;
  letter-spacing:.06em!important;color:var(--muted)!important;text-transform:uppercase!important}
[data-baseweb="select"]>div{background:var(--surface2)!important;border-color:var(--border)!important}
[data-baseweb="input"]>div{background:var(--surface2)!important;border-color:var(--border)!important}
input{color:var(--text)!important}
.section-head{font-size:.6rem;letter-spacing:.15em;text-transform:uppercase;color:var(--muted);
  font-family:'Space Mono',monospace;margin-bottom:.7rem;padding-bottom:.35rem;
  border-bottom:1px solid var(--border);font-weight:700}
.metric-card{background:var(--surface);border:1px solid var(--border);border-radius:12px;
  padding:1rem 1.2rem;margin-bottom:.65rem;position:relative;overflow:hidden}
.metric-card::before{content:'';position:absolute;top:0;left:0;width:3px;height:100%;background:var(--accent)}
.metric-label{font-size:.6rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
  color:var(--muted);margin-bottom:.25rem;font-family:'Space Mono',monospace}
.metric-value{font-size:1.45rem;font-weight:800;color:var(--accent);line-height:1}
.metric-unit{font-size:.72rem;color:var(--muted);font-family:'Space Mono',monospace}
.decision-badge{display:inline-block;padding:.45rem 1.1rem;border-radius:50px;font-size:.85rem;
  font-weight:800;letter-spacing:.08em;text-transform:uppercase;font-family:'Space Mono',monospace}
.decision-ROLL_OVER{background:rgba(127,255,110,.12);color:#7fff6e;border:1px solid #7fff6e}
.decision-PUSH{background:rgba(255,229,102,.12);color:#ffe566;border:1px solid #ffe566}
.decision-AVOID{background:rgba(255,79,79,.12);color:#ff4f4f;border:1px solid #ff4f4f}
.decision-NO_OBJECT{background:rgba(90,100,120,.15);color:var(--muted);border:1px solid var(--border)}
.obj-tag{display:inline-block;background:rgba(0,229,255,.1);border:1px solid rgba(0,229,255,.3);
  color:var(--accent);border-radius:6px;padding:.18rem .55rem;font-size:.68rem;
  font-family:'Space Mono',monospace;margin:2px;font-weight:700;text-transform:uppercase}
.obj-tag-human{background:rgba(255,107,53,.12);border-color:rgba(255,107,53,.4);color:#ff6b35}
.conf-bar-wrap{background:var(--border);border-radius:4px;height:4px;width:100%;margin-top:.3rem}
.conf-bar{height:4px;border-radius:4px;background:linear-gradient(90deg,var(--accent),#0099cc)}
.info-box{background:rgba(0,229,255,.04);border:1px solid rgba(0,229,255,.12);border-radius:8px;
  padding:.75rem .9rem;font-size:.75rem;color:#8a9ab5;line-height:1.75}
video{border-radius:10px!important;border:1px solid var(--border)!important}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
for k, v in [("results", []), ("log", []), ("frame_count", 0)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Topbar ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div class="topbar-brand">
    <div class="topbar-icon">🤖</div>
    <div>
      <div class="topbar-title">ObstacleIQ</div>
      <div class="topbar-sub">Obstacle Detection System</div>
    </div>
  </div>
  <div class="topbar-badge">YOLOv8 · WebRTC · Streamlit Cloud</div>
</div>
""", unsafe_allow_html=True)

# ── Settings expander ──────────────────────────────────────────────────────────
with st.expander("⚙  Settings — Car Parameters · Camera · Model", expanded=False):
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.markdown("**🚗 Car Parameters**")
        car_clearance  = st.slider("Ground Clearance (cm)", 1, 20, 4,  key="clearance")
        car_push_force = st.slider("Max Push Force (N)",    1, 50, 10, key="push")
        car_width      = st.slider("Car Width (cm)",        10, 60, 25, key="width")
    with col_b:
        st.markdown("**📷 Camera**")
        focal_length = st.number_input("Focal Length (px)", 100, 2000, 600, key="focal")
        conf_thresh  = st.slider("Confidence Threshold", 0.1, 1.0, 0.45, 0.05, key="conf")
    with col_c:
        st.markdown("**🧠 Model**")
        model_size   = st.selectbox("YOLOv8 Model", ["yolov8n", "yolov8s", "yolov8m"], key="model")
        detect_every = st.slider("Detect Every N Frames", 1, 6, 2, key="every",
                                  help="1=every frame (slower). 3+=faster stream.")
    with col_d:
        st.markdown("**📖 Decision Logic**")
        st.markdown("""
        <div class="info-box">
          <span style="color:#7fff6e;font-family:monospace;font-weight:700">ROLL OVER</span>
          — Height &lt; clearance &amp; light<br>
          <span style="color:#ffe566;font-family:monospace;font-weight:700">PUSH AWAY</span>
          — Manageable mass<br>
          <span style="color:#ff4f4f;font-family:monospace;font-weight:700">AVOID</span>
          — Human / too large / too heavy
        </div>""", unsafe_allow_html=True)

# ── Load detector ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_detector(model_name, conf, focal):
    return ObjectDetector(model_name=model_name, conf_threshold=conf, focal_length_px=focal)

detector = load_detector(model_size, conf_thresh, focal_length)

# ── Video processor — NO av import, uses raw numpy ────────────────────────────
class ObstacleTransformer(VideoProcessorBase):
    def __init__(self):
        self.frame_count  = 0
        self.last_results = []
        self.lock         = threading.Lock()

    def recv(self, frame):
        # frame.to_ndarray is provided by streamlit-webrtc internally;
        # we import av lazily inside recv so it never errors at module load time
        import av as _av  # noqa: F401 — needed by streamlit-webrtc internals only
        img = frame.to_ndarray(format="bgr24")

        self.frame_count += 1
        if self.frame_count % detect_every == 0:
            results = detector.detect(img)
            with self.lock:
                self.last_results = results
                st.session_state["results"]     = results
                st.session_state["frame_count"] = self.frame_count
        else:
            with self.lock:
                results = self.last_results

        annotated = detector.draw_boxes(
            img.copy(), results, car_clearance, car_push_force, car_width
        )
        cv2.putText(annotated, f"Frame {self.frame_count}",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 229, 255), 2, cv2.LINE_AA)

        import av as _av2
        return _av2.VideoFrame.from_ndarray(annotated, format="bgr24")


RTC_CONFIG = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
})

# ── Main layout ────────────────────────────────────────────────────────────────
col_feed, col_info = st.columns([3, 2], gap="large")

with col_feed:
    st.markdown('<div class="section-head">📸 Live Camera Feed</div>', unsafe_allow_html=True)
    ctx = webrtc_streamer(
        key="obstacle-iq",
        video_processor_factory=ObstacleTransformer,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True,
    )
    if ctx.state.playing:
        st.markdown("""
        <div style="font-family:'Space Mono',monospace;font-size:.7rem;color:#7fff6e;
                    margin-top:.5rem;display:flex;align-items:center;gap:8px">
          <span style="width:7px;height:7px;border-radius:50%;background:#7fff6e;display:inline-block;
                       animation:pulse 1.2s infinite"></span>LIVE — detection running
        </div>
        <style>@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}</style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="font-family:'Space Mono',monospace;font-size:.68rem;color:#2a3850;margin-top:.5rem">
          Click <strong style="color:#00e5ff">START</strong> to begin live detection
        </div>""", unsafe_allow_html=True)

# ── Results panel ──────────────────────────────────────────────────────────────
with col_info:
    st.markdown('<div class="section-head">🔍 Detection Results</div>', unsafe_allow_html=True)
    obj_ph      = st.empty()
    decision_ph = st.empty()
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-head">📊 Object Metrics</div>', unsafe_allow_html=True)
    mc1, mc2 = st.columns(2)
    with mc1:
        dist_ph   = st.empty()
        height_ph = st.empty()
    with mc2:
        weight_ph = st.empty()
        conf_ph   = st.empty()
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-head">📋 Detection Log</div>', unsafe_allow_html=True)
    log_ph = st.empty()

# ── Read results from processor ───────────────────────────────────────────────
results   = []
processor = getattr(ctx, "video_processor", None) or getattr(ctx, "video_transformer", None)
if ctx.state.playing and processor:
    with processor.lock:
        results = list(processor.last_results)

is_live = ctx.state.playing

def _idle(ph, label="—"):
    ph.markdown(f"""<div class="metric-card"><div class="metric-label">{label}</div>
    <div class="metric-value" style="color:#1e2330">—</div></div>""", unsafe_allow_html=True)

if not is_live or not results:
    obj_ph.markdown("""<div class="metric-card"><div class="metric-label">Detected Objects</div>
    <div style="color:#2a3040;font-family:monospace;font-size:.78rem;margin-top:.3rem">
    — awaiting stream —</div></div>""", unsafe_allow_html=True)
    decision_ph.markdown(f"""<div class="metric-card"><div class="metric-label">Car Decision</div>
    <span class="decision-badge decision-NO_OBJECT">{"Clear Path" if is_live else "Idle"}</span>
    </div>""", unsafe_allow_html=True)
    for ph in [dist_ph, height_ph, weight_ph, conf_ph]:
        _idle(ph)
else:
    best             = results[0]
    dist             = best["distance_cm"]
    est_h            = best["est_height_cm"]
    est_w            = best["est_weight_g"]
    conf             = best["confidence"]
    decision, reason = get_decision(best, car_clearance, car_push_force)

    tags = "".join(
        f'<span class="obj-tag {"obj-tag-human" if r["label"].lower()=="person" else ""}">'
        f'{r["label"]} {r["confidence"]:.0%}</span>'
        for r in results
    )
    obj_ph.markdown(f"""<div class="metric-card"><div class="metric-label">Detected Objects</div>
    <div style="margin-top:.35rem">{tags}</div></div>""", unsafe_allow_html=True)

    d_cls = decision.replace(" ", "_")
    decision_ph.markdown(f"""<div class="metric-card"><div class="metric-label">Car Decision</div>
    <span class="decision-badge decision-{d_cls}">{decision.replace("_"," ")}</span>
    <div style="font-size:.7rem;color:#4a5568;margin-top:.45rem;font-family:monospace">{reason}</div>
    </div>""", unsafe_allow_html=True)

    dist_ph.markdown(f"""<div class="metric-card"><div class="metric-label">Distance</div>
    <div class="metric-value">{dist:.1f}</div><div class="metric-unit">cm</div></div>""",
    unsafe_allow_html=True)
    height_ph.markdown(f"""<div class="metric-card"><div class="metric-label">Est. Height</div>
    <div class="metric-value">{est_h:.1f}</div><div class="metric-unit">cm</div></div>""",
    unsafe_allow_html=True)
    weight_ph.markdown(f"""<div class="metric-card"><div class="metric-label">Est. Weight</div>
    <div class="metric-value">{est_w:.0f}</div><div class="metric-unit">grams</div></div>""",
    unsafe_allow_html=True)
    conf_ph.markdown(f"""<div class="metric-card"><div class="metric-label">Confidence</div>
    <div class="metric-value">{conf:.0%}</div>
    <div class="conf-bar-wrap"><div class="conf-bar" style="width:{conf*100:.0f}%"></div></div>
    </div>""", unsafe_allow_html=True)

    ts    = time.strftime("%H:%M:%S")
    entry = f"[{ts}] {best['label']} | {dist:.0f}cm | {decision.replace('_',' ')}"
    log   = st.session_state.get("log", [])
    if not log or log[0].split("]")[1] != entry.split("]")[1]:
        log.insert(0, entry)
        st.session_state["log"] = log[:12]

if st.session_state.get("log"):
    rows = "".join(
        f'<div style="font-family:monospace;font-size:.68rem;color:#3a4860;'
        f'padding:.22rem 0;border-bottom:1px solid #111620">{e}</div>'
        for e in st.session_state["log"]
    )
    log_ph.markdown(
        f'<div style="background:#0d0f14;border-radius:8px;padding:.55rem .8rem">{rows}</div>',
        unsafe_allow_html=True)

if is_live:
    time.sleep(0.4)
    st.rerun()