"""
app.py - EcoCiudad CABA - Scanner de Residuos con IA y Red de Puntos Verdes
Tecnicatura en Ciencia de Datos e IA - IFTS N° 11
"""
import base64
from pathlib import Path

import av
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

from utils_vision import load_model, run_inference
from feedback_store import load_feedback, add_correction, get_total_corrections
from puntos_verdes_data import PUNTOS_VERDES_LIST, CENTROS_VERDES_LIST, CATEGORIAS_OFICIALES_GCBA

# Umbral de alta confianza: si supera esto, no se molesta al usuario con consultas
HIGH_CONF_THRESHOLD = 0.60

# ─── Configuración de Página ────────────────────────────────────────────────
st.set_page_config(
    page_title="EcoCiudad CABA | Scanner IA & Puntos Verdes",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_logo_base64() -> str:
    """Carga y codifica en Base64 el archivo logo2.png para renderizado HTML."""
    candidates = [
        Path("logo2.png"),
        Path(__file__).parent / "logo2.png",
        Path(__file__).parent.parent / "logo2.png",
        Path("d:/Backup/2025/Terciario/IA & Datos/4to Cuatri/Proyecto Integrador/logo2.png"),
    ]
    for p in candidates:
        if p.exists():
            try:
                encoded = base64.b64encode(p.read_bytes()).decode("utf-8")
                return f"data:image/png;base64,{encoded}"
            except Exception:
                pass
    return ""


LOGO_B64 = get_logo_base64()

# ─── Estado de Feedback / Memoria ───────────────────────────────────────────
if "feedback_store" not in st.session_state:
    st.session_state.feedback_store = load_feedback()

if "ultimo_residuo" not in st.session_state:
    st.session_state.ultimo_residuo = None

fb_store = st.session_state.feedback_store
total_feedbacks = get_total_corrections(fb_store)

# ─── CSS Estilos: Unified Emerald Tech Design System ────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet">

<style>
/* ─── Variables de Color y Tokens de Diseño (Emerald Tech) ─── */
:root {
    --color-bg: #F8FAFC;
    --color-surface: #FFFFFF;
    --color-primary: #0F172A;
    --color-accent: #10B981;
    --color-accent-hover: #059669;
    --color-accent-light: #ECFDF5;
    --color-accent-dark: #064E3B;
    --color-danger: #EF4444;
    --color-danger-light: #FEF2F2;
    --color-border: #E2E8F0;
    --color-border-dark: #1E293B;
    --color-text-primary: #0F172A;
    --color-text-secondary: #64748B;
    --color-tech-bg: #0F172A;
}

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    background-color: var(--color-bg) !important;
    color: var(--color-text-primary) !important;
}

/* ─── Material Symbols ─── */
.material-symbols-outlined {
    font-family: 'Material Symbols Outlined' !important;
    font-weight: normal;
    font-style: normal;
    font-size: 20px;
    line-height: 1;
    display: inline-block;
    vertical-align: middle;
}

/* ─── Top Navbar Compacto ─── */
.ecoscan-navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #FFFFFF;
    border: 1px solid var(--color-border);
    border-radius: 12px;
    padding: 0.5rem 1.25rem;
    margin-bottom: 1.1rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
}
.ecoscan-navbar-banner {
    max-height: 48px;
    width: auto;
    object-fit: contain;
}
.navbar-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #F1F5F9;
    border: 1px solid var(--color-border);
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.76rem;
    font-weight: 600;
    color: var(--color-text-secondary);
}

/* ─── Header Directo Integrado ─── */
.main-title-container {
    margin-bottom: 1rem;
}
.main-title-tag {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 0.74rem;
    font-weight: 700;
    color: var(--color-accent-dark);
    background: var(--color-accent-light);
    border: 1px solid #A7F3D0;
    padding: 3px 10px;
    border-radius: 6px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 0.35rem;
}
.main-title-text {
    font-size: 1.45rem;
    font-weight: 800;
    color: var(--color-primary);
    margin: 0 0 0.2rem 0;
    letter-spacing: -0.02em;
}
.main-title-sub {
    font-size: 0.88rem;
    color: var(--color-text-secondary);
    margin: 0;
}

/* ─── Pestañas Unificadas (Modern Segmented Tabs) ─── */
[data-testid="stTabs"] {
    margin-bottom: 1rem !important;
}
[data-testid="stTab"] {
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    color: var(--color-text-secondary) !important;
    padding: 0.55rem 1.1rem !important;
    border-radius: 8px 8px 0 0 !important;
    transition: all 0.15s ease !important;
}
[data-testid="stTab"]:hover {
    color: var(--color-accent) !important;
    background-color: #F1F5F9 !important;
}
[data-testid="stTab"][aria-selected="true"] {
    border-bottom: 3px solid var(--color-accent) !important;
    color: var(--color-accent-dark) !important;
    font-weight: 700 !important;
    background-color: #FFFFFF !important;
}

/* ─── Marco Técnico del Visor de Cámara ─── */
.scanner-viewport-box {
    background-color: var(--color-tech-bg);
    border: 1px solid var(--color-border-dark);
    border-radius: 12px;
    padding: 10px 14px;
    margin-bottom: 8px;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.12);
}
.scanner-header-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}
.scanner-status-indicator {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.76rem;
    font-weight: 700;
    color: var(--color-accent);
    letter-spacing: 0.04em;
}
.scanner-hint-text {
    font-size: 0.8rem;
    color: #94A3B8;
    display: flex;
    align-items: center;
    gap: 5px;
}
.live-dot-pulse {
    width: 8px;
    height: 8px;
    background-color: var(--color-accent);
    border-radius: 50%;
    animation: pulse-emerald 2s infinite;
}
@keyframes pulse-emerald {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
    70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

/* Estilo iframe webrtc */
[data-testid="stCustomComponentV1"] iframe {
    border-radius: 12px !important;
    background-color: #0F172A !important;
    border: 1px solid #1E293B !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.18) !important;
}

/* ─── Panel Lateral (Dashboard Widgets) ─── */
.side-widget-card {
    background: #FFFFFF;
    border: 1px solid var(--color-border);
    border-radius: 12px;
    padding: 1.15rem 1.25rem;
    margin-bottom: 0.9rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
}
.side-widget-title {
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--color-text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 6px;
}
.metric-num-highlight {
    font-size: 1.75rem;
    font-weight: 800;
    color: var(--color-accent-dark);
    line-height: 1;
}
.metric-target-sub {
    font-size: 0.85rem;
    color: var(--color-text-secondary);
    font-weight: 500;
}
.progress-track-emerald {
    width: 100%;
    background-color: #F1F5F9;
    height: 7px;
    border-radius: 9999px;
    overflow: hidden;
    margin: 0.5rem 0 0.3rem;
}
.progress-fill-emerald {
    background-color: var(--color-accent);
    height: 100%;
    border-radius: 9999px;
    transition: width 0.4s ease;
}

/* Stats de Impacto */
.impact-stat-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.5rem 0;
    border-bottom: 1px solid #F1F5F9;
}
.impact-stat-row:last-child {
    border-bottom: none;
    padding-bottom: 0;
}
.impact-stat-label {
    font-size: 0.82rem;
    color: var(--color-text-secondary);
    display: flex;
    align-items: center;
    gap: 6px;
}
.impact-stat-val {
    font-size: 0.86rem;
    font-weight: 700;
    color: var(--color-primary);
}

/* Mini Guía de Disposición */
.guide-mini-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 0.5rem 0;
    font-size: 0.8rem;
    line-height: 1.35;
    border-bottom: 1px solid #F1F5F9;
}
.guide-mini-item:last-child {
    border-bottom: none;
    padding-bottom: 0;
}
.guide-dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    margin-top: 4px;
    flex-shrink: 0;
}

/* ─── Tarjetas de Detección ─── */
.det-card {
    background: #FFFFFF;
    border: 1px solid var(--color-border);
    border-left: 5px solid var(--color-accent);
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 6px rgba(0,0,0,0.03);
}
.det-card.special {
    border-left-color: var(--color-danger);
    background: var(--color-danger-light);
}
.det-card.glass { border-left-color: #0284C7; }
.det-card.paper { border-left-color: #F59E0B; }
.det-card.metal { border-left-color: #64748B; }

.ecoscan-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 10px;
    font-size: 0.78rem;
    font-weight: 700;
    border-radius: 9999px;
    background: var(--color-accent-light);
    color: var(--color-accent-dark);
}
.ecoscan-badge-special {
    background: #FEE2E2;
    color: #991B1B;
}

/* Feedback Box */
.fb-container-stitch {
    background: #FFFFFF;
    border: 1.5px dashed #CBD5E1;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-top: 0.8rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02);
}

/* ─── Nearby Green Points List ─── */
.nearby-center-item {
    background: #FFFFFF;
    border: 1px solid var(--color-border);
    border-radius: 12px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.65rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    transition: all 0.15s ease;
}
.nearby-center-item:hover {
    background-color: #F8FAFC;
    border-color: var(--color-accent);
    transform: translateY(-1px);
}
.nearby-item-left {
    display: flex;
    align-items: center;
    gap: 12px;
}
.nearby-item-icon {
    width: 40px;
    height: 40px;
    background-color: #F1F5F9;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--color-text-secondary);
    flex-shrink: 0;
}
.nearby-center-item:hover .nearby-item-icon {
    background-color: var(--color-accent-light);
    color: var(--color-accent-dark);
}
.nearby-item-name {
    font-size: 0.88rem;
    font-weight: 700;
    color: var(--color-text-primary);
}
.nearby-item-meta {
    font-size: 0.76rem;
    color: var(--color-text-secondary);
}

/* ─── Botones Globales (UI Kit) ─── */
[data-testid="stButton"] button {
    border-radius: 8px !important;
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    padding: 0.4rem 0.95rem !important;
    border: 1px solid var(--color-border) !important;
    background: #FFFFFF !important;
    color: var(--color-text-primary) !important;
    transition: all 0.15s ease !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
}
[data-testid="stButton"] button:hover {
    background: var(--color-accent-light) !important;
    border-color: var(--color-accent) !important;
    color: var(--color-accent-dark) !important;
    transform: translateY(-1px) !important;
}
[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid var(--color-border) !important;
    border-radius: 10px !important;
    margin-bottom: 0.5rem !important;
}
[data-testid="stMetricValue"] {
    color: var(--color-accent-dark) !important;
    font-weight: 800 !important;
}
[data-testid="stMetricLabel"] {
    color: var(--color-text-secondary) !important;
    font-weight: 600 !important;
    font-size: 0.76rem !important;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)


# ─── Banner Principal Superior ──────────────────────────────────────────────
if LOGO_B64:
    banner_html = f'<img src="{LOGO_B64}" class="ecoscan-navbar-banner" alt="EcoCiudad Banner">'
else:
    banner_html = '<div style="font-size: 1.3rem; font-weight: 700; color: #0F172A; display: flex; align-items: center; gap: 8px;"><span class="material-symbols-outlined" style="font-size: 26px; color: #10B981;">recycling</span> EcoCiudad Scanner</div>'

st.markdown(f"""
<header class="ecoscan-navbar">
  <div>
    {banner_html}
  </div>
  <div class="navbar-badge">
    <span class="live-dot-pulse"></span>
    <span>IFTS N° 11 · CABA 2026</span>
  </div>
</header>
""", unsafe_allow_html=True)

# ─── Variables y Configuración ──────────────────────────────────────────────
model_choice = "yolov8s_world" if Path("yolov8s-worldv2.pt").exists() else "waste_specialized"
CONF_THRESH = 0.12
FILTER_PEOPLE = True

# ─── Carga del Modelo ────────────────────────────────────────────────────────
model = load_model(model_choice)

# ─── Dashboard y Layout Principal en 2 Columnas (70% / 30%) ───────────────────
daily_current = 124 + total_feedbacks
daily_target = 200
progress_pct = min(100, int((daily_current / daily_target) * 100))

col_main, col_side = st.columns([2.3, 1], gap="large")

with col_main:
    # Encabezado directo y limpio sin tarjeta gigante envolvente
    st.markdown("""
<div class="main-title-container">
  <div class="main-title-tag">
    <span class="material-symbols-outlined" style="font-size:14px; color:#064E3B;">psychology</span>
    Visión Computacional & Reciclaje
  </div>
  <h1 class="main-title-text">Identificador Inteligente de Residuos</h1>
  <p class="main-title-sub">Clasificación asistida por IA y geolocalización de Puntos Verdes oficiales de la Ciudad de Buenos Aires.</p>
</div>
""", unsafe_allow_html=True)

    # Navegación unificada de pestañas
    tab_live, tab_photo, tab_pv = st.tabs([
        "📹 Escanear en Vivo",
        "📁 Subir Archivo / Foto",
        "📍 Red de Puntos Verdes"
    ])

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 1 — Cámara en Vivo
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_live:
        st.markdown("""
<div class="scanner-viewport-box">
  <div class="scanner-header-bar">
    <div class="scanner-status-indicator">
      <span class="live-dot-pulse"></span>
      <span>VISOR IA ACTIVO</span>
    </div>
    <div class="scanner-hint-text">
      <span class="material-symbols-outlined" style="font-size:16px; color:#10B981;">center_focus_strong</span>
      Mostrá el residuo centrado · Detección en tiempo real
    </div>
    <span style="font-size:0.72rem; color:#64748B; font-weight:600;">CABA · 30 FPS</span>
  </div>
</div>
""", unsafe_allow_html=True)

        RTC_CONFIG = RTCConfiguration({"iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:stun.cloudflare.com:3478"]},
            {"urls": ["stun:stun.services.mozilla.com"]},
        ]})

        RTC_TRANSLATIONS = {
            "start": "Iniciar Escaneo",
            "stop": "Detener Escaneo",
            "select_device": "Seleccionar Cámara",
            "select_camera": "Cámara",
            "device_ask_permission": "Permití el acceso a la cámara para escanear.",
            "device_not_available": "No se detectó cámara disponible.",
            "device_access_denied": "Acceso a la cámara denegado.",
        }

        def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
            img_bgr = frame.to_ndarray(format="bgr24")
            annotated, _ = run_inference(
                model,
                img_bgr,
                conf_threshold=CONF_THRESH,
                filter_people=FILTER_PEOPLE,
                feedback_store=fb_store,
            )
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

        webrtc_streamer(
            key=f"stream-{model_choice}",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIG,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            desired_playing_state=True,
            async_processing=True,
            translations=RTC_TRANSLATIONS,
        )
        st.caption("💡 Para escanear varios residuos a la vez o corregir materiales para que el sistema aprenda, usá la pestaña 'Subir Archivo / Foto'.")


    # TAB 2 — Foto / Archivo (Carga de imágenes sin conflicto de cámara)
    with tab_photo:
        col_in, col_out = st.columns([1, 1], gap="large")

        with col_in:
            st.subheader("📥 Carga de Archivo de Imagen")
            uploaded = st.file_uploader(
                "Subir o arrastrar foto de tus residuos:",
                type=["jpg", "jpeg", "png", "webp", "bmp"],
                help="Soporta fotos con 1 o múltiples residuos simultáneos"
            )
            img_pil = None
            if uploaded:
                img_pil = Image.open(uploaded).convert("RGB")

        with col_out:
            st.subheader("🔍 Diagnóstico Multi-Residuo")

            if img_pil is None:
                st.markdown("""
<div style="background:#F8FAFC; border: 1.5px dashed #CBD5E1; border-radius: 12px; padding: 2.2rem 1.5rem; text-align: center;">
  <div style="width: 48px; height: 48px; background: #ECFDF5; color: #059669; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 0.75rem;">
    <span class="material-symbols-outlined" style="font-size: 24px;">upload_file</span>
  </div>
  <h4 style="margin: 0 0 0.3rem 0; font-weight: 700; color: #0F172A; font-size: 1.05rem;">Esperando imagen para diagnóstico</h4>
  <p style="margin: 0; color: #64748B; font-size: 0.84rem; line-height: 1.4;">Subí una foto a la izquierda con 1 o más materiales (potes, botellas, vapers, vasos, latas, etc.).</p>
</div>
""", unsafe_allow_html=True)
            else:
                frame_bgr = frame_bgr = np.array(img_pil)[:, :, ::-1].copy()
                annotated_bgr, detections = run_inference(
                    model,
                    frame_bgr,
                    conf_threshold=CONF_THRESH,
                    filter_people=FILTER_PEOPLE,
                    feedback_store=fb_store,
                )
                annotated_rgb = annotated_bgr[:, :, ::-1].copy()
                st.image(annotated_rgb, use_container_width=True)

                if not detections:
                    st.warning("No se detectó ningún residuo claro con el umbral actual. Probá con una toma más cercana o mejor iluminación.")
                else:
                    st.session_state.ultimo_residuo = detections[0]
                    has_special = any(d.get("es_especial", False) for d in detections)
                    top_conf = detections[0]["confianza"]
                    is_held = detections[0]["is_held_in_center"]

                    m1, m2 = st.columns(2)
                    m1.metric("Residuos detectados", len(detections))
                    m2.metric("Confianza máxima", f"{top_conf:.0%}")

                    st.markdown(f"### 📋 {len(detections)} Residuo{'s' if len(detections) > 1 else ''} Identificado{'s' if len(detections) > 1 else ''}")

                    for idx, det in enumerate(detections):
                        tipo = det["tipo"]
                        label = det["label"]
                        es_esp = det.get("es_especial", False)
                        card_cls = "det-card special" if es_esp else "det-card"
                        if not es_esp:
                            if "Vidrio" in label:
                                card_cls += " glass"
                            elif "Carton" in label or "Papel" in label:
                                card_cls += " paper"
                            elif "Metal" in label or "Lata" in label:
                                card_cls += " metal"
                            elif "CD" in label or "Disco" in label:
                                card_cls += " glass"

                        held_tag = " · 🖐️ En mano" if det["is_held_in_center"] else ""
                        boost_tag = " ✦ Auto-aprendido" if det.get("boost_applied") else ""
                        badge_class = "ecoscan-badge ecoscan-badge-special" if es_esp else "ecoscan-badge"
                        prefix = f"#{idx+1} " if len(detections) > 1 else ""

                        st.markdown(f"""
<div class="{card_cls}">
  <div style="display:flex; justify-content:space-between; align-items:center;">
    <span style="font-size:1.1rem; font-weight:700; color:#0F172A;">{prefix}{det['emoji']} {det['label']}</span>
    <div>
      <span class="{badge_class}">{det['confianza']:.0%}{boost_tag}</span>
    </div>
  </div>
  <div style="margin-top:0.5rem; font-size:0.9rem; line-height:1.5;">
    <b>Destino:</b> {det['tipo']}{held_tag}<br>
    <b>¿Qué hacer?:</b> {det['accion']}
  </div>
</div>
""", unsafe_allow_html=True)

                    # ── Opcionalidad de Puntos Verdes según el tipo de residuo ──
                    st.divider()
                    if has_special:
                        st.error("🔴 **RESIDUO ESPECIAL DETECTADO:** Al menos uno de los materiales **NO debe tirarse en el contenedor verde ni negro**. Requiere Punto Verde o Punto Verde Móvil.")
                        st.info("👉 Consultá la pestaña **'📍 Red de Puntos Verdes'** para ver el mapa y horarios del punto más cercano a tu Comuna.")
                    else:
                        st.success("🟢 **Disposición en Origen:** Podés depositar estos materiales directamente en el **Contenedor Verde** de tu cuadra (limpios y secos).")
                        with st.expander("📍 ¿Querés llevarlos a un Punto Verde o Centro de Cooperativa? (Opcional)"):
                            st.markdown("""
Si no tenés contenedor verde cerca o preferís entregarlos en mano:
- **Puntos Verdes:** Miércoles a Domingos de 11 a 14:30 h y de 15 a 19 h.
- **Centros Verdes:** Plantas de cooperativas de recuperadores urbanos.
- Consulta el mapa completo en la pestaña **'📍 Red de Puntos Verdes'**.
""")

                    # ── Auto-entrenamiento / Corrección ──
                    st.markdown("""
<div class="fb-container-stitch">
  <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
    <span class="material-symbols-outlined" style="color:#059669; font-size:20px;">psychology</span>
    <b style="color:#0F172A; font-size:0.95rem;">Auto-Entrenamiento: ¿Querés enseñarle qué material es?</b>
  </div>
  <span style="font-size:0.86rem; color:#64748B;">
    Si el sistema dudó o querés enseñarle un objeto ambiguo (como un pote o vaper), seleccionalo abajo. El aprendizaje se guardará permanentemente en el servidor.
  </span>
</div>
""", unsafe_allow_html=True)

                    quick_materials = [
                        ("plastic",   "🧴 Plástico / Pote"),
                        ("vape",      "🔋 Vaper / RAEE"),
                        ("cd",        "💿 CD / DVD"),
                        ("cardboard", "📦 Cartón"),
                        ("metal",     "🥫 Metal / Lata"),
                        ("glass",     "🍾 Vidrio"),
                        ("paper",     "📄 Papel"),
                        ("oil",       "🍳 Aceite AVU"),
                    ]

                    st.write("")
                    cols_btn = st.columns(4)
                    for idx, (mat_key, mat_label) in enumerate(quick_materials):
                        col_target = cols_btn[idx % 4]
                        if col_target.button(mat_label, key=f"btn_fb_{mat_key}", use_container_width=True):
                            new_store = add_correction(
                                predicted_cls=detections[0]["clase"],
                                correct_cls=mat_key,
                                is_held_in_center=is_held,
                                store=st.session_state.feedback_store
                            )
                            st.session_state.feedback_store = new_store
                            st.success(f"🧠 ¡Aprendizaje registrado! Se reforzó el peso para '{mat_label}' en el servidor.")
                            st.rerun()



    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 3 — Puntos y Centros Verdes
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_pv:
        st.subheader("📍 Red de Puntos Verdes y Centros Verdes de CABA")
        st.markdown("""
Visualizá las **32 plazas con Puntos Verdes** y las **14 plantas de Centros Verdes** gestionadas por Cooperativas de Recuperadores Urbanos en la Ciudad de Buenos Aires.
""")

        COMUNAS_BARRIOS = {
            1: "1 (Constitución, Montserrat, Puerto Madero, Retiro, San Nicolás, San Telmo)",
            2: "2 (Recoleta)",
            3: "3 (Balvanera, San Cristóbal)",
            4: "4 (Barracas, La Boca, Nueva Pompeya, Parque Patricios)",
            5: "5 (Almagro, Boedo)",
            6: "6 (Caballito)",
            7: "7 (Flores, Parque Chacabuco)",
            8: "8 (Villa Lugano, Villa Riachuelo, Villa Soldati)",
            9: "9 (Liniers, Mataderos, Parque Avellaneda)",
            10: "10 (Floresta, Monte Castro, Vélez Sarsfield, Versalles, Villa Luro, Villa Real)",
            11: "11 (Villa del Parque, Villa Devoto, Villa General Mitre, Villa Santa Rita)",
            12: "12 (Coghlan, Saavedra, Villa Pueyrredón, Villa Urquiza)",
            13: "13 (Belgrano, Colegiales, Núñez)",
            14: "14 (Palermo)",
            15: "15 (Agronomía, Chacarita, La Paternal, Parque Chas, Villa Crespo, Villa Ortúzar)",
        }

        # Coordenadas aproximadas de centroide por comuna
        COMUNA_CENTROIDES = {
            1: (-34.6037, -58.3750), 2: (-34.5910, -58.3980), 3: (-34.6150, -58.4030),
            4: (-34.6430, -58.3950), 5: (-34.6140, -58.4180), 6: (-34.6178, -58.4345),
            7: (-34.6310, -58.4520), 8: (-34.6680, -58.4580), 9: (-34.6530, -58.4950),
            10: (-34.6260, -58.5000), 11: (-34.6020, -58.5030), 12: (-34.5610, -58.4940),
            13: (-34.5530, -58.4620), 14: (-34.5820, -58.4240), 15: (-34.5840, -58.4580),
        }

        col_f1, col_f2 = st.columns([1.2, 1])
        with col_f1:
            comunas_disponibles = ["Todas"] + list(range(1, 16))
            filtro_comuna = st.selectbox(
                "Tu Comuna / Barrio de referencia:",
                comunas_disponibles,
                format_func=lambda c: "CABA (Todas las Comunas)" if c == "Todas" else COMUNAS_BARRIOS.get(c, str(c)),
                key="filtro_comuna_pv"
            )
        with col_f2:
            tipo_capa = st.radio(
                "Capa en mapa:",
                ["📍 Puntos Verdes (32)", "🏭 Centros Verdes (14)", "🌐 Todos (46)"],
                horizontal=True
            )

        df_pv = pd.DataFrame(PUNTOS_VERDES_LIST)
        df_cv = pd.DataFrame(CENTROS_VERDES_LIST)

        if "Puntos Verdes" in tipo_capa:
            df_display = df_pv.copy()
        elif "Centros Verdes" in tipo_capa:
            df_display = df_cv.copy()
        else:
            df_display = pd.concat([df_pv, df_cv], ignore_index=True)

        # Coordenada de referencia para cálculo según Comuna seleccionada
        if filtro_comuna != "Todas" and int(filtro_comuna) in COMUNA_CENTROIDES:
            ref_lat, ref_lon = COMUNA_CENTROIDES[int(filtro_comuna)]
            df_filtrado_mapa = df_display[df_display["comuna"] == int(filtro_comuna)]
            if df_filtrado_mapa.empty:
                df_filtrado_mapa = df_display.copy()
        else:
            ref_lat, ref_lon = -34.6037, -58.3816
            df_filtrado_mapa = df_display.copy()

        # ── Sección Puntos Verdes Cercanos (Dinámica con datos reales CABA) ──
        st.markdown("### 📍 Puntos Verdes más cercanos a tu ubicación")

        todos_puntos = PUNTOS_VERDES_LIST + CENTROS_VERDES_LIST

        def calc_dist(pt):
            dlat = np.radians(pt["lat"] - ref_lat)
            dlon = np.radians(pt["lon"] - ref_lon)
            a = np.sin(dlat / 2)**2 + np.cos(np.radians(ref_lat)) * np.cos(np.radians(pt["lat"])) * np.sin(dlon / 2)**2
            return 6371.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        puntos_con_dist = []
        for p in todos_puntos:
            p_copy = dict(p)
            p_copy["dist_km"] = calc_dist(p)
            puntos_con_dist.append(p_copy)

        puntos_con_dist.sort(key=lambda x: x["dist_km"])
        top3_cercanos = puntos_con_dist[:3]

        cards_html = ""
        for p in top3_cercanos:
            is_cv = "Centro Verde" in p.get("tipo", "")
            is_esp = "Especial" in p.get("tipo", "")

            if is_cv:
                icon_name = "factory"
                tag_color = "#19273A"
            elif is_esp:
                icon_name = "battery_charging_full"
                tag_color = "#EF4444"
            else:
                icon_name = "recycling"
                tag_color = "#059669"

            cards_html += f"""<div class="nearby-center-item"><div class="nearby-item-left"><div class="nearby-item-icon" style="color: {tag_color};"><span class="material-symbols-outlined">{icon_name}</span></div><div><div class="nearby-item-name">{p['nombre']}</div><div class="nearby-item-meta">{p.get('tipo', 'Punto Verde')} • {p.get('direccion', '')} • Comuna {p['comuna']} • <b>{p['dist_km']:.1f} km</b></div></div></div><span class="material-symbols-outlined" style="color: #94A3B8; font-size: 20px;">chevron_right</span></div>"""

        st.markdown(f'<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 10px; margin-bottom: 1.25rem;">{cards_html}</div>', unsafe_allow_html=True)

        st.markdown("#### 🗺️ Ubicaciones Geolocalizadas (Pasá el cursor o tocá un punto)")

        # Colores pydeck: Verde para Punto Verde, Rojo para Especial, Azul para Centro Verde
        def get_color_pdk(row):
            t = str(row.get("tipo", ""))
            if "Especial" in t:
                return [239, 68, 68, 220]
            elif "Centro Verde" in t:
                return [15, 23, 42, 220]
            else:
                return [16, 185, 129, 220]

        df_map_render = df_filtrado_mapa.copy()
        df_map_render["color"] = df_map_render.apply(get_color_pdk, axis=1)

        layer_points = pdk.Layer(
            "ScatterplotLayer",
            data=df_map_render,
            get_position=["lon", "lat"],
            get_color="color",
            get_radius=220,
            pickable=True,
            auto_highlight=True,
        )

        view_state = pdk.ViewState(
            latitude=float(df_map_render["lat"].mean()) if not df_map_render.empty else -34.6037,
            longitude=float(df_map_render["lon"].mean()) if not df_map_render.empty else -58.3816,
            zoom=12 if filtro_comuna != "Todas" else 11.2,
            pitch=0,
        )

        st.pydeck_chart(
            pdk.Deck(
                layers=[layer_points],
                initial_view_state=view_state,
                tooltip={
                    "html": "<div style='font-family: sans-serif; font-size: 13px; line-height: 1.4;'>"
                            "<b style='font-size: 14px; color: #A7F3D0;'>{nombre}</b><br/>"
                            "📍 <b>Dirección:</b> {direccion}<br/>"
                            "🏷️ <b>Tipo:</b> {tipo}<br/>"
                            "🕒 <b>Horario:</b> {horario}<br/>"
                            "📌 <b>Comuna:</b> {comuna}<br/>"
                            "🟢 <b>Estado:</b> {estado}</div>",
                    "style": {
                        "backgroundColor": "#0F172A",
                        "color": "#FFFFFF",
                        "borderRadius": "10px",
                        "padding": "10px 14px",
                        "boxShadow": "0 4px 12px rgba(0,0,0,0.2)"
                    }
                },
                map_style="light"
            )
        )

        st.markdown(f"#### 📋 Detalle ({len(df_filtrado_mapa)} ubicaciones)")
        for _, row in df_filtrado_mapa.iterrows():
            tipo = row.get("tipo", "Punto Verde")
            admin = f" · Admin: {row['administra']}" if "administra" in row and pd.notnull(row["administra"]) else ""
            estado_badge = "🟢 " + str(row.get("estado", "Operativo")) if "Abierto" in str(row.get("estado", "")) or "Operativo" in str(row.get("estado", "")) else "🔴 " + str(row.get("estado", ""))

            with st.expander(f"{tipo}: {row['nombre']} — Comuna {row['comuna']}{admin} ({estado_badge})"):
                st.markdown(f"""
- **Dirección:** `{row['direccion']}`
- **Barrio / Comuna:** `{row.get('barrio', '')}` (Comuna {row['comuna']})
- **Horario:** `{row.get('horario', 'Consultar')}`
- **Estado:** `{row.get('estado', 'Operativo')}`
""")

        st.divider()
        st.markdown("### 📚 Guía Oficial GCBA: Residuos Especiales")
        for cat_id, cat_info in CATEGORIAS_OFICIALES_GCBA.items():
            with st.expander(f"{cat_info['titulo']}"):
                st.markdown(f"""
- **Materiales recibidos:** {cat_info['items']}
- **Condición de entrega:** {cat_info['condicion']}
- **Destino obligatorio Punto Verde:** `{'Sí (No tirar a tachos de calle)' if cat_info['obligatorio_punto_verde'] else 'Opcional'}`
""")

# ─── Panel Lateral (Dashboard Widgets 30%) ────────────────────────────────────
with col_side:
    # ── Widget 1: Meta Diaria ──
    st.markdown(f"""
<div class="side-widget-card">
  <div class="side-widget-title">
    <span class="material-symbols-outlined" style="font-size:16px; color:#10B981;">flag</span>
    Meta Diaria de Clasificación
  </div>
  <div style="display:flex; align-items:baseline; gap:6px;">
    <span class="metric-num-highlight">{daily_current}</span>
    <span class="metric-target-sub">/ {daily_target} residuos</span>
  </div>
  <div class="progress-track-emerald">
    <div class="progress-fill-emerald" style="width: {progress_pct}%;"></div>
  </div>
  <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#64748B; font-weight:600;">
    <span>Progreso barrial</span>
    <span style="color:#059669;">{progress_pct}% completado</span>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Widget 2: Impacto Ecológico Estimado ──
    kg_estimados = daily_current * 0.25
    co2_estimado = daily_current * 0.38
    st.markdown(f"""
<div class="side-widget-card">
  <div class="side-widget-title">
    <span class="material-symbols-outlined" style="font-size:16px; color:#10B981;">eco</span>
    Impacto Ecológico Estimado
  </div>
  <div class="impact-stat-row">
    <span class="impact-stat-label">
      <span class="material-symbols-outlined" style="font-size:16px; color:#059669;">recycling</span>
      Recuperables reciclados
    </span>
    <span class="impact-stat-val">~{kg_estimados:.1f} kg</span>
  </div>
  <div class="impact-stat-row">
    <span class="impact-stat-label">
      <span class="material-symbols-outlined" style="font-size:16px; color:#059669;">cloud_off</span>
      CO₂ evitado en CABA
    </span>
    <span class="impact-stat-val">~{co2_estimado:.1f} kg</span>
  </div>
  <div class="impact-stat-row">
    <span class="impact-stat-label">
      <span class="material-symbols-outlined" style="font-size:16px; color:#059669;">domain</span>
      Trazabilidad cooperativas
    </span>
    <span class="impact-stat-val">100% GCBA</span>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Widget 3: Último Diagnóstico o Guía Rápida ──
    ultimo = st.session_state.get("ultimo_residuo")
    if ultimo:
        det_label = ultimo.get("label", "Residuo")
        det_emoji = ultimo.get("emoji", "♻️")
        det_conf = ultimo.get("confianza", 0.0)
        det_tipo = ultimo.get("tipo", "Contenedor Verde")
        det_esp = ultimo.get("es_especial", False)
        badge_bg = "#FEE2E2" if det_esp else "#ECFDF5"
        badge_txt = "#991B1B" if det_esp else "#064E3B"
        border_col = "#EF4444" if det_esp else "#10B981"
        st.markdown(f"""
<div class="side-widget-card" style="border-left: 4px solid {border_col};">
  <div class="side-widget-title">
    <span class="material-symbols-outlined" style="font-size:16px; color:{border_col};">history</span>
    Último Residuo Detectado
  </div>
  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
    <span style="font-weight:700; font-size:0.95rem; color:#0F172A;">{det_emoji} {det_label}</span>
    <span style="background:{badge_bg}; color:{badge_txt}; padding:2px 8px; border-radius:9999px; font-size:0.75rem; font-weight:700;">{det_conf:.0%}</span>
  </div>
  <div style="font-size:0.8rem; color:#64748B; line-height:1.4;">
    <b>Destino:</b> {det_tipo}
  </div>
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown("""
<div class="side-widget-card">
  <div class="side-widget-title">
    <span class="material-symbols-outlined" style="font-size:16px; color:#10B981;">lightbulb</span>
    Guía Rápida de Separación
  </div>
  <div class="guide-mini-item">
    <div class="guide-dot" style="background:#10B981;"></div>
    <div><b>Contenedor Verde:</b> Plásticos, papeles, cartones, metales y vidrios secos y limpios.</div>
  </div>
  <div class="guide-mini-item">
    <div class="guide-dot" style="background:#EF4444;"></div>
    <div><b>Punto Verde:</b> RAEE, pilas, lámparas y aceite vegetal usado.</div>
  </div>
  <div class="guide-mini-item">
    <div class="guide-dot" style="background:#64748B;"></div>
    <div><b>Contenedor Negro:</b> Restos húmedos y residuos no reciclables.</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Widget 4: Motor de Visión IA ──
    st.markdown(f"""
<div class="side-widget-card">
  <div class="side-widget-title">
    <span class="material-symbols-outlined" style="font-size:16px; color:#10B981;">neurology</span>
    Motor de Visión IA
  </div>
  <div style="display:flex; flex-direction:column; gap:8px; font-size:0.8rem; color:#64748B;">
    <div style="display:flex; justify-content:space-between;">
      <span>Modelo activo:</span>
      <b style="color:#0F172A;">YOLOv8 Residuos</b>
    </div>
    <div style="display:flex; justify-content:space-between;">
      <span>Filtro de personas:</span>
      <span style="color:#059669; font-weight:600;">Activo (Automático)</span>
    </div>
    <div style="display:flex; justify-content:space-between;">
      <span>Aprendizaje continuo:</span>
      <b style="color:#0F172A;">{total_feedbacks} correcciones</b>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)




# ─── Footer ──────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center; color:#717973; font-size:0.82rem; margin:1rem 0;'>"
    "EcoCiudad CABA · Tecnicatura en Ciencia de Datos e IA · IFTS N° 11 · 2026"
    "</p>",
    unsafe_allow_html=True,
)
