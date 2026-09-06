import os
import math
import numpy as np
import streamlit as st
from ultralytics import YOLO
from feedback_store import calculate_boosted_confidence

# ---------------------------------------------------------------------------
# MAPEO COMPLETO DE RESIDUOS (CABA) - INCLUYE RESIDUOS ESPECIALES GCBA
# ---------------------------------------------------------------------------
WASTE_MAP = {
    # Papel y Carton (Contenedor Verde)
    "cardboard":            {"label": "Carton",              "emoji": "📦", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Mantener seco, doblar o desarmar para optimizar espacio.", "es_especial": False},
    "carton":               {"label": "Carton",              "emoji": "📦", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Mantener seco, doblar o desarmar para optimizar espacio.", "es_especial": False},
    "corrugated":           {"label": "Carton",              "emoji": "📦", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Mantener seco, doblar o desarmar para optimizar espacio.", "es_especial": False},
    "cardboard box":        {"label": "Carton",              "emoji": "📦", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Mantener seco, doblar o desarmar para optimizar espacio.", "es_especial": False},
    "corrugated cardboard": {"label": "Carton",              "emoji": "📦", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Mantener seco, doblar o desarmar para optimizar espacio.", "es_especial": False},
    "flat cardboard":       {"label": "Carton",              "emoji": "📦", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Mantener seco, doblar o desarmar para optimizar espacio.", "es_especial": False},
    "paper":                {"label": "Papel",               "emoji": "📄", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Mantener limpio y seco.", "es_especial": False},
    "book":                 {"label": "Papel / Carton",       "emoji": "📦", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Mantener seco y desarmar la caja.", "es_especial": False},
    
    # Plasticos (Contenedor Verde)
    "plastic":              {"label": "Plastico / Envases",   "emoji": "🧴", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Vaciar, enjuagar y compactar.", "es_especial": False},
    "plastic bottle":       {"label": "Plastico / Envases",   "emoji": "🧴", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Vaciar, enjuagar y compactar.", "es_especial": False},
    "pet bottle":           {"label": "Plastico / Envases",   "emoji": "🧴", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Vaciar, enjuagar y compactar.", "es_especial": False},
    "bottle":               {"label": "Botella / Plastico",   "emoji": "🧴", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Vaciar, enjuagar y aplastar.", "es_especial": False},
    "cup":                  {"label": "Vaso Descartable",     "emoji": "🥤", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Asegurar que este limpio y sin restos liquidos.", "es_especial": False},
    "bowl":                 {"label": "Envase Descartable",   "emoji": "🥣", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Asegurar que este limpio y sin restos organicos.", "es_especial": False},
    "capsule":              {"label": "Capsula de Cafe",      "emoji": "☕", "tipo": "Reciclable / Punto Verde",  "color": (46, 204, 113),  "accion": "Vacia y limpia. En Punto Verde se convierte en madera plastica.", "es_especial": False},
    
    # CDs, DVDs y Discos Plasticos (Policarbonato)
    "cd":                   {"label": "CD / DVD / Disco",     "emoji": "💿", "tipo": "Reciclable (Plastico / Punto Verde)", "color": (46, 204, 113), "accion": "Material policarbonato. Apto Contenedor Verde o Punto Verde.", "es_especial": False},
    "dvd":                  {"label": "CD / DVD / Disco",     "emoji": "💿", "tipo": "Reciclable (Plastico / Punto Verde)", "color": (46, 204, 113), "accion": "Material policarbonato. Apto Contenedor Verde o Punto Verde.", "es_especial": False},
    "frisbee":              {"label": "Disco / Plastico",     "emoji": "💿", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Plastico rigido reciclable. Depositar en contenedor verde.", "es_especial": False},

    # Vidrio (Contenedor Verde)
    "glass":                {"label": "Vidrio",               "emoji": "🍾", "tipo": "Reciclable Seco (Verde)",  "color": (52, 152, 219),  "accion": "Enjuagar. Envolver si esta roto para proteger al recolector.", "es_especial": False},
    "glass bottle":         {"label": "Vidrio",               "emoji": "🍾", "tipo": "Reciclable Seco (Verde)",  "color": (52, 152, 219),  "accion": "Enjuagar. Envolver si esta roto para proteger al recolector.", "es_especial": False},
    "wine glass":           {"label": "Vidrio / Botella",     "emoji": "🍾", "tipo": "Reciclable Seco (Verde)",  "color": (52, 152, 219),  "accion": "Enjuagar. Si esta roto, envolver antes de desechar.", "es_especial": False},
    
    # Metales (Contenedor Verde)
    "metal":                {"label": "Metal / Aluminio",     "emoji": "🥫", "tipo": "Reciclable Seco (Verde)",  "color": (149, 165, 166), "accion": "Limpio y seco. Aplastar latas para ahorrar espacio.", "es_especial": False},
    "can":                  {"label": "Lata / Aluminio",      "emoji": "🥫", "tipo": "Reciclable Seco (Verde)",  "color": (149, 165, 166), "accion": "Limpio y seco. Aplastar si es posible.", "es_especial": False},
    "aluminum can":         {"label": "Lata / Aluminio",      "emoji": "🥫", "tipo": "Reciclable Seco (Verde)",  "color": (149, 165, 166), "accion": "Limpio y seco. Aplastar si es posible.", "es_especial": False},
    "tin can":              {"label": "Lata / Aluminio",      "emoji": "🥫", "tipo": "Reciclable Seco (Verde)",  "color": (149, 165, 166), "accion": "Limpio y seco. Aplastar si es posible.", "es_especial": False},
    "steel can":            {"label": "Lata / Aluminio",      "emoji": "🥫", "tipo": "Reciclable Seco (Verde)",  "color": (149, 165, 166), "accion": "Limpio y seco. Aplastar si es posible.", "es_especial": False},
    
    # RESIDUOS ESPECIALES GCBA (OBLIGATORIO PUNTO VERDE)
    "e-waste":              {"label": "RAEE / Electronico",   "emoji": "🔋", "tipo": "Residuo Especial (Punto Verde)", "color": (231, 76, 60), "accion": "Llevar a Punto Verde Movil o Fijo (hasta 10 aparatos por persona).", "es_especial": True},
    "e_waste":              {"label": "RAEE / Electronico",   "emoji": "🔋", "tipo": "Residuo Especial (Punto Verde)", "color": (231, 76, 60), "accion": "Llevar a Punto Verde Movil o Fijo (hasta 10 aparatos por persona).", "es_especial": True},
    "electronic":           {"label": "RAEE / Electronico",   "emoji": "🔋", "tipo": "Residuo Especial (Punto Verde)", "color": (231, 76, 60), "accion": "Llevar a Punto Verde Movil o Fijo (hasta 10 aparatos por persona).", "es_especial": True},
    "electronics":          {"label": "RAEE / Electronico",   "emoji": "🔋", "tipo": "Residuo Especial (Punto Verde)", "color": (231, 76, 60), "accion": "Llevar a Punto Verde Movil o Fijo (hasta 10 aparatos por persona).", "es_especial": True},
    "circuit":              {"label": "RAEE / Electronico",   "emoji": "🔋", "tipo": "Residuo Especial (Punto Verde)", "color": (231, 76, 60), "accion": "Llevar a Punto Verde Movil o Fijo (hasta 10 aparatos por persona).", "es_especial": True},
    "battery":              {"label": "Pila / Bateria",       "emoji": "🔋", "tipo": "Residuo Especial (Punto Verde)", "color": (231, 76, 60), "accion": "Llevar a buzones de pilas en Puntos Verdes o Farmacias habilitadas.", "es_especial": True},
    "cell phone":           {"label": "Celular / RAEE",       "emoji": "🔋", "tipo": "Residuo Especial (Punto Verde)", "color": (231, 76, 60), "accion": "Llevar a Punto Verde. Contiene metales pesados y componentes recuperables.", "es_especial": True},
    "laptop":               {"label": "Laptop / RAEE",        "emoji": "🔋", "tipo": "Residuo Especial (Punto Verde)", "color": (231, 76, 60), "accion": "Llevar a Punto Verde. Tratamiento diferenciado de circuitos y bateria.", "es_especial": True},
    "mouse":                {"label": "Periferico / RAEE",    "emoji": "🔋", "tipo": "Residuo Especial (Punto Verde)", "color": (231, 76, 60), "accion": "Llevar a Punto Verde junto a cables y teclados.", "es_especial": True},
    "keyboard":             {"label": "Teclado / RAEE",       "emoji": "🔋", "tipo": "Residuo Especial (Punto Verde)", "color": (231, 76, 60), "accion": "Llevar a Punto Verde Movil o Fijo.", "es_especial": True},
    "remote":               {"label": "Control / RAEE",       "emoji": "🔋", "tipo": "Residuo Especial (Punto Verde)", "color": (231, 76, 60), "accion": "Retirar pilas y llevar el control al Punto Verde.", "es_especial": True},
    "toner":                {"label": "Toner / Cartucho",     "emoji": "🖨️", "tipo": "Residuo Especial (Punto Verde)", "color": (231, 76, 60), "accion": "Llevar a Punto Verde (limpio, sin derrames, hasta 3 por persona).", "es_especial": True},
    "lamp":                 {"label": "Lampara / Tubo Fluo",  "emoji": "💡", "tipo": "Residuo Peligroso (Punto Verde)", "color": (231, 76, 60), "accion": "Llevar en caja/envoltura (contiene mercurio). Hasta 5 por persona/dia.", "es_especial": True},
    "oil":                  {"label": "Aceite Usado (AVU)",   "emoji": "🍳", "tipo": "Residuo Especial (Punto Verde)", "color": (231, 76, 60), "accion": "Enfriar y envasar en botella plastica cerrada. 1L contamina 1000L de agua.", "es_especial": True},

    # Organicos y Sanitario
    "organic":              {"label": "Residuo Organico",     "emoji": "🍂", "tipo": "Compost / Bolsa Negra",     "color": (139, 69, 19),   "accion": "Ideal para compostera domiciliaria o recepcion organica de Punto Verde.", "es_especial": False},
    "medical":              {"label": "Residuo Sanitario",    "emoji": "⚠️", "tipo": "Residuo Peligroso",        "color": (192, 57, 43),   "accion": "Descartar en bolsa cerrada especial o punto farmaceutico.", "es_especial": True},
}

# Mapa de subcadenas keyword → clave base en WASTE_MAP (fallback por substring)
_KEYWORD_FALLBACK = [
    ("cardboard",  "cardboard"),
    ("carton",     "cardboard"),
    ("corrugated", "cardboard"),
    ("paper",      "paper"),
    ("plastic",    "plastic"),
    ("glass",      "glass"),
    ("aluminum",   "aluminum can"),
    ("aluminium",  "aluminum can"),
    ("tin",        "tin can"),
    ("steel",      "steel can"),
    ("metal",      "metal"),
    ("electronic", "electronic"),
    ("circuit",    "circuit"),
    ("battery",    "battery"),
    ("organic",    "organic"),
    ("oil",        "oil"),
]

# Clases que NUNCA deben marcarse como basura
IGNORED_CLASSES = {
    "person", "hand", "face", "head", "body", "arm", "finger",
    "chair", "couch", "bed", "dining table", "tv", "laptop-screen",
    "refrigerator", "microwave", "oven", "sink", "door", "window", "cabinet",
    "clothes", "toilet", "clock", "vase", "potted plant",
    "wall", "floor", "ceiling", "shelf"
}

DEFAULT_WASTE = {
    "label": "Residuo General", "emoji": "🗑️",
    "tipo": "Basura Comun / Organicos",
    "color": (127, 140, 141),
    "accion": "Depositar en bolsa negra de residuos generales.",
    "es_especial": False
}

MODEL_PATHS = {
    "waste_specialized": "models/waste_yolov8.pt",
    "yolov8n": "yolov8n.pt",
    "yolov8s_world": "yolov8s-worldv2.pt",
}

@st.cache_resource(show_spinner="Cargando modelo de IA...")
def load_model(model_key: str = "waste_specialized") -> YOLO:
    path = MODEL_PATHS.get(model_key, "models/waste_yolov8.pt")
    if not os.path.exists(path) and model_key == "waste_specialized":
        path = "yolov8n.pt"
    model = YOLO(path)
    if "world" in str(path):
        try:
            model.set_classes([
                "plastic bottle", "glass bottle", "bottle",
                "cup", "plastic cup", "disposable cup",
                "aluminum can", "tin can", "can",
                "jar", "plastic jar", "pot", "plastic container",
                "vape", "vape pen", "electronic cigarette",
                "cell phone", "battery",
                "cardboard box", "cardboard", "paper",
                "compact disc", "cd", "tetra pak"
            ])
        except Exception as e:
            print(f"Aviso: no se pudo cargar CLIP para set_classes ({e}), usando clases por defecto.")
    return model

def get_waste_info(class_name: str) -> dict | None:
    norm = class_name.lower().strip()
    norm_spaced = norm.replace("-", " ").replace("_", " ").strip()

    # Verificación inmediata de clases ignoradas (fondo, muebles, anatomía)
    if norm in IGNORED_CLASSES or norm_spaced in IGNORED_CLASSES:
        return None

    for ignored in IGNORED_CLASSES:
        if ignored == norm or ignored.replace("-", " ").replace("_", " ") == norm_spaced:
            return None

    # 1. Busqueda exacta
    if norm in WASTE_MAP:
        return WASTE_MAP[norm]

    # 2. Normalizar guiones y underscores → espacios y buscar exacto
    if norm_spaced in WASTE_MAP:
        return WASTE_MAP[norm_spaced]

    # 3. Busqueda de palabras completas (token a token)
    tokens = norm_spaced.split()
    for token in tokens:
        if token in IGNORED_CLASSES:
            return None
        if token in WASTE_MAP:
            return WASTE_MAP[token]

    # 4. Fallback por subcadena keyword: recorre _KEYWORD_FALLBACK en orden de prioridad
    for keyword, map_key in _KEYWORD_FALLBACK:
        if keyword in norm_spaced:
            return WASTE_MAP[map_key]

    return DEFAULT_WASTE


def analyze_spatial_and_hand_context(frame_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[bool, float, bool]:
    """
    Analiza si el objeto esta en el centro visual y si existe presencia de mano/piel sosteniendolo.
    Retorna: (is_held_in_center, center_score, hand_detected)
    """
    h_img, w_img = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox

    # 1. Proximidad al centro del encuadre
    obj_cx = (x1 + x2) / 2.0
    obj_cy = (y1 + y2) / 2.0
    frame_cx = w_img / 2.0
    frame_cy = h_img / 2.0

    max_dist = math.sqrt(frame_cx**2 + frame_cy**2)
    dist = math.sqrt((obj_cx - frame_cx)**2 + (obj_cy - frame_cy)**2)
    center_score = max(0.0, 1.0 - (dist / max_dist))

    is_in_center = (0.12 <= obj_cx / w_img <= 0.88) and (0.12 <= obj_cy / h_img <= 0.88)

    # 2. Deteccion de mano/piel en la periferia del objeto
    pad = int(min(w_img, h_img) * 0.08)
    roi_x1 = max(0, x1 - pad)
    roi_y1 = max(0, y1 - pad)
    roi_x2 = min(w_img, x2 + pad)
    roi_y2 = min(h_img, y2 + pad)

    roi = frame_bgr[roi_y1:roi_y2, roi_x1:roi_x2]
    hand_detected = False

    if roi.size > 0:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 35, 45], dtype=np.uint8)
        upper_skin = np.array([28, 210, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_ratio = np.count_nonzero(skin_mask) / float(roi.shape[0] * roi.shape[1])
        if skin_ratio > 0.05:
            hand_detected = True

    is_held_in_center = is_in_center and (hand_detected or center_score > 0.55)
    return is_held_in_center, center_score, hand_detected


def run_inference(
    model: YOLO,
    frame_bgr: np.ndarray,
    conf_threshold: float = 0.20,
    filter_people: bool = True,
    feedback_store: dict | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """
    Ejecuta deteccion YOLO con analisis espacial de objeto sostenido y auto-aprendizaje.
    """
    results = model(frame_bgr, conf=conf_threshold, verbose=False)[0]
    detections = []

    for box in results.boxes:
        cls_id   = int(box.cls[0])
        cls_name = model.names[cls_id].lower()
        conf     = float(box.conf[0])

        if filter_people and cls_name in IGNORED_CLASSES:
            continue
        info = get_waste_info(cls_name)
        if info is None:
            continue

        bbox = tuple(map(int, box.xyxy[0].tolist()))
        is_held_in_center, center_score, hand_detected = analyze_spatial_and_hand_context(frame_bgr, bbox)

        effective_conf, boost_applied = calculate_boosted_confidence(
            cls_name=cls_name,
            raw_conf=conf,
            predicted_cls=cls_name,
            is_held_in_center=is_held_in_center,
            store=feedback_store
        )

        detections.append({
            "clase":             cls_name,
            "label":             info["label"],
            "emoji":             info.get("emoji", "📦"),
            "tipo":              info["tipo"],
            "accion":            info["accion"],
            "es_especial":       info.get("es_especial", False),
            "confianza":         effective_conf,
            "confianza_raw":     conf,
            "boost_applied":     boost_applied,
            "is_held_in_center": is_held_in_center,
            "center_score":      center_score,
            "hand_detected":     hand_detected,
            "bbox":              bbox,
        })

    # Priorizar por confianza efectiva + score central
    detections.sort(key=lambda d: d["confianza"] + (0.15 if d["is_held_in_center"] else 0.0), reverse=True)

    # Anotar frame (usando etiquetas ASCII limpias para evitar los '?' de OpenCV Hershey)
    annotated = frame_bgr.copy()

    for det in detections:
        info  = get_waste_info(det["clase"])
        if info is None:
            continue
        color = info["color"]
        x1, y1, x2, y2 = det["bbox"]

        thick = 3 if det["is_held_in_center"] else 2
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thick)

        # Etiqueta limpia para OpenCV (sin emojis que rompen Hershey font)
        clean_label = det["label"].encode("ascii", "ignore").decode("ascii") or det["clase"]
        tag = " [En Mano]" if det["is_held_in_center"] else ""
        star = " *" if det["boost_applied"] else ""
        label_text = f"{clean_label} {det['confianza']:.0%}{tag}{star}"

        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        top_y = max(y1, th + 10)
        cv2.rectangle(annotated, (x1, top_y - th - 8), (x1 + tw + 8, top_y), color, -1)
        cv2.putText(annotated, label_text, (x1 + 4, top_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    return annotated, detections
