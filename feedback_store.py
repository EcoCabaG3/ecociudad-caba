"""
feedback_store.py - Auto-aprendizaje adaptativo y persistencia de pesos para EcoCiudad CABA.
"""
import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
FEEDBACK_PATH = BASE_DIR / "feedback_log.json"

# Hiperparametros de auto-aprendizaje
BASE_VOTE_BOOST = 0.08      # Boost de confianza por cada confirmacion directa
HELD_CENTER_BOOST = 0.12    # Boost adicional cuando el objeto esta sostenido en el centro
MAX_TOTAL_BOOST = 0.50      # Limite maximo de boost para evitar saturacion (cap a 0.99)
DAMPING_FACTOR = 0.95       # Amortiguacion para regularizacion


def load_feedback() -> dict:
    """Carga la memoria de aprendizaje desde disco."""
    default_schema = {
        "corrections": {},          # { "clase_predicha": { "clase_real": count } }
        "held_center_weights": {},  # { "clase_real": count_held }
        "total_feedbacks": 0
    }
    if FEEDBACK_PATH.exists():
        try:
            with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "corrections" not in data:
                    data = {"corrections": data, "held_center_weights": {}, "total_feedbacks": 0}
                return data
        except Exception:
            return default_schema
    return default_schema


def save_feedback(store: dict) -> None:
    """Persiste la memoria de auto-entrenamiento a disco."""
    try:
        with open(FEEDBACK_PATH, "w", encoding="utf-8") as f:
            json.dump(store, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error al guardar feedback: {e}")


def add_correction(predicted_cls: str, correct_cls: str, is_held_in_center: bool = True, store: dict = None) -> dict:
    """
    Auto-entrena los pesos del modelo:
    Registra que cuando el detector vio `predicted_cls`, el usuario confirmo `correct_cls`.
    Si el objeto estaba sostenido en el centro, refuerza el prior espacial.
    """
    if store is None:
        store = load_feedback()

    predicted_cls = predicted_cls.lower().strip()
    correct_cls   = correct_cls.lower().strip()

    if "corrections" not in store:
        store["corrections"] = {}
    if "held_center_weights" not in store:
        store["held_center_weights"] = {}

    # Matriz de transicion/correccion
    if predicted_cls not in store["corrections"]:
        store["corrections"][predicted_cls] = {}
    store["corrections"][predicted_cls][correct_cls] = store["corrections"][predicted_cls].get(correct_cls, 0) + 1

    # Prior cuando esta sostenido en el centro
    if is_held_in_center:
        store["held_center_weights"][correct_cls] = store["held_center_weights"].get(correct_cls, 0) + 1

    store["total_feedbacks"] = store.get("total_feedbacks", 0) + 1
    save_feedback(store)
    return store


def calculate_boosted_confidence(
    cls_name: str,
    raw_conf: float,
    predicted_cls: str,
    is_held_in_center: bool,
    store: dict
) -> tuple[float, bool]:
    """
    Calcula la nueva confianza efectiva incorporando el auto-entrenamiento.
    Retorna: (confianza_efectiva, fue_mejorado_por_aprendizaje)
    """
    if not store:
        return raw_conf, False

    cls_name = cls_name.lower().strip()
    predicted_cls = predicted_cls.lower().strip()
    corrections = store.get("corrections", {})
    held_weights = store.get("held_center_weights", {})

    boost = 0.0
    boost_applied = False

    # 1. Boost por votos historicos de confirmacion
    if predicted_cls in corrections:
        votes_for_this = corrections[predicted_cls].get(cls_name, 0)
        if votes_for_this > 0:
            boost += min(votes_for_this * BASE_VOTE_BOOST, MAX_TOTAL_BOOST)
            boost_applied = True

    # 2. Boost por patron de objeto sostenido en el centro visual
    if is_held_in_center and cls_name in held_weights:
        held_votes = held_weights.get(cls_name, 0)
        if held_votes > 0:
            boost += min(held_votes * HELD_CENTER_BOOST, 0.25)
            boost_applied = True

    effective_conf = min(max(raw_conf + boost, raw_conf), 0.99)
    return effective_conf, boost_applied


def get_total_corrections(store: dict) -> int:
    """Devuelve la cantidad total de feedbacks aplicados."""
    if not store:
        return 0
    return store.get("total_feedbacks", 0)
