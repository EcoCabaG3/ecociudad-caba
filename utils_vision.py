"""
utils_vision.py - Deteccion, analisis espacial y clasificacion de residuos EcoCiudad CABA
Optimizacion estricta de recursos: Max 1 CPU Thread por operador y tope seguro < 800 MB RAM para Streamlit Cloud.
"""
import os
import sys
import math
import gc
import ctypes
import tempfile
import threading

# ─── Limites de Recursos Hardware (Tope Seguro < 800 MB RAM para Streamlit Cloud) ─
MAX_CPU_CORES = 1
MAX_MEMORY_BYTES = int(0.80 * 1024 * 1024 * 1024)  # 800 MB tope preventivo antes del limite de 1 GB de Streamlit Cloud

os.environ["OMP_NUM_THREADS"] = str(MAX_CPU_CORES)
os.environ["MKL_NUM_THREADS"] = str(MAX_CPU_CORES)
os.environ["OPENBLAS_NUM_THREADS"] = str(MAX_CPU_CORES)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(MAX_CPU_CORES)
os.environ["NUMEXPR_NUM_THREADS"] = str(MAX_CPU_CORES)
os.environ["TORCH_NUM_THREADS"] = str(MAX_CPU_CORES)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["MALLOC_ARENA_MAX"] = "2"
os.environ["MALLOC_TRIM_THRESHOLD_"] = "32768"

try:
    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        _avail_cores = sorted(os.sched_getaffinity(0))
        if len(_avail_cores) > 2:
            os.sched_setaffinity(0, set(_avail_cores[:2]))
except Exception:
    pass

import numpy as np
import torch

torch.set_num_threads(MAX_CPU_CORES)
torch.set_grad_enabled(False)
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass

import streamlit as st
from ultralytics import YOLO
from feedback_store import calculate_boosted_confidence

try:
    import cv2
    HAS_CV2 = True
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
except Exception:
    cv2 = None
    HAS_CV2 = False

try:
    _libc = ctypes.CDLL("libc.so.6")
except Exception:
    _libc = None


def trim_memory() -> None:
    """Libera referencias de Python y devuelve bloques libres de C/C++ al kernel (Linux y Windows)."""
    gc.collect()
    if _libc is not None:
        try:
            _libc.malloc_trim(0)
        except Exception:
            pass
    elif os.name == "nt":
        try:
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.psapi.EmptyWorkingSet(handle)
        except Exception:
            pass


def get_process_rss_bytes() -> int:
    """Obtiene el uso actual de memoria fisica RSS en bytes (Linux /proc/self/statm y Windows Psapi)."""
    try:
        if os.path.exists("/proc/self/statm"):
            with open("/proc/self/statm", "r", encoding="utf-8") as f:
                parts = f.read().split()
                if len(parts) >= 2:
                    return int(parts[1]) * os.sysconf("SC_PAGE_SIZE")
        elif os.name == "nt":
            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]
            pmc = PROCESS_MEMORY_COUNTERS()
            pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
                return int(pmc.WorkingSetSize)
    except Exception:
        pass
    return 0


_INFERENCE_LOCK = threading.Lock()


def _clear_predictor_memory(yolo_model: YOLO | None) -> None:
    """Elimina referencias retenidas a tensores e imagenes dentro de Ultralytics Predictor."""
    if yolo_model is None:
        return
    pred = getattr(yolo_model, "predictor", None)
    if pred is not None:
        pred.results = None
        pred.dataset = None

# ---------------------------------------------------------------------------
# MAPEO COMPLETO DE RESIDUOS (CABA) - INCLUYE RESIDUOS ESPECIALES GCBA
# ---------------------------------------------------------------------------
WASTE_MAP = {
    # Papel y Carton (Contenedor Verde)
    "cardboard":            {"label": "Carton",              "emoji": "📦", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Mantener seco, doblar o desarmar para optimizar espacio.", "es_especial": False},
    "carton":               {"label": "Carton",              "emoji": "📦", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Mantener seco, doblar o desarmar para optimizar espacio.", "es_especial": False},
    "corrugated":           {"label": "Carton",              "emoji": "📦", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Mantener seco, doblar o desarmar para optimizar espacio.", "es_especial": False},
    "cardboard box":        {"label": "Carton",              "emoji": "📦", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Mantener seco, doblar o desarmar para optimizar espacio.", "es_especial": False},
    "shoebox":              {"label": "Carton",              "emoji": "📦", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Mantener seco, doblar o desarmar para optimizar espacio.", "es_especial": False},
    "black cardboard box":  {"label": "Carton",              "emoji": "📦", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Mantener seco, doblar o desarmar para optimizar espacio.", "es_especial": False},
    "small cardboard box":  {"label": "Carton",              "emoji": "📦", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Mantener seco, doblar o desarmar para optimizar espacio.", "es_especial": False},
    "packaging box":        {"label": "Carton",              "emoji": "📦", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Mantener seco, doblar o desarmar para optimizar espacio.", "es_especial": False},
    "medicine box":         {"label": "Carton",              "emoji": "📦", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Mantener seco, doblar o desarmar para optimizar espacio.", "es_especial": False},
    "cosmetic box":         {"label": "Carton",              "emoji": "📦", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Mantener seco, doblar o desarmar para optimizar espacio.", "es_especial": False},
    "corrugated cardboard": {"label": "Carton",              "emoji": "📦", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Mantener seco, doblar o desarmar para optimizar espacio.", "es_especial": False},
    "flat cardboard":       {"label": "Carton",              "emoji": "📦", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Mantener seco, doblar o desarmar para optimizar espacio.", "es_especial": False},
    "tetra pak":            {"label": "Tetra Pak / Multicapa","emoji": "🧃", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Enjuagar, escurrir y aplastar. Apto Contenedor Verde.", "es_especial": False},
    "paper":                {"label": "Papel",               "emoji": "📄", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Mantener limpio y seco.", "es_especial": False},
    "sheet of paper":       {"label": "Papel",               "emoji": "📄", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Mantener limpio y seco.", "es_especial": False},
    "notebook paper":       {"label": "Papel",               "emoji": "📄", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Mantener limpio y seco.", "es_especial": False},
    "printed document":     {"label": "Papel",               "emoji": "📄", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Mantener limpio y seco.", "es_especial": False},
    "paper bill":           {"label": "Papel",               "emoji": "📄", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Mantener limpio y seco.", "es_especial": False},
    "invoice":              {"label": "Papel",               "emoji": "📄", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Mantener limpio y seco.", "es_especial": False},
    "brochure":             {"label": "Papel",               "emoji": "📄", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Mantener limpio y seco.", "es_especial": False},
    "flyer":                {"label": "Papel",               "emoji": "📄", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Mantener limpio y seco.", "es_especial": False},
    "pamphlet":             {"label": "Papel",               "emoji": "📄", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Mantener limpio y seco.", "es_especial": False},
    "envelope":             {"label": "Papel",               "emoji": "📄", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Mantener limpio y seco.", "es_especial": False},
    "book":                 {"label": "Papel / Carton",       "emoji": "📦", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Mantener seco y desarmar la caja.", "es_especial": False},
    
    # Plasticos, Envases, Bolsas y Potes (Contenedor Verde)
    "plastic":              {"label": "Plastico / Envases",   "emoji": "🧴", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Vaciar, enjuagar y compactar.", "es_especial": False},
    "plastic bag":          {"label": "Bolsa Plastica",       "emoji": "🛍️", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Si esta limpia y seca va al Contenedor Verde o Botella de Amor.", "es_especial": False},
    "white plastic bag":    {"label": "Bolsa Plastica",       "emoji": "🛍️", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Si esta limpia y seca va al Contenedor Verde o Botella de Amor.", "es_especial": False},
    "supermarket plastic bag": {"label": "Bolsa Plastica",    "emoji": "🛍️", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Si esta limpia y seca va al Contenedor Verde o Botella de Amor.", "es_especial": False},
    "grocery bag":          {"label": "Bolsa Plastica",       "emoji": "🛍️", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Si esta limpia y seca va al Contenedor Verde o Botella de Amor.", "es_especial": False},
    "shopping bag":         {"label": "Bolsa Plastica",       "emoji": "🛍️", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Si esta limpia y seca va al Contenedor Verde o Botella de Amor.", "es_especial": False},
    "polyethylene bag":     {"label": "Bolsa Plastica",       "emoji": "🛍️", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Si esta limpia y seca va al Contenedor Verde o Botella de Amor.", "es_especial": False},
    "plastic wrapper":      {"label": "Envoltorio Plastico",  "emoji": "🛍️", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Limpio y seco al Contenedor Verde o Botella de Amor.", "es_especial": False},
    "plastic bottle":       {"label": "Botella Plastica",     "emoji": "🧴", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Vaciar, enjuagar, aplastar y dejar con tapa en Contenedor Verde.", "es_especial": False},
    "clear plastic bottle": {"label": "Botella Plastica",     "emoji": "🧴", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Vaciar, enjuagar, aplastar y dejar con tapa en Contenedor Verde.", "es_especial": False},
    "water bottle":         {"label": "Botella Plastica",     "emoji": "🧴", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Vaciar, enjuagar, aplastar y dejar con tapa en Contenedor Verde.", "es_especial": False},
    "soda bottle":          {"label": "Botella Plastica",     "emoji": "🧴", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Vaciar, enjuagar, aplastar y dejar con tapa en Contenedor Verde.", "es_especial": False},
    "juice bottle":         {"label": "Botella Plastica",     "emoji": "🧴", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Vaciar, enjuagar, aplastar y dejar con tapa en Contenedor Verde.", "es_especial": False},
    "pet bottle":           {"label": "Botella Plastica",     "emoji": "🧴", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Vaciar, enjuagar, aplastar y dejar con tapa en Contenedor Verde.", "es_especial": False},
    "bottle":               {"label": "Botella / Plastico",   "emoji": "🧴", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Vaciar, enjuagar y aplastar.", "es_especial": False},
    "thermos":              {"label": "Termo / Botella",      "emoji": "🧴", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Limpio y seco al Contenedor Verde.", "es_especial": False},
    "cup":                  {"label": "Vaso (Vidrio / Plastico)", "emoji": "🥤", "tipo": "Reciclable Seco (Verde)", "color": (52, 152, 219),  "accion": "Asegurar que este limpio y seco. Apto Contenedor Verde.", "es_especial": False},
    "disposable cup":       {"label": "Vaso Descartable",     "emoji": "🥤", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Asegurar que este limpio y sin restos liquidos.", "es_especial": False},
    "plastic cup":          {"label": "Vaso Descartable",     "emoji": "🥤", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Asegurar que este limpio y sin restos liquidos.", "es_especial": False},
    "bowl":                 {"label": "Envase Descartable",   "emoji": "🥣", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Asegurar que este limpio y sin restos organicos.", "es_especial": False},
    "jar":                  {"label": "Pote / Frasco Plastico","emoji": "🧴", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Limpio y seco. Apto Contenedor Verde.", "es_especial": False},
    "plastic jar":          {"label": "Pote / Envase Plastico","emoji": "🧴", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Limpio y seco. Apto Contenedor Verde.", "es_especial": False},
    "cosmetic jar":         {"label": "Pote / Envase Plastico","emoji": "🧴", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Limpio y seco. Apto Contenedor Verde.", "es_especial": False},
    "cream jar":            {"label": "Pote / Envase Plastico","emoji": "🧴", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Limpio y seco. Apto Contenedor Verde.", "es_especial": False},
    "plastic tub":          {"label": "Pote / Envase Plastico","emoji": "🧴", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Limpio y seco. Apto Contenedor Verde.", "es_especial": False},
    "plastic pot":          {"label": "Pote / Envase Plastico","emoji": "🧴", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Limpio y seco. Apto Contenedor Verde.", "es_especial": False},
    "round plastic container": {"label": "Pote / Envase Plastico", "emoji": "🧴", "tipo": "Reciclable Seco (Verde)", "color": (46, 204, 113), "accion": "Limpio y seco. Apto Contenedor Verde.", "es_especial": False},
    "plastic lid":          {"label": "Tapa / Envase Plastico","emoji": "🧴", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Plastico reciclable limpio y seco. Apto Contenedor Verde.", "es_especial": False},
    "plastic cap":          {"label": "Tapa / Envase Plastico","emoji": "🧴", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Plastico reciclable limpio y seco. Apto Contenedor Verde.", "es_especial": False},
    "pot":                  {"label": "Pote / Envase Plastico","emoji": "🧴", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Limpio y seco. Apto Contenedor Verde.", "es_especial": False},
    "plastic container":    {"label": "Pote / Envase Plastico","emoji": "🧴", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Vaciar, enjuagar y compactar.", "es_especial": False},
    "plastic box":          {"label": "Caja / Envase Plastico","emoji": "🧴", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Plastico reciclable limpio y seco. Apto Contenedor Verde.", "es_especial": False},
    "container":            {"label": "Pote / Envase Plastico","emoji": "🧴", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Limpio y seco. Al Contenedor Verde.", "es_especial": False},
    "lighter":              {"label": "Encendedor (Plastico / Metal)", "emoji": "🔥", "tipo": "Basura Comun / Descarga Segura", "color": (60, 76, 231), "accion": "Verificar que este 100% sin gas antes de descartar. Por mezcla de gas, metal y plastico no va a campana verde.", "es_especial": False},
    "cigarette lighter":    {"label": "Encendedor (Plastico / Metal)", "emoji": "🔥", "tipo": "Basura Comun / Descarga Segura", "color": (60, 76, 231), "accion": "Verificar que este 100% sin gas antes de descartar. Por mezcla de gas, metal y plastico no va a campana verde.", "es_especial": False},
    "plastic lighter":      {"label": "Encendedor (Plastico / Metal)", "emoji": "🔥", "tipo": "Basura Comun / Descarga Segura", "color": (60, 76, 231), "accion": "Verificar que este 100% sin gas antes de descartar. Por mezcla de gas, metal y plastico no va a campana verde.", "es_especial": False},
    "capsule":              {"label": "Capsula de Cafe",      "emoji": "☕", "tipo": "Reciclable / Punto Verde",  "color": (46, 204, 113),  "accion": "Vacia y limpia. En Punto Verde se convierte en madera plastica.", "es_especial": False},

    # Utiles de Escritura, Medicion y Plasticos Rigidos (Mapeados a Plastico / Contenedor Verde)
    "ruler":                {"label": "Plastico / Envases",   "emoji": "🧴", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Plastico reciclable. Apto Contenedor Verde.", "es_especial": False},
    "plastic ruler":        {"label": "Plastico / Envases",   "emoji": "🧴", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Plastico reciclable. Apto Contenedor Verde.", "es_especial": False},
    "pen":                  {"label": "Lapicera / Plastico",   "emoji": "🖊️", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Plastico rigido. Retirar mina de tinta si es posible.", "es_especial": False},
    "plastic pen":          {"label": "Lapicera / Plastico",   "emoji": "🖊️", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Plastico rigido. Retirar mina de tinta si es posible.", "es_especial": False},
    "ballpoint pen":        {"label": "Lapicera / Plastico",   "emoji": "🖊️", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Plastico rigido. Retirar mina de tinta si es posible.", "es_especial": False},
    "marker":               {"label": "Marcador / Plastico",   "emoji": "🖊️", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Plastico rigido reciclable.", "es_especial": False},
    "marker pen":           {"label": "Marcador / Plastico",   "emoji": "🖊️", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Plastico rigido reciclable.", "es_especial": False},
    "permanent marker":     {"label": "Marcador / Plastico",   "emoji": "🖊️", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Plastico rigido reciclable.", "es_especial": False},
    "sharpie marker":       {"label": "Marcador / Plastico",   "emoji": "🖊️", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Plastico rigido reciclable.", "es_especial": False},
    "felt tip pen":         {"label": "Marcador / Plastico",   "emoji": "🖊️", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Plastico rigido reciclable.", "es_especial": False},
    "highlighter pen":      {"label": "Marcador / Plastico",   "emoji": "🖊️", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Plastico rigido reciclable.", "es_especial": False},
    "pencil":               {"label": "Lapiz / Madera",        "emoji": "✏️", "tipo": "Basura Comun / Seco",     "color": (127, 140, 141), "accion": "Restos de madera y grafito.", "es_especial": False},
    
    # CDs, DVDs y Discos Plasticos (Policarbonato)
    "cd":                   {"label": "CD / DVD / Disco",     "emoji": "💿", "tipo": "Reciclable (Plastico / Punto Verde)", "color": (46, 204, 113), "accion": "Material policarbonato. Apto Contenedor Verde o Punto Verde.", "es_especial": False},
    "dvd":                  {"label": "CD / DVD / Disco",     "emoji": "💿", "tipo": "Reciclable (Plastico / Punto Verde)", "color": (46, 204, 113), "accion": "Material policarbonato. Apto Contenedor Verde o Punto Verde.", "es_especial": False},
    "frisbee":              {"label": "Disco / Plastico",     "emoji": "💿", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Plastico rigido reciclable. Depositar en contenedor verde.", "es_especial": False},

    # Vidrio (Contenedor Verde)
    "glass":                {"label": "Vidrio",               "emoji": "🍾", "tipo": "Reciclable Seco (Verde)",  "color": (52, 152, 219),  "accion": "Enjuagar. Envolver si esta roto para proteger al recolector.", "es_especial": False},
    "glass bottle":         {"label": "Vidrio",               "emoji": "🍾", "tipo": "Reciclable Seco (Verde)",  "color": (52, 152, 219),  "accion": "Enjuagar. Envolver si esta roto para proteger al recolector.", "es_especial": False},
    "wine glass":           {"label": "Vidrio / Botella",     "emoji": "🍾", "tipo": "Reciclable Seco (Verde)",  "color": (52, 152, 219),  "accion": "Enjuagar. Si esta roto, envolver antes de desechar.", "es_especial": False},
    "drinking glass":       {"label": "Vaso de Vidrio",       "emoji": "🥃", "tipo": "Reciclable Seco (Verde)",  "color": (52, 152, 219),  "accion": "Vidrio reciclable. Enjuagar y depositar en Contenedor Verde.", "es_especial": False},
    "glass cup":            {"label": "Vaso de Vidrio",       "emoji": "🥃", "tipo": "Reciclable Seco (Verde)",  "color": (52, 152, 219),  "accion": "Vidrio reciclable. Enjuagar y depositar en Contenedor Verde.", "es_especial": False},
    
    # Metales (Contenedor Verde)
    "metal":                {"label": "Metal / Aluminio",     "emoji": "🥫", "tipo": "Reciclable Seco (Verde)",  "color": (149, 165, 166), "accion": "Limpio y seco. Aplastar latas para ahorrar espacio.", "es_especial": False},
    "scissors":             {"label": "Tijera / Metal",       "emoji": "✂️", "tipo": "Reciclable Seco (Verde)",  "color": (149, 165, 166), "accion": "Objeto metalico reciclable. Envolver las puntas por seguridad antes de llevar al Contenedor Verde o Punto Verde.", "es_especial": False},
    "metal scissors":       {"label": "Tijera / Metal",       "emoji": "✂️", "tipo": "Reciclable Seco (Verde)",  "color": (149, 165, 166), "accion": "Objeto metalico reciclable. Envolver las puntas por seguridad antes de llevar al Contenedor Verde o Punto Verde.", "es_especial": False},
    "pair of scissors":     {"label": "Tijera / Metal",       "emoji": "✂️", "tipo": "Reciclable Seco (Verde)",  "color": (149, 165, 166), "accion": "Objeto metalico reciclable. Envolver las puntas por seguridad antes de llevar al Contenedor Verde o Punto Verde.", "es_especial": False},
    "eyeglasses case":      {"label": "Estuche Rigido / Plastico", "emoji": "🕶️", "tipo": "Donación / Reciclable Seco", "color": (46, 204, 113), "accion": "Estuche rigido de plastico/metal. Si esta sano se puede reutilizar; si esta roto va a Contenedor Verde (sin forro textil) o Bolsa Negra.", "es_especial": False},
    "glasses case":         {"label": "Estuche Rigido / Plastico", "emoji": "🕶️", "tipo": "Donación / Reciclable Seco", "color": (46, 204, 113), "accion": "Estuche rigido de plastico/metal. Si esta sano se puede reutilizar; si esta roto va a Contenedor Verde (sin forro textil) o Bolsa Negra.", "es_especial": False},
    "hard case":            {"label": "Estuche Rigido / Plastico", "emoji": "🕶️", "tipo": "Donación / Reciclable Seco", "color": (46, 204, 113), "accion": "Estuche rigido de plastico/metal. Si esta sano se puede reutilizar; si esta roto va a Contenedor Verde (sin forro textil) o Bolsa Negra.", "es_especial": False},
    "can":                  {"label": "Lata / Aluminio",      "emoji": "🥫", "tipo": "Reciclable Seco (Verde)",  "color": (149, 165, 166), "accion": "Limpio y seco. Aplastar si es posible.", "es_especial": False},
    "aluminum can":         {"label": "Lata / Aluminio",      "emoji": "🥫", "tipo": "Reciclable Seco (Verde)",  "color": (149, 165, 166), "accion": "Limpio y seco. Aplastar si es posible.", "es_especial": False},
    "soda can":             {"label": "Lata / Aluminio",      "emoji": "🥫", "tipo": "Reciclable Seco (Verde)",  "color": (149, 165, 166), "accion": "Limpio y seco. Aplastar si es posible.", "es_especial": False},
    "beer can":             {"label": "Lata / Aluminio",      "emoji": "🥫", "tipo": "Reciclable Seco (Verde)",  "color": (149, 165, 166), "accion": "Limpio y seco. Aplastar si es posible.", "es_especial": False},
    "beverage can":         {"label": "Lata / Aluminio",      "emoji": "🥫", "tipo": "Reciclable Seco (Verde)",  "color": (149, 165, 166), "accion": "Limpio y seco. Aplastar si es posible.", "es_especial": False},
    "drink can":            {"label": "Lata / Aluminio",      "emoji": "🥫", "tipo": "Reciclable Seco (Verde)",  "color": (149, 165, 166), "accion": "Limpio y seco. Aplastar si es posible.", "es_especial": False},
    "tin can":              {"label": "Lata / Aluminio",      "emoji": "🥫", "tipo": "Reciclable Seco (Verde)",  "color": (149, 165, 166), "accion": "Limpio y seco. Aplastar si es posible.", "es_especial": False},
    "steel can":            {"label": "Lata / Aluminio",      "emoji": "🥫", "tipo": "Reciclable Seco (Verde)",  "color": (149, 165, 166), "accion": "Limpio y seco. Aplastar si es posible.", "es_especial": False},
    "spray can":            {"label": "Lata / Aerosol",       "emoji": "🥫", "tipo": "Reciclable Seco (Verde)",  "color": (149, 165, 166), "accion": "Lata de metal o aluminio vacia y despresurizada. Apto Contenedor Verde.", "es_especial": False},
    "aerosol can":          {"label": "Lata / Aerosol",       "emoji": "🥫", "tipo": "Reciclable Seco (Verde)",  "color": (149, 165, 166), "accion": "Lata de metal o aluminio vacia y despresurizada. Apto Contenedor Verde.", "es_especial": False},
    "aerosol":              {"label": "Lata / Aerosol",       "emoji": "🥫", "tipo": "Reciclable Seco (Verde)",  "color": (149, 165, 166), "accion": "Lata de metal o aluminio vacia y despresurizada. Apto Contenedor Verde.", "es_especial": False},
    "air freshener spray":  {"label": "Lata / Aerosol",       "emoji": "🥫", "tipo": "Reciclable Seco (Verde)",  "color": (149, 165, 166), "accion": "Lata de metal o aluminio vacia y despresurizada. Apto Contenedor Verde.", "es_especial": False},
    "deodorant spray":      {"label": "Lata / Aerosol",       "emoji": "🥫", "tipo": "Reciclable Seco (Verde)",  "color": (149, 165, 166), "accion": "Lata de metal o aluminio vacia y despresurizada. Apto Contenedor Verde.", "es_especial": False},
    
    # RESIDUOS ESPECIALES GCBA (OBLIGATORIO PUNTO VERDE) — Color BGR Rojo (60, 76, 231)
    "vape":                 {"label": "Vaper / Cigarrillo Electronico", "emoji": "🔋", "tipo": "Residuo Especial (Punto Verde)", "color": (60, 76, 231), "accion": "Contiene bateria de litio y circuitos. Llevar a Punto Verde.", "es_especial": True},
    "vape pen":             {"label": "Vaper / Cigarrillo Electronico", "emoji": "🔋", "tipo": "Residuo Especial (Punto Verde)", "color": (60, 76, 231), "accion": "Contiene bateria de litio y circuitos. Llevar a Punto Verde.", "es_especial": True},
    "disposable vape":      {"label": "Vaper / Cigarrillo Electronico", "emoji": "🔋", "tipo": "Residuo Especial (Punto Verde)", "color": (60, 76, 231), "accion": "Contiene bateria de litio y circuitos. Llevar a Punto Verde.", "es_especial": True},
    "vape pod":             {"label": "Vaper / Cigarrillo Electronico", "emoji": "🔋", "tipo": "Residuo Especial (Punto Verde)", "color": (60, 76, 231), "accion": "Contiene bateria de litio y circuitos. Llevar a Punto Verde.", "es_especial": True},
    "vape mod":             {"label": "Vaper / Cigarrillo Electronico", "emoji": "🔋", "tipo": "Residuo Especial (Punto Verde)", "color": (60, 76, 231), "accion": "Contiene bateria de litio y circuitos. Llevar a Punto Verde.", "es_especial": True},
    "vape bar":             {"label": "Vaper / Cigarrillo Electronico", "emoji": "🔋", "tipo": "Residuo Especial (Punto Verde)", "color": (60, 76, 231), "accion": "Contiene bateria de litio y circuitos. Llevar a Punto Verde.", "es_especial": True},
    "vape device":          {"label": "Vaper / Cigarrillo Electronico", "emoji": "🔋", "tipo": "Residuo Especial (Punto Verde)", "color": (60, 76, 231), "accion": "Contiene bateria de litio y circuitos. Llevar a Punto Verde.", "es_especial": True},
    "box mod":              {"label": "Vaper / Cigarrillo Electronico", "emoji": "🔋", "tipo": "Residuo Especial (Punto Verde)", "color": (60, 76, 231), "accion": "Contiene bateria de litio y circuitos. Llevar a Punto Verde.", "es_especial": True},
    "electronic cigarette": {"label": "Cigarrillo Electronico / Vaper", "emoji": "🔋", "tipo": "Residuo Especial (Punto Verde)", "color": (60, 76, 231), "accion": "Llevar a Punto Verde Movil o Fijo.", "es_especial": True},
    "e-cigarette":          {"label": "Cigarrillo Electronico / Vaper", "emoji": "🔋", "tipo": "Residuo Especial (Punto Verde)", "color": (60, 76, 231), "accion": "Llevar a Punto Verde Movil o Fijo.", "es_especial": True},
    "e-waste":              {"label": "RAEE / Electronico",   "emoji": "🔋", "tipo": "Residuo Especial (Punto Verde)", "color": (60, 76, 231), "accion": "Llevar a Punto Verde Movil o Fijo (hasta 10 aparatos por persona).", "es_especial": True},
    "e_waste":              {"label": "RAEE / Electronico",   "emoji": "🔋", "tipo": "Residuo Especial (Punto Verde)", "color": (60, 76, 231), "accion": "Llevar a Punto Verde Movil o Fijo (hasta 10 aparatos por persona).", "es_especial": True},
    "electronic":           {"label": "RAEE / Electronico",   "emoji": "🔋", "tipo": "Residuo Especial (Punto Verde)", "color": (60, 76, 231), "accion": "Llevar a Punto Verde Movil o Fijo (hasta 10 aparatos por persona).", "es_especial": True},
    "electronics":          {"label": "RAEE / Electronico",   "emoji": "🔋", "tipo": "Residuo Especial (Punto Verde)", "color": (60, 76, 231), "accion": "Llevar a Punto Verde Movil o Fijo (hasta 10 aparatos por persona).", "es_especial": True},
    "circuit":              {"label": "RAEE / Electronico",   "emoji": "🔋", "tipo": "Residuo Especial (Punto Verde)", "color": (60, 76, 231), "accion": "Llevar a Punto Verde Movil o Fijo (hasta 10 aparatos por persona).", "es_especial": True},
    "battery":              {"label": "Pila / Bateria",       "emoji": "🔋", "tipo": "Residuo Especial (Punto Verde)", "color": (60, 76, 231), "accion": "Llevar a buzones de pilas en Puntos Verdes o Farmacias habilitadas.", "es_especial": True},
    "aa battery":           {"label": "Pila / Bateria",       "emoji": "🔋", "tipo": "Residuo Especial (Punto Verde)", "color": (60, 76, 231), "accion": "Llevar a buzones de pilas en Puntos Verdes o Farmacias habilitadas.", "es_especial": True},
    "aaa battery":          {"label": "Pila / Bateria",       "emoji": "🔋", "tipo": "Residuo Especial (Punto Verde)", "color": (60, 76, 231), "accion": "Llevar a buzones de pilas en Puntos Verdes o Farmacias habilitadas.", "es_especial": True},
    "cylindrical battery":  {"label": "Pila / Bateria",       "emoji": "🔋", "tipo": "Residuo Especial (Punto Verde)", "color": (60, 76, 231), "accion": "Llevar a buzones de pilas en Puntos Verdes o Farmacias habilitadas.", "es_especial": True},
    "batteries":            {"label": "Pila / Bateria",       "emoji": "🔋", "tipo": "Residuo Especial (Punto Verde)", "color": (60, 76, 231), "accion": "Llevar a buzones de pilas en Puntos Verdes o Farmacias habilitadas.", "es_especial": True},
    "alkaline battery":     {"label": "Pila / Bateria",       "emoji": "🔋", "tipo": "Residuo Especial (Punto Verde)", "color": (60, 76, 231), "accion": "Llevar a buzones de pilas en Puntos Verdes o Farmacias habilitadas.", "es_especial": True},
    "small battery":        {"label": "Pila / Bateria",       "emoji": "🔋", "tipo": "Residuo Especial (Punto Verde)", "color": (60, 76, 231), "accion": "Llevar a buzones de pilas en Puntos Verdes o Farmacias habilitadas.", "es_especial": True},
    "pila":                 {"label": "Pila / Bateria",       "emoji": "🔋", "tipo": "Residuo Especial (Punto Verde)", "color": (60, 76, 231), "accion": "Llevar a buzones de pilas en Puntos Verdes o Farmacias habilitadas.", "es_especial": True},
    
    # Dispositivos moviles (RAEE)
    "cell phone":           {"label": "Celular / RAEE",       "emoji": "📱", "tipo": "Residuo Especial (Punto Verde)", "color": (60, 76, 231), "accion": "Llevar a Punto Verde. Contiene metales pesados y componentes recuperables.", "es_especial": True},
    "smartphone":           {"label": "Celular / RAEE",       "emoji": "📱", "tipo": "Residuo Especial (Punto Verde)", "color": (60, 76, 231), "accion": "Llevar a Punto Verde. Contiene metales pesados y componentes recuperables.", "es_especial": True},
    "smartphone touchscreen": {"label": "Celular / RAEE",     "emoji": "📱", "tipo": "Residuo Especial (Punto Verde)", "color": (60, 76, 231), "accion": "Llevar a Punto Verde. Contiene metales pesados y componentes recuperables.", "es_especial": True},
    "mobile phone":         {"label": "Celular / RAEE",       "emoji": "📱", "tipo": "Residuo Especial (Punto Verde)", "color": (60, 76, 231), "accion": "Llevar a Punto Verde. Contiene metales pesados y componentes recuperables.", "es_especial": True},
    "telephone":            {"label": "Celular / RAEE",       "emoji": "📱", "tipo": "Residuo Especial (Punto Verde)", "color": (60, 76, 231), "accion": "Llevar a Punto Verde. Contiene metales pesados y componentes recuperables.", "es_especial": True},
    
    # Computadoras y perifericos
    "screen":               {"label": "Monitor / RAEE",       "emoji": "🖥️", "tipo": "Residuo Especial (Punto Verde)", "color": (231, 76, 60), "accion": "Llevar a Punto Verde Especial o Centro Verde habilitado.", "es_especial": True},
    "monitor":              {"label": "Monitor / RAEE",       "emoji": "🖥️", "tipo": "Residuo Especial (Punto Verde)", "color": (231, 76, 60), "accion": "Llevar a Punto Verde Especial o Centro Verde habilitado.", "es_especial": True},
    "laptop":               {"label": "Laptop / RAEE",        "emoji": "💻", "tipo": "Residuo Especial (Punto Verde)", "color": (231, 76, 60), "accion": "Llevar a Punto Verde. Tratamiento diferenciado de circuitos y bateria.", "es_especial": True},
    "mouse":                {"label": "Periferico / RAEE",    "emoji": "🖱️", "tipo": "Residuo Especial (Punto Verde)", "color": (231, 76, 60), "accion": "Llevar a Punto Verde junto a cables y teclados.", "es_especial": True},
    "keyboard":             {"label": "Teclado / RAEE",       "emoji": "⌨️", "tipo": "Residuo Especial (Punto Verde)", "color": (231, 76, 60), "accion": "Llevar a Punto Verde Movil o Fijo.", "es_especial": True},
    "remote":               {"label": "Control / RAEE",       "emoji": "🔋", "tipo": "Residuo Especial (Punto Verde)", "color": (231, 76, 60), "accion": "Retirar pilas y llevar el control al Punto Verde.", "es_especial": True},
    "toner":                {"label": "Toner / Cartucho",     "emoji": "🖨️", "tipo": "Residuo Especial (Punto Verde)", "color": (231, 76, 60), "accion": "Llevar a Punto Verde (limpio, sin derrames, hasta 3 por persona).", "es_especial": True},
    "lamp":                 {"label": "Lampara / Tubo Fluo",  "emoji": "💡", "tipo": "Residuo Peligroso (Punto Verde)", "color": (231, 76, 60), "accion": "Llevar en caja/envoltura (contiene mercurio). Hasta 5 por persona/dia.", "es_especial": True},
    
    # Aceite Vegetal Usado (AVU)
    "oil":                  {"label": "Aceite Usado (AVU)",   "emoji": "🍳", "tipo": "Residuo Especial (Punto Verde)", "color": (231, 76, 60), "accion": "Enfriar y envasar en botella plastica cerrada. 1L contamina 1000L de agua.", "es_especial": True},
    "cooking oil":          {"label": "Aceite Usado (AVU)",   "emoji": "🍳", "tipo": "Residuo Especial (Punto Verde)", "color": (231, 76, 60), "accion": "Enfriar y envasar en botella plastica cerrada. 1L contamina 1000L de agua.", "es_especial": True},
    "cooking oil bottle":   {"label": "Aceite Usado (AVU)",   "emoji": "🍳", "tipo": "Residuo Especial (Punto Verde)", "color": (231, 76, 60), "accion": "Enfriar y envasar en botella plastica cerrada. 1L contamina 1000L de agua.", "es_especial": True},
    "oil bottle":           {"label": "Aceite Usado (AVU)",   "emoji": "🍳", "tipo": "Residuo Especial (Punto Verde)", "color": (231, 76, 60), "accion": "Enfriar y envasar en botella plastica cerrada. 1L contamina 1000L de agua.", "es_especial": True},
    "used cooking oil":     {"label": "Aceite Usado (AVU)",   "emoji": "🍳", "tipo": "Residuo Especial (Punto Verde)", "color": (231, 76, 60), "accion": "Enfriar y envasar en botella plastica cerrada. 1L contamina 1000L de agua.", "es_especial": True},
    "aceite":               {"label": "Aceite Usado (AVU)",   "emoji": "🍳", "tipo": "Residuo Especial (Punto Verde)", "color": (231, 76, 60), "accion": "Enfriar y envasar en botella plastica cerrada. 1L contamina 1000L de agua.", "es_especial": True},

    # Envases cosmeticos / cremas / lociones / sueros / esmaltes
    "lotion bottle":        {"label": "Plastico / Envases",   "emoji": "🧴", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Vaciar, enjuagar y compactar.", "es_especial": False},
    "cream bottle":         {"label": "Plastico / Envases",   "emoji": "🧴", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Vaciar, enjuagar y compactar.", "es_especial": False},
    "shampoo bottle":       {"label": "Plastico / Envases",   "emoji": "🧴", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Vaciar, enjuagar y compactar.", "es_especial": False},
    "cosmetic bottle":      {"label": "Frasco / Envase Cosmetico (Vidrio / Plastico)", "emoji": "🧴", "tipo": "Reciclable Seco (Verde)", "color": (46, 204, 113), "accion": "Vaciar completamente y limpiar antes de depositar en Contenedor Verde.", "es_especial": False},
    "nail polish bottle":   {"label": "Frasco Cosmetico / Esmalte (Vidrio + Tapa Plastica)", "emoji": "💅", "tipo": "Reciclable Seco (Verde) si esta vacio y limpio", "color": (52, 152, 219), "accion": "Frasco de vidrio con tapa/pincel plastico: si esta vacio y seco va al Contenedor Verde; con restos de esmalte o acetona cerrar bien y descartar en Bolsa Negra.", "es_especial": False},
    "nail polish":          {"label": "Frasco Cosmetico / Esmalte (Vidrio + Tapa Plastica)", "emoji": "💅", "tipo": "Reciclable Seco (Verde) si esta vacio y limpio", "color": (52, 152, 219), "accion": "Frasco de vidrio con tapa/pincel plastico: si esta vacio y seco va al Contenedor Verde; con restos de esmalte o acetona cerrar bien y descartar en Bolsa Negra.", "es_especial": False},
    "perfume bottle":       {"label": "Frasco de Vidrio / Cosmetico", "emoji": "🍾", "tipo": "Reciclable Seco (Verde)", "color": (52, 152, 219), "accion": "Frasco de vidrio reciclable. Vaciar y depositar en Contenedor Verde.", "es_especial": False},
    "small glass bottle":   {"label": "Frasco de Vidrio / Envase", "emoji": "🍾", "tipo": "Reciclable Seco (Verde)", "color": (52, 152, 219), "accion": "Vaciar y limpiar antes de depositar en Contenedor Verde.", "es_especial": False},
    "dropper bottle":       {"label": "Frasco / Envase Cosmetico (Vidrio / Plastico)", "emoji": "🧴", "tipo": "Reciclable Seco (Verde)", "color": (46, 204, 113), "accion": "Vaciar y limpiar antes de depositar en Contenedor Verde.", "es_especial": False},
    "serum bottle":         {"label": "Frasco / Envase Cosmetico (Vidrio / Plastico)", "emoji": "🧴", "tipo": "Reciclable Seco (Verde)", "color": (46, 204, 113), "accion": "Vaciar y limpiar antes de depositar en Contenedor Verde.", "es_especial": False},
    "white plastic jar":    {"label": "Pote / Envase Plastico","emoji": "🧴", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Limpio y seco. Apto Contenedor Verde.", "es_especial": False},
    "ointment jar":         {"label": "Pote / Envase Plastico","emoji": "🧴", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Limpio y seco. Apto Contenedor Verde.", "es_especial": False},

    # Organicos / No Reciclables y Sanitario
    "organic":              {"label": "Organico / No Reciclable", "emoji": "🍂", "tipo": "Compost / Bolsa Negra (No Reciclable)", "color": (139, 69, 19), "accion": "Si es resto vegetal va a compostera; si es residuo no reciclable o descartable va a Bolsa Negra.", "es_especial": False},
    "biological":           {"label": "Organico / No Reciclable", "emoji": "🍂", "tipo": "Compost / Bolsa Negra (No Reciclable)", "color": (139, 69, 19), "accion": "Si es resto vegetal va a compostera; si es residuo no reciclable o descartable va a Bolsa Negra.", "es_especial": False},
    "medical":              {"label": "Residuo Sanitario",    "emoji": "⚠️", "tipo": "Residuo Peligroso",        "color": (192, 57, 43),   "accion": "Descartar en bolsa cerrada especial o punto farmaceutico.", "es_especial": True},

    # Textiles y Calzado
    "clothes":              {"label": "Ropa / Textil",        "emoji": "👕", "tipo": "Donación / Residuo Especial", "color": (155, 89, 182), "accion": "En buen estado donar a ONGs o ferias. Ropa rota a Punto Verde o bolsa negra.", "es_especial": False},
    "folded clothes":       {"label": "Ropa / Textil",        "emoji": "👕", "tipo": "Donación / Residuo Especial", "color": (155, 89, 182), "accion": "En buen estado donar a ONGs o ferias. Ropa rota a Punto Verde o bolsa negra.", "es_especial": False},
    "folded shirt":         {"label": "Ropa / Textil",        "emoji": "👕", "tipo": "Donación / Residuo Especial", "color": (155, 89, 182), "accion": "En buen estado donar a ONGs o ferias. Ropa rota a Punto Verde o bolsa negra.", "es_especial": False},
    "t-shirt":              {"label": "Ropa / Textil",        "emoji": "👕", "tipo": "Donación / Residuo Especial", "color": (155, 89, 182), "accion": "En buen estado donar a ONGs o ferias. Ropa rota a Punto Verde o bolsa negra.", "es_especial": False},
    "shirt":                {"label": "Ropa / Textil",        "emoji": "👕", "tipo": "Donación / Residuo Especial", "color": (155, 89, 182), "accion": "En buen estado donar a ONGs o ferias. Ropa rota a Punto Verde o bolsa negra.", "es_especial": False},
    "sweater":              {"label": "Ropa / Textil",        "emoji": "👕", "tipo": "Donación / Residuo Especial", "color": (155, 89, 182), "accion": "En buen estado donar a ONGs o ferias. Ropa rota a Punto Verde o bolsa negra.", "es_especial": False},
    "hoodie":               {"label": "Ropa / Textil",        "emoji": "👕", "tipo": "Donación / Residuo Especial", "color": (155, 89, 182), "accion": "En buen estado donar a ONGs o ferias. Ropa rota a Punto Verde o bolsa negra.", "es_especial": False},
    "jacket":               {"label": "Ropa / Textil",        "emoji": "👕", "tipo": "Donación / Residuo Especial", "color": (155, 89, 182), "accion": "En buen estado donar a ONGs o ferias. Ropa rota a Punto Verde o bolsa negra.", "es_especial": False},
    "pants":                {"label": "Ropa / Textil",        "emoji": "👕", "tipo": "Donación / Residuo Especial", "color": (155, 89, 182), "accion": "En buen estado donar a ONGs o ferias. Ropa rota a Punto Verde o bolsa negra.", "es_especial": False},
    "jeans":                {"label": "Ropa / Textil",        "emoji": "👕", "tipo": "Donación / Residuo Especial", "color": (155, 89, 182), "accion": "En buen estado donar a ONGs o ferias. Ropa rota a Punto Verde o bolsa negra.", "es_especial": False},
    "blanket":              {"label": "Ropa / Textil",        "emoji": "👕", "tipo": "Donación / Residuo Especial", "color": (155, 89, 182), "accion": "En buen estado donar a ONGs o ferias. Textil limpio a Punto Verde.", "es_especial": False},
    "fleece":               {"label": "Ropa / Textil",        "emoji": "👕", "tipo": "Donación / Residuo Especial", "color": (155, 89, 182), "accion": "En buen estado donar a ONGs o ferias. Textil limpio a Punto Verde.", "es_especial": False},
    "fabric":               {"label": "Ropa / Textil",        "emoji": "👕", "tipo": "Donación / Residuo Especial", "color": (155, 89, 182), "accion": "En buen estado donar a ONGs o ferias. Textil limpio a Punto Verde.", "es_especial": False},
    "textile":              {"label": "Ropa / Textil",        "emoji": "👕", "tipo": "Donación / Residuo Especial", "color": (155, 89, 182), "accion": "En buen estado donar a ONGs o ferias. Textil limpio a Punto Verde.", "es_especial": False},
    "towel":                {"label": "Ropa / Textil",        "emoji": "👕", "tipo": "Donación / Residuo Especial", "color": (155, 89, 182), "accion": "En buen estado donar a ONGs o ferias. Textil limpio a Punto Verde.", "es_especial": False},
    "garment":              {"label": "Ropa / Textil",        "emoji": "👕", "tipo": "Donación / Residuo Especial", "color": (155, 89, 182), "accion": "En buen estado donar a ONGs o ferias. Textil limpio a Punto Verde.", "es_especial": False},
    "plush fabric":         {"label": "Ropa / Textil",        "emoji": "👕", "tipo": "Donación / Residuo Especial", "color": (155, 89, 182), "accion": "En buen estado donar a ONGs o ferias. Textil limpio a Punto Verde.", "es_especial": False},
    "shoes":                {"label": "Calzado / Zapatillas", "emoji": "👟", "tipo": "Donación / Basura Comun",     "color": (155, 89, 182), "accion": "En buen estado donar a ferias o instituciones. Calzado roto a bolsa negra.", "es_especial": False},
    "shoe":                 {"label": "Calzado / Zapatillas", "emoji": "👟", "tipo": "Donación / Basura Comun",     "color": (155, 89, 182), "accion": "En buen estado donar a ferias o instituciones. Calzado roto a bolsa negra.", "es_especial": False},
    "sneakers":             {"label": "Calzado / Zapatillas", "emoji": "👟", "tipo": "Donación / Basura Comun",     "color": (155, 89, 182), "accion": "En buen estado donar a ferias o instituciones. Calzado roto a bolsa negra.", "es_especial": False},
    "sneaker":              {"label": "Calzado / Zapatillas", "emoji": "👟", "tipo": "Donación / Basura Comun",     "color": (155, 89, 182), "accion": "En buen estado donar a ferias o instituciones. Calzado roto a bolsa negra.", "es_especial": False},
    "canvas shoes":         {"label": "Calzado / Zapatillas", "emoji": "👟", "tipo": "Donación / Basura Comun",     "color": (155, 89, 182), "accion": "En buen estado donar a ferias o instituciones. Calzado roto a bolsa negra.", "es_especial": False},
    "white sneakers":       {"label": "Calzado / Zapatillas", "emoji": "👟", "tipo": "Donación / Basura Comun",     "color": (155, 89, 182), "accion": "En buen estado donar a ferias o instituciones. Calzado roto a bolsa negra.", "es_especial": False},
    "tennis shoes":         {"label": "Calzado / Zapatillas", "emoji": "👟", "tipo": "Donación / Basura Comun",     "color": (155, 89, 182), "accion": "En buen estado donar a ferias o instituciones. Calzado roto a bolsa negra.", "es_especial": False},
    "running shoes":        {"label": "Calzado / Zapatillas", "emoji": "👟", "tipo": "Donación / Basura Comun",     "color": (155, 89, 182), "accion": "En buen estado donar a ferias o instituciones. Calzado roto a bolsa negra.", "es_especial": False},
    "pair of shoes":        {"label": "Calzado / Zapatillas", "emoji": "👟", "tipo": "Donación / Basura Comun",     "color": (155, 89, 182), "accion": "En buen estado donar a ferias o instituciones. Calzado roto a bolsa negra.", "es_especial": False},
    "footwear":             {"label": "Calzado / Zapatillas", "emoji": "👟", "tipo": "Donación / Basura Comun",     "color": (155, 89, 182), "accion": "En buen estado donar a ferias o instituciones. Calzado roto a bolsa negra.", "es_especial": False},

    # Tablets y Lectores Electronicos (RAEE)
    "tablet":               {"label": "Tablet / E-Reader (RAEE)", "emoji": "📱", "tipo": "Residuo Especial (Punto Verde)", "color": (60, 76, 231), "accion": "Aparato electronico con bateria y pantalla. Llevar a Punto Verde.", "es_especial": True},
    "e-reader":             {"label": "Tablet / E-Reader (RAEE)", "emoji": "📱", "tipo": "Residuo Especial (Punto Verde)", "color": (60, 76, 231), "accion": "Aparato electronico con bateria y pantalla. Llevar a Punto Verde.", "es_especial": True},
    "kindle":               {"label": "Tablet / E-Reader (RAEE)", "emoji": "📱", "tipo": "Residuo Especial (Punto Verde)", "color": (60, 76, 231), "accion": "Aparato electronico con bateria y pantalla. Llevar a Punto Verde.", "es_especial": True},

    # Basura General
    "trash":                {"label": "Basura General",       "emoji": "🗑️", "tipo": "Basura Comun (Bolsa Negra)",  "color": (127, 140, 141), "accion": "Depositar en bolsa negra de residuos comunes no reciclables.", "es_especial": False},
}

# Mapa de subcadenas keyword → clave base en WASTE_MAP (fallback por substring)
_KEYWORD_FALLBACK = [
    ("ruler",          "plastic"),
    ("vape",           "vape"),
    ("box mod",        "vape"),
    ("e-cigarette",    "electronic cigarette"),
    ("electronic cigarette", "electronic cigarette"),
    ("lighter",        "lighter"),
    ("plastic bag",    "plastic bag"),
    ("grocery bag",    "plastic bag"),
    ("shopping bag",   "plastic bag"),
    ("polyethylene bag", "plastic bag"),
    ("wrapper",        "plastic wrapper"),
    ("bag",            "plastic bag"),
    ("drinking glass", "drinking glass"),
    ("glass cup",      "glass cup"),
    ("glass",          "glass"),
    ("sharpie",        "marker"),
    ("permanent marker", "marker"),
    ("marker pen",     "marker"),
    ("highlighter",    "marker"),
    ("felt tip",       "marker"),
    ("marker",         "marker"),
    ("pen",            "pen"),
    ("pencil",         "pencil"),
    ("thermos",        "thermos"),
    ("cosmetic jar",   "cosmetic jar"),
    ("cream jar",      "cream jar"),
    ("ointment jar",   "ointment jar"),
    ("white plastic jar", "white plastic jar"),
    ("plastic jar",    "plastic jar"),
    ("plastic tub",    "plastic tub"),
    ("plastic pot",    "plastic pot"),
    ("plastic lid",    "plastic lid"),
    ("plastic cap",    "plastic cap"),
    ("plastic box",    "plastic box"),
    ("pot",            "plastic pot"),
    ("jar",            "plastic jar"),
    ("tub",            "plastic tub"),
    ("container",      "container"),
    ("cup",            "cup"),
    ("water bottle",   "plastic bottle"),
    ("soda bottle",    "plastic bottle"),
    ("juice bottle",   "plastic bottle"),
    ("plastic bottle", "plastic bottle"),
    ("nail polish",    "nail polish bottle"),
    ("perfume bottle", "perfume bottle"),
    ("small glass bottle", "small glass bottle"),
    ("dropper bottle", "dropper bottle"),
    ("serum bottle",   "serum bottle"),
    ("cosmetic bottle","cosmetic bottle"),
    ("bottle",         "bottle"),
    ("lotion",         "lotion bottle"),
    ("cream",          "cream bottle"),
    ("shampoo",        "shampoo bottle"),
    ("scissors",       "scissors"),
    ("glasses case",   "glasses case"),
    ("eyeglasses case","eyeglasses case"),
    ("hard case",      "hard case"),
    ("can",            "can"),
    ("tin",            "tin can"),
    ("steel",          "steel can"),
    ("metal",          "metal"),
    ("spray",          "spray can"),
    ("aerosol",        "aerosol can"),
    ("freshener",      "air freshener spray"),
    ("plastic",        "plastic"),
    ("cardboard",      "cardboard"),
    ("shoebox",        "cardboard"),
    ("packaging box",  "cardboard"),
    ("medicine box",   "cardboard"),
    ("cosmetic box",   "cardboard"),
    ("tetra pak",      "tetra pak"),
    ("carton",         "cardboard"),
    ("corrugated",     "cardboard"),
    ("invoice",        "paper"),
    ("brochure",       "paper"),
    ("flyer",          "paper"),
    ("pamphlet",       "paper"),
    ("envelope",       "paper"),
    ("document",       "paper"),
    ("bill",           "paper"),
    ("paper",          "paper"),
    ("tablet",         "tablet"),
    ("e-reader",       "e-reader"),
    ("kindle",         "kindle"),
    ("cell phone",     "cell phone"),
    ("smartphone",     "cell phone"),
    ("mobile phone",   "cell phone"),
    ("telephone",      "cell phone"),
    ("screen",         "screen"),
    ("monitor",        "screen"),
    ("electronic",     "electronic"),
    ("circuit",        "circuit"),
    ("battery",        "battery"),
    ("batteries",      "battery"),
    ("pila",           "battery"),
    ("biological",     "biological"),
    ("organic",        "organic"),
    ("blanket",        "clothes"),
    ("fleece",         "clothes"),
    ("fabric",         "clothes"),
    ("textile",        "clothes"),
    ("towel",          "clothes"),
    ("garment",        "clothes"),
    ("t-shirt",        "clothes"),
    ("shirt",          "clothes"),
    ("sweater",        "clothes"),
    ("hoodie",         "clothes"),
    ("jacket",         "clothes"),
    ("pants",          "clothes"),
    ("jeans",          "clothes"),
    ("clothes",        "clothes"),
    ("sneaker",        "shoes"),
    ("footwear",       "shoes"),
    ("shoes",          "shoes"),
    ("shoe",           "shoes"),
    ("trash",          "trash"),
    ("oil",            "oil"),
    ("aceite",         "oil"),
]

# Clases que NUNCA deben marcarse como residuo (Personas, anteojos, partes del cuerpo, mobiliario y fondo)
# Nota: Las prendas ('shirt', 't-shirt', 'hoodie', 'sweater') se filtran contextualmente solo cuando hay una persona usandolas.
IGNORORED_CLOTHING_WHEN_PERSON = {"shirt", "t-shirt", "folded shirt", "sweater", "jacket", "hoodie", "coat", "pants", "jeans"}
IGNORED_CLASSES = {
    "person", "human face", "face", "head", "man", "woman", "human", "body", "arm", "hand", "finger",
    "eyeglasses", "glasses", "spectacles", "sunglasses",
    "chair", "couch", "sofa", "bed", "dining table", "table", "wooden table", "desk", "tv", "laptop-screen",
    "refrigerator", "microwave", "oven", "sink", "door", "window", "cabinet", "wardrobe", "cupboard", "closet", "drawer",
    "toilet", "clock", "vase", "potted plant", "mirror", "picture frame", "wooden frame", "painting", "poster",
    "wall", "floor", "ceiling", "ceiling fan", "fan", "shelf", "room", "indoor", "wooden door", "furniture", "wood furniture",
    "plastic crate", "shopping basket", "grass", "lawn", "plant"
}

DEFAULT_WASTE = {
    "label": "Residuo General", "emoji": "🗑️",
    "tipo": "Basura Comun / Organicos",
    "color": (127, 140, 141),
    "accion": "Depositar en bolsa negra de residuos generales.",
    "es_especial": False
}

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATHS = {
    "yolov8s_world": str(BASE_DIR / "yolov8s-worldv2.pt"),
    "waste_specialized": str(BASE_DIR / "models" / "waste_yolov8.pt"),
    "best": str(BASE_DIR / "models" / "best.pt"),
    "yolov8n": str(BASE_DIR / "yolov8n.pt"),
}


WORLD_CLASSES = [
    # Personas, Rostros, Anteojos y Ropa puesta (descarte de usuario y vestimenta)
    "person", "human face", "eyeglasses", "glasses", "hand",
    "hoodie", "shirt", "sweater", "jacket",
    # Mobiliario y entorno de fondo (descarte de puertas, mesas, canastos, ventiladores, pasto y paredes)
    "wardrobe", "closet", "cabinet", "wooden door", "door", "shelf", "furniture",
    "wooden table", "table", "desk", "plastic crate", "shopping basket", "grass",
    "ceiling fan", "fan", "mirror", "picture frame", "wooden frame", "window", "wall",
    # Vidrio y Vasos
    "drinking glass", "glass cup", "glass bottle", "glass", "wine glass",
    # Potes Redondos, Envases Cosmeticos, Estuches, Tapas y Envases Plasticos
    "round plastic container", "white plastic jar", "ointment jar", "cosmetic jar", "cream jar",
    "plastic jar", "plastic tub", "plastic pot", "plastic container", "plastic lid", "plastic cap", "jar", "pot",
    "eyeglasses case", "glasses case", "hard case",
    # Bolsas Plasticas y Envoltorios
    "plastic bag", "white plastic bag", "supermarket plastic bag", "grocery bag", "shopping bag", "plastic wrapper",
    # Vasos
    "cup", "plastic cup", "disposable cup",
    # Encendedores
    "lighter", "cigarette lighter", "plastic lighter",
    # Botellas y Envases (con descriptores explicitos para evitar que la etiqueta cilindrica se confunda con lata o vaper)
    "clear plastic bottle", "water bottle", "soda bottle", "juice bottle", "pet bottle", "plastic bottle", "bottle", "thermos",
    "lotion bottle", "cream bottle", "shampoo bottle", "cosmetic bottle", "nail polish bottle", "perfume bottle", "small glass bottle", "dropper bottle", "serum bottle",
    # Utiles, Tijeras y Objetos Plasticos / Oficina / Medicion
    "scissors", "pair of scissors", "metal scissors",
    "plastic ruler", "ruler", "pen", "plastic pen", "ballpoint pen",
    "marker", "marker pen", "permanent marker", "sharpie marker", "felt tip pen", "highlighter pen", "pencil",
    # Aceite
    "cooking oil bottle", "oil bottle", "cooking oil", "used cooking oil",
    # Metales, Latas y Aerosoles
    "aluminum can", "soda can", "beer can", "beverage can", "drink can", "tin can", "can",
    "spray can", "aerosol can", "aerosol", "air freshener spray", "deodorant spray",
    # Papel, Facturas, Folletos y Carton
    "cardboard box", "shoebox", "black cardboard box", "small cardboard box", "packaging box", "medicine box", "cosmetic box", "plastic box",
    "cardboard", "corrugated cardboard", "paper", "sheet of paper", "notebook paper",
    "printed document", "paper bill", "invoice", "brochure", "flyer", "pamphlet", "envelope",
    # Textiles, Remeras, Mantas, Ropa doblada y Calzado / Zapatillas (para donacion o Punto Verde)
    "t-shirt", "folded shirt", "folded clothes", "blanket", "fleece", "fabric", "textile", "towel", "plush fabric", "pants", "jeans",
    "shoes", "shoe", "sneakers", "white sneakers", "canvas shoes", "tennis shoes", "running shoes", "pair of shoes", "footwear",
    # Vapers, Baterias y RAEE (solo 2 prompts especificos para evitar el sesgo multi-prompt que clasificaba todo como vaper)
    "disposable vape", "electronic cigarette",
    "battery", "batteries", "aa battery", "aaa battery", "cylindrical battery", "alkaline battery", "duracell battery",
    "smartphone touchscreen", "tablet", "e-reader", "kindle", "laptop", "mouse", "keyboard",
    # Discos y Multicapa
    "compact disc", "cd", "tetra pak"
]


def _apply_or_cache_world_classes(detector: YOLO, classes: list[str]) -> None:
    """
    Aplica los embeddings de texto (txt_feats) a YOLO-World v2.
    Si existe el cache precalculado en disco (world_txt_feats.pt, ~170 KB), lo inyecta directamente
    EVITANDO cargar el modelo CLIP ViT-B/32 (~400 MB de RAM). Si no existe, lo calcula una vez,
    lo persiste en disco y libera inmediatamente CLIP de la memoria.
    """
    cache_candidates = [
        BASE_DIR / "models" / "world_txt_feats.pt",
        Path("models/world_txt_feats.pt"),
        Path(tempfile.gettempdir()) / "ecociudad_world_txt_feats.pt",
    ]
    for c_path in cache_candidates:
        if c_path.exists():
            try:
                cached = torch.load(str(c_path), map_location="cpu", weights_only=False)
                if isinstance(cached, dict) and cached.get("classes") == classes and "txt_feats" in cached:
                    txt_feats = cached["txt_feats"]
                    detector.model.txt_feats = txt_feats
                    detector.model.nc = len(classes)
                    if hasattr(detector.model, "model") and len(detector.model.model) > 0:
                        detector.model.model[-1].nc = len(classes)
                    detector.model.names = {i: c for i, c in enumerate(classes)}
                    return
            except Exception:
                pass

    # Si no existe cache, calcular con CLIP una unica vez y guardar
    detector.set_classes(classes)
    if hasattr(detector, "model") and hasattr(detector.model, "txt_feats"):
        payload = {
            "classes": list(classes),
            "txt_feats": detector.model.txt_feats.detach().cpu(),
        }
        for save_target in (BASE_DIR / "models" / "world_txt_feats.pt", Path(tempfile.gettempdir()) / "ecociudad_world_txt_feats.pt"):
            try:
                save_target.parent.mkdir(parents=True, exist_ok=True)
                torch.save(payload, str(save_target))
                break
            except Exception:
                pass

    # Liberar el modelo CLIP interno tras calcular txt_feats (ahorra ~380-450 MB de RAM)
    if hasattr(detector, "model") and hasattr(detector.model, "clip_model"):
        detector.model.clip_model = None
        try:
            delattr(detector.model, "clip_model")
        except Exception:
            pass
    for mod_name in list(sys.modules.keys()):
        if mod_name == "clip" or mod_name.startswith("clip."):
            sys.modules.pop(mod_name, None)
    trim_memory()


@st.cache_resource(show_spinner="Iniciando Pipeline de IA Optimizado...", max_entries=1)
def load_cascade_engine() -> dict:
    """
    Carga el motor de visión híbrido en memoria con bajo consumo de RAM:
    1. Modelo Detector primario con Bounding Boxes (YOLO-World v2) usando cache de embeddings.
    2. Modelo Clasificador secundario (YOLOv8s-cls / best.pt) en modo Lazy (bajo demanda)
       para no ocupar RAM adicional hasta que realmente se requiera un fallback.
    """
    trim_memory()

    # 1. Detector primario con Bounding Boxes
    det_candidates = [
        BASE_DIR / "yolov8s-worldv2.pt",
        Path("yolov8s-worldv2.pt"),
        BASE_DIR / "models" / "waste_yolov8.pt",
        Path("models/waste_yolov8.pt"),
        BASE_DIR / "yolov8n.pt",
        Path("yolov8n.pt"),
    ]
    det_path = next((str(c) for c in det_candidates if c.exists()), "yolov8n.pt")
    detector = YOLO(det_path)

    if "world" in str(det_path).lower():
        try:
            _apply_or_cache_world_classes(detector, WORLD_CLASSES)
        except Exception as e:
            print(f"[EcoCiudad] Advertencia al configurar clases en YOLO-World: {e}")

    # 2. Clasificador secundario (ruta registrada para carga Lazy bajo demanda)
    cls_candidates = [
        BASE_DIR / "models" / "best.pt",
        Path("models/best.pt"),
        BASE_DIR / "best.pt",
        Path("best.pt"),
    ]
    cls_path = next((str(c) for c in cls_candidates if c.exists()), None)

    trim_memory()
    return {
        "detector": detector,
        "classifier": None,
        "cls_path": cls_path,
    }


def _get_lazy_classifier(engine: dict) -> YOLO | None:
    """Carga el clasificador secundario unicamente cuando el detector primario no halla objetos."""
    if not isinstance(engine, dict):
        return None
    cls_model = engine.get("classifier")
    if cls_model is not None:
        return cls_model
    cls_path = engine.get("cls_path")
    if cls_path and os.path.exists(cls_path):
        try:
            cls_model = YOLO(cls_path)
            engine["classifier"] = cls_model
            trim_memory()
            return cls_model
        except Exception:
            return None
    return None


def load_model(model_key: str = "cascade") -> dict | YOLO:
    """Carga el motor de visión en cascada por defecto o un modelo específico si se indica."""
    if model_key in ("cascade", "hybrid", "yolov8s_world", "default"):
        return load_cascade_engine()
    path = MODEL_PATHS.get(model_key, str(BASE_DIR / "yolov8s-worldv2.pt"))
    if not os.path.exists(path):
        candidates = [
            BASE_DIR / "yolov8s-worldv2.pt",
            Path("yolov8s-worldv2.pt"),
            BASE_DIR / "models" / "waste_yolov8.pt",
            Path("models/waste_yolov8.pt"),
            BASE_DIR / "models" / "best.pt",
            Path("models/best.pt"),
            BASE_DIR / "best.pt",
            Path("best.pt"),
            BASE_DIR / "yolov8n.pt",
            Path("yolov8n.pt"),
        ]
        path = next((str(c) for c in candidates if c.exists()), "yolov8n.pt")
    return YOLO(path)


def get_waste_info(class_name: str) -> dict | None:
    norm = class_name.lower().strip()
    
    # Mapeo unicamente de sinonimos estrictos de celulares
    if norm in ("smartphone", "smartphone touchscreen", "mobile phone", "telephone"):
        norm = "cell phone"
    # Mapeo directo a la categoria estandar de Plastico
    if norm in ("ruler", "plastic ruler"):
        norm = "plastic"
    norm_spaced = norm.replace("-", " ").replace("_", " ").strip()

    # 1. Busqueda exacta (prioriza clases compuestas como 'glasses case' antes del chequeo por token)
    if norm in WASTE_MAP:
        return WASTE_MAP[norm]

    # 2. Normalizar guiones y underscores → espacios y buscar exacto
    if norm_spaced in WASTE_MAP:
        return WASTE_MAP[norm_spaced]

    # Verificacion estricta de clases ignoradas (personas, rostros, muebles, fondo)
    if norm in IGNORED_CLASSES or norm_spaced in IGNORED_CLASSES:
        return None

    for ignored in IGNORED_CLASSES:
        if ignored == norm or ignored.replace("-", " ").replace("_", " ") == norm_spaced:
            return None

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

    # Si no es un residuo reconocido, retornar None (NO forzar cajas falsas)
    return None


def is_frame_blank_or_dark(frame_bgr: np.ndarray, min_mean: float = 24.0, min_std: float = 7.0) -> bool:
    """
    Determina si el fotograma esta negro, ocluido (tapa/shutter de la camara cerrado),
    o carece de informacion visual util.
    Calculo ultrarrapido (<0.2 ms) mediante submuestreo de pixeles.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return True
    sample = frame_bgr[::8, ::8]
    mean_val = float(np.mean(sample))
    if mean_val < min_mean or mean_val > 248.0:
        return True
    std_val = float(np.std(sample))
    if std_val < min_std:
        return True
    return False


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

    is_in_center = (0.15 <= obj_cx / w_img <= 0.85) and (0.15 <= obj_cy / h_img <= 0.85)

    # 2. Deteccion de mano/piel en la periferia del objeto
    pad = int(min(w_img, h_img) * 0.08)
    roi_x1 = max(0, x1 - pad)
    roi_y1 = max(0, y1 - pad)
    roi_x2 = min(w_img, x2 + pad)
    roi_y2 = min(h_img, y2 + pad)

    roi = frame_bgr[roi_y1:roi_y2, roi_x1:roi_x2]
    hand_detected = False

    if roi.size > 0:
        if HAS_CV2 and cv2 is not None:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            lower_skin = np.array([0, 35, 45], dtype=np.uint8)
            upper_skin = np.array([28, 210, 255], dtype=np.uint8)
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            skin_ratio = np.count_nonzero(skin_mask) / float(roi.shape[0] * roi.shape[1])
        else:
            b = roi[:, :, 0].astype(int)
            g = roi[:, :, 1].astype(int)
            r = roi[:, :, 2].astype(int)
            skin_mask = (r > 95) & (g > 40) & (b > 20) & ((r - g) > 15) & (r > b)
            skin_ratio = np.count_nonzero(skin_mask) / float(roi.shape[0] * roi.shape[1])
        if skin_ratio > 0.06:
            hand_detected = True

    is_held_in_center = is_in_center and (hand_detected or center_score > 0.60)
    return is_held_in_center, center_score, hand_detected


def draw_detections(frame_bgr: np.ndarray, detections: list[dict], in_place: bool = False) -> np.ndarray:
    """
    Dibuja rectangulos delimitadores y etiquetas sobre el frame BGR con escalado dinamico
    segun resolucion y tamano minimo garantizado para que nunca queden invisibles.
    """
    if not detections:
        return frame_bgr

    h_img, w_img = frame_bgr.shape[:2]
    scale = max(1.0, min(w_img, h_img) / 420.0)
    rect_thick = max(2, int(round(1.8 * scale)))
    c_thick = max(3, int(round(2.8 * scale)))
    font_scale = max(0.50, round(0.48 * scale, 2))
    font_thick = max(1, int(round(1.2 * scale)))

    annotated = frame_bgr if in_place else frame_bgr.copy()
    for det in detections:
        info = get_waste_info(det["clase"])
        if info is None:
            continue
        color = info.get("color", (46, 204, 113))
        x1, y1, x2, y2 = det["bbox"]

        # Garantizar tamano visual minimo del recuadro (>= 36x36 px) para que nunca sea una rayita invisible
        min_box_px = max(36, int(min(w_img, h_img) * 0.08))
        if (x2 - x1) < min_box_px:
            cx = (x1 + x2) // 2
            x1 = max(2, cx - min_box_px // 2)
            x2 = min(w_img - 2, cx + min_box_px // 2)
        if (y2 - y1) < min_box_px:
            cy = (y1 + y2) // 2
            y1 = max(2, cy - min_box_px // 2)
            y2 = min(h_img - 2, cy + min_box_px // 2)

        clean_label = det["label"].encode("ascii", "ignore").decode("ascii") or det["clase"]
        star = " *" if det.get("boost_applied", False) else ""
        label_text = f"{clean_label} {det['confianza']:.0%}{star}"

        if HAS_CV2 and cv2 is not None:
            # Marco visible
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, rect_thick)

            # Esquinas destacadas estilo scanner / visor tecnologico
            box_w = x2 - x1
            box_h = y2 - y1
            c_len = max(16, min(int(32 * scale), box_w // 4, box_h // 4))
            # Top-left
            cv2.line(annotated, (x1, y1), (x1 + c_len, y1), color, c_thick)
            cv2.line(annotated, (x1, y1), (x1, y1 + c_len), color, c_thick)
            # Top-right
            cv2.line(annotated, (x2, y1), (x2 - c_len, y1), color, c_thick)
            cv2.line(annotated, (x2, y1), (x2, y1 + c_len), color, c_thick)
            # Bottom-left
            cv2.line(annotated, (x1, y2), (x1 + c_len, y2), color, c_thick)
            cv2.line(annotated, (x1, y2), (x1, y2 - c_len), color, c_thick)
            # Bottom-right
            cv2.line(annotated, (x2, y2), (x2 - c_len, y2), color, c_thick)
            cv2.line(annotated, (x2, y2), (x2, y2 - c_len), color, c_thick)

            # Etiqueta limpia con fondo solido
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
            pad_v = max(6, int(6 * scale))
            top_y = max(y1, th + pad_v + 4)
            lbl_x2 = min(w_img - 2, x1 + tw + 10)
            cv2.rectangle(annotated, (x1, top_y - th - pad_v - 2), (lbl_x2, top_y), color, -1)
            cv2.putText(annotated, label_text, (x1 + 5, top_y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)
        else:
            from PIL import Image as _PILImg, ImageDraw as _PILDraw
            pil_temp = _PILImg.fromarray(annotated[:, :, ::-1])
            draw = _PILDraw.Draw(pil_temp)
            rgb_col = (color[2], color[1], color[0]) if len(color) == 3 else color
            draw.rectangle([x1, y1, x2, y2], outline=rgb_col, width=max(2, rect_thick))
            lbl_w = len(label_text) * 7 + 8
            draw.rectangle([x1, max(0, y1 - 20), min(w_img, x1 + lbl_w), y1], fill=rgb_col)
            draw.text((x1 + 4, max(0, y1 - 18)), label_text, fill=(255, 255, 255))
            annotated = np.array(pil_temp)[:, :, ::-1].copy()

    return annotated


def _detect_bounding_boxes(
    detector: YOLO,
    frame_bgr: np.ndarray,
    conf_threshold: float = 0.28,
    filter_people: bool = True,
    feedback_store: dict | None = None,
    imgsz: int = 384,
) -> list[dict]:
    """Ejecuta detector de objetos primario con cajas delimitadoras (YOLO-World v2 / YOLOv8-det)."""
    if is_frame_blank_or_dark(frame_bgr):
        return []

    h_img, w_img = frame_bgr.shape[:2]
    frame_area = w_img * h_img

    # Umbral interno reducido para no perder objetos chicos en mano (pilas AA/AAA, marcadores, papeles, tijeras)
    query_conf = min(conf_threshold, 0.14)
    with _INFERENCE_LOCK, torch.inference_mode():
        results = detector(frame_bgr, conf=query_conf, imgsz=imgsz, verbose=False)[0]

    if getattr(results, "boxes", None) is None or len(results.boxes) == 0:
        del results
        _clear_predictor_memory(detector)
        return []

    boxes_list = [
        (int(b.cls[0]), float(b.conf[0]), tuple(map(int, b.xyxy[0].tolist())))
        for b in results.boxes
    ]
    del results
    _clear_predictor_memory(detector)

    # Paso 1: Registrar ubicaciones de rostros, anteojos, personas, pilas y utiles/botellas para desambiguar
    face_or_glasses_boxes = []
    person_boxes = []
    has_person_or_clothes = False
    bottle_or_cap_boxes = []
    marker_or_pen_boxes = []
    battery_boxes = []
    for cls_id, c_score, xyxy in boxes_list:
        c_name = detector.names[cls_id].lower()
        if any(k in c_name for k in ("face", "head", "eyeglasses", "glasses", "spectacles")) and "case" not in c_name:
            face_or_glasses_boxes.append(xyxy)
        if any(k in c_name for k in ("person", "face", "man", "woman", "head")):
            has_person_or_clothes = True
            person_boxes.append(xyxy)
        if any(k in c_name for k in ("bottle", "plastic cap", "plastic lid", "nail polish", "perfume", "cosmetic", "jar")) and c_score >= 0.16:
            bottle_or_cap_boxes.append((c_name, c_score, xyxy))
        if any(k in c_name for k in ("marker", "sharpie", "pen", "highlighter", "felt tip", "pencil", "scissors")) and c_score >= 0.16:
            marker_or_pen_boxes.append((c_name, c_score, xyxy))
        if any(k in c_name for k in ("battery", "batteries", "duracell")) and c_score >= 0.14:
            battery_boxes.append((c_name, c_score, xyxy))

    _SENSITIVE_SMALL_CLASSES = {
        "battery", "batteries", "aa battery", "aaa battery", "cylindrical battery", "alkaline battery", "duracell battery",
        "scissors", "pair of scissors", "metal scissors",
        "eyeglasses case", "glasses case", "hard case",
        "marker", "marker pen", "permanent marker", "sharpie marker", "felt tip pen", "highlighter pen",
        "pen", "plastic pen", "ballpoint pen",
        "nail polish bottle", "perfume bottle", "small glass bottle", "cosmetic bottle", "dropper bottle", "serum bottle",
        "plastic bag", "white plastic bag", "supermarket plastic bag", "grocery bag", "shopping bag",
        "printed document", "paper bill", "invoice", "brochure", "flyer", "pamphlet", "envelope",
        "sheet of paper", "notebook paper",
        "t-shirt", "shirt", "folded shirt", "folded clothes", "sweater", "hoodie", "jacket", "pants", "jeans",
        "blanket", "fleece", "fabric", "textile", "towel", "plush fabric",
        "shoes", "shoe", "sneakers", "white sneakers", "canvas shoes", "tennis shoes", "running shoes", "pair of shoes", "footwear",
        "shoebox", "black cardboard box", "small cardboard box", "packaging box", "medicine box", "cosmetic box", "plastic box",
    }

    raw_detections = []
    for cls_id, conf, bbox in boxes_list:
        cls_name = detector.names[cls_id].lower()

        # Aplicar umbral por clase (permitir pilas desde 0.15, otros objetos chicos/especificos desde 0.18, resto segun conf_threshold)
        if "battery" in cls_name or "batteries" in cls_name:
            min_allowed_conf = min(conf_threshold, 0.15)
        elif cls_name in _SENSITIVE_SMALL_CLASSES:
            min_allowed_conf = min(conf_threshold, 0.18)
        else:
            min_allowed_conf = conf_threshold
        if conf < min_allowed_conf:
            continue

        # Filtro inmediato de personas, rostros, anteojos (excepto 'glasses case'), muebles, ventiladores, espejos o paredes
        if filter_people and (
            cls_name in IGNORED_CLASSES
            or (
                "case" not in cls_name
                and any(ign in cls_name for ign in (
                    "person", "face", "man", "woman", "head", "glasses", "hand",
                    "wardrobe", "closet", "cabinet", "door", "shelf", "furniture", "table", "desk",
                    "ceiling fan", "fan", "mirror", "picture frame", "wooden frame", "window", "wall", "crate", "basket", "grass"
                ))
            )
        ):
            continue

        # Si es una prenda de ropa o calzado y hay una persona/rostro en escena (selfie frente a camara),
        # descartarla para no marcar la ropa puesta ni falsos positivos de calzado cuando se sostiene una tijera/objeto en alto.
        # Si NO hay persona en la foto (ej. remera o zapatillas apoyadas en el piso/mesa), permitirlas!
        if filter_people and (
            cls_name in IGNORORED_CLOTHING_WHEN_PERSON
            or cls_name in ("shoes", "shoe", "sneakers", "white sneakers", "canvas shoes", "tennis shoes", "running shoes", "pair of shoes", "footwear")
        ):
            if has_person_or_clothes or len(face_or_glasses_boxes) > 0:
                continue

        # Mapeo estricto de celulares (solo sinonimos reales de telefono)
        if cls_name in ("smartphone", "smartphone touchscreen", "mobile phone", "telephone"):
            cls_name = "cell phone"

        # Mapeo a categoria estandar de Plastico para evitar crear clases adicionales
        if cls_name in ("ruler", "plastic ruler"):
            cls_name = "plastic"

        x1, y1, x2, y2 = bbox
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        area_ratio = (bw * bh) / float(frame_area)

        # Desambiguacion para vasos, potes y latas/aerosoles:
        # Un pote de crema blanco y ancho (ej. Crema de Ordene) suele confundirse con 'aerosol'/'can'/'cup'.
        if cls_name in ("cup", "disposable cup", "plastic cup"):
            if bw >= bh * 0.82:
                cls_name = "plastic jar"
        elif cls_name in ("spray can", "aerosol can", "aerosol", "air freshener spray", "deodorant spray", "can", "tin can", "aluminum can", "soda can", "beer can", "beverage can", "drink can"):
            # 1) Si el objeto es mas ancho o casi igual de ancho que alto (bw >= bh * 0.78), NO es un aerosol ni lata de bebida: es un Pote / Tarro Plastico!
            if bw >= bh * 0.78:
                cls_name = "plastic jar"
            else:
                # 2) Si detecta 'can'/'aluminum can' sobre la etiqueta de una botella plastica:
                #    verificar si hay una tapa plastica arriba/cerca o una botella superpuesta, o si encima/debajo de la etiqueta hay plastico transparente
                is_bottle_label = False
                for b_name, _, b_xyxy in bottle_or_cap_boxes:
                    bx1, by1, bx2, by2 = b_xyxy
                    # Si hay una tapa plastica ("plastic cap"/"plastic lid") alineada horizontalmente por encima del supuesto 'can'
                    if "cap" in b_name or "lid" in b_name:
                        cap_cx = (bx1 + bx2) / 2.0
                        if (x1 - bw * 0.4) <= cap_cx <= (x2 + bw * 0.4) and by2 <= (y1 + bh * 0.35):
                            is_bottle_label = True
                            break
                    elif "bottle" in b_name:
                        ix1, iy1 = max(x1, bx1), max(y1, by1)
                        ix2, iy2 = min(x2, bx2), min(y2, by2)
                        if ix2 > ix1 and iy2 > iy1:
                            is_bottle_label = True
                            break
                # 3) Ademas, en botellas PET (en mano o sobre pasto), la caja de 'can' suele encerrar la mitad inferior o la etiqueta,
                #    pero justo arriba ( cuello/hombro de la botella ) o abajo ( base ) continua el envase.
                if is_bottle_label or (cls_name in ("can", "aluminum can", "soda can", "drink can", "beverage can", "tin can") and conf < 0.65 and bh >= bw * 1.15):
                    # Verificar si tiene proporcion de botella o tapa asociada
                    if is_bottle_label or bh >= bw * 1.35:
                        cls_name = "plastic bottle"

        is_held_in_center, center_score, hand_detected = analyze_spatial_and_hand_context(frame_bgr, bbox)

        # Desambiguacion de Pilas Cilindricas (AA / AAA / Duracell / Energizer) confundidas con Botella, Lata, Termo o Vaper:
        # En primeros planos o fotos de producto (ej. pila vertical grande con borne '+' saliente arriba), YOLO-World suele
        # disparar 'bottle'/'thermos'/'can' con ~40%-50% por su silueta cilindrica y tapa/borne superior, pisando la deteccion de 'battery'.
        if cls_name in (
            "bottle", "plastic bottle", "clear plastic bottle", "water bottle", "soda bottle", "juice bottle", "pet bottle",
            "thermos", "can", "aluminum can", "tin can", "soda can", "beverage can", "drink can", "spray can", "aerosol can",
            "disposable vape", "electronic cigarette"
        ):
            overlaps_battery = False
            best_bat_score = 0.0
            for b_cls, b_score, b_xyxy in battery_boxes:
                ix1, iy1 = max(x1, b_xyxy[0]), max(y1, b_xyxy[1])
                ix2, iy2 = min(x2, b_xyxy[2]), min(y2, b_xyxy[3])
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    if inter / float(max(1, bw * bh)) > 0.25 and b_score >= 0.14:
                        overlaps_battery = True
                        best_bat_score = max(best_bat_score, b_score)
            # Ademas, si es un cilindro esbelto (aspect >= 2.6) con patron bicolor tipico de pila alcalina (cobre/dorado + cuerpo negro/oscuro)
            is_bicolor_battery = False
            aspect_cyl = max(bw, bh) / float(max(1, min(bw, bh)))
            if not overlaps_battery and aspect_cyl >= 2.6 and conf < 0.58 and HAS_CV2 and cv2 is not None:
                roi_bat = frame_bgr[max(0, y1):min(h_img, y2), max(0, x1):min(w_img, x2)]
                if roi_bat.size > 0:
                    hsv_bat = cv2.cvtColor(roi_bat, cv2.COLOR_BGR2HSV)
                    # Cobre/dorado (H: 8..26, S: 65..255, V: 70..255) y negro/oscuro (V < 65)
                    copper_mask = cv2.inRange(hsv_bat, np.array([8, 65, 70], dtype=np.uint8), np.array([26, 255, 255], dtype=np.uint8))
                    dark_mask = hsv_bat[:, :, 2] < 65
                    px_total = float(roi_bat.shape[0] * roi_bat.shape[1])
                    copper_ratio = np.count_nonzero(copper_mask) / px_total
                    dark_ratio = np.count_nonzero(dark_mask) / px_total
                    if copper_ratio >= 0.12 and dark_ratio >= 0.20:
                        is_bicolor_battery = True
            if overlaps_battery or is_bicolor_battery:
                cls_name = "battery"
                conf = max(conf, best_bat_score)

        # Desambiguacion estricta Anti-Sesgo Vaper vs Cajas, Marcadores y Envases Cosmeticos:
        # YOLO-World tiende a disparar 'vape' con 35%-55% en cajas oscuras (box mod), marcadores (Sharpie) o frascos con tapa larga (quitaesmalte/esmalte).
        if "vape" in cls_name or "cigarette" in cls_name:
            # 1) Si el objeto es grande (area_ratio > 0.22, como una caja de zapatillas o envase grande en primer plano), NUNCA es un vaper
            if area_ratio > 0.22:
                cls_name = "cardboard box"
            else:
                overlaps_marker = False
                for m_name, m_score, m_xyxy in marker_or_pen_boxes:
                    ix1, iy1 = max(x1, m_xyxy[0]), max(y1, m_xyxy[1])
                    ix2, iy2 = min(x2, m_xyxy[2]), min(y2, m_xyxy[3])
                    if ix2 > ix1 and iy2 > iy1 and m_score >= conf * 0.35:
                        overlaps_marker = True
                        break
                overlaps_cosmetic = False
                matched_cosmetic_cls = "nail polish bottle"
                for b_name, b_score, b_xyxy in bottle_or_cap_boxes:
                    ix1, iy1 = max(x1, b_xyxy[0]), max(y1, b_xyxy[1])
                    ix2, iy2 = min(x2, b_xyxy[2]), min(y2, b_xyxy[3])
                    if ix2 > ix1 and iy2 > iy1 and b_score >= conf * 0.35:
                        overlaps_cosmetic = True
                        if any(k in b_name for k in ("nail polish", "perfume", "small glass", "dropper", "serum", "cosmetic")):
                            matched_cosmetic_cls = b_name
                        break
                if overlaps_marker:
                    cls_name = "marker"
                elif overlaps_cosmetic:
                    cls_name = matched_cosmetic_cls
                elif conf < 0.65:
                    # Exigir alta certeza (>= 65%) para diagnosticar Vaper; si es menor, reclasificar segun su geometria real:
                    aspect_obj = max(bw, bh) / float(max(1, min(bw, bh)))
                    if aspect_obj >= 2.5:
                        cls_name = "marker"
                    elif area_ratio >= 0.14 or bw >= bh * 0.85:
                        cls_name = "cardboard box"
                    else:
                        cls_name = "nail polish bottle"

        # Desambiguacion de Cajas grandes / Cajas de zapatillas en primer plano vs Laptop:
        # Cuando se enfoca muy de cerca una caja negra/oscura (area_ratio > 0.35), YOLO-World suele llamarla 'laptop' con baja confianza (< 0.55).
        if cls_name == "laptop":
            # Si cubre gran parte del encuadre sin teclado o con baja confianza, verificar si es un E-Reader/Tablet o una Caja de Carton
            if conf < 0.42 and area_ratio > 0.35:
                cls_name = "cardboard box"
            elif 0.42 <= conf < 0.58 and bh > bw * 1.05:
                # Si es vertical/rectangular tipo Kindle/Tablet sobre mesa
                cls_name = "tablet"

        # Filtro anti-falsos positivos de 'battery' sobre cajas anchas de carton en la mano:
        # Solo reclasificar a 'cardboard box' si ocupa > 16% del frame, tiene baja confianza y NO tiene forma cilindrica alargada (aspect < 2.1).
        if cls_name in ("battery", "batteries", "aa battery", "aaa battery", "cylindrical battery", "alkaline battery", "duracell battery"):
            aspect_bat = max(bw, bh) / float(max(1, min(bw, bh)))
            if area_ratio > 0.16 and conf < 0.60 and aspect_bat < 2.1:
                cls_name = "cardboard box"

        # Filtro inteligente para carton y papel:
        # - Evita que diminutas etiquetas impresas sobre pilas, bolsas plasticas o envases (< 3.5% de la imagen) se clasifiquen como 'paper'.
        # - Permite facturas, boletas y folletos reales sobre una mesa (printed document / invoice / brochure / paper) cuando ocupan un area razonable (0.04 <= area_ratio <= 0.65).
        if cls_name in (
            "cardboard", "carton", "cardboard box", "shoebox", "black cardboard box",
            "small cardboard box", "packaging box", "medicine box", "cosmetic box",
            "corrugated cardboard", "flat cardboard",
            "paper", "sheet of paper", "notebook paper", "printed document", "paper bill",
            "invoice", "brochure", "flyer", "pamphlet", "envelope", "newspaper", "magazine"
        ):
            # 1) Bloquear micro-etiquetas / logos impresos sobre pilas, bolsas o botellas
            if cls_name == "paper" and area_ratio < 0.045:
                continue
            # 2) Si se solapa con la cara/cabeza del usuario y es grande (espejo/pared detras de la persona), descartar
            overlaps_user_head = False
            for fb in face_or_glasses_boxes:
                ix1, iy1 = max(x1, fb[0]), max(y1, fb[1])
                ix2, iy2 = min(x2, fb[2]), min(y2, fb[3])
                if ix2 > ix1 and iy2 > iy1:
                    overlaps_user_head = True
                    break
            if overlaps_user_head and area_ratio > 0.12:
                continue
            # 3) Si hay una persona en escena (selfie frente a camara) y no lo sostiene en la mano, exigir confianza alta para no marcar puertas/paredes
            if has_person_or_clothes and not hand_detected and conf < 0.62:
                continue
            # 4) Si no hay persona en escena (foto directa a la mesa o al objeto), permitir documentos/facturas/cajas desde 0.22 pero bloquear fondos que cubran toda la foto (> 80%)
            if not has_person_or_clothes and not hand_detected:
                if area_ratio > 0.80 and conf < 0.50:
                    continue
                if cls_name in ("cardboard", "paper") and area_ratio < 0.06 and conf < 0.38:
                    continue

        # Reglas anti-sesgo estrictas para 'cell phone':
        # Evita que caras con anteojos (44%), encendedores (56%) o vapers/potes (64%) se marquen como Celular
        if cls_name == "cell phone":
            if conf < 0.68:
                continue
            aspect = max(bw / float(bh), bh / float(bw))
            if aspect > 2.6 or area_ratio < 0.025:
                continue
            # Si la caja cae sobre un rostro o anteojos detectados, descartar
            overlaps_face = False
            for fb in face_or_glasses_boxes:
                ix1, iy1 = max(x1, fb[0]), max(y1, fb[1])
                ix2, iy2 = min(x2, fb[2]), min(y2, fb[3])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                if inter / float(bw * bh) > 0.20:
                    overlaps_face = True
                    break
            if overlaps_face:
                continue
            # Si el interior de la supuesta caja de celular contiene mucha piel (es una cara o mano), descartar
            if HAS_CV2 and cv2 is not None:
                roi_cell = frame_bgr[max(0, y1):min(h_img, y2), max(0, x1):min(w_img, x2)]
                if roi_cell.size > 0:
                    hsv_cell = cv2.cvtColor(roi_cell, cv2.COLOR_BGR2HSV)
                    skin_c = cv2.inRange(hsv_cell, np.array([0, 30, 40], dtype=np.uint8), np.array([28, 215, 255], dtype=np.uint8))
                    if (np.count_nonzero(skin_c) / float(roi_cell.shape[0] * roi_cell.shape[1])) > 0.22:
                        continue

        # Descartar detecciones que cubran literalmente toda la pantalla (fondo/pared > 90% ancho y alto)
        if (bw > 0.90 * w_img and bh > 0.90 * h_img) or (area_ratio > 0.85 and conf < 0.40):
            continue

        # Obtener informacion del residuo (si no es un residuo valido o es clase ignorada, se descarta)
        info = get_waste_info(cls_name)
        if info is None:
            continue

        effective_conf, boost_applied = calculate_boosted_confidence(
            cls_name=cls_name,
            raw_conf=conf,
            predicted_cls=cls_name,
            is_held_in_center=is_held_in_center,
            store=feedback_store
        )

        raw_detections.append({
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
            "cascade_fallback":  False,
        })

    if not raw_detections:
        return []

    # Non-Maximum Suppression (NMS): priorizar pilas y clases especificas sobre botellas/latas genericas y celular
    raw_detections.sort(
        key=lambda d: (
            2 if "battery" in d["clase"] or "batteries" in d["clase"]
            else (0 if d["clase"] == "cell phone" and d["confianza"] < 0.78 else 1),
            d["confianza"]
        ),
        reverse=True
    )
    
    def compute_iou(box1, box2):
        bx1 = max(box1[0], box2[0])
        by1 = max(box1[1], box2[1])
        bx2 = min(box1[2], box2[2])
        by2 = min(box1[3], box2[3])
        inter_area = max(0, bx2 - bx1) * max(0, by2 - by1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        denom = float(area1 + area2 - inter_area)
        return (inter_area / denom) if denom > 0 else 0.0

    detections = []
    for d in raw_detections:
        overlap = False
        for f in detections:
            if compute_iou(d["bbox"], f["bbox"]) > 0.40:
                overlap = True
                break
        if not overlap:
            detections.append(d)

    return detections


def _classify_focal_center(
    classifier: YOLO,
    frame_bgr: np.ndarray,
    conf_threshold: float = 0.28,
    feedback_store: dict | None = None,
) -> list[dict]:
    """Ejecuta clasificador secundario focalizado en el centro de la escena (YOLOv8s-cls) en una sola pasada."""
    if is_frame_blank_or_dark(frame_bgr):
        return []

    h_img, w_img = frame_bgr.shape[:2]

    # Recorte focalizado en el objeto central (evita procesar el fondo completo en HSV/CNN)
    view_w = int(w_img * 0.44)
    view_h = int(h_img * 0.48)
    x1 = (w_img - view_w) // 2
    y1 = (h_img - view_h) // 2
    x2 = x1 + view_w
    y2 = y1 + view_h
    bbox = (x1, y1, x2, y2)

    center_crop = frame_bgr[y1:y2, x1:x2]
    if center_crop.size == 0 or is_frame_blank_or_dark(center_crop, min_mean=28.0, min_std=8.0):
        return []

    skin_ratio = 0.0
    crop_area = float(max(1, center_crop.shape[0] * center_crop.shape[1]))

    if HAS_CV2 and cv2 is not None:
        hsv_crop = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
        crop_skin = cv2.inRange(hsv_crop, np.array([0, 30, 40], dtype=np.uint8), np.array([28, 215, 255], dtype=np.uint8))
        skin_ratio = np.count_nonzero(crop_skin) / crop_area

        # Ajustar el recuadro al contorno del objeto en primer plano (no piel) dentro de la zona central
        gray_crop = cv2.cvtColor(center_crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray_crop, (5, 5), 0), 35, 110)
        edges[crop_skin > 0] = 0
        coords = cv2.findNonZero(edges)
        del hsv_crop, crop_skin, gray_crop, edges
        if coords is not None and len(coords) > 40:
            rx, ry, rw, rh = cv2.boundingRect(coords)
            if rw > view_w * 0.22 and rh > view_h * 0.22:
                pad_x, pad_y = int(rw * 0.10), int(rh * 0.10)
                bx1 = max(x1, x1 + rx - pad_x)
                by1 = max(y1, y1 + ry - pad_y)
                bx2 = min(x2, x1 + rx + rw + pad_x)
                by2 = min(y2, y1 + ry + rh + pad_y)
                bbox = (bx1, by1, bx2, by2)
                center_crop = frame_bgr[by1:by2, bx1:bx2]

    with _INFERENCE_LOCK, torch.inference_mode():
        crop_res = classifier(center_crop, imgsz=224, verbose=False)[0]

    if getattr(crop_res, "probs", None) is None:
        del crop_res
        _clear_predictor_memory(classifier)
        return []

    p_combined = crop_res.probs.data.cpu().numpy().copy()
    del crop_res
    _clear_predictor_memory(classifier)

    # Analizar si hay una mano sosteniendo el objeto focal en el centro
    _, _, hand_detected = analyze_spatial_and_hand_context(frame_bgr, bbox)

    # Bloqueo estricto de ropa puesta o calzado falso cuando hay piel humana / mano / rostro en escena:
    # Un usuario frente a la camara o sosteniendo un objeto pequeno en la mano NUNCA debe ser diagnosticado
    # como 'shoes' ni 'clothes' por el clasificador de respaldo (que confunde dedos + objetos o ventiladores con calzado).
    # En cambio, si NO hay piel en el recorte (skin_ratio <= 0.03, ej. zapatillas o remera apoyadas en el piso/mesa), se permite!
    for idx_c, name_c in classifier.names.items():
        c_low = name_c.lower()
        if c_low in ("clothes", "shoes"):
            if skin_ratio > 0.03 or hand_detected:
                p_combined[int(idx_c)] = 0.0

    top1_id = int(np.argmax(p_combined))
    top1_conf = float(p_combined[top1_id])
    raw_top1_cls = classifier.names[top1_id].lower()

    # Si la clase dominante es ropa/calzado y hay piel/mano o confianza baja, descartar
    if raw_top1_cls in ("clothes", "shoes") and (skin_ratio > 0.03 or hand_detected or top1_conf < 0.46):
        return []
    # Si es papel/carton sin mano sosteniendolo, exigir confianza >= 0.50 para evitar fondos lisos
    if raw_top1_cls in ("paper", "cardboard") and not hand_detected and top1_conf < 0.50:
        return []

    best_cls = raw_top1_cls
    best_raw_conf = top1_conf
    best_eff_conf, best_boost = calculate_boosted_confidence(
        cls_name=raw_top1_cls,
        raw_conf=top1_conf,
        predicted_cls=raw_top1_cls,
        is_held_in_center=True,
        store=feedback_store
    )

    top5_ids = np.argsort(p_combined)[::-1][:5]
    for cid in top5_ids:
        c_name = classifier.names[int(cid)].lower()
        c_raw = float(p_combined[int(cid)])
        if c_raw <= 0.0:
            continue
        c_eff, c_boost = calculate_boosted_confidence(
            cls_name=c_name,
            raw_conf=c_raw,
            predicted_cls=raw_top1_cls,
            is_held_in_center=True,
            store=feedback_store
        )
        if c_eff > best_eff_conf:
            best_cls = c_name
            best_raw_conf = c_raw
            best_eff_conf = c_eff
            best_boost = c_boost

    min_cls_conf = max(conf_threshold, 0.48)
    if best_eff_conf >= min_cls_conf:
        info = get_waste_info(best_cls)
        if info is not None:
            return [{
                "clase":             best_cls,
                "label":             info["label"],
                "emoji":             info.get("emoji", "📦"),
                "tipo":              info["tipo"],
                "accion":            info["accion"],
                "es_especial":       info.get("es_especial", False),
                "confianza":         best_eff_conf,
                "confianza_raw":     best_raw_conf,
                "boost_applied":     best_boost,
                "is_held_in_center": True,
                "center_score":      1.0,
                "hand_detected":     (skin_ratio > 0.04),
                "bbox":              bbox,
                "cascade_fallback":  True,
            }]
    return []


def run_inference(
    model: YOLO | dict,
    frame_bgr: np.ndarray,
    conf_threshold: float = 0.28,
    filter_people: bool = True,
    feedback_store: dict | None = None,
    imgsz: int = 384,
    draw: bool = True,
    enable_cascade_fallback: bool = True,
    in_place_draw: bool = False,
) -> tuple[np.ndarray, list[dict]]:
    """
    Ejecuta el pipeline de vision inteligente:
    - Modo Cascada Hibrido (dict):
        Paso 1: Deteccion primaria con recuadros (YOLO-World v2).
        Paso 2: Si el detector no detecta ningun residuo valido y enable_cascade_fallback=True,
                carga bajo demanda y conmuta automaticamente al clasificador focalizado (YOLOv8s-cls).
    - Modo Simple (YOLO): Ejecuta el modelo individual proporcionado.
    """
    if frame_bgr is None or frame_bgr.size == 0 or is_frame_blank_or_dark(frame_bgr):
        return frame_bgr, []

    if isinstance(model, dict):
        detector = model.get("detector")
        classifier = model.get("classifier")
    else:
        if getattr(model, "task", None) == "classify":
            detector = None
            classifier = model
        else:
            detector = model
            classifier = None

    detections = []

    # ── PASO 1: Detector Primario con Bounding Boxes ──────────────────────────
    if detector is not None:
        detections = _detect_bounding_boxes(
            detector=detector,
            frame_bgr=frame_bgr,
            conf_threshold=conf_threshold,
            filter_people=filter_people,
            feedback_store=feedback_store,
            imgsz=imgsz,
        )

    # ── PASO 2: Respaldo / Cascada Automatica al Clasificador Focalizado ───────
    # Si el detector no encontro residuos validos y esta habilitado el fallback, evaluamos la zona central
    if not detections and enable_cascade_fallback:
        if classifier is None and isinstance(model, dict):
            classifier = _get_lazy_classifier(model)
        if classifier is not None:
            detections = _classify_focal_center(
                classifier=classifier,
                frame_bgr=frame_bgr,
                conf_threshold=conf_threshold,
                feedback_store=feedback_store,
            )

    annotated = draw_detections(frame_bgr, detections, in_place=in_place_draw) if draw else frame_bgr
    return annotated, detections
