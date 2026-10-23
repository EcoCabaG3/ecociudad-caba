"""
utils_vision.py - Deteccion, analisis espacial y clasificacion de residuos EcoCiudad CABA
Optimizacion de memoria y recursos para Streamlit Cloud (inferencia liviana en CPU).
"""
import os
import math
import gc
import numpy as np
import torch

# Optimizacion estricta de memoria para Streamlit Cloud (< 1 GB RAM)
torch.set_num_threads(1)
torch.set_grad_enabled(False)

import streamlit as st
from ultralytics import YOLO
from feedback_store import calculate_boosted_confidence

try:
    import cv2
    HAS_CV2 = True
except Exception:
    cv2 = None
    HAS_CV2 = False

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
    
    # Plasticos, Envases y Potes (Contenedor Verde)
    "plastic":              {"label": "Plastico / Envases",   "emoji": "🧴", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Vaciar, enjuagar y compactar.", "es_especial": False},
    "plastic bottle":       {"label": "Plastico / Envases",   "emoji": "🧴", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Vaciar, enjuagar y compactar.", "es_especial": False},
    "pet bottle":           {"label": "Plastico / Envases",   "emoji": "🧴", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Vaciar, enjuagar y compactar.", "es_especial": False},
    "bottle":               {"label": "Botella / Plastico",   "emoji": "🧴", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Vaciar, enjuagar y aplastar.", "es_especial": False},
    "thermos":              {"label": "Termo / Botella",      "emoji": "🧴", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Limpio y seco al Contenedor Verde.", "es_especial": False},
    "cup":                  {"label": "Vaso (Vidrio / Plastico)", "emoji": "🥤", "tipo": "Reciclable Seco (Verde)", "color": (52, 152, 219),  "accion": "Asegurar que este limpio y seco. Apto Contenedor Verde.", "es_especial": False},
    "disposable cup":       {"label": "Vaso Descartable",     "emoji": "🥤", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Asegurar que este limpio y sin restos liquidos.", "es_especial": False},
    "plastic cup":          {"label": "Vaso Descartable",     "emoji": "🥤", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Asegurar que este limpio y sin restos liquidos.", "es_especial": False},
    "bowl":                 {"label": "Envase Descartable",   "emoji": "🥣", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Asegurar que este limpio y sin restos organicos.", "es_especial": False},
    "jar":                  {"label": "Pote / Frasco Plastico","emoji": "🧴", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Limpio y seco. Apto Contenedor Verde.", "es_especial": False},
    "plastic jar":          {"label": "Pote / Envase Plastico","emoji": "🧴", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Limpio y seco. Apto Contenedor Verde.", "es_especial": False},
    "pot":                  {"label": "Pote / Envase Plastico","emoji": "🧴", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Limpio y seco. Apto Contenedor Verde.", "es_especial": False},
    "plastic container":    {"label": "Envase / Contenedor",  "emoji": "🧴", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Vaciar, enjuagar y compactar.", "es_especial": False},
    "container":            {"label": "Envase / Contenedor",  "emoji": "🧴", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Limpio y seco. Al Contenedor Verde.", "es_especial": False},
    "capsule":              {"label": "Capsula de Cafe",      "emoji": "☕", "tipo": "Reciclable / Punto Verde",  "color": (46, 204, 113),  "accion": "Vacia y limpia. En Punto Verde se convierte en madera plastica.", "es_especial": False},

    # Utiles de Escritura y Plasticos Rigidos (Contenedor Verde)
    "pen":                  {"label": "Lapicera / Plastico",   "emoji": "🖊️", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Plastico rigido. Retirar mina de tinta si es posible.", "es_especial": False},
    "plastic pen":          {"label": "Lapicera / Plastico",   "emoji": "🖊️", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Plastico rigido. Retirar mina de tinta si es posible.", "es_especial": False},
    "ballpoint pen":        {"label": "Lapicera / Plastico",   "emoji": "🖊️", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Plastico rigido. Retirar mina de tinta si es posible.", "es_especial": False},
    "marker":               {"label": "Marcador / Plastico",   "emoji": "🖊️", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Plastico rigido reciclable.", "es_especial": False},
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
    "can":                  {"label": "Lata / Aluminio",      "emoji": "🥫", "tipo": "Reciclable Seco (Verde)",  "color": (149, 165, 166), "accion": "Limpio y seco. Aplastar si es posible.", "es_especial": False},
    "aluminum can":         {"label": "Lata / Aluminio",      "emoji": "🥫", "tipo": "Reciclable Seco (Verde)",  "color": (149, 165, 166), "accion": "Limpio y seco. Aplastar si es posible.", "es_especial": False},
    "tin can":              {"label": "Lata / Aluminio",      "emoji": "🥫", "tipo": "Reciclable Seco (Verde)",  "color": (149, 165, 166), "accion": "Limpio y seco. Aplastar si es posible.", "es_especial": False},
    "steel can":            {"label": "Lata / Aluminio",      "emoji": "🥫", "tipo": "Reciclable Seco (Verde)",  "color": (149, 165, 166), "accion": "Limpio y seco. Aplastar si es posible.", "es_especial": False},
    
    # RESIDUOS ESPECIALES GCBA (OBLIGATORIO PUNTO VERDE)
    "vape":                 {"label": "Vaper / Cigarrillo Electronico", "emoji": "🔋", "tipo": "Residuo Especial (Punto Verde)", "color": (231, 76, 60), "accion": "Contiene bateria de litio y circuitos. Llevar a Punto Verde.", "es_especial": True},
    "vape pen":             {"label": "Vaper / Cigarrillo Electronico", "emoji": "🔋", "tipo": "Residuo Especial (Punto Verde)", "color": (231, 76, 60), "accion": "Contiene bateria de litio y circuitos. Llevar a Punto Verde.", "es_especial": True},
    "electronic cigarette": {"label": "Cigarrillo Electronico / Vaper", "emoji": "🔋", "tipo": "Residuo Especial (Punto Verde)", "color": (231, 76, 60), "accion": "Llevar a Punto Verde Movil o Fijo.", "es_especial": True},
    "e-cigarette":          {"label": "Cigarrillo Electronico / Vaper", "emoji": "🔋", "tipo": "Residuo Especial (Punto Verde)", "color": (231, 76, 60), "accion": "Llevar a Punto Verde Movil o Fijo.", "es_especial": True},
    "e-waste":              {"label": "RAEE / Electronico",   "emoji": "🔋", "tipo": "Residuo Especial (Punto Verde)", "color": (231, 76, 60), "accion": "Llevar a Punto Verde Movil o Fijo (hasta 10 aparatos por persona).", "es_especial": True},
    "e_waste":              {"label": "RAEE / Electronico",   "emoji": "🔋", "tipo": "Residuo Especial (Punto Verde)", "color": (231, 76, 60), "accion": "Llevar a Punto Verde Movil o Fijo (hasta 10 aparatos por persona).", "es_especial": True},
    "electronic":           {"label": "RAEE / Electronico",   "emoji": "🔋", "tipo": "Residuo Especial (Punto Verde)", "color": (231, 76, 60), "accion": "Llevar a Punto Verde Movil o Fijo (hasta 10 aparatos por persona).", "es_especial": True},
    "electronics":          {"label": "RAEE / Electronico",   "emoji": "🔋", "tipo": "Residuo Especial (Punto Verde)", "color": (231, 76, 60), "accion": "Llevar a Punto Verde Movil o Fijo (hasta 10 aparatos por persona).", "es_especial": True},
    "circuit":              {"label": "RAEE / Electronico",   "emoji": "🔋", "tipo": "Residuo Especial (Punto Verde)", "color": (231, 76, 60), "accion": "Llevar a Punto Verde Movil o Fijo (hasta 10 aparatos por persona).", "es_especial": True},
    "battery":              {"label": "Pila / Bateria",       "emoji": "🔋", "tipo": "Residuo Especial (Punto Verde)", "color": (231, 76, 60), "accion": "Llevar a buzones de pilas en Puntos Verdes o Farmacias habilitadas.", "es_especial": True},
    "aa battery":           {"label": "Pila / Bateria",       "emoji": "🔋", "tipo": "Residuo Especial (Punto Verde)", "color": (231, 76, 60), "accion": "Llevar a buzones de pilas en Puntos Verdes o Farmacias habilitadas.", "es_especial": True},
    "batteries":            {"label": "Pila / Bateria",       "emoji": "🔋", "tipo": "Residuo Especial (Punto Verde)", "color": (231, 76, 60), "accion": "Llevar a buzones de pilas en Puntos Verdes o Farmacias habilitadas.", "es_especial": True},
    "alkaline battery":     {"label": "Pila / Bateria",       "emoji": "🔋", "tipo": "Residuo Especial (Punto Verde)", "color": (231, 76, 60), "accion": "Llevar a buzones de pilas en Puntos Verdes o Farmacias habilitadas.", "es_especial": True},
    "small battery":        {"label": "Pila / Bateria",       "emoji": "🔋", "tipo": "Residuo Especial (Punto Verde)", "color": (231, 76, 60), "accion": "Llevar a buzones de pilas en Puntos Verdes o Farmacias habilitadas.", "es_especial": True},
    "pila":                 {"label": "Pila / Bateria",       "emoji": "🔋", "tipo": "Residuo Especial (Punto Verde)", "color": (231, 76, 60), "accion": "Llevar a buzones de pilas en Puntos Verdes o Farmacias habilitadas.", "es_especial": True},
    
    # Dispositivos moviles (RAEE)
    "cell phone":           {"label": "Celular / RAEE",       "emoji": "📱", "tipo": "Residuo Especial (Punto Verde)", "color": (231, 76, 60), "accion": "Llevar a Punto Verde. Contiene metales pesados y componentes recuperables.", "es_especial": True},
    "smartphone":           {"label": "Celular / RAEE",       "emoji": "📱", "tipo": "Residuo Especial (Punto Verde)", "color": (231, 76, 60), "accion": "Llevar a Punto Verde. Contiene metales pesados y componentes recuperables.", "es_especial": True},
    "mobile phone":         {"label": "Celular / RAEE",       "emoji": "📱", "tipo": "Residuo Especial (Punto Verde)", "color": (231, 76, 60), "accion": "Llevar a Punto Verde. Contiene metales pesados y componentes recuperables.", "es_especial": True},
    "telephone":            {"label": "Celular / RAEE",       "emoji": "📱", "tipo": "Residuo Especial (Punto Verde)", "color": (231, 76, 60), "accion": "Llevar a Punto Verde. Contiene metales pesados y componentes recuperables.", "es_especial": True},
    
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

    # Envases cosmeticos / cremas / lociones
    "lotion bottle":        {"label": "Plastico / Envases",   "emoji": "🧴", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Vaciar, enjuagar y compactar.", "es_especial": False},
    "cream bottle":         {"label": "Plastico / Envases",   "emoji": "🧴", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Vaciar, enjuagar y compactar.", "es_especial": False},
    "shampoo bottle":       {"label": "Plastico / Envases",   "emoji": "🧴", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Vaciar, enjuagar y compactar.", "es_especial": False},
    "cosmetic bottle":      {"label": "Plastico / Envases",   "emoji": "🧴", "tipo": "Reciclable Seco (Verde)",  "color": (46, 204, 113),  "accion": "Vaciar, enjuagar y compactar.", "es_especial": False},

    # Organicos y Sanitario
    "organic":              {"label": "Residuo Organico",     "emoji": "🍂", "tipo": "Compost / Bolsa Negra",     "color": (139, 69, 19),   "accion": "Ideal para compostera domiciliaria o recepcion organica de Punto Verde.", "es_especial": False},
    "biological":           {"label": "Residuo Organico / Biologico", "emoji": "🍂", "tipo": "Compost / Bolsa Negra", "color": (139, 69, 19), "accion": "Ideal para compostera domiciliaria o recepcion organica de Punto Verde.", "es_especial": False},
    "medical":              {"label": "Residuo Sanitario",    "emoji": "⚠️", "tipo": "Residuo Peligroso",        "color": (192, 57, 43),   "accion": "Descartar en bolsa cerrada especial o punto farmaceutico.", "es_especial": True},

    # Textiles y Calzado
    "clothes":              {"label": "Ropa / Textil",        "emoji": "👕", "tipo": "Donación / Residuo Especial", "color": (155, 89, 182), "accion": "En buen estado donar a ONGs o ferias. Ropa rota a Punto Verde o bolsa negra.", "es_especial": False},
    "shoes":                {"label": "Calzado / Zapatos",    "emoji": "👟", "tipo": "Donación / Basura Comun",     "color": (155, 89, 182), "accion": "En buen estado donar a ferias o instituciones. Calzado roto a bolsa negra.", "es_especial": False},
    "shoe":                 {"label": "Calzado / Zapatos",    "emoji": "👟", "tipo": "Donación / Basura Comun",     "color": (155, 89, 182), "accion": "En buen estado donar a ferias o instituciones. Calzado roto a bolsa negra.", "es_especial": False},

    # Basura General
    "trash":                {"label": "Basura General",       "emoji": "🗑️", "tipo": "Basura Comun (Bolsa Negra)",  "color": (127, 140, 141), "accion": "Depositar en bolsa negra de residuos comunes no reciclables.", "es_especial": False},
}

# Mapa de subcadenas keyword → clave base en WASTE_MAP (fallback por substring)
_KEYWORD_FALLBACK = [
    ("vape",           "vape"),
    ("drinking glass", "drinking glass"),
    ("glass cup",      "glass cup"),
    ("glass",          "glass"),
    ("pen",            "pen"),
    ("marker",         "marker"),
    ("pencil",         "pencil"),
    ("thermos",        "thermos"),
    ("pot",            "jar"),
    ("jar",            "jar"),
    ("container",      "container"),
    ("cup",            "cup"),
    ("bottle",         "bottle"),
    ("lotion",         "lotion bottle"),
    ("cream",          "cream bottle"),
    ("shampoo",        "shampoo bottle"),
    ("can",            "can"),
    ("tin",            "tin can"),
    ("steel",          "steel can"),
    ("metal",          "metal"),
    ("plastic",        "plastic"),
    ("cardboard",      "cardboard"),
    ("carton",         "cardboard"),
    ("corrugated",     "cardboard"),
    ("paper",          "paper"),
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
    ("clothes",        "clothes"),
    ("shoes",          "shoes"),
    ("shoe",           "shoes"),
    ("trash",          "trash"),
    ("oil",            "oil"),
    ("aceite",         "oil"),
]

# Clases que NUNCA deben marcarse como residuo (Personas, partes del cuerpo, mobiliario y fondo)
IGNORED_CLASSES = {
    "person", "human face", "face", "head", "man", "woman", "human", "body", "arm", "hand", "finger",
    "chair", "couch", "bed", "dining table", "tv", "laptop-screen",
    "refrigerator", "microwave", "oven", "sink", "door", "window", "cabinet", "wardrobe", "cupboard", "closet",
    "toilet", "clock", "vase", "potted plant",
    "wall", "floor", "ceiling", "shelf", "room", "indoor"
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


@st.cache_resource(show_spinner="Cargando modelo de IA...")
def load_model(model_key: str = "yolov8s_world") -> YOLO:
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
    model = YOLO(path)
    if "world" in str(path):
        # Vocabulario abierto de YOLO-World incluyendo personas para filtrar fondos automaticamente
        model.set_classes([
            # Personas y Rostros (para deteccion y descarte de fondos)
            "person", "human face", "man", "woman",
            # Vidrio y Vasos
            "drinking glass", "glass cup", "glass bottle", "glass", "wine glass",
            # Vasos y Potes Plasticos
            "cup", "plastic cup", "disposable cup", "plastic container", "plastic jar", "jar", "pot",
            # Botellas y Envases
            "plastic bottle", "bottle", "thermos",
            "lotion bottle", "cream bottle", "shampoo bottle", "cosmetic bottle",
            # Utiles y Objetos Plasticos / Oficina
            "pen", "plastic pen", "ballpoint pen", "marker", "pencil",
            # Aceite
            "cooking oil bottle", "oil bottle", "cooking oil", "used cooking oil",
            # Metales y Latas
            "aluminum can", "tin can", "can",
            # Papel y Carton
            "cardboard box", "cardboard", "corrugated cardboard", "paper",
            # RAEE y Baterias
            "cell phone", "laptop", "mouse", "keyboard",
            "vape", "vape pen", "electronic cigarette",
            "battery", "batteries", "aa battery", "alkaline battery",
            # Discos
            "compact disc", "cd", "tetra pak"
        ])
    return model


def get_waste_info(class_name: str) -> dict | None:
    norm = class_name.lower().strip()
    
    # Mapeo unicamente de sinonimos estrictos de celulares
    if norm in ("smartphone", "mobile phone", "telephone"):
        norm = "cell phone"
    norm_spaced = norm.replace("-", " ").replace("_", " ").strip()

    # Verificacion estricta de clases ignoradas (personas, rostros, muebles, fondo)
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

    # Si no es un residuo reconocido, retornar None (NO forzar cajas falsas)
    return None


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


def draw_detections(frame_bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
    """
    Dibuja rectangulos delimitadores y etiquetas sobre el frame BGR.
    Soporta tanto OpenCV como fallback puro PIL sin requerir dependencias pesadas.
    """
    if not detections:
        return frame_bgr

    annotated = frame_bgr.copy()
    for det in detections:
        info = get_waste_info(det["clase"])
        if info is None:
            continue
        color = info.get("color", (46, 204, 113))
        x1, y1, x2, y2 = det["bbox"]

        clean_label = det["label"].encode("ascii", "ignore").decode("ascii") or det["clase"]
        star = " *" if det.get("boost_applied", False) else ""
        label_text = f"{clean_label} {det['confianza']:.0%}{star}"

        if HAS_CV2 and cv2 is not None:
            # Marco sutil
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)

            # Esquinas destacadas estilo scanner / visor tecnologico
            box_w = x2 - x1
            box_h = y2 - y1
            c_len = max(14, min(26, box_w // 5, box_h // 5))
            c_thick = 2
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

            # Etiqueta limpia
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
            top_y = max(y1, th + 10)
            cv2.rectangle(annotated, (x1, top_y - th - 8), (x1 + tw + 8, top_y), color, -1)
            cv2.putText(annotated, label_text, (x1 + 4, top_y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            from PIL import Image as _PILImg, ImageDraw as _PILDraw
            pil_temp = _PILImg.fromarray(annotated[:, :, ::-1])
            draw = _PILDraw.Draw(pil_temp)
            rgb_col = (color[2], color[1], color[0]) if len(color) == 3 else color
            draw.rectangle([x1, y1, x2, y2], outline=rgb_col, width=2)
            lbl_w = len(label_text) * 7 + 6
            draw.rectangle([x1, max(0, y1 - 18), x1 + lbl_w, y1], fill=rgb_col)
            draw.text((x1 + 3, max(0, y1 - 16)), label_text, fill=(255, 255, 255))
            annotated = np.array(pil_temp)[:, :, ::-1].copy()

    return annotated


def run_inference(
    model: YOLO,
    frame_bgr: np.ndarray,
    conf_threshold: float = 0.28,
    filter_people: bool = True,
    feedback_store: dict | None = None,
    imgsz: int = 384,
) -> tuple[np.ndarray, list[dict]]:
    """
    Ejecuta deteccion YOLO optimizada en CPU y memoria para Streamlit Cloud.
    Elimina modelos secundarios pesados en memoria (CLIP) para prevenir saturacion de RAM.
    """
    h_img, w_img = frame_bgr.shape[:2]
    frame_area = w_img * h_img

    infer_imgsz = 224 if getattr(model, "task", None) == "classify" else imgsz
    with torch.inference_mode():
        results = model(frame_bgr, conf=conf_threshold, imgsz=infer_imgsz, verbose=False)[0]
    raw_detections = []

    # ── MODO 1: Modelos de clasificacion de imagen completa (YOLOv8-cls) ───────
    if getattr(results, "probs", None) is not None:
        probs = results.probs
        p_full = probs.data.cpu().numpy().copy()

        view_w = int(w_img * 0.62)
        view_h = int(h_img * 0.68)
        x1 = (w_img - view_w) // 2
        y1 = (h_img - view_h) // 2
        x2 = x1 + view_w
        y2 = y1 + view_h
        bbox = (x1, y1, x2, y2)

        center_crop = frame_bgr[y1:y2, x1:x2]
        p_crop = p_full.copy()
        skin_ratio = 0.0

        if center_crop.size > 0:
            if HAS_CV2 and cv2 is not None:
                hsv_crop = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
                skin_mask = cv2.inRange(hsv_crop, np.array([0, 35, 45], dtype=np.uint8), np.array([28, 210, 255], dtype=np.uint8))
                skin_ratio = np.count_nonzero(skin_mask) / float(center_crop.shape[0] * center_crop.shape[1])

            with torch.inference_mode():
                crop_res = model(center_crop, imgsz=infer_imgsz, verbose=False)[0]
            if getattr(crop_res, "probs", None) is not None:
                p_crop = crop_res.probs.data.cpu().numpy().copy()

        p_combined = 0.45 * p_full + 0.55 * p_crop

        # Si el usuario ocupa el centro del visor (rostro/piel > 35%), atenuar falsos positivos de tela/ropa
        if skin_ratio > 0.35:
            for idx_c, name_c in model.names.items():
                if name_c.lower() in ("clothes", "shoes"):
                    p_combined[int(idx_c)] *= 0.20

        total_p = float(np.sum(p_combined))
        if total_p > 0:
            p_combined = p_combined / total_p

        top1_id = int(np.argmax(p_combined))
        top1_conf = float(p_combined[top1_id])
        raw_top1_cls = model.names[top1_id].lower()

        best_cls = raw_top1_cls
        best_raw_conf = top1_conf
        best_eff_conf, best_boost = calculate_boosted_confidence(
            cls_name=raw_top1_cls,
            raw_conf=top1_conf,
            predicted_cls=raw_top1_cls,
            is_held_in_center=False,
            store=feedback_store
        )

        top5_ids = np.argsort(p_combined)[::-1][:5]
        for cid in top5_ids:
            c_name = model.names[int(cid)].lower()
            c_raw = float(p_combined[int(cid)])
            c_eff, c_boost = calculate_boosted_confidence(
                cls_name=c_name,
                raw_conf=c_raw,
                predicted_cls=raw_top1_cls,
                is_held_in_center=False,
                store=feedback_store
            )
            if c_eff > best_eff_conf:
                best_cls = c_name
                best_raw_conf = c_raw
                best_eff_conf = c_eff
                best_boost = c_boost

        min_cls_conf = max(conf_threshold, 0.50)
        if best_eff_conf >= min_cls_conf:
            info = get_waste_info(best_cls)
            if info is not None:
                raw_detections.append({
                    "clase":             best_cls,
                    "label":             info["label"],
                    "emoji":             info.get("emoji", "📦"),
                    "tipo":              info["tipo"],
                    "accion":            info["accion"],
                    "es_especial":       info.get("es_especial", False),
                    "confianza":         best_eff_conf,
                    "confianza_raw":     best_raw_conf,
                    "boost_applied":     best_boost,
                    "is_held_in_center": False,
                    "center_score":      1.0,
                    "hand_detected":     False,
                    "bbox":              bbox,
                })

    # ── MODO 2: Modelos con cajas delimitadoras (YOLO-World v2 / YOLOv8-det) ────
    elif getattr(results, "boxes", None) is not None and len(results.boxes) > 0:
        for box in results.boxes:
            cls_id   = int(box.cls[0])
            cls_name = model.names[cls_id].lower()
            conf     = float(box.conf[0])

            # Filtro inmediato de personas, rostros, muebles o paredes
            if filter_people and (cls_name in IGNORED_CLASSES or any(ign in cls_name for ign in ("person", "face", "man", "woman", "head"))):
                continue

            # Mapeo estricto de celulares (solo sinonimos reales de telefono)
            if cls_name in ("smartphone", "mobile phone", "telephone"):
                cls_name = "cell phone"

            # Umbral saludable para celulares (evita que reflejos o lentes de gafas se marquen como celular)
            if cls_name == "cell phone" and conf < 0.38:
                continue

            # Descartar detecciones que cubran literalmente toda la pantalla (fondo/pared > 90% ancho y alto)
            bbox = tuple(map(int, box.xyxy[0].tolist()))
            x1, y1, x2, y2 = bbox
            bw, bh = x2 - x1, y2 - y1
            area_ratio = (bw * bh) / float(frame_area)
            if (bw > 0.90 * w_img and bh > 0.90 * h_img) or (area_ratio > 0.85 and conf < 0.40):
                continue

            # Filtro para ruido de carton de muy baja confianza en el fondo
            if cls_name in ("cardboard", "carton") and conf < 0.32:
                continue

            # Obtener informacion del residuo (si no es un residuo valido o es clase ignorada, se descarta)
            info = get_waste_info(cls_name)
            if info is None:
                continue

            is_held_in_center, center_score, hand_detected = analyze_spatial_and_hand_context(frame_bgr, bbox)

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
                "is_held_in_center": False,
                "center_score":      center_score,
                "hand_detected":     hand_detected,
                "bbox":              bbox,
            })

    # Si no se detecto ningun residuo valido, retornar vacio (NUNCA inventar cajas falsas)
    if not raw_detections:
        return frame_bgr, []

    # Non-Maximum Suppression (NMS) para eliminar cajas superpuestas del mismo objeto
    raw_detections.sort(key=lambda d: d["confianza"], reverse=True)
    
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

    annotated = draw_detections(frame_bgr, detections)
    return annotated, detections
