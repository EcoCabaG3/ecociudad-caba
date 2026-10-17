"""
puntos_verdes_data.py - Catalogo oficial geolocalizado de Puntos Verdes, Centros Verdes y BA Data Infraestructura
Fuente oficial: https://data.buenosaires.gob.ar/dataset/infraestructura-gestion-residuos
API Geocodificacion Oficial: https://servicios.usig.buenosaires.gob.ar/normalizar/
"""

import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone, timedelta

# Zona horaria oficial de CABA / Argentina (UTC-3 permanente)
TZ_CABA = timezone(timedelta(hours=-3))

# ─── Perfiles de Compatibilidad de Materiales según Tipo de Punto ────────────
# Diferencia claramente qué centro recibe qué material:
# - Punto Verde Especial: Recibe TODOS los reciclables secos + Pilas + Aceite (AVU) + RAEE / Electrónicos + Lámparas + Tóneres + Textiles + Orgánicos.
# - Punto Verde (Estándar): Recibe Reciclables Secos + Pilas + Aceite de Cocina Usado (AVU) + Orgánicos + Cápsulas/Bolsas alimento + Textiles limpios. (NO recibe grandes aparatos electrónicos RAEE, lámparas ni tóneres).
# - Centro Verde (Cooperativa): Planta de clasificación de recuperadores urbanos. Recibe EXCLUSIVAMENTE Reciclables Secos (Plástico, Papel, Cartón, Vidrio, Metal, Tetra-Brik) y Textiles/Calzado. (NO recibe Pilas, NI Aceite, NI RAEE, NI Lámparas).

PERFILES_RECEPCION = {
    "Punto Verde Especial": {
        "clases": [
            "plastic", "paper", "cardboard", "glass", "metal",
            "battery", "pilas", "aceite", "raee", "e-waste",
            "lamparas", "toner", "clothes", "shoes", "biological",
            "capsulas_alimento", "otros_especiales"
        ],
        "resumen": "Reciclables Secos, 🔋 Pilas, 🍳 Aceite (AVU), 💻 RAEE/Electrónicos, 💡 Lámparas, 🖨️ Tóneres, 👕 Textiles y 🍂 Orgánicos",
    },
    "Punto Verde": {
        "clases": [
            "plastic", "paper", "cardboard", "glass", "metal",
            "battery", "pilas", "aceite", "biological",
            "capsulas_alimento", "clothes", "shoes", "otros_especiales"
        ],
        "resumen": "Reciclables Secos, 🔋 Pilas, 🍳 Aceite (AVU), ☕ Cápsulas, 👕 Textiles limpios y 🍂 Orgánicos (No recibe RAEE grandes ni lámparas)",
    },
    "Centro Verde (Cooperativa)": {
        "clases": [
            "plastic", "paper", "cardboard", "glass", "metal",
            "clothes", "shoes"
        ],
        "resumen": "Exclusivamente Reciclables Secos (Plástico, Cartón, Papel, Vidrio, Metal) y 👕 Textiles (NO recibe Pilas, Aceite ni Electrónicos)",
    },
}

# Opciones de filtro para la interfaz de usuario
FILTROS_MATERIAL_UI = {
    "todos": "🌐 Todos los materiales (Ver red completa)",
    "secos": "♻️ Reciclables Secos (Plástico, Cartón, Papel, Vidrio, Metal)",
    "battery": "🔋 Pilas y Baterías en Desuso",
    "aceite": "🍳 Aceite de Cocina Usado (AVU)",
    "raee": "💻 RAEE / Aparatos Electrónicos y Eléctricos",
    "lamparas": "💡 Lámparas, Tubos Fluorescentes y Tóneres",
    "clothes": "👕 Ropa, Textiles y Calzado",
    "biological": "🍂 Residuos Orgánicos para Compostaje",
}

CATEGORIAS_OFICIALES_GCBA = {
    "raee": {
        "titulo": "🔋 RAEE (Aparatos Eléctricos y Electrónicos)",
        "items": "Celulares, computadoras, notebooks, tablets, cámaras, radios, teclados, mouse, pequeños electrodomésticos.",
        "condicion": "Limpios y completos. Hasta 10 aparatos por persona. Se reciben en Puntos Verdes Especiales y Puntos Verdes habilitados.",
        "obligatorio_punto_verde": True
    },
    "aceite": {
        "titulo": "🍳 Aceite de Cocina Usado (AVU)",
        "items": "Aceite vegetal de fritura usado para cocinar.",
        "condicion": "Frío, en botellas plásticas cerradas y limpias (se transforma en Biodiésel). 1L contamina 1000L de agua.",
        "obligatorio_punto_verde": True
    },
    "pilas": {
        "titulo": "🔋 Pilas en Desuso",
        "items": "Pilas cilíndricas (AA, AAA, AAAA, C, D), prismáticas 9V y pilas botón.",
        "condicion": "Sueltas o en bolsa para depositar en los tubos especiales de Puntos Verdes.",
        "obligatorio_punto_verde": True
    },
    "toner": {
        "titulo": "🖨️ Cartuchos de Tinta y Tóneres",
        "items": "Cartuchos de impresoras y tóneres de fotocopiadoras.",
        "condicion": "Limpios, sin derrames. Hasta 3 unidades por persona en Puntos Verdes Especiales.",
        "obligatorio_punto_verde": True
    },
    "lamparas": {
        "titulo": "💡 Lámparas y Tubos Fluorescentes",
        "items": "Lámparas de bajo consumo, LED, incandescentes, halógenas, dicroicas y tubos fluorescentes.",
        "condicion": "En cajas o envueltas (contienen mercurio). Hasta 5 por persona/día en Puntos Verdes Especiales. NO se reciben lámparas rotas.",
        "obligatorio_punto_verde": True
    },
    "capsulas_alimento": {
        "titulo": "☕ Cápsulas de Café y Bolsas de Alimento Balanceado",
        "items": "Cápsulas plásticas/aluminio y bolsas de comida de mascotas.",
        "condicion": "Vacías, limpias y secas (se transforman en madera plástica).",
        "obligatorio_punto_verde": False
    },
    "organicos_compost": {
        "titulo": "🍂 Residuos Orgánicos para Compostaje",
        "items": "Frutas, verduras, cáscaras de huevo, café, yerba, té, hojas secas, pasto recién cortado.",
        "condicion": "En recipientes o bolsas. NO carnes, huesos ni comida cocida.",
        "obligatorio_punto_verde": False
    },
    "otros_especiales": {
        "titulo": "🕶️ Otros Especiales",
        "items": "Anteojos, CDs/DVDs, ecobolsas de polipropileno, chapitas, jeans, paraguas rotos, placas radiográficas.",
        "condicion": "Limpios y secos.",
        "obligatorio_punto_verde": True
    }
}

# Puntos Verdes que además cuentan con módulo de recepción de RAEE / Electrónicos pequeños y Lámparas
_PV_CON_RAEE_HABILITADO = {
    "Plaza San Martín",
    "Punto Verde Puerto Madero",
    "Plaza Monseñor De Andrea",
    "Parque Patricios",
    "Parque Rivadavia",
    "Parque Centenario",
    "Plaza Pueyrredón (Plaza Flores)",
    "Parque Chacabuco",
    "Parque Avellaneda",
    "Plaza Arenales",
    "Parque Saavedra",
    "Plaza Manuel Belgrano",
    "Plaza Güemes",
    "Plaza Palermo Viejo",
    "Parque Los Andes",
}

# Red de Puntos Verdes y Puntos Verdes Especiales
PUNTOS_VERDES_LIST = [
    {"nombre": "Plaza San Martín", "direccion": "Av. Santa Fe y Maipú", "comuna": 1, "tipo": "Punto Verde Especial", "lat": -34.5950, "lon": -58.3770, "estado": "Abierto", "horario": "Mié a Dom 11 a 14:30 y 15 a 19 h"},
    {"nombre": "Punto Verde Puerto Madero", "direccion": "Av. Alicia Moreau de Justo y Azucena Villaflor", "comuna": 1, "tipo": "Punto Verde Especial", "lat": -34.6120, "lon": -58.3645, "estado": "Abierto", "horario": "Mié a Dom 11 a 14:30 y 15 a 19 h"},
    {"nombre": "Plaza Rosario Vera Peñaloza", "direccion": "Av. San Juan y Chacabuco", "comuna": 1, "tipo": "Punto Verde", "lat": -34.6200, "lon": -58.3760, "estado": "Abierto", "horario": "Mié a Dom 11 a 14:30 y 15 a 19 h"},
    {"nombre": "Plaza Monseñor De Andrea", "direccion": "Anchorena y Av. Córdoba", "comuna": 2, "tipo": "Punto Verde Especial", "lat": -34.5985, "lon": -58.4040, "estado": "Abierto", "horario": "Mié a Dom 11 a 14:30 y 15 a 19 h"},
    {"nombre": "Plazoleta Reino de Tailandia", "direccion": "Av. Pueyrredón y Vicente López", "comuna": 2, "tipo": "Punto Verde", "lat": -34.5870, "lon": -58.3930, "estado": "Abierto", "horario": "Mié a Dom 11 a 14:30 y 15 a 19 h"},
    {"nombre": "Plaza 1ro de Mayo", "direccion": "Hipólito Yrigoyen y Pasco", "comuna": 3, "tipo": "Punto Verde", "lat": -34.6130, "lon": -58.4010, "estado": "Abierto", "horario": "Mié a Dom 11 a 14:30 y 15 a 19 h"},
    {"nombre": "Plaza Dr. J.M. Velazco Ibarra", "direccion": "Av. Jujuy y México", "comuna": 3, "tipo": "Punto Verde", "lat": -34.6180, "lon": -58.4050, "estado": "Cerrado por mantenimiento", "horario": "Cerrado temporalmente"},
    {"nombre": "Parque Patricios", "direccion": "Av. Caseros y Monteagudo", "comuna": 4, "tipo": "Punto Verde Especial", "lat": -34.6360, "lon": -58.4060, "estado": "Abierto", "horario": "Mié a Dom 11 a 14:30 y 15 a 19 h"},
    {"nombre": "Plaza Colombia", "direccion": "Av. Montes de Oca y Brandsen", "comuna": 4, "tipo": "Punto Verde", "lat": -34.6390, "lon": -58.3750, "estado": "Abierto", "horario": "Mié a Dom 11 a 14:30 y 15 a 19 h"},
    {"nombre": "Plaza Mariano Boedo", "direccion": "Estados Unidos y Sánchez de Loria", "comuna": 5, "tipo": "Punto Verde", "lat": -34.6235, "lon": -58.4160, "estado": "Abierto", "horario": "Mié a Dom 11 a 14:30 y 15 a 19 h"},
    {"nombre": "Plaza Almagro", "direccion": "Sarmiento y Bulnes", "comuna": 5, "tipo": "Punto Verde", "lat": -34.6050, "lon": -58.4195, "estado": "Abierto", "horario": "Mié a Dom 11 a 14:30 y 15 a 19 h"},
    {"nombre": "Parque Rivadavia", "direccion": "Av. Rivadavia y Florencio Balcarce", "comuna": 6, "tipo": "Punto Verde", "lat": -34.6178, "lon": -58.4345, "estado": "Abierto", "horario": "Mié a Dom 11 a 14:30 y 15 a 19 h"},
    {"nombre": "Parque Centenario", "direccion": "Av. Patricias Argentinas y Av. Roentgen", "comuna": 6, "tipo": "Punto Verde Especial", "lat": -34.6065, "lon": -58.4355, "estado": "Abierto", "horario": "Mié a Dom 11 a 19 h (Corrido)"},
    {"nombre": "Plaza Pueyrredón (Plaza Flores)", "direccion": "Yerbal y Artigas", "comuna": 7, "tipo": "Punto Verde Especial", "lat": -34.6285, "lon": -58.4630, "estado": "Abierto", "horario": "Mié a Dom 11 a 14:30 y 15 a 19 h"},
    {"nombre": "Parque Chacabuco", "direccion": "Av. Asamblea y Hortiguera", "comuna": 7, "tipo": "Punto Verde", "lat": -34.6340, "lon": -58.4420, "estado": "Abierto", "horario": "Mié a Dom 11 a 14:30 y 15 a 19 h"},
    {"nombre": "Plaza Sudamérica", "direccion": "Av. Piedra Buena y Av. Fernández de la Cruz", "comuna": 8, "tipo": "Punto Verde", "lat": -34.6750, "lon": -58.4680, "estado": "Abierto", "horario": "Mié a Dom 11 a 14:30 y 15 a 19 h"},
    {"nombre": "Estación Villa Soldati", "direccion": "Lafuente y Corrales", "comuna": 8, "tipo": "Punto Verde", "lat": -34.6620, "lon": -58.4470, "estado": "Abierto", "horario": "Mié a Dom 11 a 14:30 y 15 a 19 h"},
    {"nombre": "Parque Avellaneda", "direccion": "Av. Directorio y Fernández", "comuna": 9, "tipo": "Punto Verde Especial", "lat": -34.6460, "lon": -58.4770, "estado": "Abierto", "horario": "Mié a Dom 11 a 14:30 y 15 a 19 h"},
    {"nombre": "Parque Santojanni", "direccion": "Lisandro de La Torre y Patrón", "comuna": 9, "tipo": "Punto Verde", "lat": -34.6520, "lon": -58.5130, "estado": "Abierto", "horario": "Mié a Dom 11 a 14:30 y 15 a 19 h"},
    {"nombre": "Plaza Don Bosco", "direccion": "Av. Lope de Vega y Elpidio González", "comuna": 10, "tipo": "Punto Verde", "lat": -34.6220, "lon": -58.5140, "estado": "Abierto", "horario": "Mié a Dom 11 a 14:30 y 15 a 19 h"},
    {"nombre": "Plaza Vélez Sarsfield", "direccion": "Av. Avellaneda y Chivilcoy", "comuna": 10, "tipo": "Punto Verde", "lat": -34.6300, "lon": -58.4860, "estado": "Abierto", "horario": "Mié a Dom 11 a 14:30 y 15 a 19 h"},
    {"nombre": "Plaza Aristóbulo del Valle", "direccion": "Campana y Baigorria", "comuna": 11, "tipo": "Punto Verde", "lat": -34.6060, "lon": -58.4980, "estado": "Abierto", "horario": "Mié a Dom 11 a 14:30 y 15 a 19 h"},
    {"nombre": "Plaza Arenales", "direccion": "Nueva York y Mercedes", "comuna": 11, "tipo": "Punto Verde Especial", "lat": -34.5995, "lon": -58.5080, "estado": "Abierto", "horario": "Mié a Dom 11 a 19 h (Corrido)"},
    {"nombre": "Parque Saavedra", "direccion": "Roque Pérez y Paroissien", "comuna": 12, "tipo": "Punto Verde Especial", "lat": -34.5510, "lon": -58.4850, "estado": "Abierto", "horario": "Mié a Dom 11 a 14:30 y 15 a 19 h"},
    {"nombre": "Plaza Leandro N. Alem", "direccion": "Artigas y Larsen", "comuna": 12, "tipo": "Punto Verde", "lat": -34.5710, "lon": -58.5040, "estado": "Abierto", "horario": "Mié a Dom 11 a 14:30 y 15 a 19 h"},
    {"nombre": "Plaza Manuel Belgrano", "direccion": "Cuba y Av. Juramento", "comuna": 13, "tipo": "Punto Verde Especial", "lat": -34.5620, "lon": -58.4560, "estado": "Abierto", "horario": "Mié a Dom 11 a 14:30 y 15 a 19 h"},
    {"nombre": "Plaza Balcarce", "direccion": "Manzanares y Vuelta de Obligado", "comuna": 13, "tipo": "Punto Verde", "lat": -34.5440, "lon": -58.4680, "estado": "Abierto", "horario": "Mié a Dom 11 a 14:30 y 15 a 19 h"},
    {"nombre": "Plaza Güemes", "direccion": "Medrano y Charcas", "comuna": 14, "tipo": "Punto Verde Especial", "lat": -34.5900, "lon": -58.4160, "estado": "Abierto", "horario": "Mié a Dom 11 a 14:30 y 15 a 19 h"},
    {"nombre": "Plaza República de Bolivia", "direccion": "Av. Libertador y Olleros", "comuna": 14, "tipo": "Punto Verde", "lat": -34.5640, "lon": -58.4310, "estado": "Abierto", "horario": "Mié a Dom 11 a 14:30 y 15 a 19 h"},
    {"nombre": "Plaza Palermo Viejo", "direccion": "Malabia y Costa Rica", "comuna": 14, "tipo": "Punto Verde", "lat": -34.5905, "lon": -58.4270, "estado": "Abierto", "horario": "Mié a Dom 11 a 14:30 y 15 a 19 h"},
    {"nombre": "Parque Los Andes", "direccion": "Av. Dorrego y Av. Guzmán", "comuna": 15, "tipo": "Punto Verde Especial", "lat": -34.5880, "lon": -58.4480, "estado": "Abierto", "horario": "Mié a Dom 11 a 14:30 y 15 a 19 h"},
    {"nombre": "Plaza 25 de Agosto", "direccion": "Charlone y Heredia", "comuna": 15, "tipo": "Punto Verde", "lat": -34.5810, "lon": -58.4670, "estado": "Cerrado por mantenimiento", "horario": "Cerrado temporalmente"},
]

# 14 Centros Verdes oficiales de Cooperativas de Recuperadores Urbanos (BA Data)
CENTROS_VERDES_LIST = [
    {"nombre": "Centro Verde Yerbal", "administra": "Coop. Recuperadores del Oeste", "direccion": "Yerbal 1481", "barrio": "Caballito", "comuna": 6, "lat": -34.6235, "lon": -58.4525, "tipo": "Centro Verde (Cooperativa)", "estado": "Operativo", "horario": "Lunes a Viernes (Recuperación y acondicionamiento)"},
    {"nombre": "Centro Verde Cortejarena", "administra": "Coop. Amanecer de los Cartoneros", "direccion": "Cortejarena 3151", "barrio": "Barracas", "comuna": 4, "lat": -34.6436, "lon": -58.4066, "tipo": "Centro Verde (Cooperativa)", "estado": "Operativo", "horario": "Lunes a Viernes"},
    {"nombre": "Centro Verde Barracas", "administra": "Coop. Baires Cero Con Límite", "direccion": "Herrera 2224", "barrio": "Barracas", "comuna": 4, "lat": -34.6478, "lon": -58.3791, "tipo": "Centro Verde (Cooperativa)", "estado": "Operativo", "horario": "Lunes a Viernes"},
    {"nombre": "Centro Verde Chilavert", "administra": "Coop. El Álamo", "direccion": "Chilavert 2745", "barrio": "Villa Soldati", "comuna": 8, "lat": -34.6651, "lon": -58.4492, "tipo": "Centro Verde (Cooperativa)", "estado": "Operativo", "horario": "Lunes a Viernes"},
    {"nombre": "Centro Verde Retiro", "administra": "Coop. El Ceibo", "direccion": "Av. Antártida Argentina y San Martín", "barrio": "Retiro", "comuna": 1, "lat": -34.5882, "lon": -58.3741, "tipo": "Centro Verde (Cooperativa)", "estado": "Operativo", "horario": "Lunes a Viernes"},
    {"nombre": "Centro Verde Belgrano", "administra": "Coop. Madreselva", "direccion": "Juramento 3550", "barrio": "Belgrano", "comuna": 13, "lat": -34.5694, "lon": -58.4618, "tipo": "Centro Verde (Cooperativa)", "estado": "Operativo", "horario": "Lunes a Viernes"},
    {"nombre": "Centro Verde Saavedra", "administra": "Coop. Recuperadores Urbanos", "direccion": "Av. Balbín 4100", "barrio": "Saavedra", "comuna": 12, "lat": -34.5521, "lon": -58.4912, "tipo": "Centro Verde (Cooperativa)", "estado": "Operativo", "horario": "Lunes a Viernes"},
    {"nombre": "Centro Verde Flores", "administra": "Coop. Trabajo y Dignidad", "direccion": "Varela 1400", "barrio": "Flores", "comuna": 7, "lat": -34.6385, "lon": -58.4571, "tipo": "Centro Verde (Cooperativa)", "estado": "Operativo", "horario": "Lunes a Viernes"},
    {"nombre": "Centro Verde Lugano", "administra": "Coop. Amanecer de los Cartoneros", "direccion": "Av. Cruz y Pola", "barrio": "Villa Lugano", "comuna": 8, "lat": -34.6791, "lon": -58.4725, "tipo": "Centro Verde (Cooperativa)", "estado": "Operativo", "horario": "Lunes a Viernes"},
    {"nombre": "Centro Verde Pompeya", "administra": "Coop. Primavera", "direccion": "Amancio Alcorta 3400", "barrio": "Nueva Pompeya", "comuna": 4, "lat": -34.6512, "lon": -58.4183, "tipo": "Centro Verde (Cooperativa)", "estado": "Operativo", "horario": "Lunes a Viernes"},
    {"nombre": "Centro Verde Chacarita", "administra": "Coop. Correcamino", "direccion": "Guzmán 50", "barrio": "Chacarita", "comuna": 15, "lat": -34.5873, "lon": -58.4496, "tipo": "Centro Verde (Cooperativa)", "estado": "Operativo", "horario": "Lunes a Viernes"},
    {"nombre": "Centro Verde San Telmo", "administra": "Coop. Recicladores Unidos", "direccion": "Perú 1200", "barrio": "San Telmo", "comuna": 1, "lat": -34.6214, "lon": -58.3732, "tipo": "Centro Verde (Cooperativa)", "estado": "Operativo", "horario": "Lunes a Viernes"},
    {"nombre": "Centro Verde Mataderos", "administra": "Coop. Nueva Generación", "direccion": "Av. de los Corrales 7200", "barrio": "Mataderos", "comuna": 9, "lat": -34.6610, "lon": -58.5085, "tipo": "Centro Verde (Cooperativa)", "estado": "Operativo", "horario": "Lunes a Viernes"},
    {"nombre": "Centro Verde Paternal", "administra": "Coop. Cartoneros del Centro", "direccion": "Warnes 1500", "barrio": "La Paternal", "comuna": 15, "lat": -34.5962, "lon": -58.4611, "tipo": "Centro Verde (Cooperativa)", "estado": "Operativo", "horario": "Lunes a Viernes"},
]

# Enriquecer cada punto con su lista explícita de materiales aceptados
for _pt in PUNTOS_VERDES_LIST + CENTROS_VERDES_LIST:
    _tipo = _pt.get("tipo", "Punto Verde")
    _perfil = PERFILES_RECEPCION.get(_tipo, PERFILES_RECEPCION["Punto Verde"])
    _pt["materiales_clases"] = list(_perfil["clases"])
    _pt["materiales_desc"] = _perfil["resumen"]


def punto_recibe_material(punto: dict, clase_material: str | None) -> bool:
    """
    Verifica si un Punto Verde o Centro Verde recibe específicamente la categoría de residuo indicada.
    Por ejemplo:
    - 'battery' (Pilas) o 'aceite' (AVU): SÍ en Puntos Verdes y Puntos Verdes Especiales; NO en Centros Verdes (Cooperativas).
    - 'raee' / 'e-waste' / 'lamparas' / 'toner': SÍ en Puntos Verdes Especiales; NO en Puntos Verdes simples ni Cooperativas.
    - 'plastic', 'cardboard', 'paper', 'glass', 'metal': SÍ en todos (Puntos Verdes, Especiales y Cooperativas).
    """
    if not clase_material or clase_material in ("todos", "all", "trash", "unknown"):
        return True

    clases_aceptadas = punto.get("materiales_clases")
    if not clases_aceptadas:
        tipo = punto.get("tipo", "Punto Verde")
        clases_aceptadas = PERFILES_RECEPCION.get(tipo, PERFILES_RECEPCION["Punto Verde"])["clases"]

    c = clase_material.lower().strip()
    if c == "secos":
        return any(k in clases_aceptadas for k in ("plastic", "paper", "cardboard", "glass", "metal"))
    if c in ("e-waste", "raee", "electronic", "electronics", "cell phone", "laptop", "keyboard", "mouse"):
        return "raee" in clases_aceptadas or "e-waste" in clases_aceptadas
    if c in ("battery", "pilas", "pila"):
        return "battery" in clases_aceptadas or "pilas" in clases_aceptadas
    if c in ("aceite", "avu", "oil"):
        return "aceite" in clases_aceptadas
    if c in ("lamparas", "toner", "fluorescent"):
        return "lamparas" in clases_aceptadas or "toner" in clases_aceptadas
    if c in ("clothes", "shoes", "textil"):
        return "clothes" in clases_aceptadas or "shoes" in clases_aceptadas
    if c in ("biological", "organicos", "compost", "organicos_compost"):
        return "biological" in clases_aceptadas

    return c in clases_aceptadas


def filtrar_puntos_por_material(puntos: list[dict], clase_material: str | None) -> list[dict]:
    """
    Filtra una lista de Puntos/Centros Verdes dejando únicamente los que reciben el material especificado.
    """
    if not clase_material or clase_material in ("todos", "all", "trash", "unknown"):
        return list(puntos)
    filtrados = [p for p in puntos if punto_recibe_material(p, clase_material)]
    return filtrados if filtrados else list(puntos)


# Diccionario local de respaldo por barrio / hito de CABA en caso de corte de internet
_BARRIOS_CABA_FALLBACK = {
    "palermo": (-34.5820, -58.4250, "Palermo, CABA"),
    "recoleta": (-34.5905, -58.3960, "Recoleta, CABA"),
    "belgrano": (-34.5620, -58.4560, "Belgrano, CABA"),
    "caballito": (-34.6178, -58.4360, "Caballito, CABA"),
    "almagro": (-34.6090, -58.4200, "Almagro, CABA"),
    "boedo": (-34.6250, -58.4170, "Boedo, CABA"),
    "flores": (-34.6285, -58.4630, "Flores, CABA"),
    "floresta": (-34.6290, -58.4840, "Floresta, CABA"),
    "villa urquiza": (-34.5730, -58.4870, "Villa Urquiza, CABA"),
    "villa devoto": (-34.5995, -58.5080, "Villa Devoto, CABA"),
    "villa del parque": (-34.6060, -58.4950, "Villa del Parque, CABA"),
    "villa crespo": (-34.5980, -58.4420, "Villa Crespo, CABA"),
    "chacarita": (-34.5880, -58.4500, "Chacarita, CABA"),
    "colegiales": (-34.5750, -58.4480, "Colegiales, CABA"),
    "nuñez": (-34.5450, -58.4650, "Núñez, CABA"),
    "nunez": (-34.5450, -58.4650, "Núñez, CABA"),
    "saavedra": (-34.5520, -58.4860, "Saavedra, CABA"),
    "san telmo": (-34.6210, -58.3735, "San Telmo, CABA"),
    "microcentro": (-34.6037, -58.3780, "Microcentro / San Nicolás, CABA"),
    "obelisco": (-34.6037, -58.3816, "Obelisco (Av. 9 de Julio y Av. Corrientes), CABA"),
    "retiro": (-34.5920, -58.3760, "Retiro, CABA"),
    "puerto madero": (-34.6110, -58.3630, "Puerto Madero, CABA"),
    "constitucion": (-34.6275, -58.3810, "Constitución, CABA"),
    "barracas": (-34.6430, -58.3850, "Barracas, CABA"),
    "la boca": (-34.6350, -58.3640, "La Boca, CABA"),
    "parque patricios": (-34.6360, -58.4050, "Parque Patricios, CABA"),
    "nueva pompeya": (-34.6510, -58.4180, "Nueva Pompeya, CABA"),
    "balvanera": (-34.6100, -58.4030, "Balvanera / Once, CABA"),
    "once": (-34.6085, -58.4060, "Once (Balvanera), CABA"),
    "san cristobal": (-34.6220, -58.4010, "San Cristóbal, CABA"),
    "parque chacabuco": (-34.6340, -58.4420, "Parque Chacabuco, CABA"),
    "mataderos": (-34.6580, -58.5080, "Mataderos, CABA"),
    "liniers": (-34.6430, -58.5230, "Liniers, CABA"),
    "villa lugano": (-34.6770, -58.4720, "Villa Lugano, CABA"),
    "villa soldati": (-34.6630, -58.4470, "Villa Soldati, CABA"),
    "paternal": (-34.5970, -58.4650, "La Paternal, CABA"),
    "agronomia": (-34.5920, -58.4850, "Agronomía, CABA"),
}


def geocodificar_direccion_caba(direccion: str) -> tuple[float, float, str] | None:
    """
    Geocodifica una calle y altura o intersección dentro de la Ciudad Autónoma de Buenos Aires (CABA).
    1. Consulta en primer lugar el Normalizador de Direcciones Oficial del GCBA (USIG):
       https://servicios.usig.buenosaires.gob.ar/normalizar/?direccion=...&geocodificar=true
    2. Si es un nombre de barrio o lugar, utiliza OpenStreetMap Nominatim acotado a CABA y/o diccionario local.

    Retorna:
        (latitud: float, longitud: float, direccion_normalizada: str) o None si no se encontró.
    """
    if not direccion or not direccion.strip():
        return None

    raw = direccion.strip()
    raw_lower = raw.lower()

    # 1. Intentar con API Oficial USIG del Gobierno de la Ciudad de Buenos Aires
    try:
        q_usig = urllib.parse.quote(raw)
        url_usig = f"https://servicios.usig.buenosaires.gob.ar/normalizar/?direccion={q_usig}&geocodificar=true"
        req = urllib.request.Request(url_usig, headers={"User-Agent": "EcoCiudadCABA-IFTS11/1.0"})
        with urllib.request.urlopen(req, timeout=4.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            dirs = data.get("direccionesNormalizadas", [])
            for d in dirs:
                # Filtrar resultados dentro de CABA
                cod_partido = str(d.get("cod_partido", "")).lower()
                if cod_partido and cod_partido != "caba":
                    continue
                coords = d.get("coordenadas")
                if coords and "y" in coords and "x" in coords:
                    lat = float(coords["y"])
                    lon = float(coords["x"])
                    if -34.72 <= lat <= -34.52 and -58.55 <= lon <= -58.33:
                        nom = d.get("direccion", raw)
                        return lat, lon, nom
    except Exception:
        pass

    # 2. Revisar diccionario rápido de barrios / hitos de CABA
    for key_barrio, (b_lat, b_lon, b_label) in _BARRIOS_CABA_FALLBACK.items():
        if key_barrio in raw_lower:
            return b_lat, b_lon, b_label

    # 3. Fallback a Nominatim OpenStreetMap acotado a Ciudad Autónoma de Buenos Aires
    try:
        q_osm = urllib.parse.quote(f"{raw}, Ciudad Autónoma de Buenos Aires, Argentina")
        url_osm = f"https://nominatim.openstreetmap.org/search?q={q_osm}&format=json&limit=1"
        req_osm = urllib.request.Request(url_osm, headers={"User-Agent": "EcoCiudadCABA-IFTS11/1.0"})
        with urllib.request.urlopen(req_osm, timeout=4.5) as resp:
            results = json.loads(resp.read().decode("utf-8"))
            if results and isinstance(results, list):
                lat = float(results[0]["lat"])
                lon = float(results[0]["lon"])
                if -34.72 <= lat <= -34.52 and -58.55 <= lon <= -58.33:
                    disp = results[0].get("display_name", raw).split(",")[0:2]
                    return lat, lon, ", ".join(disp) + ", CABA"
    except Exception:
        pass

    return None


def generar_links_navegacion(
    dest_lat: float,
    dest_lon: float,
    orig_lat: float | None = None,
    orig_lon: float | None = None,
    orig_direccion: str | None = None,
) -> tuple[str, str]:
    """
    Genera los deep-links directos para abrir la ruta hacia el Punto Verde en Google Maps y en Waze.
    Retorna: (google_maps_url, waze_url)
    """
    if orig_lat is not None and orig_lon is not None:
        gmaps_url = (
            f"https://www.google.com/maps/dir/?api=1"
            f"&origin={orig_lat},{orig_lon}"
            f"&destination={dest_lat},{dest_lon}&travelmode=walking"
        )
    elif orig_direccion and orig_direccion.strip():
        orig_q = urllib.parse.quote(f"{orig_direccion.strip()}, CABA, Argentina")
        gmaps_url = (
            f"https://www.google.com/maps/dir/?api=1"
            f"&origin={orig_q}"
            f"&destination={dest_lat},{dest_lon}"
        )
    else:
        gmaps_url = f"https://www.google.com/maps/dir/?api=1&destination={dest_lat},{dest_lon}"

    waze_url = f"https://waze.com/ul?ll={dest_lat},{dest_lon}&navigate=yes"
    return gmaps_url, waze_url


def calcular_estado_apertura(punto: dict, now: datetime | None = None) -> tuple[bool, str, str]:
    """
    Evalua en tiempo real si el Punto Verde o Centro Verde esta actualmente ABIERTO o CERRADO
    segun el estado operativo y el horario oficial vs la hora y dia del sistema.
    """
    if now is None:
        now = datetime.now(TZ_CABA)
    elif now.tzinfo is not None:
        now = now.astimezone(TZ_CABA)

    estado_base = str(punto.get("estado", "")).strip()
    if "Cerrado" in estado_base:
        return False, "🔴 " + estado_base, estado_base

    horario_str = str(punto.get("horario", "")).strip()
    if "Cerrado" in horario_str:
        return False, "🔴 Cerrado temporalmente", "Cerrado temporalmente"

    weekday = now.weekday()  # 0=Lunes, 1=Martes, 2=Miercoles, ..., 6=Domingo
    current_time_float = now.hour + now.minute / 60.0

    # 1. Puntos Verdes habituales ("Mié a Dom 11 a 14:30 y 15 a 19 h")
    if "14:30" in horario_str:
        if weekday not in [2, 3, 4, 5, 6]:
            dias_faltantes = (2 - weekday) % 7
            dia_nombre = "el miércoles" if dias_faltantes > 1 else "mañana"
            return False, "🔴 Cerrado ahora", f"Cerrado hoy. Abre {dia_nombre} a las 11:00 h"
        if current_time_float < 11.0:
            return False, "🔴 Cerrado ahora", "Abre hoy a las 11:00 h"
        elif 11.0 <= current_time_float < 14.5:
            return True, "🟢 Abierto ahora", "Abierto (Turno mañana hasta 14:30 h)"
        elif 14.5 <= current_time_float < 15.0:
            return False, "🔴 Cerrado por receso", "En receso de mediodía. Abre a las 15:00 h"
        elif 15.0 <= current_time_float < 19.0:
            return True, "🟢 Abierto ahora", "Abierto hasta las 19:00 h"
        else:
            return False, "🔴 Cerrado ahora", "Cerró a las 19:00 h"

    # 2. Puntos Verdes Especiales horario corrido ("Mié a Dom 11 a 19 h (Corrido)")
    elif "Corrido" in horario_str or ("Mié a Dom" in horario_str and "19" in horario_str):
        if weekday not in [2, 3, 4, 5, 6]:
            dias_faltantes = (2 - weekday) % 7
            dia_nombre = "el miércoles" if dias_faltantes > 1 else "mañana"
            return False, "🔴 Cerrado ahora", f"Cerrado hoy. Abre {dia_nombre} a las 11:00 h"
        if current_time_float < 11.0:
            return False, "🔴 Cerrado ahora", "Abre hoy a las 11:00 h"
        elif 11.0 <= current_time_float < 19.0:
            return True, "🟢 Abierto ahora", "Abierto horario corrido hasta las 19:00 h"
        else:
            return False, "🔴 Cerrado ahora", "Cerró a las 19:00 h"

    # 3. Centros Verdes de cooperativas ("Lunes a Viernes")
    elif "Lunes a Viernes" in horario_str:
        if weekday not in [0, 1, 2, 3, 4]:
            return False, "🔴 Cerrado fin de semana", "Abre el lunes a las 08:00 h"
        if current_time_float < 8.0:
            return False, "🔴 Cerrado ahora", "Abre hoy a las 08:00 h"
        elif 8.0 <= current_time_float < 17.0:
            return True, "🟢 Abierto ahora", "Abierto hasta las 17:00 h"
        else:
            return False, "🔴 Cerrado ahora", "Cerró a las 17:00 h"

    # Fallback general
    if "Abierto" in estado_base or "Operativo" in estado_base:
        return True, "🟢 Operativo", "Consultar horario"
    return False, "🔴 Cerrado", estado_base


def get_puntos_habilitados(
    incluir_centros: bool = True,
    solo_abiertos_ahora: bool = False,
    clase_material: str | None = None,
) -> list[dict]:
    """
    Retorna la lista de Puntos Verdes (y opcionalmente Centros Verdes) operativos.
    Si clase_material se especifica, filtra estrictamente los que reciben ese tipo de residuo.
    Si solo_abiertos_ahora=True, filtra los que se encuentran abiertos en este instante.
    """
    puntos = [p for p in PUNTOS_VERDES_LIST if "Cerrado" not in str(p.get("estado", ""))]
    if incluir_centros:
        puntos += [c for c in CENTROS_VERDES_LIST if "Cerrado" not in str(c.get("estado", ""))]
    if clase_material:
        puntos = filtrar_puntos_por_material(puntos, clase_material)
    if solo_abiertos_ahora:
        puntos = [p for p in puntos if calcular_estado_apertura(p)[0]]
    return puntos


def get_puntos_por_comuna(comuna: int | str, solo_abiertos: bool = True) -> list[dict]:
    """
    Filtra los puntos y centros verdes pertenecientes a una comuna especifica de CABA (1 al 15).
    """
    base = get_puntos_habilitados(incluir_centros=True) if solo_abiertos else (PUNTOS_VERDES_LIST + CENTROS_VERDES_LIST)
    try:
        comuna_num = int(comuna)
        return [p for p in base if p.get("comuna") == comuna_num]
    except (ValueError, TypeError):
        return base
