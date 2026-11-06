"""
config.py
=========
Configuración central del proyecto "Cartografía Automatizada".

Todo lo que comparten el entrenamiento (Colab), la evaluación y la app web
vive aquí: rutas, definición de clases, colores, división de datos y
hiperparámetros por defecto. Cambiar un valor aquí lo cambia en todo el
proyecto.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                      # aquí se descomprime el zip de Kaggle
SEG_DIR = RAW_DIR / "Dataset segmentacion"      # recorte_XX.png + ground_truth_recorte_XX.npy
EXTRA_DIR = RAW_DIR / "Recortes adicionales"    # imágenes sin etiqueta (demo / inferencia)
EXTERNAL_DIR = DATA_DIR / "pruebas_externas"    # imágenes de otras fuentes (p. ej. capturas de Google Maps)
SPLIT_FILE = DATA_DIR / "splits.json"           # división train/val/test reproducible
WEIGHTS_DIR = ROOT / "models" / "weights"       # pesos .pt de los mejores modelos
OUTPUTS_DIR = ROOT / "outputs"                  # métricas, curvas, figuras para el informe

KAGGLE_DATASET = "cetaamb/reto-1-cartografa-automatizada"

# --------------------------------------------------------------------------
# Datos del equipo (se muestran en la app web). Edita los nombres aquí.
# --------------------------------------------------------------------------
AGENCY = "Agencia Orbital Chicamocha"      # agencia ficticia de la misión
MISSION = "GEOTRAZA-01"                     # código de la misión
TEAM = ["Natalia Serrano Ardila", "Dariana Andrea Torrecilla"]
COURSE = "Electiva IV · Deep Computer Vision · Taller 3, 3.er corte"
PROFESSOR = "Ing. Carlos Daniel Pimentel Díaz, M.Sc."
UNIVERSITY = "Universidad Santo Tomás · Seccional Bucaramanga · Ingeniería Mecatrónica"

# --------------------------------------------------------------------------
# Clases
# --------------------------------------------------------------------------
# Canales del ground truth (.npy de forma H x W x 4), en el orden del dataset:
GT_CHANNELS = ["Área Caminable", "Área Construida", "Zona Verde", "Carretera"]

# Mapa semántico interno: 0 = fondo (píxeles sin etiqueta), 1..4 = clases.
CLASS_NAMES = ["Fondo", "Área Caminable", "Área Construida", "Zona Verde", "Carretera"]
NUM_CLASSES = len(CLASS_NAMES)           # 5 (incluye fondo)
EVAL_CLASSES = [1, 2, 3, 4]              # las métricas del taller se reportan sobre las 4 clases

# Segmentación panóptica: "things" (objetos contables, con instancias) y
# "stuff" (regiones amorfas). Las manzanas construidas y las zonas verdes se
# pueden contar como regiones separadas; carretera y andén son redes continuas.
THING_CLASSES = [2, 3]                   # Área Construida, Zona Verde
STUFF_CLASSES = [0, 1, 4]                # Fondo, Área Caminable, Carretera

# Prioridad para resolver píxeles con más de una etiqueta en el GT
# (p. ej. copas de árboles sobre andenes). Primero = gana. Se usa lo que se
# ve desde el satélite: la vegetación tapa lo que hay debajo.
OVERLAP_PRIORITY = [3, 4, 1, 2]          # Zona Verde > Carretera > Caminable > Construida

# Colores RGB para visualización (coherentes en notebook, informe y app)
CLASS_COLORS = {
    0: (40, 44, 52),       # Fondo   - gris carbón
    1: (186, 104, 255),    # Caminable - violeta
    2: (255, 92, 92),      # Construida - rojo coral
    3: (64, 220, 120),     # Zona verde - verde
    4: (255, 196, 0),      # Carretera - ámbar
}

# Protocolo de evaluación principal. True = los píxeles SIN anotación del GT se
# ignoran ("void", como en Cityscapes/ADE20K): en este dataset hay construcciones
# visibles que no fueron anotadas, así que "sin etiqueta" no siempre significa
# "ninguna clase". Las métricas del otro protocolo (estricto) se reportan siempre
# al lado, en res["alt"].
IGNORE_VOID = True

# Instancias más pequeñas que esto (en píxeles, a resolución original) se
# consideran ruido al generar instancias por componentes conexas.
MIN_INSTANCE_AREA = 64

# --------------------------------------------------------------------------
# Datos
# --------------------------------------------------------------------------
SEED = 42
SPLIT_RATIOS = (0.70, 0.15, 0.15)        # train / val / test
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --------------------------------------------------------------------------
# Hiperparámetros por defecto (se pueden sobreescribir por línea de comandos)
# --------------------------------------------------------------------------
MASK2FORMER_DEFAULTS = dict(
    checkpoint="facebook/mask2former-swin-base-coco-panoptic",
    crop_size=512,
    batch_size=4,
    epochs=80,
    lr=1e-4,
    backbone_lr_mult=0.1,
    weight_decay=0.05,
    crops_per_image=4,          # recortes aleatorios por imagen en cada época
    infer_size=1024,            # lado mayor en inferencia
)

EFFICIENTPS_DEFAULTS = dict(
    backbone="tf_efficientnet_b5.ns_jft_in1k",
    crop_size=512,
    batch_size=4,
    epochs=100,
    lr=3e-4,
    weight_decay=1e-4,
    crops_per_image=4,
    infer_size=1024,
)


# --------------------------------------------------------------------------
# Lectura/escritura de JSON siempre en UTF-8 (Windows usa otra codificación
# por defecto y rompía los nombres con tildes, p. ej. "Área").
# --------------------------------------------------------------------------
def read_json(path):
    import json
    raw = Path(path).read_bytes()
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return json.loads(raw.decode(enc))
        except UnicodeDecodeError:
            continue


def write_json(path, obj):
    import json
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
