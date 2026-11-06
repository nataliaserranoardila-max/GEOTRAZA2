# 🛰️ GEOTRAZA — Cartografía automatizada con segmentación panóptica

*Misión GEOTRAZA-01 de la Agencia Orbital Chicamocha (agencia ficticia del proyecto). Equipo: Natalia Serrano Ardila y Dariana Andrea Torrecilla.*

Taller 3 · 3.er corte — Electiva IV *Deep Computer Vision* · Ingeniería Mecatrónica, USTA Bucaramanga

Dos arquitecturas de **segmentación panóptica** — **Mask2Former** y **EfficientPS** — entrenadas sobre imágenes satelitales RGB para identificar **Área Caminable, Área Construida, Zona Verde y Carretera**, evaluadas con IoU, mAP, F1, precisión (y PQ), y desplegadas en una aplicación web interactiva (**GEOTRAZA**) que compara ambos modelos sobre imágenes nuevas.

| Flujo | Dónde |
|---|---|
| Entrenamiento (GPU) | Google Colab → `notebooks/Entrenamiento_Colab_CARTO1.ipynb` |
| App web + evaluación | Tu PC (CPU o GPU) → `python main.py` |

---

## 1. Estructura del proyecto

```
cartografia-automatizada/
├── config.py                 # clases, colores, rutas, splits, hiperparámetros
├── main.py                   # servidor web FastAPI (la app)
├── train.py                  # entrenamiento de ambos modelos (--model ...)
├── evaluate.py               # reproduce las métricas en test/val
├── report_figures.py         # tablas y gráficas para el informe Word
├── data/
│   ├── dataset.py            # carga, preprocesamiento, aumento, Dataset de PyTorch
│   ├── prepare_data.py       # splits + estadísticas + figuras del dataset
│   └── raw/                  # aquí va el dataset de Kaggle (no se versiona)
├── models/
│   ├── mask2former_model.py  # Mask2Former (Hugging Face) + inferencia panóptica
│   ├── efficientps.py        # EfficientPS en PyTorch: 2-way FPN, LSFE/DPC/MC, Mask R-CNN, fusión
│   ├── common.py             # TTA, preparación de imagen, Predictor base
│   ├── postprocess.py        # salida panóptica unificada
│   ├── predictor.py          # carga cualquier .pt
│   └── weights/              # mask2former_best.pt, efficientps_best.pt
├── utils/
│   ├── metrics.py            # IoU, precisión, recall, F1, mAP (COCO), PQ
│   ├── visualization.py      # colores, overlays, mapas de error/confianza
│   └── plots.py              # curvas de pérdida, barras, matrices de confusión
├── templates/index.html      # interfaz
├── static/css/style.css
├── static/js/app.js
├── notebooks/Entrenamiento_Colab_CARTO1.ipynb
├── outputs/                  # métricas JSON, curvas, figuras (lo llena el entrenamiento)
├── requirements.txt          # PC
└── requirements-colab.txt    # Colab
```

## 2. Preparación de datos

Dataset: [`cetaamb/reto-1-cartografa-automatizada`](https://www.kaggle.com/datasets/cetaamb/reto-1-cartografa-automatizada) — 100 recortes con ground truth (`recorte_XX.png` + `ground_truth_recorte_XX.npy` de forma H×W×4) y 80 recortes adicionales sin etiqueta (se usan para la demo de la app).

Descomprime el zip en `data/raw/` para que quede `data/raw/Dataset segmentacion/` y `data/raw/Recortes adicionales/`, o hazlo automático (necesita `~/.kaggle/kaggle.json`):

```bash
python data/prepare_data.py --download     # descarga + splits + estadísticas
python data/prepare_data.py                # si ya descomprimiste a mano
```

Qué hace el preprocesamiento (`data/dataset.py`):

1. **Solapes del GT.** Las 4 máscaras pueden marcar el mismo píxel (árboles sobre andenes). Se resuelve con la prioridad *Zona Verde > Carretera > Área Caminable > Área Construida* (lo visible desde arriba). Píxeles sin etiqueta = **Fondo**.
2. **Panóptico.** *Things*: Área Construida y Zona Verde, una instancia por región conexa (8-vecindad, área ≥ 64 px). *Stuff*: Área Caminable, Carretera y Fondo.
3. **Split** 70/15/15 con semilla 42 → `data/splits.json` (se versiona: test siempre son las mismas imágenes).
4. **Aumento** (solo train): escala 0.75–1.5, recorte 512×512, 8 simetrías D4, brillo/contraste/saturación/tono y desenfoque leve.

## 3. Cómo entrenar (Google Colab)

1. Comprime esta carpeta como `cartografia-automatizada.zip` y súbela a la carpeta `DEEP COMPUTER` de tu Google Drive (si la pones en otro lado, el notebook la busca solo).
2. Abre `notebooks/Entrenamiento_Colab_CARTO1.ipynb` en Colab con **GPU**.
3. Ejecuta las celdas en orden. Te pedirá tu `kaggle.json`.
4. Al final descarga `carto1_resultados.zip` (también queda copiado en Drive).

Tiempos aproximados en una T4: Mask2Former 80 épocas ≈ 1–1.5 h; EfficientPS 120 épocas ≈ 1–1.5 h. Cada época se guarda en `DEEP COMPUTER/carto_ckpt/` en Drive; si Colab se desconecta, vuelve a ejecutar las celdas y sigue con `--resume`.

Mismo entrenamiento desde la terminal (Colab o un PC con GPU):

```bash
python train.py --model mask2former --epochs 80  --batch-size 4 --resume
python train.py --model efficientps --epochs 100 --batch-size 4 --resume   # peso ×2 a Área Caminable por defecto
```

Opciones útiles: `--batch-size 2 --accum 2` (poca memoria), `--no-amp` (si aparecen pérdidas NaN), `--backbone tf_efficientnet_b4.ns_jft_in1k`, `--checkpoint facebook/mask2former-swin-large-coco-panoptic` (si tienes A100), `--limit 8` (prueba rápida del código).

## 4. Cómo ejecutar el proyecto en tiempo real (PC)

**Forma rápida (Windows):** descomprime `carto1_resultados.zip` dentro de esta carpeta, luego doble clic en **`instalar.bat`** (instala Python 3.11, PyTorch con CUDA si hay GPU NVIDIA, las librerías y el dataset si encuentra `kaggle.json` en Descargas) y después doble clic en **`iniciar.bat`** para abrir la app.

**Forma manual:**

Requisitos: Python 3.10 o superior.

```bash
# 1) entorno
python -m venv .venv
.venv\Scripts\activate              # Windows   (Linux/Mac: source .venv/bin/activate)
pip install torch torchvision      # o la variante CUDA de https://pytorch.org
pip install -r requirements.txt

# 2) resultados de Colab: descomprime carto1_resultados.zip en esta carpeta
#    quedan models/weights/*.pt, outputs/ y data/splits.json

# 3) (recomendado) dataset en data/raw/ para la galería de escenas y métricas en vivo

# 4) lanzar la app
python main.py                      # abre http://127.0.0.1:8000
python main.py --infer-size 768     # más rápido en CPU
```

### Uso de la interfaz

La página está pensada para seguir la presentación de 10 minutos de la Parte 4, de arriba abajo: **Misión** (equipo y mejores métricas) → **Datos** (cifras del dataset, distribución de clases, preprocesamiento y ejemplos de GT) → **Entrenamiento** (arquitecturas, hiperparámetros, curvas de pérdida train/validación) → **Análisis en vivo** (inferencia) → **Resultados** (tabla comparativa, protocolos, ajuste del post-procesamiento, matrices de confusión y ejemplos). Los nombres del equipo se editan en `config.py` (`TEAM`).

- **Nueva imagen**: arrastra o elige un PNG/JPG/TIFF. Opcionalmente adjunta su ground truth `.npy` para calcular métricas en vivo.
- **Archivo de escenas**: imágenes del dataset por split. Las de **test** traen GT, así que la app calcula IoU, mAP, F1, precisión y PQ de esa escena para ambos modelos.
- **Modo**: *Nativo* (cabeza panóptica de cada modelo) o *TTA ×8* (más preciso, ~8× más lento).
- **Visor**: pestañas Mask2Former / EfficientPS / **Comparar** (uno a cada lado del deslizador) / Ground truth; vistas **Semántica**, **Panóptica** (cada instancia con su tono), **Confianza** y **Errores vs GT**. Las clases se activan o desactivan desde la leyenda; el cursor muestra clase, confianza e instancia de cada píxel.
- **Telemetría**: métricas con chequeo contra los mínimos del taller, cobertura por clase, conteo de instancias, tiempo de inferencia y mapa de desacuerdo entre modelos.
- **Banco de pruebas**: tabla comparativa del split de test y curvas de pérdida/validación del entrenamiento.
- **Guardar vista / reporte**: PNG de lo que se ve y JSON con todas las métricas de la escena.

## 5. Cómo reproducir las métricas

**Windows (rápido):** doble clic en **`evaluar.bat`**: evalúa los dos modelos en test (modo nativo y TTA) y regenera tablas y figuras en `outputs/`.

**Manual:**

```bash
python evaluate.py --weights models/weights/mask2former_best.pt --mode native
python evaluate.py --weights models/weights/mask2former_best.pt --mode tta
python evaluate.py --weights models/weights/efficientps_best.pt --mode native
python evaluate.py --weights models/weights/efficientps_best.pt --mode tta
python report_figures.py           # tablas CSV/Markdown + gráficas para el Word
```

Definición de las métricas (`utils/metrics.py`):

| Métrica | Cálculo |
|---|---|
| IoU (mIoU) | TP/(TP+FP+FN) por clase sobre todos los píxeles de test, promedio de las 4 clases |
| Precisión | TP/(TP+FP) por clase, promedio |
| F1 | 2·P·R/(P+R) por clase (equivale a Dice), promedio |
| mAP | estilo COCO: AP de máscara en IoU 0.50:0.05:0.95, 101 puntos, promedio de clases. Instancia GT = región conexa de cada clase; predicción = segmento panóptico con su confianza |
| mAP50 | igual, solo umbral IoU 0.5 |
| mAP things | mAP solo sobre las clases *thing* (Área Construida, Zona Verde), convención de COCO/Cityscapes panoptic: el AP se define para objetos contables, no para regiones amorfas como la red vial |
| PQ | Panoptic Quality = SQ·RQ con emparejamiento IoU > 0.5 |

Mínimos del taller: **IoU ≥ 0.70, mAP ≥ 0.50, F1 ≥ 0.80**.

### Dos protocolos de evaluación (se reportan siempre ambos)

| Protocolo | Píxeles sin anotar en el GT | Uso |
|---|---|---|
| **void** (principal, `config.IGNORE_VOID = True`) | se ignoran: no cuentan como acierto ni como error; los segmentos predichos que caen > 50 % sobre esa zona se descartan | estándar en Cityscapes / ADE20K |
| **estricto** | cuentan como clase *Fondo*: predecir una clase ahí es un error | columna "estricto" en tablas y app |

Motivo: en el dataset hay construcciones claramente visibles que no fueron anotadas (p. ej. galpones industriales), así que "sin etiqueta" no siempre significa "ninguna de las 4 clases". La matriz de confusión del protocolo estricto lo muestra: ~40 % de los píxeles sin anotar se predicen como Área Construida.

### Ajuste del post-procesamiento (sin reentrenar)

`python tune_postprocess.py` busca en **validación** (nunca en test) el suavizado de probabilidades, el área mínima de las regiones predichas y el puntaje de ranking que maximizan (mIoU + mF1 + mAP)/3, y lo guarda en `outputs/postprocess.json`. Después `evaluate.py`, `report_figures.py` y la app lo usan automáticamente. El ground truth nunca se modifica.

## 6. Arquitecturas

**Mask2Former** (Cheng et al., CVPR 2022). Swin-B → pixel decoder multiescala con *deformable attention* → decodificador transformer con *masked attention* y 100 consultas, cada una predice (clase, máscara). Pesos iniciales: COCO-panoptic; se reemplaza la cabeza de clases por 5 clases. Inferencia panóptica: consultas con confianza > 0.5, asignación por argmax de score·máscara, se descartan máscaras tapadas en > 20 %.

**EfficientPS** (Mohan & Valada, IJCV 2021), implementado aquí en PyTorch. EfficientNet-B5 → **2-way FPN** (top-down + bottom-up) → cabeza semántica (**LSFE** en 1/4 y 1/8, **DPC** en 1/16 y 1/32, **MC** entre escalas) + cabeza de instancias **Mask R-CNN** → **fusión panóptica** (promedio de probabilidad de máscara y probabilidad semántica de la clase dentro de la caja; el resto de píxeles sale de la cabeza semántica). Cambios frente al paper por el tamaño de batch: GroupNorm en vez de iABN sincronizado y BN del backbone congelado.

## 7. Problemas comunes

| Síntoma | Solución |
|---|---|
| `CUDA out of memory` en Colab | `--batch-size 2 --accum 2` |
| `pérdida no finita` muchas veces | añade `--no-amp` |
| La app dice "sin pesos" | revisa que existan `models/weights/mask2former_best.pt` y `efficientps_best.pt` |
| La galería está vacía | descomprime el dataset en `data/raw/` |
| Inferencia lenta en el PC | `python main.py --infer-size 768` o usa el modo *Nativo* |
| Tipografías distintas | la app usa Google Fonts; sin internet cae a fuentes del sistema (funciona igual) |

## Pesos de los modelos (.pt)

Los archivos `mask2former_best.pt` (≈ 204 MB) y `efficientps_best.pt` (≈ 89 MB) superan el límite de 100 MB por archivo de GitHub, por eso están publicados en la sección **Releases** de este repositorio. Descárgalos y colócalos en `models/weights/` antes de ejecutar `iniciar.bat` o `evaluate.py`.
