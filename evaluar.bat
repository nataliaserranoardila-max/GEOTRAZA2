@echo off
chcp 65001 >nul
title GEOTRAZA - Reproducir metricas
cd /d "%~dp0"
call ".venv\Scripts\activate.bat" || (echo Primero ejecuta instalar.bat & pause & exit /b 1)
echo.
echo  Reproduciendo las metricas de test de ambos modelos (unos minutos)...
echo.
for %%M in (mask2former efficientps) do (
  for %%D in (native tta) do (
    python evaluate.py --weights models\weights\%%M_best.pt --mode %%D
  )
)
echo.
echo  Generando tablas y figuras del informe...
python report_figures.py --n 5
echo.
echo  Listo. Resultados en la carpeta outputs\  (la app los muestra al reiniciarla)
pause
