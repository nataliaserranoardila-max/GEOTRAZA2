@echo off
chcp 65001 >nul
title GEOTRAZA - Instalacion
cd /d "%~dp0"
echo.
echo  ==============================================
echo    GEOTRAZA - Instalacion automatica
echo  ==============================================
echo.

echo %CD% | find /I "OneDrive" >nul && (
  echo  [!] El proyecto esta dentro de OneDrive. Va a funcionar, pero muy lento
  echo      porque OneDrive sincroniza miles de archivos. Recomendado: mover la
  echo      carpeta a C:\carto1\ y volver a ejecutar este archivo.
  echo.
  choice /M "  Continuar de todas formas"
  if errorlevel 2 exit /b
)

rem ---------- 1. Python (3.11 / 3.12 / 3.10) ----------
set "PYV="
call :findpy
if not defined PYV (
  echo  [1/5] Instalando Python 3.11 ...
  echo  ^(si te hace preguntas, responde Y o S y presiona Enter^)
  py install 3.11
  call :findpy
)
if not defined PYV (
  winget install -e --id Python.Python.3.11 --accept-source-agreements --accept-package-agreements
  call :findpy
)
if not defined PYV (
  echo.
  echo  [x] No pude instalar Python automaticamente.
  echo      Si arriba dice que se instalo, CIERRA esta ventana y vuelve a abrir instalar.bat.
  echo      Si no, instala Python 3.11 desde https://www.python.org/downloads/release/python-3119/
  echo      ^(boton "Windows installer 64-bit", y marca "Add python.exe to PATH"^)
  pause & exit /b 1
)
echo  [1/5] Python %PYV% OK

rem ---------- 2. Entorno virtual ----------
if not exist ".venv\Scripts\python.exe" (
  echo  [2/5] Creando entorno virtual .venv ...
  py -%PYV% -m venv .venv || (echo  [x] Error creando el entorno & pause & exit /b 1)
)
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip -q
echo  [2/5] Entorno virtual OK

rem ---------- 3. PyTorch (GPU si hay NVIDIA) ----------
python -c "import torch" >nul 2>nul
if errorlevel 1 (
  nvidia-smi >nul 2>nul
  if errorlevel 1 (
    echo  [3/5] No se detecto GPU NVIDIA: instalando PyTorch para CPU ...
    pip install torch torchvision
  ) else (
    echo  [3/5] GPU NVIDIA detectada: instalando PyTorch con CUDA ^(~2.5 GB, paciencia^) ...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
  )
)
echo  [3/5] PyTorch OK

rem ---------- 4. Resto de librerias ----------
echo  [4/5] Instalando librerias del proyecto ...
pip install -r requirements.txt -q || (echo  [x] Error instalando requirements & pause & exit /b 1)

rem ---------- 5. Dataset ----------
if not exist "data\raw\Dataset segmentacion" (
  if not exist "%USERPROFILE%\.kaggle\kaggle.json" if exist "%USERPROFILE%\Downloads\kaggle.json" (
    mkdir "%USERPROFILE%\.kaggle" 2>nul
    copy "%USERPROFILE%\Downloads\kaggle.json" "%USERPROFILE%\.kaggle\kaggle.json" >nul
  )
  if exist "%USERPROFILE%\.kaggle\kaggle.json" (
    echo  [5/5] Descargando el dataset de Kaggle ...
    kaggle datasets download -d cetaamb/reto-1-cartografa-automatizada -p data\raw --unzip
  ) else (
    echo  [5/5] No encontre kaggle.json en Descargas. La galeria de escenas quedara vacia
    echo        ^(la app funciona igual subiendo imagenes^).
  )
) else (
  echo  [5/5] Dataset OK
)

echo.
echo  ---------------- Verificacion ----------------
python -c "import torch; print('  GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no (se usara CPU)')"
if exist "models\weights\mask2former_best.pt" (echo   Mask2Former: OK) else (echo   [!] Falta models\weights\mask2former_best.pt  ^(descomprime carto1_resultados.zip aqui^))
if exist "models\weights\efficientps_best.pt" (echo   EfficientPS: OK) else (echo   [!] Falta models\weights\efficientps_best.pt)
if exist "data\raw\Dataset segmentacion" (echo   Dataset: OK) else (echo   Dataset: no)
echo.
echo  Listo. Para abrir la app haz doble clic en  iniciar.bat
echo.
pause
exit /b 0

:findpy
for %%V in (3.11 3.12 3.10) do (
  if not defined PYV (
    py -%%V -c "import sys" >nul 2>nul && set "PYV=%%V"
  )
)
exit /b 0
