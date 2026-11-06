@echo off
chcp 65001 >nul
title GEOTRAZA - Estacion de analisis orbital
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
  echo Primero ejecuta instalar.bat
  pause & exit /b 1
)
call ".venv\Scripts\activate.bat"
echo.
echo  Cargando modelos... el navegador se abrira solo en unos segundos.
echo  Para cerrar la app, cierra esta ventana.
echo.
start "" /min cmd /c "timeout /t 20 >nul & start http://127.0.0.1:8000"
python main.py
pause
