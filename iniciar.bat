@echo off
TITLE Analizador de Voz - ESCOM
SETLOCAL EnableDelayedExpansion

echo ============================================================
echo   CONFIGURANDO E INICIANDO ENTORNO - ANALIZADOR DE VOZ
echo ============================================================
echo.

:: 1. Verificar si Python esta instalado
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python no esta instalado o no se encuentra en el PATH.
    echo Por favor, instala Python antes de continuar.
    pause
    exit /b
)

:: 2. Definir ruta del entorno virtual (.venv)
set VENV_DIR=%~dp0.venv

:: 3. Crear el entorno virtual si no existe
if not exist "%VENV_DIR%" (
    echo [INFO] No se encontro un entorno virtual. Creando uno nuevo en .venv...
    python -m venv "%VENV_DIR%"
    if !errorlevel! neq 0 (
        echo [ERROR] Hubo un problema al crear el entorno virtual.
        pause
        exit /b
    )
    echo [OK] Entorno virtual creado exitosamente.
    echo.
    
    echo [INFO] Activando entorno e instalando dependencias...
    call "%VENV_DIR%\Scripts\activate.bat"
    
    echo [INFO] Actualizando pip...
    python -m pip install --upgrade pip
    
    if exist "%~dp0requirements.txt" (
        echo [INFO] Instalando requerimientos desde requirements.txt...
        pip install -r "%~dp0requirements.txt"
        if !errorlevel! neq 0 (
            echo [ERROR] Error durante la instalacion de dependencias.
            pause
            exit /b
        )
        echo [OK] Dependencias instaladas correctamente.
    ) else (
        echo [WARN] No se encontro el archivo requirements.txt en la raiz.
    )
) else (
    echo [OK] Entorno virtual existente detectado. Activando...
    call "%VENV_DIR%\Scripts\activate.bat"
)

echo.
echo ============================================================
echo   LANZANDO APLICACION WEB (STREAMLIT)
echo ============================================================
echo.

:: 4. Ejecutar streamlit
streamlit run "%~dp0src/gui.py"

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] La aplicacion web se detuvo de forma inesperada.
    pause
)

deactivate