Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "   CONFIGURANDO E INICIANDO ENTORNO (POWERSHELL) " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Verificar si existe el entorno virtual
if (-not (Test-Path ".\.venv")) {
    Write-Host "[INFO] No se encontro entorno virtual. Creando uno nuevo en .venv..." -ForegroundColor Yellow
    python -m venv .venv
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] No se pudo crear el entorno virtual." -ForegroundColor Red
        Read-Host "Presiona Enter para salir"
        exit
    }
    
    Write-Host "[OK] Entorno virtual creado." -ForegroundColor Green
    Write-Host "[INFO] Activando e instalando dependencias..." -ForegroundColor Yellow
    
    # Activar entorno de forma nativa en PowerShell
    .\.venv\Scripts\Activate.ps1
    
    python -m pip install --upgrade pip
    pip install -r requirements.txt
} else {
    Write-Host "[OK] Entorno virtual existente detectado. Activando..." -ForegroundColor Green
    .\.venv\Scripts\Activate.ps1
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "   LANZANDO APLICACION WEB (STREAMLIT)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# 2. Ejecutar la app
streamlit run .\src\gui.py

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "   PROCESO TERMINADO" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Read-Host "Presiona Enter para cerrar esta ventana..."