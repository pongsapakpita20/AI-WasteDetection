@echo off
REM ============================================
REM AI Waste Sorter - DVC Pipeline Runner
REM ============================================
echo.
echo ============================================
echo   AI Waste Sorter Pipeline
echo ============================================
echo.

REM ตรวจสอบว่า DVC ติดตั้งแล้วหรือยัง
where dvc >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] DVC not found. Please install DVC first.
    echo Run: pip install dvc
    pause
    exit /b 1
)

REM 1. Pull dataset จาก DVC storage (ถ้ามี)
echo [1/4] Pulling dataset from DVC storage...
dvc pull waste-detection.dvc
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Failed to pull dataset. Continuing anyway...
)

REM 2. รัน DVC pipeline (train + promote + evaluate)
echo.
echo [2/4] Running DVC pipeline (train + promote + evaluate)...
echo This may take a long time depending on your GPU/CPU...
dvc repro evaluate
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Pipeline failed!
    pause
    exit /b 1
)

REM 3. Push artifacts ไป DVC storage
echo.
echo [3/4] Pushing artifacts to DVC storage...
dvc push
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Failed to push artifacts. Check your DVC remote configuration.
)

REM 4. แจ้งผลลัพธ์
echo.
echo [4/4] Pipeline completed successfully!
echo.
echo Results:
echo   - Model: artifacts\models\waste-sorter-best.pt
echo   - Metrics: artifacts\eval\metrics.json
echo.
pause

