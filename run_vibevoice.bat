@echo off
REM VibeVoice Windows Launcher
REM This script activates the virtual environment and launches VibeVoice

echo Starting VibeVoice...

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo Error: Virtual environment not found!
    echo Please run the installation instructions first to create a virtual environment.
    echo.
    echo To create a virtual environment:
    echo   python -m venv venv
    echo   venv\Scripts\activate
    echo   pip install -e .
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if .env file exists
if not exist ".env" (
    echo Warning: .env file not found!
    echo Please copy .env-sample to .env and add your API keys:
    echo   copy .env-sample .env
    echo.
)

REM Launch VibeVoice
echo Launching VibeVoice...
echo.
echo ========================================
echo VibeVoice will be available at:
echo http://localhost:7590
echo ========================================
echo.
echo Note: If FlashAttention2 is not available, VibeVoice will
echo automatically fall back to SDPA (Scaled Dot Product Attention).
echo This ensures compatibility across different hardware configurations.
echo.

REM Standard mode (loads model on startup)
python main.py

REM Load-on-demand mode (faster startup, loads model when needed)
REM Uncomment the line below and comment out the line above to use load-on-demand mode:
REM python main.py --lod

echo.
echo VibeVoice has stopped.
pause
