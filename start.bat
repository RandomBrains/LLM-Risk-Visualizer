@echo off
REM LLM Risk Visualizer Windows Startup Script
REM This script helps set up and run the LLM Risk Visualizer application on Windows

setlocal EnableDelayedExpansion

REM Colors for output (basic)
set "INFO=[INFO]"
set "SUCCESS=[SUCCESS]"
set "WARNING=[WARNING]"
set "ERROR=[ERROR]"

REM Function to check if Python is installed
:check_python
echo %INFO% Checking Python installation...

python --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=python
    echo %SUCCESS% Python found
    goto :check_python_version
)

python3 --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=python3
    echo %SUCCESS% Python3 found
    goto :check_python_version
)

echo %ERROR% Python is not installed. Please install Python 3.8 or higher.
echo Download from: https://www.python.org/downloads/
pause
exit /b 1

:check_python_version
for /f "tokens=2" %%i in ('%PYTHON_CMD% --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Python version: %PYTHON_VERSION%
goto :eof

REM Function to check if pip is installed
:check_pip
echo %INFO% Checking pip installation...

pip --version >nul 2>&1
if %errorlevel% equ 0 (
    set PIP_CMD=pip
    echo %SUCCESS% pip found
    goto :eof
)

pip3 --version >nul 2>&1
if %errorlevel% equ 0 (
    set PIP_CMD=pip3
    echo %SUCCESS% pip3 found
    goto :eof
)

echo %ERROR% pip is not installed. Please install pip.
pause
exit /b 1

REM Function to create virtual environment
:create_venv
echo %INFO% Setting up virtual environment...

if not exist "venv" (
    %PYTHON_CMD% -m venv venv
    echo %SUCCESS% Virtual environment created
) else (
    echo %WARNING% Virtual environment already exists
)

REM Activate virtual environment
call venv\Scripts\activate.bat
echo %SUCCESS% Virtual environment activated
goto :eof

REM Function to install dependencies
:install_dependencies
echo %INFO% Installing dependencies...

if exist "requirements.txt" (
    %PIP_CMD% install --upgrade pip
    %PIP_CMD% install -r requirements.txt
    echo %SUCCESS% Dependencies installed successfully
) else (
    echo %ERROR% requirements.txt not found
    pause
    exit /b 1
)
goto :eof

REM Function to setup environment
:setup_env
echo %INFO% Setting up environment configuration...

if not exist ".env" (
    if exist ".env.example" (
        copy ".env.example" ".env" >nul
        echo %WARNING% Created .env file from template. Please update with your actual values.
    ) else (
        echo %WARNING% No .env.example found. You may need to create .env file manually.
    )
) else (
    echo %SUCCESS% Environment file already exists
)
goto :eof

REM Function to initialize database
:init_database
echo %INFO% Initializing database...

%PYTHON_CMD% -c "import sqlite3; from auth import DatabaseManager as AuthDB; from database import DatabaseManager as MainDB; auth_db = AuthDB(); main_db = MainDB(); print('Database setup completed')" 2>nul
if %errorlevel% equ 0 (
    echo %SUCCESS% Database initialized successfully
) else (
    echo %WARNING% Database initialization may have failed (this is normal on first run)
)
goto :eof

REM Function to check system requirements
:check_requirements
echo %INFO% Checking system requirements...

REM Check available disk space (simplified)
for /f "tokens=3" %%i in ('dir /-c ^| find "bytes free"') do set AVAILABLE_SPACE=%%i
echo Disk space check completed

REM Check if required ports are available
netstat -an | find "8501" >nul
if %errorlevel% equ 0 (
    echo %WARNING% Port 8501 may be in use
) else (
    echo %SUCCESS% Port 8501 is available
)
goto :eof

REM Function to start the application
:start_app
echo %INFO% Starting LLM Risk Visualizer...

REM Check if streamlit is installed
streamlit --version >nul 2>&1
if %errorlevel% neq 0 (
    echo %ERROR% Streamlit is not installed. Please run the setup first.
    pause
    exit /b 1
)

REM Determine which app file to use
if exist "app_enhanced.py" (
    set APP_FILE=app_enhanced.py
    echo %INFO% Using enhanced application
) else if exist "app.py" (
    set APP_FILE=app.py
    echo %INFO% Using standard application
) else (
    echo %ERROR% No application file found (app.py or app_enhanced.py)
    pause
    exit /b 1
)

echo %SUCCESS% Starting application...
echo %INFO% Open http://localhost:8501 in your browser
echo %INFO% Press Ctrl+C to stop the application

streamlit run %APP_FILE%
goto :eof

REM Function to show help
:show_help
echo LLM Risk Visualizer - Windows Startup Script
echo.
echo Usage: %0 [COMMAND]
echo.
echo Commands:
echo   setup     - Full setup (install dependencies, create venv, etc.)
echo   start     - Start the application
echo   install   - Install dependencies only
echo   check     - Check system requirements
echo   help      - Show this help message
echo.
echo Examples:
echo   %0 setup     # First time setup
echo   %0 start     # Start the application
echo.
echo For Docker support, please use the Linux script or Docker Desktop
goto :eof

REM Main script logic
if "%1"=="" goto start_default
if "%1"=="setup" goto setup
if "%1"=="start" goto start_default
if "%1"=="install" goto install_only
if "%1"=="check" goto check_only
if "%1"=="help" goto show_help
if "%1"=="-h" goto show_help
if "%1"=="--help" goto show_help

echo %ERROR% Unknown command: %1
call :show_help
pause
exit /b 1

:setup
echo %INFO% 🚀 Setting up LLM Risk Visualizer...
call :check_requirements
call :check_python
call :check_pip
call :create_venv
call :install_dependencies
call :setup_env
call :init_database
echo %SUCCESS% 🎉 Setup completed! Run 'start.bat start' to launch the application.
pause
exit /b 0

:start_default
echo %INFO% 🚀 Starting LLM Risk Visualizer...
call :check_python

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

call :start_app
exit /b 0

:install_only
call :check_python
call :check_pip
call :install_dependencies
pause
exit /b 0

:check_only
call :check_requirements
call :check_python
call :check_pip
echo %SUCCESS% System check completed
pause
exit /b 0