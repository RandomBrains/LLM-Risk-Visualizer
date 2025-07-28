#!/bin/bash

# LLM Risk Visualizer Startup Script
# This script helps set up and run the LLM Risk Visualizer application

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is installed
check_python() {
    print_status "Checking Python installation..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
        print_success "Python3 found: $(python3 --version)"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
        print_success "Python found: $(python --version)"
    else
        print_error "Python is not installed. Please install Python 3.8 or higher."
        exit 1
    fi
    
    # Check Python version
    PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [ $(echo "$PYTHON_VERSION >= 3.8" | bc -l) -eq 0 ]; then
        print_error "Python 3.8 or higher is required. Found: $PYTHON_VERSION"
        exit 1
    fi
}

# Check if pip is installed
check_pip() {
    print_status "Checking pip installation..."
    
    if command -v pip3 &> /dev/null; then
        PIP_CMD="pip3"
        print_success "pip3 found"
    elif command -v pip &> /dev/null; then
        PIP_CMD="pip"
        print_success "pip found"
    else
        print_error "pip is not installed. Please install pip."
        exit 1
    fi
}

# Create virtual environment
create_venv() {
    print_status "Setting up virtual environment..."
    
    if [ ! -d "venv" ]; then
        $PYTHON_CMD -m venv venv
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        # Windows (Git Bash)
        source venv/Scripts/activate
    else
        # Linux/macOS
        source venv/bin/activate
    fi
    
    print_success "Virtual environment activated"
}

# Install dependencies
install_dependencies() {
    print_status "Installing dependencies..."
    
    if [ -f "requirements.txt" ]; then
        $PIP_CMD install --upgrade pip
        $PIP_CMD install -r requirements.txt
        print_success "Dependencies installed successfully"
    else
        print_error "requirements.txt not found"
        exit 1
    fi
}

# Setup environment variables
setup_env() {
    print_status "Setting up environment configuration..."
    
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
            print_warning "Created .env file from template. Please update with your actual values."
        else
            print_warning "No .env.example found. You may need to create .env file manually."
        fi
    else
        print_success "Environment file already exists"
    fi
}

# Initialize database
init_database() {
    print_status "Initializing database..."
    
    $PYTHON_CMD -c "
import sqlite3
from auth import DatabaseManager as AuthDB
from database import DatabaseManager as MainDB

# Initialize authentication database
auth_db = AuthDB()
print('✅ Authentication database initialized')

# Initialize main database
main_db = MainDB()
print('✅ Main database initialized')

print('Database setup completed successfully')
" 2>/dev/null || print_warning "Database initialization may have failed (this is normal on first run)"
}

# Check system requirements
check_requirements() {
    print_status "Checking system requirements..."
    
    # Check available disk space (in GB)
    if command -v df &> /dev/null; then
        AVAILABLE_SPACE=$(df . | tail -1 | awk '{print $4}')
        AVAILABLE_GB=$((AVAILABLE_SPACE / 1024 / 1024))
        
        if [ $AVAILABLE_GB -lt 1 ]; then
            print_warning "Low disk space: ${AVAILABLE_GB}GB available. At least 1GB recommended."
        else
            print_success "Disk space: ${AVAILABLE_GB}GB available"
        fi
    fi
    
    # Check memory (Linux/macOS)
    if command -v free &> /dev/null; then
        MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
        if [ $MEMORY_GB -lt 2 ]; then
            print_warning "Low memory: ${MEMORY_GB}GB available. At least 2GB recommended."
        else
            print_success "Memory: ${MEMORY_GB}GB available"
        fi
    fi
}

# Start the application
start_app() {
    print_status "Starting LLM Risk Visualizer..."
    
    # Check if streamlit is installed
    if ! command -v streamlit &> /dev/null; then
        print_error "Streamlit is not installed. Please run the setup first."
        exit 1
    fi
    
    # Determine which app file to use
    if [ -f "app_enhanced.py" ]; then
        APP_FILE="app_enhanced.py"
        print_status "Using enhanced application"
    elif [ -f "app.py" ]; then
        APP_FILE="app.py"
        print_status "Using standard application"
    else
        print_error "No application file found (app.py or app_enhanced.py)"
        exit 1
    fi
    
    print_success "Starting application..."
    print_status "Open http://localhost:8501 in your browser"
    print_status "Press Ctrl+C to stop the application"
    
    streamlit run $APP_FILE
}

# Docker operations
build_docker() {
    print_status "Building Docker image..."
    
    if [ ! -f "Dockerfile" ]; then
        print_error "Dockerfile not found"
        exit 1
    fi
    
    docker build -t llm-risk-visualizer .
    print_success "Docker image built successfully"
}

run_docker() {
    print_status "Running Docker container..."
    
    docker run -p 8501:8501 llm-risk-visualizer
}

# Show help
show_help() {
    echo "LLM Risk Visualizer - Startup Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  setup     - Full setup (install dependencies, create venv, etc.)"
    echo "  start     - Start the application"
    echo "  install   - Install dependencies only"
    echo "  docker    - Build and run with Docker"
    echo "  check     - Check system requirements"
    echo "  help      - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 setup     # First time setup"
    echo "  $0 start     # Start the application"
    echo "  $0 docker    # Use Docker"
    echo ""
}

# Main script logic
main() {
    case "${1:-start}" in
        "setup")
            print_status "🚀 Setting up LLM Risk Visualizer..."
            check_requirements
            check_python
            check_pip
            create_venv
            install_dependencies
            setup_env
            init_database
            print_success "🎉 Setup completed! Run './start.sh start' to launch the application."
            ;;
        "start")
            print_status "🚀 Starting LLM Risk Visualizer..."
            check_python
            
            # Activate virtual environment if it exists
            if [ -d "venv" ]; then
                if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
                    source venv/Scripts/activate
                else
                    source venv/bin/activate
                fi
            fi
            
            start_app
            ;;
        "install")
            check_python
            check_pip
            install_dependencies
            ;;
        "docker")
            print_status "🐳 Using Docker..."
            build_docker
            run_docker
            ;;
        "check")
            check_requirements
            check_python
            check_pip
            print_success "System check completed"
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"