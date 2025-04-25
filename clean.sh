#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Cleaning project..."

# Remove __pycache__ directories
echo "Removing __pycache__ directories..."
find . -type d -name "__pycache__" -exec rm -r {} +

# Remove .pyc files
echo "Removing .pyc files..."
find . -type f -name "*.pyc" -delete

# Remove build artifacts
echo "Removing build artifacts..."
rm -rf build/ dist/ *.egg-info

echo "Linting and formatting..."

# Check if uv is installed
if ! command -v uv &> /dev/null
then
    echo "uv command not found. Please install uv: https://github.com/astral-sh/uv"
    exit 1
fi

# Check if black and flake8 are installed, install if not
if ! black --version &> /dev/null || ! flake8 --version &> /dev/null || ! isort --version &> /dev/null || ! autoflake --version &> /dev/null
then
    echo "Installing/updating formatting tools using uv..."
    uv add --dev black flake8 isort autoflake
fi

# Sort imports with isort
echo "Running isort..."
if ! isort . --profile black --skip .venv ; then
    echo "Error running isort. Please ensure it's properly installed in your current Python environment"
    exit 1
fi

# Format code with black
echo "Running black..."
if ! black . --exclude=".venv" ; then
    echo "Error running black. Please ensure it's properly installed in your current Python environment"
    exit 1
fi

# Auto-fix some flake8 issues with autoflake
echo "Running autoflake..."
set +e  # Temporarily disable exit on error
autoflake --recursive --in-place --remove-all-unused-imports --remove-unused-variables --exclude .venv .
set -e  # Re-enable exit on error

# Lint code with flake8
echo "Running flake8..."
# Run flake8 but don't exit on linting errors
set +e  # Temporarily disable exit on error
flake8 . --exclude=.venv
FLAKE8_EXIT_CODE=$?
set -e  # Re-enable exit on error

if [ $FLAKE8_EXIT_CODE -eq 1 ]; then
    echo "Flake8 found style issues. Some may have been auto-fixed, please review remaining issues."
elif [ $FLAKE8_EXIT_CODE -gt 1 ]; then
    echo "Error running flake8. Please ensure it's properly installed in your current Python environment"
    exit 1
fi

echo "Cleaning and linting complete."
