#!/bin/bash

# Installation script for ComfyUI Flame Integration
echo "================================================"
echo "ComfyUI Flame Integration - Dependency Installer"
echo "================================================"
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "✗ pip3 not found. Please install pip first."
    exit 1
fi

echo "✓ pip3 is available"
echo ""

# Create virtual environment (optional but recommended)
read -p "Create virtual environment? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✓ Virtual environment created and activated"
fi

# Install dependencies
echo ""
echo "Installing dependencies from requirements.txt..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ All dependencies installed successfully!"
else
    echo ""
    echo "✗ Some dependencies failed to install. Please check the error messages above."
    exit 1
fi

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p workflows
mkdir -p /tmp/flame_comfyui
mkdir -p /tmp/comfyui_cache

echo "✓ Directories created"

# Summary
echo ""
echo "================================================"
echo "Installation Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Ensure ComfyUI is running on localhost:8188"
echo "2. Copy workflow JSON files to ./workflows directory"
echo "3. For Flame integration:"
echo "   - Copy flame_comfyui_pybox.py to your Flame Python hooks directory"
echo "   - Usually: /opt/Autodesk/<flame_version>/python/"
echo "4. Run the GUI: python3 comfyui_workflow_manager.py"
echo ""
echo "Documentation: See README.md for detailed usage instructions"
echo "================================================"
