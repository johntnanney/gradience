#!/bin/bash
set -euo pipefail

# Gradience Development Environment Setup
# 
# This script sets up a Python virtual environment and installs dependencies
# from the repo root directory to prevent common installation pitfalls.
#
# Usage: ./scripts/setup_venv.sh

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Setting up Gradience development environment${NC}"
echo

# Check that we're in the repo root (directory containing pyproject.toml)
if [[ ! -f "pyproject.toml" ]]; then
    echo -e "${RED}❌ Error: pyproject.toml not found${NC}"
    echo "   This script must be run from the repo root directory."
    echo "   Expected directory structure:"
    echo "     /path/to/gradience/           <- You should be here"
    echo "     ├── pyproject.toml           <- This file should exist"
    echo "     ├── gradience/               <- Python package directory"
    echo "     └── scripts/setup_venv.sh    <- This script"
    echo
    echo "   Current directory: $(pwd)"
    exit 1
fi

echo -e "${GREEN}✅ Found pyproject.toml - in correct directory${NC}"

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
echo -e "${BLUE}🐍 Using Python: ${python_version}${NC}"

# Create virtual environment if it doesn't exist
if [[ ! -d ".venv" ]]; then
    echo -e "${BLUE}📦 Creating virtual environment in .venv/${NC}"
    python3 -m venv .venv
else
    echo -e "${YELLOW}📦 Virtual environment .venv/ already exists${NC}"
fi

# Activate virtual environment
echo -e "${BLUE}🔌 Activating virtual environment${NC}"
source .venv/bin/activate

# Upgrade pip
echo -e "${BLUE}⬆️  Upgrading pip${NC}"
python -m pip install --upgrade pip

# Install the package in editable mode with dependencies
echo -e "${BLUE}📚 Installing Gradience in editable mode${NC}"
pip install -e .

echo
echo -e "${GREEN}🎉 Setup complete!${NC}"
echo
echo "To activate your environment in the future, run:"
echo -e "  ${BLUE}source .venv/bin/activate${NC}"
echo
echo "To verify your installation:"
echo -e "  ${BLUE}python -c 'import gradience; print(\"Gradience installed successfully!\")'${NC}"
echo
echo "To run benchmarks:"
echo -e "  ${BLUE}python -m gradience.bench.run_bench --help${NC}"