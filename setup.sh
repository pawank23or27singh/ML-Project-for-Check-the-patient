#!/bin/bash

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate  # On Windows, use: .\venv\Scripts\activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "\nSetup complete! To run the application:"
echo "1. Activate virtual environment:"
echo "   - On Linux/Mac: source venv/bin/activate"
echo "   - On Windows: .\\venv\\Scripts\\activate"
echo "2. Run the app: streamlit run app.py"
