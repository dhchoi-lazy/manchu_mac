# Manchu LLaMA 3.2 11B Vision Model

This repository contains scripts to run the `dhchoi/manchu-llama32-11b-vision-merged` model for Manchu script OCR and text generation.

**Note:** This application is optimized for macOS with Apple Silicon (MPS) or CPU support only.

## Setup

### 1. Clone Repository

```bash
git clone https://github.com/dhchoi-lazy/manchu_mac.git
cd manchu_mac
```

### 2. Clean Virtual Environment (if exists)

```bash
# Remove existing virtual environment for clean setup
rm -rf .venv
```

### 3. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Web Interface (Streamlit)

The easiest way to use the Manchu OCR model is through the web interface:

```bash
python run_streamlit.py
```

The web app will open automatically in your browser at `http://localhost:10011`

### Command Line Interface

#### Manchu OCR Model

```bash
# Interactive mode
python run_manchu_model.py

# Single image OCR
python run_manchu_model.py ./samples/validation_sample_0000.jpg

# Batch directory processing
python run_manchu_model.py ./samples/
```

#### Model Evaluation

```bash
python run_manchu_eval.py
```

## Model Information

- **Model**: `dhchoi/manchu-llama32-11b-vision-merged`
- **Base**: LLaMA 3.2 11B Vision
- **Purpose**: Manchu script OCR and text generation
- **Size**: ~21.3 GB
- **Device Support**: Apple Silicon (MPS) or CPU

## System Requirements

- **OS**: macOS (Apple Silicon recommended)
- **Python**: 3.10+
- **Memory**: 16GB+ RAM recommended
- **Storage**: 25GB+ free space for model cache

## Performance

- **Apple Silicon (MPS)**: ~30-40 seconds per image
- **CPU**: ~60-120 seconds per image (depending on CPU)
