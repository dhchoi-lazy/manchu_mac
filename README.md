# Manchu LLaMA 3.2 11B Vision Model

This repository contains scripts to run the `dhchoi/manchu-llama32-11b-vision-merged` model for Manchu script OCR and text generation.

## Setup

### 1. Clone Repository

```bash
git clone https://github.com/dhchoi-lazy/manchu_mac.git
cd manchu_mac
```

### 2. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### 🌐 Web Interface (Streamlit)

The easiest way to use the Manchu OCR model is through the web interface:

```bash
python streamlit_app.py
```

**Features:**

- 📤 Drag-and-drop file upload
- 🖼️ Image preview with details
- 📝 Real-time OCR processing
- 📊 Performance metrics
- 💾 Downloadable results
- 📱 Mobile-friendly interface

The web app will open automatically in your browser at `http://localhost:8501`

### 💻 Command Line Interface

#### Manchu OCR Model

```bash
# Interactive mode
python run_manchu_model.py

# Single image OCR
python run_manchu_model.py image.jpg

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

## 📚 Additional Resources

- `POETRY_SETUP.md` - Detailed Poetry setup and usage guide
- `poetry_commands.py` - Helper script showing common Poetry commands
