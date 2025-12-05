# Gemini CUA Vision - GUI Grounding Evaluation

A comprehensive evaluation framework for testing Google's Gemini Computer Use Agent (CUA) on GUI grounding tasks. This tool evaluates how well Gemini can understand natural language instructions and locate the correct UI elements on screenshots.

## Overview

This framework:
- **Evaluates** Gemini CUA's ability to locate UI elements based on natural language instructions
- **Supports** multiple datasets: ScreenSpot, UI-Vision, and ScreenSpot Pro
- **Analyzes** results by application, platform, UI element type, and more
- **Generates** detailed reports with success rates and failure patterns

---

## Prerequisites

Before you begin, ensure you have:

1. **Python 3.8 or higher** (tested with Python 3.9+)
   ```bash
   python --version
   ```

2. **Git** installed
   ```bash
   git --version
   ```

3. **Git LFS** (Large File Storage) - required for downloading dataset images
   ```bash
   git lfs --version
   ```
   If not installed:
   - **macOS**: `brew install git-lfs`
   - **Linux**: `sudo apt-get install git-lfs` or follow [Git LFS installation guide](https://git-lfs.github.com/)
   - **Windows**: Download from [Git LFS website](https://git-lfs.github.com/)

4. **Google API Key** for Gemini API access
   - Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

---

## Installation & Setup

Follow these steps carefully to set up the evaluation environment:

### Step 1: Clone the Repository

```bash
cd /path/to/your/project
git clone <repository-url>
cd omniparser_pipeline/GeminiCUA_Vision
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt when activated.

### Step 3: Install Python Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

This will install:
- `google-genai` - Gemini API client
- `playwright` - Browser automation
- `python-dotenv` - Environment variable management
- `pydantic` - Data validation
- `Pillow` - Image processing
- `datasets` - HuggingFace datasets
- `numpy`, `matplotlib` - Data analysis

### Step 4: Install Playwright Browsers

Playwright needs browser binaries to run:

```bash
playwright install chromium
```

This downloads Chromium browser used for automation.

### Step 5: Configure API Key

Create a `.env` file in the `GeminiCUA_Vision` directory:

```bash
# Create .env file
touch .env  # On macOS/Linux
# or use your text editor on Windows
```

Add your Google API key:

```bash
GOOGLE_API_KEY=your_actual_api_key_here
```

**Important**: 
- Replace `your_actual_api_key_here` with your actual API key
- Never commit the `.env` file to version control
- The `.env` file is already in `.gitignore`

### Step 6: Setup Datasets

Choose which datasets you want to evaluate:

#### Option A: ScreenSpot (Auto-downloaded)

No setup required! The ScreenSpot dataset will be automatically downloaded from HuggingFace on first use.

#### Option B: UI-Vision Dataset (Recommended for full dataset)

1. **Create datasets directory and clone the repository** (uses Git LFS):
   ```bash
   cd GeminiCUA_Vision
   mkdir -p datasets
   cd datasets
   git clone https://huggingface.co/datasets/ServiceNow/ui-vision
   cd ui-vision
   ```

2. **Install Git LFS and pull images**:
   ```bash
   git lfs install
   git lfs pull
   ```
   This downloads the actual image files (not just pointers).

3. **Verify setup**:
   ```bash
   ls images/element_grounding/ | head -5  # Should show image files
   ```

#### Option C: ScreenSpot Pro Dataset (Recommended for comprehensive evaluation)

1. **Create datasets directory and clone the repository**:
   ```bash
   cd GeminiCUA_Vision
   mkdir -p datasets
   cd datasets
   git clone https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding ScreenSpot-Pro
   cd ScreenSpot-Pro
   ```

2. **Install Git LFS and pull images**:
   ```bash
   git lfs install
   git lfs pull
   ```

3. **Verify setup**:
   ```bash
   ls images/ | head -5  # Should show image files
   ls annotations/ | head -5  # Should show JSON annotation files
   ```

### Step 7: Verify Installation

Run a quick test to ensure everything works:

```bash
# Test that Python can import required packages
python -c "from google import genai; print('✓ Gemini API imported')"
python -c "from playwright.sync_api import sync_playwright; print('✓ Playwright imported')"
python -c "from PIL import Image; print('✓ PIL imported')"

# Check API key is loaded
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('✓ API Key:', 'SET' if os.getenv('GOOGLE_API_KEY') else 'NOT SET')"
```

---

## Quick Start

### Run Evaluation

**ScreenSpot Pro (Recommended)**:
```bash
python src/evaluate_grounding.py \
    --dataset screenspot-pro \
    --num_samples 10 \
    --output_dir results/screenspot_pro_evaluation_results \
    --screenspot_pro_path datasets/ScreenSpot-Pro
```

**UI-Vision**:
```bash
python src/evaluate_grounding.py \
    --dataset ui-vision \
    --num_samples 10 \
    --output_dir results/evaluation_results_ui_vision \
    --task_type element_grounding \
    --subtask basic \
    --dataset_path datasets/ui-vision
```

**ScreenSpot (Auto-downloaded)**:
```bash
python src/evaluate_grounding.py \
    --dataset screenspot \
    --num_samples 10 \
    --output_dir results/evaluation_results_screenspot
```

### Analyze Results

After running evaluations, generate analysis reports:

```bash
# For ScreenSpot Pro
python src/analyze_evaluation_results.py \
    --results_dir results/screenspot_pro_evaluation_results \
    --annotations_dir datasets/ScreenSpot-Pro/annotations \
    --output results/evaluation_analysis_report.txt

# View the report
cat results/evaluation_analysis_report.txt
```

---

## Usage Guide

### Command-Line Arguments

#### `evaluate_grounding.py`

Main evaluation script with the following options:

```
Required:
  --dataset DATASET_NAME    Dataset to use: 'screenspot', 'ui-vision', or 'screenspot-pro'

Optional:
  --num_samples N           Number of samples to evaluate (default: 10)
  --output_dir DIR          Output directory for results (default: results/evaluation_results)
  --random_seed SEED        Random seed for reproducibility
  
Dataset-specific:
  --screenspot_pro_path PATH     Path to ScreenSpot-Pro directory (default: datasets/ScreenSpot-Pro)
  --dataset_path PATH            Path to dataset directory (default: datasets/ui-vision)
  --task_type TYPE               Task type: 'element_grounding' or 'layout_grounding' (ui-vision)
  --subtask SUBTASK              Subtask: 'basic', 'functional', or 'spatial' (ui-vision)
```

#### `analyze_evaluation_results.py`

Analysis script for evaluation results:

```
Required:
  --results_dir DIR         Directory containing evaluation results (default: results/screenspot_pro_evaluation_results)

Optional:
  --annotations_dir DIR     Path to annotations directory (default: datasets/ScreenSpot-Pro/annotations)
  --output FILE             Output filename for report (default: results/evaluation_analysis_report.txt)
```

### Example Workflows

**Workflow 1: Evaluate 100 ScreenSpot Pro samples**
```bash
python src/evaluate_grounding.py \
    --dataset screenspot-pro \
    --num_samples 100 \
    --output_dir results/screenspot_pro_evaluation_results \
    --screenspot_pro_path datasets/ScreenSpot-Pro \
    --random_seed 42

python src/analyze_evaluation_results.py \
    --results_dir results/screenspot_pro_evaluation_results \
    --annotations_dir datasets/ScreenSpot-Pro/annotations
```

**Workflow 2: Continue evaluation (skip already evaluated samples)**
```bash
# Run another batch - automatically skips already evaluated samples
python src/evaluate_grounding.py \
    --dataset screenspot-pro \
    --num_samples 100 \
    --output_dir results/screenspot_pro_evaluation_results \
    --screenspot_pro_path datasets/ScreenSpot-Pro
```

---

## Understanding Results

### Output Structure

Evaluation results are saved in the output directory:

```
output_dir/
├── evaluation_results.json           # Summary statistics
├── success/
│   └── sample_XXXX/
│       ├── original.png             # Original screenshot
│       ├── annotated.png            # With predictions (red) & ground truth (green)
│       └── metadata.json            # Full evaluation details
└── failed/
    └── sample_YYYY/
        ├── original.png
        ├── annotated.png
        └── metadata.json
```

### Analysis Report

The analysis report (`evaluation_analysis_report.txt`) includes:

- **Overall Success Rate**: Percentage of correct predictions
- **By Application**: Success rates per application (Excel, Photoshop, etc.)
- **By Platform**: Success rates for Windows, macOS, Linux
- **By UI Element Type**: Icon vs. Text element performance
- **By Query Length**: How instruction length affects performance
- **By Bounding Box Size**: Performance on small vs. large targets

### Key Metrics

- **Point in Bbox Rate**: Percentage of predictions that fall within the ground truth bounding box
- **Success/Fail Counts**: Number of correct and incorrect predictions
- **Error Analysis**: Common failure patterns

---

## Project Structure

```
GeminiCUA_Vision/
├── src/                              # Source code
│   ├── evaluate_grounding.py         # Main evaluation script ⭐
│   ├── analyze_evaluation_results.py # Result analysis tool ⭐
│   ├── dataset_loaders.py            # Modular dataset loading system
│   ├── pipeline/                     # Core Gemini agent implementation
│   │   └── services/
│   │       ├── gemini/               # Gemini agent code
│   │       │   ├── agent.py          # Agent execution logic
│   │       │   └── actions.py        # Browser action helpers
│   │       ├── models.py             # Data models
│   │       └── pipeline_runner.py    # Pipeline orchestration
│   └── scripts/                      # Utility scripts
│       ├── test_pipeline_with_ratios.py  # Screen ratio testing
│       └── run_gemini_pipeline.py    # Custom task runner
│
├── datasets/                         # All datasets
│   ├── ui-vision/                    # UI-Vision dataset (cloned)
│   │   ├── images/                   # Screenshots (Git LFS)
│   │   └── annotations/              # JSON annotations
│   └── ScreenSpot-Pro/               # ScreenSpot Pro dataset (cloned)
│       ├── images/                   # Screenshots (Git LFS)
│       └── annotations/              # JSON annotations
│
├── results/                          # Evaluation results (ignored by git)
│   ├── screenspot_pro_evaluation_results/
│   ├── evaluation_results_ui_vision/
│   ├── screenspot_evaluation_results/
│   └── evaluation_analysis_report.*
│
├── docs/                             # Documentation
│   ├── FOLDER_STRUCTURE_GUIDE.md     # Detailed folder documentation
│   └── ARCHITECTURE_EXPLANATION.md   # Architecture details
│
├── .gitignore                        # Git ignore rules
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
└── venv/                             # Virtual environment (ignored by git)
```

**Note**: The `results/` directory is ignored by git (see `.gitignore`). Evaluation results are generated locally and not committed to version control.

For detailed folder documentation, see [docs/FOLDER_STRUCTURE_GUIDE.md](docs/FOLDER_STRUCTURE_GUIDE.md).

---

## Troubleshooting

### Problem: "ModuleNotFoundError" when running scripts

**Solution**: Make sure your virtual environment is activated:
```bash
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows
```

### Problem: "Cannot identify image file" error

**Solution**: Git LFS files not downloaded. Run:
```bash
cd datasets/ui-vision  # or datasets/ScreenSpot-Pro
git lfs install
git lfs pull
```

### Problem: "API key not found" or authentication errors

**Solution**: 
1. Verify `.env` file exists in `GeminiCUA_Vision/` directory
2. Check API key is correctly formatted: `GOOGLE_API_KEY=your_key_here`
3. Ensure no extra spaces or quotes around the key
4. Verify API key is valid at [Google AI Studio](https://makersuite.google.com/app/apikey)

### Problem: Playwright browser errors

**Solution**: Reinstall Playwright browsers:
```bash
playwright install chromium
playwright install --help  # For more options
```

### Problem: Out of memory errors with large datasets

**Solution**: 
- Process samples in smaller batches: `--num_samples 50`
- Close other applications to free up RAM
- Consider using a machine with more memory for large evaluations

### Problem: Evaluation hangs or times out

**Solution**:
- Check your internet connection (API calls require connectivity)
- Verify API key has sufficient quota
- Check `evaluation.log` for detailed error messages
- Try reducing `--num_samples` to test connectivity first

### Problem: Dataset path not found

**Solution**: 
- Verify dataset directory exists: `ls datasets/ui-vision/` or `ls datasets/ScreenSpot-Pro/`
- Check path is correct in command: `--dataset_path datasets/ui-vision` or `--screenspot_pro_path datasets/ScreenSpot-Pro`
- Ensure you're running commands from `GeminiCUA_Vision/` directory

---

## Advanced Usage

### Custom Evaluation

You can modify evaluation parameters in `evaluate_grounding.py` or create custom evaluation scripts using the dataset loaders:

```python
from src.dataset_loaders import get_dataset_loader

loader = get_dataset_loader("screenspot-pro", dataset_path="datasets/ScreenSpot-Pro")
samples = loader.load_samples(num_samples=10, random_seed=42)

for sample in samples:
    image = loader.get_sample_image(sample)
    instruction = loader.get_sample_instruction(sample)
    bbox = loader.get_sample_bbox(sample)
    # ... your evaluation logic
```

### Testing Different Screen Ratios

Test how screen aspect ratios affect performance:

```bash
python src/scripts/test_pipeline_with_ratios.py --ratio ultrawide
```

Available ratios: `default`, `standard`, `ultrawide`, `wide`, `superwide`

---

## Contributing

When adding new features:

1. Follow the existing code structure
2. Add new dataset loaders by extending `BaseDatasetLoader` in `dataset_loaders.py`
3. Update this README with new usage instructions
4. Test thoroughly before submitting

---

## License

[Add your license information here]

---

## Contact & Support

For issues or questions:
- Check `evaluation.log` for detailed error messages
- Review `FOLDER_STRUCTURE_GUIDE.md` for detailed documentation
- Open an issue on the repository

---

## Acknowledgments

- **Gemini CUA**: Google's Computer Use Agent
- **ScreenSpot**: GUI grounding dataset from Roots Automation
- **UI-Vision**: GUI grounding dataset from ServiceNow
- **ScreenSpot Pro**: Comprehensive GUI grounding dataset

---

**Last Updated**: See git history for latest changes.
