Application of Transformer Attention Mechanism-Based Multimodal Deep Learning Model in the Diagnosis of Papillary Thyroid Carcinoma
Project Overview
This project implements a deep learning framework for multimodal data analysis (medical imaging, radiomics features, clinical data) designed for pathology-related tasks such as pathological complete response (pCR) prediction. Key features include:

Multimodal Fusion Architecture: Flexible integration of any number of image, radiomics, and clinical data branches

Enhanced Vision Backbone: EfficientNetV2S with CBAM (Convolutional Block Attention Module) attention mechanism

High-Stability Training Configuration: Includes memory optimization, stability patches, and reproducibility settings

One-Click Training Pipeline: End-to-end training solution based on the Onekey framework

Project Structure
text
project_root/
├── run_multigroup.py           # Main training script (multimodal fusion configuration)
├── efficientnetv2s_cbam.py     # Custom EfficientNetV2S_CBAM model
├── requirements.txt            # Dependency list (to be created based on environment)
├── LICENSE                     # License file
└── README.md                   # Project documentation
Environment Requirements
Python Version
Python 3.8 or higher

Main Dependencies
text
torch>=1.9.0
torchvision>=0.10.0
timm>=0.6.0
numpy>=1.19.0
pandas>=1.3.0
safetensors>=0.3.0 (optional, for safe weight loading)
onekey-algo                     # Multimodal fusion framework (requires separate installation)
Environment Setup
bash
# Create conda environment (recommended)
conda create -n multimodal_fusion python=3.8
conda activate multimodal_fusion

# Install PyTorch (select based on CUDA version)
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# Install other dependencies
pip install timm pandas numpy
pip install safetensors  # Optional
Quick Start
1. Data Preparation
Organize data according to the following structure:

text
data_directory/
├── imaging_data/
│   ├── group1/          # Corresponds to "V" in IMAGE_GROUPS
│   └── ...
├── radiomics_features/
│   └── rad_features_cleaned.csv
└── clinical_data/
    └── clinical.csv
2. Configuration
Modify the configuration section in run_multigroup.py:

python
# Output directory (modify to your path)
OUTPUT_ROOT = r"your_output_directory"

# Imaging data paths
IMAGE_GROUPS = {
    "V": r"your_imaging_data_path",
}

# Radiomics features configuration
RADIOMICS_GROUPS = {
    "RAD1": {
        "feature_file": r"radiomics_feature_file_path",
        "input_dim": 1282,  # Important: Set to the number of feature columns
        # ... other parameters
    },
}

# Clinical data configuration
CLINICAL_GROUPS = {
    "CLI1": {
        "feature_file": r"clinical_data_file_path",
        "input_dim": 10,  # Important: Set to the number of feature columns
        # ... other parameters
    },
}

# Task configuration
task_settings = {
    'pCR': {
        'label_file': r'label_file_path',
        'type': 'clf', 
        'num_classes': 2
    },
}
3. Run Training
bash
python run_multigroup.py
Code File Descriptions
run_multigroup.py - Main Training Script
This script configures and initiates multimodal fusion training with the following features:

Stability Patches:

Disables ONNX export during training to save memory

Disables TensorBoard computation graph logging

Modifies DataLoader to single-process mode to avoid deadlocks

Multimodal Configuration:

Supports any number of imaging branches

Supports any number of radiomics feature branches

Supports any number of clinical data branches

Training Parameters:

Configurable batch size, learning rate, epochs

Fixed random seeds for reproducibility

Gradient accumulation support (for memory constraints)

efficientnetv2s_cbam.py - Custom Vision Model
Vision backbone based on EfficientNetV2S with CBAM attention mechanism:

Core Components:

CBAM attention module (channel attention + spatial attention)

Dynamic input channel adapter (supports 1-channel → 3-channel conversion)

Local pretrained weight loading support (.bin or .safetensors format)

Model Features:

Supports both feature extraction and classification forward modes

Option to freeze stem layers for faster training

Compatible with timm library's features_only mode

Configuration Details
Imaging Branch Configuration
python
IMAGE_GROUPS = {
    "branch_name": "image_path",
    # Multiple branches can be added
    # "V2": "second_image_path",
}
Feature Branch Configuration
python
RADIOMICS_GROUPS = {
    "branch_name": {
        "feature_file": "CSV_file_path",
        "input_dim": feature_dimension,   # Must be set correctly
        "norm": True,                     # Whether to normalize
        "hidden_unit": [32,64,128],       # DNN hidden layer structure
        "dropout": 0.5,                   # Dropout rate
    },
}
Training Parameters
Parameter	Description	Recommended Value
batch_size	Batch size	16-32 (adjust based on GPU memory)
epochs	Training epochs	50-100
init_lr	Initial learning rate	3e-4
weight_decay	Weight decay	5e-4
trans_dim	Feature transformation dimension	64
Output Description
After training completion, the following structure will be generated in the OUTPUT_ROOT directory:

text
output_directory/
├── checkpoints/          # Model checkpoints
├── logs/                 # Training logs
├── tensorboard/          # TensorBoard records
└── results/             # Evaluation results
Frequently Asked Questions
Q1: Memory Insufficient Error During Training
Reduce batch_size (e.g., from 32 to 16)

Enable gradient accumulation (add "grad_accum_steps": 2 to parameters)

Ensure num_workers=0 (set by default)

Q2: How to Load Local Pretrained Weights?
python
# Set in run_multigroup.py
LOCAL_WEIGHTS = r"your_weight_file_path"
# Supports .pth, .bin, .safetensors formats
Q3: How to Add New Task Types?
Add new tasks in task_settings:

python
task_settings = {
    'pCR': {...},  # Classification task
    'OS': {        # Survival analysis task
        'label_file': 'survival_data.csv',
        'type': 'sur',
        'event_column': 'event',
        'duration_column': 'duration'
    },
}
Q4: How to Reproduce Experimental Results?
Set random seed to 42 (default)

Use identical data preprocessing pipeline

Ensure all non-deterministic operations are disabled

Model Architecture
text
Input Data
    ├── Imaging Branch → EfficientNetV2S_CBAM → Feature Extraction (256-dim)
    ├── Radiomics Branch → DNN → Feature Extraction (configurable)
    └── Clinical Data Branch → DNN → Feature Extraction (configurable)
        ↓
    Feature Fusion Layer (trans_dim=64)
        ↓
    Fusion DNN ([16,32,16] hidden layers)
        ↓
    Task-Specific Output Layer
Important Notes
Data Paths: Use r"path" format for Windows paths to avoid escape issues

Feature Dimensions: input_dim must exactly match the number of feature columns in CSV files

GPU Memory Management: Large images or batches may cause memory issues; start testing with smaller configurations

Onekey Framework: Requires prior installation and configuration of the Onekey multimodal fusion framework

Citation
If you use this code, please cite:

bibtex
@software{multimodal_fusion,
  author = {Your Name},
  title = {Application of Transformer Attention Mechanism-Based Multimodal Deep Learning Model in the Diagnosis of Papillary Thyroid Carcinoma},
  year = {2024},
  url = {https://github.com/yourusername/yourrepository}
}
