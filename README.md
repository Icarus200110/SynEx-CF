# Train Files: Spatial Transcriptomics Training Modules

This directory contains three main modules for spatial transcriptomics analysis:

1. **RACF**: Gene expression recommendation system (strongly recommended for ML-based expression prediction)
2. **SynEx\_CF**: Gene expression prediction using MambaVision and KA-GNN
3. **SpotFormer**: Tissue type classification using Vision Transformer

***

## Module 1: RACF (Recommended for Expression Prediction)

RACF  is a lightweight module for gene expression recommendation. **Highly recommended for machine learning-based expression prediction models due to its simplicity and versatility.**

### Features

- Simple and easy to use - minimal setup required
- Supports gene expression data with arbitrary gene ordering
- Automatic gene alignment via gene order files
- Fast similarity search using FAISS
- Works with various ML models for expression prediction

### Quick Start

```python
from RACF import RACF

recommender = RACF(
    database_path="database.tsv",
    database_gene_order="gene_order.txt",
    k=10
)

recommendations = recommender.run(
    spot_data="spot_data.tsv",
    spot_gene_order="spot_gene_order.txt"
)
```

### Installation

```bash
pip install numpy pandas scikit-learn faiss-cpu
```

### More Information

For detailed usage, see [RACF README](../RACF/README.md) or run the demo:

```bash
cd ../RACF
python demo.py
```

***

## Directory Structure

```
train_files/
├── SynEX_CF_train.py          # SynEx_CF distributed training script
├── SynEx_CF.py                # SynEx_CF model definition
├── dataset.py                 # Dataset loaders (HER2ST, CSCC, BDS)
├── predict.py                 # Prediction and evaluation utilities
├── train_spotformer.py        # SpotFormer training script
├── inference_spotformer.py    # SpotFormer inference script
├── evaluate_spotformer.py     # SpotFormer evaluation script
├── predict_spotformer.py      # SpotFormer prediction script
├── src/
│   ├── models/
│   │   └── spotformer.py      # SpotViT/SpotResNet model implementation
│   ├── data/
│   │   └── spotformer_dataset.py
│   └── utils/
│       ├── label_utils.py
│       └── metrics.py
├── configs/
│   └── spotformer_config.yaml
├── requirements.txt
└── README.md
```

***

## Module 2: SynEx\_CF (Gene Expression Prediction)

### Model Architecture

SynEx\_CF integrates multiple components:

- **TissueTypeEmbedding**: Embeds 3x3 neighborhood tissue type labels
- **SpatialContextKAGNN**: KA-GNN based spatial context feature extraction
- **MambaVisionImageFeatureExtractor**: MambaVision based image feature extraction
- **SSDFNet**: Spatial-Semantic Dynamic Fusion Network for feature fusion
- **GenePredictor**: Gene expression prediction from fused features

### Training

```bash
python SynEX_CF_train.py --data her2st --fold 12 --gpus 0,1
```

#### Parameters

| Parameter         | Default                   | Description                       |
| ----------------- | ------------------------- | --------------------------------- |
| `--gpus`          | '0,1,2,3'                 | GPU IDs to use                    |
| `--fold`          | 12                        | Dataset fold for cross-validation |
| `--lr`            | 1e-5                      | Learning rate                     |
| `--epochs`        | 401                       | Number of training epochs         |
| `--data`          | 'her2st'                  | Dataset: her2st, cscc, or BDS     |
| `--train_samples` | "A1,B1,C1,D1,E1,F1,G2,H1" | Training samples                  |
| `--freeze_vit`    | False                     | Freeze MambaVision parameters     |

### Output

- Model checkpoints: `./model/NewHGGEP/`
- TensorBoard logs: `./logs/NewHGGEP_{dataset}_{name}/`

***

## Module 3: SpotFormer (Tissue Type Classification)

### Model Architecture

SpotFormer uses Vision Transformer for tissue type classification:

- **SpotViT**: Lightweight Vision Transformer
- **Model Variants**: spotvit\_tiny, spotvit\_small, spotvit\_base
- **Supported Classes**: breast glands, cancer in situ, immune infiltrate, invasive cancer, necrosis, stroma, undetermined

### Training

```bash
python train_spotformer.py --config configs/spotformer_config.yaml
```

### Inference

```bash
python inference_spotformer.py --model <model_path> --config configs/spotformer_config.yaml --data_root data --samples A1 B1
```

### Evaluation

```bash
python evaluate_spotformer.py --model <model_path> --config configs/spotformer_config.yaml
```

### Configuration

Key configuration options in `configs/spotformer_config.yaml`:

```yaml
data:
  data_root: "data"
  patch_size: 64
  batch_size: 1
  undersample_majority: true

model:
  type: "spotvit_small"

training:
  epochs: 100
  early_stopping_patience: 15

optimizer:
  learning_rate: 0.0005

loss:
  type: "focal"
  gamma: 2.5
```

### Output

```
outputs/spotresnet_YYYYMMDD_HHMMSS/
├── checkpoints/
│   ├── best.pth
│   └── latest.pth
├── logs/
├── visualizations/
└── classification_report.txt
```

***

## Installation

### 1. Create Environment

```bash
conda create -n synex python=3.10 -y
conda activate synex
```

### 2. Install PyTorch

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install MambaVision

```bash
pip install mambavision
```

Or from source:

```bash
git clone https://github.com/NVlabs/MambaVision.git
cd MambaVision
pip install -e .
```

### 4. Install KA-GNN Dependencies

```bash
pip install dgl
pip install torch_geometric
```

### 5. Install Other Dependencies

```bash
pip install -r requirements.txt
```

***

## Dependencies

```
torch>=1.10
pytorch-lightning>=1.4
scanpy>=1.8
torch_geometric
dgl
timm
einops
scipy
scikit-learn
pandas
numpy
matplotlib
seaborn
Pillow
scprep
anndata
opencv-python
pyyaml
tqdm
```

***

## Data Preparation

### HER2ST Dataset

```bash
git clone https://github.com/almaan/her2st.git data/her2st
```

### Data Directory Structure

```
data/
├── ST-imgs/                    # Image data
│   ├── A/
│   │   └── A1/
│   │       └── A1.jpg
│   └── B/
│       └── B1/
│           └── B1.jpg
└── ST-pat/
    └── lbl/                    # Label data
        ├── A1_labeled_coordinates.tsv
        └── B1_labeled_coordinates.tsv
```

### Label File Format (TSV)

```
pixel_x	pixel_y	x	y	label
1234.5	2345.6	10	20	invasive cancer
1345.6	2456.7	11	21	breast glands
```

***

## References

- [MambaVision: A Hybrid Mamba-Transformer Vision Backbone](https://github.com/NVlabs/MambaVision) (CVPR 2025)
- [KA-GNN: Kolmogorov-Arnold Graph Neural Networks](https://github.com/LongLee220/KA-GNN) (Nature Machine Intelligence, 2025)

## Citation

<br />

