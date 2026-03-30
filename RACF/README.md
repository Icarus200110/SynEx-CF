# RACF

## Features

- Supports gene expression data with arbitrary gene ordering
- Automatic gene alignment via gene order files
- Composite similarity-based recommendation (weighted cosine similarity + Jaccard similarity)
- Fast similarity search using FAISS
- Diverse recommendations: clustering-selection strategy ensures result diversity

## Installation

```bash
pip install -r requirements.txt
```

Core dependencies:

- numpy
- pandas
- scikit-learn
- faiss-cpu or faiss-gpu

## Data Format

### Expression Data File (TSV format)

First column is sample ID, remaining columns are gene expression values:

```
sample_id    GENE_001    GENE_002    GENE_003    ...
DB_001       1.23        0.45        -0.67       ...
DB_002       0.89        1.21        0.33        ...
```

### Gene Order File (one gene name per line)

```
GENE_003
GENE_001
GENE_002
```

## Quick Start

### 1. Basic Usage

```python
from RACF import RACF

recommender = RACF(
    database_path="path/to/database.tsv",
    database_gene_order="path/to/gene_order.txt",
    k=10
)

recommendations = recommender.run(
    spot_data="path/to/spot_data.tsv",
    spot_gene_order="path/to/spot_gene_order.txt"
)

for spot_idx, rec_indices in recommendations.items():
    print(f"Spot {spot_idx}: Recommended {rec_indices}")
```

### 2. With Final Expression Calculation

```python
recommendations, final_expressions = recommender.run(
    spot_data="path/to/spot_data.tsv",
    spot_gene_order="path/to/spot_gene_order.txt",
    calculate_final_expression=True
)
```

### 3. Without Gene Order File

If gene order file is not provided, the system assumes the first column is ID and remaining columns follow column names:

```python
recommender = RACF(
    database_path="path/to/database.tsv",
    k=10
)

recommendations = recommender.run(spot_data="path/to/spot_data.tsv")
```

### 4. Direct DataFrame Input

```python
import pandas as pd
from RACF import RACF

database_df = pd.read_csv("database.tsv", sep='\t')
spot_df = pd.read_csv("spot_data.tsv", sep='\t')

recommender = RACF(database_path=database_df, k=10)
recommendations = recommender.run(spot_df)
```

## Parameters

### RACF Class

| Parameter            | Type          | Default | Description                          |
| -------------------- | ------------- | ------- | ------------------------------------ |
| `database_path`     | str/DataFrame | Required | Expression database path or DataFrame |
| `k`                 | int           | 10      | Number of recommended database vectors per spot |
| `n_candidates`       | int           | 5000    | Initial candidate count               |
| `n_precise`         | int           | 1000    | Precise candidate count              |
| `database_gene_order`| str           | None    | Database gene order file path        |

### run() Method

| Parameter                  | Type          | Default | Description                              |
| -------------------------- | ------------- | ------- | ---------------------------------------- |
| `spot_data`               | str/DataFrame | Required | Spot expression data path or DataFrame    |
| `spot_gene_order`          | str           | None    | Spot gene order file path                |
| `calculate_final_expression`| bool          | False   | Whether to calculate final expression    |

## Algorithm Flow

1. **Data Loading & Alignment**: Load database and spot data, align gene order via gene order files
2. **Preprocessing**: Missing value filling, normalization
3. **FAISS Index Building**: Fast search index using L2 distance
4. **Similarity Calculation**: Weighted cosine similarity (70%) + Jaccard similarity (30%)
5. **Diverse Recommendation**: Clustering-selection strategy ensures diversity

## Run Demo

```bash
python demo.py
```

Demo includes:

- Basic usage
- Handling data with different gene ordering
- Final expression calculation
- Programmatic DataFrame input

## File Structure

```
RACF/
├── RACF.py           # Main module
├── demo.py           # Demo script
├── requirements.txt  # Dependencies
└── README.md         # This file
```
