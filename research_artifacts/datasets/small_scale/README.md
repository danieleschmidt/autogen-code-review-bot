# Small Scale Dataset

## Overview
This dataset was generated for research validation of novel algorithmic approaches in code analysis and optimization.

## Dataset Characteristics
- **Samples**: 1,000
- **Features**: 10
- **Complexity**: low
- **Generation Seed**: 42
- **Noise Level**: 0.1

## Feature Categories
- Code Metrics: Structural properties of code
- Complexity Metrics: Algorithmic complexity measures  
- Quality Metrics: Code quality indicators
- Performance Metrics: Runtime performance measures

## Target Variables
- `target_continuous`: Continuous performance improvement score
- `target_binary`: Binary classification of improvement (above/below median)

## Algorithm Performance Comparisons
- `performance_baseline`: Baseline algorithm performance
- `performance_quantum_hybrid`: Quantum-classical hybrid algorithm
- `performance_neural_semantic`: Neural semantic analysis algorithm
- `performance_consciousness_inspired`: Consciousness-inspired algorithm
- `performance_temporal_optimization`: Temporal optimization algorithm

## File Formats
- **CSV**: `small_scale.csv` - Standard comma-separated values
- **Parquet**: `small_scale.parquet` - Compressed columnar format
- **Pickle**: `small_scale_full.pkl.gz` - Complete Python object with all metadata
- **JSON**: `small_scale_metadata.json` - Dataset metadata and statistics

## Usage Example
```python
import pandas as pd
import pickle
import gzip

# Load CSV
df = pd.read_csv('small_scale.csv')

# Load full dataset
with gzip.open('small_scale_full.pkl.gz', 'rb') as f:
    full_dataset = pickle.load(f)
```

## Citation
If you use this dataset in your research, please cite:
```
@dataset{small_scale_2025,
  title={{Research Validation Dataset: Small Scale}},
  author={Research Breakthrough Engine},
  year={2025},
  version={1.0}
}
```

## License
This dataset is released under the MIT License for research and educational purposes.
