# Large Dataset Files

The following large dataset files are excluded from git due to GitHub size limitations:

- `research_artifacts/datasets/large_scale/large_scale.csv` (198.51 MB)
- `research_artifacts/datasets/enterprise_scale/enterprise_scale.csv` (580.43 MB) 
- `research_artifacts/datasets/large_scale/large_scale.parquet` (97.70 MB)
- `research_artifacts/datasets/large_scale/large_scale_full.pkl.gz` (77.68 MB)

## How to Regenerate

These files can be regenerated using the research breakthrough engine:

```python
from execute_research_breakthrough import ResearchBreakthroughEngine

engine = ResearchBreakthroughEngine()
datasets = engine.generate_benchmark_datasets()
```

## Production Deployment

For production deployment, use the optimized algorithms without the raw research datasets. The key algorithms and validation results are included in the repository.

## Alternative Access

Large research datasets can be made available through:
- Cloud storage (S3, GCS, Azure Blob)
- Git LFS (Large File Storage)
- Academic data repositories
- Direct download from research collaboration partners

Contact: research@terragonlabs.com for access to full datasets.