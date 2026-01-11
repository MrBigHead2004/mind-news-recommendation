# MIND News Recommendation with NRMS

A PyTorch implementation of the **NRMS (Neural News Recommendation with Multi-Head Self-Attention)** model for personalized news recommendation, trained on the [Microsoft News Dataset (MIND)](https://msnews.github.io/).

## 📋 Overview

This project implements a neural news recommendation system that learns to predict which news articles a user will click based on their reading history. The model uses:

- **RoBERTa** for encoding news titles
- **Multi-Head Self-Attention** for capturing word-level and news-level interactions
- **Additive Attention** for aggregating representations

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         NRMS Model                          │
├─────────────────────────────────────────────────────────────┤
│  News Encoder:                                              │
│    Title → RoBERTa → Multi-Head Self-Attention              │
│         → Additive Attention → News Vector                  │
├─────────────────────────────────────────────────────────────┤
│  User Encoder:                                              │
│    History News Vectors → Multi-Head Self-Attention         │
│         → Additive Attention → User Vector                  │
├─────────────────────────────────────────────────────────────┤
│  Click Prediction:                                          │
│    Score = dot(Candidate News Vector, User Vector)          │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Download Dataset

The project uses the MIND dataset from Kaggle:

```python
from src.download_dataset import download_dataset
download_dataset()
```

Or manually download from [Kaggle](https://www.kaggle.com/datasets/nhthongl/mind-dataset) and place it in the `MIND/` directory:

```
MIND/
├── MINDsmall_train/
│   ├── news.tsv
│   └── behaviors.tsv
├── MINDsmall_dev/
│   ├── news.tsv
│   └── behaviors.tsv
└── MINDlarge_test/
    ├── news.tsv
    └── behaviors.tsv
```

### Training

Run the Jupyter notebook `mind_news_rec.ipynb` for an interactive training experience, or use the training modules directly:

```python
from src.config import config, print_config
from src.data import load_news_data, load_behaviors_data, tokenize_news, create_behavior_samples
from src.models import get_model
from src.train import train_model

# Print configuration
print_config()

# Load and preprocess data
news_df = load_news_data()
train_behaviors_df, val_behaviors_df = load_behaviors_data()
news_features, news_categories = tokenize_news(news_df)

# Create model
model = get_model('nrms', config)

# Train (see notebook for full training loop)
```

### Prediction

Generate predictions for the MIND competition:

```bash
python predict.py --checkpoint mind_news_rec.pth --output prediction.txt
```

Options:

- `--checkpoint`: Path to trained model checkpoint (default: `mind_news_rec.pth`)
- `--output`: Output file for predictions (default: `prediction.txt`)
- `--batch_size`: Batch size for inference (default: 32)
- `--news_path`: Path to test news.tsv
- `--behaviors_path`: Path to test behaviors.tsv

## 📁 Project Structure

```
mind-news-recommendation/
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration settings
│   ├── data.py             # Data loading and preprocessing
│   ├── download_dataset.py # Dataset download utility
│   ├── metrics.py          # Evaluation metrics (AUC, MRR, NDCG)
│   ├── train.py            # Training loop
│   ├── utils.py            # Utility functions
│   └── models/
│       ├── __init__.py     # Model registry
│       ├── attention.py    # Attention mechanisms
│       ├── base.py         # Base model class
│       └── nrms.py         # NRMS implementation
├── mind_news_rec.ipynb     # Main training notebook
├── predict.py              # Prediction script for competition
├── requirements.txt        # Python dependencies
└── README.md
```

## ⚙️ Configuration

Key configuration options in `src/config.py`:

| Parameter             | Default        | Description                   |
| --------------------- | -------------- | ----------------------------- |
| `MODEL_NAME`          | `roberta-base` | Pretrained language model     |
| `EMBEDDING_DIM`       | 768            | Embedding dimension           |
| `NUM_ATTENTION_HEADS` | 16             | Number of attention heads     |
| `MAX_TITLE_LEN`       | 10             | Maximum title token length    |
| `MAX_HISTORY_LEN`     | 20             | Maximum user history length   |
| `NEG_SAMPLES`         | 4              | Negative samples per positive |
| `BATCH_SIZE`          | 8              | Training batch size           |
| `EPOCHS`              | 3              | Number of training epochs     |
| `LR`                  | 4e-5           | Learning rate                 |
| `DROPOUT`             | 0.2            | Dropout rate                  |

## 📊 Evaluation Metrics

The model is evaluated using standard news recommendation metrics:

- **AUC**: Area Under the ROC Curve
- **MRR**: Mean Reciprocal Rank
- **NDCG@5**: Normalized Discounted Cumulative Gain at 5
- **NDCG@10**: Normalized Discounted Cumulative Gain at 10

## 🔧 Requirements

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- NumPy, Pandas
- scikit-learn
- tqdm
- kagglehub

See `requirements.txt` for complete dependencies.

## 📚 References

- [NRMS: Neural News Recommendation with Multi-Head Self-Attention](https://aclanthology.org/D19-1671/) (Wu et al., EMNLP 2019)
- [MIND: A Large-scale Dataset for News Recommendation](https://msnews.github.io/) (Wu et al., ACL 2020)

## 📝 License

This project is for educational purposes. Please refer to the MIND dataset license for data usage terms.

## 🙏 Acknowledgments

- Microsoft Research for the MIND dataset
- Hugging Face for the Transformers library
