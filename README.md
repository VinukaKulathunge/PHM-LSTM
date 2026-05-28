# Personal Health Mention (PHM) Tweet Classification using Stacked LSTMs

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.0+-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io/)
[![Natural Language Processing](https://img.shields.io/badge/NLP-Bi--LSTM-blue?style=flat-square)](https://en.wikipedia.org/wiki/Natural_language_processing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

An end-to-end deep learning project utilizing advanced Recurrent Neural Networks (RNNs) to classify Twitter/X posts. The objective is to identify **Personal Health Mentions (PHM)**—tweets where a user expresses a personal health issue, symptom, or diagnosis—from general tweets. Accurate PHM classification plays a critical role in digital epidemiology, biosurveillance, and real-time public health monitoring.

---

## Features

- **Custom Text Preprocessing Pipeline**: Cleans noisy social media text (removes URLs, user mentions, and non-alphabetic characters) while selectively preserving critical personal pronouns (`i`, `my`, `me`, `im`, `ive`, `myself`) that act as key semantic indicators of personal health claims.
- **Deep Sequence Architectures**:
  - **Stacked LSTM**: 2 stacked Long Short-Term Memory layers with spatial dropout and recurrent dropout to avoid overfitting.
  - **Stacked Bidirectional LSTM (Bi-LSTM)**: 2 stacked bidirectional layers to capture sequential dependencies in both forward and backward directions for richer contextual representation.
- **Addressing Class Imbalance**: Uses compute-based class weights during model training (`class_weight_dict = {0: 1.0, 1: 1.4}`) to handle the relatively sparse nature of PHM tweets compared to non-PHM tweets.
- **Robust Training Workflow**: Implements learning rate decay (`ReduceLROnPlateau`), checkpointing (`ModelCheckpoint`), and early stopping to guarantee optimal model convergence.
- **Decision Threshold Tuning**: Custom decision threshold set to `0.4` (rather than standard `0.5`) to optimize the sensitivity/recall of personal health mention detection, crucial for epidemiological tracking.

---

## Deep Learning Architectures

### 1. Stacked LSTM Network
- **Embedding Layer**: Projects token IDs into dense embeddings with masking enabled (`mask_zero=True`) to neglect padding tokens.
- **Spatial Dropout**: Performs 1D spatial dropout to drop entire feature maps across steps.
- **Stacked LSTMs**: Dual-layered LSTM network with Recurrent Dropout (`0.2`) to prevent recurrent state overfitting.
- **Dense Output**: A Sigmoid output layer for binary classification.

### 2. Stacked Bidirectional LSTM (Bi-LSTM) Network
- Includes the same embedding and spatial dropout structures.
- **Bidirectional Wrapping**: Wraps the stacked LSTM layers in PyTorch/Keras Bidirectional layers, letting the model process context from past-to-future and future-to-past simultaneously.
- Offers superior representation capacity for nuanced social media texts.

---

## Project Structure

```bash
personal-health-tweet-lstm/
├── .gitignore                           # Excludes checkpoints, environments, and temp files
├── phm_tweet_classification_lstm.ipynb  # Primary Jupyter notebook containing the pipeline
└── README.md                            # Professional documentation (this file)
```

---

## Getting Started

### Prerequisites
Make sure you have Python 3.8+ installed on your local machine.

### Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/VinukaKulathunge/personal-health-tweet-lstm.git
   cd personal-health-tweet-lstm
   ```

2. **Create and Activate a Virtual Environment**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install --upgrade pip
   pip install tensorflow numpy pandas matplotlib nltk scikit-learn jupyter
   ```

4. **Download NLTK Stopwords**
   The notebook downloads the required datasets automatically, but you can also pre-fetch them:
   ```python
   import nltk
   nltk.download('stopwords')
   ```

5. **Run the Notebook**
   ```bash
   jupyter notebook phm_tweet_classification_lstm.ipynb
   ```

---

## Evaluation & Results

Both models are evaluated on a separate test set across precision, recall, and F1-score. A decision threshold of `0.4` is utilized to capture a higher volume of positive cases (PHM), resulting in improved recall for public health surveillance tasks.

---

## License
This project is licensed under the MIT License - see the LICENSE file for details.
