🚀 Sentence Contradiction Classification

🎯 Goal

This project aims to classify pairs of sentences into one of three categories: Contradiction, Entailment, or Neutral based on their semantic meaning. The classification is achieved using various machine learning and deep learning models, including traditional algorithms, neural networks, and transformer-based architectures like BERT and XLM-R.

📂 Dataset

The dataset consists of two files:

train.csv: Labeled dataset containing sentence pairs and their relationships.

id: Unique identifier for each sentence pair.

sentence1: The first sentence (Premise).

sentence2: The second sentence (Hypothesis).

label: Relationship between the sentences:

0: Contradiction (Sentences have opposite meanings)

1: Neutral (Sentences are related but do not imply each other)

2: Entailment (One sentence logically follows from the other)

test.csv: Unlabeled dataset for predictions.

🔍 Project Workflow

📊 1. Exploratory Data Analysis (EDA)

Visualized class distribution (Contradiction, Entailment, Neutral).

Analyzed sentence structures (length, word distribution, common words).

Checked for missing values and outliers.

🛠️ 2. Text Preprocessing

Tokenization (splitting sentences into words).

Lowercasing and removal of stopwords, special characters, and punctuation.

Lemmatization/Stemming for text normalization.

Feature extraction using TF-IDF, Word2Vec, or Transformer embeddings (BERT, XLM-R).

🤖 3. Model Training

Baseline Models: Logistic Regression, Decision Tree, Random Forest, XGBoost.

Neural Networks: Custom Artificial Neural Network (ANN).

Sequence Models: LSTM/GRU for sequential learning.

Transformer-Based Models: Fine-tuning BERT and XLM-R for better contextual understanding.

📈 4. Model Evaluation

Metrics used: Accuracy, Precision, Recall, F1-score.

Confusion Matrix to analyze misclassifications.

AUC-ROC curve to evaluate classification performance.

📊 Model Performance:

Baseline Models:

Random Forest: 43.89% accuracy

Logistic Regression: 43.64% accuracy

ANN: 43.23% accuracy

XGBoost: 42.90% accuracy

Decision Tree: 41.00% accuracy

LSTM: 33.00% accuracy

Transformer Models:

BERT Performance:

Accuracy: 61%

Macro F1-score: 0.61

XLM-R Performance:

Accuracy: 63%

Macro F1-score: 0.63

🔧 5. Hyperparameter Tuning

Experimented with optimizers (Adam, SGD) and activation functions.

Adjusted learning rate, batch size, and epochs.

Used Grid Search and Random Search for optimization.

🏆 Final Performance Evaluation

Best-performing model: XLM-R (63% accuracy, best macro F1-score)

Classification metrics: Accuracy, Precision, Recall, F1-score, and AUC-ROC.

Confusion Matrix visualization.

📜 Expected Deliverables

Jupyter Notebook (.ipynb) with:

EDA (visualizations included)

Text preprocessing pipeline

Model training and evaluation

Hyperparameter tuning (if applicable)

Performance Report detailing classification results.

🛠️ Installation & Setup

Prerequisites

Ensure you have Python and necessary libraries installed:

pip install numpy pandas scikit-learn tensorflow transformers matplotlib seaborn

▶️ Running the Notebook

Execute the Jupyter Notebook to train and evaluate the model:

jupyter notebook sentence-contradiction-classification.ipynb

📂 Repository Structure

│-- dataset/
│   │-- train.csv
│   │-- test.csv
│-- notebooks/
│   │-- sentence-contradiction-classification.ipynb
│-- README.md

🚀 Future Work

Experimenting with larger transformer models like RoBERTa, DeBERTa.

Investigating zero-shot and few-shot learning approaches.

Enhancing dataset augmentation techniques.

📜 License

This project is open-source and available under the MIT License.

Enhancing dataset augmentation techniques.

📜 License

This project is open-source and available under the MIT License.
