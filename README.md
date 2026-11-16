# 🌀 Irony Detection

<!-- Live Demo: [IronyScope- Irony Detector](https://ironyscope.streamlit.app/)
<img src="https://github.com/user-attachments/assets/fa501fb0-beaf-4524-a95b-c1ba7fa634fa" alt="Demo GIF" width="900"/> -->

This compact NLP project, developed for the **_Introduction to Data Science_** course, explores irony detection in Twitter data using a machine learning pipeline. It uses the small version of the SemEval-2018 Task-3 dataset, a benchmark for irony detection, and applies NLTK-based text preprocessing techniques, including tokenisation, stemming, stopword removal, and emoji and emoticon conversion.

The processed text is transformed into numerical features using TF-IDF and used to train and evaluate machine learning classifiers. The project explores several classification approaches, including Multinomial Naive Bayes, LinearSVC, Stochastic Gradient Descent (SGD), Decision Tree, and Multi-layer Perceptron (MLP). Model performance is assessed using accuracy, precision, recall, F1-score, and confusion matrices.

The original comparative evaluation reported SGD as the best-performing classifier, with a score of 65%. The latest notebook also contains an MLPClassifier implementation, which is evaluated separately. This project demonstrates practical applications of text preprocessing, feature engineering, and supervised classification for social media text analysis.

This project is intended for demonstration purposes. While the model can make predictions, it may sometimes produce unexpected results due to limitations in the dataset and model performance. Its primary goal is to showcase practical applications of irony detection and NLP techniques.

---

## 🚀 Features
- **Preprocessing**: Tokenisation, language detection, accented character removal, stemming, stopword removal, and emoji and emoticon conversion.
- **Feature Extraction**: TF-IDF features generated from unigrams, bigrams, and trigrams.
- **Classifier Comparison**: Evaluation of Multinomial Naive Bayes, LinearSVC, SGD, Decision Tree, and MLP classifiers.
- **Evaluation**: Accuracy, precision, recall, F1-score, confusion matrices, and classification reports.

## 📊 Dataset
- **SemEval-2018 Task-3 (Small version)** [C. Van Hee et al., 2018 – ACL Anthology](https://aclanthology.org/S18-1005.pdf)

## 🧠 Model Architecture
The project uses a traditional machine learning pipeline for text classification:

- **CountVectorizer:** Transforms text into token-count matrices using unigrams, bigrams, and trigrams (1–3 n-grams).
- **TF-IDF Transformer:** Converts token counts into TF-IDF scores using L1 normalisation and smoothed IDF values.
- **Classifiers:** Scikit-learn implementations of Multinomial Naive Bayes, LinearSVC, SGD, Decision Tree, and MLPClassifier.
- **MLP Configuration:** The latest notebook implementation uses a feedforward neural network with one hidden layer of 100 units, ReLU activation, and the Adam optimiser.

## 📈 Evaluation
The original comparative evaluation reported the following results:

| Classifier | Accuracy | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: | ---: |
| Multinomial Naive Bayes | 0.50 | 0.50 | 0.50 | 0.50 |
| Decision Tree | 0.30 | 0.30 | 0.30 | 0.30 |
| LinearSVC | 0.55 | 0.55 | 0.55 | 0.55 |
| **SGD** | **0.65** | **0.65** | **0.65** | **0.65** |
| MLP | 0.55 | 0.60 | 0.60 | 0.55 |

The original presentation identified SGD as the best-performing classifier, reporting a micro-average F1-score of 65%. The MLP results are included separately for comparison. These are historical evaluation results from the project presentation and should not be interpreted as a guarantee of performance from the latest notebook implementation.

## 🛠️ Tools and Libraries
- Python
- scikit-learn
- NLTK
- pandas
- numpy
- langdetect
- matplotlib
- seaborn
- emoji
- emoticon_fix

## 💻 How to Run Locally
1️⃣ Clone the Repository

```bash
git clone https://github.com/adrikachowdhury/IronyScope.git
cd IronyScope
```

2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

3️⃣ To check out the evaluation scores or modify the optimal settings before deployment, you may run the notebook `colab-version.ipynb`.

## 🔑 Notes
- The trained model (`irony_model.pkl`) is already included, so no training is required.
- NLTK data (`punkt`, `stopwords`) will be downloaded automatically if missing.
- Ensure your Python version is >=3.9 for full compatibility.

## 💫 Credits
Feel free to explore the documentation, and please give **credit to the owner** when using content from this repo!

Many thanks!🙌
