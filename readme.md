# Fake News Detection Using Machine Learning

A machine learning web application built with Python that classifies news articles as **Fake** or **Real** using Natural Language Processing (NLP) techniques.

## Overview

The Fake News Detection system analyzes the text of a news article and predicts whether it resembles fake or real news based on patterns learned from a labelled dataset.

The project uses TF-IDF for text feature extraction and Logistic Regression for classification. A Streamlit interface allows users to enter news articles and view the model's predictions.

## Features

- News classification into Fake and Real categories
- Text preprocessing using Pandas
- TF-IDF feature extraction
- Logistic Regression classifier
- Model evaluation using accuracy, precision, recall, F1-score, and a confusion matrix
- Saved model pipeline using Joblib
- Interactive web interface built with Streamlit

## Tech Stack

- **Programming Language:** Python
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **NLP:** TF-IDF Vectorization
- **Classification Algorithm:** Logistic Regression
- **Web Framework:** Streamlit
- **Model Persistence:** Joblib
- **Development Environment:** VS Code

## Dataset

This project uses the Fake and Real News Dataset, which contains two CSV files:

- `Fake.csv` — articles labelled as fake
- `True.csv` — articles labelled as real

Dataset source: [Fake and Real News Dataset on Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

The dataset contains historical news articles. Its labels and writing patterns may not represent current news or all forms of misinformation.

## Machine Learning Workflow

1. Load the fake and real news datasets.
2. Assign numerical labels to the articles.
3. Combine the headline and article text.
4. Handle missing values and remove duplicate articles.
5. Split the data into training and testing sets.
6. Convert text into numerical features using TF-IDF.
7. Train a Logistic Regression classifier.
8. Evaluate the model on held-out test data.
9. Save the trained pipeline using Joblib.
10. Use Streamlit to accept article text and display predictions.

## Model Performance

The initial model achieved the following results on the held-out test split:

| Metric | Result |
|---|---:|
| Accuracy | 99.00% |
| Fake News Precision | 99% |
| Fake News Recall | 98% |
| Real News Precision | 99% |
| Real News Recall | 100% |

**Evaluation note:** These metrics describe performance on the selected test split only. They do not establish equivalent accuracy on new articles from different sources. Further testing on independent, source-separated and recent news data is needed.

## Project Structure

```text
fake-news-detection-ml/
├── app.py
├── train_model.py
├── predict.py
├── fake_news_model.joblib
├── requirements.txt
├── README.md
└── data/
    ├── Fake.csv
    └── True.csv
```

The `data/` directory is needed to retrain the model. The deployed application can load the saved model without the CSV files if it does not retrain during startup.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Abhishek00134/fake-news-detection-ml.git
cd fake-news-detection-ml
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment (recommended)

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Train the model

Ensure `Fake.csv` and `True.csv` are present inside the `data/` directory, then run:

```bash
python train_model.py
```

The training script evaluates the classifier and generates `fake_news_model.joblib`.

If the saved model is already available, you can skip training.

### 5. Run the web application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal to use the application.

## Usage

1. Launch the Streamlit application.
2. Enter or paste a news headline or article into the text box.
3. Click **Analyze News**.
4. View the predicted category: Fake or Real.

The result represents the model's learned classification, not a verified assessment of the article's factual accuracy.

## Limitations

- The model learns statistical patterns in the training dataset rather than verifying claims against evidence.
- Dataset-specific language and source patterns can influence predictions.
- Historical training data may not generalize to current events.
- Short summaries and full articles may receive different predictions.
- High test accuracy does not guarantee reliable misinformation detection in real-world use.

## Future Improvements

- Compare Logistic Regression with Multinomial Naive Bayes and Linear SVM.
- Evaluate performance on independent datasets and unseen news sources.
- Add confusion-matrix visualizations and a model evaluation dashboard.
- Integrate reputable fact-checking sources for independent verification.
- Explore transformer-based NLP models for improved contextual understanding.
- Improve the user interface and deployment experience.

## Author

**Abhishek**

GitHub: [Abhishek](https://github.com/Abhishek00134)
