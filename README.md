# Email Spam Classifier

A machine learning-based email spam classifier built in Python. This project detects whether an email is spam or ham (not spam) using NLP techniques and scikit-learn models. It also features a Streamlit web app for real-time predictions.

## Project Overview

Emails are an essential form of communication, but spam emails waste time and can pose security risks. This project implements a robust spam detection system using natural language processing (NLP) and machine learning.

## Key Features:

Classifies emails as Spam or Ham.

Uses LinearSVC as the primary classifier (previously experimented with LogisticRegression).

Supports real-time predictions via a Streamlit app.

Saves vectorizer and model as .pkl files for easy reuse.

## Tech Stack

Language: Python 3.x

Libraries: scikit-learn, pandas, numpy, Streamlit

Machine Learning Models: LinearSVC, LogisticRegression (for experimentation)

Tools: Jupyter Notebook, VS Code / PyCharm

## Project Structure
Email_Spam_Classifier_Project/
│
├── app.py                # Streamlit application
├── main.ipynb            # Jupyter Notebook for model training
├── vectorizer.pkl        # Saved CountVectorizer / TfidfVectorizer
├── model.pkl             # Saved trained LinearSVC model
├── .gitignore            # Git ignore file
└── README.md             # Project documentation

## Setup and Installation

Clone the repository:

git clone https://github.com/joshiyash710/Email-Spam-Classifier.git
cd Email-Spam-Classifier


## Create and activate a virtual environment:

python -m venv spam-env
#### Windows
spam-env\Scripts\activate
#### macOS/Linux
source spam-env/bin/activate


### Install dependencies:

pip install -r requirements.txt


### Run the Streamlit app:

streamlit run app.py


Open the provided URL in your browser to test the spam classifier.

## Usage

You can test the classifier by entering email content in the Streamlit app.
Examples:

### Spam Email:

Congratulations! You won a lottery. Claim your prize now!


Predicted: Spam

### Ham Email:

Hi, can we meet tomorrow for lunch?


Predicted: Ham

Contributions

Contributions are welcome! You can:

Improve the model accuracy.

Add preprocessing for better NLP feature extraction.

Extend the web app with additional features like email history.

License

This project is open-source and free to use.
