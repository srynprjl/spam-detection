# Spam Detection System

This project is a Spam Detection System that uses Natural Language Processing (NLP) and machine learning to classify messages as either "Spam" or "Ham" (legitimate). It includes a full pipeline from data preprocessing and exploratory data analysis (EDA) to model training and a web-based interface for real-time testing.

---

## What we did
For the development of this Spam Detection System we first used the dataset of Enron email and linguistic datasets to ensure the model could handle a wide variety of communication styles. During the preprocessing phase, we removed many null values & duplicated values and merge subject and text. 

To move beyond simple keyword matching, we did feature engineering by calculating metrics like digit ratios and average word lengths, special symbols and URLs. 

After the preprocessing phase, we implemented a Hybrid Naive Bayes Classifier that combines Multinomial Naive Bayes for text-based word analysis with Gaussian Naive Bayes for calculating the probability of numerical features. This approach allowed the model to achieve an accuracy of approximately 94%.

Finally, we created a web interface using Dash framework, allowing users to input custom messages and receive instant Spam or Ham predictions.

---

## Features
- Text preprocessing and cleaning
- Tokenization and stopword removal
- Feature extraction using Bag of Words
- Classificiation using Naive Bayes
- Model evaluation with performance metrics
- Spam prediction for new messages

---

## Technologies Used
- Pandas
- NumPy
- Dash
- Matplotlib 
- Seaborn

---
## Project Structure
`EDA & Preprocessing.ipynb`: Handles data loading, cleaning, and feature engineering. It merges multiple datasets (like Enron and Linguist) and extracts features such as digit ratios, symbol counts, and URL presence.

`SpamDetection.ipynb`: Contains the machine learning logic, including data splitting, model training, and performance evaluation.

`app.py`: A Dash-based web application that allows users to input custom messages and get instant classification results using the trained model.

`spam_dataset.csv`: The primary processed dataset used for training, containing the raw text and engineered features.

## Primary Dataset

- Enron Spam Dataset
- Linguist Dataset

---


## Usage
Preprocessing: Run ` EDA & Preprocessing.ipynb` to clean the raw data and generate the spam_dataset.csv file. <br />
Training: Run `SpamDetection.ipynb` to train the Naive Bayes model and export it as a pickle file <br />
Dashboard:  Run `python app.py` to open the Dashboard <br />


Input :
```
Subject: Re: Q3 Marketing Sync
Body:
Hi Sarah,
I've attached the updated slide deck for tomorrow's meeting. I adjusted the budget projections on slide 4 based on the feedback from the finance team. Let me know if you have any questions before we go live!
Best,
Mark

```

Output:

```
Ham
```

---
