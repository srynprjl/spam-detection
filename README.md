# Email Spam Detection System

A complete machine learning pipeline and interactive web interface for classifying emails as either Spam or Ham emails. The system utilizes Natural Language Processing (NLP) techniques and a Hybrid Naive Bayes classifier to achieve high prediction accuracy.

## Overview

This project implements a full lifecycle for spam detection, covering data collection from multiple sources, rigorous preprocessing, feature engineering, and a unique classification approach. The hybrid model combines Multinomial Naive Bayes for word-based analysis and Gaussian Naive Bayes for numerical features, resulting in a balanced and effective prediction system.

### Core Methodology
- Hybrid Classifier: Integrates Multinomial Naive Bayes (for text content) and Gaussian Naive Bayes (for engineered numerical features).
- Feature Engineering: Goes beyond simple keyword matching by analyzing digit ratios, URL presence, lexical diversity, and average word lengths.
- Real-time Interface: Provides an interactive dashboard for users to test custom messages.

## Live Demo
A live version of the interactive dashboard is available at: https://spam.sysnefo.com

## Features

- Data Ingestion: Processes and merges multiple datasets (Enron, Ling-Spam) into a unified format.
- Preprocessing Pipeline: Automates the removal of null values, duplicates, and performs field merging (Subject + Body).
- NLP Processing: Handles tokenization, stopword removal, and Bag-of-Words vectorization.
- Engineered Metrics: Extracts metadata features including:
  - Digit Ratio: Frequency of numerical digits in the message.
  - URL Presence: Detection of common web link patterns (http, www, .com).
  - Lexical Diversity: Ratio of unique words to total word count.
  - Average Word Length: Statistical measure of message complexity.
- Evaluation Suite: Comprehensive performance tracking including accuracy metrics (approximately 94%).

## Project Structure

```text
├── app.py                     # Dash web application and model inference logic
├── datasets/                  # Source and processed data files
│   ├── spam_enron.csv         # Enron email dataset source
│   ├── spam_ling.csv          # Ling-Spam dataset source
│   ├── spam_dataset.csv       # Final processed dataset used for training
│   └── spam_test.csv          # Evaluation dataset
├── notebooks/
│   ├── EDA & Preprocessing.ipynb # Data cleaning and feature extraction logic
│   └── SpamDetection.ipynb    # Model training, validation, and export
├── pyproject.toml             # Project configuration and metadata
├── requirements.txt           # Python dependency specifications
└── README.md                  # Project documentation
```

## Technologies Used

- Data Analysis: Pandas, NumPy
- Web Interface: Dash, Plotly
- Visualization: Matplotlib, Seaborn
- Development: JupyterLab, Python 3.14+

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/srynprjl/spam-detection
   cd spam-detection
   ```

### Option A: Using uv (Recommended)
`uv` will automatically manage the correct Python version and dependencies for you.
```bash
uv sync
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Option B: Using pip
Ensure you have Python 3.14+ installed on your system.
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Note: This project is configured for Python 3.14 and above as specified in the project configuration.

## Usage

### 1. Data Processing
Run the `notebooks/EDA & Preprocessing.ipynb` notebook to clean the raw CSV files in the `datasets` folder and generate the consolidated `spam_dataset.csv`.

### 2. Model Training
Execute the `notebooks/SpamDetection.ipynb` notebook. This process will:
- Split the data into training and testing sets.
- Train the Hybrid Naive Bayes components.
- Export the trained weights and vocabulary to `models/spam_detection.pkl`.

### 3. Running the Dashboard
Start the interactive web interface by running:
```bash
python app.py
```
By default, the application will be accessible at `http://127.0.0.1:8050/`.

## Performance Metrics

The Hybrid Naive Bayes approach achieved an accuracy of approximately 94% on the test dataset. By combining both linguistic patterns (word frequencies) and structural features (digit ratios, URLs), the model maintains robustness across different communication styles found in the Enron and Ling-Spam datasets.

## Example Input/Output

Input:
```text
Subject: Re: Q3 Marketing Sync
Body:
Hi Sarah,
I've attached the updated slide deck for tomorrow's meeting. I adjusted the budget projections on slide 4 based on the feedback from the finance team. Let me know if you have any questions before we go live!
Best,
Mark
```

Output:
```text
Classification: HAM
```
