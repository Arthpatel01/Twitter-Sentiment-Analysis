# Twitter Sentiment Analysis

This project performs sentiment analysis on Twitter data using a Multinomial Naive Bayes classifier. It aims to predict the sentiment (positive, negative, or neutral) of tweets based on their content.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Preprocessing](#preprocessing)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Visualizations](#visualizations)
- [Contributing](#contributing)
- [License](#license)
- [Developer](#developer)

## Introduction

Sentiment analysis is a natural language processing task that involves determining the sentiment expressed in a piece of text. In this project, we focus on sentiment analysis of tweets related to various companies using a Multinomial Naive Bayes classifier.

## Installation

To run this project locally, you need to have Python and the required packages installed. You can install the required packages using the following command:

```bash
pip install numpy pandas nltk scikit-learn matplotlib
```

## Usage

Clone the GitHub repository and navigate to the project directory:

```bash
git clone https://github.com/Arthpatel01/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis
```

Run the Jupyter Notebook file `Twitter_Sentiment_Analysis.ipynb` to execute the entire project workflow.

## Data

The training and validation datasets used for this project are available in the files `twitter_training.csv` and `twitter_validation.csv`. The datasets contain columns such as `Tweet ID`, `entity`, `sentiment`, and `Tweet content`.

## Preprocessing

Data preprocessing is an essential step in natural language processing tasks. In this project, we perform the following preprocessing steps:
- Convert text to lowercase
- Remove stopwords and non-alphabetic characters
- Apply stemming

## Model Training

We use the Multinomial Naive Bayes classifier for sentiment prediction. The classifier is trained on the training data after text data is transformed into numerical features using TF-IDF vectorization.

## Evaluation

The performance of the trained model is evaluated using accuracy, classification report, and confusion matrix on the test dataset.

## Results

The model achieved an accuracy of [insert accuracy value here] on the test dataset.

## Visualizations

We provide visualizations of the sentiment distribution in the training and test datasets, as well as a line chart comparing actual and predicted sentiment distributions.

## Contributing

Contributions to this project are welcome. If you find any issues or want to add new features, feel free to submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Developer

This project is developed by Arth Patel. You can find more about the developer on GitHub: [Arth Patel](https://github.com/Arthpatel01/).
```

