# NLP Projects

This repository contains various Natural Language Processing (NLP) projects. The projects focus on different aspects of NLP, including data extraction, cleaning, and analysis.

## Projects

1. **10-K Report Analysis**:
   - Implemented a data extraction pipeline to automatically extract and clean useful information from 10-K reports filed with the SEC.
   - Utilized Python libraries such as Pandas and BeautifulSoup to transform raw HTML content into a structured dataframe.
   - Designed and implemented a data schema to store cleaned data for over 50 companies in PostgreSQL, reducing data querying complexities by approximately 20%.
   - Enhanced the data extraction process by employing regular expressions and Python logic to selectively parse and clean financial data, significantly reducing processing time and improving data reliability for subsequent analysis.
   - Conducted thorough data cleaning to remove HTML tags, entities, and unnecessary content, ensuring high-quality data for analysis.
   - Extracted key information such as fiscal year-end dates, legal proceedings, signature dates, and signers from the cleaned text for further analysis.

2. **Sentiment Analysis on Product Quality**:
   - Performed sentiment analysis on consumer reviews to understand perceptions of product quality.
   - Utilized the dataset `ReviewsTraining.csv` to predict the sentiment score based on the text in the "Summary" and "Text" columns.
   - Preprocessed the text by expanding contractions, removing special characters, and tokenizing into unigrams and bigrams.
   - Built and trained a Keras Artificial Neural Network (ANN) to predict sentiment scores, classifying scores 1-3 as negative and 4-5 as positive.
   - Implemented a bonus feature to train a multi-class Keras ANN for predicting scores 1-5.
   - Analyzed the most frequent words and recurring phrases in negative reviews to identify underlying problems and provide strategic recommendations.

## Requirements

- Python 3.x
- Pandas
- BeautifulSoup
- PostgreSQL
- NumPy
- Matplotlib
- Keras
- TensorFlow

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/TheGreatCorrine/python-collection.git
   cd python-collection