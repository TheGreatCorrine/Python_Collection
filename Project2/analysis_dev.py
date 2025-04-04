#!/usr/bin/env python
# coding: utf-8
# This is only the devleopment version of the analysis.py

# In[2]:
"""
Step 0: Import the necessary dependencies and download NLTK resources
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from contractions import CONTRACTION_MAP

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from nltk import ngrams

from collections import Counter

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('punkt_tab')

np.random.seed(42)

"""
Step 1: Load the training data and quick EDA
"""
train_data = pd.read_csv('ReviewsTraining.csv')
test_data = pd.read_csv('ReviewsTest.csv')

# Set the display options for pandas to show all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 2000)
pd.set_option('display.expand_frame_repr', False)

class DataLoader:
    def __init__(self):
        # Create a set of stopwords
        self.stop_words = set(stopwords.words('english'))
        # Remove negation words from stopwords as they're important for sentiment
        # TODO: 只保留这些不够吧
        self.negation_words = {'no', 'not', 'nor', 'none', 'never', 'neither', 'hardly', 'barely'}
        self.stop_words = self.stop_words - self.negation_words

    def explore_data(self, data):
        """
        This function explores the data structure and prints the first few rows
        :param data: The input data
        :return: None
        """
        # Check the structure
        print("Data shape:", data.shape)
        print("\nColumns:", data.columns.tolist())
        print("\nSample data:")
        print(data.head())

        # Check for missing values
        print("\nMissing values:")
        print(data.isnull().sum())

    def plot_score_distribution(self, data):
        """
        This function plots the distribution of review scores
        :param data: The input data
        :return: None
        """
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(x='Score', data=data)
        plt.title('Distribution of Review Scores')
        plt.xlabel('Score')
        plt.ylabel('Count')

        # Add count labels on top of bars
        for p in ax.patches:
            ax.annotate(f'{p.get_height()}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='baseline',
                        xytext=(0, 5), textcoords='offset points')

        plt.tight_layout()
        plt.show()

# # Uncomment the following lines to explore the training data
# # Create an instance of the DataLoader class
# data_loader = DataLoader()
# data_loader.explore_data(train_data) # Explore the training data
# data_loader.plot_score_distribution(train_data) # Plot the distribution of review scores

"""
Step 2: Text Preprocessing
    2.1 Expand contractions
    2.2 Text cleaning to lowercase, remove punctuation, and numbers
    2.3 Tokenization
    2.4 Remove stopwords but keep sentiment words, especially negation words
    2.5 Lemmatization
"""
# TODO: 先remove stopwords还是先tokenize？
class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Remove negation words from stopwords as they're important for sentiment
        self.negation_words = {'no', 'not', 'nor', 'none', 'never', 'neither', 'hardly', 'barely', 'but'}
        self.stop_words = self.stop_words - self.negation_words


    def expand_contractions(self, text, contraction_mapping=CONTRACTION_MAP):
        """
        This function expands contractions in the text
        :param text: The input text
        :param contraction_mapping: The contraction mapping
        :return: The text with expanded contractions

        @example: 'don't' -> 'do not', 'can't' -> 'cannot'
        """
        if isinstance(text, str):
            contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                              flags=re.IGNORECASE | re.DOTALL)

            def expand_match(contraction):
                match = contraction.group(0)
                first_char = match[0]
                expanded_contraction = contraction_mapping.get(match) \
                    if contraction_mapping.get(match) \
                    else contraction_mapping.get(match.lower())
                expanded_contraction = first_char + expanded_contraction[1:] if expanded_contraction else match
                return expanded_contraction

            expanded_text = contractions_pattern.sub(expand_match, text)
            return expanded_text
        else:
            return ""


    def preprocess_text(self, text):
        """
        Apply all preprocessing steps described above
        :param text: Raw text input
        :return: Preprocessed text
        """
        if not isinstance(text, str) or len(text.strip()) == 0:
            return ""

        # Expand contractions
        text = self.expand_contractions(text)
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        # Lemmatize tokens
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        # Join tokens back into text
        return ' '.join(tokens)


    def preprocess_dataframe(self, df, text_col='Text', summary_col='Summary'):
        """
        Preprocess both text and summary columns and combine them

        :param df: DataFrame containing text data
        :param text_col: Name of the column containing the main text
        :param summary_col: Name of the column containing the summary
        :return: DataFrame with preprocessed text
        """
        df_processed = df.copy()

        df_processed[f'{summary_col}_processed'] = df_processed[summary_col].apply(self.preprocess_text)
        df_processed[f'{text_col}_processed'] = df_processed[text_col].apply(self.preprocess_text)

        # Combine the processed columns
        df_processed['combined_text'] = df_processed[f'{summary_col}_processed'] + ' ' + df_processed[
            f'{text_col}_processed']

        return df_processed

# # Uncomment the following lines to test the TextPreprocessor class
# text_preprocessor = TextPreprocessor()
# example_text = "I can't believe it's not butter!"
# print(f"Original: {example_text}")
# print(f"Preprocessed: {text_preprocessor.preprocess_text(example_text)}")
# print()
# for line in test_data['Text']:
#     print(f"Original: {line}")
#     print(f"Preprocessed: {text_preprocessor.preprocess_text(line)}")
#     print()

text_preprocessor = TextPreprocessor()
train_data_processed = text_preprocessor.preprocess_dataframe(train_data)
test_data_processed = text_preprocessor.preprocess_dataframe(test_data)

"""
Step 3: Feature Engineering
    3.1 Convert scores to binary sentiment (1 = positive, 0 = negative)
    3.2 Vectorize text data using TF-IDF
    3.3 Split the data for training and validation
"""
# Convert scores to binary sentiment (scores 1-3 = negative, 4-5 = positive)
train_data_processed['sentiment_binary'] = train_data_processed['Score'].apply(lambda x: 1 if x >= 4 else 0)

# Unigrams only
unigram_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 1))
X_unigram = unigram_vectorizer.fit_transform(train_data_processed['combined_text'])

# Bigrams + Unigrams
bigram_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_bigram = bigram_vectorizer.fit_transform(train_data_processed['combined_text'])


# Split the data for training and validation
X_unigram_train, X_unigram_val, y_train_unigram, y_val_unigram = train_test_split(
    X_unigram, train_data_processed['sentiment_binary'], test_size=0.2, random_state=42
)

X_bigram_train, X_bigram_val, y_train_bigram, y_val_bigram = train_test_split(
    X_bigram, train_data_processed['sentiment_binary'], test_size=0.2, random_state=42
)

"""
Step 4: Model Building and Training
    4.1 Build unigram model
    4.2 Build bigram model
    4.3 Compare model performance
"""
def create_model(input_dim):
    """
    Creates a Keras ANN model for sentiment classification
    """
    model = Sequential([
        Dense(256, input_shape=(input_dim,), activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    return model


# Define early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

# Train unigram model
print("\nTraining unigram model:")
unigram_model = create_model(X_unigram_train.shape[1])
unigram_model.summary()

history_unigram = unigram_model.fit(
    X_unigram_train, y_train_unigram,
    epochs=10,
    batch_size=64,
    validation_data=(X_unigram_val, y_val_unigram),
    callbacks=[early_stopping],
    verbose=1
)

# Train bigram model
print("\nTraining bigram model:")
bigram_model = create_model(X_bigram_train.shape[1])
bigram_model.summary()

history_bigram = bigram_model.fit(
    X_bigram_train, y_train_bigram,
    epochs=10,
    batch_size=64,
    validation_data=(X_bigram_val, y_val_bigram),
    callbacks=[early_stopping],
    verbose=1
)

""" 4.3 Evaluate and compare model performance
    You can choose to display the full classification report or just the accuracy.
        - evaluate_model(model, X_val, y_val, model_name)
    method: evaluate_model(model, X_val, y_val, model_name)
    Finally, save the best model and vectorizer for future use.
"""
class ModelEvaluator:
    def __init__(self):
        pass

    def evaluate_model(self, model, X_val, y_val, model_name, plot=False):
        """
        This function evaluates the model on the validation set.
        It also displays the classification report and accuracy of the model
        """
        y_pred_prob = model.predict(X_val)
        y_pred = (y_pred_prob > 0.5).astype(int)

        if y_pred.ndim > 1:
            y_pred = y_pred.flatten()

        accuracy = accuracy_score(y_val, y_pred)

        if plot:
            print(f"\n{model_name} model evaluation:")
            print(f"Accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_val, y_pred))

        return accuracy, y_pred

    def plot_training_history(self, history_unigram, history_bigram, save_path='model_performance.png'):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history_unigram.history['accuracy'], label='Unigram Train')
        plt.plot(history_unigram.history['val_accuracy'], label='Unigram Val')
        plt.plot(history_bigram.history['accuracy'], label='Bigram Train')
        plt.plot(history_bigram.history['val_accuracy'], label='Bigram Val')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()

        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()

    def compare_models(self, unigram_accuracy, bigram_accuracy, unigram_model, bigram_model,
                      unigram_vectorizer, bigram_vectorizer):
        """Compare the performance of two models and return the best model"""
        print("\nBelow are the Model performance comparison:")
        print(f"Unigram model accuracy: {unigram_accuracy:.4f}")
        print(f"Bigram model accuracy: {bigram_accuracy:.4f}")

        if bigram_accuracy > unigram_accuracy:
            best_model = bigram_model
            best_vectorizer = bigram_vectorizer
            best_model_name = 'bigram'
        else:
            best_model = unigram_model
            best_vectorizer = unigram_vectorizer
            best_model_name = 'unigram'

        print(f"\n{best_model_name} model is the best model here.")

        return best_model, best_vectorizer, best_model_name

model_evaluator = ModelEvaluator()
# Set plot to True to display full details of the classification report
unigram_accuracy, y_pred_unigram = model_evaluator.evaluate_model(unigram_model, X_unigram_val, y_val_unigram, 'Unigram', plot=False)
bigram_accuracy, y_pred_bigram = model_evaluator.evaluate_model(bigram_model, X_bigram_val, y_val_bigram, 'Bigram', plot=False)
best_model, best_vectorizer, best_model_name = model_evaluator.compare_models(unigram_accuracy, bigram_accuracy, unigram_model, bigram_model,unigram_vectorizer, bigram_vectorizer)
model_evaluator.plot_training_history(history_unigram, history_bigram)

"""
Step 5: Generating Predictions for Test Data
    5.1 Transform test data
    5.2 Generate some predictions and display the prediction summary
    5.3 Save the prediction results to a new file called `Team6predictions.txt`
"""

# Transform test data
X_test = best_vectorizer.transform(test_data_processed['combined_text'])

# Generate predictions
y_pred_proba = best_model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int)

# Display prediction summary
print("\nTest data prediction summary:")
print(f"Total predictions: {len(y_pred)}")
print(f"Positive sentiment (1): {np.sum(y_pred)} samples ({np.sum(y_pred) / len(y_pred) * 100:.2f}%)")
print(
    f"Negative sentiment (0): {len(y_pred) - np.sum(y_pred)} samples ({(len(y_pred) - np.sum(y_pred)) / len(y_pred) * 100:.2f}%)")

# Save predictions to file
np.savetxt('Team1predictions.txt', y_pred, fmt='%d')
print("\nPredictions saved to 'Team6predictions.txt'")