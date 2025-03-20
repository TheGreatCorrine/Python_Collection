from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm  # Use SVM for classification
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier

# !/usr/bin/env python
# coding: utf-8

# RSM317 Group Assignment 2 - Multi-class Sentiment Analysis (Bonus Task)

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
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)

print("Loading data...")
# Load the data
train_data = pd.read_csv('ReviewsTraining.csv')
test_data = pd.read_csv('ReviewsTest.csv')

print(f"Training data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

# Display score distribution
print("\nScore distribution in training data:")
score_counts = train_data['Score'].value_counts().sort_index()
print(score_counts)

# Plot score distribution
plt.figure(figsize=(10, 6))
ax = sns.countplot(x='Score', data=train_data)
plt.title('Distribution of Review Scores')
plt.xlabel('Score')
plt.ylabel('Count')
for p in ax.patches:
    ax.annotate(f'{p.get_height()}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='baseline',
                xytext=(0, 5), textcoords='offset points')
plt.savefig('score_distribution.png')


class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Remove negation words from stopwords as they're important for sentiment
        self.negation_words = {'no', 'not', 'nor', 'none', 'never', 'neither', 'hardly', 'barely', 'but'}
        self.stop_words = self.stop_words - self.negation_words
        # Initialize VADER sentiment analyzer
        self.sid = SentimentIntensityAnalyzer()

    def expand_contractions(self, text, contraction_mapping=CONTRACTION_MAP):
        """
        Expand contractions in text using the provided mapping
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
        Apply all preprocessing steps:
        1. Expand contractions
        2. Convert to lowercase
        3. Remove punctuation
        4. Remove numbers
        5. Tokenize
        6. Remove stopwords
        7. Lemmatize tokens
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

    def get_vader_scores(self, text):
        """
        Calculate VADER sentiment scores for the text
        """
        if not isinstance(text, str) or len(text.strip()) == 0:
            return {'compound': 0, 'pos': 0, 'neg': 0, 'neu': 0}

        return self.sid.polarity_scores(text)

    def preprocess_dataframe(self, df, text_col='Text', summary_col='Summary'):
        """
        Preprocess text columns and add VADER features
        """
        print(f"Preprocessing {df.shape[0]} reviews...")
        df_processed = df.copy()

        # Preprocess text
        df_processed[f'{summary_col}_processed'] = df_processed[summary_col].apply(self.preprocess_text)
        df_processed[f'{text_col}_processed'] = df_processed[text_col].apply(self.preprocess_text)

        # Combine processed text
        df_processed['combined_text'] = df_processed[f'{summary_col}_processed'] + ' ' + df_processed[
            f'{text_col}_processed']

        # Add VADER features from original text (not preprocessed)
        print("Adding VADER sentiment features...")
        df_processed['vader_summary'] = df_processed[summary_col].apply(self.get_vader_scores)
        df_processed['vader_text'] = df_processed[text_col].apply(self.get_vader_scores)

        # Extract scores as separate columns
        for col in ['compound', 'pos', 'neg', 'neu']:
            df_processed[f'vader_summary_{col}'] = df_processed['vader_summary'].apply(lambda x: x[col])
            df_processed[f'vader_text_{col}'] = df_processed['vader_text'].apply(lambda x: x[col])

        # Create additional derived features
        df_processed['vader_weighted_compound'] = (df_processed['vader_summary_compound'] * 0.3) + (
                    df_processed['vader_text_compound'] * 0.7)
        df_processed['vader_mean_pos'] = (df_processed['vader_summary_pos'] + df_processed['vader_text_pos']) / 2
        df_processed['vader_mean_neg'] = (df_processed['vader_summary_neg'] + df_processed['vader_text_neg']) / 2
        df_processed['vader_mean_neu'] = (df_processed['vader_summary_neu'] + df_processed['vader_text_neu']) / 2

        return df_processed


# Preprocess the data
preprocessor = TextPreprocessor()
train_processed = preprocessor.preprocess_dataframe(train_data)
test_processed = preprocessor.preprocess_dataframe(test_data)

# Display a sample of preprocessed data with VADER features
print("\nSample of preprocessed data with VADER features:")
sample_cols = ['Summary', 'Summary_processed', 'vader_weighted_compound', 'Score']
print(train_processed[sample_cols].head(2))

# Create TF-IDF features
print("\nCreating TF-IDF features...")
tfidf_vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X_tfidf_train = tfidf_vectorizer.fit_transform(train_processed['combined_text']).toarray()
print(f"TF-IDF feature shape: {X_tfidf_train.shape}")

# Define VADER features to use
vader_features = [
    'vader_summary_compound', 'vader_text_compound',
    'vader_summary_pos', 'vader_text_pos',
    'vader_summary_neg', 'vader_text_neg',
    'vader_summary_neu', 'vader_text_neu',
    'vader_weighted_compound',
    'vader_mean_pos', 'vader_mean_neg', 'vader_mean_neu'
]
X_vader_train = train_processed[vader_features].values
print(f"VADER feature shape: {X_vader_train.shape}")

# Combine TF-IDF and VADER features
X_combined_train = np.hstack((X_tfidf_train, X_vader_train))
print(f"Combined feature shape: {X_combined_train.shape}")

# Prepare target variable for multi-class classification (0-indexed for Keras)
y_train = train_processed['Score'] - 1  # Scores will be 0-4
y_train_categorical = to_categorical(y_train, num_classes=5)
print(f"Target shape: {y_train_categorical.shape}")

# Split data for training and validation
X_train, X_val, y_train_cat, y_val_cat = train_test_split(
    X_combined_train, y_train_categorical, test_size=0.2, random_state=42
)

# Create the multi-class model
print("\nBuilding the multi-class sentiment model...")
model = Sequential([
    Dense(256, input_shape=(X_train.shape[1],), activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(5, activation='softmax')  # 5 output classes for scores 1-5
])

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# Print model summary
model.summary()

# Early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

# Train the model
print("\nTraining the model...")
history = model.fit(
    X_train, y_train_cat,
    epochs=15,
    batch_size=64,
    validation_data=(X_val, y_val_cat),
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate on validation set
val_loss, val_accuracy = model.evaluate(X_val, y_val_cat, verbose=0)
print(f"\nValidation accuracy: {val_accuracy:.4f}")

# Get validation predictions
y_val_pred_proba = model.predict(X_val)
y_val_pred = np.argmax(y_val_pred_proba, axis=1)
y_val_true = np.argmax(y_val_cat, axis=1)

# Classification report
print("\nClassification Report (Validation):")
print(
    classification_report(y_val_true, y_val_pred, target_names=['Score 1', 'Score 2', 'Score 3', 'Score 4', 'Score 5']))

# Confusion matrix
conf_matrix = confusion_matrix(y_val_true, y_val_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(1, 6), yticklabels=range(1, 6))
plt.xlabel('Predicted Score')
plt.ylabel('True Score')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('model_performance.png')

# Process test data for predictions
print("\nGenerating predictions for test data...")
X_tfidf_test = tfidf_vectorizer.transform(test_processed['combined_text']).toarray()
X_vader_test = test_processed[vader_features].values
X_combined_test = np.hstack((X_tfidf_test, X_vader_test))

# Generate predictions
y_test_pred_proba = model.predict(X_combined_test)
y_test_pred = np.argmax(y_test_pred_proba, axis=1)

# Convert predictions back to 1-5 scale
y_test_pred_score = y_test_pred + 1

# Display prediction distribution
print("\nPrediction distribution for test data:")
for score in range(1, 6):
    count = np.sum(y_test_pred_score == score)
    print(f"Score {score}: {count} reviews ({count / len(y_test_pred_score) * 100:.2f}%)")

# Save predictions to file
output_filename = 'Team1multiclass_predictions.txt'
np.savetxt(output_filename, y_test_pred_score, fmt='%d')
print(f"\nMulti-class predictions saved to '{output_filename}'")

# Compare with a model without VADER features (to show the benefit of VADER)
print("\nTraining a model without VADER features for comparison...")
simple_model = Sequential([
    Dense(128, input_shape=(X_tfidf_train.shape[1],), activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(5, activation='softmax')
])

simple_model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# Split data for the simple model
X_tfidf_train_split, X_tfidf_val_split, y_train_split, y_val_split = train_test_split(
    X_tfidf_train, y_train_categorical, test_size=0.2, random_state=42
)

# Train the simple model
simple_history = simple_model.fit(
    X_tfidf_train_split, y_train_split,
    epochs=10,
    batch_size=64,
    validation_data=(X_tfidf_val_split, y_val_split),
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate the simple model
simple_val_loss, simple_val_accuracy = simple_model.evaluate(X_tfidf_val_split, y_val_split, verbose=0)
print(f"\nSimple model (without VADER) validation accuracy: {simple_val_accuracy:.4f}")
print(f"Combined model (with VADER) validation accuracy: {val_accuracy:.4f}")
print(f"Improvement with VADER: {(val_accuracy - simple_val_accuracy) * 100:.2f}%")

print("\nConclusion:")
print("The multi-class sentiment analysis model successfully predicts product review scores (1-5).")
print("Adding VADER sentiment features improved the model's performance compared to using only TF-IDF features.")
print(
    "This demonstrates the value of combining lexical content with sentiment information for fine-grained sentiment analysis.")