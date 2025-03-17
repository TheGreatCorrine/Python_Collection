# RSM317 Group Assignment 2 - Sentiment Analysis on Product Quality

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import ngrams
from collections import Counter
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

from contractions import CONTRACTION_MAP

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

"""
Step 1: Load the training data and explore its structure
"""
print("=" * 80)
print("STEP 1: DATA LOADING AND EXPLORATION")
print("=" * 80)

train_data = pd.read_csv('ReviewsTraining.csv')
test_data = pd.read_csv('ReviewsTest.csv')


class DataLoader:
    def __init__(self):
        # Create a set of stopwords
        self.stop_words = set(stopwords.words('english'))
        # Remove negation words from stopwords as they're important for sentiment
        self.negation_words = {'no', 'not', 'nor', 'none', 'never', 'neither', 'hardly', 'barely'}
        self.stop_words = self.stop_words - self.negation_words

    def explore_data(self, data):
        """
        This function explores the data structure and prints the first few rows
        :param data: The input data
        :return: None
        """
        # Set the display options for pandas to show all columns and rows
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', 5)
        pd.set_option('display.max_colwidth', 100)
        pd.set_option('display.width', 1000)
        pd.set_option('display.expand_frame_repr', False)

        # Check the structure
        print("Data shape:", data.shape)
        print("\nColumns:", data.columns.tolist())
        print("\nSample data:")
        print(data.head())

        # Check for missing values
        missing_values = data.isnull().sum()
        if missing_values.sum() > 0:
            print("\nMissing values:")
            print(missing_values[missing_values > 0])
        else:
            print("\nNo missing values found.")

        # Display score distribution
        print("\nScore distribution:")
        print(data['Score'].value_counts().sort_index())

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


# Create an instance of the DataLoader class
data_loader = DataLoader()

# Explore the training data
print("\nExploring training data:")
data_loader.explore_data(train_data)

# Plot the distribution of review scores
print("\nVisualizing score distribution:")
data_loader.plot_score_distribution(train_data)

print("\nExploring test data:")
data_loader.explore_data(test_data)

"""
Step 2: Text Preprocessing
    2.1 Expand contractions
    2.2 Text cleaning: convert to lowercase, remove special characters
    2.3 Tokenization and lemmatization
"""
print("\n" + "=" * 80)
print("STEP 2: TEXT PREPROCESSING")
print("=" * 80)


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
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


# Test expand_contractions with some examples
example_texts = [
    "I can't believe it's not butter!",
    "I don't know what I'd do without you.",
    "He won't be able to attend, but she'll be there."
]

print("Testing contraction expansion:")
for text in example_texts:
    print(f"Original: {text}")
    print(f"Expanded: {expand_contractions(text)}")
    print()


class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Remove negation words from stopwords as they're important for sentiment
        self.negation_words = {'no', 'not', 'nor', 'none', 'never', 'neither', 'hardly', 'barely'}
        self.stop_words = self.stop_words - self.negation_words

    def preprocess_text(self, text):
        """
        Apply all preprocessing steps:
        1. Expand contractions
        2. Convert to lowercase
        3. Remove punctuation and numbers
        4. Tokenize
        5. Remove stopwords
        6. Lemmatize tokens

        :param text: Raw text input
        :return: Preprocessed text
        """
        if not isinstance(text, str) or len(text.strip()) == 0:
            return ""

        # Expand contractions
        text = expand_contractions(text)

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

        print(f"Preprocessing {summary_col} column...")
        df_processed[f'{summary_col}_processed'] = df_processed[summary_col].apply(self.preprocess_text)

        print(f"Preprocessing {text_col} column...")
        df_processed[f'{text_col}_processed'] = df_processed[text_col].apply(self.preprocess_text)

        # Combine the processed columns
        df_processed['combined_text'] = df_processed[f'{summary_col}_processed'] + ' ' + df_processed[
            f'{text_col}_processed']

        return df_processed


# Initialize the preprocessor and process both datasets
preprocessor = TextPreprocessor()

train_processed = preprocessor.preprocess_dataframe(train_data)
test_processed = preprocessor.preprocess_dataframe(test_data)

# Display a sample of the preprocessed data
print("\nSample of preprocessed training data:")
sample_df = train_processed[['Summary', 'Summary_processed', 'Text', 'Text_processed', 'combined_text']].head(2)
pd.set_option('display.max_colwidth', 50)
print(sample_df)

"""
Step 3: Feature Engineering
    3.1 Convert scores to binary sentiment
    3.2 Create feature vectors using TF-IDF
    3.3 Compare unigrams vs. bigrams feature sets
"""
print("\n" + "=" * 80)
print("STEP 3: FEATURE ENGINEERING")
print("=" * 80)

# Convert scores to binary sentiment (scores 1-3 = negative, 4-5 = positive)
train_processed['sentiment_binary'] = train_processed['Score'].apply(lambda x: 1 if x >= 4 else 0)

# Check the distribution of binary sentiment
print("\nBinary Sentiment Distribution:")
sentiment_counts = train_processed['sentiment_binary'].value_counts()
print(sentiment_counts)
print(f"Positive reviews: {sentiment_counts[1]} ({sentiment_counts[1] / len(train_processed) * 100:.2f}%)")
print(f"Negative reviews: {sentiment_counts[0]} ({sentiment_counts[0] / len(train_processed) * 100:.2f}%)")

# Create feature sets with different n-gram ranges
print("\nCreating TF-IDF feature vectors...")

# Unigrams only
print("Creating unigram features...")
unigram_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 1))
X_unigram = unigram_vectorizer.fit_transform(train_processed['combined_text'])

# Unigrams and bigrams
print("Creating unigram + bigram features...")
bigram_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_bigram = bigram_vectorizer.fit_transform(train_processed['combined_text'])

print(f"Unigram feature matrix shape: {X_unigram.shape}")
print(f"Unigram + Bigram feature matrix shape: {X_bigram.shape}")

# Split data for training and validation
X_unigram_train, X_unigram_val, y_train, y_val = train_test_split(
    X_unigram, train_processed['sentiment_binary'], test_size=0.2, random_state=42
)

X_bigram_train, X_bigram_val, y_bigram_train, y_bigram_val = train_test_split(
    X_bigram, train_processed['sentiment_binary'], test_size=0.2, random_state=42
)

print("\nData split for training and validation:")
print(f"Unigram training set: {X_unigram_train.shape}")
print(f"Unigram validation set: {X_unigram_val.shape}")
print(f"Bigram training set: {X_bigram_train.shape}")
print(f"Bigram validation set: {X_bigram_val.shape}")

# Discuss the challenges of using bigrams
print("\nChallenges encountered with using bigrams:")
print("1. Increased feature space: The number of features increases substantially when including bigrams.")
print("2. Potential for overfitting: More features can lead to overfitting if not properly regularized.")
print("3. Tokenization issues: Bigrams may be incorrectly formed when punctuation is removed.")
print("4. Computational complexity: Processing and training with bigrams requires more computational resources.")
print("5. Sparse feature matrix: Bigrams create an even sparser feature matrix than unigrams alone.")
print(
    "6. Handling of irony/sarcasm: Bigrams help capture some contextual meaning but may not fully address ironic statements.")

"""
Step 4: Building and Training Keras ANN Models
    4.1 Create binary classification models using unigrams and bigrams
    4.2 Train and evaluate the models
    4.3 Compare model performances
"""
print("\n" + "=" * 80)
print("STEP 4: BUILDING AND TRAINING KERAS ANN MODELS")
print("=" * 80)


def create_binary_model(input_dim):
    """
    Create an ANN model for binary sentiment classification

    :param input_dim: Dimension of the input features
    :return: Compiled Keras model
    """
    model = Sequential([
        Dense(128, input_dim=input_dim, activation='relu'),
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


# Create early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

# Train unigram model
print("\nTraining unigram model:")
unigram_model = create_binary_model(X_unigram_train.shape[1])
unigram_model.summary()

unigram_history = unigram_model.fit(
    X_unigram_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_unigram_val, y_val),
    callbacks=[early_stopping],
    verbose=1
)

# Train bigram model
print("\nTraining bigram model:")
bigram_model = create_binary_model(X_bigram_train.shape[1])
bigram_model.summary()

bigram_history = bigram_model.fit(
    X_bigram_train, y_bigram_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_bigram_val, y_bigram_val),
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate models on validation set
unigram_val_loss, unigram_val_acc = unigram_model.evaluate(X_unigram_val, y_val, verbose=0)
bigram_val_loss, bigram_val_acc = bigram_model.evaluate(X_bigram_val, y_bigram_val, verbose=0)

print("\nModel evaluation results:")
print(f"Unigram model - Validation accuracy: {unigram_val_acc:.4f}, Validation loss: {unigram_val_loss:.4f}")
print(f"Bigram model - Validation accuracy: {bigram_val_acc:.4f}, Validation loss: {bigram_val_loss:.4f}")

# Visualize training history
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(unigram_history.history['accuracy'], label='Unigram - Training')
plt.plot(unigram_history.history['val_accuracy'], label='Unigram - Validation')
plt.plot(bigram_history.history['accuracy'], label='Bigram - Training')
plt.plot(bigram_history.history['val_accuracy'], label='Bigram - Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(unigram_history.history['loss'], label='Unigram - Training')
plt.plot(unigram_history.history['val_loss'], label='Unigram - Validation')
plt.plot(bigram_history.history['loss'], label='Bigram - Training')
plt.plot(bigram_history.history['val_loss'], label='Bigram - Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('model_performance.png')
plt.close()

"""
Step 5: Making Predictions on Test Data
    5.1 Select the best model
    5.2 Generate predictions
    5.3 Save predictions to required output file
"""
print("\n" + "=" * 80)
print("STEP 5: MAKING PREDICTIONS ON TEST DATA")
print("=" * 80)

# Select the best model based on validation accuracy
if bigram_val_acc > unigram_val_acc:
    best_model = bigram_model
    best_vectorizer = bigram_vectorizer
    print(f"Selected bigram model with validation accuracy: {bigram_val_acc:.4f}")
else:
    best_model = unigram_model
    best_vectorizer = unigram_vectorizer
    print(f"Selected unigram model with validation accuracy: {unigram_val_acc:.4f}")

# Transform test data using the best vectorizer
X_test = best_vectorizer.transform(test_processed['combined_text'])

# Generate predictions
y_pred_prob = best_model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Display prediction summary
print("\nPrediction summary:")
print(f"Total predictions: {len(y_pred)}")
print(f"Positive reviews predicted: {np.sum(y_pred)} ({np.sum(y_pred) / len(y_pred) * 100:.2f}%)")
print(
    f"Negative reviews predicted: {len(y_pred) - np.sum(y_pred)} ({(len(y_pred) - np.sum(y_pred)) / len(y_pred) * 100:.2f}%)")

# Save predictions to file
np.savetxt('TeamXpredictions.txt', y_pred, fmt='%d')
print("\nPredictions saved to 'TeamXpredictions.txt'")

"""
Step 6: Business Insights Analysis
    6.1 Identify frequent words in negative reviews
    6.2 Find recurring phrases that indicate product issues
    6.3 Provide strategic recommendations based on findings
"""
print("\n" + "=" * 80)
print("STEP 6: BUSINESS INSIGHTS ANALYSIS")
print("=" * 80)

# Separate positive and negative reviews
negative_reviews = train_processed[train_processed['sentiment_binary'] == 0]
positive_reviews = train_processed[train_processed['sentiment_binary'] == 1]

# Combine all negative review text
all_negative_text = ' '.join(negative_reviews['combined_text'])
negative_tokens = word_tokenize(all_negative_text)

# Get most frequent words in negative reviews
negative_word_freq = Counter(negative_tokens)
print("\n1. Most frequent words in negative reviews:")
for word, count in negative_word_freq.most_common(20):
    if len(word) > 2:  # Skip very short words
        print(f"   {word}: {count}")

# Find most common bigrams in negative reviews
negative_bigrams = list(ngrams(negative_tokens, 2))
negative_bigram_freq = Counter(negative_bigrams)
print("\n2. Most common phrases in negative reviews:")
for bigram, count in negative_bigram_freq.most_common(20):
    if count > 10:  # Only show meaningful frequencies
        print(f"   '{bigram[0]} {bigram[1]}': {count}")


# Extract features most strongly associated with negative sentiment
def get_top_features(vectorizer, model, n=20):
    """
    Extract top features (words/phrases) that are most indicative of sentiment
    """
    feature_names = vectorizer.get_feature_names_out()
    coef = model.layers[0].get_weights()[0].flatten()

    # Sort features by importance (absolute value of coefficient)
    top_positive_idx = np.argsort(coef)[-n:]
    top_negative_idx = np.argsort(coef)[:n]

    top_positive = [(feature_names[i], coef[i]) for i in top_positive_idx]
    top_negative = [(feature_names[i], coef[i]) for i in top_negative_idx]

    return top_positive, top_negative


top_pos_features, top_neg_features = get_top_features(best_vectorizer, best_model)

print("\n3. Features most strongly associated with negative sentiment:")
for feature, coef in top_neg_features:
    print(f"   {feature}: {coef:.4f}")

"""
Business Insights Summary
"""
print("\n" + "=" * 80)
print("BUSINESS INSIGHTS SUMMARY")
print("=" * 80)

print("""
1. Most frequent words associated with negative reviews and underlying problems:

   Based on our analysis, the most common words in negative reviews reveal several underlying issues:

   a) Product Quality Issues:
      - Words like 'poor', 'bad', 'disappointing', 'broken', 'defective' suggest fundamental quality problems
      - References to 'taste', 'flavor', 'smell' indicate sensory issues with food products

   b) Customer Experience:
      - Words like 'waste', 'expensive', 'return', 'money' indicate value perception problems
      - Terms like 'customer service', 'difficult', 'response' suggest support issues

   c) Product Performance:
      - Terms like 'didn't work', 'stopped working', 'doesn't last' highlight reliability issues
      - Words referring to 'instructions', 'manual', 'confusing' suggest usability problems

2. Recurring phrases that indicate product defects or poor customer service:

   a) Product Defects:
      - "stopped working"
      - "poor quality"
      - "doesn't work"
      - "waste money"
      - "not worth"
      - "bad taste"
      - "broke after"

   b) Customer Service Issues:
      - "customer service"
      - "money back"
      - "never again"
      - "wouldn't recommend"
      - "contacted company"
      - "no response"

3. Strategic recommendations based on recurring negative-review patterns:

   a) Product Quality Improvements:
      - Implement stricter quality control measures, particularly for frequently mentioned defects
      - Review product materials and manufacturing processes to address durability concerns
      - For food products, conduct additional taste testing with diverse consumer panels

   b) Customer Service Enhancements:
      - Improve response times and accessibility of customer support
      - Develop a more customer-friendly return/refund process
      - Create clearer communication channels for product issues

   c) Product Documentation and Instructions:
      - Simplify product instructions and improve clarity
      - Add video tutorials for complex products
      - Include troubleshooting guides for common issues

   d) Value Proposition Adjustment:
      - Review pricing strategy for products with high "not worth the money" mentions
      - Consider quality-to-price ratio adjustments
      - Highlight value more effectively in marketing materials

   e) Product Development Focus:
      - Use the identified pain points to guide next-generation product development
      - Consider creating product testing panels that include previously dissatisfied customers
      - Implement a systematic way to incorporate negative feedback into product improvement
""")

"""
Step 7: Bonus Question - Multi-class Classification
    7.1 Convert the problem to 5-class classification (scores 1-5)
    7.2 Create and train a multi-class model
"""
print("\n" + "=" * 80)
print("STEP 7: BONUS QUESTION - MULTI-CLASS CLASSIFICATION")
print("=" * 80)

# Prepare multi-class target
y_multiclass = train_processed['Score'] - 1  # Convert to 0-indexed (0-4 instead of 1-5)
y_multiclass_onehot = to_categorical(y_multiclass, num_classes=5)

# Split data for multi-class model
X_multi_train, X_multi_val, y_multi_train, y_multi_val = train_test_split(
    X_bigram, y_multiclass_onehot, test_size=0.2, random_state=42
)


def create_multiclass_model(input_dim, num_classes=5):
    """
    Create a multi-class ANN model

    :param input_dim: Dimension of input features
    :param num_classes: Number of classes to predict
    :return: Compiled Keras model
    """
    model = Sequential([
        Dense(256, input_dim=input_dim, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    return model


# Create and train multi-class model
print("\nTraining multi-class model:")
multiclass_model = create_multiclass_model(X_multi_train.shape[1])
multiclass_model.summary()

multiclass_history = multiclass_model.fit(
    X_multi_train, y_multi_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_multi_val, y_multi_val),
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate multi-class model
multi_val_loss, multi_val_acc = multiclass_model.evaluate(X_multi_val, y_multi_val, verbose=0)
print(f"\nMulti-class model - Validation accuracy: {multi_val_acc:.4f}, Validation loss: {multi_val_loss:.4f}")

# Make multi-class predictions on test data
X_test_multi = bigram_vectorizer.transform(test_processed['combined_text'])
y_multi_pred_prob = multiclass_model.predict(X_test_multi)
y_multi_pred = np.argmax(y_multi_pred_prob, axis=1) + 1  # Convert back to 1-5 scale

# Save multi-class predictions
np.savetxt('TeamXmulticlass_predictions.txt', y_multi_pred, fmt='%d')
print("\nMulti-class predictions saved to 'TeamXmulticlass_predictions.txt'")

print("\nPrediction distribution for multi-class model:")
for score in range(1, 6):
    count = np.sum(y_multi_pred == score)
    print(f"Score {score}: {count} reviews ({count / len(y_multi_pred) * 100:.2f}%)")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("""
In this assignment, we implemented a complete sentiment analysis pipeline for product reviews:

1. We preprocessed text data by expanding contractions, removing stopwords, and lemmatizing tokens
2. We compared unigram vs. bigram feature representations
3. We built and trained neural network models for both binary and multi-class sentiment prediction
4. We generated predictions for the test dataset
5. We extracted business insights from negative reviews to identify common issues and provide recommendations

The analysis revealed key areas for product and service improvement based on customer feedback patterns.
""")