# RSM317 Group Assignment 2 - Sentiment Analysis on Product Quality

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import ngrams
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

`import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping`
from tensorflow.keras.utils import to_categorical

from contractions import CONTRACTION_MAP

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

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
        self.negation_words = {'no', 'not', 'nor', 'none', 'never', 'neither', 'hardly', 'barely', 'rarely', 'seldom'}
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
data_loader.plot_score_distribution(train_data)

print("\nExploring test data:")
data_loader.explore_data(test_data)

"""
Step 2: Text Preprocessing
    2.1 Expand contractions
    2.2 Text cleaning with focus on maintaining sentiment indicators
    2.3 Handle special patterns relevant to sentiment
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


class SentimentAwarePreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # Keep negation and intensity words that are important for sentiment
        self.sentiment_words = {
            'no', 'not', 'nor', 'none', 'never', 'neither', 'hardly', 'barely', 'rarely', 'seldom',
            'very', 'extremely', 'absolutely', 'completely', 'totally', 'utterly', 'really',
            'quite', 'rather', 'somewhat', 'almost', 'just', 'nearly', 'virtually'
        }
        self.stop_words = self.stop_words - self.sentiment_words

        # Sentiment-specific lexicons
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'outstanding',
            'superb', 'awesome', 'terrific', 'fabulous', 'perfect', 'best', 'love', 'impressive',
            'exceptional', 'delightful', 'pleasant', 'brilliant', 'remarkable', 'splendid'
        }

        self.negative_words = {
            'bad', 'poor', 'terrible', 'awful', 'horrible', 'disappointing', 'worst', 'fail',
            'mediocre', 'inferior', 'inadequate', 'useless', 'defective', 'faulty', 'broken',
            'pathetic', 'dreadful', 'substandard', 'abysmal', 'atrocious', 'hate'
        }

        # Irony and sarcasm detection patterns
        self.irony_patterns = [
            r'yeah right',
            r'sure+\W+as if',
            r'right\W+like',
            r'exactly\W+not',
            r'of course\W+not',
            r'so\W+called',
            r'supposedly',
            r'allegedly'
        ]

    def detect_special_patterns(self, text):
        """
        Detect special patterns that indicate sentiment, irony, or sarcasm
        """
        has_irony = False

        # Check for irony/sarcasm patterns
        for pattern in self.irony_patterns:
            if re.search(pattern, text.lower()):
                has_irony = True
                break

        # Check for quotes which might indicate non-literal meaning
        quotes_count = text.count('"') + text.count("'")
        has_quotes = quotes_count >= 2

        # Check for excessive punctuation which might indicate strong sentiment
        exclamation_count = text.count('!')
        question_count = text.count('?')
        has_emphasis = exclamation_count > 1 or question_count > 1

        # Check for ALL CAPS words which might indicate emphasis
        words = re.findall(r'\b[A-Z]{2,}\b', text)
        has_caps = len(words) > 0

        return {
            'has_irony': has_irony,
            'has_quotes': has_quotes,
            'has_emphasis': has_emphasis,
            'has_caps': has_caps,
            'exclamation_count': exclamation_count,
            'question_count': question_count
        }

    def clean_text(self, text):
        """
        Performs sentiment-aware text cleaning
        """
        if not isinstance(text, str):
            return ""

        # First capture special patterns before cleaning
        special_patterns = self.detect_special_patterns(text)

        # Convert to lowercase but preserve detected special patterns
        text = text.lower()

        # Remove URLs and HTML tags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'<.*?>', '', text)

        # Extract emoticons before removing punctuation (as they contain punctuation)
        emoticons_happy = set([
            ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
            ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
            '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
            'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
            '<3'
        ])
        emoticons_sad = set([
            ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
            ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
            ':c', ':{', '>:\\', ';('
        ])

        # Count emoticons
        happy_count = sum(1 for emoticon in emoticons_happy if emoticon in text)
        sad_count = sum(1 for emoticon in emoticons_sad if emoticon in text)

        # Handle negations (e.g., "not good" -> "not_good")
        negation_words = {'no', 'not', 'never', 'none', 'nobody', 'nowhere', 'neither', 'nor'}
        words = text.split()
        negated = False
        result = []
        for i, word in enumerate(words):
            # Check if this word is a negation
            if word in negation_words or word.endswith("n't"):
                negated = True
                result.append(word)
            # If we're in a negation context, attach 'NEG_' to sentiment words
            elif negated and (word in self.positive_words or word in self.negative_words):
                result.append('NEG_' + word)
            else:
                result.append(word)
                # Reset negation after punctuation
                if any(p in word for p in '.!?,:;'):
                    negated = False

        text = ' '.join(result)

        # Replace multi-exclamation/question with single versions but note the count
        text = re.sub(r'!+', ' EXCLAM ', text)
        text = re.sub(r'\?+', ' QMARK ', text)

        # Remove punctuation except in emoticons
        punct_to_remove = string.punctuation.replace('_', '')  # Keep underscore for negations
        text = ''.join(c if c not in punct_to_remove else ' ' for c in text)

        # Remove numbers but keep emoticons
        text = re.sub(r'\d+', ' NUM ', text)

        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()

        # Add extracted features back to text as special tokens
        if special_patterns['has_irony']:
            text += ' HAS_IRONY'
        if special_patterns['has_quotes']:
            text += ' HAS_QUOTES'
        if special_patterns['has_emphasis']:
            text += ' HAS_EMPHASIS'
        if special_patterns['has_caps']:
            text += ' HAS_CAPS'

        # Add emoticon sentiment
        if happy_count > 0:
            text += f' HAPPY_EMOTICON'
        if sad_count > 0:
            text += f' SAD_EMOTICON'

        if special_patterns['exclamation_count'] > 0:
            text += f' EXCLAM_COUNT'
        if special_patterns['question_count'] > 0:
            text += f' QMARK_COUNT'

        return text

    def remove_stopwords(self, text):
        """
        Removes stopwords but preserves sentiment-related words
        """
        if not isinstance(text, str):
            return ""

        tokens = word_tokenize(text)
        filtered_tokens = [token for token in tokens if token.lower() not in self.stop_words]
        return ' '.join(filtered_tokens)

    def lemmatize_text(self, text):
        """
        Lemmatizes text but preserves sentiment markers
        """
        if not isinstance(text, str):
            return ""

        tokens = word_tokenize(text)

        # Don't lemmatize special tokens
        special_tokens = {'HAS_IRONY', 'HAS_QUOTES', 'HAS_EMPHASIS', 'HAS_CAPS',
                          'HAPPY_EMOTICON', 'SAD_EMOTICON', 'EXCLAM_COUNT', 'QMARK_COUNT',
                          'EXCLAM', 'QMARK', 'NUM'}

        lemmatized_tokens = []
        for token in tokens:
            # Don't lemmatize negated words or special tokens
            if token.startswith('NEG_') or token in special_tokens:
                lemmatized_tokens.append(token)
            else:
                lemmatized_tokens.append(self.lemmatizer.lemmatize(token))

        return ' '.join(lemmatized_tokens)

    def preprocess(self, text, expand_contractions_flag=True, remove_stopwords_flag=True, lemmatize_flag=True):
        """
        Full preprocessing pipeline with sentiment awareness
        """
        if not isinstance(text, str):
            return ""

        # Expand contractions
        if expand_contractions_flag:
            text = expand_contractions(text)

        # Clean text while preserving sentiment markers
        text = self.clean_text(text)

        # Remove stopwords
        if remove_stopwords_flag:
            text = self.remove_stopwords(text)

        # Lemmatize
        if lemmatize_flag:
            text = self.lemmatize_text(text)

        return text


# Initialize the text preprocessor
preprocessor = SentimentAwarePreprocessor()

# Apply preprocessing to the datasets
print("Preprocessing training data...")
train_data['processed_summary'] = train_data['Summary'].apply(preprocessor.preprocess)
train_data['processed_text'] = train_data['Text'].apply(preprocessor.preprocess)

# Display some examples to verify preprocessing
print("\nExample of preprocessed text:")
for i in range(2):
    print(f"Original Summary: {train_data['Summary'].iloc[i]}")
    print(f"Processed Summary: {train_data['processed_summary'].iloc[i]}")
    print(f"Original Text: {train_data['Text'].iloc[i][:100]}...")
    print(f"Processed Text: {train_data['processed_text'].iloc[i][:100]}...")
    print("-" * 80)

# Apply preprocessing to test data
print("\nPreprocessing test data...")
test_data['processed_summary'] = test_data['Summary'].apply(preprocessor.preprocess)
test_data['processed_text'] = test_data['Text'].apply(preprocessor.preprocess)

# Combine summary and text
train_data['combined_text'] = train_data['processed_summary'] + ' ' + train_data['processed_text']
test_data['combined_text'] = test_data['processed_summary'] + ' ' + test_data['processed_text']

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
train_data['sentiment_binary'] = train_data['Score'].apply(lambda x: 1 if x >= 4 else 0)

# Check the distribution of binary sentiment
print("\nBinary Sentiment Distribution:")
sentiment_counts = train_data['sentiment_binary'].value_counts()
print(sentiment_counts)
print(f"Positive reviews: {sentiment_counts[1]} ({sentiment_counts[1] / len(train_data) * 100:.2f}%)")
print(f"Negative reviews: {sentiment_counts[0]} ({sentiment_counts[0] / len(train_data) * 100:.2f}%)")


# Identify most common words in positive and negative reviews
def get_top_n_words(corpus, n=20):
    """
    Extract the top N words from a corpus
    """
    all_words = ' '.join(corpus).split()
    # Count word frequencies
    word_counts = Counter(all_words)
    # Return most common words
    return word_counts.most_common(n)


# Get top words for positive and negative reviews
positive_reviews = train_data[train_data['sentiment_binary'] == 1]['combined_text']
negative_reviews = train_data[train_data['sentiment_binary'] == 0]['combined_text']

print("\nTop words in positive reviews:")
for word, count in get_top_n_words(positive_reviews):
    print(f"  {word}: {count}")

print("\nTop words in negative reviews:")
for word, count in get_top_n_words(negative_reviews):
    print(f"  {word}: {count}")


# Extract bigrams from text
def get_top_n_bigrams(corpus, n=20):
    """
    Extract the top N bigrams from a corpus
    """
    bigrams_list = []
    for text in corpus:
        tokens = text.split()
        # Create bigrams
        bigrams_list.extend(list(ngrams(tokens, 2)))
    # Count bigram frequencies
    bigram_counts = Counter(bigrams_list)
    # Return most common bigrams
    return bigram_counts.most_common(n)


print("\nTop bigrams in positive reviews:")
for bigram, count in get_top_n_bigrams(positive_reviews):
    print(f"  {bigram[0]} {bigram[1]}: {count}")

print("\nTop bigrams in negative reviews:")
for bigram, count in get_top_n_bigrams(negative_reviews):
    print(f"  {bigram[0]} {bigram[1]}: {count}")

# Feature extraction using TF-IDF
print("\nCreating feature vectors...")

# Unigrams only
print("Creating unigram features...")
unigram_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 1))
X_unigram = unigram_vectorizer.fit_transform(train_data['combined_text'])

# Unigrams and bigrams
print("Creating unigram + bigram features...")
bigram_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_bigram = bigram_vectorizer.fit_transform(train_data['combined_text'])

print(f"Unigram feature matrix shape: {X_unigram.shape}")
print(f"Unigram + Bigram feature matrix shape: {X_bigram.shape}")

# Split the data for training and validation
X_unigram_train, X_unigram_val, y_train_unigram, y_val_unigram = train_test_split(
    X_unigram, train_data['sentiment_binary'], test_size=0.2, random_state=42
)

X_bigram_train, X_bigram_val, y_train_bigram, y_val_bigram = train_test_split(
    X_bigram, train_data['sentiment_binary'], test_size=0.2, random_state=42
)

print("\nData split for training and validation:")
print(f"Training set: {X_unigram_train.shape[0]} samples")
print(f"Validation set: {X_unigram_val.shape[0]} samples")

# Discuss challenges with using bigrams
print("\nChallenges encountered with using bigrams:")
print("1. Increased dimensionality: Bigrams significantly increase the feature space.")
print("2. Sparsity: Bigram matrices are even sparser than unigram matrices.")
print("3. Overfitting risk: More features can lead to model overfitting.")
print("4. Computational cost: Training with bigrams requires more memory and processing time.")
print("5. Context limitations: Bigrams only capture immediate word pairs, not broader context.")
print("6. Tokenization issues: Punctuation removal can create artificial bigrams that don't exist in original text.")
print("7. Irony/sarcasm: Bigrams may not fully capture ironic expressions like 'yeah right'.")

"""
Step 4: Model Building and Training
    4.1 Build unigram model
    4.2 Build bigram model
    4.3 Compare model performance
"""
print("\n" + "=" * 80)
print("STEP 4: MODEL BUILDING AND TRAINING")
print("=" * 80)


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

# Evaluate unigram model
y_pred_unigram = (unigram_model.predict(X_unigram_val) > 0.5).astype(int)
unigram_accuracy = accuracy_score(y_val_unigram, y_pred_unigram)
print("\nUnigram model evaluation:")
print(f"Accuracy: {unigram_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_val_unigram, y_pred_unigram))

# Evaluate bigram model
y_pred_bigram = (bigram_model.predict(X_bigram_val) > 0.5).astype(int)
bigram_accuracy = accuracy_score(y_val_bigram, y_pred_bigram)
print("\nBigram model evaluation:")
print(f"Accuracy: {bigram_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_val_bigram, y_pred_bigram))

# Plot training history
plt.figure(figsize=(12, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history_unigram.history['accuracy'], label='Unigram Train')
plt.plot(history_unigram.history['val_accuracy'], label='Unigram Val')
plt.plot(history_bigram.history['accuracy'], label='Bigram Train')
plt.plot(history_bigram.history['val_accuracy'], label='Bigram Val')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history_unigram.history['loss'], label='Unigram Train')
plt.plot(history_unigram.history['val_loss'], label='Unigram Val')
plt.plot(history_bigram.history['loss'], label='Bigram Train')
plt.plot(history_bigram.history['val_loss'], label='Bigram Val')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.savefig('model_performance.png')
plt.show()

# Compare model performance
print("\nModel performance comparison:")
print(f"Unigram model accuracy: {unigram_accuracy:.4f}")
print(f"Bigram model accuracy: {bigram_accuracy:.4f}")

# Choose the best model
best_model = bigram_model if bigram_accuracy > unigram_accuracy else unigram_model
best_vectorizer = bigram_vectorizer if bigram_accuracy > unigram_accuracy else unigram_vectorizer
print(f"\nSelected {'bigram' if bigram_accuracy > unigram_accuracy else 'unigram'} model as the best model.")

"""
Step 5: Generating Predictions for Test Data
"""
print("\n" + "=" * 80)
print("STEP 5: GENERATING PREDICTIONS FOR TEST DATA")
print("=" * 80)

# Transform test data
X_test = best_vectorizer.transform(test_data['combined_text'])

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
print("\nPredictions saved to 'Team1predictions.txt'")

"""
Step 6: Business Insights Analysis
    6.1 Identify frequent words in negative reviews
    6.2 Find recurring phrases that indicate product issues
    6.3 Provide strategic recommendations
"""
print("\n" + "=" * 80)
print("STEP 6: BUSINESS INSIGHTS ANALYSIS")
print("=" * 80)

# Get most frequent words in negative reviews
negative_text = ' '.join(negative_reviews)
negative_words = negative_text.split()
negative_word_freq = Counter(negative_words)

print("\n1. Most frequent words in negative reviews:")
for word, count in negative_word_freq.most_common(20):
    if len(word) > 2:  # Skip very short words
        print(f"   {word}: {count}")

# Analyze common bigrams in negative reviews
negative_bigrams = list(ngrams(' '.join(negative_reviews).split(), 2))
negative_bigram_freq = Counter(negative_bigrams)

print("\n2. Most common phrases in negative reviews:")
for bigram, count in negative_bigram_freq.most_common(20):
    print(f"   '{bigram[0]} {bigram[1]}': {count}")


# Extract important features for negative sentiment
def get_important_features(vectorizer, model, class_index=0, top_n=20):
    """
    Extract features most strongly associated with a particular class
    """
    feature_names = vectorizer.get_feature_names_out()
    weights = model.layers[0].get_weights()[0]

    # For binary classification, use weights for the output neuron
    if weights.shape[1] == 1:
        feature_weights = weights.flatten()
    else:
        # For multi-class, use weights for the specified class
        feature_weights = weights[:, class_index]

    # Get indices of top features
    top_indices = np.args