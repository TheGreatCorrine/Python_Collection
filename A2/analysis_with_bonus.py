# Group 6 - Corrine and Jessica
# Complete implementation with both binary and multi-class prediction
# We reimplement (customize )the entire pipeline with Vader sentiment analysis tools
# The binary model has an accuracy of 0.90 and the multi-class model has an accuracy of 0.69
# Just click on 'run' and you will see the results of the analysis in the terminal.


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
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from contractions import CONTRACTION_MAP

# Custom VADER Sentiment Analyzer with domain-specific lexicon augmentation
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class EnhancedVaderAnalyzer(SentimentIntensityAnalyzer):
    """
    Enhanced VADER sentiment analyzer with domain-specific additions for product reviews
    """

    def __init__(self):
        # Initialize with default lexicon loading
        super().__init__()

        # Add product review specific booster words
        self.constants.BOOSTER_DICT.update({
            "very": 0.35,
            "extremely": 0.35,
            "incredibly": 0.35,
            "really": 0.35,
            "so": 0.35,
            "too": 0.35,
            "totally": 0.35,
            "absolutely": 0.35,
            "completely": 0.35,
            "definitely": 0.35,
            "highly": 0.35,
            "strongly": 0.35
        })

        # Add domain-specific words for product reviews
        # These words and values are based on common product review terms
        self.lexicon.update({
            # Product quality related
            'excellent': 3.0,
            'amazing': 3.0,
            'awesome': 3.0,
            'fantastic': 3.0,
            'wonderful': 3.0,
            'superb': 3.0,
            'outstanding': 3.0,
            'perfect': 3.0,
            'best': 3.0,
            'great': 2.5,
            'good': 2.0,
            'nice': 1.5,
            'decent': 1.0,
            'acceptable': 0.5,
            'mediocre': -0.5,
            'disappointing': -2.0,
            'poor': -2.5,
            'terrible': -3.0,
            'horrible': -3.0,
            'awful': -3.0,
            'worst': -3.0,

            # Product defects
            'defective': -2.5,
            'broken': -2.5,
            'damaged': -2.0,
            'faulty': -2.5,
            'useless': -2.5,
            'unusable': -2.5,

            # Value for money
            'expensive': -1.5,
            'overpriced': -2.0,
            'cheap': -1.0,  # Could be positive or negative depending on context
            'bargain': 2.0,
            'worth': 1.5,
            'value': 1.5,
            'waste': -2.5,

            # Common product review phrases
            'highly recommend': 3.0,
            'recommend': 2.0,
            'would recommend': 2.0,
            'not recommend': -2.0,
            'would not recommend': -2.5,
            'dont recommend': -2.5,
            'do not recommend': -2.5,
            'works great': 2.5,
            'works well': 2.0,
            'works perfectly': 3.0,
            'doesnt work': -2.5,
            'does not work': -2.5,
            'stopped working': -2.5,
            'broke': -2.0,
            'returned': -1.5,
            'return': -1.0,
            'refund': -1.0,
            'money back': -1.5,
            'disappointed': -2.0,
            'disappointment': -2.0,
            'satisfactory': 1.0,
            'satisfied': 1.5,
            'dissatisfied': -2.0,
            'happy': 2.0,
            'unhappy': -2.0,
            'impressed': 2.0,
            'unimpressed': -1.5,
            'love': 3.0,
            'hate': -3.0,

            # Food product specific
            'delicious': 3.0,
            'tasty': 2.5,
            'yummy': 2.5,
            'flavor': 0.0,  # Neutral without context
            'taste': 0.0,  # Neutral without context
            'bland': -1.5,
            'stale': -2.0,
            'fresh': 2.0,
            'expired': -2.5,
            'spoiled': -2.5,

            # Tech product specific
            'fast': 2.0,
            'slow': -1.5,
            'responsive': 2.0,
            'unresponsive': -2.0,
            'lag': -1.5,
            'crash': -2.0,
            'crashed': -2.0,
            'bug': -1.5,
            'buggy': -2.0,
            'glitch': -1.5,
            'user friendly': 2.0,
            'user unfriendly': -2.0,
            'intuitive': 2.0,
            'unintuitive': -1.5,
            'complicated': -1.0,
            'simple': 1.5,

            # Customer service
            'service': 0.0,  # Neutral without context
            'customer service': 0.0,  # Neutral without context
            'support': 0.0,  # Neutral without context
            'helpful': 2.0,
            'unhelpful': -2.0,
            'responsive': 2.0,
            'unresponsive': -2.0,
            'rude': -2.5,
            'polite': 2.0,
        })

    def score_review(self, summary, text):
        """
        Score a product review using both summary and text

        Args:
            summary: The review summary/title
            text: The main review text

        Returns:
            Dict containing sentiment scores and combined score
        """
        # Get scores for summary and text
        if pd.isna(summary) or summary == '':
            summary_scores = {'compound': 0, 'pos': 0, 'neg': 0, 'neu': 1.0}
        else:
            summary_scores = self.polarity_scores(summary)

        if pd.isna(text) or text == '':
            text_scores = {'compound': 0, 'pos': 0, 'neg': 0, 'neu': 1.0}
        else:
            text_scores = self.polarity_scores(text)

        # Create weighted compound score (summary often has more concentrated sentiment)
        weighted_compound = (summary_scores['compound'] * 0.4) + (text_scores['compound'] * 0.6)

        # Calculate sentiment features
        result = {
            'summary_compound': summary_scores['compound'],
            'summary_pos': summary_scores['pos'],
            'summary_neg': summary_scores['neg'],
            'summary_neu': summary_scores['neu'],
            'text_compound': text_scores['compound'],
            'text_pos': text_scores['pos'],
            'text_neg': text_scores['neg'],
            'text_neu': text_scores['neu'],
            'weighted_compound': weighted_compound,
            'sentiment_diff': abs(summary_scores['compound'] - text_scores['compound']),
            'max_sentiment': max(summary_scores['compound'], text_scores['compound']),
            'min_sentiment': min(summary_scores['compound'], text_scores['compound'])
        }

        return result


class TextPreprocessor:
    def __init__(self):
        # Initialize lemmatizer
        self.lemmatizer = WordNetLemmatizer()

        # Get stopwords but keep negation words
        self.stop_words = set(stopwords.words('english'))
        self.negation_words = {'no', 'not', 'nor', 'none', 'never', 'neither', 'hardly', 'barely'}
        self.stop_words = self.stop_words - self.negation_words

        # Initialize enhanced VADER analyzer
        self.vader = EnhancedVaderAnalyzer()

    def expand_contractions(self, text, contraction_mapping=CONTRACTION_MAP):
        """
        Expand contractions in the text (e.g., "don't" -> "do not")
        """
        if not isinstance(text, str) or len(text.strip()) == 0:
            return ""

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

    def preprocess_text(self, text):
        """
        Apply full preprocessing pipeline to text
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
        processed_text = ' '.join(tokens)

        return processed_text

    def preprocess_dataframe(self, df, text_col='Text', summary_col='Summary'):
        """
        Preprocess both text and summary columns and add VADER features

        IMPORTANT: Uses raw text (with contractions) for VADER analysis
        and processed text (with expanded contractions) for TF-IDF features
        """
        df_processed = df.copy()

        # Process text columns for TF-IDF features
        df_processed[f'{summary_col}_processed'] = df_processed[summary_col].apply(self.preprocess_text)

        df_processed[f'{text_col}_processed'] = df_processed[text_col].apply(self.preprocess_text)

        # Combine processed columns
        df_processed['combined_text'] = df_processed[f'{summary_col}_processed'] + ' ' + df_processed[
            f'{text_col}_processed']

        vader_scores = []

        for _, row in df.iterrows():
            # Use original text with contractions for VADER
            scores = self.vader.score_review(row[summary_col], row[text_col])
            vader_scores.append(scores)

        # Convert VADER scores to DataFrame columns
        vader_df = pd.DataFrame(vader_scores)

        # Concatenate with processed DataFrame
        df_processed = pd.concat([df_processed, vader_df], axis=1)

        return df_processed


class SentimentModel:
    def __init__(self, use_vader=True):
        self.use_vader = use_vader
        self.binary_model = None
        self.multiclass_model = None
        self.tfidf_vectorizer = None
        self.vader_features = [
            'summary_compound', 'text_compound',
            'summary_pos', 'text_pos',
            'summary_neg', 'text_neg',
            'summary_neu', 'text_neu',
            'weighted_compound', 'sentiment_diff',
            'max_sentiment', 'min_sentiment'
        ]

        # Weight VADER features more heavily for improved accuracy
        self.vader_weight = 2.0

    def create_feature_matrix(self, df, is_training=True):
        """
        Create feature matrix from preprocessed DataFrame
        """
        # TF-IDF features
        if is_training:
            # Use a smaller set of features to reduce noise and prevent overfitting
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=3000,  # Reduced from 5000
                ngram_range=(1, 2),
                min_df=5,  # Ignore terms that appear in less than 5 documents
                max_df=0.9,  # Ignore terms that appear in more than 90% of documents
                sublinear_tf=True  # Apply sublinear tf scaling (1+log(tf))
            )
            X_tfidf = self.tfidf_vectorizer.fit_transform(df['combined_text'])
        else:
            X_tfidf = self.tfidf_vectorizer.transform(df['combined_text'])

        if self.use_vader:
            # VADER features - these are more reliable for sentiment
            X_vader = df[self.vader_features].values

            # Enhance the impact of VADER features by repeating them
            # This effectively gives VADER features higher weight
            X_vader_weighted = np.repeat(X_vader, int(self.vader_weight), axis=1)

            # Combine TF-IDF and weighted VADER features
            X = np.hstack((X_tfidf.toarray(), X_vader_weighted))
        else:
            X = X_tfidf.toarray()

        return X

    def create_binary_model(self, input_dim):
        """
        Create binary classification model
        """
        model = Sequential([
            Dense(256, input_dim=input_dim, activation='relu'),
            Dropout(0.4),  # Increased dropout
            Dense(128, activation='relu'),
            Dropout(0.4),  # Increased dropout
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])

        # Using a smaller learning rate for better convergence
        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(learning_rate=0.0005),
            metrics=['accuracy']
        )

        return model

    def create_multiclass_model(self, input_dim, num_classes=5):
        """
        Create multi-class classification model with improved architecture
        """
        model = Sequential([
            # Wider first layer
            Dense(384, input_dim=input_dim, activation='relu'),
            Dropout(0.4),
            # Additional layer
            Dense(196, activation='relu'),
            Dropout(0.4),
            Dense(96, activation='relu'),
            Dropout(0.3),
            Dense(48, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])

        # Using a smaller learning rate and different optimizer for better performance
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=0.0005),
            metrics=['accuracy']
        )

        return model

    def train_binary_model(self, X, y, validation_split=0.2, epochs=10, batch_size=64):
        """
        Train binary sentiment model
        """
        # Split data for training and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )

        # Create model
        self.binary_model = self.create_binary_model(X_train.shape[1])

        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        )

        # Train model
        history = self.binary_model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=1
        )

        # Evaluate on validation set
        val_loss, val_accuracy = self.binary_model.evaluate(X_val, y_val, verbose=0)
        print(f"Binary model - Validation accuracy: {val_accuracy:.4f}")

        # Get validation predictions
        y_val_pred = (self.binary_model.predict(X_val) > 0.5).astype(int).flatten()

        # Classification report
        print("\nBinary Classification Report (Validation):")
        print(classification_report(y_val, y_val_pred, target_names=['Negative', 'Positive']))

        return history

    def train_multiclass_model(self, X, y, validation_split=0.2, epochs=15, batch_size=64):
        """
        Train multi-class sentiment model
        """
        # Convert labels to categorical format
        y_cat = to_categorical(y, num_classes=5)

        # Split data for training and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_cat, test_size=validation_split, random_state=42
        )

        # Create model
        self.multiclass_model = self.create_multiclass_model(X_train.shape[1])

        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        )

        # Train model
        history = self.multiclass_model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=1
        )

        # Evaluate on validation set
        val_loss, val_accuracy = self.multiclass_model.evaluate(X_val, y_val, verbose=0)
        print(f"Multi-class model - Validation accuracy: {val_accuracy:.4f}")

        # Get validation predictions
        y_val_pred_proba = self.multiclass_model.predict(X_val)
        y_val_pred = np.argmax(y_val_pred_proba, axis=1)
        y_val_true = np.argmax(y_val, axis=1)

        # Classification report
        print("\nMulti-class Classification Report (Validation):")
        print(classification_report(y_val_true, y_val_pred,
                                    target_names=['Score 1', 'Score 2', 'Score 3', 'Score 4', 'Score 5']))

        # Plot confusion matrix
        conf_matrix = confusion_matrix(y_val_true, y_val_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=range(1, 6), yticklabels=range(1, 6))
        plt.xlabel('Predicted Score')
        plt.ylabel('True Score')
        plt.title('Multi-class Confusion Matrix')
        plt.savefig('Bonus_multiclass_confusion_matrix.png')
        plt.close()

        return history

    def predict_binary(self, X):
        """
        Make binary predictions
        """
        if self.binary_model is None:
            raise ValueError("Binary model has not been trained yet.")

        # Generate predictions
        y_pred_prob = self.binary_model.predict(X)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()

        return y_pred

    def predict_multiclass(self, X):
        """
        Make multi-class predictions
        """
        if self.multiclass_model is None:
            raise ValueError("Multi-class model has not been trained yet.")

        # Generate predictions
        y_pred_proba = self.multiclass_model.predict(X)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Convert 0-indexed to 1-indexed (scores 1-5)
        y_pred = y_pred + 1

        return y_pred


def analyze_negative_reviews(df, preprocessor):
    """
    Analyze negative reviews to extract business insights
    """
    print("\n" + "=" * 80)
    print("BUSINESS INSIGHTS ANALYSIS")
    print("=" * 80)

    # Separate positive and negative reviews (1-3 = negative, 4-5 = positive)
    df['sentiment_binary'] = df['Score'].apply(lambda x: 1 if x >= 4 else 0)
    negative_reviews = df[df['sentiment_binary'] == 0]

    print(f"Analyzing {len(negative_reviews)} negative reviews...")

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

    # Average VADER scores for negative reviews
    avg_neg_compound = negative_reviews['weighted_compound'].mean()
    print(f"\nAverage VADER compound score for negative reviews: {avg_neg_compound:.4f}")

    # Group by score to see patterns
    score_groups = df.groupby('Score')
    score_stats = score_groups['weighted_compound'].agg(['mean', 'std'])
    print("\nVADER compound score statistics by Score:")
    print(score_stats)


    return negative_word_freq, negative_bigram_freq


def main():
    """
    Main function to execute the complete sentiment analysis pipeline
    Just click on 'run' and you will see the results of the analysis in the terminal.
    """
    print("=" * 80)
    print("BONUS - Both Binary and Multi-class Sentiment Analysis")
    print("=" * 80)

    # Load data
    train_data = pd.read_csv('ReviewsTraining.csv')
    test_data = pd.read_csv('ReviewsTest.csv')

    # Visualize current score distribution
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
    plt.savefig('Bonus_score_distribution.png')
    plt.close()

    # Initialize text preprocessor
    preprocessor = TextPreprocessor()

    # Preprocess data
    train_processed = preprocessor.preprocess_dataframe(train_data)
    test_processed = preprocessor.preprocess_dataframe(test_data)


    # Create binary labels (1-3 = negative, 4-5 = positive)
    train_processed['sentiment_binary'] = train_processed['Score'].apply(lambda x: 1 if x >= 4 else 0)

    # Initialize sentiment model
    model = SentimentModel(use_vader=True)

    # Create feature matrices
    X_train = model.create_feature_matrix(train_processed, is_training=True)

    # Prepare labels for binary and multi-class models
    y_binary = train_processed['sentiment_binary'].values
    y_multiclass = train_processed['Score'].values - 1  # Convert to 0-indexed

    # Train binary model
    print("\n" + "=" * 80)
    print("TRAINING BINARY SENTIMENT MODEL")
    print("=" * 80)
    binary_history = model.train_binary_model(X_train, y_binary)

    # Train multi-class model (bonus task)
    print("\n" + "=" * 80)
    print("TRAINING MULTI-CLASS SENTIMENT MODEL (BONUS TASK)")
    print("=" * 80)
    multiclass_history = model.train_multiclass_model(X_train, y_multiclass)

    # Plot training histories
    plt.figure(figsize=(15, 6))

    # Binary model history
    plt.subplot(1, 2, 1)
    plt.plot(binary_history.history['accuracy'], label='Training')
    plt.plot(binary_history.history['val_accuracy'], label='Validation')
    plt.title('Binary Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Multi-class model history
    plt.subplot(1, 2, 2)
    plt.plot(multiclass_history.history['accuracy'], label='Training')
    plt.plot(multiclass_history.history['val_accuracy'], label='Validation')
    plt.title('Multi-class Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('Bonus_model_histories.png')
    plt.close()

    # Analyze test data
    print("\n" + "=" * 80)
    print("GENERATING PREDICTIONS FOR TEST DATA")
    print("=" * 80)

    # Create test feature matrix
    X_test = model.create_feature_matrix(test_processed, is_training=False)

    # Binary predictions
    binary_predictions = model.predict_binary(X_test)

    # Multi-class predictions
    multiclass_predictions = model.predict_multiclass(X_test)

    # Save predictions to file
    team_number = "6"  # Replace with your team number
    binary_output_file = f'Bonus_Team{team_number}predictions.txt'
    multiclass_output_file = f'Bonus_Team{team_number}multiclass_predictions.txt'

    np.savetxt(binary_output_file, binary_predictions, fmt='%d')
    np.savetxt(multiclass_output_file, multiclass_predictions, fmt='%d')

    print(f"\nBinary predictions saved to '{binary_output_file}'")
    print(f"Multi-class predictions saved to '{multiclass_output_file}'")

    # Binary prediction summary
    print("\nBinary prediction summary:")
    print(f"Total predictions: {len(binary_predictions)}")
    print(
        f"Positive reviews predicted: {np.sum(binary_predictions)} ({np.sum(binary_predictions) / len(binary_predictions) * 100:.2f}%)")
    print(
        f"Negative reviews predicted: {len(binary_predictions) - np.sum(binary_predictions)} ({(len(binary_predictions) - np.sum(binary_predictions)) / len(binary_predictions) * 100:.2f}%)")

    # Multi-class prediction summary
    print("\nMulti-class prediction summary:")
    for score in range(1, 6):
        count = np.sum(multiclass_predictions == score)
        print(f"Score {score}: {count} reviews ({count / len(multiclass_predictions) * 100:.2f}%)")

    # Analyze negative reviews for business insights
    negative_word_freq, negative_bigram_freq = analyze_negative_reviews(train_processed, preprocessor)


if __name__ == "__main__":
    main()