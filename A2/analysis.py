"""
Step 0: Import the necessary dependencies and download NLTK resources
"""
# TODO: spaCy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from contractions import CONTRACTION_MAP
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# TODO: from tensorflow.keras.preprocessing.text import Tokenizer
# 一下sklearn是用来vectorize的
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm # Use SVM for classification

# TODO: Do we import keras or tensorflow.keras?
# from keras.preprocessing.text import Tokenizer
# from keras.models import Sequential
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
# from tensorflow.keras.callbacks import EarlyStopping

# TODO: stemming?
# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
# TODO: should I download this or just simply change clean_text()?
# TODO: we can simply use words = text.split() instead of word_tokenize(text)

np.random.seed(42)

"""
Step 1
    1.1 Load the training data
    1.2 Explore the structure of the training data and plot the distribution of review scores
"""
# TODO: ReviewsTraining or ReviewsTraining.csv?
# 感觉似乎是在写作业的时候用reviewsTraining.csv，但是交作业为了方便用Test
# TODO: ReviewsTest.csv那个文件好像和训练文件格式不太一样，一个有Id，一个没有
# TODO: 这个helpfulnessnumerator和helpfulnessdenominator是什么意思？怎么基本上总是一样
train_data = pd.read_csv('ReviewsTraining.csv')
test_data = pd.read_csv('ReviewsTest.csv')

class DataLoader:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))


    def explore_data(self, data):
        """
        This function explores the data structure and prints the first few rows
        :param data: The input data
        :return: None
        """
        # Set the display options for pandas to show all columns and rows
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.width', 2000)
        pd.set_option('display.expand_frame_repr', False)

        # Check the structure
        print("Data shape:", data.shape)
        print("\nColumns:", data.columns.tolist())
        print("\nSample data:")
        print(data.head())

        # Check for missing values
        print("\nMissing values:")
        print(data.isnull().sum()) # If there are no missing values, the output should be 0


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


"""
Step 2 Text Preprocessing
    2.1 Expand contractions
    2.2 Text cleaning: convert to lowercase, remove special characters, remove stopwords
    2.3 Tokenize the reviews
    2.4 Stemming Lemmatization
"""
# TODO: stemming and lemmatization
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
#
# # Test the function expand_contractions
# example_text = "I can't believe it's not butter!"
# print(f"Original: {example_text}")
# print(f"Expanded: {expand_contractions(example_text)}")
# for line in test_data['Text']:
#     print(f"Original: {line}")
#     print(f"Expanded: {expand_contractions(line)}")
#     print()


def clean_text(text):
    """
    This function cleans the text by converting to lowercase, removing special characters, and removing stopwords
    :param text: The input text
    :return: The cleaned text
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) # TODO: 有必要吗？

    # Remove stopwords # TODO: For sentiment analysis, we may want to keep some of the stopwords while removing the rest
    text_tokens = word_tokenize(text)
    filtered_words = [word for word in text_tokens if word not in set(stopwords.words('english'))]

    # Join the words back into a single string
    text = ' '.join(filtered_words) # TODO: 有必要吗？

    return text


# Apply preprocessing to test data
print("Preprocessing test data...")
test_data['processed_summary'] = test_data['Summary'].fillna('').apply(clean_text)
test_data['processed_text'] = test_data['Text'].fillna('').apply(clean_text)
test_data['processed_combined'] = test_data['processed_summary'] + ' ' + test_data['processed_text']

# Show an example of preprocessed text
print("\nExample of preprocessing:")
sample_idx = 0 # Test on the first review
print(f"Original summary: {test_data.iloc[sample_idx]['Summary']}")
print(f"Processed summary: {test_data.iloc[sample_idx]['processed_summary']}")
print(f"Original text: {test_data.iloc[sample_idx]['Text'][:200]}...")
print(f"Processed text: {test_data.iloc[sample_idx]['processed_text'][:200]}...")


"""
Step 3: Feature Engineering
"""
train_data['sentiment'] = train_data['Score'].apply(lambda x: 1 if x >= 4 else 0)

# Check the distribution of binary sentiment
plt.figure(figsize=(8, 6))
ax = sns.countplot(x='sentiment', data=train_data)
plt.title('Distribution of Binary Sentiment')
plt.xlabel('Sentiment (0=Negative, 1=Positive)')
plt.ylabel('Count')

# Add count labels
for p in ax.patches:
    ax.annotate(f'{p.get_height()}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='baseline',
                xytext=(0, 5), textcoords='offset points')

plt.tight_layout()
plt.show()

# Unigram features with TF-IDF
# TfidVectorizer itself has encapsulated the process of tokenization
print("Creating unigram features...")
unigram_vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=5,
    max_df=0.8,
    strip_accents='unicode',
    use_idf=True,
    smooth_idf=True,
    sublinear_tf=True
)

unigram_features = unigram_vectorizer.fit_transform(train_data['processed_combined'])

# Bigram features with TF-IDF (includes both unigrams and bigrams)
print("Creating bigram features...")
bigram_vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=5,
    max_df=0.8,
    ngram_range=(1, 2),  # both unigrams and bigrams
    strip_accents='unicode',
    use_idf=True,
    smooth_idf=True,
    sublinear_tf=True
)

bigram_features = bigram_vectorizer.fit_transform(train_data['processed_combined'])

print(f"Unigram features shape: {unigram_features.shape}")
print(f"Bigram features shape: {bigram_features.shape}")


# Display top unigrams and bigrams with highest IDF (most informative)
def display_top_features(vectorizer, feature_names, top_n=20):
    # Get the IDF values
    idf = vectorizer.idf_

    # Create a DataFrame with feature names and IDF values
    feature_df = pd.DataFrame({
        'feature_name': feature_names,
        'idf': idf
    })

    # Sort by IDF in descending order
    feature_df = feature_df.sort_values('idf', ascending=False).head(top_n)

    return feature_df


print("\nTop Unigrams by IDF:")
unigram_names = unigram_vectorizer.get_feature_names_out()
top_unigrams = display_top_features(unigram_vectorizer, unigram_names)
print(top_unigrams)

print("\nTop Bigrams by IDF:")
bigram_names = bigram_vectorizer.get_feature_names_out()
# Filter only bigrams (those with a space in the name)
bigram_only_names = [name for name in bigram_names if ' ' in name]
# Create a mock vectorizer with only the bigram idfs
bigram_idfs = [bigram_vectorizer.idf_[bigram_names.tolist().index(name)] for name in bigram_only_names]
bigram_only_vectorizer = bigram_vectorizer
bigram_only_vectorizer.idf_ = np.array(bigram_idfs)

top_bigrams = display_top_features(bigram_only_vectorizer, bigram_only_names)
print(top_bigrams)
