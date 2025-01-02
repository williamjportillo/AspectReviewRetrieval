import os
import pandas as pd
from nltk.corpus import stopwords, wordnet
from collections import Counter, defaultdict
from nltk.stem import WordNetLemmatizer
from torch.nn.functional import threshold

# Get the current directory
current_dir = os.getcwd()

# Define the path to the data file
data_file = os.path.join(current_dir, '..', 'data', 'reviews_segment.pkl')

# Load the data into a DataFrame
df = pd.read_pickle(data_file)

# Load stopwords
stopwords = set(stopwords.words('english'))

# Remove stopwords from the review text
df['filtered_text'] = df['review_text'].apply(
    lambda x: ' '.join([word for word in x.split() if word.lower() not in stopwords])
)

# Calculate word frequencies
all_words = ' '.join(df['filtered_text']).split()
word_freq = Counter(all_words)

# Remove infrequent words (words with frequency < 5)
df['filtered_text'] = df['filtered_text'].apply(
    lambda x: ' '.join([word for word in x.split() if word_freq[word] >= 5])
)

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Lemmatize the words in the filtered text
df['filtered_text'] = df['filtered_text'].apply(
    lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()])
)

# Define the path to save the processed DataFrame
df_file = os.path.join(current_dir, '..', 'data', 'processed_reviews.csv')