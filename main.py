import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from baseline import create_inverted_index, boolean_search
from method2 import preprocess_text, get_documents_using_embeddings
from method3 import asp_boolean_search_with_embeddings, detect_sentiment
from nltk.tokenize import word_tokenize
import os
from preprocess import df
import gensim.downloader as api
from sentence_transformers import SentenceTransformer, util
from src.method2 import boolean_search_with_embeddings_and_sentiment, is_negative_sentiment

# Get user input for aspect and opinion
aspect_input = input("Enter the aspect: ")
opinion_input = input("Enter the opinion: ")

# Tokenize and preprocess the user inputs
aspect_tokens = word_tokenize(aspect_input)
opinion_tokens = word_tokenize(opinion_input)
aspect = ' '.join(aspect_tokens)
opinion = ' '.join(opinion_tokens)

# Create inverted index from the dataframe
inverted_index = create_inverted_index(df)

# Perform boolean search using baseline method
print("Performing boolean search using baseline method...")
bool_results = boolean_search(inverted_index, aspect, opinion)
bool_ids = df.loc[list(bool_results), 'review_id'].str.strip("'").reset_index(drop=True).to_frame()

# METHOD 2: Boolean search with embeddings and sentiment analysis
embedModel = api.load("word2vec-google-news-300")
print("Performing boolean search with embeddings and sentiment analysis (Method 2)...")
embed_results = boolean_search_with_embeddings_and_sentiment(inverted_index, aspect, opinion, embedModel)
negative_sentiment = any(is_negative_sentiment(word) for word in opinion_tokens)

# Filter reviews based on sentiment
df['customer_review_rating'] = pd.to_numeric(df['customer_review_rating'], errors='coerce')
embed_results_list = list(embed_results)

if negative_sentiment:
    print("Filtering reviews based on negative sentiment...")
    filtered_reviews = df.loc[embed_results_list]
    filtered_reviews = filtered_reviews[filtered_reviews['customer_review_rating'] <= 3]
else:
    print("Filtering reviews based on positive or neutral sentiment...")
    filtered_reviews = df.loc[embed_results_list]

filtered_reviews['review_id'] = filtered_reviews['review_id'].str.replace("'", "")
filtered_reviews = filtered_reviews.reset_index(drop=True)[['review_id']]

# METHOD 3: Boolean search with embeddings and bigram analysis
model3 = SentenceTransformer('all-MiniLM-L6-v2')
print("Performing boolean search with embeddings and bigram analysis (Method 3)...")
asp_Embeds = asp_boolean_search_with_embeddings(inverted_index, aspect, embedModel)
valid_indices = list(asp_Embeds)
reviews = df.loc[valid_indices, 'filtered_text']
encoded_reviews = model3.encode(reviews.tolist(), convert_to_tensor=True)
encoded_query = model3.encode(opinion, convert_to_tensor=True)

# Compute cosine similarity and filter top reviews
cosine_scores = util.pytorch_cos_sim(encoded_query, encoded_reviews)
threshold = .15
print("Computing cosine similarity and filtering top reviews...")
top_reviews = [reviews.iloc[idx] for score, idx in zip(cosine_scores[0], range(len(cosine_scores[0]))) if score.item() > threshold]

# Extract and count bigrams from top reviews
vectorizer = CountVectorizer(ngram_range=(2, 2))
X = vectorizer.fit_transform(top_reviews)
ngrams_df = pd.DataFrame.sparse.from_spmatrix(X, columns=vectorizer.get_feature_names_out())
bigram_counts = ngrams_df.sum(axis=0)
print("Extracting and counting bigrams from top reviews...")
frequent_bigrams = bigram_counts[bigram_counts > 1]

# Create a DataFrame with the bigram and its count
frequent_bigrams_df = frequent_bigrams.reset_index()
frequent_bigrams_df.columns = ['bigram', 'count']
frequent_bigrams_df = frequent_bigrams_df.sort_values(by='count', ascending=False)

# Filter bigrams by sentiment
sentiment = detect_sentiment(opinion)
print("Filtering bigrams by sentiment...")
filtered_bigrams_df = frequent_bigrams_df[frequent_bigrams_df['bigram'].apply(detect_sentiment) == sentiment]

# Compute cosine similarity for filtered bigrams
threshold = 0.5
results = []

for bigram in filtered_bigrams_df['bigram']:
    encoded_query = model3.encode(bigram, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(encoded_query, encoded_reviews)
    relevant_results = [(bigram, score.item(), idx, reviews.iloc[idx]) for score, idx in zip(cosine_scores[0], range(len(cosine_scores[0]))) if score.item() > threshold]
    print(f"Found {len(relevant_results)} relevant results for bigram: {bigram}")
    results.extend(relevant_results)

# Create a DataFrame from the results
results_df = pd.DataFrame(results, columns=['bigram', 'score', 'index', 'review'])

# Extract review IDs for method 3
method3df = pd.DataFrame(columns=['review_id'])
for idx in results_df['index']:
    review_id = df.loc[idx, 'review_id']
    review_id = review_id.replace("'", "")  # Remove single quotes
    new_row = pd.DataFrame({'review_id': [review_id]})
    method3df = pd.concat([method3df, new_row], ignore_index=True)

# Create output directory if it doesn't exist
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# Define the file names based on the aspect and opinion
baseline_filename = f"{output_dir}/{aspect.replace(' ', '_')}_{opinion.replace(' ', '_')}_baseline.pkl"
method2_filename = f"{output_dir}/{aspect.replace(' ', '_')}_{opinion.replace(' ', '_')}_method2.pkl"
method3_filename = f"{output_dir}/{aspect.replace(' ', '_')}_{opinion.replace(' ', '_')}_method3.pkl"

# Save the DataFrames as .pkl files
print("Saving the results to .pkl files...")
bool_ids.to_pickle(baseline_filename)
filtered_reviews.to_pickle(method2_filename)
method3df.to_pickle(method3_filename)
print("Results saved successfully.")