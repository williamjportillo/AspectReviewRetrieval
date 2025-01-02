from method2 import preprocess_text
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
from sklearn.metrics.pairwise import cosine_similarity

def get_asp_documents_using_embeddings(term, inverted_index, embedding_model):
    """
    Retrieve documents related to the input term using word embeddings.

    Args:
        term (str): The input term to search for.
        inverted_index (dict): The inverted index mapping terms to document indices.
        embedding_model (gensim.models.KeyedVectors): The word embedding model.

    Returns:
        set: Set of document indices that match the term.
    """
    terms = term.split()  # Split the input term into individual words
    documents = None

    for t in terms:
        t = preprocess_text(t)  # Preprocess the term (remove punctuation, lowercase, etc.)
        if t in embedding_model:  # Check if the word has a vector in the embedding model
            term_embedding = embedding_model[t]
            related_docs = set()

            # Compare the term embedding with each word in the inverted index
            for word, docs in inverted_index.items():
                word = preprocess_text(word)  # Preprocess the inverted index word as well
                if word in embedding_model:
                    word_embedding = embedding_model[word]
                    similarity = cosine_similarity([term_embedding], [word_embedding])[0][0]

                    # If similarity is above a threshold (e.g., 0.7), consider the word relevant
                    if similarity > 0.7:
                        related_docs.update(docs)

            # Perform set intersection across terms (AND operation)
            if documents is None:
                documents = related_docs
            else:
                documents &= related_docs
        else:
            return set()  # If the term isn't in the embedding model, return an empty set

    return documents if documents else set()

def asp_boolean_search_with_embeddings(inverted_index, aspect, embedding_model):
    """
    Perform a boolean search using word embeddings.

    Args:
        inverted_index (dict): The inverted index mapping terms to document indices.
        aspect (str): The aspect term to search for.
        embedding_model (gensim.models.KeyedVectors): The word embedding model.

    Returns:
        set: Set of document indices that match the aspect.
    """
    aspect_docs = get_asp_documents_using_embeddings(aspect, inverted_index, embedding_model)
    return aspect_docs

analyzer = SentimentIntensityAnalyzer()

def detect_sentiment(text):
    """
    Determine the sentiment of the input text.

    Args:
        text (str): The input text to analyze.

    Returns:
        str: The sentiment of the text ('positive', 'negative', or 'neutral').
    """
    # Handle negations
    text = re.sub(r'\bnot good\b', 'bad', text)
    text = re.sub(r'\bnot bad\b', 'good', text)

    # Analyze sentiment
    sentiment_score = analyzer.polarity_scores(text)
    if sentiment_score['compound'] >= 0.05:
        return 'positive'
    elif sentiment_score['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

