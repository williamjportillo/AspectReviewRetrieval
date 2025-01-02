
from sklearn.metrics.pairwise import cosine_similarity
import re
from textblob import TextBlob

def preprocess_text(text):
    """
    Preprocess the input text by removing punctuation and normalizing case.

    Args:
        text (str): The input text to preprocess.

    Returns:
        str: The preprocessed text.
    """
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove punctuation
    return text.lower()

def get_documents_using_embeddings(term, inverted_index, embedding_model):
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

def is_negative_sentiment(text):
    """
    Determine if the sentiment of the input text is negative.

    Args:
        text (str): The input text to analyze.

    Returns:
        bool: True if the sentiment is negative, False otherwise.
    """
    analysis = TextBlob(text)
    return analysis.sentiment.polarity < 0

def boolean_search_with_embeddings_and_sentiment(inverted_index, aspect, opinion, embedding_model):
    """
    Perform a boolean search using word embeddings and sentiment analysis.

    Args:
        inverted_index (dict): The inverted index mapping terms to document indices.
        aspect (str): The aspect term to search for.
        opinion (str): The opinion term to search for.
        embedding_model (gensim.models.KeyedVectors): The word embedding model.

    Returns:
        set: Set of document indices that match both the aspect and opinion.
    """
    # Get documents for the aspect and opinion using word embeddings
    aspect_docs = get_documents_using_embeddings(aspect, inverted_index, embedding_model)
    opinion_docs = get_documents_using_embeddings(opinion, inverted_index, embedding_model)

    # Perform the AND operation (intersection of the document sets)
    result_docs = aspect_docs & opinion_docs

    return result_docs