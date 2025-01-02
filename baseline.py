from collections import defaultdict


def create_inverted_index(input_file):
    """
    Create an inverted index from the input DataFrame.

    Args:
        input_file (pd.DataFrame): DataFrame containing the reviews.

    Returns:
        dict: Inverted index mapping terms to document indices.
    """
    df = input_file
    inverted_index = defaultdict(set)  # Use a set to eliminate duplicates

    # Populate the inverted index
    for idx, review in df['filtered_text'].items():
        for word in review.split():
            inverted_index[word].add(idx)  # Add index to the set

    # Convert sets to sorted lists
    for term in inverted_index:
        inverted_index[term] = sorted(inverted_index[term])

    return inverted_index


def boolean_search(inverted_index, aspect, opinion):
    """
    Perform a boolean search on the inverted index.

    Args:
        inverted_index (dict): Inverted index mapping terms to document indices.
        aspect (str): Aspect term to search for.
        opinion (str): Opinion term to search for.

    Returns:
        set: Set of document indices that match both the aspect and opinion.
    """

    def get_documents(term):
        """
        Get documents for a term from the inverted index.

        Args:
            term (str): Term to search for.

        Returns:
            set: Set of document indices containing the term.
        """
        terms = term.split()
        documents = None

        # Perform set intersection to find common documents
        for t in terms:
            if t in inverted_index:
                if documents is None:
                    documents = set(inverted_index[t])
                else:
                    documents &= set(inverted_index[t])
            else:
                return set()  # If any term doesn't exist, return an empty set

        return documents if documents else set()

    # Get documents for aspect and opinion
    aspect_docs = get_documents(aspect)
    opinion_docs = get_documents(opinion)

    # Perform the AND operation (intersection of the document sets)
    result_docs = aspect_docs & opinion_docs

    return result_docs