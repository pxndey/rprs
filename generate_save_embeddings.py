import pickle
from sentence_transformers import SentenceTransformer


def load_keywords_from_file(file_name):
    """
    Load keywords from a text file, removing leading and trailing spaces from each line.
    """
    with open(file_name, 'r', encoding='utf-8') as f:
        # Using strip() to remove both leading and trailing spaces
        keywords = [line.strip() for line in f.readlines() if line.strip()]
    return keywords


def generate_embeddings(keywords, model_name='all-MiniLM-L6-v2'):
    """
    Generate SBERT embeddings for each keyword using the specified model.
    Args:
    - keywords: List of keywords.
    - model_name: Name of the SBERT model.
    Returns:
    - A dictionary with keywords as keys and their corresponding embeddings as values.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(keywords, show_progress_bar=True)
    return dict(zip(keywords, embeddings))



def save_embeddings_to_pickle(file_name, embeddings_dict):
    """
    Save the embeddings dictionary to a pickle file.
    Args:
    - file_name: The pickle file where embeddings will be saved.
    - embeddings_dict: Dictionary of embeddings.
    """
    with open(file_name, 'wb') as f:
        pickle.dump(embeddings_dict, f)
    print(f"Embeddings saved to {file_name}")



def main():
    text_file = 'keywords_list.txt'  # Text file with keywords (one keyword per line)
    pickle_file = 'sbert_embeddings.pkl'  # Output pickle file
    
    # Step 1: Load keywords
    keywords = load_keywords_from_file(text_file)
    print(f"Loaded {len(keywords)} keywords.")

    # Step 2: Generate SBERT embeddings
    embeddings_dict = generate_embeddings(keywords)
    print(f"Generated embeddings for {len(embeddings_dict)} keywords.")

    # Step 3: Save embeddings to a pickle file
    save_embeddings_to_pickle(pickle_file, embeddings_dict)

if __name__ == "__main__":
    main()
