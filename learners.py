import csv
from gensim.models import Word2Vec
from tqdm import tqdm

def train_word2vec_from_csv(csv_file, vector_size=128, window=5, min_count=1, workers=4, output_model="word2vec.model"):
    """
    Train a Word2Vec model on random walks data stored in a large CSV file with progress bars.
    
    Parameters:
        csv_file (str): Path to the CSV file containing random walks.
        vector_size (int): Dimensionality of the word vectors.
        window (int): Maximum distance between the current and predicted word within a sentence.
        min_count (int): Ignores all words with total frequency lower than this.
        workers (int): Number of worker threads to train the model.
        output_model (str): Path to save the trained Word2Vec model.
    """
    
    # First, count total lines (walks) in the CSV file for progress tracking
    with open(csv_file, newline='') as f:
        total_walks = sum(1 for line in f)

    # Define a generator to yield walks line-by-line from the CSV file with a progress bar
    def walk_generator():
        with open(csv_file, newline='') as f:
            reader = csv.reader(f, delimiter=';')
            for row in tqdm(reader, total=total_walks, desc="Reading walks from CSV"):
                yield row  # Each row is a walk, represented as a list of nodes
    
    # Initialize Word2Vec model
    Word2Vec()
    model = Word2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers
    )
    
    # Build vocabulary with progress bar
    print("Building vocabulary...")
    model.build_vocab(tqdm(walk_generator(), total=total_walks, desc="Building vocab"))
    
    # Train the model with progress bar
    print("Training model...")
    model.train(tqdm(walk_generator(), total=total_walks, desc="Training model"), total_examples=model.corpus_count, epochs=model.epochs)
    
    # Save the trained model to disk
    model.save(output_model)
    print(f"Model saved to {output_model}")

# Example usage

if __name__ == "__main__":    
    # Train Word2Vec on the generated walks
    embedding_size = 128
    window_size = 10
    min_count = 1
    train_word2vec_from_csv("walks.csv", vector_size=128, window=window_size, min_count=min_count, workers=8, output_model="word2vec.model")
    exit