from gensim.models import Word2Vec

def get_similar_nodes(model_path, node, top_n=10):
    """
    Load a Word2Vec model and find nodes similar to a given node.
    
    Parameters:
        model_path (str): Path to the saved Word2Vec model.
        node (str): The node for which to find similar nodes.
        top_n (int): Number of similar nodes to retrieve.
        
    Returns:
        list of tuples: Each tuple contains a similar node and its similarity score.
    """
    # Load the trained Word2Vec model
    model = Word2Vec.load(model_path)
    
    # Check if the node exists in the model's vocabulary
    if node not in model.wv:
        raise ValueError(f"Node '{node}' not found in the model vocabulary.")
    
    # Retrieve the most similar nodes
    similar_nodes = model.wv.most_similar(node, topn=top_n)
    
    return similar_nodes

# Example usage
 

if __name__ == '__main__':
    similar_nodes = get_similar_nodes("word2vec.model", node="53e9993fb7602d970217a39b", top_n=10)
    print(similar_nodes)
