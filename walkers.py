from gensim.models import Word2Vec
import csv
import networkx as nx
import random
import pickle
import time

def load_data_from_pickle(pickle_file_path):
    """
    Load the data from the pickle file.

    Parameters:
    pickle_file_path (str): Path to the pickle file containing the paper data.

    Returns:
    dict: The dictionary mapping paper IDs to their citations and keywords.
    """
    with open(pickle_file_path, 'rb') as f:
        data = pickle.load(f)
    
    return data


def build_citation_graph(paper_data):
    """
    Build a citation graph using the paper data.

    Parameters:
    paper_data (dict): Dictionary containing paper IDs, citations, and keywords.

    Returns:
    G (nx.DiGraph): A directed graph where nodes are papers and edges represent citations.
    """
    G = nx.DiGraph()  # Directed graph since citations are directional

    for paper_id, details in paper_data.items():
        # Add the paper itself as a node
        G.add_node(paper_id, keywords=details['keywords'])
        
        # Add citation edges (directed)
        for citation_id in details['citations']:
            if citation_id in paper_data:  # Only add citations that exist in the dataset
                G.add_edge(paper_id, citation_id)  # Edge from paper_id -> citation_id

    return G


def generate_and_save_walks(G, num_walks, walk_length, output_file):
    """
    Generate random walks on a graph and save them to a CSV file.
    
    Parameters:
        G (networkx.Graph): The input graph.
        num_walks (int): Number of random walks to start from each node.
        walk_length (int): Length of each random walk.
        output_file (str): Path to the CSV file where walks will be saved.
    """
    nodes = list(G.nodes())
    
    with open(output_file, "a", newline="") as f:
        writer = csv.writer(f, delimiter=';')  # Set delimiter to semicolon
        
        for node in nodes:
            if not list(G.neighbors(node)):
                continue
            
            for _ in range(num_walks):
                walk = [node]
                
                while len(walk) < walk_length:
                    current_node = walk[-1]
                    neighbors = list(G.neighbors(current_node))
                    
                    if not neighbors:
                        break  # End walk if no neighbors are found                    
                    next_node = random.choice(neighbors)
                    walk.append(next_node)
                
                writer.writerow(walk)

# Example usage
# generate_and_save_walks(G, num_walks=10, walk_length=30, output_file="random_walks.csv")

if __name__ == '__main__':
    start_time = time.time()
    # Path to your pickle file
    pickle_file_path = 'D:\\Projects\\3Credit_project\\Aminer Dataset\\paper_data.pkl' 

    print(f"Starting pickle load")

    # Load and build the graph
    paper_data = load_data_from_pickle(pickle_file_path)
    print(f"starting graph build")
    G = build_citation_graph(paper_data)
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    pre_write = time.time()
    print(f"load time = {time.time()-start_time } seconds")
    # Generate and save walks to a CSV file
    print(f"starting to walk")
    generate_and_save_walks(G, num_walks=10, walk_length=10, output_file="walks.csv")
    print("walks generated and saved")
    print(f"time taken = {time.time()-pre_write} seconds")
    # Delete the graph to free up memory
    del G
    print(f"total time taken = {time.time() - start_time} seconds")
