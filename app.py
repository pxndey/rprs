import ijson
import pickle

def extract_data_from_large_json(json_file_path):
    """
    Extract paper IDs, citations, and keywords from a large JSON array file.

    Parameters:
    json_file_path (str): Path to the large JSON file.

    Returns:
    dict: A dictionary mapping paper IDs to their citations and keywords.
    """
    paper_data = {}

    # Open the JSON file and use ijson to parse it incrementally
    with open(json_file_path, 'r', encoding='utf-8') as f:
        papers = ijson.items(f, 'item')  # 'item' treats each element in the JSON array as a separate object
        
        for paper in papers:
            paper_id = paper.get('id')
            citations = paper.get('references', [])  # Ensure citations default to empty list if not present
            keywords = paper.get('keywords', [])

            if paper_id:
                paper_data[paper_id] = {
                    'citations': citations,
                    'keywords': keywords
                }

    return paper_data

def save_to_pickle(data, pickle_file_path):
    """
    Save the extracted data into a pickle file.

    Parameters:
    data (dict): The paper data to save.
    pickle_file_path (str): Path where the pickle file will be saved.
    """
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(data, f)

def main():
    json_file_path = 'dblp_v14.json'  # Path to the large JSON file
    pickle_file_path = 'paper_data.pkl'  # Output file where parsed data will be stored

    # Extract data from the large JSON file
    print("Extracting data from large JSON...")
    paper_data = extract_data_from_large_json(json_file_path)

    # Save extracted data to Pickle
    print(f"Saving data to {pickle_file_path}...")
    save_to_pickle(paper_data, pickle_file_path)

    print("Data extraction and saving completed!")

if __name__ == '__main__':
    main()