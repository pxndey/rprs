import ijson
from tqdm import tqdm

def extract_keywords_from_paper(paper):
    """
    Extracts keywords from a paper, using the 'keywords' field.
    Returns an empty set if no keywords are found.
    """
    keywords = set()
    
    # Extract from 'keywords' field if available
    if 'keywords' in paper and paper['keywords']:
        keywords.update(paper['keywords'])
    
    return keywords

def extract_all_keywords(json_file_path, max_papers=None):
    """
    Extracts all keywords from the given JSON file (AMiner dataset) using ijson.
    Args:
    - json_file_path: Path to the JSON file containing paper data.
    - max_papers: Limit the number of papers to process for testing purposes (optional).
    Returns:
    - A list of all extracted keywords (a unique set).
    """
    all_keywords = set()

    # Use ijson to parse the file incrementally
    with open(json_file_path, 'r', encoding='utf-8') as f:
        papers = ijson.items(f, 'item')  # Each 'item' is a paper entry
        for idx, paper in tqdm(enumerate(papers), desc="Processing papers"):
            if max_papers and idx >= max_papers:
                break
            keywords = extract_keywords_from_paper(paper)
            if keywords:
                all_keywords.update(keywords)

    return list(all_keywords)

def main():
    json_file_path = 'dblp_v14.json'
    max_papers = None  # Set a limit for testing, set to None to process all papers
    keywords = extract_all_keywords(json_file_path, max_papers=max_papers)
    
    # Save the keywords for future use
    with open('keywords_list.txt', 'w', encoding='utf-8') as f:
        for keyword in keywords:
            f.write(f"{keyword}\n")
    
    print(f"Extracted {len(keywords)} unique keywords.")

if __name__ == "__main__":
    main()
