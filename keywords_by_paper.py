import ijson
import json
import pickle
import numpy as np
from tqdm import tqdm

def main():
    print("starting pickel load")
    # Load the pkl file containing keyword embeddings
    with open("sbert_embeddings.pkl", "rb") as f:
        keyword_embeddings = pickle.load(f)

    print("pickel loaded")
    print("starting embedGen")
    # Open the output file in write mode
    with open("embeddings.jsonl", "w") as out_file:
        # Open the large JSON array file and stream each object in the array
        with open("dblp_v14.json", "r") as json_file:
            # Use ijson to parse each item in the JSON array as a separate object
            objects = ijson.items(json_file, "item")
            
            # Wrap the streaming iterator with tqdm for a progress bar
            for obj in tqdm(objects, desc="Processing objects"):
                # Get the ID and array of keywords from the current object
                obj_id = obj.get("id")
                keywords = obj.get("keywords", [])

                # Retrieve embeddings for each keyword
                embeddings = [keyword_embeddings[key] for key in keywords if key in keyword_embeddings]

                if embeddings:
                    # Calculate the average embedding if embeddings are found
                    avg_embedding = np.mean(embeddings, axis=0).tolist()
                else:
                    # Set embedding to 0 if no embeddings are found
                    avg_embedding = np.zeros(shape=(384,)).tolist()

                # Write each result as a JSON line
                result = {"id": obj_id, "embedding": avg_embedding}
                out_file.write(json.dumps(result) + "\n")
    print("done")

if __name__ == "__main__":
    main()
