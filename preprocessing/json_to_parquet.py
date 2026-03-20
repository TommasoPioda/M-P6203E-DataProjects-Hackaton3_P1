import pandas as pd
import os
import sys

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define paths relative to the script directory
    # Expected structure:
    # root/
    #   preprocessing/
    #     json_to_parquet.py
    #   data/
    #     DBLP-Citation-network-V18/
    #       DBLP-Citation-network-V18.jsonl
    #     parquet/
    
    DATA_PATH = os.path.join(script_dir, "..", "data", "DBLP-Citation-network-V18.jsonl")
    OUTPUT_DIR = os.path.join(script_dir, "..", "data", "parquet")
    
    # Normalize paths
    dataset_path = os.path.normpath(DATA_PATH)
    output_dir = os.path.normpath(OUTPUT_DIR)

    print(f"Dataset path: {dataset_path}")
    print(f"Output directory: {output_dir}")

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    try:
        print("Reading JSON file in chunks...")
        chunks = pd.read_json(
            dataset_path,
            lines=True,
            chunksize=100000
        )

        for i, chunk in enumerate(chunks):
            output_file = os.path.join(output_dir, f"data_{i}.parquet")
            print(f"Processing chunk {i}, writing to {output_file}")
            chunk.to_parquet(
                output_file,
                engine="pyarrow",
                index=False
            )
        print("Conversion complete.")

    except ImportError as e:
        print(f"ImportError: {e}")
        print("It seems 'pyarrow' or one of its dependencies is missing.")
        print("Try reinstalling pyarrow: pip install pyarrow --force-reinstall")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
