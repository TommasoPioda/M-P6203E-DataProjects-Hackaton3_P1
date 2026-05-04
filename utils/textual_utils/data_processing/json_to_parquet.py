import pandas as pd
import os
import glob


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
    
    DATA_PATH = os.path.join(script_dir, "..", "data", "DBLP-Citation-network-V18", "DBLP-Citation-network-V18.jsonl")
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
            
            print(f"Processing chunk {i}")

            chunk["year"] = pd.to_numeric(chunk["year"], errors="coerce")
            
            chunk_nan = chunk[chunk["year"].isna()]
            chunk_valid = chunk[chunk["year"].notna()]


            if not chunk_nan.empty:
                nan_dir = os.path.join(output_dir, "nan")
                os.makedirs(nan_dir, exist_ok=True)

                nan_file = os.path.join(nan_dir, f"{year}_part_{i}.parquet")        
                
                chunk_nan.to_parquet(
                    nan_file,
                    engine="pyarrow",
                    index=False
                    )

            chunk_valid["year"] = chunk_valid["year"].astype(int)

            chunk_valid = chunk_valid.sort_values("year")

            for year, group in chunk_valid.groupby("year"):

                if year < 1970:
                    
                    before_1970_dir = os.path.join(output_dir, "b_1970")
                    os.makedirs(before_1970_dir, exist_ok=True)

                    output_file = os.path.join(before_1970_dir, f"{year}_part_{i}.parquet")        
                    
                    group.to_parquet(
                        output_file,
                        engine="pyarrow",
                        index=False
                        )
                    
                else:
                    
                    year_dir = os.path.join(output_dir, str(int(year)))
                    os.makedirs(year_dir, exist_ok=True)
                    output_file = os.path.join(year_dir, f"{year}_part_{i}.parquet")        
                    
                    group.to_parquet(
                        output_file,
                        engine="pyarrow",
                        index=False
                        )

        print("Conversion complete.")
        
        print("\nStart concatenations of the files")
            
        for folder in os.listdir(OUTPUT_DIR):
            folder_path = os.path.join(OUTPUT_DIR, folder)
            
            if not os.path.isdir(folder_path):
                continue
            
            print(f"Processing folder: {folder}")
            
            files = glob.glob(os.path.join(folder_path, "*.parquet"))
            
            if not files:
                print("No parquet files found, skipping...")
                

            # Read and concatenate the filees
            dfs = []
            for f in files:
                dfs.append(pd.read_parquet(f))
                
            combined = pd.concat(dfs, ignore_index=True)
            
            
            # Save single file
            output_file = os.path.join(folder_pah, f"{folder}.parquet")
            combined.to_parquet(
                output_file,
                engine="pyarrow",
                index=False
                )
            
            print(f"Saved merged file: {output_file}")
            
            # Delete old files
            for f in files:
                if f != output_file:
                    os.remove(f)
                    
            print(f"Deleted all old files")

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
