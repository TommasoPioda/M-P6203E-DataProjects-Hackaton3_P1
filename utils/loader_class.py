import pandas as pd
import pyarrow as pa
import os
from tqdm.auto import tqdm


class DBLP_Loader:
    """ Loader class to handle the streaming and processing of the DBLP dataset.
    - Initializes with the folder path containing the parquet files.
    - Provides methods to load all data at once (with caution) or to stream and process files one by one.
    - Includes a specific method to fill missing author details using an authors registry.
    - Offers a safe checkpointing method to save processed data in chunks to avoid memory issues."""
    def __init__(self, folder_path):
        self.folder = os.path.abspath(folder_path)
        
        self.files = sorted([f for f in os.listdir(self.folder) if f.endswith('.parquet')])
        self.current_data = []

        # column types lists for processing and checkpointing
        self.string_columns = []
        self.numeric_columns = []
        self.array_of_str_columns = []
        self.array_of_dict_columns = []  

    def get_full_df(self):
        """Loads all files into one DataFrame (Careful with RAM!)"""
        all_chunks = []
        for f in tqdm(self.files, desc="Loading Parquet Files"):
            path = os.path.join(self.folder, f)
            all_chunks.append(pd.read_parquet(path))
        return pd.concat(all_chunks, ignore_index=True)
    
    def stream_and_process(self, process_func, update_internal=True, take_current=True, **kwargs):
        """Processes files one by one and optionally updates internal state."""
        results = []

        if take_current and self.current_data:
            print("Processing current internal data...")
            for c in tqdm(self.current_data, desc="Processing Current Data Chunks"):
                processed_chunk = process_func(c, **kwargs)
                results.append(processed_chunk)
        
        elif take_current or not self.current_data:
            print("No current internal data to process, streaming from files...")
            for f in tqdm(self.files, desc="Processing Batches"):
                path = os.path.join(self.folder, f)
                df = pd.read_parquet(path)
                processed_df = process_func(df, **kwargs)
                results.append(processed_df)

        if update_internal == True:
            self.current_data = results

        return results
    
    def fill_author_gaps(self, author_registry, chunks=None, update_internal=True):
        """
        Specific process function to fill missing author details.
        - Fills missing `id` from `official_name`.
        - Fills missing `official_name` from `id`.
        - Fills missing `org` from `id` and `year`.
        """
        chunks = chunks if chunks is not None else self.current_data  
        reg = author_registry.copy()

        # mapping
        print("Creating mapping dictionaries from the authors registry...")

        print("\tname to official name mapping...")
        # name -> official_name
        # name_to_official_name = reg.explode('name').set_index('name')['official_name'].to_dict()
        name_exploded = reg.explode('name', ignore_index=True)
        name_to_official_name = dict(zip(name_exploded['name'].values, name_exploded['official_name'].values))
        
        print("\tid to official name mapping...")
        # id -> official_name
        # name_to_id = reg.set_index('official_name')['id'].to_dict()
        name_to_id = dict(zip(reg['official_name'].values, reg['id'].values))

        print("\tofficial name to id mapping...")
        # official_name -> id
        # id_to_name = reg.set_index('id')['official_name'].to_dict()
        id_to_name = dict(zip(reg['id'].values, reg['official_name'].values))

        print("\tid-year to org mapping...")
        # id, year -> org
        def valid_format_org_year(cell):
            if isinstance(cell, dict) and 'org' in cell and 'year' in cell:
                return True  # Wrap in a list to standardize the format
            else:
                if isinstance(cell, list) and len(cell) == 2 and isinstance(cell[0], str) and isinstance(cell[1], (int, float)):
                    print(f"Warning: 'org_year' is a list instead of dict. Converting to dict format for consistency.")
                return False  # Invalid format, will be filtered out later

        registry_exploded = reg.explode('org_year', ignore_index=True)
        # filter out rows where 'org_year' is not a valid dict
        mask_valid = registry_exploded['org_year'].map(valid_format_org_year)
        registry_exploded = registry_exploded[mask_valid]
        # divide 'org_year' in two different columns, 'org' and 'year'
        org_year_df = pd.json_normalize(registry_exploded['org_year']) # json normalization is faster
        registry_exploded = pd.concat([registry_exploded[['id']], org_year_df[['org', 'year']]]
                                      , axis=1).dropna(subset=['org', 'year', 'id'])

        # convert types for memory efficiency
        registry_exploded['year'] = registry_exploded['year'].astype('int32', errors='ignore')
        registry_exploded['org'] = registry_exploded['org'].astype('category')


        id_year_to_org = dict(zip(zip(registry_exploded['id'].values, registry_exploded['year'].values),
                                  registry_exploded['org'].values))
        
        del reg, org_year_df, registry_exploded, name_exploded  # free memory

        # function to fill the gaps in the authors data
        def fill_missing(authors, year):
            if authors is None or not isinstance(authors, list) or len(authors) == 0:
                return authors  # No authors to process, skip to next row
            
            for a in authors:
                if a is None or pd.isna(a) or not isinstance(a, dict):
                    continue  # Skip invalid author entries
                
                name = a.get('name')
                a_id = a.get('id')

                if name and name in name_to_official_name:
                    official_name = name_to_official_name.get(name)
                    a['name'] = official_name

                    # no ID, but official name is known
                    if pd.isna(a_id) and official_name in name_to_id:
                        a['id'] = name_to_id.get(official_name)

                # no name, but ID is known
                elif pd.isna(name) and a_id in id_to_name:
                    a['name'] = id_to_name.get(a_id)

                # not org, but ID and year are known
                if pd.isna(a.get('org')) and a_id and isinstance(year, (int, float)) and (a_id, year) in id_year_to_org:
                    a['org'] = id_year_to_org.get((a_id, year))

            return authors
        
        # process chunks
        results = []
        for c in tqdm(chunks, desc="Filling Authors Gaps Chunks:"):
            c = c.copy()
            # fill missing values
            c['authors'] = [fill_missing(authors, year) for authors, year in zip(c['authors'], c['year'])]
            
            results.append(c)

        if update_internal == True:
            self.current_data = results

        return results
    
    def fix_venue_mismatch(self, chunks=None, update_internal=True):
        """ Checks for mismatches between 'venue' and 'doc_type' and attempts to fix them based on simple heuristics in every chunk."""
        chunks = chunks if chunks is not None else self.current_data
        results = []

        n_conf_as_journ = 0
        n_journ_as_conf = 0
        n_no_doc_type_conf_venue = 0
        n_no_doc_type_journ_venue = 0
        
        def check_and_fix_venue_mismatch(df):
            """ Checks for mismatches between 'venue' and 'doc_type' and attempts to fix them based on simple heuristics."""
            # Example: If 'Conference' in venue name but 'Journal' in doc_type, set doc_type to 'Conference'
            conf_as_journ = (df['venue'].str.contains('Conference', case=False, na=False)) & (df['doc_type'].str.contains('Journal', case=False, na=False))
            journ_as_conf = (df['venue'].str.contains('Journal', case=False, na=False)) & (df['doc_type'].str.contains('Conference', case=False, na=False))
            no_doc_type_conf_venue = (df['doc_type'].isna()) & (df['venue'].str.contains('Conference', case=False, na=False))
            no_doc_type_journ_venue = (df['doc_type'].isna()) & (df['venue'].str.contains('Journal', case=False, na=False))

            n_conf_as_journ = conf_as_journ.sum()
            n_journ_as_conf = journ_as_conf.sum()
            n_no_doc_type_conf_venue = no_doc_type_conf_venue.sum()
            n_no_doc_type_journ_venue = no_doc_type_journ_venue.sum()
            
            df.loc[conf_as_journ, 'doc_type'] = 'conference'
            df.loc[journ_as_conf, 'doc_type'] = 'journal'
            df.loc[no_doc_type_conf_venue, 'doc_type'] = 'conference'
            df.loc[no_doc_type_journ_venue, 'doc_type'] = 'journal'

            return df, n_conf_as_journ, n_journ_as_conf, n_no_doc_type_conf_venue, n_no_doc_type_journ_venue
        
        for c in tqdm(chunks, desc="Fixing Venue Mismatch Chunks:"):
            c = c.copy()
            c, n1, n2, n3, n4 = check_and_fix_venue_mismatch(c)
            
            n_conf_as_journ += n1
            n_journ_as_conf += n2
            n_no_doc_type_conf_venue += n3
            n_no_doc_type_journ_venue += n4

            results.append(c)
        
        if update_internal == True:
            self.current_data = results

        print(f"Found and fixed {n_conf_as_journ} mismatches where 'Conference' is in venue but doc_type is 'Journal'.")
        print(f"Found and fixed {n_journ_as_conf} mismatches where 'Journal' is in venue but doc_type is 'Conference'.")
        print(f"Found and fixed {n_no_doc_type_conf_venue} mismatches where 'Conference' is in venue but doc_type is missing.")
        print(f"Found and fixed {n_no_doc_type_journ_venue} mismatches where 'Journal' is in venue but doc_type is missing.")

        return results
    
    def set_column_types(self, string_cols, numeric_cols, array_of_str_cols, array_of_dict_cols):
        """
        Set the column types for processing and checkpointing.
        This method allows us to define which columns are of which type, so that we can apply the appropriate processing and ensure consistent schema when saving to Parquet."""
        self.string_columns = string_cols
        self.numeric_columns = numeric_cols
        self.array_of_str_columns = array_of_str_cols
        self.array_of_dict_columns = array_of_dict_cols

    def pyarrow_schema_from_df(self, data):
        """
        Generates a PyArrow schema based on the columns present in the DataFrame.
        This is used to ensure that when we save to Parquet, we maintain consistent data types and structure.
        """
        fields = []
        for col in self.string_columns:
            if col in data.columns:
                fields.append(pa.field(col, pa.string()))

        for col in self.numeric_columns:
            if col in data.columns:
                fields.append(pa.field(col, pa.float64()))

        for col in self.array_of_str_columns:
            if col in data.columns:
                fields.append(pa.field(col, pa.list_(pa.string())))

        for col in self.array_of_dict_columns:
            if col in data.columns:
                if col == 'authors':
                    # Definiamo i campi fissi dello struct per gli autori
                    author_struct = pa.struct([
                        pa.field('id', pa.string()),
                        pa.field('name', pa.string()),
                        pa.field('org', pa.string())
                    ])
                    fields.append(pa.field(col, pa.list_(author_struct)))
                else:
                    pass
                    # fields.append(pa.field(col, pa.list_(pa.struct([pa.field(k, pa.string()) for k in v.keys()]))))

        return pa.schema(fields)
    
    def safe_checkpoint(self, base_name='01_cleaned_data', path='data/checkpoints', chunk_size = 100000, mode_divide_by=False):
        """
        Save the processed data to Parquet files in chunks.
        Parameters:
            - data: DataFrame to save
            - path: Directory where to save the Parquet files
            - base_name: The base name for the saved files
        """
        if not os.path.exists(path):
            os.makedirs(path)
        
        data = pd.concat(self.current_data, ignore_index=True)

        # print("Replacing null placeholders with NaN for checkpointing...")
        # for col in data.columns:
        #     data[col] = data[col].astype(str).replace(NULL_PLACEHOLDERS, np.nan)

        # print("Forcing column types for checkpointing...")
        # for col in self.numeric_columns:
        #     if col in data.columns:
        #         data[col] = pd.to_numeric(data[col], errors='coerce')
                
        # for col in self.array_of_str_columns + self.array_of_dict_columns:
        #     if col in data.columns:
        #         try:
        #             data[col] = data[col].apply(lambda x: list(x))
        #         except Exception as e:
        #             print(f"Error occurred while processing column {col}: {e}")
        
        print("Retrieving PyArrow schema for checkpointing...")
        schema = self.pyarrow_schema_from_df(data)
        print(f"\nGenerated PyArrow schema for checkpointing: \n{schema}\n")
        print(f"Saving checkpoint with {len(data)} records...")
        if mode_divide_by:
            col = mode_divide_by
            # If mode_divide_by is active, we will save one file per each value in the column specified (e.g., 'year'), instead of splitting by chunk size.
            unique_values = data[col].dropna().unique()
            unique_values.sort()

            for value in tqdm(unique_values, desc="Saving Parquet Files by Value"):
                value_data = data[data[col] == value]
                file_path = os.path.join(path, f"{base_name}_{value}.parquet")
                value_data.to_parquet(file_path, engine='pyarrow', schema=schema, index=False)

            print(f"Checkpoint saved to {path} with files divided by {col}.")
            return
        
        else:
            # Split data into smaller chunks and save each as a Parquet file
            total_chunks = len(data) // chunk_size + (len(data) % chunk_size > 0)
            
            for i in tqdm(range(total_chunks), desc="Saving Parquet Files"):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(data))
                
                chunk = data.iloc[start_idx:end_idx]
                file_path = os.path.join(path, f"{base_name}_part_{i + 1}.parquet")
                chunk.to_parquet(file_path, engine='pyarrow', schema=schema, index=False)

            print(f"Checkpoint saved to {path}")