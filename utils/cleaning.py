import pandas as pd
import numpy as np
import pyarrow as pa
from matplotlib import pyplot as plt
import os
import re
import glob
import pyarrow.parquet as pq
from tqdm.auto import tqdm
from IPython.display import display

def tmp_authors_registry_chunks(authors_registry, chunk_size = 500_000, output_dir = "data/authors_registry_chunks"):
    """Safe checkpoints to safe the raw data of authors registry"""
    os.makedirs(output_dir, exist_ok=True)

    def convert_row(x):
        return list(x) if isinstance(x, (set, list)) else x

    def convert_org_year(x):
        if isinstance(x, (set, list)):
            return [
                [org, int(year) if pd.notna(year) else None]
                for org, year in x
            ]
        return x

    for i in tqdm(range(0, len(authors_registry), chunk_size), desc="Processing chunks"):
        chunk_id = i // chunk_size
        chunk_path = f"{output_dir}/part_{chunk_id}.parquet"

        if os.path.exists(chunk_path):
            continue

        chunk = authors_registry.iloc[i:i+chunk_size].copy()

        # conversions
        for col in ['name', 'keywords', 'lang']:
            chunk[col] = [convert_row(x) for x in chunk[col]]

        chunk['org_year'] = [convert_org_year(x) for x in chunk['org_year']]

        # safe
        chunk.to_parquet(chunk_path)

def tmp_authors_registry_merge_chunks(data_dir = "data/authors_registry_chunks", output_dir = "data/authors_registry_full.parquet"):
    """merge in an unique parquet file"""
    files = sorted(glob.glob(f"{data_dir}/part_*.parquet"))

    writer = None

    for f in tqdm(files, desc="Merging chunks"):
        table = pq.read_table(f)

        if writer is None:
            writer = pq.ParquetWriter(
                output_dir,
                table.schema
            )

        writer.write_table(table)

    if writer:
        writer.close()

    print("Merge completed!")


from collections import Counter 
import ast

def is_valid_name(name):
    # ^ : start of the string
    # [A-Za-z\s]+ : one or more letters (upper or lower case) or whitespace characters
    # $ : end of the string
    return bool(re.match(r'^[A-Za-z\s]+$', name))

def union_sets(series):
        return set().union(*series)

def union_org_year(series):
    unique_orgs = set()
    for sublist in series:
        if isinstance(sublist, (list, set)):
            for item in sublist:
                if isinstance(item, (list, tuple)):
                    unique_orgs.add(item)
                elif isinstance(item, dict) and 'org' in item and 'year' in item:
                    unique_orgs.add((item['org'], item['year']))
    return list(unique_orgs.values())

def extract_author_info(df):
    """
    Extract authors' info from a chunk of papers
    """
    author_records = []

    for _, row in df.iterrows():
        authors_list = row.get('authors')
        # print(f'debug 1: {len(authors_list)}')
        
        # if not isinstance(authors_list, list):
        #     continue

        if isinstance(authors_list, str):
            try:
                authors_list = ast.literal_eval(authors_list)
            except:
                continue
            
        keywords = set(row.get('keywords')) if isinstance(row.get('keywords'), list) else set()
        lang = row.get('lang')
        year = row.get('year')
        # print(f'\tdebug 2: {lang}, {year}')

        for auth in authors_list:
            auth_id = auth.get('id')
            auth_name = auth.get('name')
            auth_org = auth.get('org')
            # print(f'\t\tdebug 3: {auth_id}, {auth_name}, {auth_org}')

            # id is the primary key otherwise the name
            has_id = bool(auth_id and auth_id != '')
            key = auth_id if has_id else auth_name
            if not key: 
                continue
            # print(f'\t\t\tdebug 4:{has_id}, {key}')

            author_records.append({
                'id_or_name': key,
                'is_id': has_id,
                'name': [auth_name] if auth_name else list(),
                'org_year': {(auth_org, year)} if auth_org and not pd.isna(year) else set(),
                'keywords': keywords,
                'lang': {lang} if lang and not pd.isna(lang) else set()
            })

    # create a temp dataframe with chunk's info
    if not author_records:
        return pd.DataFrame()

    temp_df = pd.DataFrame(author_records)
    
    # chunk aggregation
    return temp_df.groupby('id_or_name').agg({
        'name': 'sum',
        'is_id': 'max',
        'org_year': union_org_year,
        'keywords': union_sets,
        'lang': union_sets
    }).reset_index()


def assigns_ids(df):
    """Assign an ID to authors that don't have it, based on their official name."""
    # mapping official_name to ID
    name_to_id = df[df['is_id'] == True].explode('official_name').set_index('official_name')['id_or_name'].to_dict()

    # assign IDs to authors without an ID based on their official name
    def assign_name_to_id(row):
        if not row['is_id'] and isinstance(row['official_name'], str) and row['official_name']:
            if row['official_name'] in name_to_id:
                return name_to_id[row['official_name']]
        return row['id_or_name'] if row['is_id'] else None

    df['id'] = df.apply(assign_name_to_id, axis=1)
    return df

def assign_official_name(df):
    """Assign an official name to each author based on the most common name used."""
    def most_common_name(names):
        if isinstance(names, list) and names:
            return Counter(names).most_common(1)[0][0]
        return None

    df['official_name'] = df['name'].apply(most_common_name)
    return df

def clean_names(df):
    """Clean the names by removing invalid entries."""
    def filter_names(names):
        # keep only valid names (alphabets and spaces)
        return [name for name in names if isinstance(name, str) and is_valid_name(name)]
    
    df['name'] = df['name'].apply(filter_names)
    return df

def clean_org_year(df):
    """Clean the org_year column to ensure it's in the correct format."""
    def clean_org_year_item(org_year):
        if isinstance(org_year, (set, list)):
            cleaned = []
            for item in org_year:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    org, year = item
                    cleaned.append((org, int(year) if pd.notna(year) else None))
                elif isinstance(item, dict) and 'org' in item and 'year' in item:
                    cleaned.append((item['org'], int(item['year']) if pd.notna(item['year']) else None))
            return cleaned
        return org_year

    df['org_year'] = df['org_year'].apply(clean_org_year_item)
    return df

def safe_authors_registry(df, path):
    """Save the refined authors registry to a parquet file."""
    df.drop(columns=['is_id', 'id_or_name']).to_parquet(path)
    print(f"Authors registry saved to {path}")    

def refine_authors_df(df, path):
    """ 
    refine the df of authors:
    - assign an id to the one that don't have it
    - assign an unique name ('official_name') based on the most used one
    and save the df as a parquet
    """
    tqdm.pandas(desc="Refining Authors DataFrame")
    # df = df.pipe(assigns_ids).pipe(assign_official_name).pipe(clean_org_year)
    print("\tCleaning the names...")
    df = clean_names(df)
    print("\tCleaning the organization and year information...")
    df = clean_org_year(df)
    print("\tAssigning official names...")
    df = assign_official_name(df)
    print("\tAssigning IDs...")
    df = assigns_ids(df)

    safe_authors_registry(df, path=path)
    return df

