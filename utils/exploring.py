from datetime import datetime

import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib import pyplot as plt
from tqdm.auto import tqdm


def inspect_authors_field(df):
    """Inspect the 'authors' field for potential issues."""
    print("Inspecting 'authors' field for potential issues...")
    # count NaN values
    num_nan = df['authors'].isna().sum()
    print(f"\tNumber of NaN values in 'authors': {num_nan}")

    # validate structure
    def is_valid_author_entry(authors):
        if isinstance(authors, np.ndarray):
            for a in authors:
                if not isinstance(a, dict):
                    return False
                if ('name' not in a) or ('id' not in a) or ('org' not in a):
                    return False
            return True
        return False  # anything else is invalid

    # check for invalid structures
    invalid_structure_mask = ~df['authors'].apply(is_valid_author_entry)
    num_invalid_structures = invalid_structure_mask.sum()
    print(f"\tNumber of entries with invalid structure in 'authors': {num_invalid_structures}")

    # check for empty arrays/lists
    def is_empty_array(authors):
        if isinstance(authors, np.ndarray):
            return len(authors) == 0
        return False

    empty_array_mask = df['authors'].apply(is_empty_array)
    num_empty_arrays = empty_array_mask.sum()
    print(f"\tNumber of empty array entries in 'authors': {num_empty_arrays}")

    exploded = df.explode('authors', ignore_index=True)['authors']
    exploded_names = exploded.apply(lambda a: a.get('name') if isinstance(a, dict) and 'name' in a else np.nan)
    n_unique_authors = exploded_names.dropna().nunique()
    print(f"\tNumber of unique authors in 'authors': {n_unique_authors}")

def inspect_keywords_field(df):
    """Inspect the 'keywords' field for potential issues."""
    print("Inspecting 'keywords' field for potential issues...")
    # count NaN values
    num_nan = df['keywords'].isna().sum()
    print(f"\tNumber of NaN values in 'keywords': {num_nan}")

    # check for empty arrays/lists
    def is_empty_array(keywords):
        if isinstance(keywords, np.ndarray):
            return len(keywords) == 0
        return False

    empty_array_mask = df['keywords'].dropna().apply(is_empty_array)
    num_empty_arrays = empty_array_mask.sum()
    print(f"\tNumber of empty array entries in 'keywords': {num_empty_arrays}")

    # check for invalid structures
    def is_valid_keyword_entry(keywords):
        if isinstance(keywords, np.ndarray) or isinstance(keywords, list):
            return all(isinstance(k, str) for k in keywords)
        return False
    invalid_structure_mask = ~df['keywords'].apply(is_valid_keyword_entry)
    num_invalid_structures = invalid_structure_mask.sum()
    print(f"\tNumber of entries with invalid structure in 'keywords': {num_invalid_structures}")

    # check for number of unique keywords
    unique_keywords = df['keywords'].explode().dropna().nunique()
    print(f"\tNumber of unique keywords: {unique_keywords}")

    # check for number of keywords per paper
    num_keywords = df['keywords'].apply(lambda x: len(x) if isinstance(x, np.ndarray) else 0)
    print(f"\tAverage number of keywords per paper: {num_keywords.mean():.2f}")
    print(f"\tMax number of keywords for a single paper: {num_keywords.max()}")

    # check for most common keywords
    all_keywords = df['keywords'].explode().dropna()
    all_keywords = all_keywords.apply(lambda x: x.strip().lower() if isinstance(x, str) else x)  # normalize keywords
    top_keywords = all_keywords.value_counts().head(10)
    print("\tTop 10 most common keywords:")
    for i, k in enumerate(top_keywords.index):
        print(f"\t\t{i}) {k}: {top_keywords[k]} occurrences")

def inspect_venue_field(df):
    """Inspect the 'venue' field for potential issues."""
    print("Inspecting 'venue' field for potential issues...")
    # count NaN values
    num_nan = df['venue'].isna().sum()
    print(f"\tNumber of NaN values in 'venue': {num_nan}")

    # check for number of unique venues
    n_unique_venues = df['venue'].nunique()
    print(f"\tNumber of unique venues: {n_unique_venues}")

    # check for most common venues
    top_venues = df['venue'].value_counts().head(10)
    print("\tTop 10 most common venues:")
    for i, v in enumerate(top_venues.index):
        print(f"\t\t{i}) {v}: {top_venues[v]} occurrences")

def inspect_doc_type_field(df):
    """Inspect the 'doc_type' field for potential issues."""
    print("Inspecting 'doc_type' field for potential issues...")
    # count NaN values
    num_nan = df['doc_type'].isna().sum()
    print(f"\tNumber of NaN values in 'doc_type': {num_nan}")

    # check for number of unique doc types
    n_unique_doc_types = df['doc_type'].dropna().nunique()
    print(f"\tNumber of unique doc types: {n_unique_doc_types}")

    # check for most common doc types
    top_doc_types = df['doc_type'].value_counts().head(10)
    print("\tTop 10 most common doc types:")
    for i, dt in enumerate(top_doc_types.index):
        print(f"\t\t{i}) {dt}: {top_doc_types[dt]} occurrences")

def inspect_year_field(df):
    """Inspect the 'year' field for potential issues."""
    print("Inspecting 'year' field for potential issues...")
    # count NaN values
    num_nan = df['year'].isna().sum()
    print(f"\tNumber of NaN values in 'year': {num_nan}")

    # check for valid year range
    current_year = datetime.now().year
    invalid_years = df[(df['year'] > current_year) | (df['year'] <= 1800)]['year']
    num_invalid_years = invalid_years.count()
    print(f"\tNumber of invalid years (<=1800 or >{current_year}): {num_invalid_years}")

    # check for most common years
    top_years = df['year'].value_counts().head(10)
    print("\tTop 10 most common publication years:")
    for i, y in enumerate(top_years.index):
        print(f"\t\t{i}) {y}: {top_years[y]} occurrences")

def global_inspection(df):
    df = df.copy()

    # explode properly
    df = df.explode('authors', ignore_index=True)
    df = df.explode('keywords', ignore_index=True)

    # base describe
    df_descr = df.describe(include='all')

    # create rows with same index as df_descr
    dtypes = pd.Series(df.dtypes, name='dtype')
    nan_values = pd.Series(df.isna().sum(), name='num_nan')
    not_nan_values = pd.Series(df.notna().sum(), name='num_not_nan')

    # convert these Series into DataFrames with matching index
    dtypes_df = dtypes.to_frame().T
    nan_df = nan_values.to_frame().T
    not_nan_df = not_nan_values.to_frame().T

    # assign proper index labels
    dtypes_df.index = ['dtype']
    nan_df.index = ['num_nan']
    not_nan_df.index = ['num_not_nan']

    # concatenate vertically
    global_df = pd.concat([nan_df, not_nan_df,df_descr, dtypes_df], axis=0)

    print("Global DataFrame Description:")
    display(global_df)
    return global_df




# inspect the 'references' field for potential issues, including validation of reference years against the paper's year
def inspect_references_field(df, id_to_year, null_placeholders, example=False):
    df = df.copy()
    """Inspect the 'references' field for potential issues."""
    print("Inspecting 'references' field for potential issues...")
    # count NaN values
    def count_missing_references(df):
        def is_really_missing(x):
            if isinstance(x, (list, np.ndarray)):
                return False
            # NaN or it's in the list of null placeholders
            if pd.isna(x) or x in null_placeholders:
                return True
            # also consider as missing if it's a float (which could be a result of some parsing issues)
            if isinstance(x, float):
                return True
            return False
        # apply the function to the 'references' column and sum the results
        count = df['references'].apply(is_really_missing).sum()
        return count
    num_nan = count_missing_references(df)
    print(f"\tNumber of NaN values in 'references': {num_nan}")

    # check for empty arrays/lists
    def is_empty_array(references):
        if isinstance(references, np.ndarray):
            return len(references) == 0
        return False

    empty_array_mask = df['references'].dropna().apply(is_empty_array)
    num_empty_arrays = empty_array_mask.sum()
    print(f"\tNumber of empty array entries in 'references': {num_empty_arrays}")

    # check for invalid structures
    def is_valid_references_entry(references):
        if isinstance(references, np.ndarray) or isinstance(references, list):
            return all(isinstance(r, str) for r in references)
        return False
    
    df_no_nan = df['references'].dropna()
    invalid_structure_mask_excluding_nan = ~df_no_nan.apply(is_valid_references_entry)
    num_invalid_structures = invalid_structure_mask_excluding_nan.sum()
    print(f"\tNumber of entries with invalid structure (excluding NaN) in 'references': {num_invalid_structures}")

    # check for references newer than the paper itself
    def count_newer_references(references, paper_year):
        if isinstance(references, np.ndarray) and isinstance(paper_year, (int, float)):
            newer_count = 0
            for ref in references:
                ref_year = id_to_year.get(ref)
                if ref_year and ref_year > paper_year:
                    newer_count += 1
            return newer_count
        return 0
    
    df['num_newer_references'] = df.apply(lambda row: count_newer_references(row['references'], row['year']), axis=1)
    print(f"\tNumber of references newer than the paper itself: {df['num_newer_references'].sum()}")
    
    # example of reference year validation for the first paper in the chunk
    if example:
        print()
        print()
        print("Example of reference year validation for the first paper in the chunk:")
        c = df.head(1)
        print(f"\tPaper ID: {c['id'].values[0]}, Year: {c['year'].values[0]}")

        print(f'\tHere the complete references of the paper: {[r for r in list(c["references"].values[0])]}')
        print()
        for ref in c['references'].values[0]:
            print(f"\t\tReference ID: {ref}, Year: {id_to_year.get(ref)}\tis valid? {id_to_year.get(ref)<=c['year'].values[0]}")


def explore_year_trends(chunks):
    """Explore trends in publication years across the dataset."""
    print("Exploring yearly trends in publication data...")
    all_years = []
    for c in tqdm(chunks, desc="Extracting Years from Chunks"):
        all_years.extend(c['year'].dropna().tolist())
    
    if not all_years:
        print("\tNo valid year data found.")
        return
    
    years_series = pd.Series(all_years)
    year_counts = years_series.value_counts().sort_index()
    
    print("\tPublication counts by year:")
    # Plotting the distribution of publication years
    year_counts.plot(kind='bar', figsize=(20, 6), title='Distribution of Publication Years')
    plt.xlabel('Publication Year')
    plt.ylabel('Number of Publications')
    plt.show()

    print("\tTop 10 least common publication years:")
    for i, y in enumerate(year_counts.nsmallest(10).index):
        print(f"\t\t{i}) {y}: {year_counts[y]} occurrences")
    print()
    print("\tTop 10 most common publication years:")
    for i, y in enumerate(year_counts.nlargest(10).index):
        print(f"\t\t{i}) {y}: {year_counts[y]} occurrences")
    print()
    print("\tYears with less than 100 publications:")
    for y in year_counts[year_counts < 100].index:
        print(f"\t\t{y}: {year_counts[y]} occurrences")
    print()
    print("\tOverall publication year statistics:")
    print(f"\t\tAverage publication year: {years_series.mean():.2f}")
    print(f"\t\tMedian publication year: {years_series.median():.2f}")
    print(f"\t\tOldest publication year: {years_series.min()}")
    print(f"\t\tNewest publication year: {years_series.max()}")

def explore_missing_values(chunks, not_values=[]):
    """Explore the distribution of missing values across different columns."""
    print("Exploring distribution of missing values across columns...")
    cols = chunks[0].columns
    total_rows = sum(len(c) for c in chunks)
    column_nan_counts = {col: 0 for col in cols}

    for c in tqdm(chunks, desc="Counting Missing Values in Chunks"):
        for col in column_nan_counts.keys():
            column_nan_counts[col] += c[col].isna().sum()

    print("\n\tMissing value counts and percentages by column:")
    for col, count in column_nan_counts.items():
        percentage = (count / total_rows) * 100 if total_rows > 0 else 0
        print(f"\t\t{col}: {count} missing values ({percentage:.2f}%)")

    if len(not_values) > 0:
        print("\n\tAdditional values treated as missing:")
        for val in not_values:
            print(f"\t\t{val}: {sum(c[col].isin([val]).sum() for c in chunks)} occurrences across all columns")

    print("\n\tColumns sorted by percentage of missing values:")
    sorted_columns = sorted(column_nan_counts.items(), key=lambda x: (x[1] / total_rows) if total_rows > 0 else 0, reverse=True)
    for col, count in sorted_columns:
        percentage = (count / total_rows) * 100 if total_rows > 0 else 0
        print(f"\t\t{col}: {count} missing values ({percentage:.2f}%)")
    
    # plotting the missing value distribution
    # sort by percentage of missing values (ascending)
    sorted_items = sorted(column_nan_counts.items(), key=lambda x: (x[1] / total_rows) if total_rows > 0 else 0)
    cols_sorted = [col for col, _ in sorted_items]
    vals_sorted = [(count / total_rows) * 100 for _, count in sorted_items]

    plt.figure(figsize=(12, 6))
    plt.bar(cols_sorted, vals_sorted)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Percentage of Missing Values')
    plt.title('Percentage of Missing Values by Column')
    plt.show()


