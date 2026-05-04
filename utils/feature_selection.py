import numpy as np


# if we remove page_start and page_end, we will lose the possibility to calculate the number of pages, which can be a useful feature for some analysis, so we add it before removing the columns
def add_n_pages(df):
    """Add a new column 'n_pages' to the DataFrame, calculated as page_end - page_start + 1."""
    df = df.copy()
    df['n_pages'] = (df['page_end'] - df['page_start'] + 1).where(df['page_end'].notna() & df['page_start'].notna(), np.nan)
    return df

# remove_columns = lambda df, cols: df.drop(columns=[col for col in cols if col in df.columns], axis=1) if cols in df.columns else df
def remove_columns(df, cols):
    return df.drop(columns=[c for c in cols if c in df.columns], errors="ignore")
