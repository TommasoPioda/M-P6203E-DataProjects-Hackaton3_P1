Problems:
- persone a cui mancano dati (e.g.: id, org, ...), dobbiamo riuscire a reperirli basandoci su altri articoli, ma per esempio per **org** non bisogna prendere il più recente ma quello di uno stesso articolo dello stesso anno

- ricerca contando anche piccoli type errors e accenti sbagliati (sempre basati su data)

- type error:ex\.g. in venue there's written conf but in doc_type (conf/journal) --> journal

CHAT GPT:
I want to add dtypes, nan_values, not_nan_values as rows in the dataset with indexes 'dtype', 'num_nan', 'num_not_nan': def global_inspection(df):
    df = df.copy()
    df.explode('authors', ignore_index=True)
    df.explode('keywords', ignore_index=True)

    df_descr = df.describe(include='all')
    dtypes = df.dtypes
    nan_values = df.isna().sum()
    not_nan_values = df.notna().sum()
    
    global_df = pd.concat([df_descr, dtypes.rename('dtype'), nan_values.rename('num_nan'), not_nan_values.rename('num_not_nan')], axis=1)

    print("Global DataFrame Description:")

    return global_df

global_inspection(peek_file(0, rows=1000))



You're very close — the only issue is that your extra rows (dtype, num_nan, num_not_nan) must have the same index as df.describe(), otherwise pandas aligns them incorrectly.

Also, your explode calls do nothing because you're not assigning the result back.

Here’s a clean, correct version that produces exactly what you want.

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
    global_df = pd.concat([df_descr, dtypes_df, nan_df, not_nan_df], axis=0)

    print("Global DataFrame Description:")
    return global_df
