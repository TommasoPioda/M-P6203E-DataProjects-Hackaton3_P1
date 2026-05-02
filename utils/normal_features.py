import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder

def encode_categorical(df_train, df_val, df_test, ohe_cols=['lang_article', 'lang_ref', 'doc_type_article', 'doc_type_ref'], cols_top=['lang_article', 'lang_ref'], n_top=5):
    """ 
    Perform the encoding on the given columns, in addition is possible to perform a TOP_N encoding on some columns, where only the most N-frequent values will be encoded, th others will be encoded in a column named other
    """
    # Select the N more frequent
    for col in cols_top:
        top_n = df_train[col].value_counts().nlargest(n_top).index.tolist()
        
        for df in [df_train, df_val, df_test]:
            df[col] = df[col].apply(lambda x: x if x in top_n else 'Other')

    # One-Hot Encoding
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    # Fit on train, transform on others
    # to exclude nan we put a tmp string
    train_ohe = encoder.fit_transform(df_train[ohe_cols].fillna('UNKNOWN_VAL'))
    val_ohe = encoder.transform(df_val[ohe_cols].fillna('UNKNOWN_VAL'))
    test_ohe = encoder.transform(df_test[ohe_cols].fillna('UNKNOWN_VAL'))
    
    # Create df
    ohe_features = encoder.get_feature_names_out(ohe_cols)
    # Exclude nan
    cols_to_keep = [c for c in ohe_features if 'UNKNOWN_VAL' not in c]
    
    df_train_ohe = pd.DataFrame(train_ohe, columns=ohe_features, index=df_train.index)[cols_to_keep]
    df_val_ohe = pd.DataFrame(val_ohe, columns=ohe_features, index=df_val.index)[cols_to_keep]
    df_test_ohe = pd.DataFrame(test_ohe, columns=ohe_features, index=df_test.index)[cols_to_keep]
    
    df_train = pd.concat([df_train.drop(columns=ohe_cols), df_train_ohe], axis=1)
    df_val = pd.concat([df_val.drop(columns=ohe_cols), df_val_ohe], axis=1)
    df_test = pd.concat([df_test.drop(columns=ohe_cols), df_test_ohe], axis=1)
    
    return df_train, df_val, df_test


from sklearn.feature_extraction import FeatureHasher

def hash_features(df_train, df_val, df_test, hash_cols=['keywords_article', 'keywords_ref'], n_features=16):
    """
    Perform the hash encoding with the given number of features to make the hashing
    """
    hasher = FeatureHasher(n_features=n_features, input_type='string')
    
    for col in hash_cols:
        # exclude nan
        t_data = df_train[col].apply(lambda x: list(x) if isinstance(x, (list, np.ndarray)) else [])
        v_data = df_val[col].apply(lambda x: list(x) if isinstance(x, (list, np.ndarray)) else [])
        te_data = df_test[col].apply(lambda x: list(x) if isinstance(x, (list, np.ndarray)) else [])

        train_hashed = hasher.transform(t_data).toarray()
        val_hashed = hasher.transform(v_data).toarray()
        test_hashed = hasher.transform(te_data).toarray()
        
        cols = [f"{col}_hash_{i}" for i in range(n_features)]
        
        df_train = pd.concat([df_train.drop(columns=[col]), 
                              pd.DataFrame(train_hashed, columns=cols, index=df_train.index)], axis=1)
        df_val = pd.concat([df_val.drop(columns=[col]), 
                            pd.DataFrame(val_hashed, columns=cols, index=df_val.index)], axis=1)
        df_test = pd.concat([df_test.drop(columns=[col]), 
                             pd.DataFrame(test_hashed, columns=cols, index=df_test.index)], axis=1)
            
    return df_train, df_val, df_test

