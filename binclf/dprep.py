"""
binclf dprep
==================================
Utils library for data cleaning and preparation.

Author: Casokaks (https://github.com/Casokaks/)
Created on: Nov 1st 2018

"""


import statistics as stat
from sklearn.preprocessing import LabelEncoder


def drop_nan(df, nan_max=0.9, verbose=False):
    '''To remove features with too many nan from dataframe
    
    Parameters
    ----------
    
    Returns
    -------
    
    '''    

    nan_prc = df.isnull().sum()/len(df)
    cols_drop = list(nan_prc[nan_prc>nan_max].index)
    if verbose == True:
        print('Columns dropped due to high number of nan: {}'.format(cols_drop))
    return df.drop(labels=cols_drop, axis=1, inplace=False)


def drop_corr(df, corr_max=0.95, verbose=False):
    '''To remove correlated features from dataframe
    
    Parameters
    ----------
    
    Returns
    -------
    
    '''
    
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > corr_max)]
    if verbose == True:
        print('Columns dropped due to high correlation: {}'.format(to_drop))
    return df.drop(labels=to_drop, axis=1, inplace=False)


def encode_df_cols(ts, encoder):
    '''To encode a categorical timeseries
    
    Parameters
    ----------
    
    Returns
    -------
    
    '''
    
    cnt_nan = ts.isnull().sum()
    
    # if no nan
    if cnt_nan == 0:
        ts = encoder.fit_transform(ts)
    
    # keep nan
    else:
        idx = list(np.where(ts.isnull())[0])
        ts = ts.astype(str)
        ts = encoder.fit_transform(ts)
        ts = ts.astype(float)
        ts[idx] = np.nan 
    
    return ts


def decompose_date(ts, date_format='%d-%m-%Y', to_type='categ'):
    '''To decompose date timeseries into multiple columns.
    
    Parameters
    ----------
    ts : pandas timeseries
        timeseries of dates
    format : string, optional
        format of input dates
    to_type : string ['categ','numeric'], optional
        to make new date feature columns either categorical or numeric dtype
    
    Returns
    -------
    pandas dataframe with date_decomposed columns: 
    day of the month, day of the week, week, month, year.
    
    '''
    
    if to_type == 'categ':
        to_type = 'str'
    elif to_type == 'numeric':
        to_type = 'int'
    else:
        print('ERROR: to_type param not recognized')
        sys.exit(-1)
        
    ts = pd.to_datetime(ts, format=date_format)
    df = pd.DataFrame(ts)    
    df['{}_DAYMONTH'.format(ts.name)] = ts.dt.day.astype(to_type)
    df['{}_DAYWEEK'.format(ts.name)] = ts.dt.weekday.astype(to_type)
    df['{}_WEEK'.format(ts.name)] = ts.dt.week.astype(to_type)
    df['{}_MONTH'.format(ts.name)] = ts.dt.month.astype(to_type)
    df['{}_YEAR'.format(ts.name)] = ts.dt.year.astype(to_type)
    
    return df
    

def fill_nan(ts, strategy, verbose=True):
    '''To replace nan with selected strategy. 
    To be used column by column in case of dataframe.
    
    Parameters
    ----------
    ts : timeseries
        original timeseries (dataframe column) 
    strategy : string
        strategy to replace missing values 
        [None, 'median','mean','zero','negone'] if numeric columns
        [None, 'nan','mode'] if string columns 
    verbose : boolean, optional
        print of not informations
    
    Returns
    -------
    filled timeseries
    
    '''
    
    nan_count = ts.isnull().sum()
    
    if strategy != None:
        try:
            if strategy == 'median':
                fill = np.nanmedian(ts.values)
            elif strategy == 'mean':
                fill = np.nanmean(ts.values)
            elif strategy == 'zero':
                fill = 0
            elif strategy == 'negone':
                fill = -1
            elif strategy == 'mode':
                fill = stat.mode(list(ts.values))
            elif strategy == 'nan':
                fill = 'nan'
            else:
                print('ERROR: strategy not recognized')
                sys.exit(-1)
    
            ts = ts.fillna(fill)
            nan_count_new = ts.isnull().sum()
            
            if verbose == True:
                print('{} NaNs replaced with {} strategy (NaNs from {} to {})'.format(
                        ts.name, strategy, nan_count, nan_count_new))
    
        except:
            print('''ERROR: fill_nan(ts={}, strategy={}) operation aborted 
                  due to some problem (fill={})'''.format(ts.name, strategy, fill))
            sys.exit(-1)
    
    else:
        if verbose == True:
            print('{} NaNs not filled due to {} strategy ({} NaNs)'.format(
                    ts.name, strategy, nan_count))
    
    return ts 


def drop_cols(df, drop_cols):
    '''To drop columns from dataframe
    
    Parameters
    ----------
    df : dataframe
        original dataframe
    drop_cols : list
        list of column names (string) to be removed from df
    
    Returns
    -------
    dataframe without the specified columns
    
    '''
    
    print('Input shape:',df.shape)
    
    for col in drop_cols:
        if col in df.columns:
            df.drop(labels=col, axis=1, inplace=True)
        else:
            print('{} already not present..'.format(col))

    print('Output shape:',df.shape)
    return df


def reorder_df_cols(df, label_col):
    '''Todo documentation
    
    Parameters
    ----------
    
    Returns
    -------
    
    '''
    
    new_cols = sorted(list(df.columns))
    new_cols.remove(label_col)
    new_cols = [label_col] + new_cols
    df = df.reindex(columns=new_cols)
    return df    
    

def prepare_label(df, label_col, verbose=True):
    '''Todo documentation
    
    Parameters
    ----------
    
    Returns
    -------
    
    '''
    
    print('\n--- Preparing LABEL columns ---\n')
    print('Input shape:',df.shape)
    print('Tot no. of NAN =',df.isnull().sum().sum())

    for col in [label_col]:
        print('')
        if col in df.columns:
            if df[label_col].isnull().sum() > 0: # missing
                print('ERROR: Label column {} contains missing values...'.format(col))
                sys.exit(-1)
            df[col] = df[col].astype(str)
            df[col] = encode_df_cols(ts=df[col], encoder=LabelEncoder())
            if verbose==True:
                print('Label column {} encoded with LabelEncoder'.format(col))    
        else:
            print('WARNING: Label column {} not present...'.format(col))
            
    print('')
    print('Output shape:',df.shape)
    print('Tot no. of NAN =',df.isnull().sum().sum())
    print('\n--- LABEL column preparation completed ---\n')
    return df


def prepare_numerics(df, numeric_cols=[], numeric_nan='mean', numeric_col_na=True,
                     drop_still_nan=False, verbose=True):
    '''Todo documentation
    
    Parameters
    ----------
    
    Returns
    -------
    
    '''
    
    print('\n--- Preparing NUMERIC columns ---\n')
    print('Input shape:',df.shape)
    print('Tot no. of NAN =',df.isnull().sum().sum())
    
    for col in numeric_cols:
        print('')
        if col in df.columns:
            enc_cols = []
            
            # force to float
            # if not able then drop the column from dataframe and move to next column
            try:
                df[col] = df[col].astype(float)
            except:
                df.drop(labels=[col], axis=1, inplace=True)
                print('WARNING: Numeric column {} dropped ' 
                      'because unable to make it float'.format(col))
                continue
            
            # add nan column to dataframe (encode later)
            if numeric_col_na==True:
                cnt_nan = df[col].isnull().sum()
                new_name = '{}_NA'.format(col)
                if cnt_nan > 0:
                    df[new_name] = np.nan
                    df.loc[df[col].isnull()==True,new_name] = 1
                    df.loc[df[col].isnull()==False,new_name] = 0
                    enc_cols.extend([new_name])
                    if verbose==True:
                        print('Column {} created based on column {} NaNs'.format(new_name, col))
                else:
                    if verbose==True:
                        print('Column {} not created since no NaN was found ' 
                              'in column {}'.format(new_name, col))                    
                
            # fill nan
            df[col] = fill_nan(df[col], numeric_nan)
            cnt_nan = df[col].isnull().sum()
            if verbose==True:
                print('Numeric column {} filled with strategy {} ({} NaNs)'.format(
                        col, numeric_nan, cnt_nan))        
            
            # drop if nan still present
            # but keep the new nan column
            if cnt_nan > 0: 
                if drop_still_nan == True:
                    df.drop(inplace=True, axis=1, labels=[col])
                    print('WARNING: Numeric column {} dropped due to unfixed NaNs '
                          '({} NaNs)'.format(col, cnt_nan))
                else:
                    print('WARNING: Numeric column {} still contains NaNs '
                          '({} NaNs)'.format(col, cnt_nan)) 
            else:
                if verbose==True:
                    print('Numeric column {} with no NaN'.format(col))
                
            # encode columns
            # at this point the list might be empty or not
            # but if not empty columns need to be encoded
            for ec in enc_cols:
                df[ec] = encode_df_cols(ts=df[ec], encoder=LabelEncoder())
                                
        else:
            print('WARNING: Numeric column {} not present...'.format(col))

    # return
    print('')
    print('Output shape:',df.shape)
    print('Tot no. of NAN =',df.isnull().sum().sum())
    print('\n--- NUMERIC columns preparation completed ---\n')    
    return df


def prepare_categoricals(df, categ_cols=[], categ_nan='mode', categ_col_na=True,
                         one_hot=True, categ_limit=55, drop_still_nan=True, verbose=True):
    '''Todo documentation
    
    Parameters
    ----------
    
    Returns
    -------
    
    '''
    
    print('\n--- Preparing CATEGORICAL columns ---\n')
    print('Input shape:',df.shape)
    print('Tot no. of NAN =',df.isnull().sum().sum())
        
    for col in categ_cols:
        print('')
        if col in df.columns:
            enc_cols = [col]
            
            # force the dataframe columns to string
            # if not successful, drop it and pass to next column
            try:
                df[col] = df[col].astype(str)
                df.loc[df[col]=='nan',col] = np.nan
            except:
                df.drop(labels=[col], axis=1, inplace=True)
                print('WARNING: Categorical column {} dropped and preparation process stopped '
                      'because unable to make it str'.format(col))
                continue

            # add nan column to the dataframe if there are NANs
            if categ_col_na==True:
                cnt_nan = df[col].isnull().sum()
                new_name = '{}_NA'.format(col)
                if cnt_nan > 0:
                    df[new_name] = np.nan
                    df.loc[df[col].isnull()==True,new_name] = 1
                    df.loc[df[col].isnull()==False,new_name] = 0
                    enc_cols.extend([new_name])
                    if verbose==True:
                        print('Column {} created based on column {} NaNs'.format(new_name, col))
                else:
                    if verbose==True:
                        print('Column {} not created since no NaN was found ' 
                              'in column {}'.format(new_name, col))                    

            # fill nan of the dataframe column 
            df[col] = fill_nan(df[col], categ_nan)            
            cnt_nan = df[col].isnull().sum()
            if verbose==True:
                print('Categorical column {} filled with strategy {} ({} NaNs)'.format(
                        col, categ_nan, cnt_nan))        
                    
            # build one hot encoding and add columns to dataframe
            if one_hot == True:
                
                # if too many values, then do not perform one-hot:
                # delete the dataframe column (keep the new nan column if added)
                # and pass to next column
                cnt_val = len(list(set(df[col].values)))
                if cnt_val > categ_limit:
                    df.drop(inplace=True, axis=1, labels=[col])
                    enc_cols.remove(col)
                    print('WARNING: Categorical column {} not one-hot encoded and dropped '
                          'due to many unique values ({} values vs limit of {})'.format(
                          col, cnt_val, categ_limit))
                    continue
                
                # if not too many values perform one-hot
                # create new columns, add them to the dataframe
                # drop the original column
                else:
                    
                    onehot_df = pd.get_dummies(df[col], prefix=col, prefix_sep='_', dummy_na=False)
                    df.drop(inplace=True, axis=1, labels=[col])
                    enc_cols.remove(col)
                    df = df.merge(onehot_df, left_index=True, right_index=True)
                    enc_cols.extend(list(onehot_df.columns))
                    if verbose==True:
                        print('Categorical column {} transformed to One-Hot columns '
                              'and dropped ({}: {})'.format(col, len(enc_cols), enc_cols)) 
                 
            # in case ont-hot is true: original column is always dropped from dataframe
            # in case of false, need to make sure there are no NAN left (if param is true)
            # while keep the new nan (if added), since it is inherently with no nan
            else:
                if cnt_nan > 0:
                    if drop_still_nan==True:
                        df.drop(inplace=True, axis=1, labels=[col])
                        enc_cols.remove(col)
                        print('WARNING: Categorical column {} dropped due to unfixed NaNs '
                              '({} NaNs)'.format(col, cnt_nan))
                    else:
                        print('WARNING: Categorical column {} still contains NaNs '
                              '({} NaNs)'.format(col, cnt_nan))     
                else:
                    if verbose==True:
                        print('Categorical column {} with no NaN'.format(col))
    
            # encode columns
            # at this point the list might be empty or not
            # but if not empty columns need to be encoded
            for ec in enc_cols:
                df[ec] = encode_df_cols(ts=df[ec], encoder=LabelEncoder())
                          
        else:
            print('WARNING: Categorical column {} not present...'.format(col))

    # return dataframe 
    print('')
    print('Output shape:',df.shape)
    print('Tot no. of NAN =',df.isnull().sum().sum())
    print('\n--- CATEGORICAL columns preparation completed ---\n')    
    return df


def prepare_dates(df, date_cols=[], date_format='%d-%b-%y', date_to_type='categ', 
                  one_hot=True, categ_limit=55, drop_nan=True, verbose=True):
    '''Todo documentation
    one_hot only used is date_to_type is categ
    
    Parameters
    ----------
    
    Returns
    -------
    
    '''
    
    print('\n--- Preparing DATE columns ---\n')
    print('Input shape:',df.shape)
    print('Tot no. of NAN =',df.isnull().sum().sum())

    for col in date_cols:
        print('')
        if col in df.columns:
            
            # decomposing dates into additional features
            date_features = decompose_date(df[col], date_format=date_format, 
                                           to_type=date_to_type)
            if verbose==True:
                print('Date column {} expanded with {} new date_features {}'.format(col, 
                      len(date_features.columns), list(date_features.columns)))
            
            # encode columns
            if date_to_type=='numeric':
                date_features = prepare_numerics(df=date_features, 
                                                 numeric_cols=list(date_features.columns), 
                                                 numeric_nan=None, numeric_col_na=True,
                                                 drop_still_nan=drop_nan, verbose=verbose)
            
            elif date_to_type=='categ':
                date_features = prepare_categoricals(df=date_features, 
                                                     categ_cols=list(date_features.columns), 
                                                     categ_nan=None, categ_col_na=True,
                                                     one_hot=one_hot, categ_limit=categ_limit, 
                                                     drop_still_nan=drop_nan, verbose=verbose)

            else:
                print('ERROR: date_to_type not recognized')
                sys.exit(-1)
            
            # merge with df
            df.drop(labels=col, axis=1, inplace=True)
            df = df.merge(date_features, left_index=True, right_index=True)
                
        else:
            print('WARNING: Date column {} not present...'.format(col))

    print('')
    print('Output shape:',df.shape)
    print('Tot no. of NAN =',df.isnull().sum().sum())
    print('\n--- DATE columns preparation completed ---\n')
    
    return df


def prepare_df_clf(df, label_col, 
                   numeric_cols=[], numeric_nan='mean', numeric_col_na=True,
                   categ_cols=[], categ_nan='mode', categ_col_na=True,
                   date_cols=[], date_format='%d-%b-%y', date_to_type='categ',
                   one_hot=True, categ_limit=55, drop_still_nan=True, verbose=True):
    '''To prepare dataframe for Classification. 
    Very simple missing value strategy implemented, suggested to properly handle missing values before this.
    
    Parameters
    ----------
    df : dataframe
        original dataframe with full dataset
    label_col : string
        name of the column to treat as label (apply sklearn LabelEncoder)
    numeric_cols : list
        list of column names (string) to be treated as numeric ()
    numeric_nan : string, optional
        strategy to replace missing values [None, 'median','mean','zero','negone'] 
    numeric_col_na : boolean, optional
        to add or not a new one-hot column to track missing values of numeric columns 
    categ_cols : list
        list of column names (string) to be treated as categorical 
        (apply pandas get_dummy for one-hot encoding)
    categ_nan : string, optional
        strategy to replace missing values [None, 'nan','mode']
    categ_col_na : boolean, optional
        to add or not a new one-hot column to track missing values of categ columns 
    date_cols : list, optional
        list of column names (string) to treat as date (geenrate dates fetures)
    date_format : string, optional
        format of date_cols to be used for conversion
    date_to_type : string ['categ','numeric'], optional
        to make new date feature columns either categoricat or numeric dtype
    one_hot : boolean, optional
        apply or not one-hot encoding (pandas get_dummy)
    categ_limit : integer, optional
        maximum number of values for a categorical column to be kept 
        (and maybe transformed into one-hot encoding).
        If the number of values exceed the limit then the column is dropped.
    drop_still_nan : boolean, optional
        after cleaning and preparation, drop columns that still present NANs. 
    verbose : boolean, optional
        to or not to print actions taken.
    
    Returns
    -------
    clean dataframe ready for classification. 
    Numeric columns are returned as float columns,
    categorical columns are returned as int columns (one-hot).
    
    '''

    # label
    df = prepare_label(df=df, label_col=label_col, verbose=verbose)

    # date columns
    df = prepare_dates(df=df, date_cols=date_cols, 
                       date_format=date_format, date_to_type=date_to_type, 
                       one_hot=one_hot, categ_limit=categ_limit, 
                       drop_nan=drop_still_nan, verbose=verbose)
    
    # numeric columns 
    df = prepare_numerics(df=df, numeric_cols=numeric_cols, 
                          numeric_nan=numeric_nan, numeric_col_na=numeric_col_na,
                          drop_still_nan=drop_still_nan, verbose=verbose)
        
    # categorical columns
    df = prepare_categoricals(df=df, categ_cols=categ_cols, 
                              categ_nan=categ_nan, categ_col_na=categ_col_na,
                              one_hot=one_hot, categ_limit=categ_limit, 
                              drop_still_nan=drop_still_nan, verbose=verbose)

    # reordering columns by name
    df = reorder_df_cols(df=df, label_col=label_col)

    # last check on columns (all must be dtype)
    print('\n--- FINAL check and cleaning ---\n')
    for col in df.columns:
        dtp = df[col].dtype
        if dtp not in ['int64','float64','int32','float32']:
            df.drop(labels=col, axis=1, inplace=True)            
            print('!!! >>> WARNING: column {} dropped due to dtype {}, '
                  'when only either int64 or float64 are expected'.format(col, dtp))            

    print('\n--- Data preparation process completed ---\n')
    return df






























