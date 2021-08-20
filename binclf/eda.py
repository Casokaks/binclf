"""
binclf eda
==================================
Utils library for Exploratory data analysis (EDA).

Author: Casokaks (https://github.com/Casokaks/)
Created on: Nov 1st 2018

"""


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display


def feature_stats(df, features_list, label, nan=False):
    '''Plot and return features simple statistics for each label class.
    
    Parameters
    ----------
    df : pandas dataframe
        data
    features_list : list
        list of features to analyze
    label: string
        name of the label column 
    nan : boolean, optional
        keep nan or not
    
    Returns
    -------
    dictionary of dataframes
        
    '''        
    
    df_dict = {}
    for f in range(len(features_list)):
        
        feature = features_list[f]
        vals = list(set(df[feature].dropna().values))
        idx = ['ALL'] + vals 
        if nan == True:
            idx = idx + [np.nan]
            
        plot_df = pd.DataFrame(index=idx,columns=['AVG','SUM','CNT','SUP'])
        plot_df.index.name = feature
        for idx_value in idx:
            if idx_value == 'ALL':
                plot_df.loc[idx_value,'AVG'] = np.nanmean(df[label].values)*100
                plot_df.loc[idx_value,'SUM'] = np.nansum(df[label].values)
                plot_df.loc[idx_value,'CNT'] = len(df[label].values)
                plot_df.loc[idx_value,'SUP'] = plot_df.loc[idx_value,'CNT']*100/len(df[label])
            else:
                plot_df.loc[idx_value,'AVG'] = np.nanmean(df[label][df[feature]==idx_value].values)*100
                plot_df.loc[idx_value,'SUM'] = np.nansum(df[label][df[feature]==idx_value].values)
                plot_df.loc[idx_value,'CNT'] = len(df[label][df[feature]==idx_value].values)
                plot_df.loc[idx_value,'SUP'] = plot_df.loc[idx_value,'CNT']*100/len(df[label])
        plot_df.fillna(value=np.nan,inplace=True)
        
        display(plot_df.sort_values(by='AVG',axis=0,ascending=False).transpose().round(2))
        df_dict[f] = plot_df    

    return df_dict


def feature_analysis(feature_ts, label_ts, density_kernel='gau', pointplot_estimator=np.mean, 
                     hsize=18, vsize=5, pics_save_path=None):
    '''Plot feature analysis. Can handle nan.
    
    Parameters
    ----------
    feature_ts : TimeSeries
        timeseries of the feature to analyze
    feature_ts : TimeSeries
        timeseries of the label
    density_kernel: string, {‘gau’ | ‘cos’ | ‘biw’ | ‘epa’ | ‘tri’ | ‘triw’ }, optional
        kernel to use for density estimation in seaborn kdeplot plot
    pointplot_estimator : callable, optinal
        Statistical function (np.mean, np.median, etc) to estimate within each categorical bin.
    hsize : positive integer, optional
        horizontal size of the plot
    vsize : positive integer or None, optional
        vertical size of the plot
    
    Returns
    -------
        
    '''    
    
    sns.set_style('whitegrid')
    
    feature_type = feature_ts.dtype.type
    df = pd.DataFrame(data=feature_ts, index=feature_ts.index, columns=[feature_ts.name])
    df = df.merge(pd.DataFrame(data=label_ts, index=label_ts.index, columns=[label_ts.name]), 
                  left_index=True, right_index=True)    
    legend_label = 'Label'
    df.loc[:,legend_label] = np.nan
    df.loc[df[label_ts.name]==0,legend_label] = '{} = 0'.format(label_ts.name)
    df.loc[df[label_ts.name]==1,legend_label] = '{} = 1'.format(label_ts.name)                      
    
    # numeric: 
    if feature_type in [np.float64, np.int64] and len(list(set(feature_ts.values))) > 2:        
        f, axes = plt.subplots(nrows=1, ncols=3, figsize=(hsize, vsize))
        
        # (LX) conditional distribution
        ax = axes.flatten()[0]        
        for lval in list(set(label_ts.values)):
            fts = df[df[label_ts.name]==lval][feature_ts.name]
            fts.name = '{} = {}'.format(label_ts.name, lval)
            sns.kdeplot(fts, kernel=density_kernel, shade=True, legend=True, ax=ax)        
        ax.set_xlabel(feature_ts.name)
        ax.set_ylabel('Density')
        ax.set_title('Density plot')
        
        # (CX) conditional violin plot
        ax = axes.flatten()[1]   
        ax = sns.violinplot(data=df, x=feature_ts.name, y=label_ts.name, hue=legend_label,
                            orient='h', ax=ax, cut=0)
        ax.set_title('Violin plot')
        ax.legend(loc='center right')
        
        # (RX) conditional point plot 
        ax = axes.flatten()[2]
        ax = sns.pointplot(data=df, y=label_ts.name, x=feature_ts.name, hue=legend_label, 
                           estimator=pointplot_estimator, orient='h', ax=ax)        
        ax.set_title('Point plot')
        ax.legend(loc='center right')

        plt.show()
        
        if isinstance(pics_save_path,str):
            f.savefig('{}/{}.png'.format(pics_save_path,feature_ts.name)) 

    # one-hot: 
    elif feature_type == np.int64 and sorted(list(set(feature_ts.values))) == [0,1]:
        f, axes = plt.subplots(nrows=1, ncols=3, figsize=(hsize, vsize))
        
        # (LX) count plot
        ax = axes.flatten()[0]        
        ax = sns.countplot(data=df, x=feature_ts.name, hue=legend_label, 
                           orient='h', ax=ax)
        ax.set_title('Count plot')

        # (CX) ratio plot
        dfrx = df.loc[:,[label_ts.name,feature_ts.name]]
        dfr = dfrx[dfrx[label_ts.name]==1].groupby(by=feature_ts.name).count()
        dfr = dfr / dfrx.groupby(by=feature_ts.name).count()
        dfr = dfr.merge(pd.DataFrame(1-dfr), right_index=True, left_index=True)
        dfr.columns = ['{} = {}'.format(label_ts.name, 1), '{} = {}'.format(label_ts.name, 0)]
        ax = axes.flatten()[1]      
#        ax.bar(x=[0,1], tick_label=list(dfr.index), height=dfr.iloc[:,1], bottom=None,
#               label=dfr.columns[1])
#        ax.bar(x=[0,1], tick_label=list(dfr.index), height=dfr.iloc[:,0], bottom=dfr.iloc[:,1], 
#               label=dfr.columns[0])
        ax.bar(x=[0,1], tick_label=list(dfr.index), height=dfr.iloc[:,0], bottom=None, 
               label=dfr.columns[0], color='grey')
        ax.set_xlabel(feature_ts.name)
        ax.set_ylabel(label_ts.name)
        ax.set_ylim(top=max(dfr.iloc[:,0])*1.20)
        ax.set_title('Ratio plot')
        ax.legend(loc='upper right')

        # (RX) point plot
        ax = axes.flatten()[2]        
        ax = sns.pointplot(data=df, y=label_ts.name, x=feature_ts.name, hue=legend_label,
                           estimator=pointplot_estimator, orient='h', ax=ax)
        ax.set_title('Point plot')
        ax.legend(loc='center right')

        plt.show()

        if isinstance(pics_save_path,str):
            f.savefig('{}/{}.png'.format(pics_save_path,feature_ts.name)) 

    else:
        print('ERROR: {} column type {} with values {} not supported...'.format(
                feature_ts.name,    feature_type, sorted(list(set(feature_ts.values)))))

    
def plot_corr(df, label_name, size=None):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        label_name: string, name of the label
        size: vertical and horizontal size of the plot
        
    Output:
        correlation matrix
    
    '''

    corr = df.corr()
    ord_cols = list(corr.sum().sort_values(ascending=False).index)
    corr = corr[ord_cols].reindex(index=ord_cols)    

    if size == None:
        size = len(corr)/2
    fig, ax = plt.subplots(figsize=(size, round(size*0.8)))
    
    labels = list(corr.columns.values)
    #labels[labels==label_name] = '>>>>> LABEL >>>>> {}'.format(label_name)
    #print(labels)

    sns.set(style='white')
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, xticklabels=labels, yticklabels=labels, ax=ax, mask=mask,               
                cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    ax.set_title('Features correlation matrix')
    plt.show()
    
    return corr
    
    
def hmap_label_through_time(df, label_col, yeal_col='YEAR', month_col='MONTH', hsize=18, vsize=8):
    '''Plot feature analysis.
    
    Parameters
    ----------
    df : pandas dataframe
        containing label, years and months
    label_col : string
        name of the columns containing the label (binary [0,1])
    year_col : string
        name of the columns containing years
    month_col : string
        name of the columns containing months
    hsize : positive integer, optional
        horizontal size of the plot
    vsize : positive integer or None, optional
        vertical size of the plot
    
    Returns
    -------
    two pandas dataframe with label average and sample counts by months and years
        
    '''    
    
    years = sorted(list(set(df[yeal_col].astype(int))))
    months = sorted(list(set(df[month_col].astype(int))))
    hmdf_avg = pd.DataFrame(index=months,columns=years)
    hmdf_cnt = pd.DataFrame(index=months,columns=years)
    for y in years:
        for m in months:
            avg = np.nanmean(df[(df[yeal_col]==str(y)) & (df[month_col]==str(m))][label_col].values)
            cnt = len(df[(df[yeal_col]==str(y)) & (df[month_col]==str(m))][label_col].values)
            hmdf_avg.loc[m,y] = avg
            hmdf_cnt.loc[m,y] = cnt
    hmdf_avg.fillna(value=np.nan,inplace=True)
    hmdf_cnt.fillna(value=np.nan,inplace=True)
    
    # plot avg
    fig = plt.figure()
    fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(hsize,vsize))
    sns.heatmap(hmdf_avg*100, cmap='RdYlGn_r', linewidths=0.5, annot=True, ax=ax[0])
    sns.heatmap(hmdf_cnt, cmap='RdYlGn_r', linewidths=0.5, annot=True, ax=ax[1])
    ax[0].set(xlabel=yeal_col, ylabel=month_col, title='Label Average')
    ax[1].set(xlabel=yeal_col, ylabel=month_col, title='No. of Samples')
    plt.show()
    
    return hmdf_avg, hmdf_cnt

    
    