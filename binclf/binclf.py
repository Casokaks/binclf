"""
binclf binclf
==================================
Utils library for Binary Classification.

Author: Casokaks (https://github.com/Casokaks/)
Created on: Nov 1st 2018

"""


from copy import deepcopy
import math
import statistics as stat
import itertools
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics 
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display


def performance_full_analysis(clf_perf_list, classes, left='train', right='test', 
                              round_dec=3, hsize=15, vsize=5):
    '''Plot full x-validation performance analysis of a classifier.
    
    Parameters
    ----------
    clf_perf_list : list 
        list (x-val iterations) of dictionaries (train and test) of dictionary (sklearn metrics)  
    classes : list
        label classes
    left : string, optional
        ['train','test'] depending on which roc graph should be plotted on the left.
    right : None or string, optional
        None to not plot a second roc graph. ['train','test'] to plot the respective roc graph on the right.
    round_dec : integer, optional
        to round decimals
    hsize : positive integer, optional
        horizontal size of the plot
    vsize : positive integer, optional
        vertical size of the plot
    
    Returns
    -------
    
    '''

    # plot xval performance    
    df = xval_performances(performance_list=clf_perf_list, dsets=[left,right], round_dec=3)
    
    print('\n>>> Performance overall')
    display(df[['accuracy','log-loss','cohen-kappa']])
    
    for n in ['micro','0','1']:
        print('\n>>> Performance of class = {}'.format(n))
        display(df[['precision.{}'.format(n),'recall.{}'.format(n),
                    'f1-score.{}'.format(n),'auc.{}'.format(n),'avg-prc-rec.{}'.format(n)]])
    
    # plot confusion matrix
    for norm in [False,True]:
        for agg in ['sum','mean']:
            tit = '\n>>> X-Val Confusion-Matrix aggregated by {}'.format(agg)
            if norm == True:
                tit = '{} (percentage)'.format(tit)
            else:
                tit = '{} (absolute value)'.format(tit)                
            print(tit)
            plot_confusion_matrix(performance_list=clf_perf_list, classes=classes, 
                                  confmatrix_name='confusion-matrix', left=left, right=right, 
                                  normalize=norm, aggregate=agg, cmap=None, hsize=hsize, 
                                  vsize=vsize)

    # plot roc
    for n in ['micro','0','1']:
        print('\n>>> ROC plot of class = {}'.format(n))
        plot_roc_curve(performance_list=clf_perf_list, left='train', right='test', 
                       fp_name='fpr-curve.{}'.format(n), tp_name='tpr-curve.{}'.format(n), 
                       hsize=hsize, vsize=vsize)
    
    # plot racall / precision curve + average precision performance
    for n in ['micro','0','1']:  
        print('\n>>> Recall-Precision plot of class = {}'.format(n))
        plot_precision_recall_curve(performance_list=clf_perf_list, left='train', right='test', 
                                    prc_name='precision-curve.{}'.format(n), 
                                    rec_name='recall-curve.{}'.format(n), 
                                    apr_name='avg-prc-rec.{}'.format(n), 
                                    hsize=hsize, vsize=vsize)


def __prepare_confmatrix_plot(cm, target_names, title, normalize, cmap, ax):
    
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    #ax.colorbar()

    try:
        target_names = target_names[0]
        if type(target_names) != type(None):
            ax.set_xticks(ticks=target_names)
            ax.set_yticks(ticks=target_names)
    except:
        pass
    
    #for tick in ax.get_xticklabels():
    #    tick.set_rotation(45)

    if normalize:
        cm = cm.astype('float') / cm.sum().sum()
        #cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            ax.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            ax.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))


def plot_confusion_matrix(performance_list, classes, confmatrix_name='confusion-matrix', 
                          left='train', right='test', normalize=False, aggregate='sum',
                          cmap=None, hsize=18, vsize=6):
    '''Plot nice confusion matrix give a xval list of performances.
    
    Parameters
    ----------
    performance_list : list
        list of performances (of different xval iterations) containing confusion matrices,
        as returned by metrics.confusion_matrix
    classes : list
        list of classes name
    confmatrix_name : string
        dictionary key of the field within performance_list containing confusion matrices. 
    left : string, optional
        ['train','test'] depending on which roc graph should be plotted on the left.
    right : None or string, optional
        None to not plot a second roc graph. ['train','test'] to plot the respective roc graph on the right.
    normalize : boolean, optional
        plot normalized or absolute count confusion matrix. 
    aggregate : string, optional
        use sum to summ all confusion matrices, or mean to compute the mean. 
    cmap : string or None, optional
        the gradient of the values displayed from matplotlib.pyplot.cm
       see http://matplotlib.org/examples/color/colormaps_reference.html
       plt.get_cmap('jet') or plt.cm.Blues
    hsize : positive integer, optional
        horizontal size of the plot
    vsize : positive integer, optional
        vertical size of the plot
    
    Returns
    -------
    
    '''

    # Initialize plot axes.
    f, axes = plt.subplots(nrows=1, ncols=2, figsize=(hsize, vsize))
    
    # Plot LEFT        
    cnf_list = [p[left][confmatrix_name] for p in performance_list]
    if aggregate == 'sum':
        confmatrix = sum(cnf_list)
    elif aggregate == 'mean':
        confmatrix = sum(cnf_list)/len(cnf_list)
    else:
        print('ERROR: aggregate must be either sum or mean')
        sys.exit(-1)
    ax = axes.flatten()[0]
    tit = 'Confusion matrix ({})'.format(left) 
    __prepare_confmatrix_plot(cm=confmatrix, target_names=classes, title=tit, 
                              normalize=normalize, cmap=cmap, ax=ax)

    # Plot RIGHT        
    cnf_list = [p[right][confmatrix_name] for p in performance_list]  
    if aggregate == 'sum':
        confmatrix = sum(cnf_list)
    elif aggregate == 'mean':
        confmatrix = sum(cnf_list)/len(cnf_list)
    else:
        print('ERROR: aggregate must be either sum or mean')
        sys.exit(-1)
    ax = axes.flatten()[1]
    tit = 'Confusion matrix ({})'.format(right) 
    __prepare_confmatrix_plot(cm=confmatrix, target_names=classes, title=tit, 
                              normalize=normalize, cmap=cmap, ax=ax)

    plt.tight_layout()
    plt.show()
       

def __prepare_roc_plot(fprs, tprs, title, ax):
    
    # Plot ROC for each K-Fold + compute AUC scores.
    mean_fpr = np.linspace(0, 1, 100)
    tprs_interp = []
    aucs = []
    for i, (fpr, tpr) in enumerate(zip(fprs, tprs)):
        tprs_interp.append(np.interp(mean_fpr, fpr, tpr))
        tprs_interp[-1][0] = 0.0
        roc_auc = metrics.auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        
    # Plot the baseline.
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Baseline', alpha=.8)
    
    # Plot the mean ROC.
    mean_tpr = np.mean(tprs_interp, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b', lw=2, alpha=.8,
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc))
    
    # Plot the standard deviation around the mean ROC.
    std_tpr = np.std(tprs_interp, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    
    # Fine tune and show the plot.
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend()#loc="lower right")
    ax.grid(True)
    plt.rc('grid', linestyle='--', color='grey', alpha=0.5)
    

def plot_roc_curve(performance_list, left='train', right='test', 
                   fp_name='fpr-curve', tp_name='tpr-curve', hsize=18, vsize=6):
    '''Plot the Receiver Operating Characteristic from a list 
    of true positive rates and false positive rates.
    
    Parameters
    ----------
    performance_list : list
        list of performances (of different xval iterations) containing true and false positive curves.     
        a true or false positive rate curve is an array as returned by sklearn.metrics.roc_curve.
    left : string, optional
        ['train','test'] depending on which roc graph should be plotted on the left.
    right : None or string, optional
        None to not plot a second roc graph. ['train','test'] to plot the respective roc graph on the right.
    fp_name : string, optional
        the dictionary key within performance_list of the false positive rate curve 
    tp_name : string, optional
        the dictionary key within performance_list of the true positive rate curve 
    hsize : positive integer, optional
        horizontal size of the plot
    vsize : positive integer, optional
        vertical size of the plot
    
    Returns
    -------
        
    '''
    
    # Initialize plot axes.
    f, axes = plt.subplots(nrows=1, ncols=2, figsize=(hsize,vsize))

    # Plot LEFT        
    fprs, tprs = [], [] 
    for f in range(len(performance_list)):
        fprs.append(performance_list[f][left][fp_name])
        tprs.append(performance_list[f][left][tp_name])
    ax = axes.flatten()[0]
    tit = 'Receiver operating characteristic ({})'.format(left) 
    __prepare_roc_plot(fprs=fprs, tprs=tprs, title=tit, ax=ax)
        
    # Plot RIGHT
    if right != None:         
        fprs, tprs = [], [] 
        for f in range(len(performance_list)):
            fprs.append(performance_list[f][right][fp_name])
            tprs.append(performance_list[f][right][tp_name])
        ax = axes.flatten()[1]
        tit = 'Receiver operating characteristic ({})'.format(right) 
        __prepare_roc_plot(fprs=fprs, tprs=tprs, title=tit, ax=ax)
    
    plt.show()
    
    
def evaluate_binclass(clf, X_train, X_test, y_train, y_test, pos_label=1, clf_threshold=0.5):
    '''Evaluate a binary classifier performance on train and test sets. 
    
    Parameters
    ----------
    clf : Classifier object
        classifier instance (e.g. sklearn.ensemble.RandomForestClassifier() return) 
    X_train : dataframe
        dataframe of features used to train the classifier
    y_train : dataframe
        dataframe of label to predict used to train the classifier
    X_test : dataframe
        dataframe of features to be used for out of sample testing
    y_test : dataframe
        dataframe of label to be used for out of sample testing
    pos_label : integer, optional
        the positive class. [0,1] for binary classification
    clf_threshold : float, optional
        threshold [0,1] to be used to predict either class depending on probability 
    
    Returns
    -------
    classes
    predictions (train + test) of the classifier
    performance (train + test) of the classifier  
        
    '''
    
    # labels
    classes = list(set(y_train))        

    # prepare datasets
    X, y = {}, {}
    X['train'], y['train'] = X_train, y_train
    X['test'], y['test'] = X_test, y_test
    
    # create 1 column per class on y
    for d in ['train','test']: 
        y[d] = pd.DataFrame(1-y[d]).merge(pd.DataFrame(y[d]), left_index=True, right_index=True)
        y[d].columns = [0,1]
    
    # predict: proba and class 
    yhat = {'train':{},'test':{}}
    for d in ['train','test']: 
        yhat[d]['proba'] = clf.predict_proba(X[d])
        yhat[d]['class'] = (yhat[d]['proba'][:,pos_label] > clf_threshold).astype(int)
        yhat[d]['class'] = pd.DataFrame(1-yhat[d]['class']).merge(pd.DataFrame(yhat[d]['class']), 
            left_index=True, right_index=True)
        yhat[d]['class'].columns = [0,1]
    
    # evaluate
    perf = {'train':{},'test':{}}
    for d in ['train','test']: 
    
        # confusion matrix
        perf[d]['confusion-matrix'] = metrics.confusion_matrix(y_true=y[d][pos_label], 
            y_pred=yhat[d]['class'][pos_label], labels=classes)
        
        # accuracy
        perf[d]['accuracy'] = metrics.accuracy_score(y_true=y[d], y_pred=yhat[d]['class'])
        
        # log loss
        perf[d]['log-loss'] = metrics.log_loss(y_true=y[d], y_pred=yhat[d]['proba'])

        # cohen kappa
        perf[d]['cohen-kappa'] = metrics.cohen_kappa_score(y1=y[d][pos_label], 
            y2=yhat[d]['class'][pos_label])

        # micro precision
        perf[d]['precision.micro'] = metrics.precision_score(y_true=y[d], 
            y_pred=yhat[d]['class'], pos_label=pos_label, average='micro')
        
        # micro recall
        perf[d]['recall.micro'] = metrics.recall_score(y_true=y[d], 
            y_pred=yhat[d]['class'], pos_label=pos_label, average='micro')
                
        # micro f1
        perf[d]['f1-score.micro'] = metrics.f1_score(y_true=y[d], 
            y_pred=yhat[d]['class'], pos_label=pos_label, average='micro')
                
        # micro ROC 
        fpr, tpr, thr = metrics.roc_curve(y_true=y[d].values.ravel(), 
            y_score=yhat[d]['proba'].ravel(),pos_label=pos_label)
        perf[d]['fpr-curve.micro'] = fpr
        perf[d]['tpr-curve.micro'] = tpr
        perf[d]['fpr-tpr-thr-curve.micro'] = thr

        # micro AUC
        perf[d]['auc.micro'] = metrics.auc(x=perf[d]['fpr-curve.micro'], 
            y=perf[d]['tpr-curve.micro'])

        # precision and recall curve
        prc, rec, thr = metrics.precision_recall_curve(y_true=y[d].values.ravel(), 
            probas_pred=yhat[d]['proba'].ravel(), pos_label=pos_label)
        perf[d]['precision-curve.micro'] = prc
        perf[d]['recall-curve.micro'] = rec
        perf[d]['prc-rec-thr-curve.micro'] = thr

        # average precision recall
        perf[d]['avg-prc-rec.micro'] = metrics.average_precision_score(y_true=y[d], 
            y_score=yhat[d]['proba'], average='micro')

        # compute metrics for each class 
        for c in classes:
    
            # precision
            perf[d]['precision.{}'.format(c)] = metrics.precision_score(y_true=y[d][c], 
                y_pred=yhat[d]['class'][c], pos_label=1, average='binary')
             
            # recall
            perf[d]['recall.{}'.format(c)] = metrics.recall_score(y_true=y[d][c], 
                y_pred=yhat[d]['class'][c], pos_label=1, average='binary')

            # f1
            perf[d]['f1-score.{}'.format(c)] = metrics.f1_score(y_true=y[d][c], 
                y_pred=yhat[d]['class'][c], pos_label=1, average='binary')

            # roc
            fpr, tpr, thr = metrics.roc_curve(y_true=y[d][c], 
                y_score=yhat[d]['proba'][:,c], pos_label=1)
            perf[d]['fpr-curve.{}'.format(c)] = fpr 
            perf[d]['tpr-curve.{}'.format(c)] = tpr
            perf[d]['fpr-tpr-thr-curve.{}'.format(c)] = thr
        
            # auc
            perf[d]['auc.{}'.format(c)] = metrics.auc(x=perf[d]['fpr-curve.{}'.format(c)], 
                y=perf[d]['tpr-curve.{}'.format(c)])

            # precision and recall curve
            prc, rec, thr = metrics.precision_recall_curve(y_true=y[d][c], 
                probas_pred=yhat[d]['proba'][:,c], pos_label=1)
            perf[d]['precision-curve.{}'.format(c)] = prc
            perf[d]['recall-curve.{}'.format(c)] = rec
            perf[d]['prc-rec-thr-curve.{}'.format(c)] = thr
                        
            # average precision recall
            perf[d]['avg-prc-rec.{}'.format(c)] = metrics.average_precision_score(y_true=y[d][c], 
                y_score=yhat[d]['proba'][:,c], average=None)
    
    # beautify yhat with dataframe predictions
    yhat['train']['preds'] = yhat['train']['class'].merge(pd.DataFrame(data=yhat['train']['proba']), 
        left_index=True, right_index=True, suffixes=['_class','_proba'])
    yhat['test']['preds']  = yhat['test']['class'].merge(pd.DataFrame(data=yhat['test']['proba']), 
        left_index=True, right_index=True, suffixes=['_class','_proba'])
    yhat['train']['preds'].index = X_train.index
    yhat['test']['preds'].index  = X_test.index
    yhat['train']['preds'].index.name = X_train.index.name
    yhat['test']['preds'].index.name  = X_test.index.name

    '''    
    yhat['train']['proba'] = pd.DataFrame(data=yhat['train']['proba'])
    yhat['test']['proba']  = pd.DataFrame(data=yhat['test']['proba'])    
    yhat['train']['class'].index = X_train.index
    yhat['train']['proba'].index = X_train.index
    yhat['test']['class'].index  = X_test.index
    yhat['test']['proba'].index  = X_test.index
    yhat['train']['class'].index.name = X_train.index.name
    yhat['train']['proba'].index.name = X_train.index.name
    yhat['test']['class'].index.name  = X_test.index.name
    yhat['test']['proba'].index.name  = X_test.index.name
    '''
    
    return (classes, yhat, perf)
    
    
def xval_performances(performance_list, dsets=['train','test'], round_dec=3):
    '''Computes cross validation performance averages and std.
    
    Parameters
    ----------
    performance_list : list
        list of performances (of different xval iterations) containing true and false positive curves.     
        a true or false positive rate curve is an array as returned by sklearn.metrics.roc_curve.
    dsets : list, optional
        the type of dataset to compute metrics on. at least one of train or test must be included. 
    round_dec : integer, optional
        number of decimals to keep in the final dataframe
    
    Returns
    -------
    dataframe with all metrics as columns and train/test average and stad performance values on rows. 
    
    '''
    
    # find metrics
    perf_metrics = []
    for k in list(performance_list[0][dsets[0]].keys()):
        if isinstance(performance_list[0][dsets[0]][k],float):
            perf_metrics.append(k)      
    
    # initialize dataframe
    dfindex =[]
    for dset in dsets:
        dfindex.append('{}.avg'.format(dset))
        dfindex.append('{}.std'.format(dset))
    perf_df = pd.DataFrame(index=dfindex, columns=perf_metrics)

    # compute avg and std    
    for pmetric in perf_metrics:
        pm = []
        for dset in dsets:
            p = [p[dset][pmetric] for p in performance_list]
            pm.append(round(np.nanmean(p),round_dec))
            pm.append(round(np.nanstd(p),round_dec))
        perf_df.loc[:,pmetric] = pm 
    
    return perf_df        

        
def xval_classify(clf, cv, X, y, pos_label=1, cv_folds=10, clf_threshold=0.5, 
                  verbose=True, metrics_to_print=['log-loss']):        
    '''Runs cross validation on the input classifier and returns performances.
    
    Parameters
    ----------
    clf : Classifier object
        classifier instance (e.g. sklearn.ensemble.RandomForestClassifier() return) 
    cv : Cross validation object
        cross validation object from sklearn.model_selection 
    X : dataframe
        dataframe of features used to train the classifier
    y : dataframe
        dataframe of label to predict used to train the classifier
    pos_label : integer
        the positive class. [0,1] for binary classification
    X_test : dataframe
        dataframe of features to be used for out of sample testing
    cv_folds : positive integer, optional
        number of folds to use for the validation 
    clf_threshold : float, optional
        threshold [0,1] to be used to predict either class depending on probability 
    verbose : boolean, optional
        print cross validation progress
    metrics_to_print : list, optional
        performance metrics to print during x-val.
        can be any of the metrics returned by evaluate_binclass.
        
    Returns
    -------
    classes
    list of predicitons  (one for each validation iteration)
    list of performances (one for each validation iteration)  
    list of classes      (one for each validation iteration)
    
    '''

    if verbose == True:
        print('\n--- X-Val started @ {} ---\n'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        print('PREDICTORS shape =\t', X.shape)
        print('TARGET LABEL shape =\t', y.shape)

    cv = cv.split(X=X, y=y)    
    classes_list, yhat_list, perf_list = [], [], []    
    for (train, test), i in zip(cv, range(cv_folds)):       
        
        X_train, X_test = X.iloc[train], X.iloc[test] 
        y_train, y_test = y.iloc[train], y.iloc[test] 
                    
        clf.fit(X_train, y_train)
        classes, yhat, perf = evaluate_binclass(clf=clf, clf_threshold=clf_threshold, pos_label=pos_label,
                                       X_train=X_train, X_test=X_test, 
                                       y_train=y_train, y_test=y_test)
        
        yhat_list.append(yhat)
        perf_list.append(perf)
        classes_list.append(classes)
        
        if verbose == True:
            print('\n> Iteration {}:\t\t train:\t\t| samples = {}\t| pos_lables = {}\t| pos_rate = {:.4f}'.format(
                  i, len(y_train), y_train.sum(), round(y_train.sum()/len(y_train),4))) 
            print('\t\t\t test:\t\t| samples = {}\t| pos_lables = {}\t| pos_rate = {:.4f}'.format(
                  len(y_test), y_test.sum(), round(y_test.sum()/len(y_test),4))) 
            for metric in metrics_to_print:
                change = (perf['test'][metric]-perf['train'][metric])/perf['train'][metric]
                if len(metric) > 7:
                    print('\t\t\t {}:\t| train = {:.4f}\t| test = {:.4f} ({:.2f}%)'.format(metric, 
                          round(perf['train'][metric],4), round(perf['test'][metric],4), round(change*100,4)))
                else:
                    print('\t\t\t {}:\t\t| train = {:.4f}\t| test = {:.4f} ({:.2f}%)'.format(metric, 
                          round(perf['train'][metric],4), round(perf['test'][metric],4), round(change*100,4)))
        
    if verbose == True:
        print('\n\n--- X-Val completed @ {} ---\n'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    return (classes_list, yhat_list, perf_list)
        

def __prepare_precision_recall_plot(prcs, recs, aprs, title, ax):
    
    # Plot the base line.
    ax.plot([0, 1], [1, 0], linestyle='--', lw=2, color='r', label='Baseline', alpha=.5)

    # Plot the precision recall curves for each K-Fold.
    mean_rec = np.linspace(0, 1, 100)
    prcs_interp = []
    for i, (rec, prc) in enumerate(zip(recs, prcs)):
        ax.plot(rec, prc, lw=1, alpha=0.5, label='PR fold %d (APR = %0.2f)' % (i, aprs[i]))
        prcs_interp.append(np.interp(x=mean_rec, xp=rec[::-1], fp=prc[::-1]))
        
    # Plot the mean curve.
    mean_prc = np.mean(prcs_interp, axis=0)
    mean_apr = np.nanmean(aprs)
    std_apr = np.nanstd(aprs)
    ax.plot(mean_rec, mean_prc, color='b', lw=2, alpha=.8,
            label='Mean PR (APR = %0.2f $\pm$ %0.2f)' % (mean_apr, std_apr))
    
    # Plot the standard deviation.
    std_prc = np.nanstd(prcs_interp, axis=0)
    prcs_upper = np.minimum(mean_prc + std_prc, 1)
    prcs_lower = np.maximum(mean_prc - std_prc, 0)
    ax.fill_between(x=mean_rec, y1=prcs_upper, y2=prcs_lower, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    
    # Fine tune
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim([-0.05,1.05])
    ax.set_ylim([-0.05,1.05])
    ax.set_title(title)
    ax.legend()#loc='lower left')
    ax.grid(True)
    plt.rc('grid', linestyle='--', color='grey', alpha=0.5)
    return ax    
    

def plot_precision_recall_curve(performance_list, left='train', right='test', 
                                prc_name='precision-curve', rec_name='recall-curve', apr_name='avg-prc-rec',
                                hsize=18, vsize=6):
    '''Plot Precision-Recall curves from the list of cross valideated performances.
    
    Parameters
    ----------
    performance_list : list
        list of performances (of different xval iterations) containing precision and recall curves,
        as returned by sklearn.metrics.precision_recall_curve.
    left : string, optional
        ['train','test'] depending on which roc graph should be plotted on the left.
    right : None or string, optional
        None to not plot a second roc graph. ['train','test'] to plot the respective roc graph on the right.
    prc_name : string, optional
        the dictionary key within performance_list of the precision curve 
    rec_name : string, optional
        the dictionary key within performance_list of the recall curve 
    apr_name : string, optional
        the dictionary key within performance_list of average precision recall 
    hsize : positive integer, optional
        horizontal size of the plot
    vsize : positive integer, optional
        vertical size of the plot
    
    Returns
    -------
        
    '''
    
    # Initialize plot axes.
    f, axes = plt.subplots(nrows=1, ncols=2, figsize=(hsize,vsize))

    # Plot LEFT        
    prcs, recs, aprs = [], [], [] 
    for f in range(len(performance_list)):
        prcs.append(performance_list[f][left][prc_name])
        recs.append(performance_list[f][left][rec_name])
        aprs.append(performance_list[f][left][apr_name])
    ax = axes.flatten()[0]
    tit = 'Precision vs Recall Curves ({})'.format(left) 
    __prepare_precision_recall_plot(prcs=prcs, recs=recs, aprs=aprs, title=tit, ax=ax)
        
    # Plot RIGHT
    if right != None:         
        prcs, recs, aprs = [], [], [] 
        for f in range(len(performance_list)):
            prcs.append(performance_list[f][right][prc_name])
            recs.append(performance_list[f][right][rec_name])
            aprs.append(performance_list[f][right][apr_name])
        ax = axes.flatten()[1]
        tit = 'Precision vs Recall Curves ({})'.format(right) 
    __prepare_precision_recall_plot(prcs=prcs, recs=recs, aprs=aprs, title=tit, ax=ax)
    
    plt.show()


def plot_features_importance(clf, features_index, no_topfeatures=None, desired_importance=0.95,
                             hsize=12, vsize=None):
    '''Compute ad plot feature importance of a random forest classifier.
    
    Parameters
    ----------
    clf : RandomForestClassifier
        sklearn random forest classifier 
    features_index : index
        ordered index of feature names, as passed to RandomForestClassifier for fitting
    no_topfeatures : integer or None, optional
        number of top important features to plot. 
        None will automatically set it to the number of features needed to reach desired_importance.
    desired_importance : float [0,1], optional
        cumulative importance desired    
    hsize : positive integer, optional
        horizontal size of the plot
    vsize : positive integer or None, optional
        vertical size of the plot. if None will be set to no_topfeatures/2
    
    Returns
    -------
        
    '''    
    
    feature_imp = pd.Series(clf.feature_importances_, index=features_index)
    feature_imp = feature_imp.sort_values(ascending=False) #.head(no_topfeatures)
    sorted_importances = list(feature_imp.values)
    sorted_features = list(feature_imp.index)
    cumulative_importances = np.cumsum(sorted_importances)
    no_important_features = len(cumulative_importances[cumulative_importances <= desired_importance])+1
    
    if no_important_features < len(features_index):
        print('\n{} importance reached with the {} most important features'.format(desired_importance*100,
              no_important_features))
    else:
        print('\nAll {} features needed to reach {} importance'.format(len(features_index),
              cumulative_importances[-1]))

    if type(no_topfeatures) == type(None):
        no_topfeatures = no_important_features

    if vsize == None:
        vsize = no_topfeatures/2
        vsize_cum = 8

    # Plot features importance
    f, ax = plt.subplots(nrows=1, ncols=1, figsize=(hsize, vsize))
    
    sns.barplot(x=feature_imp[:no_topfeatures], 
                y=feature_imp.index[:no_topfeatures])
    
    ax.set_xlabel('Feature Importance Score')
    ax.set_ylabel('Features')
    ax.set_title('Visualizing Important Features')
    ax.grid(True)
    plt.rc('grid', linestyle='--', color='grey', alpha=0.5)
    #plt.tight_layout()
    plt.show()
    
    # Plot cumulative importance
    f, ax = plt.subplots(nrows=1, ncols=1, figsize=(hsize, vsize_cum))
    x_values = list(range(len(sorted_importances[:no_topfeatures])))
    
    ax.hlines(y=desired_importance, xmin=0, xmax=len(sorted_importances[:no_topfeatures]), 
              color='red', linestyles='dashed', alpha=0.8, linewidth=3, 
              label='{}% cumulative importance'.format(desired_importance*100))
    
    ax.plot(x_values, cumulative_importances[:no_topfeatures], 
            color='green', marker='o', linestyle='-', linewidth=3, markersize=12, 
            label='Cumulative importance')
    
    ax.set_xticklabels(labels=sorted_features[:no_topfeatures]) 
    ax.set_xlabel('Variable')
    plt.xticks(x_values, sorted_features[:no_topfeatures], rotation=90)
    #for tick in ax.get_xticklabels():
    #    tick.set_rotation(9 0)
    ax.set_ylabel('Cumulative Importance')
    ax.set_ylim(bottom=-0.05, top=1.05)
    ax.set_title('Visualizing Important Features')
    ax.grid(True)
    ax.legend(loc='lower right')
    plt.rc('grid', linestyle='--', color='grey', alpha=0.5)
    #plt.tight_layout()
    plt.show()
    
    imp_df = pd.DataFrame(feature_imp[:no_topfeatures], columns=['Single importance'])
    cum_df = pd.DataFrame(cumulative_importances[:no_topfeatures], index=imp_df.index, 
                          columns=['Cumulative importance'])
    imp_df['Cumulative importance'] = cum_df
    imp_df.index.name = 'Features'

    return imp_df


def preds_table(yhat_list, y):
    '''Compute prediction table from x-val predictions.
    
    Parameters
    ----------
    yhat_list : list
        list of predictions as returned by xval_classify 
    y : pandas series
        actual labels (train and test if yhat contains train and test)
        
    Returns
    -------
    Dictionary of two (train and test) dataframes. 
    Each of them contains: label series and predicitons (classes and probabilities).     
    
    '''    

    preds = {}
    for d in yhat_list[0].keys():
        preds[d] = yhat_list[0][d]['preds']
        for i in range(1,len(yhat_list)):
            preds[d] = preds[d].append(yhat_list[i][d]['preds'])
        preds[d] = (pd.DataFrame(y)).merge(preds[d], how='inner', left_index=True, right_index=True)

    return preds


def init_bins(preds_df_dict, no_bins='10', bins_by='proba'):
    '''Predictions sensitivity analysis: binned predicted probabilities are compared to the label.
    
    Parameters
    ----------
    preds_df_dict : dictionary
        Dictionary of two (train and test) dataframes. 
        Each of them contains: label series and predicitons (classes and probabilities).     
    no_bins : integer, optional
        number of bins to use to discretize the positive class probabilities
    bins_by : string, optional
        split bins by same probability interval width (proba) or by same number of samples per bin (samples)
        
    Returns
    -------
    Probability bins    

    '''    
    
    if bins_by == 'proba':
        step = 1/no_bins
        bins = [(round(i*step,4),round((i+1)*step,4)) for i in range(no_bins)]
        
    elif bins_by == 'samples':
        df = preds_df_dict['train'].sort_values('1_proba', ascending=True)
        max_end = len(df)
        step = math.ceil(max_end/no_bins)
        bins = []
        for i in range(no_bins):
            start_idx = int(i*step)
            end_idx = int(min(max_end-1, (i+1)*step))
            boundaries = (df.iloc[start_idx]['1_proba'], df.iloc[end_idx]['1_proba'])
            bins.append(boundaries)
        bins[0] = (0.00, bins[0][1])
        bins[-1] = (bins[-1][0], 1.00)
        
    else:
        print('ERROR: bins_by not recognizd')
        sys.exit(-1)
    
    return bins

def preds_sensitivity_table(yhat_list, y, label_column, pos_proba_column='1_proba', 
                            pos_class_column='1_class', neg_class_column='0_class', no_bins=10, bins_by='proba'):
    '''Predictions sensitivity analysis: binned predicted probabilities are compared to the label.
    
    Parameters
    ----------
    yhat_list : list
        list of predictions as returned by xval_classify 
    y : pandas series
        actual labels
    label_column : string
        column name of the column containing the label
    pos_proba_column : string, optional
        column name of the column containing the predicted probability of positive class
    pos_class_column : string, optional
        column name of the column containing the predicted positive class
    neg_class_column : string, optional
        column name of the column containing the predicted negative class
    no_bins : integer, optional
        number of bins to use to discretize the positive class probabilities
    bins_by : string, optional
        split bins by same probability interval width (proba) or by same number of samples per bin (samples)
        
    Returns
    -------
    2 items:
    1. Dictionary of two (train and test) dataframes (predictions). 
    Each of them contains: label series and predicitons (classes and probabilities).     
    2. Dictionary of two (train and test) dataframes (binned summary). 
    Each of them contains: no_preds, 0_class_preds, 1_class_preds, 0_NB_DEF_IND, 1_NB_DEF_IND, RATIO_NB_DEF_IND.   
    
    '''    

    # build full predictions table
    preds_df_dict = preds_table(yhat_list=yhat_list, y=y)
    
    # init bins
    bins = init_bins(preds_df_dict=preds_df_dict, no_bins=no_bins, bins_by=bins_by)
    
    # init results dictionary and table
    cols = ['no_preds', 'prc_preds',
            neg_class_column+'_preds', pos_class_column+'_preds',
            '0_'+label_column, '1_'+label_column, 'RATIO_'+label_column]
    
    preds_binned_dict = {}
    for d in preds_df_dict.keys():
        preds_binned = pd.DataFrame(index=bins, columns=cols)
        tot = len(preds_df_dict[d])
        for i in range(len(bins)):
            
            low, high = bins[i][0], bins[i][1]
            if i == 0: # first bin: both boundaries inclusive 
                pr = preds_df_dict[d][((preds_df_dict[d][pos_proba_column]>=low) & 
                                       (preds_df_dict[d][pos_proba_column]<=high))]
            else: # other bins: lower bound not inclusive
                pr = preds_df_dict[d][((preds_df_dict[d][pos_proba_column]>low) & 
                                       (preds_df_dict[d][pos_proba_column]<=high))]

            no_samples = len(pr)
            try: 
                prc_samples = no_samples/tot
            except: 
                prc_samples = np.nan
            no_preds_neg = len(pr[pr[neg_class_column]==1])
            no_preds_pos = len(pr[pr[pos_class_column]==1])
            no_actual_neg = len(pr[pr[label_column]==0])
            no_actual_pos = len(pr[pr[label_column]==1])
            
            try:
                ratio = no_actual_pos / (no_actual_pos+no_actual_neg)
            except:
                ratio = np.nan
                
            preds_binned.iloc[i,:] = [no_samples, prc_samples,
                                      no_preds_neg, no_preds_pos,
                                      no_actual_neg, no_actual_pos, ratio]
                                      
        preds_binned_dict[d] = preds_binned
        preds_binned_dict[d].index.name = 'yhat_proba'
    
    return preds_df_dict, preds_binned_dict


def plot_binned_preds_single(preds_binned_dict, column, label, scale_type, hsize=18, vsize=5):
            
         # data
        datasets = list(preds_binned_dict.keys())
        f, axes = plt.subplots(nrows=1, ncols=len(datasets), figsize=(hsize, vsize))

        # train and test dataframes
        for i in range(len(datasets)):
            d, ax = datasets[i], axes[i]            
            df = deepcopy(preds_binned_dict[d]).sort_index(ascending=False)
            df.loc[df['0_class_preds']>df['1_class_preds'],'predicted_class'] = 'PRED CLASS = 0'
            df.loc[df['0_class_preds']<df['1_class_preds'],'predicted_class'] = 'PRED CLASS = 1'            
            yval = df[column]
            
            # scale
            if scale_type == 'abs':
                pass
            elif scale_type == 'log':
                yval.replace(to_replace=0, value=np.nan, inplace=True)
                yval = np.log(yval)
            elif scale_type == 'prc':
                yval = yval/np.nansum(yval)    
            else:
                print('ERROR: scale not recognized')
                sys.exit(-1)
                
            # y axis scale upper bound
            try:
                max0 = np.nanmax(preds_binned_dict[datasets[0]][column])
                max1 = np.nanmax(preds_binned_dict[datasets[1]][column])
            except:
                max0, max1 = 1.3, 1.3
            if scale_type == 'abs':
                ymax = max(max0,max1)*1.30
            elif scale_type == 'log':
                ymax = np.log(max(max0,max1))*1.30
            elif scale_type == 'prc':
                ymax = 1.30
            else:
                print('ERROR: scale not recognized')
                sys.exit(-1)

            # compute baselines
            if column.startswith('RATIO') == True:
                try:
                    baseline = np.nansum(df['1_{}'.format(label)])/np.nansum(df['no_preds'])
                except:
                    baseline = np.nan
                baseline_title = 'AVG POS/TOT RATE = {}'.format(round(baseline,4))
            else:
                try:
                    baseline = np.nansum(yval)/len(yval)
                except:
                    baseline = np.nan
                baseline_title = 'AVG NO_SAMPLES/BUCKET = {}'.format(round(baseline,4))
            
            # plot
            try:
                sns.barplot(x=df.index, y=yval, hue='predicted_class', data=df, ax=ax,
                            palette=sns.color_palette("RdBu", n_colors=5, desat=1))  
                ax.hlines(y=baseline, xmin=0, xmax=len(df.index), label=baseline_title,
                          color='black', linestyles='dashed', alpha=0.8, linewidth=2)
                ax.grid(True)
                plt.rc('grid', linestyle='--', color='grey', alpha=0.5)
                ax.set_title('{} - {} ({} scale)'.format(column, d, scale_type))
                ax.legend(loc='upper right')
                ax.set_ylim(top=ymax)
                ax.set_ylabel(column)
                ax.set_xlabel('Predicted probability ({} bins)'.format(len(df)))
                for tick in ax.get_xticklabels():
                    tick.set_rotation(90)
            except:
                pass
                
        #plt.tight_layout()
        plt.show()
    

def plot_binned_preds(preds_binned_dict, label, columns_list, percscale=[], logscale=[],   
                      hsize=18, vsize=5):
    '''Barplot of provided column vs binned predicted probabilities.
    
    Parameters
    ----------
    preds_binned_dict : dictionary
        dictionary (train and test) of datasets as returned by preds_sensitivity_table.
        predictions and lables for each binned predicted probability.
    label : string
        name on the label column
    columns_list : list
        list of column names of the columns to plot
    percscale : list, optional
        print percentage scaled values of included columns
    logscale : list, optional
        print log scaled values of included columns
    hsize : integer, optional
        horizontal size of the plot
    vsize : integer, optional
        vertical size of the plot
    
    Returns
    -------
    
    '''    
    
    sns.set_style('whitegrid')
    for column in columns_list:
        print('\n>>> {} sentitivity analysis \n'.format(column))
        
        plot_binned_preds_single(preds_binned_dict=preds_binned_dict, column=column, label=label, 
                                 scale_type='abs', hsize=hsize, vsize=vsize)
        
        if column in percscale:
            plot_binned_preds_single(preds_binned_dict=preds_binned_dict, column=column, label=label,
                                     scale_type='prc', hsize=hsize, vsize=vsize)
            
        if column in logscale:
            plot_binned_preds_single(preds_binned_dict=preds_binned_dict, column=column, label=label,
                                     scale_type='log', hsize=hsize, vsize=vsize)
            

def binned_preds_analysis(yhat_list, y, label, no_bins=10, bins_by='proba', hsize=18, vsize=5):
    '''Plot full binned predictions analysis, after building all it's needed.
    
    Parameters
    ----------
    yhat_list : list
        list of predictions as returned by xval_classify 
    y : pandas series
        actual labels
    label : string
        name on the label column
    no_bins : integer, optional
        number of bins to use to discretize the positive class probabilities
    bins_by : string, optional
        split bins by same probability interval width (proba) or by same number of samples per bin (samples)
    hsize : integer, optional
        horizontal size of the plot
    vsize : integer, optional
        vertical size of the plot
    
    Returns
    -------
    2 items:
    1. Dictionary of two (train and test) dataframes (predictions). 
    Each of them contains: label series and predicitons (classes and probabilities).     
    2. Dictionary of two (train and test) dataframes (binned summary). 
    Each of them contains: no_preds, 0_class_preds, 1_class_preds, 0_LABEL, 1_LABEL, RATIO_LABEL.   
    
    '''    
    
    # build binned prediction summary table
    preds_df_dict, preds_binned_dict = preds_sensitivity_table(yhat_list=yhat_list, y=y, label_column=y.name, 
                                                pos_proba_column='1_proba', pos_class_column='1_class', 
                                                neg_class_column='0_class', no_bins=no_bins, bins_by=bins_by)
        
    # plot analysis
    plot_binned_preds(preds_binned_dict=preds_binned_dict, label=label, 
                      columns_list=['RATIO_{}'.format(label),'no_preds','1_{}'.format(label),'0_{}'.format(label)], 
                      percscale=['no_preds','1_{}'.format(label),'0_{}'.format(label)], 
                      logscale=['no_preds','1_{}'.format(label),'0_{}'.format(label)], 
                      hsize=18, vsize=5)
    
    return preds_df_dict, preds_binned_dict


