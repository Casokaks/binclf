# -*- coding: utf-8 -*-
'''
Utils library for XGBoost optimization 
Created on Nov 19th 2018
@author: Andrea Casati
'''


import math 
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import xgboost as xgb
from xgboost.sklearn import XGBClassifier


def xgb_opt_no_estimators(alg, dtrain, target, predictors, eval_metric='logloss', 
                          num_boost_round=1000, early_stopping_rounds=50,
                          cv_folds=5, cv_stratified=False, cv_shuffle=True,
                          seed=8888, verbose=None):
    '''Find the optimal number of weak estimators (trees)
    
    Parameters
    ----------
    
    Returns
    -------
    
    '''
    
    print('\n--- n_estimators optimization started @ {} ---\n'.format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    
    xgb_param = alg.get_xgb_params()
    xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
    
    if cv_stratified == True:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=cv_shuffle, random_state=seed)
        print('StratifiedKFold cross-validation instantiated...')
    else:
        cv = KFold(n_splits=cv_folds, shuffle=cv_shuffle, random_state=seed)
        print('KFold cross-validation instantiated...')
        
    print('Search started...\n')
    cvresult = xgb.cv(params=xgb_param, 
                      dtrain=xgtrain, 
                      metrics=eval_metric,
                      num_boost_round=num_boost_round, 
                      early_stopping_rounds=early_stopping_rounds,
                      folds = cv,
                      nfold=cv_folds,
                      stratified=cv_stratified,
                      shuffle=cv_shuffle,
                      seed=seed, 
                      verbose_eval=verbose)
    
    no_estimators = cvresult.shape[0]
    score = cvresult.iloc[-1]
    print('\nOptimal number of estimators = {}'.format(no_estimators))
    print('Train {} = {:.4f} +/- {:.4f}'.format(eval_metric, score[2], score[3]))
    print('Test {} = {:.4f} +/- {:.4f}'.format(eval_metric, score[0],score[1]))
    print('\n--- n_estimators optimization completed @ {} ---\n'.format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    
    return no_estimators, score
    

def print_opt_params_report(results, n_top=3):
    '''Print top best results and parameters.
    
    Parameters
    ----------
    
    Returns
    -------
    
    '''

    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        print('')
        for candidate in candidates:
            print('Model with rank: {0}'.format(i))
            print('Mean validation score: {0:.4f} (std: {1:.4f})'.format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print('Parameters: {0}'.format(results['params'][candidate]))
            print('')
            
            
def _update_params(params, results, keep_top, params_dict):
    '''Re-Initialize parameters for search, based on given parameters dictionary and last results to keep.
    
    Parameters
    ----------
    
    Returns
    -------
    
    '''    
    
    # update min and max based on last results to keep
    for k in params.keys():
        param_temp = np.array([])
        for n in range(1,keep_top+1):
            idx = np.flatnonzero(results['rank_test_score'] == n)[0]
            param_temp = np.append(param_temp, results['params'][idx][k])
        params_dict[k]['min'] = min(param_temp)
        params_dict[k]['max'] = max(param_temp) 

    # re-initialize params
    return _init_params(params_dict)
    

def _init_params(params_dict):
    '''Initialize parameters for search, based on given parameters dictionary.
    
    Parameters
    ----------
    
    Returns
    -------
    
    '''    

    params = {}
    for k in params_dict.keys():
        
        # addictive range
        if params_dict[k]['type'] == 'add': 
        
            params[k] = np.arange(params_dict[k]['min'], params_dict[k]['max'], params_dict[k]['step'])
        
        # multiplication range
        elif params_dict[k]['type'] == 'multi':
            
            params[k] = np.array([])
            next_value = params_dict[k]['min']
            while next_value <= params_dict[k]['max']:
                params[k] = np.append(params[k], next_value)
                next_value = next_value * params_dict[k]['step']
            
        # error
        else:
            print('ERROR: params type can only be either add or multi')
            sys.exit(-1)
        
        # last check to add range max if not already included
        if params_dict[k]['max'] not in params[k]: 
            params[k] = np.append(params[k], params_dict[k]['max'])
            
        # if range of integers then tranform the array into an integer array
        if sum(params[k] == np.array(params[k], dtype=int)) == len(params[k]):            
            params[k] = np.array(params[k], dtype=int)
                
    return params
            
            
def xgb_opt_params(X, y, estimator, scoring='neg_log_loss', search_type='random',
                   params_dict = {'max_depth':{'init':5,'min':2,'max':10,'step':1,'type':'add'},
                                  'min_child_weight':{'init':1,'min':1,'max':20,'step':1,'type':'add'},
                                  'gamma':{'init':0.00,'min':0.00,'max':1.00,'step':0.01,'type':'add'},
                                  'subsample':{'init':0.80,'min':0.50,'max':1.00,'step':0.10,'type':'add'},
                                  'colsample_bytree':{'init':0.80,'min':0.50,'max':1.00,'step':0.10,'type':'add'},
                                  'reg_alpha':{'init':0,'min':1e-7,'max':1e+1,'step':10,'type':'multi'},
                                  'reg_lambda':{'init':1,'min':1e-4,'max':1e+3,'step':10,'type':'multi'},
                                  'learning_rate':{'init':0.1,'min':1e-4,'max':1e+1,'step':10,'type':'multi'},
                                 }, 
                   n_iter_max=5, keep_top_perc=0.20, n_iter_rnd=15, 
                   cv_folds=5, cv_stratified=False, cv_shuffle=True, iid=False, 
                   n_jobs=1, seed=8888, verbose=True):
    '''Find optimal parameter(s) values exploring the defined search space through random or grid search.
    Basic search algorithms (i.e. sklearn GridSearchCV, RandomizedSearchCV) are re-iterated for n_iter_max times, 
    narrowing the search boundaries according to the last keep_top_perc best solutions.
    
    Parameters
    ----------
    
    Returns
    -------
    
    '''
    
    print('\n--- {} optimization started @ {} ---\n'.format(
            list(params_dict.keys()), datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    if verbose == True:
        verbose = cv_folds
    else:
        verbose = 0

    # init parmeters
    params = _init_params(params_dict)
    search_space = np.prod(np.array([len(list(params[k])) for k in params.keys()]))

    n_iter = 0
    while n_iter < n_iter_max and search_space > 1:
        print('Starting iteration {}, with a search space of {}'.format(n_iter+1, search_space))
        for k in params.keys():
            print('{} = {}'.format(k,params[k]))

        # cross validation setup           
        if cv_stratified == True:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=cv_shuffle, random_state=seed)
            print('StratifiedKFold cross-validation instantiated...')
        else:
            cv = KFold(n_splits=cv_folds, shuffle=cv_shuffle, random_state=seed)
            print('KFold cross-validation instantiated...')

        # search
        if search_type == 'random':
            print('RANDOM search in progress...\n')
            n_iter_rnd = min(n_iter_rnd,search_space)
            search = RandomizedSearchCV(estimator=estimator, 
                                        param_distributions=params,
                                        n_iter=n_iter_rnd,
                                        scoring=scoring,
                                        cv=cv, 
                                        iid=iid,
                                        random_state=seed,
                                        n_jobs=n_jobs,
                                        verbose=verbose)
            
        elif search_type == 'grid':
            print('GRID search in progress...\n')
            search = GridSearchCV(estimator=estimator, 
                                  param_grid=params,
                                  scoring=scoring,
                                  cv=cv, 
                                  iid=iid,
                                  n_jobs=n_jobs,
                                  verbose=verbose)
            
        else:
            print('ERROR: search_type must be either random or grid')
            sys.exit(-1)
                   
        search.fit(X,y)
        results = search.cv_results_
        best_result = {'params':search.best_params_,
                       'score': search.best_score_} 

        # define solutions to prepare next itaration        
        keep_top = int(math.ceil(keep_top_perc * len(results['rank_test_score'])))
        keep_top = min(max(keep_top, 1),len(results['rank_test_score']))
        print_opt_params_report(results=results, n_top=keep_top)
        
        # re-define params for next itaration
        params = _update_params(params, results, keep_top, params_dict)
        search_space = np.prod(np.array([len(list(params[k])) for k in params.keys()]))
        n_iter += 1
        
    print('\n--- {} optimization completed @ {} in {} iterations, with left {} iterations and a search space of {} ---\n'.format(
            list(params_dict.keys()), datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
            n_iter, n_iter_max-n_iter, search_space))

    return best_result, results


def _map_eval_scoring_metric(eval_metric):

    if eval_metric == 'logloss':
        scoring_metric = 'neg_log_loss'
        
    else:
        scoring_metric = eval_metric
    
    return scoring_metric 


def xgb_full_params_optimization(dtrain, target, predictors, eval_metric='logloss', search_type='random',
                                 params_dict = {'max_depth':{'init':5,'min':2,'max':10,'step':1,'type':'add'},
                                                'min_child_weight':{'init':1,'min':1,'max':20,'step':1,'type':'add'},
                                                'gamma':{'init':0.00,'min':0.00,'max':1.00,'step':0.01,'type':'add'},
                                                'subsample':{'init':0.80,'min':0.50,'max':1.00,'step':0.10,'type':'add'},
                                                'colsample_bytree':{'init':0.80,'min':0.50,'max':1.00,'step':0.10,'type':'add'},
                                                'reg_alpha':{'init':0,'min':1e-7,'max':1e+1,'step':10,'type':'multi'},
                                                'reg_lambda':{'init':1,'min':1e-4,'max':1e+3,'step':10,'type':'multi'},
                                                'learning_rate':{'init':0.1,'min':1e-4,'max':1e+1,'step':10,'type':'multi'},
                                        },
                                 num_boost_round=1000, early_stopping_rounds=50,
                                 cv_folds=5, cv_stratified=True, cv_shuffle=True, iid=False, 
                                 n_iter_max=5, keep_top_perc=0.20, n_iter_rnd=15, 
                                 n_jobs=1, seed=8888, verbose=False):    
    '''XGBoost full optimization pipeline:
    Step 0: Set initial guess for parameters (high learning rate);
    Step 1: Tune number of estimators;
    Step 2: Tune max_depth and min_child_weight;
    Step 3: Tune gamma;
    Step 4: Tune subsample and colsample_bytree;
    Step 5: Tuning Regularization Parameters reg_alpha, reg_lambda;
    Step 6: Tune learning rate;
    Step 7: Optimize no_estimators again.
    
    Parameters
    ----------
    
    Returns
    -------
    Dictionary of best parameters   
    
    '''
    
    print('\n> Optimization pipeline started @ {}\n'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    # Step 0: Set initial guess for parameters (high learning rate)      
    xgb_opt = XGBClassifier(n_estimators=num_boost_round,
                            max_depth=params_dict['max_depth']['init'], 
                            min_child_weight=params_dict['min_child_weight']['init'], 
                            gamma=params_dict['gamma']['init'], 
                            subsample=params_dict['subsample']['init'],
                            colsample_bytree=params_dict['colsample_bytree']['init'], 
                            reg_alpha=params_dict['reg_alpha']['init'], 
                            reg_lambda=params_dict['reg_lambda']['init'], 
                            learning_rate=params_dict['learning_rate']['init'], 
                            n_jobs=n_jobs, 
                            seed=seed)
                   
    # Step 1: Tune number of estimators 
    no_estimators, _ = xgb_opt_no_estimators(alg=xgb_opt, dtrain=dtrain, target=target, predictors=predictors, 
                                             eval_metric=eval_metric, num_boost_round=num_boost_round,
                                             early_stopping_rounds=early_stopping_rounds,
                                             cv_folds=cv_folds, cv_stratified=cv_stratified, cv_shuffle=cv_shuffle,
                                             seed=seed, verbose=verbose)
    
    # Step 2: Tune max_depth and min_child_weight 
    xgb_opt = XGBClassifier(n_estimators=no_estimators,
                            max_depth=params_dict['max_depth']['init'], 
                            min_child_weight=params_dict['min_child_weight']['init'], 
                            gamma=params_dict['gamma']['init'], 
                            subsample=params_dict['subsample']['init'],
                            colsample_bytree=params_dict['colsample_bytree']['init'], 
                            reg_alpha=params_dict['reg_alpha']['init'], 
                            reg_lambda=params_dict['reg_lambda']['init'], 
                            learning_rate=params_dict['learning_rate']['init'], 
                            n_jobs=n_jobs, 
                            seed=seed)

    scoring_metric = _map_eval_scoring_metric(eval_metric)
    params_to_opt = {}
    for k in ['max_depth','min_child_weight']:
        params_to_opt[k] = params_dict[k]
    
    best_result, _ = xgb_opt_params(X=dtrain[predictors], y=dtrain[target], estimator=xgb_opt,
                                    scoring=scoring_metric, search_type=search_type, params_dict=params_to_opt,    
                                    n_iter_max=n_iter_max, keep_top_perc=keep_top_perc, n_iter_rnd=n_iter_rnd, 
                                    cv_folds=cv_folds, cv_stratified=cv_stratified, cv_shuffle=cv_shuffle, 
                                    iid=iid, n_jobs=n_jobs, seed=seed, verbose=verbose)
            
    max_depth = best_result['params']['max_depth']
    min_child_weight = best_result['params']['min_child_weight']
    
    # Step 3: Tune gamma
    xgb_opt = XGBClassifier(n_estimators=no_estimators,
                            max_depth=max_depth, 
                            min_child_weight=min_child_weight, 
                            gamma=params_dict['gamma']['init'], 
                            subsample=params_dict['subsample']['init'],
                            colsample_bytree=params_dict['colsample_bytree']['init'], 
                            reg_alpha=params_dict['reg_alpha']['init'], 
                            reg_lambda=params_dict['reg_lambda']['init'], 
                            learning_rate=params_dict['learning_rate']['init'], 
                            n_jobs=n_jobs, 
                            seed=seed)
    
    scoring_metric = _map_eval_scoring_metric(eval_metric)
    params_to_opt = {}
    for k in ['gamma']:
        params_to_opt[k] = params_dict[k]
    
    best_result, _ = xgb_opt_params(X=dtrain[predictors], y=dtrain[target], estimator=xgb_opt,
                                    scoring=scoring_metric, search_type=search_type, params_dict=params_to_opt,    
                                    n_iter_max=n_iter_max, keep_top_perc=keep_top_perc, n_iter_rnd=n_iter_rnd, 
                                    cv_folds=cv_folds, cv_stratified=cv_stratified, cv_shuffle=cv_shuffle, 
                                    iid=iid, n_jobs=n_jobs, seed=seed, verbose=verbose)

    gamma = best_result['params']['gamma']    
    
    # Step 4: Tune subsample and colsample_bytree
    xgb_opt = XGBClassifier(n_estimators=no_estimators,
                            max_depth=max_depth, 
                            min_child_weight=min_child_weight, 
                            gamma=gamma, 
                            subsample=params_dict['subsample']['init'],
                            colsample_bytree=params_dict['colsample_bytree']['init'], 
                            reg_alpha=params_dict['reg_alpha']['init'], 
                            reg_lambda=params_dict['reg_lambda']['init'], 
                            learning_rate=params_dict['learning_rate']['init'], 
                            n_jobs=n_jobs, 
                            seed=seed)

    scoring_metric = _map_eval_scoring_metric(eval_metric)
    params_to_opt = {}
    for k in ['subsample','colsample_bytree']:
        params_to_opt[k] = params_dict[k]
    
    best_result, _ = xgb_opt_params(X=dtrain[predictors], y=dtrain[target], estimator=xgb_opt,
                                    scoring=scoring_metric, search_type=search_type, params_dict=params_to_opt,    
                                    n_iter_max=n_iter_max, keep_top_perc=keep_top_perc, n_iter_rnd=n_iter_rnd, 
                                    cv_folds=cv_folds, cv_stratified=cv_stratified, cv_shuffle=cv_shuffle, 
                                    iid=iid, n_jobs=n_jobs, seed=seed, verbose=verbose)
    
    subsample = best_result['params']['subsample']    
    colsample_bytree = best_result['params']['colsample_bytree']    
    
    # Step 5: Tuning Regularization Parameters reg_alpha, reg_lambda
    xgb_opt = XGBClassifier(n_estimators=no_estimators,
                            max_depth=max_depth, 
                            min_child_weight=min_child_weight, 
                            gamma=gamma, 
                            subsample=subsample,
                            colsample_bytree=colsample_bytree, 
                            reg_alpha=params_dict['reg_alpha']['init'], 
                            reg_lambda=params_dict['reg_lambda']['init'], 
                            learning_rate=params_dict['learning_rate']['init'], 
                            n_jobs=n_jobs, 
                            seed=seed)

    scoring_metric = _map_eval_scoring_metric(eval_metric)
    params_to_opt = {}
    for k in ['reg_alpha','reg_lambda']:
        params_to_opt[k] = params_dict[k]
    
    best_result, _ = xgb_opt_params(X=dtrain[predictors], y=dtrain[target], estimator=xgb_opt,
                                    scoring=scoring_metric, search_type=search_type, params_dict=params_to_opt,    
                                    n_iter_max=n_iter_max, keep_top_perc=keep_top_perc, n_iter_rnd=n_iter_rnd, 
                                    cv_folds=cv_folds, cv_stratified=cv_stratified, cv_shuffle=cv_shuffle, 
                                    iid=iid, n_jobs=n_jobs, seed=seed, verbose=verbose)
    
    reg_alpha = best_result['params']['reg_alpha']    
    reg_lambda = best_result['params']['reg_lambda']    
    
    # Step 6: Tune Learning Rate 
    xgb_opt = XGBClassifier(n_estimators=no_estimators,
                            max_depth=max_depth, 
                            min_child_weight=min_child_weight, 
                            gamma=gamma, 
                            subsample=subsample,
                            colsample_bytree=colsample_bytree, 
                            reg_alpha=reg_alpha, 
                            reg_lambda=reg_lambda, 
                            learning_rate=params_dict['learning_rate']['init'], 
                            n_jobs=n_jobs, 
                            seed=seed)

    scoring_metric = _map_eval_scoring_metric(eval_metric)
    params_to_opt = {}
    for k in ['learning_rate']:
        params_to_opt[k] = params_dict[k]
    
    best_result, _ = xgb_opt_params(X=dtrain[predictors], y=dtrain[target], estimator=xgb_opt,
                                    scoring=scoring_metric, search_type=search_type, params_dict=params_to_opt,    
                                    n_iter_max=n_iter_max, keep_top_perc=keep_top_perc, n_iter_rnd=n_iter_rnd, 
                                    cv_folds=cv_folds, cv_stratified=cv_stratified, cv_shuffle=cv_shuffle, 
                                    iid=iid, n_jobs=n_jobs, seed=seed, verbose=verbose)
    
    learning_rate = best_result['params']['learning_rate']  

    # Step 7: Tune number of estimators again 
    xgb_opt = XGBClassifier(n_estimators=num_boost_round,
                            max_depth=max_depth, 
                            min_child_weight=min_child_weight, 
                            gamma=gamma, 
                            subsample=subsample,
                            colsample_bytree=colsample_bytree, 
                            reg_alpha=reg_alpha, 
                            reg_lambda=reg_lambda, 
                            learning_rate=learning_rate, 
                            n_jobs=n_jobs, 
                            seed=seed)
    
    no_estimators, _ = xgb_opt_no_estimators(alg=xgb_opt, dtrain=dtrain, target=target, predictors=predictors, 
                                             eval_metric=eval_metric, num_boost_round=num_boost_round,
                                             early_stopping_rounds=early_stopping_rounds,
                                             cv_folds=cv_folds, cv_stratified=cv_stratified, cv_shuffle=cv_shuffle,
                                             seed=seed, verbose=verbose)

    # Return optimal params   
    print('\n> Optimization pipeline completed @ {}\n'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    return {'max_depth':max_depth,
            'min_child_weight':min_child_weight,
            'gamma':gamma,
            'subsample':subsample,
            'colsample_bytree':colsample_bytree,
            'reg_alpha':reg_alpha,
            'reg_lambda':reg_lambda,
            'n_estimators':no_estimators,
            'learning_rate':learning_rate}

    

    