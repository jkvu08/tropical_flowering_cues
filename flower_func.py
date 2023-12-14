# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 14:05:32 2022
Objective of this project is to identify the weather cues that tropical plants use to regulate their flowering and the timescales at which they occur. 
Functions for Bayesian logistic regression pipeline to predict and visualize the probability of flowering in a species given the mean weather conditions in a preceding time window. 
@author: Jannet
"""
# load packages/modules
import os, time
import arviz as az
import matplotlib.pyplot as pyplot
import numpy as np
import pandas as pd
from pandas import read_csv, DataFrame
import pymc3 as pm
import seaborn as sns
import theano.tensor as tt
from sklearn.metrics import roc_curve, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.metrics import classification_report, average_precision_score, log_loss, roc_auc_score, accuracy_score, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
import joblib
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import imblearn
from imblearn.metrics import geometric_mean_score
from imblearn.over_sampling import RandomOverSampler
import gc
from pymc3.distributions.continuous import Lognormal
from PIL import Image

# assign labels for the visualization
LABELS = ["no flower","flower"]

# cue dictionary that associates the weather cue name with column index in covariate array
# used to subset covariate array
cue_dict = {'rain': 0,
            'drought': 0,
            'temp': 1,
            'solar': 2}

#################################
### DATA FORMATTING FUNCTIONS ###
#################################
def data_gen(data, params):
    """
    Generate species specific response and covariate data.

    Parameters
    ----------
    data : species level phenology data
    params : model dictionary
    
    Returns
    -------
    datasub : focal species phenology data
    X : focal species covariates
    y : focal species phenology

    """
    datasub = data[(data['Species'] == params['species']) & (pd.notna(data['flower']))].reset_index() # subset out focal species data and get rid of flowering NA values
    X = joblib.load(params['species']+'_rain_tmin_solar_mean_norm_binom.pkl') # import species specific covariate arrays which will be used as predictors
    # This is species specific because it maps to the exact date that the species was surveyed
    # The data contains the normalized weatheer data for all cue window and lag time combinations with respect to the survey date
    cue_list =[] # create empty list for weather cue keys 
    # populate cue keys in a list
    for cue in params['covariates']: # for each weather condition being tested as a cue 
        cue_list += [cue_dict[cue]] # extract the relevant weather cue id based on the cue dictionary
    X = X[:,:,:,np.r_[cue_list]] # use cue key list to subset out the relevant covariates
    y = np.array(datasub.loc[:, ['flower','n_fl','prop_fl']]) # subset out only columns pertaining to flowering phenology
    if len(X.shape) == 3: # if the shape of the covariate array is 3D make it 4D
        X = X[:,:,:,np.newaxis]
    return datasub, X, y

def train_test_data(X,y,datasub,gss):
    """
    generate training and testing data

    Parameters
    ----------
    X : focal species covariates
    y: focal species responses
    datasub : focal species dataset
    gss : group split

    Returns
    -------
    train_X : list of training indices
    valid_list : list of valid indices

    """
    # for each kfold get the training and validation indices
    # for this analysis there is only 1 kfold, but this would be extended to multiple kfolds
    for train_index,valid_index in gss.split(X[0,0,:,0], y, groups = datasub['year']):
        train_X, valid_X = X[:,:,train_index,:], X[:,:,valid_index,:]
        train_ds, valid_ds = datasub.iloc[train_index,:], datasub.iloc[valid_index,:] 
        train_y, valid_y = y[train_index,:], y[valid_index,:]
    return train_X, valid_X, train_y, valid_y, train_ds, valid_ds

###############################
### DATA MODELING FUNCTIONS ###
###############################
def single_model(X, y, params):
    """
    architecture for Bayesian flowering model when plant assumed to be cued by a single weather condition

    Parameters
    ----------
    X : covariate array
    y : observed number of individuals flowering 
    params: model parameters

    Returns
    -------
    single_model: model 
        
    """
    samples = int(params['ni']/params['nt']) # calculate number of samples to store
    # create model and add priors  
    with pm.Model() as single_model: 
        X = pm.Data('X',X) # assign predictors
        n = pm.Data('n',y[:,1]) # assign n for binomial (number of individuals sampled on a given day)
        # alpha0 is the logistic intercept that flowering occurs even when the cue threshold is not met
        alpha0=pm.Normal('alpha0', mu=0, sigma=params['alpha_sd']) # normal prior for alpha
        if (params['direction'][0]) == 'positive': # flowering occurs when weather conditions exceed a threshold (reserved for rain, warm temp and high light cues)
            # beta0 is the logistic slope, which describes the relationship between the weather conditions during the cue period and the prob of flowering
            beta0=pm.Exponential('beta0', lam=0.1) # Exponential prior for beta
        else: # flowering occurs when weather conditions fall below a threshold (reserved for drought, low temp and low light cues)
            prebeta0 = pm.Exponential('prebeta0', lam=0.1) # Exponential prior for prebeta 
            beta0 = pm.Deterministic('beta0', prebeta0*-1) # multiply by -1 to ensure the negative directionality between weather conditions and the threshold 
        
        # determine cue period, which is defined by lag and window
        # lag0 is the number of days between the cue window and flowering event      
        lag0 = pm.DiscreteUniform('lag0', lower=params['lower_lag0'], upper=params['upper_lag0']) # discrete uniform prior for lag
        # constrain lag time to the a priori determined upper limit of lag times being tested
        # this limits to the search space to the more immediate weather conditions, since weather can be cyclicty
        # also need to do this since the weather cues only calculated within the a priori determined range
        lag0 = tt.switch(tt.gt(lag0,params['upper_lag0']),params['upper_lag0'], lag0) # if the lag is greater than the upper limit, reassign to the upper limit 
        lag0 = tt.switch(tt.lt(lag0,0),0, lag0) # if lag is lower than 0 reassign to 0 since lag cannot be negative
        
        # cue window0 is the number of consecutive days in which the weather cue occurs 
        window0 = pm.DiscreteUniform('window0', lower=1, upper=params['upper_lb']) # discrete uniform prior for window
        # constrain window to a priori determined upper limit for time window
        # since many tropical plants flower sub-annually, this helps to limit the cue to the immediate cycle being assessed
        window0 = tt.switch(tt.gt(window0,params['upper_lb']),params['upper_lb'], window0) # if window is greater than the upper limit, reassign to the upper limit
        window0 = tt.switch(tt.lt(window0,1),1, window0) # if the window is lower than 1 reassign to 1 since need at least one day of cues
        
        # two modeling options with out without threshold criteria
        if params['threshold'] == True: # if prob of flowering increases after threshold condition met
            # since all weather conditions are normalized, threshold weather condition must also fall within the 0-1 range
            threshold0 = pm.Uniform('threshold0', lower= 0, upper = 1) # normal prior for threshold
            # prob of flowering increases once weather conditions exceed threshold during the cue period
            if params['direction'][0] == 'positive':
                # if weather condition does not meet threshold, then reassign to 0, otherwise weather condition0 - threshold0  
                w0 = pm.Deterministic('w0', tt.switch(tt.lt(X[lag0,window0,:,0]-threshold0,0),0,X[lag0,window0,:,0]-threshold0)) # if weather conditons < threshold then ressign to 0
            # prob of flowering increases once weather conditions drops below a threshold during the cue period
            else:
                # if weather condition does not meet threshold, then reassign to 0, otherwise weather condition0 - threshold0  
                w0 = pm.Deterministic('w0', tt.switch(tt.gt(X[lag0,window0,:,0]-threshold0,0),0,X[lag0,window0,:,0]-threshold0)) # if weather conditions > threshold reassign to 0 
        else: # otherwise use non-threshold model, flowering prob is function of weather conditions during cue period
            w0 = pm.Deterministic('w0', X[lag0,window0,:,0])
        
        # generate probability that the species flowers
        p = pm.Deterministic('p', pm.math.invlogit(alpha0 + beta0*w0)) 
        x0 = pm.Deterministic('x0', X[lag0,window0,:,0]) # get the weather conditions during the cue period
        observed = pm.Binomial("y", n=n, p=p, observed=y[:,0]) # add observed number of individuals flowering on each survey day
    return single_model

def double_model(X,y,params):
    """
    architecture for Bayesian flowering model when plant assumed to be cued by a two weather conditions

    Parameters
    ----------
    X : covariate array
    y : observed number of individuals flowering 
    params: model parameters

    Returns
    -------
    double_model: model 

    """
    # create model and add priors  
    with pm.Model() as double_model: 
        X = pm.Data('X',X) # assign predictors
        n = pm.Data('n',y[:,1]) # assign n for binomial (number of individuals sampled on a given day)
        
        # alpha is the logistic intercept that flowering occurs even when the cue threshold for the a particular cue is not met
        alpha0=pm.Normal('alpha0', mu=0, sd=params['alpha_sd']) # normal prior for alpha for cue 0
        alpha1=pm.Normal('alpha1', mu=0, sd=params['alpha_sd']) # normal prior for alpha for cue 1
        
        # beta is the logistic slope, which describes the relationship between the weather conditions of a particular cue during the cue period and the prob of flowering    
        if (params['direction'][0]) == 'positive': # flowering occurs when weather conditions of cue 0 exceed a threshold (reserved for rain, warm temp and high light cues)
            beta0=pm.Exponential('beta0', lam=0.1) # exponential prior for beta on cue 0
        else: # flowering occurs when weather conditions fall below a threshold (reserved for drought, low temp and low light cues)
            prebeta0 = pm.Exponential('prebeta0', lam=0.1) # assign a prebeta exponential prior 
            beta0 = pm.Deterministic('beta0', prebeta0*-1) # then multiple the prebeta by -1 to generate a negative beta
        if (params['direction'][1]) == 'positive':# if the directionality of the cue 1 is positive
            beta1=pm.Exponential('beta1', lam=0.1) # exponential prior for beta on cue 1
        else: #if the directionality of the cue 1 is negative
            prebeta1 = pm.Exponential('prebeta1',lam=0.1) # assign a prebeta exponential prior 
            beta1 = pm.Deterministic('beta1', prebeta1*-1) # then multiple the prebeta by -1 to generate a negative beta
        
        # determine cue period, which is defined by lag and window
        # lag is the number of days between the cue window and flowering event  
        # cue window is the number of consecutive days in which the weather cue occurs
        # sample the window for cue 1
        window1 = pm.DiscreteUniform('window1', lower=1, upper=params['upper_lb']) # discrete uniform prior for window
        # constrain window to a priori determined upper limit for time window
        window1 = tt.switch(tt.gt(window1,params['upper_lb']),params['upper_lb'], window1) # if the window is greater than the upper limit, reassign to the upper limit 
        window1 = tt.switch(tt.lt(window1,1),1, window1)  # if the window is lower than 1 reassign to 1, since cue must occur for at least 1 day
        
        if params['sequential'] == False: # if cues periods can overlap
            # sample the lag for cue 1
            lag1 = pm.DiscreteUniform('lag1', lower=params['lower_lag1'], upper=params['upper_lag1']) # discrete uniform prior for lag 1
            # constrain lag time to the a priori determined upper and lower limit of lag times being tested
            lag1 = tt.switch(tt.gt(lag1,params['upper_lag1']),params['upper_lag1'], lag1) # if the lag is greater than the upper limit, reassign to the upper limit 
            lag1 = tt.switch(tt.lt(lag1,params['lower_lag1']),params['lower_lag1'], lag1)  # if the lag lower than lower limit, then reassign to lower limit   
            
            # sample the window for cue 0
            window0 = pm.DiscreteUniform('window0', lower=1, upper=params['upper_lb']) # discrete uniform prior for window
            # constrain window to the a priori determined upper limit of cue windows being tested
            window0 = tt.switch(tt.gt(window0,params['upper_lb']),params['upper_lb'], window0) # if the window is greater than the upper limit, reassign to the upper limit 
            window0 = tt.switch(tt.lt(window0,1),1, window0)  # if the window is lower than 1 reassign to 1, 1 is the window lower limit
            
            # sample the lag for cue 0
            lag0 = pm.DiscreteUniform('lag0', lower=0, upper=params['upper_lag0']) # discrete uniform prior for lag 0
            # constrain lag time to the a priori determined upper limit of lag times being tested
            lag0 = tt.switch(tt.gt(lag0,params['upper_lag0']),params['upper_lag0'], lag0) # if the lag is greater than the upper limit, reassign to the upper limit 
            lag0 = tt.switch(tt.lt(lag0,0),0, lag0)  # if the lag lower than 0, then reassign to lower limit   
        
        # constrain lag and window to ensure sequential cues
        else: 
            # sample the lag for cue 1
            lag1 = pm.DiscreteUniform('lag1', lower=params['lower_lag1']+1, upper=params['upper_lag1']) # discrete uniform prior for lag
            # constrain lag time to the a priori determined upper and lower limit of lag times being tested
            lag1 = tt.switch(tt.gt(lag1,params['upper_lag1']),params['upper_lag1'], lag1) # if the lag is greater than the upper limit, reassign to the upper limit 
            lag1 = tt.switch(tt.lt(lag1,params['lower_lag1']+1),params['lower_lag1']+1, lag1)  # if the lag lower than the 1 + lower limit , then reassign to 1 + lower limit
            
            # sample the window for cue 0
            window0 = pm.DiscreteUniform('window0', lower=1, upper=lag1) # dicrete uniform prior for window0 with lag1 as upper limit
            # constrain window to the a priori determined upper limit of cue windows being tested and the lag
            window0 = tt.switch(tt.gt(window0,lag1),lag1, window0) # if the window is greater than the lag1, reassign to lag1
            window0 = tt.switch(tt.lt(window0,1),1, window0)  # if the window is lower than 1 reassign to 1
            
            # sample the lag for cue 0
            ulim = lag1-window0 # get upper limit for lag 0, to ensure the two cue periods don't overlap
            lag0 = pm.DiscreteUniform('lag0', lower=0, upper=ulim) # discrete uniform prior for lag 0
            lag0 = tt.switch(tt.gt(lag0,ulim),ulim, lag0) # if the lag is greater than the upper limit, reassign to the upper limit 
            lag0 = tt.switch(tt.lt(lag0,0),0, lag0)  # if the lag lower than 0, then reassign to lower limit   
        
        # two modeling options with out without threshold criteria
        if params['threshold'] == True: # if prob of flowering increases after threshold condition met
            # since all weather conditions are normalized, threshold weather condition must also fall within the 0-1 range
            threshold0 = pm.Uniform('threshold0', lower=0, upper=1) # uniform prior for threshold
            threshold1 = pm.Uniform('threshold1', lower=0, upper=1) # uniform prior for threshold
            # prob of flowering increases once weather conditions exceed threshold during the cue period for cue 0
            if params['direction'][0]== 'positive': 
                w0 = pm.Deterministic('w0', tt.switch(tt.lt(X[lag0,window0,:,0]-threshold0,0),0,X[lag0,window0,:,0]-threshold0)) # if weather conditions < threshold, reassign to 0
            # prob of flowering increases once weather conditions drops below threshold during the cue period for cue 0
            else:  
                w0 = pm.Deterministic('w0', tt.switch(tt.gt(X[lag0,window0,:,0]-threshold0,0),0,X[lag0,window0,:,0]-threshold0)) # if weather conditions > threshold, reassign to 0
            # prob of flowering increases once weather conditions exceed threshold during the cue period for cue 1
            if params['direction'][1]== 'positive': # if weather condition does not exceed threshold, then reassign to 0, otherwise keep weather condition - threshold  
                w1 = pm.Deterministic('w1', tt.switch(tt.lt(X[lag1,window1,:,1]-threshold1,0),0,X[lag1,window1,:,1]-threshold1)) # if weather conditions < threshold, reassign to 0
            # prob of flowering increases once weather conditions drops below threshold during the cue period for cue 1
            else:
                w1 = pm.Deterministic('w1', tt.switch(tt.gt(X[lag1,window1,:,1]-threshold1,0),0,X[lag1,window1,:,1]-threshold1)) # if weather conditions > threshold, reassign to 0
        # otherwise use nonthreshold model
        else: # flowering prob is function of weather conditions during cue period
            w0 = pm.Deterministic('w0', X[lag0,window0,:,0])
            w1 = pm.Deterministic('w1', X[lag1,window1,:,1])
        
        # generate probability that the species flowers
        p0 = pm.Deterministic('p0', pm.math.invlogit(alpha0 + beta0*w0)) # generate probability cue 0 triggers flowering 
        p1 = pm.Deterministic('p1', pm.math.invlogit(alpha1 + beta1*w1)) # generate probability cue 1 triggers flowering
        p = pm.Deterministic('p', p0*p1) # generate joint probability of flowering
        x0 = pm.Deterministic('x0', X[lag0,window0,:,0]) # get the weather conditions 0 at lag0 and window0
        x1 = pm.Deterministic('x1', X[lag1,window1,:,1]) # get the weather conditions 1 at lag0 and window0
        observed=pm.Binomial("y", n=n, p=p, observed=y[:,0]) # add observed number of individuals flowering on each survey day
        return double_model

def run_model(X,y, model, params, save = None):
    """
    run model and sample from posterior distribution

    Parameters
    ----------
    X : covariates table (validation)
    y : observed target data (number of individuals flowering) for validation
    model : model
    params: model parameters
    save : output saving mode. full = saves all results, trace = only saves trace, None = does not save results
         The default is None.

    Returns
    -------
    results in form of dictionary
        {'model': model 
         'trace': trace output from model
         'inference': inference data
         'ppc': posterior predictive samples training
         'vppc': posterior predictive samples validation 
         'filename': filename for results 
         }

    """
    start_time = time.time() # generate start time to monitor computation time
    with model:
        step1=pm.NUTS(target_accept=params['nuts_target']) # set sampler for continuous variables and assign mean acceptance probability (i.e., nuts_target)
        step2=pm.Metropolis() # set sampler for discrete variables
        step = [step1,step2] # put sampler steps together
        # draw samples
        trace=pm.sample(draws =params['ni'], # number of draws
                        step=step, 
                        return_inferencedata=False, # do not return inferenceData
                        chains=params['nc'], # number of chains
                        tune=params['nb']) # number of burn-ins
        # thin samples
        trace = trace._slice(slice(0,params['ni'],params['nt'])) 
        # sample posterior predictions for training data
        postpred = pm.sample_posterior_predictive(trace=trace, 
                                                  var_names = params['name_var']) 
        # get inference data
        infdata = az.from_pymc3(trace = trace, 
                                log_likelihood = True) 
        print('model training took', (time.time()-start_time)/60, 'minutes') # output how long the model took  
        
        # sample posterior predictions for validation data
        pm.set_data({'X': X,
                     'n': y[:,1].astype('int32')})
        vppc = pm.sample_posterior_predictive(trace,
                                              var_names =params['name_var'])
        
    print('model traing & validation took', (time.time()-start_time)/60, 'minutes') # output how long the model took  
        
    # assign filename
    if params['threshold'] == True:    
        filename = params['species'] + '_' + '_'.join(params['covariates']) + '_'+ params['relation'] + params['mtype'] 
    else:
        filename = params['species'] + '_' + '_'.join(params['covariates']) + '_'+ params['relation'] +'_nt'+ params['mtype']
 
    # put results in dictionary
    results = {'model': model,
               'trace': trace,
               'inference': infdata,
               'ppc': postpred,
               'vppc': vppc,
               'filename':filename}
    
    # save output
    if save == 'full':# save all model components
        start_time = time.time() # restart the time to see how long saving takes
        # put model and components into a dictionary and save as pickle
        joblib.dump(results,filename+'.pkl')    
    elif save == 'trace': # only save the trace
        joblib.dump(infdata, filename+'_trace.pkl')
    else:
        print('model not saved')
    print('model saving took', str((time.time()-start_time)/60), 'minutes') # output how long saving takes
    
    return results

#################################
### DATA PREDICTION FUNCTIONS ###
#################################
def bern_pred_gen(y, ypred, ds = 'train'):
    '''
    generate bernoulli predictions 

    Parameters
    ----------
    y : observed counts of the individuals flowering
    ypred : predicted counts of individuals flowering 
    ds : datasplit identifier, 'train' or 'valid'
        DEFAULT = 'train'

    Returns
    -------
    bern_tab : dataframe of obs and pred bernoulli trials of species flowering with obs and pred prob of flowering 

    '''
    # create empty list to store data for
    obs = [] # observed absence/presence of individuals flowering (0/1)
    probs = [] # observed prob of flowering 
    preds = [] # predicted absence/presence of individuals flowering (0/1)
    pprobs = [] # predicted prob of flowering 
    # for each survey date
    for i in range(len(y)):    
        # generate observed 0/1
        ones= np.ones(int(y[i, 0])) # generate 1's 
        zeros= np.zeros(int(y[i,1]-y[i,0])) # generate 0's
        obs_oz = np.concatenate([ones,zeros]) # concatenate 1's and 0's
        np.random.shuffle(obs_oz) # randomly shuffle - can randomly shuffle under the assumption that individual identity is not important
        proba = np.repeat(y[i,2], int(y[i,1])) # duplicate the obs prob of flowering
        
        # generate predicted 0/1 
        pones= np.ones(int(ypred[i, 0])) # generate 1's 
        pzeros= np.zeros(int(ypred[i,1]-ypred[i,0])) # generate 0's
        pred_oz = np.concatenate([pones,pzeros]) # concatenate 1's and 0's
        np.random.shuffle(pred_oz) # randomly shuffle
        pproba = np.repeat(ypred[i,2], int(ypred[i,1])) # duplicate the pred prob of flowering
        
        # append results to lists
        obs += [obs_oz] # obs bernoulli trials of species flowering 
        probs += [proba] # obs prob of species flowering
        preds += [pred_oz] # pred bernoulli trials of species flowering 
        pprobs += [pproba] # pred prob of species flowering
    
    # convert from list to array
    obs = np.concatenate(obs) 
    probs = np.concatenate(probs)
    preds = np.concatenate(preds)
    pprobs = np.concatenate(pprobs)
    
    bern_tab = pd.DataFrame({'dataset': [ds]*len(obs),
                             'obs': obs, 
                             'obs_prob': probs, 
                             'pred': preds, 
                             'pred_prob': pprobs})
    
    return bern_tab

def prob_pred_bern(results,train_y,valid_y, path, save = True):
    """
    Generate class probability and predictions and save into file

    Parameters
    ----------
    results : result file from mcmc
    y: observed counts of individuals flowering
    path: path to save output

    Returns
    -------
    train_prob : training class probability
    train_pred : training class prediction
    valid_prob : validation class probability
    valid_pred : validation class prediction

    """
    train_prob = np.median(results['ppc']['p'],axis=0) # get mean posterior predictions (prob of flowering)
    train_pred = np.random.binomial(n=train_y[:,1].astype('int32'), p=train_prob) # generate predictions (counts of individuals flowering)
    train_predictions = np.column_stack((train_pred,train_y[:,1], train_prob)) # organize into same array
    
    valid_prob = np.median(results['vppc']['p'],axis=0) # get mean posterior predictions(prob of flowering)
    valid_pred = np.random.binomial(n=valid_y[:,1].astype('int32'), p=valid_prob) # generate predictions (counts of individuals flowering)
    valid_predictions = np.column_stack((valid_pred,valid_y[:,1], valid_prob)) # organize into same array
    
    obs = np.concatenate((train_y[:,0], valid_y[:,0]), axis= 0) # organize observations (obs counts of individuals flowering)
    probs= np.concatenate((train_prob, valid_prob), axis= 0) # organize probabilities (median pred prob of individuals flowering)
    pred = np.concatenate((train_pred, valid_pred), axis= 0) # organize predictions (pred counts of individuals flowering)

    # organize binomial predictions
    binom_pred = pd.DataFrame({'dataset': ['train']*len(train_y) + ['valid']*len(valid_y),
                             'obs': obs, 
                             'prob': probs, 
                             'pred': pred}) 
    
    train_tab = bern_pred_gen(train_y,train_predictions,'train') # generate bernoulli predictions for training data (expand counts of individuals flowering to 0 and 1s)
    valid_tab = bern_pred_gen(valid_y,valid_predictions,'valid') # generate bernoulli predictions for validation data (expand counts of individuals flowering to 0 and 1s)
    
    bern_pred = pd.concat([train_tab, valid_tab], axis = 0) # put training and validation predictions into same dataframe
    
    # to save or not
    if save == True:
        binom_pred.to_csv(path+results['filename'] + 'binom_predictions.csv')
        bern_pred.to_csv(path+results['filename'] + 'bern_predictions.csv')
    return train_tab, valid_tab, binom_pred

########################
### METRIC FUNCTIONS ###
########################
def confusion_mat(y, y_pred, LABELS, normalize = 'true'):
    """
    generates and visualizes the confusion matrix

    Parameters
    ----------
    y : labeled true values
    y_pred : labeled predictions
    LABELS : class labels
    normalize: true = data converted to proportions, None = data left as raw counts

    """
    cm = confusion_matrix(y,y_pred, normalize = normalize) # generate confusion matrix
    # generate the confusion matrix graphic
    if normalize == None: # visualize counts
        sns.heatmap(cm, 
                    xticklabels=LABELS, 
                    yticklabels=LABELS, 
                    annot=True, 
                    fmt ='d') 
    else: # visualize proportions
        sns.heatmap(cm, 
                    xticklabels=LABELS, 
                    yticklabels=LABELS, 
                    annot=True) 
        
    pyplot.title("Confusion matrix") # add title
    pyplot.ylabel('True class') # add y axis title
    pyplot.xlabel('Predicted class') # add x axis title

def class_report(y, y_pred):
    """
    generate the class report to get the recall, precision, f1 and accuracy per class and overall

    Parameters
    ----------
    y : labeled true values
    y_pred : labeled predictions

    Returns
    -------
    class_rep : classification report

    """
    # generatate classification report as dictionary
    class_rep = classification_report(y,
                                      y_pred, 
                                      zero_division = 0, 
                                      output_dict = True) 
    class_rep = DataFrame(class_rep).transpose() # convert dictionary to dataframe
    return class_rep 

def class_rep_split(train_y,train_pred, valid_y, valid_pred):
    """
    generate and merge classification reports for training and validation data

    Parameters
    ----------
    train_y : observed training phenology
    train_pred : training predictions
    valid_y : observed validation phenology
    valid_pred : validation predictions

    Returns
    -------
    class_rep : combined training and validation classification report as dataframe

    """
    train_class_rep = class_report(train_y,train_pred) # generate classification report for training 
    train_class_rep = round(train_class_rep,3) # round the classification report 
    train_class_rep.columns = 'train_' + train_class_rep.columns # rename column
    
    valid_class_rep = class_report(valid_y,valid_pred) # generate classification report for validation 
    valid_class_rep = round(valid_class_rep,3) # round the classification report
    valid_class_rep.columns = 'valid_' +valid_class_rep.columns # rename column
    
    class_rep = pd.concat([train_class_rep, valid_class_rep], axis=1) # concatenate the training and validation reports
    return class_rep

def get_hdi(ary, hdi_prob, var_names):
    """
    extract the credible intervals

    Parameters
    ----------
    ary : array used to calculate credible intervals
    hdi_prob : set the credible interval density
    var_names : specify the variables for which to calculate the intervals

    Returns
    -------
    ci : table of credible intervals for each variable of interest 

    """
    # get credible intervals
    ci = az.hdi(ary=ary,
                hdi_prob =float(hdi_prob), 
                var_names=var_names) 
    ci = ci.to_dataframe().transpose() # transpose and turn into dataframe
    ci.columns = ['lower_'+hdi_prob[2:4], 'upper_'+hdi_prob[2:4]] # rename columns
    return ci

def summary_tab(results, params):
    """
    generate and save summary table 

    Parameters
    ----------
    results : dictionary of results from model
    params : model parameters

    Returns
    -------
    sum_df : trace summary table
    hdi_df : creedible interval summary
    
    """
    # summarize parameter estimates
    sum_df = pm.summary(results['inference'], 
                        var_names = params['variables'])
    
    # calculate the median value
    medval = results['inference']['posterior'].median().values() 
    med_df = pd.DataFrame(data = medval) # convert median values to dataframe
    med_index = list(results['inference']['posterior'].keys()) # get list of parameters in posterior
    # create dataframe out of median values with index as parameter names
    med_df = pd.DataFrame(data = medval, 
                          index = med_index, 
                          columns = ['median']) 
    med_df = med_df.loc[sum_df.index,:] # sort the median values according to the parameter order in the parameter summary data frame
    
    sum_df = pd.concat([med_df, sum_df], axis = 1) # concatenate the median and summary tables
    
    # get credible intervals
    hdi_95 = get_hdi(results['inference'], 
                     '0.95', # get 95% credible intervals
                     var_names = params['variables'])
    hdi_90 = get_hdi(results['inference'], 
                     '0.90',  # get 90% credible intervals
                     var_names = params['variables'])
    hdi_80 = get_hdi(results['inference'], 
                     '0.80', # get 80% credible intervals
                     var_names = params['variables']) 
    hdi_50 = get_hdi(results['inference'], 
                     '0.50', # get 50% credible intervals
                     var_names = params['variables']) 
    hdi_df = pd.concat([hdi_95, hdi_90, hdi_80, hdi_50], axis = 1) # concatenate credible intervals into a single table
    
    # backtransform threshold values (unnormalize)
    if params['covariates'][0] == 'temp':
            sum_df.loc['threshold0',['median','mean','sd','hdi_3%','hdi_97%']] = sum_df.loc['threshold0',['median','mean','sd','hdi_3%','hdi_97%']]*(32.1225-14.71125)+14.71125 
            hdi_df.loc['threshold0',:] = hdi_df.loc['threshold0',:]*(32.1225-14.71125)+14.71125 
    elif params['covariates'][0] == 'solar':
        sum_df.loc['threshold0',['median','mean','sd','hdi_3%','hdi_97%']] = sum_df.loc['threshold0',['median','mean','sd','hdi_3%','hdi_97%']]*(8846-279)+279 
        hdi_df.loc['threshold0',:] = hdi_df.loc['threshold0',:]*(8846-279)+279
    else:
        sum_df.loc['threshold0',['median','mean','sd','hdi_3%','hdi_97%']] = sum_df.loc['threshold0',['median','mean','sd','hdi_3%','hdi_97%']]*(206.8) 
        hdi_df.loc['threshold0',:] = hdi_df.loc['threshold0',:]*(206.8)  

    if len(params['covariates']) == 2: # if there is a second cue
        if params['covariates'][1] == 'temp':
            sum_df.loc['threshold1',['median','mean','sd','hdi_3%','hdi_97%']] = sum_df.loc['threshold1',['median','mean','sd','hdi_3%','hdi_97%']]*(32.1225-14.71125)+14.71125
            hdi_df.loc['threshold1',:] = hdi_df.loc['threshold1',:]*(32.1225-14.71125)+14.71125
        elif params['covariates'][1] == 'solar':
            sum_df.loc['threshold1',['median','mean','sd','hdi_3%','hdi_97%']] = sum_df.loc['threshold1',['median','mean','sd','hdi_3%','hdi_97%']]*(8846-279)+279 
            hdi_df.loc['threshold1',:] = hdi_df.loc['threshold1',:]*(8846-279)+279
        else:
            sum_df.loc['threshold1',['median','mean','sd','hdi_3%','hdi_97%']] = sum_df.loc['threshold1',['median','mean','sd','hdi_3%','hdi_97%']]*(206.8) 
            hdi_df.loc['threshold1',:] = hdi_df.loc['threshold1',:]*(206.8)  
            
    summary_df = pd.concat([sum_df, hdi_df], axis = 1) # concatenate credible intervals to summary tables
    hdi_df = round(hdi_df,3) # round values
    sum_df = round(sum_df,3) # round values
    return sum_df, hdi_df

def comp_tab(results, path):
    """
    generate and save comparison metrics for model selection (loo, WAIC, deviance)

    Parameters
    ----------
    results : results from mcmc run
    path : directory to save output

    Returns
    -------
    comp_df : comparision table with model selection metrics

    """
    # calculate the loo metric
    loo = az.loo(results['inference'],
                 results['model']) 
    # calculate the deviance
    dloo = az.loo(results['inference'],
                  results['model'], 
                  scale ='deviance') 
    # calculate waic
    waic = az.waic(results['inference'],
                   results['model'])
    # concatenate metrics into a dataframe
    comp_df = pd.DataFrame((loo[0:5].values,waic[0:5].values,dloo[0:5].values), 
                           index = ['loo','waic','deviance'],  
                           columns = ['metric','se','p','n_samples','n_datapoints']) 
    # save metrics
    comp_df.to_csv(path+results['filename']+'_modcomp.csv')
    # save results 
    joblib.dump(loo, results['filename']+'_loo.pkl')
    # round the metrics
    comp_df = round(comp_df,3) 
    return comp_df

def output_metrics(bern_tab, binom_tab):
    """
    Calculates five metrics to assess predictive performance: log loss, accuracy, roc auc, pr auc and f1-score

    Parameters
    ----------
    bern_tab: observed and predicted bernoulli trials of flowering with observed and predicted prob of flowering
    binom_tab: observed and predicted counts of individuals flowering and prob of flowering 

    Returns
    -------
    list of metrics

    """
    y = bern_tab['obs'] # observed absence/presence of flowering (0/1)
    y_pred = bern_tab['pred'] # predicted absence/presence of flowering (0/1)
    y_prob = bern_tab['pred_prob'] # pred prob of flowering
    loss = log_loss(y, y_prob) # calculate log loss
    acc = accuracy_score(y,y_pred) # calculate accuracy 
    prec = precision_score(y,y_pred) # calculate precision
    rc = recall_score(y,y_pred) # calculate recall
    pr = average_precision_score(y, y_prob) # calculate weighed mean of precisions at each threshold given the recall
    roc = roc_auc_score(y, y_prob) # calculate tarea under the receiver operating characteristic curve
    f1 = f1_score(y,y_pred) # calculate the f1-score
    gmean = geometric_mean_score(y, y_pred) # calculate the geometric mean
    rmse = mean_squared_error(binom_tab['obs'], binom_tab['pred'], squared = False) # calculate the root mean squared error
    return [loss,acc,prec,rc,roc,pr,f1,gmean,rmse]

def outtab(train_tab, valid_tab, binom_tab, filename, path):
    '''
    format performance metrics into a dataframe

    Parameters
    ----------
    train_tab : training dataframe of observed and predicted bernoulli trials for flowering
    valid_tab : validation dataframe of observed and predicted bernoulli trials for flowering 
    binom_tab : dataframe of observed and predicted counts of individuals flowering and prob of flowering
    filename : file name 
    path : path to save file

    Returns
    -------
    output : dataframe of performance metrics

    '''
    # create empty dataframe and assign metrics as column names
    output = DataFrame(columns=['loss','acc','prec','rec','roc','pr','f1','gmean','rmse'])
    # generate training performance metrics
    train_results = output_metrics(train_tab, binom_tab[binom_tab['dataset'] == 'train'])
    output.loc[len(output.index)] = train_results # add to dataframe
    # generate validation performance metrics
    valid_results = output_metrics(valid_tab, binom_tab[binom_tab['dataset']=='valid'])
    output.loc[len(output.index)] = valid_results # add to dataframe
    output.index = ['train','valid'] # assign indices
    output.to_csv(path+ filename +'_output_metrics.csv') # save performance metric table
    return output

def get_ci(ary, alpha = 0.95):
    '''
    extract credible intervals from posterior samples

    Parameters
    ----------
    ary : array of posterior samples 
    alpha : credible interval threshold
        DEFAULT = 0.95

    Returns
    -------
    lpi : array of lower credible interval values 
    upi : array of upper credible interval values
    '''
    # assign lower and upper cutoffs
    lci = (1-alpha)/2 # lower cutoff
    uci = 1-lci # upper cutoff 
    
    # creat empty list to population credible interval values
    lpi = [] # lower credible intervals
    upi = [] # upper credible intervals
    
    # get indicese of lower and upper limit of the credible interval values
    llim = int(ary.shape[0]*lci)-1
    ulim = int(ary.shape[0]*uci)-1
    
    # extract credible interval values
    for i in range(ary.shape[1]): 
        svalues = np.sort(ary[:,i]) # sort the array chronologically 
        # add credible interval value based on credible interval indices
        lpi.append(svalues[llim]) 
        upi.append(svalues[ulim]) 
    # convert list to array
    lpi = np.array(lpi)
    upi = np.array(upi)
    return lpi, upi

###############################
### VISUALIZATION FUNCTIONS ###
###############################
def roc_plot(y, y_prob):
    """
    generate receiving operator curve graphic
    
    Parameters
    ----------
    y : labeled true values
    y_prob : predicted probability for positive class

    Returns
    -------
    None.

    """
    # get false positive rate, true positive rate and thresholds
    false_pos_rate, true_pos_rate, thresholds = roc_curve(y, y_prob) 
    # get auc roc score
    roc_auc = roc_auc_score(y,y_prob) 
    # generate roc curve
    pyplot.plot(false_pos_rate, 
                true_pos_rate, 
                linewidth=5, 
                label='ROC_AUC = %0.3f'% roc_auc) 
    pyplot.plot([0,1],[0,1], linewidth=5)  # plot 1 to 1 true positive: false positive line
    pyplot.xlim([-0.01, 1]) # set x limits
    pyplot.ylim([0, 1.01]) # set y limits
    pyplot.legend(loc='lower right') # add legend
    pyplot.title('ROC') # add title
    pyplot.ylabel('True Positive Rate') # add y axis title
    pyplot.xlabel('False Positive Rate') # add x axis title

def pr_plot(y, y_prob):
    """
    generate precision recall curve graphic

    Parameters
    ----------
    y : labeled true values
    y_prob : predicted probability for positive class

    Returns
    -------
    None.

    """
    precision, recall, thresholds = precision_recall_curve(y, y_prob) # get precision, recall and threshold values
    pr_auc = average_precision_score(y, y_prob) # get aucpr score
    # plot precision-recall curve
    pyplot.plot(recall,
                precision, 
                linewidth=5, 
                label='PR_AUC = %0.3f'% pr_auc) 
    baseline = len(y[y==1]) / len(y) # get baseline expection, minority class proportion
    # plot the baseline
    pyplot.plot([0,1],
                [baseline,baseline], 
                linewidth=5) 
    pyplot.xlim([-0.01, 1]) # set x limit
    pyplot.ylim([0, 1.01]) # set y limit
    pyplot.legend(loc='upper right') # add legend
    pyplot.title('PR curve') # add title
    pyplot.ylabel('Precision') # add y axis title 
    pyplot.xlabel('Recall') # add x axis title

def trace_image(results, params, path):
    '''
    generate and save trace plots

    Parameters
    ----------
    results : dictionary of results from model
    params : model parameters
    path : directory to save output
    
    Returns
    -------
    None.

    '''
    # generate trace plot
    az.plot_trace(results['inference'], 
                  compact = True, #  multiple chains on same plot
                  var_names = params['variables'], # specify variables to plot
                  divergences = params['divergence'], # specify divergence location
                  backend = 'matplotlib', # plot aesthetic 
                  backend_kwargs = {'tight_layout': True}) # tight layout 
    fig = pyplot.gcf() # get last figure (trace plots)
    fig.savefig(path+results['filename']+'_trace.jpg', dpi=150) # save the trace plots as an image to flatten
    pyplot.close() # close the plot

def covplot(data, y,cov='x0', covariate ='solar', medthreshold = 0, modname = None, legend = False):
    """
    Posterior predictive check. Plot y and mean predictive probability of y with 95% credible intervals against cue conditions

    Parameters
    ----------
    data : posterior predictive check
    y : labeled true values (obs prob of flowering)
    cov: covariate variable name to plot
        DEFAULT is 'x0'
    covariate: weather condition: 'rain', 'temp', 'solar'
        DEFAULT is 'solar'
    medthreshold: median threshold cue condition
    modname: plot title
        DEFAULT is None
    legend: boolean to add legend or not
        DEFAULT is False

    """
    # extract traces
    input_x = data[cov] # ppc for weather conditions during cue period
    if covariate == 'rain': # backtransform if rain data
        input_x = input_x*(206.8) 
    elif covariate == 'temp': # backtransform if temp data
        input_x = input_x*(32.1225-14.71125)+14.71125
    elif covariate == 'solar': # backtransform if solar data
        input_x = input_x*(8846-279)+279 
    else:
        raise Exception ('invalid covariate')
    input_p = data['p'] # predictions for probability of flowering
    p = np.median(input_p,0) # get median predictive probability of flowering for each survey date
    x = np.median(input_x,0) # get median estimated weather conditions during cue period for each survey date
    input_pi = data['y']/y[:,1] # divide binom pred counts of individuals flowering by number of individuals sampled

    # get the indices of cue values sorted in descending order
    idx = np.argsort(x)
    # get the 95% credible intervals
    az.plot_hdi(x,input_pi,0.95,smooth = False, fill_kwargs={"alpha": 0.2, 
                                                             "color": "grey", 
                                                             "label": "95% CI"})
    # get the 95% prediction intervals   
    az.plot_hdi(x,input_p,0.95,smooth = False, fill_kwargs={"alpha": 0.6, 
                                                            "color": "dimgrey", 
                                                            "label": "95% PI"})
    # plot vertical line for median threshold condition
    pyplot.vlines(x=medthreshold, 
                  ymin=0, 
                  ymax=1, 
                  color='black', 
                  ls='--')  
    # plot median predictive probability of flowering against median estimated cue
    pyplot.plot(x[idx], 
                p[idx], 
                color='black', 
                lw=1.5, 
                alpha = 0.5, 
                label="median p")
    # plot observed flowering prob
    pyplot.scatter(x, 
                   np.random.normal(y[:,2], 0.001), 
                   marker='.', 
                   alpha = 0.7,
                   c = 'black',
                   label="observed")
    
    # add x-axis titles
    if covariate == 'rain':
        pyplot.xlabel('rainfall (mm)')# add x axis title
    elif covariate == 'temp':
        pyplot.xlabel('temperature (C)')# add x axis title
    elif covariate == 'solar':
        pyplot.xlabel('solar radiation (W/m2)')# add x axis title
    else:
        raise Exception ('invalid covariate')
    
    # option to add legend
    if legend == True:
        pyplot.legend()
        
    # assign primary axes labels  
    pyplot.ylabel('p', rotation=0) # add y axis title
    pyplot.title(modname)

def time_series_plot(train_ds, valid_ds, train_y, valid_y, results, params, cov = 'x0', covariate = 'solar', ci = 0.95, modname =''):
    '''
    Plot time series of flowering prob and weather conditions with predition and credible intervals

    Parameters
    ----------
    train_ds : species specific training data 
    valid_ds : species specific validation data
    train_y : training flowering phenology data
    valid_y : validation flowering phenology data
    results : model results
    params : model parameters
    cov : covariate variable being plotted 
        The default is 'x0'.
    covariate : cue condition
        The default is 'solar'.
    ci : prediction/credible interval alpha  
        The default is 0.95.
    modname : plot title
    The default is ''.

    Raises
    ------
    Exception
        invalid covariate specified

    Returns
    -------
    None.

    '''

    # get posterior preditions for flowering prob
    data_vp = results['vppc']['p'] # validation data
    data_tp = results['ppc']['p'] # training data
    
    # regenerate probabilities from binomially sampled counts of individuals flowering
    data_vpi = results['vppc']['y']/valid_y[:,1] # validation
    data_tpi = results['ppc']['y']/train_y[:,1] # training
    
    # concatenate the two predicted probabilities together
    pred_pi = np.concatenate((data_tpi,data_vpi),axis =1) # validation
    pred_tv = np.concatenate((data_tp, data_vp), axis = 1) #  training
    
    # get weather conditions during cue period
    data_vx = results['vppc'][cov]
    data_tx = results['ppc'][cov]
    cov_tv = np.concatenate((data_tx, data_vx), axis = 1) # concatenate weather conditions together
    
    # backtransform weather condition values
    if covariate == 'rain':
        cov_tv = cov_tv*206.8
    elif covariate == 'temp':
        cov_tv = cov_tv*(32.1225-14.71125)+14.71125
    elif covariate == 'solar':
        cov_tv = cov_tv*(8846-279)+279
    else: 
        raise Exception ('invalid covariate')
    
    # get median posterior predictive
    p = np.median(pred_tv,0) # probability of flowering
    x = np.median(cov_tv,0) # weather conditions during cue period
    
    # get prediction/credible intervals
    plp95, pup95 = get_ci(pred_tv, ci)
    xlp95, xup95 = get_ci(cov_tv, ci)
    pilp95, piup95 = get_ci(pred_pi, ci)
    
    # concatenate datasub training and validation together
    obs_tv = pd.concat((train_ds, valid_ds), axis = 0)
   # obs_tv['dataset'] = np.concatenate([np.repeat('train', len(train_y)), np.repeat('valid', len(valid_y))])
    d = obs_tv['Date'].values # get dates
    d = [pd.to_datetime(t) for t in d] # convert to date time format 
    idx = np.argsort(d) # get chronologically ordered indices
    y = obs_tv['prop_fl'].values # get observe prop of individual flowering ( = prob of flowering)
    
    # plot time series of flowering and weather cues with prediction and credible intervals 
    fig, ax1 = pyplot.subplots(figsize = (10,3))
    ax2 = ax1.twinx()
    
    # prediction interval for flowering prob
    ax1.fill_between(np.sort(d), 
                     pilp95[idx], 
                     piup95[idx], 
                     color='#98FB98', 
                     alpha=0.6,
                     label='95% PI') 
    
    # credible interval for flowering prob
    ax1.fill_between(np.sort(d), 
                     plp95[idx], 
                     pup95[idx], 
                     color='#308014', 
                     alpha=0.6,
                     label ='95% CI') 
    
    # prediction interval for cue condition
    ax2.fill_between(np.sort(d), 
                     xlp95[idx], 
                     xup95[idx], 
                     color='dimgrey', 
                     alpha=0.6,
                     label='95% PI') 
    
    # median prob of flowering 
    ax1.plot(np.sort(d), 
             p[idx], 
             color='black', 
             lw=1, 
             label="median p") 
    
    # median estimate for cue condition 
    ax2.plot(np.sort(d), 
             x[idx], 
             color='black', 
             lw=1, 
             linestyle = 'dashed', 
             label="median x")
    
    # plot training observed flowering prob
    ax1.scatter(d[:len(train_ds)], 
                np.random.normal(y[:len(train_ds)],0.001), 
                marker='^', 
                alpha=0.6, 
                c = '#377eb8', 
                label="training")  
    
    # plot validation observed flowering prob
    ax1.scatter(d[len(train_ds):], 
                np.random.normal(y[len(train_ds):],0.001), 
                marker='o', 
                alpha=0.6, 
                c = '#a65628', 
                label="validation")
    
    # add legend
    # adjust plot size
    box1 = ax1.get_position()
    ax1.set_position([box1.x0, 
                      box1.y0 + box1.height * 0.2,
                      box1.width, 
                      box1.height*0.8])
    box2 = ax2.get_position()
    ax2.set_position([box2.x0, 
                      box2.y0 + box2.height * 0.2,
                      box2.width, 
                      box2.height*0.8])
    # add legend below plots
    ax1.legend(ncol= 5, loc='upper center', bbox_to_anchor=(0.5, -0.3), frameon=False)
    ax2.legend(ncol= 2, loc='upper center', bbox_to_anchor=(0.5, -0.45),frameon=False)
  
    # assign primary axes labels
    ax1.set_xlabel('date') # add x axis title
    ax1.set_ylabel('flowering proportion') # add y axis title
    # assign secondary axes labels
    if covariate == 'rain':
        ax2.set_ylabel('rainfall (mm)') # add y axis title
    elif covariate == 'temp':
        ax2.set_ylabel('temperature (C)') # add y axis title
    elif covariate == 'solar':
        ax2.set_ylabel('solar radiation (W/m2)') # add y axis title
    else: 
        raise Exception ('invalid covariate')
    ax1.set_title(modname) # assign title
    
def plot_bayes_split(train_y, train_pred, train_prob,valid_y, valid_pred, valid_prob):
    '''
    organize confusion matrices, roc curve, and precision-recall curve into a single figure
    
    Parameters
    ----------
    y : labeled true values
    y_pred : predicted class
    y_prob : predicted probability for positive class
    

    Parameters
    ----------
    train_y : training observed y
    train_pred : training y predictions 
    train_prob : training probability predictions
    valid_y : validation observed y
    valid_pred : validation y predictions
    valid_prob : validation probabilties predictions
    Returns
    -------
    fig : figure of the confusion matrices, roc curce and AUCPR curve

    '''
    fig, axis = pyplot.subplots(2,4,figsize=(12,6)) # generate figure subplots, axes and size 
    # plot training results
    pyplot.subplot(2,4,1) # designate subplot
    confusion_mat(train_y,train_pred, LABELS = LABELS, normalize = 'true') # plot confusion matrix with proportions
    pyplot.subplot(2,4,2)
    confusion_mat(train_y,train_pred, LABELS = LABELS, normalize = None) # plot confusion matrix with counts
    pyplot.subplot(2, 4, 3) 
    roc_plot(train_y, train_prob) # plot roc curve
    pyplot.subplot(2, 4, 4)  
    pr_plot(train_y, train_prob) # plot precision recall curve
    # plot validation results
    pyplot.subplot(2,4,5) # designate subplot
    confusion_mat(valid_y,valid_pred, LABELS = LABELS, normalize = 'true') # plot confusion matrix with proportions
    pyplot.subplot(2,4,6)
    confusion_mat(valid_y,valid_pred, LABELS = LABELS, normalize = None) # plot confusion matrix with counts
    pyplot.subplot(2, 4, 7) 
    roc_plot(valid_y, valid_prob) # plot roc curve
    pyplot.subplot(2, 4, 8)  
    pr_plot(valid_y, valid_prob) # plot precision recall curve
    fig.tight_layout() # set tight layout
    return fig

def pymc_vis_split(results, train_y, valid_y, train_ds, valid_ds, params, path):
    """
    wrapper function to visualize, organize and save pymc results into a single pdf

    Parameters
    ----------
    results : dictionary of results with model, trace, posterior predictions, features and filename
    train_y : observed number of individuals flowering (training)
    valid_y : observed number of individuals flowering (validation)
    train_ds : data subset for species (training)
    valid_ds : data subset for species (validation)
    params: model parameters
    path : directory to save results
    
    Returns
    -------
    train_pred : predicted classes for training data 
    train_prob : predicted probability for positive class for training data 
    valid_pred : predicted classes for validation data 
    valid_prob : predicted probability for positive cwlass for validation data 

    """
    # generate summary table and credible interval table 
    sum_df, hdi_df = summary_tab(results, 
                                 params) 
    # generate  model selection metrics 
    comp_df = comp_tab(results, 
                       path) 
    # generate trace image
    trace_image(results, 
                params, 
                path) 
    traceplt = Image.open(path+results['filename']+'_trace.jpg') # open trace plot images
    # generate predictions
    train_tab, valid_tab, binom_pred = prob_pred_bern(results, 
                                                       train_y, 
                                                       valid_y, 
                                                       path) 
    # generate classification reports
    class_rep = class_rep_split(train_tab['obs'],
                                train_tab['pred'],
                                valid_tab['obs'],
                                valid_tab['pred'])
    # generate pdfs
    with PdfPages(path+results['filename']+'_results.pdf') as pdf:
        pyplot.figure(figsize=(8.5, 9)) # assign figure size 
        pyplot.imshow(traceplt) # plot trace plot image
        pyplot.axis('tight') # show all data
        pyplot.axis('off') # turn off axis labels and lines
        pdf.savefig(dpi=150) # save figure
        pyplot.close() # close page
        
        # generate forest/ caterpillar plots with 95% credible intervals
        az.plot_forest(results['inference'], 
                       var_names = params['variables'], 
                       combined = True,
                       hdi_prob = 0.95, 
                       figsize=(6,3))
        pyplot.grid() # add grid to plot
        pdf.savefig() # save figure
        pyplot.close() # close page
        
        # generate forest/ caterpillar plots with 90% credible intervals
        az.plot_forest(results['inference'], 
                       var_names = params['variables'], 
                       combined = True,
                       hdi_prob = 0.90, 
                       figsize=(6,3))
        pyplot.grid() # add grid to plot
        pdf.savefig() # save figure
        pyplot.close() # close page
        
        # generate posterior predictive check plot
        hcol = len(params['covariates']) # number of cuues 
        fig,ax = pyplot.subplots(hcol,2,figsize =(8,3*hcol))
        # for each variable plot observed, predictions with credible intervals against weather conditions during cue period
        for i in range(hcol):
            pyplot.subplot(hcol,2,2*i+1)
            # generate for training
            covplot(data=results['ppc'],
                    y=train_y,
                    cov='w'+str(i),
                    covariate = params['covariates'][i],
                    medthreshold = sum_df.loc['threshold'+str(i), 'median'],
                    modname = params['species'] + ' training',
                    legend = False)
            pyplot.subplot(hcol,2,2*i+2)
             # generate for validation
            covplot(data=results['vppc'],
                    y=valid_y,
                    cov='w'+str(i),
                    covariate = params['covariates'][i],
                    medthreshold = sum_df.loc['threshold'+str(i), 'median'],
                    modname = params['species'] + ' validation',
                    legend = True)
        fig.tight_layout()
        pdf.savefig() # save figure
        pyplot.close() # close page
        
        # generate time series plots for each cue
        for i in range(hcol):
            time_series_plot(train_ds, 
                             valid_ds, 
                             train_y, 
                             valid_y, 
                             results = results, 
                             params = params,
                             cov='w'+str(i),    
                             covariate= params['covariates'][i],
                             ci = 0.95,
                             modname = params['species'])
            pdf.savefig() # save figure
            pyplot.close() # close page
            
        # generate confusion matrices, roc curves and pr curves  
        pb_plot = plot_bayes_split(train_tab['obs'],
                                   train_tab['pred'], 
                                   train_tab['pred_prob'], 
                                   valid_tab['obs'],
                                   valid_tab['pred'], 
                                   valid_tab['pred_prob'])
        pdf.savefig(pb_plot) # save figure
        pyplot.close() # close page
        
        # generate subplots to organize table data 
        fig, ax = pyplot.subplots(figsize=(8.5,12),nrows=4)
        # format classification report into graphic
        metrictab = ax[0].table(cellText=class_rep.values,
                                colLabels = class_rep.columns, 
                                rowLabels=class_rep.index,
                                loc='center')
        metrictab.auto_set_font_size(False) # trun off auto font
        metrictab.set_fontsize(9) # set font size to 9
        ax[0].axis('tight') 
        ax[0].axis('off')

        # format summary table into graphic
        sumtab = ax[1].table(cellText=sum_df.values,
                             colLabels = sum_df.columns, 
                             rowLabels=sum_df.index,
                             loc='center')
        sumtab.auto_set_font_size(False)
        sumtab.set_fontsize(9)
        ax[1].axis('tight')
        ax[1].axis('off')
        
        # format credible interval table into graphic
        hditab = ax[2].table(cellText=hdi_df.values,
                             colLabels = hdi_df.columns, 
                             rowLabels=hdi_df.index,
                             loc='center')
        hditab.auto_set_font_size(False)
        hditab.set_fontsize(9)
        ax[2].axis('tight')
        ax[2].axis('off')
        
        # format metric table into graphic
        comptab = ax[3].table(cellText=comp_df.values,
                              colLabels=comp_df.columns,
                              rowLabels=comp_df.index, 
                              loc='center')
        comptab.auto_set_font_size(False)
        comptab.set_fontsize(9)
        ax[3].axis('tight')
        ax[3].axis('off')
        pdf.savefig() # save figure
        pyplot.close() # close page
    return train_tab, valid_tab, binom_pred

#################################
### PIPELINE WRAPPER FUNCTION ###
#################################
def flower_model_wrapper(data, gss, params, path, species, covariates, threshold = True, direction = None, relation = None):
    """
    wrapper function to run Bayesian logistic regression cue models

    Parameters
    ----------
    data : formatted phenology data for all species
    gss: data split designation
    params : model parameters
    path : path to save model outputs
    species : focal species being modeled
    covariates: covariates of interest
    threshold: Boolean indicating whether the model incorporates thresholds or not
    direction: list of cue threshold directionality ('positive' = exceeds threshold, 'negative' = falls below threshold)
    relation: file name suffix used to specify the directionality of the cue relationships, but can take on any string
    
    Raises
    ------
    Exception
        print error if invalid covariate used

    Returns
    -------
    None.

    """
    first_time = time.time() # monitor computation time
    params['species'] = species # reassign species 
    params['covariates'] = covariates # reassign covariates
    params['threshold'] = threshold # reassign threshold 
    params['relation'] = relation # reassign relation
    params['direction'] = direction # reassign direction
    # subset focal species phenology data, covariate array and response array
    datasub, X, y = data_gen(data, params) 
    # split data into training and validation
    train_X, valid_X, train_y, valid_y, train_ds, valid_ds = train_test_data(X,
                                                                             y,
                                                                             datasub,
                                                                             gss)
    # run flowering cue model
    if len(params['covariates']) ==1: # if single weather condition assumed to cue flowering
        params['name_var'] = ['y','p','w0','x0'] # reassign variable names
        params['variables'] = ['~w0','~p','~x0'] # reassign variables
        # generate single cue model architecture
        model = single_model(train_X,
                             train_y,
                             params) 
    elif len(params['covariates'])==2: # if two weather conditions assumed to cue flowering
        params['name_var'] = ['y','p','p0','p1','w0','w1','x0','x1'] # reassign variable names
        params['variables'] = ['~p','~p0','~p1','~w0','~w1','~x0','~x1'] # reassign variables 
        # if rain and drought are assumed to be the cues, the cues must occur sequentially due to autocorrelation 
        if 'rain' in params['covariates'] and 'drought' in params['covariates']:
            params['sequential'] = True
        # if solar and temp are assumed to be the cues, the cues must occur sequentially due to high correlation
        elif 'solar' in params['covariates'] and 'temp' in params['covariates']:
            params['sequential'] = True
        # cue periods may overlap for other cue combinations
        else: 
            params['sequential'] = False
        # generate double cue model architecture 
        model = double_model(train_X,
                             train_y,
                             params) 
    else: # otherwise raise exception
        raise Exception ('too few or too many covariates entered')
    # run model and save output 
    results = run_model(valid_X,
                        valid_y,
                        model,
                        params, 
                        save=params['save']) 
    print('training took ' + str(round((time.time()-first_time)/60,3)) + ' mins') # output run time

    # save model output
    save_time = time.time() # monitor saving time
    joblib.dump(results,results['filename']+'.pkl')    
    print('model saving took: ' + str((time.time()-save_time)/60) + ' minutes') # output how long it takes to save
    
    # visualize results 
    train_tab, valid_tab, binom_tab = pymc_vis_split(results, 
                                                     train_y, 
                                                     valid_y, 
                                                     train_ds,
                                                     valid_ds,
                                                     params, 
                                                     path)
    print('visualization done')
    
    # generate performance metrics
    output = outtab(train_tab, 
                    valid_tab, 
                    binom_tab, 
                    results['filename'], 
                    path)
    print('took ' + str((time.time()-first_time)/60) + 'minutes') # output how long it takes to generate and save metrics
    gc.collect() # dump model from memory
    gc.collect() # dump model from memory
    gc.collect() # dump model from memory
    return output