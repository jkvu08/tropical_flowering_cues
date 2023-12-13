# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 14:05:32 2022

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

# set working directory
os.chdir("C:\\Users\\Jannet\\Documents\\Dissertation\\codes\\tropical_flowering_cues")
# set directory where results will be saved
path = "C:\\Users\\Jannet\\Documents\\Dissertation\\codes\\tropical_flowering_cues\\BGLM_flower\\"

# import data
#data = read_csv('kian_phenology_bernoulli_09062022.csv',header=0).iloc[:,1:]
data = read_csv('kian_phenology_binomial_format.csv',header=0).iloc[:,1:]
# data attributions:
# Date - date (YYYY-MM-DD)
# Species - species local name
# year - year 
# month - month
# week - week of the year
# biweekly - biweek of the year
# time - Julian time? 
# n_fl - number of individuals surveyed on Date for flowering
# flowers - number of individuals flowering on Date
# prop_fl - proportion of individuals flowering on Date 
# n_fr - number of individuals surveyed on Date for flowering
# fruit - number of individuals fruiting on Date
# prop_fr - proportion of individuals fruiting on Date 
# rain - ?
# temp.mean - mean daily temperature (C)
# solar.rad - cumulative daily solar radiation (W/m^2)

# assign labels for the visualization
LABELS = ["no flower","flower"]

# cue dictionary that associates the weather cue name with column index in covariate array
cue_dict = {'rain': 0,
            'drought': 0,
            'temp': 1,
            'solar': 2}

# generator that will be used to split the training and validation dataset
# use group split cause we are spliting by year
gss = GroupShuffleSplit(n_splits=1,  # only need 1 split: training and testing
                        train_size = 0.6, # use about 60% on training data and set a random state
                        random_state = 235) # optional to assigned random state

# define data formating functions
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

def train_test_data(X,y, datasub,gss):
    """
    generate training and testing data

    Parameters
    ----------
    X : focal species covariates
    datasub : focal species dataset
    gss : group split

    Returns
    -------
    train_list : list of training indices
    valid_list : list of valid indices

    """
    # for each kfold get the training and validation indices
    # for this analysis there is only 1 kfold, but this would be extended to multiple kfolds
    for train_index,valid_index in gss.split(X[0,0,:,0], y, groups = datasub['year']):
        train_X, valid_X = X[:,:,train_index,:], X[:,:,valid_index,:]
        train_y, valid_y = y[train_index,:], y[valid_index,:]
    return train_X, valid_X, train_y, valid_y

# define model functions
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
        # alpha is the logistic intercept that flowering occurs even when the cue threshold is not met
        alpha=pm.Normal('alpha', mu=0, sigma=params['alpha_sd']) # normal prior for alpha
        if (params['direction']) == 'positive': # flowering occurs when weather conditions exceed a threshold (reserved for rain, warm temp and high light cues)
            # beta is the logistic slope, which describes the relationship between the weather conditions during the cue period and the prob of flowering
            beta=pm.Exponential('beta', lam=0.1) # Exponential prior for beta
        else: # flowering occurs when weather conditions fall below a threshold (reserved for drought, low temp and low light cues)
            prebeta = pm.Exponential('prebeta', lam=0.1) # Exponential prior for prebeta 
            beta = pm.Deterministic('beta', prebeta*-1) # multiply by -1 to ensure the negative directionality between weather conditions and the threshold 
        
        # determine cue period, which is defined by lag and window
        # lag is the number of days between the cue window and flowering event      
        lag = pm.DiscreteUniform('lag', lower=params['lower_lag'], upper=params['upper_lag']) # discrete uniform prior for lag
        # constrain lag time to the a priori determined upper limit of lag times being tested
        # this limits to the search space to the more immediate weather conditions, since weather can be cyclicty
        # also need to do this since the weather cues only calculated within the a priori determined range
        lag = tt.switch(tt.gt(lag,params['upper_lag']),params['upper_lag'], lag) # if the lag is greater than the upper limit, reassign to the upper limit 
        lag = tt.switch(tt.lt(lag,0),0, lag) # if lag is lower than 0 reassign to 0 since lag cannot be negative
        
        # cue window is the number of consecutive days in which the weather cue occurs 
        window = pm.DiscreteUniform('window', lower=1, upper=params['upper_lb']) # discrete uniform prior for window
        # constrain window to a priori determined upper limit for time window
        # since many tropical plants flower sub-annually, this helps to limit the cue to the immediate cycle being assessed
        window = tt.switch(tt.gt(window,params['upper_lb']),params['upper_lb'], window) # if window is greater than the upper limit, reassign to the upper limit
        window = tt.switch(tt.lt(window,1),1, window) # if the window is lower than 1 reassign to 1 since need at least one day of cues
        
        # two modeling options with out without threshold criteria
        if params['threshold'] == True: # if prob of flowering increases after threshold condition met
            # since all weather conditions are normalized, threshold weather condition must also fall within the 0-1 range
            threshold = pm.Uniform('threshold', lower= 0, upper = 1) # normal prior for threshold
            # prob of flowering increases once weather conditions exceed threshold during the cue period
            if params['direction'] == 'positive':
                # if weather condition does not meet threshold, then reassign to 0, otherwise weather condition - threshold  
                w = pm.Deterministic('w', tt.switch(tt.lt(X[lag,window,:,0]-threshold,0),0,X[lag,window,:,0]-threshold)) # if weather onditons < threshold then ressign to 0
            # prob of flowering increases once weather conditions drops below a threshold during the cue period
            else:
                # if weather condition does not meet threshold, then reassign to 0, otherwise weather condition - threshold  
                w = pm.Deterministic('w', tt.switch(tt.gt(X[lag,window,:,0]-threshold,0),0,X[lag,window,:,0]-threshold)) # if weather conditions > threshold reassign to 0 
        else: # otherwise use non-threshold model, flowering prob is function of weather conditions during cue period
            w = pm.Deterministic('w', X[lag,window,:,0])
        
        # generate probability that the species flowers
        p = pm.Deterministic('p', pm.math.invlogit(alpha + beta*w)) 
        x = pm.Deterministic('x', X[lag,window,:,0]) # get the weather conditions during the cue period
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

def run_model(X, y, model, params, save = None):
    """
    run models

    Parameters
    ----------
    X : covariates table
    y : observed target data (number of individuals flowering)
    model : model
    params: model parameters
    save : output saving mode. full = saves all results, trace = only saves trace, None = does not save results
         The default is None.

    Returns
    -------
    results in form of dictionary
        {'model': model 
         'trace': trace output from model
         'ppc': posterior predictive sample
         'filename': filename for results 
         }

    """
    start_time = time.time() # generate start time to monitor computation time
    with model:
        step1=pm.NUTS(target_accept=params['nuts_target']) # set sampler for continous variables and mean acceptance probability
        step2=pm.Metropolis() # set sampler for discrete variables
        step = [step1,step2] # put sampler steps together
        trace=pm.sample(draws =params['ni'], step=step, return_inferencedata=False, # draw samples specifying number of draws, sampling method, do not return inferenceData
                              chains=params['nc'], tune=params['nb']) # number of chains, number of burn-ins
        trace = trace._slice(slice(0,params['ni'],params['nt'])) # thin samples
        postpred = pm.sample_posterior_predictive(trace=trace, var_names = params['name_var']) # sample posterior predictions
        infdata = az.from_pymc3(trace = trace, log_likelihood = True) # get inference data
    print('model took', (time.time()-start_time)/60, 'minutes') # output how long the model took
     # set filename
    if params['threshold'] == True:    
        filename = params['species'] + '_' + '_'.join(params['covariates']) + '_'+ params['mtype'] +'_'+ params['relation']  # assign filename
    else:
        filename = params['species'] + '_' + '_'.join(params['covariates']) + '_'+ params['mtype'] +'_'+ params['relation']+'_nt'  # assign filename
    
    if save == 'full':
        start_time = time.time() # restart the time to see how long saving takes
        # put model and components into a dictionary and save as pickle
        joblib.dump({'model': model,
                     'trace': trace,
                     'inference': infdata,
                     'ppc': postpred,
                     'filename':filename},filename+'.pkl')    
        print('model saving took', str((time.time()-start_time)/60), 'minutes') # output how long saving takes
    elif save == 'trace': # only save the trace
        joblib.dump(infdata, filename+'_trace.pkl')
        print('model saving took', str((time.time()-start_time)/60), 'minutes') # output how long saving takes
    
    return {'model': model,
            'trace': trace,
            'inference':infdata,
            'ppc': postpred,
            'filename':filename}

# define metric functions
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
    if normalize == None: # generate the confusion matrix graphic
        sns.heatmap(cm, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt ='d') # specific integers
    else:
        sns.heatmap(cm, xticklabels=LABELS, yticklabels=LABELS, annot=True)
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
    class_rep = classification_report(y,y_pred, zero_division = 0, output_dict = True) # generatate classification report as dictionary
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
    ci = az.hdi(ary=ary,hdi_prob =float(hdi_prob), var_names=var_names) # get credible intervals
    ci = ci.to_dataframe().transpose() # transpose and turn into dataframe
    ci.columns = ['lower_'+hdi_prob[2:4], 'upper_'+hdi_prob[2:4]] # rename columns
    return ci

def summary_tab(results, params,path):
    """
    generate and save summary table 

    Parameters
    ----------
    results : dictionary of results from model
    params : model parameters
    path : directory to save output

    Returns
    -------
    sum_df : trace summary table
    hdi_df : creedible interval summary
    
    """
    sum_df = pm.summary(results['inference'], var_names = params['variables']) # summarize parameter estimates
    medval = results['inference']['posterior'].median().values() # calculate the median value
    med_df = pd.DataFrame(data = medval) # convert median values to dataframe
    med_index = list(results['inference']['posterior'].keys()) # get list of parameters in posterior 
    med_df = pd.DataFrame(data = medval, index = med_index, columns = ['median']) # create dataframe out median values with index as parameter names
    med_df = med_df.loc[sum_df.index,:] # sort the median values according to the parameter order in the parameter summary data frame
    sum_df = pd.concat([med_df, sum_df], axis = 1) # concatenate the median and summary tables
    hdi_95 = get_hdi(results['inference'], '0.95', var_names = params['variables']) # get 95% credible intervals
    hdi_90 = get_hdi(results['inference'], '0.90', var_names = params['variables']) # get 90% credible intervals
    hdi_80 = get_hdi(results['inference'], '0.80', var_names = params['variables']) # get 80% credible intervals
    hdi_50 = get_hdi(results['inference'], '0.50', var_names = params['variables']) # get 50% credible intervals
    hdi_df = pd.concat([hdi_95, hdi_90, hdi_80, hdi_50], axis = 1) # concatenate credible intervals into a single table
    # backtransform threshold values 
    if len(params['covariates']) == 2: # if these results are from a double cue model 
        if params['covariates'][0] == 'temp':
            sum_df.loc['threshold0',['median','mean','sd','hdi_3%','hdi_97%']] = sum_df.loc['threshold0',['median','mean','sd','hdi_3%','hdi_97%']]*(32.1225-14.71125)+14.71125 
            hdi_df.loc['threshold0',:] = hdi_df.loc['threshold0',:]*(32.1225-14.71125)+14.71125 
        elif params['covariates'][0] == 'solar':
            sum_df.loc['threshold0',['median','mean','sd','hdi_3%','hdi_97%']] = sum_df.loc['threshold0',['median','mean','sd','hdi_3%','hdi_97%']]*(8846-279)+279 
            hdi_df.loc['threshold0',:] = hdi_df.loc['threshold0',:]*(8846-279)+279
        else:
            sum_df.loc['threshold0',['median','mean','sd','hdi_3%','hdi_97%']] = sum_df.loc['threshold0',['median','mean','sd','hdi_3%','hdi_97%']]*(206.8) 
            hdi_df.loc['threshold0',:] = hdi_df.loc['threshold0',:]*(206.8)  
        if params['covariates'][1] == 'temp':
            sum_df.loc['threshold1',['median','mean','sd','hdi_3%','hdi_97%']] = sum_df.loc['threshold1',['median','mean','sd','hdi_3%','hdi_97%']]*(32.1225-14.71125)+14.71125
            hdi_df.loc['threshold1',:] = hdi_df.loc['threshold1',:]*(32.1225-14.71125)+14.71125
        elif params['covariates'][1] == 'solar':
            sum_df.loc['threshold1',['median','mean','sd','hdi_3%','hdi_97%']] = sum_df.loc['threshold1',['median','mean','sd','hdi_3%','hdi_97%']]*(8846-279)+279 
            hdi_df.loc['threshold1',:] = hdi_df.loc['threshold1',:]*(8846-279)+279
        else:
            sum_df.loc['threshold1',['median','mean','sd','hdi_3%','hdi_97%']] = sum_df.loc['threshold1',['median','mean','sd','hdi_3%','hdi_97%']]*(206.8) 
            hdi_df.loc['threshold1',:] = hdi_df.loc['threshold1',:]*(206.8)  
    else: # otherwise the results are from a single cue model 
        if params['covariates'][0] == 'temp':
            sum_df.loc['threshold',['median','mean','sd','hdi_3%','hdi_97%']] = sum_df.loc['threshold',['median','mean','sd','hdi_3%','hdi_97%']]*(32.1225-14.71125)+14.71125
            hdi_df.loc['threshold',:] = hdi_df.loc['threshold',:]*(32.1225-14.71125)+14.71125
        elif params['covariates'][0] == 'solar':
            sum_df.loc['threshold',['median','mean','sd','hdi_3%','hdi_97%']] = sum_df.loc['threshold',['median','mean','sd','hdi_3%','hdi_97%']]*(8846-279)+279 
            hdi_df.loc['threshold',:] = hdi_df.loc['threshold',:]*(8846-279)+279
        else:
            sum_df.loc['threshold',['median','mean','sd','hdi_3%','hdi_97%']] = sum_df.loc['threshold',['median','mean','sd','hdi_3%','hdi_97%']]*(206.8) 
            hdi_df.loc['threshold',:] = hdi_df.loc['threshold',:]*(206.8)  
    summary_df = pd.concat([sum_df, hdi_df], axis = 1) # concatenate credible intervals to summary tables
 #   summary_df.to_csv(path+results['filename']+'_summary.csv') # save summary tables
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
    loo = az.loo(results['inference'],results['model']) # calculate the loo metric
    dloo = az.loo(results['inference'],results['model'], scale ='deviance') # calculate the deviance
    waic = az.waic(results['inference'],results['model']) # calculate waic
    comp_df = pd.DataFrame((loo[0:5].values,waic[0:5].values,dloo[0:5].values), index = ['loo','waic','deviance'],  columns = ['metric','se','p','n_samples','n_datapoints']) # concatenate metrics into a dataframe
    comp_df.to_csv(path+results['filename']+'_modcomp.csv') # save metrics
    joblib.dump(loo, results['filename']+'_loo.pkl') # save results 
    comp_df = round(comp_df,3) # round the metrics
    return comp_df

# define visualization functions
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
    false_pos_rate, true_pos_rate, thresholds = roc_curve(y, y_prob) # get false positive rate, true positive rate and thresholds
    roc_auc = roc_auc_score(y,y_prob) # get auc roc score
    pyplot.plot(false_pos_rate, true_pos_rate, linewidth=5, label='ROC_AUC = %0.3f'% roc_auc) # generate roc curve
    pyplot.plot([0,1],[0,1], linewidth=5) # plot 1 to 1 true positive: false positive line
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
    pyplot.plot(recall,precision, linewidth=5, label='PR_AUC = %0.3f'% pr_auc) # plot precision-recall curve
    baseline = len(y[y==1]) / len(y) # get baseline expection, minority class proportion
    pyplot.plot([0,1],[baseline,baseline], linewidth=5) # plot the baseline
    pyplot.xlim([-0.01, 1]) # set x limit
    pyplot.ylim([0, 1.01]) # set y limit
    pyplot.legend(loc='upper right') # add legend
    pyplot.title('PR curve') # add title
    pyplot.ylabel('Precision') # add y axis title 
    pyplot.xlabel('Recall') # add x axis title

def trace_image(results,params, path):
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
    az.plot_trace(results['inference'], compact = True, var_names = params['variables'], # multiple chains on same plot, specific variables to plot
              divergences = params['divergence'], backend = 'matplotlib', backend_kwargs = {'tight_layout': True}) # specific divergence location, plot aesthetic and tight layout 
    fig = pyplot.gcf() # get last figure (trace plots)
    fig.savefig(path+results['filename']+'_trace.jpg', dpi=150) # save the trace plots as an image to flatten
    pyplot.close() # close the plot

def dbplot(data,y,cov='w',valid = False, legend = False):
    """
    Posterior predictive check. Plot y and mean predictive probability of y with 95% credible intervals

    Parameters
    ----------
    data : inference data
    y : labeled true values
    cov: variable to plot
    modname: model name 
    legend: boolean, add legend if True
    
    Returns
    -------
    None.

    """
    if type(data) == dict:
        # extract traces
        input_w = data[cov] # ppc for cue
        input_p = data['p']# ppc for probability of positive class
        p = np.median(input_p,0) # get median predictive probability of positive class
        w = np.median(input_w,0) # get median estimated cue
        input_pi = data['y']/y[:,1]
        
    else:
        input_w = data['posterior'][cov].to_numpy() # trace for cue
        input_p = data['posterior']['p'].to_numpy() # trace for probability of positive class
        p = np.median(input_p,0)  # get mean predictive probability of positive class
        p = np.median(p,0)  
        # get mean estimated cue
        w0 = np.median(input_w,0)
        w = np.median(w0, 0)
    
    # get the indices of cue values sorted in descending order
    idx = np.argsort(w)
   
    # plot vertical line for mean cue
    pyplot.vlines(np.median(w), 0, 1, color='k', ls='--', label ='median') 
    # get the 95% prediction intervals
    try:
        az.plot_hdi(w,input_pi,0.95,fill_kwargs={"alpha": 0.2, "color": "green", "label": "p 95% prediction intervals"})
        # get the 95% credible intervals
        az.plot_hdi(w,input_p,0.95,fill_kwargs={"alpha": 0.6, "color": "darkgreen", "label": "p 95% credible interval"})
    except:
        print(cov + ' no variation in median')
    
    # plot median predictive probability of positive class against median estimated cue
    pyplot.plot(w[idx], p[idx], color='black', lw=2, label="Median p")
    # plot observed values of y
    pyplot.scatter(w, np.random.normal(y[:,2], 0.001), marker='.', alpha=0.5, c = 'dimgrey',
                   label="Data", )
    pyplot.xlabel(cov)# add x axis title
    pyplot.ylabel('p', rotation=0) # add y axis title
    if valid == True:
        pyplot.title('posterior predictive check valid')
    else:
        pyplot.title('posterior predictive check')
    if legend == True:
        pyplot.legend(fontsize=8, loc='center left', framealpha= 0.5) # add legend
    
    
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
    pyplot.subplot(2,4,1) # designate subplot
    confusion_mat(train_y,train_pred, LABELS = LABELS, normalize = 'true') # plot confusion matrix with proportions
    pyplot.subplot(2,4,2)
    confusion_mat(train_y,train_pred, LABELS = LABELS, normalize = None) # plot confusion matrix with counts
    pyplot.subplot(2, 4, 3) 
    roc_plot(train_y, train_prob) # plot roc curve
    pyplot.subplot(2, 4, 4)  
    pr_plot(train_y, train_prob) # plot precision recall curve
    pyplot.subplot(2,4,5) # designate subplot
    confusion_mat(valid_y,valid_pred, LABELS = LABELS, normalize = 'true') # plot confusion matrix with proportions
    pyplot.subplot(2,4,6)
    confusion_mat(valid_y,valid_pred, LABELS = LABELS, normalize = None) # plot confusion matrix with counts
    pyplot.subplot(2, 4, 7) 
    roc_plot(valid_y, valid_prob) # plot roc curve
    pyplot.subplot(2, 4, 8)  
    pr_plot(valid_y, valid_prob) # plot precision recall curve
   # pyplot.suptitle(title)
    fig.tight_layout() # set tight layout
    return fig

def prob_pred_bern(results,train_y,valid_y, path, save = True):
    """
    Generate class probability and predictions and save into file

    Parameters
    ----------
    results : result file from mcmc
    y: observed values
    path: path to save output

    Returns
    -------
    train_prob : training class probability
    train_pred : training class prediction
    valid_prob : validation class probability
    valid_pred : validation class prediction

    """
    train_prob = np.median(results['ppc']['p'],axis=0) # get mean posterior predictions
    train_pred = np.random.binomial(n=train_y[:,1].astype('int32'), p=train_prob) # generate predictions
    train_predictions = np.column_stack((train_pred,train_y[:,1], train_prob)) # organize into same array
    
    valid_prob = np.median(results['vppc']['p'],axis=0) # get mean posterior predictions
    valid_pred = np.random.binomial(n=valid_y[:,1].astype('int32'), p=valid_prob) # generate predictions
    valid_predictions = np.column_stack((valid_pred,valid_y[:,1], valid_prob)) # organize into same array
    
    obs = np.concatenate((train_y[:,0], valid_y[:,0]), axis= 0) # organize observations
    probs= np.concatenate((train_prob, valid_prob), axis= 0) # organize probabilities
    pred = np.concatenate((train_pred, valid_pred), axis= 0) # organize predicitons

    binom_pred = pd.DataFrame({'dataset': ['train']*len(train_y) + ['valid']*len(valid_y),
                             'obs': obs, 'prob': probs, 'pred': pred}) # organize binomial predictions
    
    train_tab = bern_pred_gen(train_y,train_predictions,'train') # generate bernoulli predictions for training data
    valid_tab = bern_pred_gen(valid_y,valid_predictions,'valid') # generate bernoulli predictions for validation data
    
    bern_pred = pd.concat([train_tab, valid_tab], axis = 0) # put training nad validation predictions into same dataframe
    
    # to save or not
    if save == True:
        binom_pred.to_csv(path+results['filename'] + 'binom_predictions.csv')
        bern_pred.to_csv(path+results['filename'] + 'bern_predictions.csv')
    return train_tab, valid_tab, binom_pred

def pymc_vis_split(results, train_y, valid_y, params, path):
    """
    visualize, organize and save pymc results into a single pdf

    Parameters
    ----------
    results : dictionary of results with model, trace, posterior predictions, features and filename
    y : observed y
    variables : parameters of interest
    path : directory to save results
    
    Returns
    -------
    train_pred : predicted classes for training data 
    train_prob : predicted probability for positive class for training data 
    valid_pred : predicted classes for validation data 
    valid_prob : predicted probability for positive cwlass for validation data 

    """
    sum_df, hdi_df = summary_tab(results, params, path) # generate summary table and credible interval table 
    comp_df = comp_tab(results, path) # generate the commonly used model selection metrics 
    trace_image(results, params, path) # generate trace image
    traceplt = Image.open(path+results['filename']+'_trace.jpg') # open trace plot images
    
    #train_prob, train_pred, valid_prob, valid_pred = prob_pred_split(results, train_y, valid_y, path)
    train_tab, valid_tab, binom_pred = prob_pred_bern(results, train_y, valid_y, path)
    
    class_rep = class_rep_split(train_tab['obs'],train_tab['pred'],
                                valid_tab['obs'],valid_tab['pred'])
    
    
    # generate pdfs
    with PdfPages(path+results['filename']+'_results.pdf') as pdf:
        pyplot.figure(figsize=(8.5, 9)) # assign figure size 
        pyplot.imshow(traceplt) # plot image
        pyplot.axis('tight') # show all data
        pyplot.axis('off') # turn off axis labels and lines
        pdf.savefig(dpi=150) # save figure
        pyplot.close() # close page
        # generate forest/ caterpillar plots
        az.plot_forest(results['inference'], var_names = params['variables'], 
                    combined = True,hdi_prob = 0.95, figsize=(6,3))
        pyplot.grid() # add grid to plot
        pdf.savefig() # save figure
        pyplot.close() # close page
        
        az.plot_forest(results['inference'], var_names = params['variables'], 
                    combined = True,hdi_prob = 0.90, figsize=(6,3))
        pyplot.grid() # add grid to plot
        pdf.savefig() # save figure
        pyplot.close() # close page
        
        # generate posterior predictive check plot
        name_var = params['name_var'][2:]
        hcol = len(name_var)
        fig,ax = pyplot.subplots(hcol,2,figsize =(8,3*hcol))
        for i in range(hcol):
            pyplot.subplot(hcol,2,2*i+1)
            dbplot(results['ppc'],train_y, cov=name_var[i],valid = False)        
            pyplot.subplot(hcol,2,2*i+2)
            dbplot(results['vppc'],valid_y, cov=name_var[i],valid = True)
        fig.tight_layout()
        pdf.savefig() # save figure
        pyplot.close() # close page
                
        # generate confusion matrices, roc curves and pr curves  
        pb_plot = plot_bayes_split(train_tab['obs'],train_tab['pred'], train_tab['pred_prob'], 
                                   valid_tab['obs'],valid_tab['pred'], valid_tab['pred_prob'])
        pdf.savefig(pb_plot) # save figure
        pyplot.close() # close page
        
        # generate subplots to organize table data 
        fig, ax = pyplot.subplots(figsize=(8.5,12),nrows=4)
        # format classification report into graphic
        metrictab = ax[0].table(cellText=class_rep.values,colLabels = class_rep.columns, rowLabels=class_rep.index,loc='center')
        metrictab.auto_set_font_size(False) # trun off auto font
        metrictab.set_fontsize(9) # set font size to 9
        ax[0].axis('tight') 
        ax[0].axis('off')

        # format summary table into graphic
        sumtab = ax[1].table(cellText=sum_df.values,colLabels = sum_df.columns, rowLabels=sum_df.index,loc='center')
        sumtab.auto_set_font_size(False)
        sumtab.set_fontsize(9)
        ax[1].axis('tight')
        ax[1].axis('off')
        
        # format credible interval table into graphic
        hditab = ax[2].table(cellText=hdi_df.values,colLabels = hdi_df.columns, rowLabels=hdi_df.index,loc='center')
        hditab.auto_set_font_size(False)
        hditab.set_fontsize(9)
        ax[2].axis('tight')
        ax[2].axis('off')
        
        # format metric table into graphic
        comptab = ax[3].table(cellText=comp_df.values,colLabels=comp_df.columns,rowLabels=comp_df.index, loc='center')
        comptab.auto_set_font_size(False)
        comptab.set_fontsize(9)
        ax[3].axis('tight')
        ax[3].axis('off')
        pdf.savefig() # save figure
        pyplot.close() # close page
    return train_tab, valid_tab, binom_pred
     
def bern_pred_gen(y, ypred, ds):
    obs = []
    probs = []
    preds = []
    pprobs = []
    for i in range(len(y)):    
        ones= np.ones(int(y[i, 0]))
        zeros= np.zeros(int(y[i,1]-y[i,0]))
        obs_oz = np.concatenate([ones,zeros])
        np.random.shuffle(obs_oz)
    
        proba = np.repeat(y[i,2], int(y[i,1]))
        
        pones= np.ones(int(ypred[i, 0]))
        pzeros= np.zeros(int(ypred[i,1]-ypred[i,0]))
        pred_oz = np.concatenate([pones,pzeros])
        np.random.shuffle(pred_oz)
        
        pproba = np.repeat(ypred[i,2], int(ypred[i,1]))
        
        obs += [obs_oz]
        probs += [proba]
        preds += [pred_oz]
        pprobs += [pproba]
    
    obs = np.concatenate(obs)
    probs = np.concatenate(probs)
    preds = np.concatenate(preds)
    pprobs = np.concatenate(pprobs)
    
    bern_tab = pd.DataFrame({'dataset': [ds]*len(obs),
                             'obs': obs, 'obs_prob': probs, 'pred': preds, 'pred_prob': pprobs})
    
    return bern_tab

def prob_pred_split(results,train_y,valid_y, path):
    """
    Generate class probability and predictions and save into file

    Parameters
    ----------
    results : result file from mcmc
    y: observed values
    path: path to save output

    Returns
    -------
    train_prob : training class probability
    train_pred : training class prediction
    valid_prob : validation class probability
    valid_pred : validation class prediction

    """
    train_y = train_y.astype('int32')
    valid_y = valid_y.astype('int32')
    
    train_prob = results['ppc']['p'].mean(axis=0) # get mean posterior predictions
    train_pred = np.random.binomial(n=train_y[:,1], p=train_prob) # generate predictions
    
    valid_prob = results['vppc']['p'].mean(axis=0) # get mean posterior predictions
    valid_pred = np.random.binomial(n=valid_y[:,1], p=valid_prob) # generate predictions
    
    obs = np.concatenate((train_y[:,0], valid_y[:,0]), axis= 0)
    probs= np.concatenate((train_prob, valid_prob), axis= 0)
    pred = np.concatenate((train_pred, valid_pred), axis= 0)
    
    pred_tab = pd.DataFrame({'dataset': ['train']*len(train_y) + ['valid']*len(valid_y),
                             'obs': obs, 'prob': probs, 'pred': pred})
    pred_tab.to_csv(path+results['filename'] + '_predictions.csv')
    return train_prob, train_pred, valid_prob, valid_pred

def output_metrics(bern_tab, binom_tab):
    """
    Calculates five metrics to assess predictive performance: log loss, accuracy, roc auc, pr auc and f1-score

    Parameters
    ----------
    y : true observed target values
    y_pred : predicted class
    y_prob : positive class prediction probability

    Returns
    -------
    list of metrics

    """
    y = bern_tab['obs']
    y_pred = bern_tab['pred']
    y_prob = bern_tab['pred_prob']
    loss = log_loss(y, y_prob)
    acc = accuracy_score(y,y_pred)
    prec = precision_score(y,y_pred)
    rc = recall_score(y,y_pred)
    pr = average_precision_score(y, y_prob)
    roc = roc_auc_score(y, y_prob)
    f1 = f1_score(y,y_pred)
    gmean = geometric_mean_score(y, y_pred)
    rmse = mean_squared_error(binom_tab['obs'], binom_tab['pred'], squared = False)
    return [loss,acc,prec,rc,roc,pr,f1,gmean,rmse]

def outtab(train_tab, valid_tab, binom_tab, filename, path):
    output = DataFrame(columns=['loss','acc','prec','rec','roc','pr','f1','gmean','rmse'])
    train_results = output_metrics(train_tab, binom_tab[binom_tab['dataset'] == 'train'])
    output.loc[len(output.index)] = train_results
    valid_results = output_metrics(valid_tab, binom_tab[binom_tab['dataset']=='valid'])
    output.loc[len(output.index)] = valid_results
    output.index = ['train','valid']
    output.to_csv(path+ filename +'_output_metrics.csv')
    return output

def wrapper_fun(data, params, species, covariates, relation = None, direction = None, threshold = True):
    """
    wrapper function to run Bayesian logistic regression cue models

    Parameters
    ----------
    data : phenology data set with target variable 
    params : model parameters
    species : focal species being modelled
    covariates: covariates of interest
    threshold: Boolean indicating whether the model incorporates thresholds or not
    
    Raises
    ------
    Exception
        print error is invalid covariate is used

    Returns
    -------
    None.

    """
    first_time = time.time()
    params['species'] = species
    params['covariates'] = covariates
    params['threshold'] = threshold
    datasub, X, y = data_gen(data, params)
    train_X, valid_X, train_y, valid_y = train_test_data(X,y,datasub,gss)
    params['relation'] = relation
    params['direction'] = direction
        
    if len(params['covariates']) ==1:
        params['name_var'] = ['y','p','w','x']
        params['variables'] = ['~w','~p','~x']
        model = single_model(train_X,train_y,params)
    elif len(params['covariates'])==2:
        params['name_var'] = ['y','p','p0','p1','w0','w1','x0','x1']
        params['variables'] = ['~p','~p0','~p1','~w0','~w1','~x0','~x1']
        

        if 'rain' in params['covariates'] and 'drought' in params['covariates']:
            params['sequential'] = True
        elif 'solar' in params['covariates'] and 'temp' in params['covariates']:
            params['sequential'] = True
        else:
            params['sequential'] = False
        model = double_model(train_X,train_y,params)
    else:
        raise Exception ('invalid covariates entered')
    results = run_model(train_X,train_y,model,params, save=params['save'])
    print('training took ' + str(round((time.time()-first_time)/60,3)) + ' mins')
    
    # make predictions on validation set
    with results['model']:
        pm.set_data({'X': valid_X,'n': valid_y[:,1].astype('int32')})
        results['vppc'] = pm.sample_posterior_predictive(results['trace'],
                                              var_names =params['name_var'])
# start here
    # save model output
    save_time = time.time()
    joblib.dump(results,results['filename']+'.pkl')    
    print('model saving took: ' + str((time.time()-save_time)/60) + ' minutes') # output how long saving takes
    
    # visualize results 
    train_tab, valid_tab, binom_tab = pymc_vis_split(results, train_y, valid_y, params, path+params['species']+'\\')
    print('visualization done')    
    
    # generate performance metrics
    output = outtab(train_tab, valid_tab, binom_tab, results['filename'], path)
    print('took ' + str((time.time()-first_time)/60) + 'minutes')
    gc.collect() # dump model from memory
    gc.collect() # dump model from memory
    gc.collect() # dump model from memory
    return output
   
# assign parameters
params = {'species': 'Ambora',
          'lower_lag':0,
          'upper_lag':110,
          'upper_lb': 100,
          'lower_lag0': 0,
          'upper_lag0':110,
          'lower_lag1': 0,
          'upper_lag1':110,
          'ni': 1000,
          'nb':1000,
          'nc': 4,
          'nt':1,
          'covariates': ['temp','solar'],
          'alpha_sd': 10,
          'beta_sd': 10,
          'threshold_sd':10,
          'nuts_target':0.95,
          'variables':['~w','~p'],
          'threshold': True,
          'relation': 'double_neg',
          'direction': ['negative','negative'],
          'sequential':True,
          'save': 'full',
          'divergence':None,
          'name_var': ['y','w','p','x0','x1'],
          'mtype': 'chen'}

#run single cue models
# wrapper_fun(data, params, 'Rahiaka', ['rain'],False)
# wrapper_fun(data, params, 'Rahiaka', ['temp'],False)
# wrapper_fun(data, params, 'Rahiaka', ['solar'],False)

# wrapper_fun(data, params, 'Ravenala', ['rain'],True)
# wrapper_fun(data, params, 'Ravenala', ['temp'],True)
# wrapper_fun(data, params, 'Ravenala', ['solar'],True)


# to run
params['ni'] = 50000
params['nb'] = 50000
wrapper_fun(data, params, 'Ambora', ['temp','solar'], 'double_neg', ['negative','negative'], True)
wrapper_fun(data, params, 'Rahiaka', ['solar','temp'], 'double_posneg_os', ['positive','negative'], True)
wrapper_fun(data, params, 'Rahiaka', ['rain','solar'], 'double_pos50', ['positive','positive'], True)
wrapper_fun(data, params, 'Rahiaka', ['drought','solar'], 'double_neg50', ['negative','negative'], True)
params['ni'] = 100000
params['nb'] = 100000
wrapper_fun(data, params, 'Ampaly', ['rain','temp'], 'double_posneg100', ['positive','negative'], True)


wrapper_fun(data, params, 'Lafa', ['solar','temp'], 'double_neg50', ['negative','positive'], True)
wrapper_fun(data, params, 'Lafa', ['rain','drought'], 'double_posneg50', ['positive','negative'], True)
wrapper_fun(data, params, 'Rahiaka', ['temp','solar'], 'double_posneg50', ['positive','negative'], True)
wrapper_fun(data, params, 'Ambora', ['rain','solar'], 'double_pos50', ['positive','positive'], True)

params['ni'] = 50000
params['nb'] = 50000
wrapper_fun(data, params, 'Ampaly', ['rain','solar'], 'double_neg50', ['negative','negative'], True)
wrapper_fun(data, params, 'Rahiaka', ['rain','temp'], 'double_posneg50', ['positive','negative'], True)
wrapper_fun(data, params, 'Rahiaka', ['rain','solar'], 'double_posneg50', ['positive','negative'], True)
wrapper_fun(data, params, 'Rahiaka', ['temp','solar'], 'double_posneg50', ['positive','negative'], True)
params['ni'] = 20000
params['nb'] = 20000
wrapper_fun(data, params, 'Ambora', ['temp','solar'], 'double_posneg', ['positive','negative'], True)


wrapper_fun(data, params, 'Ampaly', ['rain','solar'], 'double_pos', ['positive','positive'], True)
wrapper_fun(data, params, 'Ampaly', ['rain','temp'], 'double_negpos', ['negative','positive'], True)
wrapper_fun(data, params, 'Ampaly', ['drought','rain'], 'double_negpos', ['negative','positive'], True)
wrapper_fun(data, params, 'Ampaly', ['rain','drought'], 'double_posneg', ['positive','negative'], True)
wrapper_fun(data, params, 'Ampaly', ['rain','temp'], 'double_posneg', ['positive','negative'], True)
wrapper_fun(data, params, 'Ampaly', ['rain','solar'], 'double_posneg', ['positive','negative'], True)
wrapper_fun(data, params, 'Ambora', ['rain','solar'], 'double_posneg', ['positive','negative'], True)
wrapper_fun(data, params, 'Ambora', ['drought','solar'], 'double_neg', ['negative','negative'], True)
wrapper_fun(data, params, 'Ambora', ['temp','solar'], 'double_neg', ['negative','negative'], True)
wrapper_fun(data, params, 'Ambora', ['solar','temp'], 'double_neg', ['negative','negative'], True)
wrapper_fun(data, params, 'Rahiaka', ['solar','temp'], 'double_negpos', ['negative','positive'], True)
wrapper_fun(data, params, 'Rahiaka', ['temp','solar'], 'double_negpos', ['negative','positive'], True)
wrapper_fun(data, params, 'Rahiaka', ['temp','solar'], 'double_posneg', ['positive','negative'], True)
wrapper_fun(data, params, 'Rahiaka', ['rain','solar'], 'double_posneg', ['positive','negative'], True)
wrapper_fun(data, params, 'Rahiaka', ['solar','temp'], 'double_posneg', ['positive','negative'], True)
wrapper_fun(data, params, 'Rahiaka', ['drought','solar'], 'double_neg', ['negative','negative'], True)
wrapper_fun(data, params, 'Rahiaka', ['solar','temp'], 'double_neg', ['negative','negative'], True)
wrapper_fun(data, params, 'Rahiaka', ['temp','solar'], 'double_neg', ['negative','negative'], True)

# need to run
wrapper_fun(data, params, 'Ambora', ['temp','solar'], 'double_posneg', ['positive','negative'], True)

wrapper_fun(data, params, 'Rahiaka', ['drought','temp'], 'double_neg', ['negative','negative'], True)
wrapper_fun(data, params, 'Ravenala', ['solar','temp'], 'double_pos', ['positive','positive'], True)

wrapper_fun(data, params, 'Rahiaka', ['temp'], 'single_neg', ['negative'], True)
wrapper_fun(data, params, 'Vatsilana', ['temp'], 'single_pos', ['positive'], True)
wrapper_fun(data, params, 'Vatsilana', ['rain'], 'single_neg', ['negative'], True)


wrapper_fun(data, params, 'Ampaly', ['solar','temp'], 'double_negpos', ['negative','positive'], True)


wrapper_fun(data, params, 'Ampaly', ['temp','solar'], 'double_pos', ['positive','positive'], True)
wrapper_fun(data, params, 'Ampaly', ['temp','solar'], 'double_negpos', ['negative','positive'], True)

wrapper_fun(data, params, 'Ambora', ['rain','temp'], 'double_posneg', ['positive','negative'], True)

wrapper_fun(data, params, 'Lafa', ['rain','solar'], True)
wrapper_fun(data, params, 'Ampaly', ['solar','temp'], True)
params['relation']  = 'double_pos50'
params['direction'] = ['positive','positive']
wrapper_fun(data, params, 'Lafa', ['rain','solar'], True)
wrapper_fun(data, params, 'Lafa', ['temp','solar'], True)
params['relation'] = 'double_neg50'
params['direction'] = ['negative','negative']
wrapper_fun(data, params, 'Lafa', ['temp','solar'], True)
wrapper_fun(data, params, 'Lafa', ['solar','temp'], True)



wrapper_fun(data, params, 'Rahiaka', ['solar','temp'], True)
wrapper_fun(data, params, 'Rahiaka', ['rain','solar'], True)
wrapper_fun(data, params, 'Rahiaka', ['rain','drought'], True)

params['relation'] = 'double_negpos50'
params['direction'] = ['negative','positive']
wrapper_fun(data, params, 'Lafa', ['solar','temp'], True)
wrapper_fun(data, params, 'Lafa', ['solar','temp'], True)
wrapper_fun(data, params, 'Lafa', ['temp','solar'], True)


wrapper_fun(data, params, 'Ampaly', ['rain','solar'], True)
wrapper_fun(data, params, 'Ampaly', ['rain','temp'], True)
params['relation']  = 'double_pos'
params['direction'] = ['positive','positive']
wrapper_fun(data, params, 'Ampaly', ['temp','solar'], True)
params['relation'] = 'double_negpos'
params['direction'] = ['negative','positive']
wrapper_fun(data, params, 'Ampaly', ['rain','solar'], True)
wrapper_fun(data, params, 'Ampaly', ['rain','temp'], True)
params['relation'] = 'double_posneg'
params['direction'] = ['positive','negative']
wrapper_fun(data, params, 'Ampaly', ['temp','solar'], True)


wrapper_fun(data, params, 'Ravenala', ['solar','temp'], True)
params['relation'] = 'double_negpos50'
params['direction'] = ['negative','positive']
wrapper_fun(data, params, 'Ravenala', ['solar','temp'], True)

params['lower_lag0'] = 12
params['upper_lag0'] = 109
params['relation'] = 'double_posneg55x102_12x109'
params['direction'] = ['positive','negative']
wrapper_fun(data, params, 'Lafa', ['rain','solar'], True)


params['ni'] = 50000
params['nb'] = 50000
wrapper_fun(data, params, 'Ravenala', ['rain','temp'], True)



wrapper_fun(data, params, 'Rahiaka', ['rain','drought'], True)
params['relation'] = 'add'
wrapper_fun(data, params, 'Rahiaka', ['rain','drought'], True)

wrapper_fun(data, params, 'Ambora', ['rain','temp'], True) 
wrapper_fun(data, params, 'Ambora', ['rain','solar'], True) 
wrapper_fun(data, params, 'Ambora', ['solar','temp'], True)
wrapper_fun(data, params, 'Ambora', ['temp','solar'], True)

wrapper_fun(data, params, 'Ampaly', ['solar','temp'], True)



wrapper_fun(data, params, 'Rahiaka', ['solar','temp'], True)
wrapper_fun(data, params, 'Rahiaka', ['rain','temp'], True) 
wrapper_fun(data, params, 'Rahiaka', ['temp','solar'], True)
wrapper_fun(data, params, 'Rahiaka', ['rain','drought'], True)
wrapper_fun(data, params, 'Rahiaka', ['rain','solar'], True) 

# run double cue models
wrapper_fun(data, params, 'Rahiaka', ['solarcloud'],True)


#wrapper_fun(data, params, 'Rahiaka', ['rain','solarcloud'], True) 
wrapper_fun(data, params, 'Rahiaka', ['solar','temp'], True)
#wrapper_fun(data, params, 'Rahiaka', ['solarcloud','temp'], True)

#wrapper_fun(data, params, 'Rahiaka', ['temp','solarcloud'], True)
wrapper_fun(data, params, 'Ravenala', ['temp','solar'], True)

results = joblib.load('abc')
b = np.array(results['inference']['posterior']['var']).flatten()



################################
### Compare converged models ###
################################
def compare_plots(comp_dict,path,species):
    """
    generate and loo loo comparison table and plots

    Parameters
    ----------
    comp_dict : model traces to compare
    path : path to store results
    species : focaal species

    Returns
    -------
    None.

    """
    filename = path+species+'\\'+species # generate file name
    comp_df = az.compare(comp_dict, 'loo') # generate comparison table using loo
    comp_df.to_csv(filename + '_loo_comparison.csv') # save comparison tables
    comp_df.iloc[:,1:7] = round(comp_df.iloc[:,1:7],3)  # round the comparison table for visualization
    fig, ax = pyplot.subplots(figsize = (6,6), nrows=2) # generate subplots
    # format summary table into graphic
    metrictab = ax[0].table(cellText=comp_df.values,colLabels = comp_df.columns, rowLabels=comp_df.index,
                              cellLoc = 'center', loc='center') # add comparison table to first subplot
    metrictab.auto_set_font_size(False) # trun off auto font # turn off auto font
    metrictab.set_fontsize(8) # set font size to 9
    metrictab.auto_set_column_width(col=list(range(len(comp_df.columns)))) # adjust width to column content
    ax[0].axis('tight') 
    ax[0].axis('off') # turn graph axis off 
      
    # plot comparison table in second subplot
    az.plot_compare(comp_df, plot_ic_diff = False,ax=ax[1])
    pyplot.ylabel('ranked models') # add y axis label
    pyplot.xlabel('loo (log)') # add x axis label
    pyplot.legend(labels = ['loo without \nparameter \npenalty', 'best loo', 'loo'],
                    loc = 'center left', bbox_to_anchor=(1,0.5)) # generate legend
    pyplot.title(species + ' model comparison') # add title
    fig.tight_layout() # tighten layout
    fig.savefig(filename + '_model_comparison.png', dpi=150) # save pdf
    pyplot.close() # close the plot

# load trace plots
t1 = joblib.load('Ravenala_rain_wright_single_pos_loo.pkl')  
t2 = joblib.load('Ravenala_solar_wright_single_pos_loo.pkl')
t3 = joblib.load('Ravenala_temp_wright_single_pos_loo.pkl')
t4 = joblib.load('ambora_rain_solar_cumsum_prod_trace.pkl')

comp_dict = {'rain':joblib.load('Ravenala_rain_wright_single_pos_loo.pkl'),
            'cool':joblib.load('Ravenala_temp_wright_single_neg_loo.pkl'),
            'warm':joblib.load('Ravenala_temp_wright_single_pos_loo.pkl'),
            'drought':joblib.load('Ravenala_rain_wright_single_neg_loo.pkl'),
            'high light':joblib.load('Ravenala_solar_wright_single_pos_loo.pkl'),
            'low light':joblib.load('Ravenala_solar_wright_single_neg_loo.pkl'),
            'cool x low light':joblib.load('Ravenala_temp_solar_wright_double_neg_loo.pkl'),
            'drought x cool':joblib.load('Ravenala_rain_temp_wright_double_neg_loo.pkl'),
            'drought x low light':joblib.load('Ravenala_rain_solar_wright_double_neg_loo.pkl'),
            'drought x high light':joblib.load('Ravenala_rain_solar_wright_double_negpos50_loo.pkl'),
            'drought x rain':joblib.load('Ravenala_rain_drought_wright_double_negpos_loo.pkl'),
            'drought x warm':joblib.load('Ravenala_rain_temp_wright_double_negpos_loo.pkl'),
            'low light x warm':joblib.load('Ravenala_solar_temp_wright_double_negpos50_loo.pkl')}

comp_dict = {'cool':joblib.load('Ampaly_temp_wright_single_neg_loo.pkl'),
            'drought':joblib.load('Ampaly_rain_wright_single_neg_loo.pkl'),
            'low light':joblib.load('Ampaly_solar_wright_single_neg_loo.pkl'),
            'high light': joblib.load('Ampaly_solar_wright_single_pos_loo.pkl'),
            'warm 0x10':joblib.load('Ampaly_temp_wright_single_pos0x10_loo.pkl'),
            'warm 10x20':joblib.load('Ampaly_temp_wright_single_pos10x20_loo.pkl'),
            'low light x cool':joblib.load('Ampaly_solar_temp_wright_double_neg_loo.pkl')}

comp_dict = {'rain':joblib.load('Lafa_rain_wright_single_pos_loo.pkl'),
            'warm':joblib.load('Lafa_temp_wright_single_pos_loo.pkl'),
            'cool':joblib.load('Lafa_temp_wright_single_neg0x32_loo.pkl'),
            'high light': joblib.load('Lafa_solar_wright_single_pos_loo.pkl'),
            'low light':joblib.load('Lafa_solar_wright_single_neg_loo.pkl'),
            'rain x cool':joblib.load('Lafa_rain_temp_wright_double_posneg_loo.pkl')}

comp_dict = {'cool':joblib.load('Ambora_temp_wright_single_neg_loo.pkl'),
            'drought':joblib.load('Ambora_rain_wright_single_neg_loo.pkl'),
            'high light':joblib.load('Ambora_solar_wright_single_pos_loo.pkl'),
            'warm': joblib.load('Ambora_temp_wright_single_pos_loo.pkl'),
            'drought x cool':joblib.load('Ambora_rain_temp_wright_double_neg_loo.pkl'),
            'high light x warm':joblib.load('Ambora_solar_temp_wright_double_pos50_loo.pkl'),
            'high light x cool':joblib.load('Ambora_solar_temp_wright_double_posneg50_loo.pkl')}

# organize trace plots into dictionary
comp_dict = {'low light':t1,
             'high light':t2,
             'warm':t3,
             'rain x solar':t4}

comp_dict = {'rain':loo1,
             'temp':loo2,}
# run plot comparisons 
compare_plots(comp_dict, path, 'Ambora')

# check the loo values and pareto statistics for models with pareto warnings
a = az.loo(t1,pointwise = True)
len([idx for idx, val in enumerate(a.pareto_k) if val > 0.5])

pred = read_csv('C:\\Users\\Jannet\\Documents\\Dissertation\\results\\BGLM_flower\\Ravenala\\Ravenala_temp_wright_single_negbinom_predictions.csv', header= 0, index_col = 0)
datasub, X, y = data_gen(data, params)
for train_index,valid_index in gss.split(datasub['Date'].values, y, groups = datasub['year']):
        train_ds, valid_ds = datasub.iloc[train_index,:], datasub.iloc[valid_index,:]
        
newdata = pd.concat([train_ds, valid_ds])
newdata = pd.concat([newdata, pred], axis= 1)
newdata.to_csv('ravenala_coolmodel_preds.csv')


train_ds['dataset'] = 
valid_ds['dataset'] = 'valid'

pred['date'] = pd.concat([train_ds, valid_ds])
def train_test_data(X,y, datasub,gss):
    """
    generate training and testing data

    Parameters
    ----------
    X : focal specise covariates
    datasub : focal species dataset
    gss : group split

    Returns
    -------
    train_list : list of training indices by fold 
    valid_list : list of valid indices by fold

    """
    #train_list = [] # create empty list to populate train fold indices
    #valid_list = [] # create empty list to populate valid fold indices
    # for each indicies in kfold
    for train_index,valid_index in gss.split(X[0,0,:,0], y, groups = datasub['year']):
        train_X, valid_X = X[:,:,train_index,:], X[:,:,valid_index,:]
        train_y, valid_y = y[train_index,:], y[valid_index,:]
    return train_X, valid_X, train_y, valid_y


files = ['Rahiaka_temp_solar_wright_double_pos',
         'Rahiaka_solar_temp_wright_double_negpos',
         'Rahiaka_drought_temp_wright_double_negpos100',
         'Rahiaka_temp_solar_wright_double_negpos',
         'Rahiaka_temp_wright_single_pos',
         'Rahiaka_solar_wright_single_pos',
         'Rahiaka_solar_temp_wright_double_posneg']

params['species'] = 'Ravenala'
params['covariates'] = ['solar']
results = joblib.load(files[10]+'.pkl')
datasub, X, y = data_gen(data, params)
_, _, train_y, valid_y = train_test_data(X, y, datasub, gss)

filelist = []
for file in glob.glob("Lafa*wright_double*.pkl"):
    filelist.append(file)
filelist = [item for item in filelist if 'loo' not in item]

cov_list = [['rain','solar'],['rain','temp'],['rain','temp'],['solar','temp'],['solar','temp']]
for i in range(len(filelist)):
    params['covariates'] = cov_list[i]
    datasub, X, y = data_gen(data, params)
    results =joblib.load(filelist[i])
    train_X, valid_X, train_y, valid_y = train_test_data(X,y,datasub,gss)
    pymc_vis_split(results, train_y, valid_y, params, path + 'Lafa\\')
    
# results['vppc']['p'] = results['vppc']['p'][np.r_[30000:50000,80000:100000, 130000:150000, 180000:200000]] 
# results['ppc']['p'] = results['ppc']['p'][np.r_[30000:50000,80000:100000, 130000:150000, 180000:200000]] 


for file in files:
    results =joblib.load(file+'.pkl')
    datasub, X, y = data_gen(data, params)
    _, _, train_y, valid_y = train_test_data(X, y, datasub, gss)
    out_train, out_val = prob_pred_bernoulli(results, train_y,valid_y, path+'predictions\\', file)
    
    
def prob_pred_bernoulli(results, train_y, valid_y, path, filename):
    '''
    generate posterior predictions from the posterior probabilities and calculate performance metrics

    Parameters
    ----------
    results : model results 
    train_y : training targets
    valid_y : validation targets
    path : path to save the output metrics
    filename : filename

    Returns
    -------
    out_train : posterior prediciton training metrics 
    out_valid : posterior prediction validation metrics

    '''
    start_time = time.time() # get start time to compute running time
    train_prob = results['ppc']['p'] # extract the posterior predictions
    train_prob = train_prob[-80000:,:]
    
    train_list = [] # create empty list for training predictions
    for i in range(len(train_prob)):
        train_pred = np.random.binomial(n=train_y[:,1].astype('int32'), p=train_prob[i,:]) # generate predictions
        train_list.append(train_pred) # add prediction to the list
    train_predictions = np.stack(train_list) # turn into array

    valid_prob = results['vppc']['p'] # get mean posterior predictions
    valid_prob = train_prob[-valid:,:]
    
    valid_list = [] # create empty list for validation rpedictions
    for i in range(len(valid_prob)): 
        valid_pred = np.random.binomial(n=valid_y[:,1].astype('int32'), p=valid_prob[i,:]) # generate predictions
        valid_list += [valid_pred] # add predictions to the validation list
    valid_predictions = np.stack(valid_list) # turn into array
     
    # generate 3d array with probablities and predictions
    train_3d = np.dstack((train_prob, train_predictions)) 
    valid_3d = np.dstack((valid_prob, valid_predictions))
    
    # Generating the observed bernoulli positive/negative classes from their flowering probability  
    train_by, train_bprob = bern_gen(train_y[:, 0], train_y[:, 1], train_y[:, 2]) # for training
    train_observed = np.column_stack((train_by, train_bprob)) # put positive and negative observations and probablity of flowering together
    valid_by, valid_bprob = bern_gen(valid_y[:, 0], valid_y[:, 1], valid_y[:, 2]) # for validation
    valid_observed = np.column_stack((valid_by, valid_bprob)) # put positive and negative observations and probablity of flowering together
    
    new_time = time.time() 
    print('generated the binomial predictions & formatted bern & binom observations' + str((new_time-start_time)/60) + ' mins')
   
    # create empty dataframe for generating the bern predictions
    train_pred_by = []
    train_pred_bprob = []
    valid_pred_by = []
    valid_pred_bprob = []
    # generate bern predictions for training
    for i in range(train_3d.shape[0]): # for each posterior prediciton
        tb, tp = bern_gen(train_3d[i,:,1], train_y[:,1],train_3d[i,:,0]) # generate the bernoulli prediction
        train_pred_by +=[tb] # add to list
        train_pred_bprob +=[tp] # add to list
    train_predictions_by = np.stack(train_pred_by) # format
    train_predictions_bprob = np.stack(train_pred_bprob) # format
    train_pred_3d = np.dstack((train_predictions_bprob, train_predictions_by)) # put into 3d array
    print('generated bern train predictions ' + str((time.time()-new_time)/60) + ' mins') # get runtime
    new_time = time.time() 
    # generate bern predicitons for validation
    for i in range(valid_3d.shape[0]): # for each posterior prediction
        vb, vp = bern_gen(valid_3d[i,:,1], valid_y[:,1],valid_3d[i,:,0]) # generate the bern prediction
        valid_pred_by +=[vb] # add to list
        valid_pred_bprob +=[vp] # add to list
    valid_predictions_by = np.stack(valid_pred_by) # format
    valid_predictions_bprob = np.stack(valid_pred_bprob) # format
    valid_pred_3d = np.dstack((valid_predictions_bprob,valid_predictions_by)) # put into 3d array
    print('generated bern valid predictions ' + str((time.time() -new_time)/60) + ' mins') # get runtime
    
    # put raw predictions into a diction
    pred_dict = {'binom_train_obs': train_y,
                 'binom_valid_obs': valid_y,
                 'binom_train_pred': train_3d,
                 'binom_valid_pred': valid_3d,
                 'bern_train_obs': train_observed,
                 'bern_valid_obs': valid_observed,
                 'bern_train_pred': train_pred_3d,
                 'bern_valid_pred': valid_pred_3d}
    
    joblib.dump(pred_dict, filename +'_pred_raw.pkl') # save dictionary
    print('raw predictions saved')
    new_time = time.time()
    
    # generate output metrics
    out_list_train = [] # list for training metrics
    out_list_valid = [] # list for validation metrics
    # get output metrics for the training data
    for i in range(train_3d.shape[0]):  # for each posterior prediction 
        outmet = output_mets(train_observed[:,0],train_pred_3d[i,:,1],train_pred_3d[i,:,0]) # generate the output metrics
        rmse = mean_squared_error(train_y[:,0], train_3d[i,:,1], squared = False) # also get the RMSE
        outmet += [rmse] # add RMSE to the other metrics
        out_list_train += [outmet] # add the output metric for this posterior prediction into the list
    out_train = np.stack(out_list_train) # format
    out_train = pd.DataFrame(out_train, columns = ['loss','acc','prec','rc','roc','pr','f1','gmean','rmse']) # turn into dataframe
    out_train.to_csv(path +'predictions//'+ filename + '_train.csv') # save file
    print('generated train output metrics ' + str((time.time() - new_time)/60) + ' mins') # get runtime
    new_time = time.time()
    
    # get the output metrics for the validation data
    for i in range(valid_3d.shape[0]): # for each posterior prediction
        outmet = output_mets(valid_observed[:,0],valid_pred_3d[i,:,1],valid_pred_3d[i,:,0]) # generate the output metrics
        rmse = mean_squared_error(valid_y[:,0], valid_3d[i,:,1], squared = False) # also get the RMSE     
        outmet += [rmse] # add Rmse to the other metrics
        out_list_valid += [outmet] # add the output metric for this posterior prediction into the list
    out_valid = np.stack(out_list_valid) # format
    out_valid = pd.DataFrame(out_valid, columns = ['loss','acc','prec','rc','roc','pr','f1','gmean','rmse']) # turn into dataframe
    out_valid.to_csv(path +'predictions//'+ '_valid.csv') # save file
    
    print('generated valid output metrics ' + str((time.time() -new_time)/60) + ' mins')
    print('total runtime: ' + str((time.time()-start_time)/60) + ' mins')
    return out_train, out_valid

def output_mets(y, y_pred, y_prob):
    '''
    generate output metrics to evaluate model performances

    Parameters
    ----------
    y : target y
    y_pred : bern predicted y
    y_prob : probability of positive class for y

    Returns
    -------
    return list of output metrics

    '''
    y = y.astype('int32') # set datatype to int32 as required by metric functions
    y_pred = y_pred.astype('int32') # set datatype to int32 as required by metric functions
    loss = log_loss(y, y_prob) # calculate the log loss
    acc = accuracy_score(y,y_pred) # calculate accuracy
    prec = precision_score(y,y_pred) # calculate the precision
    rc = recall_score(y,y_pred) # calculate the recall
    pr = average_precision_score(y, y_prob) # calculate AUCPR
    roc = roc_auc_score(y, y_prob) # calculate AUC ROC
    f1 = f1_score(y,y_pred) # calculate the f1 score
    gmean = geometric_mean_score(y, y_pred) # calculate the geometric mean
    return [loss,acc,prec,rc,roc,pr,f1,gmean] # output the metrics

def bern_gen(y_pres, yn, y_prob):
    '''
    Generating bernoulli predictions from the binomial prediction 
    Parameters
    ----------
    y_pres : number of positive class predictions
    yn : # number of predictions
    y_prob : # probablity of prediction 

    Returns
    -------
    obs : observation 
    probs : probablities

    '''
    obs = [] # empty list for observation
    probs = [] # empty list for probabilities
    for i in range(len(y_pres)): # for each prediction
        ones= np.ones(int(y_pres[i])) # generate ones for the positive classes
        zeros= np.zeros(int(yn[i]-y_pres[i])) # generate zero for the negative classes
        obs_oz = np.concatenate([ones,zeros]) # combine them together
        np.random.shuffle(obs_oz) # shuffle around, do this since it is unlikely the predictions will always be order all 1's predicted first and then 0's second
        proba = np.repeat(y_prob[i], int(yn[i])) # match the probability to the bernoulli prediction
        obs += [obs_oz] # add the bernoulli predictions
        probs += [proba] # add the probabilit yof the bernoulli prediction
    obs = np.concatenate(obs) # merge into array 
    obs = obs.astype('int32')  # set datatype
    probs = np.concatenate(probs) # merge into array
    return obs, probs # return the bernoulli prediction and their prediction probability

tout, vout = prob_pred_bernoulli(results, train_y, valid_y, path, files[10])
