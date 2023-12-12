# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 13:17:06 2023

@author: Jannet
"""
import os, time
#import arviz as az
#import matplotlib.pyplot as pyplot
import numpy as np
import pandas as pd
from pandas import read_csv#, DataFrame
#import pymc3 as pm
#import seaborn as sns
#import theano.tensor as tt
#from sklearn.metrics import roc_curve, confusion_matrix, precision_recall_curve, classification_report
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score, accuracy_score, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
import joblib
#from matplotlib.backends.backend_pdf import PdfPages
#from PIL import Image
#import imblearn
from imblearn.metrics import geometric_mean_score
import gc
import glob
#from pymc3.distributions.continuous import Lognormal

# set working directory
os.chdir("C:\\Users\\Jannet\\Documents\\Dissertation\\data")
# set directory where results will be saved
path = "C:\\Users\\Jannet\\Documents\\Dissertation\\results\\BGLM_flower\\"

# import data
#data = read_csv('kian_phenology_bernoulli_09062022.csv',header=0).iloc[:,1:]
data = read_csv('kian_phenology_binomial_09062022.csv',header=0).iloc[:,1:]
# data attributions:
# Date - date (YYYY-MM-DD)
# Species - species local name
# year - year 
# mont - month
# n_fl - number of individuals surveyed on Date
# flowers - number of individuals flowering on Date
# prop_fl - proportion of individuals flowering on Date 

# assign labels for the visualization
LABELS = ["no flower","flower"]

# cue dictionary that associates covariate name with column index in covariate array
cue_dict = {'rain': 0,
            'drought': 0,
            'temp': 1,
            'solar': 2}

# generate splits that will be used to split the training and validation dataset
# use group split cause we are spliting by year
# only need 1 split: training and testing
# use about 60% on training data and set a random state
gss = GroupShuffleSplit(n_splits=1, train_size = 0.6, random_state = 235) 

# define data formating functions 
def data_gen(data, params):
    """
    generate species specific data

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
    # This is species specific because it maps the exact date the species was surveyed
    # The data contains the normalized climate data for all cue window and lag time combinations with respec to the survey date
    cue_list =[] # create empty list for climatic cue keys 
    # populate cue keys in a list
    for cue in params['covariates']: # for each climatic condition being tested as a cue 
        cue_list += [cue_dict[cue]] # extract the relevant climatic cue id based on the cue dictionary
    X = X[:,:,:,np.r_[cue_list]] # use cue key list to subset out the relevant covariates
    y = np.array(datasub.loc[:, ['flower','n_fl','prop_fl']]) # subset out only the phenology specific columns
    if len(X.shape) == 3: # if the shape of the covariate array is 3D make it 4D
        X = X[:,:,:,np.newaxis]
    return datasub, X, y

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
    valid_prob = valid_prob[-80000:,:]
    
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
    print('generated the binomial predictions & formatted bern & binom observations ' + str((new_time-start_time)/60) + ' mins')
   
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
    
    joblib.dump(pred_dict, path+'predictions//'+filename +'_preds.pkl') # save dictionary
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
    out_valid.to_csv(path +'predictions//'+ filename+'_valid.csv') # save file
    
    print('generated valid output metrics ' + str((time.time() -new_time)/60) + ' mins')
    print('total runtime: ' + str((time.time()-start_time)/60) + ' mins')
    return out_train, out_valid

# assign parameters
params = {'species': 'Rahiaka',
          'lower_lag':0,
          'upper_lag':110,
          'upper_lb': 100,
          'lower_lag0': 0,
          'upper_lag0':110,
          'lower_lag1': 0,
          'upper_lag1':110,
          'ni': 20000,
          'nb':20000,
          'nc': 4,
          'nt':1,
          'covariates': ['solar'],
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
          'mtype': 'wright'}

# 'Rahiaka_temp_solar_wright_double_pos',
files = ['Ravenala_rain_solar_wright_double_neg',
         'Ambora_solar_temp_wright_double_posneg',
         'Ampaly_rain_wright_single_neg',
         'Rahiaka_temp_wright_single_pos']

for file in files:
    datasub, X, y = data_gen(data, params)
    _, _, train_y, valid_y = train_test_data(X, y, datasub, gss)
    results =joblib.load(file+'.pkl')
    out_train, out_val = prob_pred_bernoulli(results, train_y,valid_y, path, file)

# summarize the outputs 
filetrain = [] # get file names for training output metrics
filevalid = [] # get file names for validation output metrics
for file in glob.glob(path + 'predictions//*_train.csv'):
    filename = file.split('\\')[-1]
    filetrain.append(filename)
    
for file in glob.glob(path + 'predictions//*_valid.csv'):
    filename = file.split('\\')[-1]
    filevalid.append(filename)

metrics = ['loss', 'acc', 'prec', 'rc', 'roc', 'pr', 'f1', 'gmean', 'rmse']
def metsum_fun(df, species, metric, model, direction):
    '''
    sumamrize posterior prediction output metrics

    Parameters
    ----------
    df : posterior prediction output metrics
    species : species
    metric : metric of interest
    model : covariates
    direction: model directionality
    

    Returns
    -------
    summary of metrics 

    '''
    values = np.array(df[metric].sort_values())# sort the the metric in order
    mval = np.mean(values) # calculate mean
    medval = np.median(values) # calculate median
    minval = np.min(values) # calculate min
    maxval = np.max(values) # calculate max
    l95 = values[1999] # get 95 CI
    u95 = values[77999]
    l90 = values[3999] # get 90 CI
    u90 = values[75999]
    l80 = values[7999] # get 80 CI
    u80 = values[71999]
    l50 = values[19999] # get 50 CI
    u50 = values[59999]
    return [species, model, direction, metric, mval, medval, minval, maxval, l95,u95,l90,u90,l80,u80,l50,u50] # output metric summary

trainlist = []
for file in filetrain:
    met_tab = read_csv(path+'predictions//'+file, header=0, index_col=0)
    namesplit = file.split('_')
    species = namesplit[0]
    direction = namesplit[-2]
    if namesplit[2] == 'wright':
        model = namesplit[1]
    else:
        model = namesplit[1] + ' x ' + namesplit[2]
    for metric in metrics:
        trainlist += [metsum_fun(met_tab, species, metric, model, direction)]
        
traintab = np.stack(trainlist)
traintab = pd.DataFrame(traintab, columns = ['species', 'cues', 'direction', 'metric', 'mean','median',
                                             'min', 'max', 'l95','u95','l90','u90','l80','u80','l50','u50'])
traintab.to_csv('top_elpd_predictions_train.csv')

validlist = []
for file in filevalid:
    met_tab = read_csv(path+'predictions//'+file, header=0, index_col=0)
    namesplit = file.split('_')
    species = namesplit[0]
    direction = namesplit[-2]
    if namesplit[2] == 'wright':
        model = namesplit[1]
    else:
        model = namesplit[1] + ' x ' + namesplit[2]
    for metric in metrics:
        validlist += [metsum_fun(met_tab, species, metric, model, direction)]
        
validtab = np.stack(validlist)
validtab = pd.DataFrame(validtab, columns = ['species', 'cues', 'direction', 'metric', 'mean','median',
                                             'min', 'max', 'l95','u95','l90','u90','l80','u80','l50','u50'])

validtab.to_csv('top_elpd_predictions_valid.csv')
