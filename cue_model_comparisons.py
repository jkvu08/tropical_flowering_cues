# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 14:05:32 2022
Objective of this project is to identify the weather cues that tropical plants use to regulate their flowering and the timescales at which they occur. 
Compare competing flowering cue models
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

import flower_func as flm

######################
### Compare models ###
######################
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
    comp_df.to_csv(filename + '_loo_comparison.csv') # save comparison table
    comp_df.iloc[:,1:7] = round(comp_df.iloc[:,1:7],3)  # round the comparison table for visualization
    
    # format summary table into graphic
    fig, ax = pyplot.subplots(figsize = (6,6), nrows=2) # format figure layout
    # add comparison table to first subplot
    metrictab = ax[0].table(cellText=comp_df.values,
                            colLabels = comp_df.columns, 
                            rowLabels=comp_df.index,
                            cellLoc = 'center', 
                            loc='center') 
    metrictab.auto_set_font_size(False) # turn off auto font
    metrictab.set_fontsize(8) # set font size
    metrictab.auto_set_column_width(col=list(range(len(comp_df.columns)))) # adjust width to column content
    ax[0].axis('tight') 
    ax[0].axis('off') # turn graph axis off 
      
    # plot comparison table in second subplot
    az.plot_compare(comp_df, 
                    plot_ic_diff = False,
                    ax=ax[1])
    pyplot.ylabel('ranked models') # add y axis label
    pyplot.xlabel('loo (log)') # add x axis label
    # generate legend
    pyplot.legend(labels = ['loo without \nparameter \npenalty', 'best loo', 'loo'],
                    loc = 'center left', 
                    bbox_to_anchor=(1,0.5)) 
    pyplot.title(species + ' model comparison') # add title
    fig.tight_layout() # tighten layout
    fig.savefig(filename + '_model_comparison.png', dpi=150) # save figure
    pyplot.close() # close the plot

# load loo data
mod1 = joblib.load('Rahiaka_temp_single_pos_2023_loo.pkl')  
mod2 = joblib.load('Rahiaka_solar_temp_double_negpos_2023_loo.pkl')

# organize loo data into dictionary
comp_dict = {'warm': mod1,
            'low light x warm': mod2}
             
# run plot comparisons 
compare_plots(comp_dict, path, 'Rahiaka')

# merge performance metrics for each model into a single dataframe


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
