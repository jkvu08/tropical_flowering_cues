# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 14:05:32 2022
Objective of this project is to identify the weather cues that tropical plants use to regulate their flowering and the timescales at which they occur. 
Implement Bayesian logistic regression to predict the probability of flowering in a species given the mean weather conditions in a preceding time window. 
A unique model was run for each potential cue/ cue combination.

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

# import functions from file
import flower_func as flm

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
# rain - daily rainfall (mm)
# temp.mean - mean daily temperature (C)
# solar.rad - daily solar radiation (W/m^2)

# view data structure
data.info()

# generator that will be used to split the training and validation dataset
# used group split in order to split by year
gss = GroupShuffleSplit(n_splits=1,  # only need 1 split: training and testing
                        train_size = 0.6, # use about 60% on training data and set a random state
                        random_state = 235) # optional to assigned random state

### RUN MODELS
# assign parameters
params = {'species': 'Rahiaka', # focal species
          'threshold': True, # assign whether threshold model is used
          'covariates': ['temp'], # weather cues 
          'direction': ['positive'], # assign cue directionality
          'sequential':True,
          'upper_lb': 100, # maximum cue window for cue 0
          'lower_lag0': 0, # minimum number of days expected between weather cue and flowering event for cue 0
          'upper_lag0':110, # maximum number of days expected between weather cue and flowering event for cue 0
          'lower_lag1': 0, # minimum number of days expected between weather cue and flowering event for cue 1
          'upper_lag1':110, # maximum number of days expected between weather cue and flowering event for cue 1
          'ni': 20000, # number of draws afer burn-ins
          'nb':20000, # number of burn-ins
          'nc': 4, # number of chains
          'nt':1, # thinning ratio
          'alpha_sd': 10, # standard deviation for alpha priors
          'nuts_target':0.95, # mean acceptance probability for NUTS sampler, influences the step size
          'variables':['~w0','~p','~x0'], # assign variables
          'name_var': ['y','p','w0','x0'], # assign variable names
          'save': 'full', # specify how much of model results to save
          'divergence':None, # specify position of divergence draws
          'relation': 'single_pos', # assign file name suffix 
          'mtype': '_2023'} # optional unique model identifier 

# run each model component 
# subset species level data, covariates (4-D array window x lag x survey date x covariate) and reponses
datasub, X, y = flm.data_gen(data, params)

# get sense of data
datasub.head() # view subset of species data 
X[0:10,0:10,0,0] # view subset of normalized weather conditions preceding a specific survey day and covariate
y[0:10,0:3] # view subset of responses (number of individuals flowering, total individuals sampled, prop of individuals flowering)

# split data into training and validation by year
train_X, valid_X, train_y, valid_y, train_ds, valid_ds = flm.train_test_data(X,
                                                                             y,
                                                                             datasub,
                                                                             gss)

# generate model architecture
model = flm.single_model(train_X, 
                         train_y, 
                         params)

# run model using training data
results = flm.run_model(valid_X,
                        valid_y,
                        model,
                        params, 
                        save=params['save']) 

# explore results
results.keys()
# explore pymc3 inference outputs
results['inference']['posterior'].info()
results['inference']['posterior']
results['inference']['log_likelihood']
results['inference']['sample_stats']
results['inference']['observed_data']
results['inference']['constant_data']

# explore posterior predictions 
results['ppc'] # for training data
results['vppc'] # for prediction data

# output pymc3 model summary of parameter estimates with credible intervals and save 
sum_df, hdi_df = flm.summary_tab(results, params) 

# get sense of parameter estimates and convergence (rhat)
sum_df 

# get sense of uncertainty about parameter estimates
hdi_df

# calculate model selection metrics 
comp_df = flm.comp_tab(results, path)

# print out model selection metrics
# this only makes sense relative to other models and will be used later to compare models
comp_df

# plot caterpillar plots for each parameter of interest (alpha, beta, lag, window, threshold )
# visual in file to get a sense of parameter estimates and convergence
flm.trace_image(results, params, path)

# generate bernoulli (0/1) and binomial (counts of individuals) predictions of flowering 
train_tab, valid_tab, binom_pred = flm.prob_pred_bern(results, 
                                                      train_y, 
                                                      valid_y, 
                                                      path) 

# generate classification report
class_rep = flm.class_rep_split(train_tab['obs'],
                                train_tab['pred'],
                                valid_tab['obs'],
                                valid_tab['pred'])

# get sense of training and validation performance by class and overall
class_rep


# generate confusion matrices, roc curves and pr curves to further visualize performance metrics
pb_plot = flm.plot_bayes_split(train_tab['obs'],
                               train_tab['pred'], 
                               train_tab['pred_prob'], 
                               valid_tab['obs'],
                               valid_tab['pred'], 
                               valid_tab['pred_prob'])

# generate posterior predictive check plots to visualize the relationsihp between weather cue and flowering prob
hcol = len(params['covariates']) # number of cuues 
fig,ax = pyplot.subplots(hcol,2,figsize =(8,3*hcol))
# for each variable plot observed, predictions with credible intervals against weather conditions during cue period
for i in range(hcol):
    pyplot.subplot(hcol,2,2*i+1)
    # generate for training
    flm.covplot(data=results['ppc'],
            y=train_y,
            cov='x'+str(i),
            covariate = params['covariates'][i],
            medthreshold = sum_df.loc['threshold'+str(i), 'median'],
            modname = params['species'] + ' training',
            legend = False)
    pyplot.subplot(hcol,2,2*i+2)
     # generate for validation
    flm.covplot(data=results['vppc'],
            y=valid_y,
            cov='x'+str(i),
            covariate = params['covariates'][i],
            medthreshold = sum_df.loc['threshold'+str(i), 'median'],
            modname = params['species'] + ' validation',
            legend = True)

fig.tight_layout()

# generate time series plots to visualize the observed versus predicted prob of flowering in relation to the weather conditions during the cue period
for i in range(hcol):
    flm.time_series_plot(train_ds, 
                     valid_ds, 
                     train_y, 
                     valid_y, 
                     results = results, 
                     params = params,
                     cov='x'+str(i),    
                     covariate= params['covariates'][i],
                     ci = 0.95,
                     modname = params['species'])

# wrapper function to visualize model results and save into pdf
train_tab, valid_tab, binom_tab = flm.pymc_vis_split(results, 
                                                     train_y, 
                                                     valid_y, 
                                                     train_ds,
                                                     valid_ds,
                                                     params, 
                                                     path)

# get sense of predictions table structure
train_tab.head()
valid_tab.head()
binom_tab.head()

# performance metrics
output = flm.outtab(train_tab, 
                    valid_tab, 
                    binom_tab, 
                    results['filename'], 
                    path)

output

# run wrapper with entire model pipeline with visualization output
# single cue model
single_metrics = flm.flower_model_wrapper(data = data, 
                                         gss = gss,
                                         params = params, 
                                         path = path,
                                         species = 'Rahiaka', 
                                         covariates = ['temp'],
                                         threshold = True,
                                         direction = ['positive'],
                                         relation = 'warm')

single_metrics

# double cue model
double_metrics = flm.flower_model_wrapper(data = data, 
                                         gss = gss,
                                         params = params, 
                                         path = path,
                                         species = 'Rahiaka', 
                                         covariates = ['solar','temp'],
                                         threshold = True,
                                         direction = ['negative','positive'],
                                         relation = 'double_negpos')

double_metrics

single_pos