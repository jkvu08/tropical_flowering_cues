# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 14:05:32 2022
Objective of this project is to identify the weather cues that tropical plants use to regulate their flowering and the timescales at which they occur. 
Run the Bayesian model without a wrapper functions and visualize results 
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
import flower_func as flm

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
# rain - daily rainfall (mm)
# temp.mean - mean daily temperature (C)
# solar.rad - daily solar radiation (W/m^2)

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
          'ni': 10000, # number of draws
          'nb':1000, # number of burn-ins
          'nc': 4, # number of chains
          'nt':1, # thinning ratio
          'alpha_sd': 10, # standard deviation for alpha priors
          'nuts_target':0.95, # mean acceptance probability for NUTS sampler, influences the step size
          'variables':['~w0','~p','~x0'], # assign variables
          'name_var': ['y','p','w0','x0'], # assign variable names
          'save': 'full', # specify how much of model results to save
          'divergence':None, # specify position of divergence draws
          'relation': 'single_warm', # assign file name suffix 
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
model = flm.single_model(train_X, train_y, params)

# run model using training data
results = flm.run_model(train_X,
                        train_y,
                        model,
                        params, 
                        save=params['save']) 


# predict using validation data
with results['model']:
        pm.set_data({'X': valid_X,
                     'n': valid_y[:,1].astype('int32')})
        results['vppc'] = pm.sample_posterior_predictive(results['trace'],
                                              var_names =params['name_var'])
    
# explore results
results.keys()
# explore pymc3 inference outputs
results['inference']['posterior']
results['inference']['log_likelihood']
results['inference']['sample_stats']
results['inference']['observed_data']
results['inference']['constant_data']

# explore posterior predictions 
results['ppc'] # for training data
results['vppc'] # for prediction data

# save results
joblib.dump(results, results['filename']+'.pkl') 

# output pymc3 model summary of parameter estimates with credible intervals and save 
sum_df, hdi_df = flm.summary_tab(results, params) 
sum_df
hdi_df

# calculate model selection metrics 
comp_df = flm.comp_tab(results, path)
comp_df

# plot caterpillar plots
flm.trace_image(results, params, path) # view in file

# generate bernoulli (0/1) and binomial (counts of individuals) predictions of flowering 
train_tab, valid_tab, binom_pred = flm.prob_pred_bern(results, 
                                                      train_y, 
                                                      valid_y, 
                                                      path) 

# gneerate classification report
class_rep = flm.class_rep_split(train_tab['obs'],
                                train_tab['pred'],
                                valid_tab['obs'],
                                valid_tab['pred'])
class_rep

# generate posterior predictive check plot
hcol = len(params['covariates']) # number of cuues 
fig,ax = pyplot.subplots(hcol,2,figsize =(8,3*hcol))
# for each variable plot observed, predictions with credible intervals against weather conditions during cue period
for i in range(hcol):
    pyplot.subplot(hcol,2,2*i+1)
    # generate for training
    flm.covplot(data=results['ppc'],
            y=train_y,
            cov='w'+str(i),
            covariate = params['covariates'][i],
            medthreshold = sum_df.loc['threshold'+str(i), 'median'],
            modname = params['species'] + ' training')
    pyplot.subplot(hcol,2,2*i+2)
     # generate for validation
    flm.covplot(data=results['vppc'],
            y=valid_y,
            cov='w'+str(i),
            covariate = params['covariates'][i],
            medthreshold = sum_df.loc['threshold'+str(i), 'median'],
            modname = params['species'] + ' validation')
fig.tight_layout()

# generate time series plots
for i in range(hcol):
    flm.time_series_plot(train_ds, 
                     valid_ds, 
                     train_y, 
                     valid_y, 
                     results = results, 
                     params = params,
                     cov='w'+str(i),    
                     covariate= params['covariates'][i],
                     ci = 0.95,
                     modname = params['species'])

# generate confusion matrices, roc curves and pr curves  
pb_plot = flm.plot_bayes_split(train_tab['obs'],
                               train_tab['pred'], 
                               train_tab['pred_prob'], 
                               valid_tab['obs'],
                               valid_tab['pred'], 
                               valid_tab['pred_prob'])

# wrapper function to visualize model results
train_tab, valid_tab, binom_tab = flm.pymc_vis_split(results, 
                                                     train_y, 
                                                     valid_y, 
                                                     train_ds,
                                                     valid_ds,
                                                     params, 
                                                     path)

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

# run wrapper with entire model pipeline with visualization
model_metrics = flm.flower_model_wrapper(data = data, 
                                         gss = gss,
                                         params = params, 
                                         species = 'Rahiaka', 
                                         covariates = ['temp'],
                                         threshold = True,
                                         direction = ['positive'],
                                         relation = 'warm')


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
