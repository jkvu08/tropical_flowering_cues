# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 14:05:32 2022
Objective of this project is to identify the weather cues that tropical plants use to regulate their flowering and the timescales at which they occur. 
Compare competing flowering cue models
@author: Jannet
"""
# load packages/modules
import os
import arviz as az
import matplotlib.pyplot as pyplot
#import numpy as np
import pandas as pd
from pandas import read_csv
import joblib

# set working directory
os.chdir("C:\\Users\\Jannet\\Documents\\Dissertation\\codes\\tropical_flowering_cues")
# set directory where results will be saved
path = "C:\\Users\\Jannet\\Documents\\Dissertation\\codes\\tropical_flowering_cues\\BGLM_flower\\"

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
    filename = path + species # generate file name
    comp_df = az.compare(comp_dict, 'loo') # generate comparison table using loo
    comp_df.to_csv(filename + '_loo_comparison.csv') # save comparison table
    comp_df.iloc[:,1:7] = round(comp_df.iloc[:,1:7],3)  # round the comparison table for visualization
    
    # format summary table into graphic
    fig, ax = pyplot.subplots(figsize = (8,6), nrows=2) # format figure layout
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
    return fig

# load loo data
mod1 = joblib.load('Rahiaka_temp_single_pos_2023_loo.pkl')  
mod2 = joblib.load('Rahiaka_solar_temp_double_negpos_2023_loo.pkl')

# organize loo data into dictionary
comp_dict = {'warm': mod1,
            'low light x warm': mod2}
             
# run plot comparisons 
# compares loo and ranks models based on loo
comp_fig = compare_plots(comp_dict, path, 'Rahiaka')
comp_fig

# merge performance metrics for each model into a single dataframe
# load performance metrics 
metrics1 = read_csv(path + 'Rahiaka_temp_single_pos_2023_output_metrics.csv', header =0)
metrics2 = read_csv(path + 'Rahiaka_solar_temp_double_negpos_2023_output_metrics.csv', header =0)

# assing column name for datatype
metrics1.rename(columns={'Unnamed: 0': 'datatype'},
                inplace = True)

metrics2.rename(columns={'Unnamed: 0': 'datatype'},
                inplace = True)

# add column for model identifier
metrics1['model'] = 'warm'
metrics2['model'] = 'low light x warm'

# combine performance metrics for the models into a single dataframe
combined_metrics = pd.concat([metrics1,metrics2], 
                             ignore_index = True)

# sort by the datatype and the model
combined_metrics.sort_values(by=['datatype', 'model'],
                             inplace = True)

# view to assess relative performance
combined_metrics

# low light x warm model slightly outperforms warm model across all metrics examined

# save comparison of performance metrics
combined_metrics.to_csv(path + 'Rahiaka_cue_model_comparisons.csv')
