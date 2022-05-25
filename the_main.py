# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 16:20:25 2022

@author: John
"""

'''IMPORT LIBS'''
import os
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.insert(1, 'C:\\Users\\User\\DSS EXPERIMENTS\\EMERALD_CAD_SCRIPTS\\')
import logging
import numpy as np

'''CONFIGURE LOGS'''



'''CUSTOM SCRIPTS'''
import configurations
import tasks
import routines
import data_preprocessing



'''GET THE CONFIGURATIONS'''
# in dictionary format
import configurations
OPTIONS_PREPROCESSING,OPTIONS_DATA_ANALYTICS,OPTIONS_MODE,OPTIONS_TRAINING,OPTIONS_DICOM_LOAD = configurations.get_options ()



#%%
'''FUNCTIONALITIES'''

# LOAD DATA
DATA_REST_AC, DATA_REST_NAC, DATA_STRESS_AC, DATA_STRESS_NAC, LABELS,EXPERT_LABELS, excel_file, ATT, INFO,error,time_seconds = data_preprocessing.load_cad_dicoms_sa001(OPTIONS_DICOM_LOAD)

# Sample image
img = DATA_REST_AC[1,1,:,:,:]
import matplotlib.pyplot as plt
plt.imshow(img)

#%%

# TRAIN NETWORK
import tasks
folds_metrics,predictions_all,predictions_all_num,test_labels,duration,history,FINAL_METRICS = tasks.train_image_only (OPTIONS_MODE,OPTIONS_TRAINING,OPTIONS_PREPROCESSING,DATA_REST_AC,DATA_REST_NAC,DATA_STRESS_AC,DATA_STRESS_NAC,ATT,LABELS,EXPERT_LABELS,INFO)


# TRAIN A CLINICAL_ONLY MODEL
import tasks
folds_metrics_clinical,predictions_all_clinical,predictions_all_num_clinical,test_labels_clinical,duration_clinical,history_clinical,FINAL_METRICS_clinical = tasks.train_clinical_only (OPTIONS_MODE,OPTIONS_TRAINING,OPTIONS_PREPROCESSING,DATA_REST_AC,DATA_REST_NAC,DATA_STRESS_AC,DATA_STRESS_NAC,ATT,LABELS,EXPERT_LABELS,INFO)





#%%

# PLOT RESULTS
routines.history_plots (history,OPTIONS_DICOM_LOAD['class_names'],predictions_all_num,LABELS)

routines.conf_matrix_plot(FINAL_METRICS['CNF'], OPTIONS_TRAINING['class_names'])

# GRAD CAM


