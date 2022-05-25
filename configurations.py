# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 16:20:25 2022

@author: John
"""


def get_options ():
    
    OPTIONS_DICOM_LOAD = {
              'verbose'           : True,
              'example'           : True,
              'normalization'     : True,
              'shuffle'           : True,
              'no_of_slices'      : 10,
              'path'              : "C:\\Users\\User\\EMERALD DATA\\Processed Datasets\\CAD\\1st version (only SA fiels nothing else)\\",
              'label_path'        : "C:\\Users\\User\\EMERALD DATA\\Processed Datasets\\CAD\\TF SPECT_labelsv2_with_normals.xlsx",
              'label_col_name'    : 'pos/neg >70',
              'class_names'       : ["Healthy","Parathyroid"]
              }
    
    
    
    
    
    
    
    OPTIONS_PREPROCESSING = {
                        "W"              : 64,
                        "H"              : 64,
                        "D"              : 10,
                        'shape'          : (10,64,64,1)
        }
    
    # pass to a function which returns: image data in numpy array
    # and auxilliary variables
    
    
    
    OPTIONS_DATA_ANALYTICS = {
                        }
    
    
    # pass to a function and return analytics from the data (e.g. PCA, RF, Correlation, Means etc)
    
    
    
    OPTIONS_MODE = {
                        "image_only"         : True,
                        "cnn"                : "3d_main",
                        "classifier"         : "default", # or FCM or RF,
                        "grad_cam"           : True,
                        "feature_maps"       : True,
                        "importances"        : True,# only works for RF
                        'clinical_only_model': 'rf'
        }
    
    
    
    
    
    OPTIONS_TRAINING = {
                        "epochs"             : 150,
                        "k-split"            : 10,
                        "validation"         : 10,
                        "classes"            : 2,
                        'tune'               : 'frozen', # 'scratch','number_of_trainable'
                        "class_names"        : ["Healthy", "Parathyrid"],
                        'augmentation'       : True,
                        "batch_size"         : 64,
                        "plot_CM"            : True,
                        "verbose"            : True,
                        "verbose_metrics"    : True,
                        "save_model_after"   : True,}
    # independent options (global)
    
    return OPTIONS_PREPROCESSING,OPTIONS_DATA_ANALYTICS,OPTIONS_MODE,OPTIONS_TRAINING,OPTIONS_DICOM_LOAD



