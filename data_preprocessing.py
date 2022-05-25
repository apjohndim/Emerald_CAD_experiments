# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 17:21:52 2022

@author: John
"""


import os
import sys
import pandas as pd
from shutil import copy2
import numpy as np
import pydicom as pyd
import matplotlib.pyplot as plt
import datetime
from copy import deepcopy
import time
from keras.utils import to_categorical




def load_dicom(path, full):
    
    start = time.time()
    
    dicom = pyd.read_file(path)
    img_series = dicom.pixel_array
    
    info = {
    "number_of_imgs":0,
    "machine_model":0,
    "machine_manu":0,
    "slice_thickness":0,
    "study_desc":0,
    "img_max_value":0
    }
    
    if full:
        number_of_imgs = int(dicom.NumberOfFrames)
        machine_model = dicom.ManufacturerModelName
        machine_manu = dicom.Manufacturer
        slice_thickness = dicom.SliceThickness
        study_desc = dicom.StudyDescription
        img_max_value = img_series.max()
        
        info = {
            "number_of_imgs":number_of_imgs,
            "machine_model":machine_model,
            "machine_manu":machine_manu,
            "slice_thickness":slice_thickness,
            "study_desc":study_desc,
            "img_max_value":img_max_value
            }
    
    return img_series, info
    
    

def load_cad_dicoms_sa001(OPTIONS_DICOM_LOAD):
    
    ''' TOTAL LOADER
    
    Give paths to labels and patient folders
    give the column name that is used for reference (1 output mode)
    give the CONFIG
    
    This functions does the following:
        searches in the patient folders and loads the 4 dicoms inside them. The dicoms must have specific names
        loads the dicoms and makes them 4D, channel last format. It keeps only the central slices according to config
        reads the excel file and constructs:
            a. a labels list (binary)
            b. a dataframe with every info
            c. an ATT dataframe which contains only the ML attributes and the patient ID -> this must go to index later
        The ATT file goes for the ML models
        The excel_file contains all the data
        
    For a correct operation, specify the columns that are unecessary in the list excluded_columns. Those columns are excluded from the ATT
    
    All columns must 
    
    
    ''' 
    start = time.time()
    
    path = OPTIONS_DICOM_LOAD['path']
    label_path = OPTIONS_DICOM_LOAD['label_path']
    label_col_name = OPTIONS_DICOM_LOAD['label_col_name']
    
    excluded_columns = ['NAME',
                        'EXPERT DIAGNOSIS: STRESS DEFECT SIZE\nNORM=0/SMALL=1/MED=2/LARGE=3',
                        'LM>50','LAD>70','LCX >70','RCA > 70','MULTIVESSEL >70','0,1,2,3 VS',
                        'SCAN 1','SCAN > 1','SCAN 2','pos/neg >70','EXPERT DIAGNOSIS: REVERSIBLE DEFECT SIZE (ISCHEMIA)',
                        'Expert Diagnosis Binary']
    
    if OPTIONS_DICOM_LOAD['verbose']:
        print('')
        print('===========================================')
        print('===== INITIALIZING CAD DICOM LOADER  ======')
        print('===========================================')
        print('Datetime: {}'.format(datetime.datetime.now()))
        print('')
        
    # EMPTY DATA LISTS
    DATA_STRESS_AC = []
    DATA_STRESS_NAC = []
    DATA_REST_AC = []
    DATA_REST_NAC = []
    
    # LABELS LIST
    LABELS = []
    EXPERT_LABELS = []
    

    
    
    # LOAD EXCEL FILE
    excel_file = pd.read_excel(label_path)
    
    # VERY IMPORTANT INFORMATION. THE COLUMN OF THE GROUND TRUTH
    GROUND_TRUTH = label_col_name
    column = excel_file.columns.get_loc(GROUND_TRUTH)
    expert_column = excel_file.columns.get_loc('Expert Diagnosis Binary')
    
    # CONSTRUCT THE INFO DATAFRAME   
    columns = ["patient_no",
               "posneg>70",
               "vessels",
               "expert",
               "dicom_path",
               "number_of_imgs",
               "model",
               "manufacturer",
               "thickness",
               "study",
               "max_val"
        ]
    
    INFO = pd.DataFrame(columns = columns)
    
    '''HANDLE THE ATTRIBUTES OF THE FCM MODEL'''
    
    #construct a dataframe with only the attributes
    attributes = deepcopy(excel_file)
    for col in excluded_columns:
        print(col)
        attributes.drop(col,1,inplace=True)
    
    # the SEX is in f,m format -> turn to binary
    mapping = {'f': 0, 'm': 1}
    attributes['SEX'] = attributes['SEX'].map(mapping) 
    

    # ATRIBUTE SET
    ATT = pd.DataFrame(columns = attributes.columns)

    
    # FIND THE SUBFOLDERS OF THE MAIN PATH
    folder_patients = os.listdir(path)
    print('FOUND {} PATIENT FOLDERS'.format(len(folder_patients)))
    
    error = []
    # FOR EACH SUBFOLDER (PATEINT CASE)
    for patient in folder_patients:
        if OPTIONS_DICOM_LOAD['verbose']:
            print('')
            print('--> Working on folder: {}'.format(patient))
        the_path = os.path.join(path,patient)
        the_files = os.listdir(the_path)
        
        
        
        # EXCEL MATCH
        try:
            pat_id = int(patient[:4])
        except Exception as e:
            print(e)
            continue
        
        try:
            row = excel_file[excel_file['No']==pat_id].index[0]
            

            
            label = int(excel_file.iloc[row,column])
            label_expert = int(excel_file.iloc[row,expert_column])
            
        except Exception as e:
            print (e)
            print ('Patient {} not matched with excel'.format(pat_id))
            continue
        
        #LABELS.append(label)
        
        
        
        
        # MATCH THE FILE NAMES WITH THE EXPECTED REST-STRESS AC-NAC ENTITIES
        ok = True
        for filename in the_files:
            complete_filepath = os.path.join(the_path,filename)
            if "STRESS_AC" in filename:
                stress_ac_path = os.path.join(the_path,filename)
            elif "STRESS_NAC" in filename:
                stress_nac_path = os.path.join(the_path,filename)
            elif "REST_AC" in filename:
                rest_ac_path = os.path.join(the_path,filename)
            elif "REST_NAC" in filename:
                rest_nac_path = os.path.join(the_path,filename)
            else:
                print('Problem in folder. Could not match the stress-rest ac-nac files')
                ok = False
        
        if not ok:
            continue
        else:
            if OPTIONS_DICOM_LOAD['verbose']:
                print('Stress-Rest-AC-NAC files match. Proceeding')
            
            try:    
                # LOAD A STRESS AC
                
                stress_nac,info = load_dicom(path=stress_nac_path, full=False)
                rest_ac,info = load_dicom(path=rest_ac_path, full=False)
                rest_nac,info = load_dicom(path=rest_nac_path, full=False)
                stress_ac,info = load_dicom(path=stress_ac_path, full=True)
                
                # KEEP THE DESIRED NO. OF SLICES
                if (stress_ac.shape[0]%OPTIONS_DICOM_LOAD["no_of_slices"])%2 == 1:
                    stress_ac = stress_ac[ int(OPTIONS_DICOM_LOAD["no_of_slices"]/2-1) : int(OPTIONS_DICOM_LOAD["no_of_slices"]/2-1) + OPTIONS_DICOM_LOAD["no_of_slices"],  :,:]
                else:
                    stress_ac = stress_ac[ int(OPTIONS_DICOM_LOAD["no_of_slices"]/2+1) : int(OPTIONS_DICOM_LOAD["no_of_slices"]/2+1) + OPTIONS_DICOM_LOAD["no_of_slices"],  :,:]
                
                # NORMALIZE TO (0,1)
                if OPTIONS_DICOM_LOAD['normalization']:
                    stress_ac = stress_ac/(stress_ac.max())
                
                # EXPAND DIMS
                stress_ac = np.expand_dims(stress_ac, axis=3)
                
                
                
                
                # LOAD A STRESS NAC
                
                # KEEP THE DESIRED NO. OF SLICES
                if (stress_nac.shape[0]%OPTIONS_DICOM_LOAD["no_of_slices"])%2 == 1:
                    stress_nac = stress_nac[ int(OPTIONS_DICOM_LOAD["no_of_slices"]/2-1) : int(OPTIONS_DICOM_LOAD["no_of_slices"]/2-1) + OPTIONS_DICOM_LOAD["no_of_slices"],  :,:]
                else:
                    stress_nac = stress_nac[ int(OPTIONS_DICOM_LOAD["no_of_slices"]/2+1) : int(OPTIONS_DICOM_LOAD["no_of_slices"]/2+1) + OPTIONS_DICOM_LOAD["no_of_slices"],  :,:]
                # NORMALIZE TO (0,1)
                if OPTIONS_DICOM_LOAD['normalization']:
                    stress_nac = stress_nac/(stress_nac.max())
                # EXPAND DIMS
                stress_nac = np.expand_dims(stress_nac, axis=3)            
                
                
                
                
                
                # LOAD A REST AC
                
                # KEEP THE DESIRED NO. OF SLICES
                if (rest_ac.shape[0]%OPTIONS_DICOM_LOAD["no_of_slices"])%2 == 1:
                    rest_ac = rest_ac[ int(OPTIONS_DICOM_LOAD["no_of_slices"]/2-1) : int(OPTIONS_DICOM_LOAD["no_of_slices"]/2-1) + OPTIONS_DICOM_LOAD["no_of_slices"],  :,:]
                else:
                    rest_ac = rest_ac[ int(OPTIONS_DICOM_LOAD["no_of_slices"]/2+1) : int(OPTIONS_DICOM_LOAD["no_of_slices"]/2+1) + OPTIONS_DICOM_LOAD["no_of_slices"],  :,:]
                # NORMALIZE TO (0,1)
                if OPTIONS_DICOM_LOAD['normalization']:
                    rest_ac = rest_ac/(rest_ac.max())
                # EXPAND DIMS
                rest_ac = np.expand_dims(rest_ac, axis=3)            
                
                
                
                
                
                # LOAD A REST NAC
                
                # KEEP THE DESIRED NO. OF SLICES
                if (rest_nac.shape[0]%OPTIONS_DICOM_LOAD["no_of_slices"])%2 == 1:
                    rest_nac = rest_nac[ int(OPTIONS_DICOM_LOAD["no_of_slices"]/2-1) : int(OPTIONS_DICOM_LOAD["no_of_slices"]/2-1) + OPTIONS_DICOM_LOAD["no_of_slices"],  :,:]
                else:
                    rest_nac = rest_nac[ int(OPTIONS_DICOM_LOAD["no_of_slices"]/2+1) : int(OPTIONS_DICOM_LOAD["no_of_slices"]/2+1) + OPTIONS_DICOM_LOAD["no_of_slices"],  :,:]
                # NORMALIZE TO (0,1)
                if OPTIONS_DICOM_LOAD['normalization']:
                    rest_nac = rest_nac/(rest_nac.max())
                # EXPAND DIMS
                rest_nac = np.expand_dims(rest_nac, axis=3)
                
                
                DATA_STRESS_AC.append(stress_ac)
                DATA_STRESS_NAC.append(stress_nac)
                DATA_REST_AC.append(rest_ac)
                DATA_REST_NAC.append(rest_nac)
                
    
                LABELS.append(label)
                EXPERT_LABELS.append(label_expert)
                
                # FIND THE INDEX OF THE ROW WITH PATIENT ID EQUAL TO THE CURRENT ID AND APPEND IT TO THE ATT DATAFRAME
                where_is = attributes.index[attributes['No'] == pat_id].tolist()[0]
                ATT = ATT.append(attributes.iloc[where_is])
                
                INFODICT = {
                    "patient_no":pat_id,
                   "posneg>70":label,
                   "vessels":excel_file.iloc[where_is]['0,1,2,3 VS'],
                   "expert":excel_file.iloc[where_is]['EXPERT DIAGNOSIS: REVERSIBLE DEFECT SIZE (ISCHEMIA)'],
                   "dicom_path":'na',
                   "number_of_imgs":info['number_of_imgs'],
                   "model":info['machine_model'],
                   "manufacturer":info['machine_manu'],
                   "thickness":info['slice_thickness'],
                   "study":info['study_desc'],
                   "max_val":info['img_max_value']
                   }
                
                INFO = INFO.append(INFODICT,ignore_index=True)
                
                assert rest_nac.shape[0]+rest_ac.shape[0]+stress_nac.shape[0]+stress_ac.shape[0] == 4*rest_nac.shape[0], 'img_series do not have the same legnth'
                
            except Exception as e:
                print (e)
                error.append(pat_id)
    
    DATA_REST_AC = np.stack(DATA_REST_AC)
    DATA_REST_NAC = np.stack(DATA_REST_NAC)
    DATA_STRESS_AC = np.stack(DATA_STRESS_AC)
    DATA_STRESS_NAC = np.stack(DATA_STRESS_NAC)
    
    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # NEED TO RETURN ICA_LABELS, HUMAN_LABELS
    # NEED TO RETURN ATT ARRAY WITHOUT ANY LABELS AND WITHOUT PATIENT NO
    
    ATT = ATT.drop("No",1)
    
    LABELS = to_categorical(LABELS)
    EXPERT_LABELS = to_categorical(EXPERT_LABELS)
    
    
    
    end = time.time()
    
    time_seconds = round(end-start,3)
    print ("Time taken: {} seconds".format (time_seconds))
    return DATA_REST_AC, DATA_REST_NAC, DATA_STRESS_AC, DATA_STRESS_NAC, LABELS,EXPERT_LABELS, excel_file, ATT, INFO,error,time_seconds