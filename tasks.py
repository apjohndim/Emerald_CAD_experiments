# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 17:07:24 2022

@author: John
"""
import routines
import time
from sklearn.model_selection import KFold
import numpy as np
# start = time.time()
# end = time.time()
# duration = (end - start)

import data_preprocessing
from image_data_gen import VoxelDataGenerator

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train_image_only (OPTIONS_MODE,OPTIONS_TRAINING,OPTIONS_PREPROCESSING,DATA_REST_AC,DATA_REST_NAC,DATA_STRESS_AC,DATA_STRESS_NAC,ATT,LABELS,EXPERT_LABELS,INFO):
    start = time.time()
    
    # storage
    predictions_all = np.empty(0) # here, every fold predictions will be kept
    test_labels = np.empty([0,OPTIONS_TRAINING['classes']])
    test_labels_onehot = np.empty(0)#here, every fold labels are kept
    predictions_all_num = np.empty([0,OPTIONS_TRAINING['classes']])
    folds_metrics = []
    
    summ = True
    
    for train_index, test_index in KFold(OPTIONS_TRAINING['k-split']).split(LABELS):

            
        
        # KFOLD SPLIT DATA
        train_stress_ac, test_stress_ac = DATA_STRESS_AC[train_index], DATA_STRESS_AC[test_index]
        train_stress_nac, test_stress_nac = DATA_STRESS_NAC[train_index], DATA_STRESS_NAC[test_index]
        train_rest_ac, test_rest_ac = DATA_REST_AC[train_index], DATA_REST_AC[test_index]
        train_rest_nac, test_rest_nac = DATA_REST_NAC[train_index], DATA_REST_NAC[test_index]
        
        # KFOLD SPLIT LABELS
        train_label_ica, test_label_ica = LABELS[train_index], LABELS[test_index]
        train_label_expert, test_label_expert = EXPERT_LABELS[train_index], EXPERT_LABELS[test_index]
        
        # assign ground truth
        train_label = train_label_expert
        test_label = test_label_expert
        
        # PICK THE MODEL: CALL ANOTHER FUNCTION TO PICK IT
        model = routines.pick_model(OPTIONS_MODE, OPTIONS_PREPROCESSING, OPTIONS_TRAINING)
        if summ:
            summ = False
            model.summary()
        
        # params
        epochs = int(OPTIONS_TRAINING['epochs'])
        batch_size = int(OPTIONS_TRAINING['batch_size'])
        

        
        
        # PLACEHOLDER
        if OPTIONS_TRAINING['augmentation']:
            print ('Under Development')
            augmentation = False

        else:
            augmentation = False
            
        
        # TRAIN
        
        if augmentation:
            c = VoxelDataGenerator(rotate_axis=1, rotate_angle=10)
            h = c.build(data=train_stress_ac, label=train_label, batch_size=batch_size)
            i = c.build(data=train_stress_nac, label=train_label, batch_size=batch_size)
            j = c.build(data=train_rest_ac, label=train_label, batch_size=batch_size)
            k = c.build(data=train_rest_nac, label=train_label, batch_size=batch_size)
            history = model.fit_generator([h,i,j,k], steps_per_epoch=50, epochs=30)
        else:
            history = model.fit([train_stress_ac,train_stress_nac,train_rest_ac,train_rest_nac], train_label, validation_split=0.05, epochs=epochs, batch_size=batch_size)
    
    
        # predict the unseen
        predict = model.predict([test_stress_ac,test_stress_nac,test_rest_ac,test_rest_nac])
        predict_num = predict
        predict = predict.argmax(axis=-1)
        
        
        test_label_onehot = np.argmax(test_label, axis=-1) #make the labels 1column array
        predictions_all = np.concatenate([predictions_all, predict])
        predictions_all_num = np.concatenate([predictions_all_num, predict_num])
        test_labels = np.concatenate([test_labels, test_label])
        test_labels_onehot = np.concatenate([test_labels_onehot, test_label_onehot])
        
        # CALL METRICS FUNCTION
        fold_metrics = routines.metrics(predict, predict_num, test_label, test_label_onehot)
        print ('Test Accuracy: {}, SEN: {}, SPE {}'.format(round(fold_metrics['Accuracy'],2),round(fold_metrics['Sensitivity'],2),round(fold_metrics['Specificity'],2)))
        folds_metrics.append(fold_metrics)
        
    FINAL_METRICS = routines.metrics(predictions_all, predictions_all_num, test_labels, test_labels_onehot)     
    end = time.time()
    duration = round(end - start,2) 
    return folds_metrics,predictions_all,predictions_all_num,test_labels,duration,history,FINAL_METRICS

    

def train_clinical_only (OPTIONS_MODE,OPTIONS_TRAINING,OPTIONS_PREPROCESSING,DATA_REST_AC,DATA_REST_NAC,DATA_STRESS_AC,DATA_STRESS_NAC,ATT,LABELS,EXPERT_LABELS,INFO):
    start = time.time()
    
    # storage
    predictions_all = np.empty(0) # here, every fold predictions will be kept
    test_labels = np.empty([0,OPTIONS_TRAINING['classes']])
    test_labels_onehot = np.empty(0)#here, every fold labels are kept
    predictions_all_num = np.empty([0,OPTIONS_TRAINING['classes']])
    folds_metrics = []
    
    summ = True
    
    dataset = np.array(ATT.fillna(0))
    
    for train_index, test_index in KFold(OPTIONS_TRAINING['k-split']).split(LABELS):

        train_clinical, test_clinical = dataset[train_index], dataset[test_index]
        train_clinical = train_clinical.astype(float)
             
        # KFOLD SPLIT LABELS
        train_label_ica, test_label_ica = LABELS[train_index], LABELS[test_index]
        train_label_expert, test_label_expert = EXPERT_LABELS[train_index], EXPERT_LABELS[test_index]
        
        # assign ground truth
        train_label = train_label_expert
        test_label = test_label_expert
        
        # PICK THE MODEL: CALL ANOTHER FUNCTION TO PICK IT
        model = routines.pick_model_clinical(OPTIONS_MODE, OPTIONS_PREPROCESSING, OPTIONS_TRAINING)


        # PLACEHOLDER
        if OPTIONS_TRAINING['augmentation']:
            print ('Under Development')
            augmentation = False

        else:
            augmentation = False
            
        
        # TRAIN

        model.fit(train_clinical, train_label)
        #f_i = model.feature_importances_
        
    
        # predict the unseen
        predict = model.predict(test_clinical)
        predict_num = predict
        predict = predict.argmax(axis=-1)
        
        
        test_label_onehot = np.argmax(test_label, axis=-1) #make the labels 1column array
        predictions_all = np.concatenate([predictions_all, predict])
        predictions_all_num = np.concatenate([predictions_all_num, predict_num])
        test_labels = np.concatenate([test_labels, test_label])
        test_labels_onehot = np.concatenate([test_labels_onehot, test_label_onehot])
        
        # CALL METRICS FUNCTION
        fold_metrics = routines.metrics(predict, predict_num, test_label, test_label_onehot)
        print ('Test Accuracy: {}, SEN: {}, SPE {}'.format(round(fold_metrics['Accuracy'],2),round(fold_metrics['Sensitivity'],2),round(fold_metrics['Specificity'],2)))
        folds_metrics.append(fold_metrics)
    
    
    if OPTIONS_MODE['clinical_only_model'] == 'rf':
        model.fit(dataset, LABELS)
        f_i = model.feature_importances_
        importance_dict = {}
        for col,imp in zip(ATT.columns,f_i):
            importance_dict.update({col: round(imp,2)})
    else: importance_dict = 0
        
            
    
    FINAL_METRICS = routines.metrics(predictions_all, predictions_all_num, test_labels, test_labels_onehot)     
    history = [model,importance_dict]
    end = time.time()
    duration = round(end - start,2) 
    return folds_metrics,predictions_all,predictions_all_num,test_labels,duration,history,FINAL_METRICS





'''


RANDOM FOREST + CNN INTEGRATION



'''

def train_integration (OPTIONS_MODE,OPTIONS_TRAINING,OPTIONS_PREPROCESSING,DATA_REST_AC,DATA_REST_NAC,DATA_STRESS_AC,DATA_STRESS_NAC,ATT,LABELS,EXPERT_LABELS,INFO):
    start = time.time()
    
    # storage
    predictions_all_img = np.empty(0) # here, every fold predictions will be kept
    test_labels_img = np.empty([0,OPTIONS_TRAINING['classes']])
    test_labels_onehot_img = np.empty(0)#here, every fold labels are kept
    predictions_all_num_img = np.empty([0,OPTIONS_TRAINING['classes']])
    folds_metrics_img = []
    
    summ = True
    
    for train_index, test_index in KFold(OPTIONS_TRAINING['k-split']).split(LABELS):

            
        
        # KFOLD SPLIT DATA
        train_stress_ac, test_stress_ac = DATA_STRESS_AC[train_index], DATA_STRESS_AC[test_index]
        train_stress_nac, test_stress_nac = DATA_STRESS_NAC[train_index], DATA_STRESS_NAC[test_index]
        train_rest_ac, test_rest_ac = DATA_REST_AC[train_index], DATA_REST_AC[test_index]
        train_rest_nac, test_rest_nac = DATA_REST_NAC[train_index], DATA_REST_NAC[test_index]
        
        # KFOLD SPLIT LABELS
        train_label_ica, test_label_ica = LABELS[train_index], LABELS[test_index]
        train_label_expert, test_label_expert = EXPERT_LABELS[train_index], EXPERT_LABELS[test_index]
        
        # assign ground truth
        train_label = train_label_expert
        test_label = test_label_expert
        
        # PICK THE MODEL: CALL ANOTHER FUNCTION TO PICK IT
        model = routines.pick_model(OPTIONS_MODE, OPTIONS_PREPROCESSING, OPTIONS_TRAINING)
        if summ:
            summ = False
            model.summary()
        
        # params
        epochs = int(OPTIONS_TRAINING['epochs'])
        batch_size = int(OPTIONS_TRAINING['batch_size'])
        

        
        
        # PLACEHOLDER
        if OPTIONS_TRAINING['augmentation']:
            print ('Under Development')
            augmentation = False

        else:
            augmentation = False
            
        
        # TRAIN
        
        if augmentation:
            c = VoxelDataGenerator(rotate_axis=1, rotate_angle=10)
            h = c.build(data=train_stress_ac, label=train_label, batch_size=batch_size)
            i = c.build(data=train_stress_nac, label=train_label, batch_size=batch_size)
            j = c.build(data=train_rest_ac, label=train_label, batch_size=batch_size)
            k = c.build(data=train_rest_nac, label=train_label, batch_size=batch_size)
            history = model.fit_generator([h,i,j,k], steps_per_epoch=50, epochs=30)
        else:
            history = model.fit([train_stress_ac,train_stress_nac,train_rest_ac,train_rest_nac], train_label, validation_split=0.05, epochs=epochs, batch_size=batch_size)
    
    
        # predict the unseen
        predict_img = model.predict([test_stress_ac,test_stress_nac,test_rest_ac,test_rest_nac])
        predict_num_img = predict_img
        predict_img = predict.argmax(axis=-1)
        
        
        test_label_onehot_img = np.argmax(test_label, axis=-1) #make the labels 1column array
        predictions_all_img = np.concatenate([predictions_all_img, predict_img])
        predictions_all_num_img = np.concatenate([predictions_all_num_img, predict_num_img])
        test_labels_img = np.concatenate([test_labels_img, test_label])
        test_labels_onehot_img = np.concatenate([test_labels_onehot_img, test_label_onehot_img])
        
        # CALL METRICS FUNCTION
        fold_metrics_img = routines.metrics(predict_img, predict_num_img, test_label, test_label_onehot_img)
        print ('Test Accuracy: {}, SEN: {}, SPE {}'.format(round(fold_metrics_img['Accuracy'],2),round(fold_metrics_img['Sensitivity'],2),round(fold_metrics_img['Specificity'],2)))
        folds_metrics_img.append(fold_metrics_img)
        
    FINAL_METRICS_img = routines.metrics(predictions_all_img, predictions_all_num_img, test_labels_img, test_labels_onehot_img)     
    end = time.time()
    duration = round(end - start,2) 
    



    return 

'''APP DEVELOPMENT FUNCTIONS: FUTURE'''



def load_trained_model ():
    return

def predict_new_sinle_input ():
    return

