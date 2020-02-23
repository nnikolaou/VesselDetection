#!/usr/bin/env python
# coding: utf-8

# Dark Vessel Detection Project -- A simple Vessel Detector CNN implemented in Keras

# The script repeats n_runs times a train/validation(optionally)/test split of
# the data, followed by training and evaluation of the CNN. It saves all Keras
# models produced & prints various evaluation measures (average & standard error*).
# *standard error = standard deviation / sqrt(n_runs) 

from os import listdir
from os.path import isfile, join
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
#from tensorflow.keras.utils import plot_model

n_runs = 5 #Set number of runs (default was 5)
include_validation = 1 #1 to use a separate validation set during training, 0 to use test set (default was 1)
include_noise = 0 #1 to add gaussian noise to training examples, 0 to use clean ones (default was 0)

# Load data:
#Dataset directories:
vessel_dir = 'Chips/vessels'
nonvessel_dir = 'Chips/nonvessels'

#Get names of vessel data and nonvessel data files:
vessel_files = [f for f in listdir(vessel_dir) if isfile(join(vessel_dir, f))]
nonvessel_files = [f for f in listdir(nonvessel_dir) if isfile(join(nonvessel_dir, f))]

#Specify dimensionality of .tiffs:
num_rows, num_cols = 144, 144

vessels = np.zeros((num_cols, num_cols, len(vessel_files)))
nonvessels = np.zeros((num_cols, num_cols, len(nonvessel_files)))

#Load datapoints:
for i in range(len(vessel_files)):
    vessels[:,:,i] = np.array(Image.open(vessel_dir +"/"+ vessel_files[i]))
 
for i in range(len(nonvessel_files)):
    nonvessels[:,:,i] = np.array(Image.open(nonvessel_dir +"/"+ nonvessel_files[i]))

# Preprocessing
#Concatenate positives (vessels) & negatives (nonvessels)
# & construct label vector accordingly, using 1 & 0 to denote the two respective classes:
X = np.concatenate((vessels,nonvessels), axis = 2).T
y = np.concatenate((np.ones((1, len(vessel_files))), np.zeros((1, len(nonvessel_files)))), axis = 1).T

#Normalize data to [0,1] - Rescale all images by 1/65535 (65535 is the "naive" max value)
X = X/65535 # comment this out if using augmentation, or -TODO- move after augmentation

#Define arrays to store evaluation measures:
TN = np.zeros((n_runs, 1))
FP = np.zeros((n_runs, 1))
TP = np.zeros((n_runs, 1))
FN = np.zeros((n_runs, 1))
Accuracy = np.zeros((n_runs, 1))
Recall = np.zeros((n_runs, 1))
Precision = np.zeros((n_runs, 1))
F1_score = np.zeros((n_runs, 1))
Jaccard_index = np.zeros((n_runs, 1))
Expected_Accuracy = np.zeros((n_runs, 1))
Cohens_Kappa = np.zeros((n_runs, 1))
AUC_decisions = np.zeros((n_runs, 1))
AUC_scores = np.zeros((n_runs, 1))
brier_score = np.zeros((n_runs, 1))

for run in range(n_runs):
    print("Training model "+str(run+1)+" of "+str(n_runs)+".")
    #Split to train (80%) and test (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    if include_validation == 1: 
        #Further split training data into training (75%) & validation (25%) sets
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)
    else:
        X_val = X_test
        y_val = y_test
    
    # Preprocessing: Rebalance training set by oversampling negatives (optional)
    # Indices of examples from each class
    i_class0 = np.where(y_train == 0)[0] #non-vessels (minority)
    i_class1 = np.where(y_train == 1)[0] #vessels (majority)
    # Number of examples in each class
    n_class0 = len(i_class0)
    n_class1 = len(i_class1)
    # For every example in class 1, randomly sample from class 0 with replacement
    i_class0_upsampled = np.random.choice(i_class0, size=n_class1, replace=True)
    # Join together class 0's upsampled target vector with class 1's target vector
    y_train = np.concatenate((y_train[i_class0_upsampled], y_train[i_class1]))
    #As above, for the features:
    X_train = np.concatenate((X_train[i_class0_upsampled,:,:], X_train[i_class1,:,:]))
    #Upsampled training set needs to be shuffled again:
    X_train, y_train = shuffle(X_train, y_train)
    
    # Preprocessing: Add some Gaussian noise to the training data (optional)
    #Add Gaussian noise with mean 0 & variance that of corresponding pixel accross training data
    if include_noise == 1:
        pixelwise_mean = np.nanmean(X_train, axis = 0)
        pixelwise_std = np.nanstd(X_train, axis = 0)
        X_train = X_train + np.random.normal(np.zeros(X_train.shape), pixelwise_std)
        #Renormalize:
        X_train = (X_train - np.nanmin(X_train)) / (np.nanmax(X_train) - np.nanmin(X_train))
    
    # Reshape data to match Keras conventions:
    X_train = X_train.reshape(-1, num_rows, num_cols, 1)
    X_test = X_test.reshape(-1, num_rows, num_cols, 1) 
    X_val = X_val.reshape(-1, num_rows, num_cols, 1)
    
    #Training the CNN
    
    #Define the CNN architecture:
    model = models.Sequential()
    model.add(layers.Conv2D(32, kernel_size = 3, activation='relu', input_shape=(num_rows, num_cols, 1)))
    model.add(layers.Conv2D(32, kernel_size = 3, activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, kernel_size = 3, activation='relu'))
    model.add(layers.Conv2D(64, kernel_size = 3, activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, kernel_size = 3, activation='relu'))
    model.add(layers.Conv2D(128, kernel_size = 3, activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Conv2D(256, kernel_size = 3, activation='relu'))
    #model.add(layers.Conv2D(256, kernel_size = 3, activation='relu'))
    #model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    ##Visualize & save model architecture
    #print(model.summary())
    ##Needs pydot & graphviz installed:
    #plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    
    # Optimization:
    #Compile model, defining loss function and optimizer:
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-5),
                  metrics=['acc'])#use accuracy as the evaluation metric
    
    
    # Early stopping, train model & track training history:
    #The training will use early stopping based on the validation loss:
    es = EarlyStopping(monitor='val_loss',
                       mode='min',
                       verbose=1,
                       patience=20,
                       restore_best_weights = True)
    
    #Train model & track training history
    history = model.fit(X_train, y_train, 
                        epochs=100,
                        batch_size=4,
                        shuffle=True,
                        validation_data=(X_val, y_val),
                        callbacks=[es]) 
    
    #Store some of the model's training history in easy to use variables for plotting
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    
    #Save model after training
    if include_validation == 1:
        if include_noise == 1:
            model.save('VesselDetectorVal'+str(run+1)+'of'+str(n_runs)+'Noisy.h5')
        else:
            model.save('VesselDetectorVal'+str(run+1)+'of'+str(n_runs)+'.h5')
        
    else:
        if include_noise == 1:
            model.save('VesselDetector'+str(run+1)+'of'+str(n_runs)+'Noisy.h5')
        else:
            model.save('VesselDetector'+str(run+1)+'of'+str(n_runs)+'.h5')
    
    # Plot validation & training accuracy per epoch:
    plt.figure()
    plt.plot(epochs, acc, color = 'blue', linestyle='-.', label='Training acc')
    plt.plot(epochs, val_acc, color = 'red', linestyle='-', label='Validation acc')
    plt.title('Training and validation accuracy of model '+str(run+1)+' of '+str(n_runs))
    plt.legend()
       
    #Plot validation & training loss per epoch:
    plt.figure()
    plt.plot(epochs, loss, 'blue', linestyle='-.', label='Training loss')
    plt.plot(epochs, val_loss, 'red',  linestyle='-', label='Validation loss')
    plt.title('Training and validation loss of model '+str(run+1)+' of '+str(n_runs))
    plt.legend()
    
    # Evaluation on test data
    
    # Store final model's predictions on the test set:
    y_pred = model.predict_classes(X_test)# predict classes for test set
    y_prob_pred = model.predict_proba(X_test)#[:,1]# predict class probabilities for test set 
    
    #Get confusion matrix (and isolate entries) on the test set:
    conf_mat = metrics.confusion_matrix(y_test, y_pred)
    TN[run] = conf_mat[0][0]
    FP[run] = conf_mat[0][1]
    TP[run] = conf_mat[1][1]
    FN[run] = conf_mat[1][0]
      
    # Show confusion matrix:
    print("Confusion matrix of model "+str(run+1)+" of "+str(n_runs))
    print(conf_mat)   
    
    #Compute accuracy, precision, recall, f1-score, Jaccard index & Cohen's Kappa:
    Accuracy[run] = (TP[run]+TN[run]) / (TP[run]+FN[run]+FP[run]+TN[run])
    Recall[run] = TP[run] / (TP[run]+FN[run])
    Precision[run] = TP[run] / (TP[run]+FP[run]) 
    F1_score[run] = 2*(Recall[run]*Precision[run]) / (Recall[run]+Precision[run])
    Jaccard_index[run] = TP[run] / (TP[run]+FN[run]+FP[run])
    
    Expected_Accuracy[run] = ((TN[run]+FP[run])*(TN[run]+FN[run])+(FN[run]+TP[run])*(FP[run]+TP[run])) / (TN[run]+TP[run]+FN[run]+FP[run])**2
    Cohens_Kappa[run] = (Accuracy[run] - Expected_Accuracy[run]) / (1 - Expected_Accuracy[run])
    
    #Compute Area Under ROC curve:
    #1. Using class predictions (i.e. after decisions):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    AUC_decisions[run] = metrics.auc(fpr, tpr)
    #2. Using scores (i.e. before decisions):
    AUC_scores[run] = metrics.roc_auc_score(y_test, y_prob_pred)
    #fpr_scores, tpr_scores, thresholds_scores = metrics.roc_curve(y_test, y_prob_pred)
    
    #Compute Brier score
    brier_score[run] = metrics.brier_score_loss(y_test, y_prob_pred) 

#Calculate average and std of evaluation measures on test set accross all runs:   
avg_TN = np.nanmean(TN, axis = 0)
avg_FP = np.nanmean(FP, axis = 0)
avg_TP = np.nanmean(TP, axis = 0)
avg_FN = np.nanmean(FN, axis = 0)
avg_Accuracy = np.nanmean(Accuracy, axis = 0)
avg_Recall = np.nanmean(Recall, axis = 0)
avg_Precision = np.nanmean(Precision, axis = 0)
avg_F1_score = np.nanmean(F1_score, axis = 0)
avg_Jaccard_index = np.nanmean(Jaccard_index, axis = 0)
avg_Expected_Accuracy = np.nanmean(Expected_Accuracy, axis = 0)
avg_Cohens_Kappa = np.nanmean(Cohens_Kappa, axis = 0)
avg_AUC_decisions = np.nanmean(AUC_decisions, axis = 0)
avg_AUC_scores = np.nanmean(AUC_scores, axis = 0)
avg_brier_score = np.nanmean(brier_score, axis = 0)

std_TN = np.nanstd(TN, axis = 0)
std_FP = np.nanstd(FP, axis = 0)
std_TP = np.nanstd(TP, axis = 0)
std_FN = np.nanstd(FN, axis = 0)
std_Accuracy = np.nanstd(Accuracy, axis = 0)
std_Recall = np.nanstd(Recall, axis = 0)
std_Precision = np.nanstd(Precision, axis = 0)
std_F1_score = np.nanstd(F1_score, axis = 0)
std_Jaccard_index = np.nanstd(Jaccard_index, axis = 0)
std_Expected_Accuracy = np.nanstd(Expected_Accuracy, axis = 0)
std_Cohens_Kappa = np.nanstd(Cohens_Kappa, axis = 0)
std_AUC_decisions = np.nanstd(AUC_decisions, axis = 0)
std_AUC_scores = np.nanstd(AUC_scores, axis = 0)
std_brier_score = np.nanstd(brier_score, axis = 0)

np.set_printoptions(precision=3)#decimal digits to print

#Print average and std of evaluation measures on test set accross all runs:
print("Test set results (average +/- standard error) across all "+str(n_runs)+" runs:")
print("-------Confusion matrix:-------")
print("TP: "+str(avg_TP)+"+/-"+str(std_TP/np.sqrt(n_runs))+"\t(Higher better)")
print("TN: "+str(avg_TN)+"+/-"+str(std_TN/np.sqrt(n_runs))+"\t(Higher better)")
print("FP: "+str(avg_FP)+"+/-"+str(std_FP/np.sqrt(n_runs))+"\t(Lower better)")
print("FN: "+str(avg_FN)+"+/-"+str(std_FN/np.sqrt(n_runs))+"\t(Lower better)")
print("-------Asymmetry:-------")
print("Expected Accuracy: "+str(avg_Expected_Accuracy)+"+/-"+str(std_Expected_Accuracy/np.sqrt(n_runs))+", i.e.  Accuracy of a classifier randomly assigning examples to the 2 classes (deviation from 0.5 indicates class imbalance).")
print("-------Classification evaluation measures:-------")
print("Accuracy: "+str(avg_Accuracy)+"+/-"+str(std_Accuracy/np.sqrt(n_runs))+"\t(Higher better)")
print("Recall: "+str(avg_Recall)+"+/-"+str(std_Recall/np.sqrt(n_runs))+"\t(Higher better)")
print("Precision: "+str(avg_Precision)+"+/-"+str(std_Precision/np.sqrt(n_runs))+"\t(Higher better)")
print("F1-score: "+str(avg_F1_score)+"+/-"+str(std_F1_score/np.sqrt(n_runs))+"\t(Higher better)")
print("Jaccard Index: "+str(avg_Jaccard_index)+"+/-"+str(std_Jaccard_index/np.sqrt(n_runs))+"\t(Higher better)")
print("Cohen's Kappa: "+str(avg_Cohens_Kappa)+"+/-"+str(std_Cohens_Kappa/np.sqrt(n_runs))+"\t(Higher better)")
print("AUC (using class predictions): "+str(avg_AUC_decisions)+"+/-"+str(std_AUC_decisions/np.sqrt(n_runs))+"\t(Higher better)")
print("AUC (using scores): "+str(avg_AUC_scores)+"+/-"+str(std_AUC_scores/np.sqrt(n_runs))+"\t(Higher better)")
print("-------Probability estimation evaluation:-------")
print("Brier Score: "+str(avg_brier_score)+"+/-"+str(std_brier_score/np.sqrt(n_runs))+"\t(Lower better)")
