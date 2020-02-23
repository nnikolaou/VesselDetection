# Vessel Detection from Satelite Images
Code for training & evaluating a simple Vessel Detector CNN.

Includes code for balancing the training data, for augmenting the training data or adding noise, training the CNN in TensorFlow-Keras, test set evaluation using various evaluation measures, calibration of probability estimates or threshold manipulation and various visualisations (filters, intermediate activations, Grad-CAM heat maps).

## Requirements
### Data:
In their current form, the notebook & the script both require being in the same folder with the "Chips" folder containing 2 subfolders: "vessels" & "nonvessels", each with the examples of the corresponding class.

### Packages:
The main code has been tested with the following packages: Scikit-learn 0.22, TensorFlow 2.0.0 & Keras 2.2.4-tf

The additional code for the visualization of the CNN filters & Grad-CAM heatmaps in the notebook requires TensorFlow 1.10.0 & Keras 2.2.2

## Description

### Jupyter Notebook:
"Vessel Detector.ipynb" provides options for all functionalities described in the introduction. The notebook itself and the code within it contain all necessary documentation.

"Optional" (in bold): indicates parts of the notebook that can be ommited, but if not, they will have an effect on cells further down (e.g. data preprocessing / augmentation).

"TODO": indicates parts of the notebook that are to be modified, finalized or added.

"Visualization": indicates parts of the notebook that is used only for visualization / printing of results, with no effect further down.

### Python Script:
"Vessel Detector.py": repeats n_runs times a train/validation(optionally)/test split of the data, followed by training and evaluation of the CNN. It saves all Keras models produced and prints various evaluation measures (average & standard error* accross all runs).

*standard error = standard deviation / sqrt(n_runs) 

## Suggested settings

### Hyperparameters
The CNN architecture & hyperparameter setup in the notebook & script were the optimal found during hyperparameter optimisation; the default values in the notebook are guaranteed to produce a model with good predictive performance.

### Data preprocessing 
It is suggested to rebalance the training set, but not to use data augmentation via transformations or add Gaussian noise (these can improve performance of the resulting models but have not been tested fully).

### Train/Test vs. Train/Validation/Test split
If all datapoints in "Chips" are assumed to be used for training and a separate test set is provided, use this the Train/Test split. Default setup gets to ~ 91% Accuracy. If part of the dataset given is assumed to only be used for the final evaluation use a Train/Validation/Test split. Default setup gets to ~ 88% Accuracy.

### Fast training vs. best results
For faster training, use a learning rate of 10e-4,  early stopping with a patience of 10 epochs and a total of 50; for best results, use a learning rate of 10e-5 and early stopping with a patience of 20 and a total of 100 epochs.

### Calibration / threshold shifting
In the current version, calibration / threshold shifting do not provide tangible benefits. It can be improved if in the future we get access to more data / improve data augmentation. Calibration with the proper Scikit-learn function requires that the model be trained using a Scikit-learn wrapper for Keras (i.e. training option 2 in the "Vessel Detector.ipynb").

### Visualizations & loading stored models
All visualizations can be performed with loaded models trained on the same dataset. Currently no saved models are in the project's GitHub page due to size limitations but you can easily store/load them in your local copy.

## Sample Results

### Train/Validation/Test split - no augmentation - upsampling minority class - no noise added
Performing 5 random train/validation/test splits, training a CNN with default parameters and evaluating on test set yielded the following results (to reproduce, run "Vessel Detection.py" w/o changes):

Test set results (average +/- standard error) across all 5 runs:

-------Confusion matrix:-------

TP: [331.200]+/-[2.143]   (Higher better)
 
TN: [168.400]+/-[1.951]   (Higher better)

FP: [35.000]+/-[1.414]    (Lower better)

FN: [34.400]+/-[1.951]    (Lower better)

-------Asymmetry:-------

Expected Accuracy: [0.541]+/-[0.002], i.e. of a classifier randomly assigning examples to the 2 classes (deviation from 0.5 indicates class imbalance)

-------Classification evaluation measures:-------

Accuracy: [0.878]+/-[0.002]             		     (Higher better)

Recall: [0.906]+/-[0.005]               		     (Higher better)

Precision: [0.904]+/-[0.004]            		     (Higher better)

F1-score: [0.905]+/-[0.002]             		     (Higher better)

Jaccard Index: [0.827]+/-[0.003]        		     (Higher better)

Cohen's Kappa: [0.734]+/-[0.004].                (Higher better)

AUC (using class predictions): [0.867]+/-[0.002] (Higher better)

AUC (using scores): [0.944]+/-[0.004]            (Higher better)

-------Probability estimation evaluation:-------

Brier Score: [0.088]+/-[0.002]  (Lower better)

-------Epochs Traned:-------

Epochs Trained: 52.800 +/- 6.508
