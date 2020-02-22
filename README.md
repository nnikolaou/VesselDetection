# Vessel Detection from Satelite Images
Code for training & evaluating a simple Vessel Detector CNN. Includes code for balancing the training data, for augmenting the training data or adding noise, training the CNN in Keras, test set evaluation using various evaluation measures, calibration of probability estimates or threshold manipulation and various visualisations (filters, intermediate activations, Grad-CAM heat maps).

## Requirements
The main code has been tested with the following packages: Scikit-learn 0.22, TensorFlow 2.0.0 & Keras 2.2.4-tf
The additional code for the visualization of the CNN filters & Grad-CAM heatmaps requires TensorFlow 1.10.0 & Keras 2.2.2

## Description
__Optional__ : indicates parts of the notebook that can be ommited, but if not it will have an effect on cells further down
TODO (in red): indicates parts of the notebook that are to be modified, finalized or added
Visualization (in blue) indicates parts of the notebook that is used only for visualization / printing of results, with no effect further down

## Suggested settings
The CNN architecture & hyperparameter setup in the notebook were the opttimal found during hyperparameter optimisation
It is suggested to rebalance the training set, not to use data augmentation via transformations, not to add Gaussian noise;
Best results are obtained w/o a validation split (the hyperparameters were optimised in advance using one). For faster training use early stopping with a patience of 10 epochs and a total of 50; for best results use a patience of 20 and a total of 100 epochs; In the current version, calibration / threshold shifting do not provide tangible benefits.
