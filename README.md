# Vessel Detection from Satelite Images
Code for training & evaluating a simple Vessel Detector CNN.

Includes code for balancing the training data, for augmenting the training data or adding noise, training the CNN in TensorFlow-Keras, test set evaluation using various evaluation measures, calibration of probability estimates or threshold manipulation and various visualisations (filters, intermediate activations, Grad-CAM heat maps).

## Requirements
In its current form, the notebook requires being in the same folder with the "Chips" folder containing 2 subfolders: "vessels" & "nonvessels", each with the examples of the corresponding class.

The main code has been tested with the following packages: Scikit-learn 0.22, TensorFlow 2.0.0 & Keras 2.2.4-tf

The additional code for the visualization of the CNN filters & Grad-CAM heatmaps requires TensorFlow 1.10.0 & Keras 2.2.2

## Description
"Vessel Detection.ipynb" does only a training/test split & evaluates on test data. If all datapoints given are assumed to be used for training, use this. Default setup gets to ~ 90-92% Accuracy.

"Vessel Detection-Val.ipynb" is the same notebook with an additional training/validation split. If part of the dataset given is assumed to only be used for the final evaluation, use this. Default setup gets to ~ 86-87% Accuracy.

In the notebooks:
"Optional" (in bold): indicates parts of the notebook that can be ommited, but if not, they will have an effect on cells further down (e.g. data preprocessing / augmentation).

"TODO" (in red): indicates parts of the notebook that are to be modified, finalized or added.

"Visualization" (in blue) indicates parts of the notebook that is used only for visualization / printing of results, with no effect further down.

The notebook itself and the code within it contain all necessary documentation.

"Vessel Detection.py": repeats n_runs times a train-(optionally validation)-test split of the data, training and evaluating of the CNN. It saves all keras models produced and prints various evaluation measures (average & standard error* accross all runs).

*standard error = standard deviation / sqrt(n_runs) 

## Suggested settings

The CNN architecture & hyperparameter setup in the notebook were the optimal found during hyperparameter optimisation; the default values in the notebook are guaranteed to produce a model with good predictive performance.

It is suggested to rebalance the training set, but not to use data augmentation via transformations or add Gaussian noise (these can improve performance of the resulting models but have not been tested fully).

Best results are obtained w/o a validation split (run "Vessel Detection.ipynb"; the default hyperparameters were optimised in advance using one).

For faster training, use a learning rate of 10e-4,  early stopping with a patience of 10 epochs and a total of 50; for best results, use a learning rate of 10e-5 and early stopping with a patience of 20 and a total of 100 epochs.

In the current version, calibration / threshold shifting do not provide tangible benefits. Calibration with the proper Scikit-learn function requires that the model be trained using a Scikit-learn wrapper for Keras (i.e. training option 2).

All visualizations can be performed with loaded models trained on the same dataset. Currently no saved models are in the project's GitHub page due to size limitations but you can easily store/load them in your local copy.
