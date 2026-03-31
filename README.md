# PCA from Scratch

This project implements Principal Component Analysis (PCA) from scratch in Python using **NumPy** and **Pandas**, without relying on machine learning libraries such as scikit-learn.

The implementation follows the standard PCA pipeline:

* Mean centering the dataset
* Computing the covariance matrix
* Performing eigenvalue and eigenvector decomposition
* Selecting the top *k* principal components
* Projecting the data onto the reduced feature space

The model is tested on the **Iris dataset**, which is preloaded in scikit-learn.
To validate correctness, the results from the custom implementation are compared with the PCA implementation provided by scikit-learn.
