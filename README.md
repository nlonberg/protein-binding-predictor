# Protein-Protein Interactions Predictor

```
|__ code
|   |__ 01_collection.ipynb   
|   |__ 02_cleaning.ipynb
|   |__ 03_main.ipynb
|   |__ autocovariance.ipynb
|   |__ conjoint_triad_method.ipynb
|   |__ modeling.ipynb
|   |__ helper_classes.ipynb
|__ data
|   |__ train.csv
|   |__ train_cleaned.csv
|__ images
|   |__ confusion_matrix.png
|__ README.md
|__ capstone_presentation.pdf
|__ sources.md

```

## Problem Statement
The strength of machine learning as a system of analysis lies in its ability to understand complex relationships in data that lie beyond our intuition.  As a data scientist interested in biochemistry, I see the problem of understanding protein relationships as well-suited for machine learning. This General Assembly Capstone is a test of deep learning's capacity to predict protein-protein interactions based off of sequence data. Given a pair of proteins' amino acid sequences, predict if they will interact.

## Executive Summary
Protein-protein interactions are the essence of all biological function. Understanding them will advance science and medicine. Modeling them will save us time and money. A lot of research has been put into this complex and important problem. Some researchers use 3D images to predict protein-protein interactions. Others use 2D interpretations of protein's important structural motifs. We are interested in research done to predict PPIs based on proteins' primary structure amino acid sequence. The widespread promotion of protein sequencing has made primary structure the most universal data that exists on proteins. We will just use amino acid sequences because we want our model to be trained on a large dataset and available for recently discovered proteins like those on SARS-COV-2. We will explore varying techniques in feature extraction and deep learning to create a model that can accurately predict protein-protein interactions.

Deep learning is well-suited to discover many of the relationships between amino acid sequence and PPIs without us spelling it out. Our decision to use deep learning reinforces our commitment to building an accurate model regardles of its inferential capability. To improve model performance, there have to be ways we can help it. Ways to reorganize our data. Tidbits of information we can give it. These processes are called feature extraction.

Feature extraction involves the preprocessing of our input data into predictive feature variables. It is the most time-consuming and difficult part of the machine learning pipeline. We will spend lots of time delving into different forms of feature extraction.

The primary forms of feature extraction we will perform are autocovariance and the conjoint triad method. These methods are inspired by the work of Sun et. al.

Our dataset, containing pairs of proteins that interact and pairs of proteins that do not interact and their amino acid sequences, was downloaded from UniProt.

We brought in an additional dataset containing 7 physiochemical properties for each of the 20 amino acids.

For the modeling stage, we gridsearched hyperparameters to find the best configuration of neural network.

Our final model was Convolutional Neural Network with Conjoint Triad Method sequence encoding.  The model predicted PPIs with 76% prediction accuracy. This is a 19% improvement on the baseline accuracy of 57%. Additional hyperparameter tuning and feature extraction should be performed, but this is a strong start.