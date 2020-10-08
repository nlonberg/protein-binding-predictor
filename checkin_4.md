# Protein-Protein Interactions Predictor

```
|__ code
|   |__ 01_collection.ipynb   
|   |__ 02_cleaning.ipynb
|   |__ 03_clustering.ipynb
|   |__ 04_autocovariance.ipynb
|   |__ 05_conjoint_modules.ipynb
|   |__ 06_res2vec.ipynb
|   |__ 07_modeling.ipynb
|   |__ 08_predictions.ipynb
|   |__ helper_classes.ipynb
|__ data
|   |__ 
|__ images
|   |__ 
|__ README.md
|__ presentation.pdf
|__ sources.md

```

## Problem Statement
SARS-COV-2 has infected at least 30 million people worldwide. The global efforts to find a vaccine, treat COVID-19, and understand the virus can only be described as the largest scientific project every undertaken. One branch of this project is mapping interactions between SARS-COV-2 surface proteins to human membrane receptors. The goal of this project is to train a binary classifier to identify whether two inputted proteins interact or not. The model will be applied to recently sequenced SARS-COV-2 surface proteins to determine their likely human targets.

## Executive Summary
Protein-protein interactions are the essence of all biological function. Understanding them will advance science and medicine. Modeling them will save us time and money. A lot of research has been put into this complex and important problem. Some researchers use 3D images to predict protein-protein interactions. Others use 2D interpretations of protein's important structural motifs. We are interested in research done to predict PPIs based on proteins' primary structure amino acid sequence. The widespread promotion of protein sequencing has made primary structure the most universal data that exists on proteins. We will just use amino acid sequences because we want our model to be trained on a large dataset and available for recently discovered proteins like those on SARS-COV-2. We will explore varying techniques in feature extraction and deep learning to create a model that can accurately predict protein-protein interactions.

Deep learning is well-suited to discover many of the relationships between amino acid sequence and PPIs without us spelling it out. But to improve model performance, there have to be ways we can help it. Ways to reorganize our data. Tidbits of information we can give it. These processes are called feature extraction.

Feature extraction involves the preprocessing of our input data into predictive feature variables. It is the most time-consuming and difficult part of the machine learning pipeline. We will spend lots of time delving into different forms of feature extraction.

The primary forms of feature extraction we will perform are autocovariance, the conjoint triad method, and feature embedding. These methods are widely used for time series and natural language processing problems and are supported in literature as ways to help neural networks to understand relationships in sequence data.

Our dataset, containing pairs of proteins that interact and pairs of proteins that do not interact and their amino acid sequences, was downloaded from UniProt. The testing set of SARS-COV-2 spike proteins was also downloaded from UniProt.

We will bring in an additional dataset containing 7 physiochemical properties for each of the 20 amino acids. This data will help us cluster the amino acids into chemically similar groups to form conjoint amino acid modeules and will help us produce autocovariances for each sequence. The dataset comes from the supplementary files of Sun et. al. 

For the modeling stage, we will gridsearch hyperparameters to find the best configuration of neural network.

Lastly, we will perform predictive analysis on SARS-COV-2 spike proteins to determine which receptors they are likely to bind to.
