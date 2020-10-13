DeepDTA: deep drug–target binding affinity prediction.
Hakime Öztürk, Arzucan Özgür, Elif Ozkirimli LINK
Deep learning can help predict Drug-Target binding interaction
Can be a binary classification (Interacts-Doesn’t Interact) or Regression (binding affinity)
“Deep learning with feature embedding for compound-protein interaction prediction” by Wan and Zheng has protein-protein binding dataset
Random Forest models oversimplify protein complex; Deep Learning favored
Use only sequences to predict pba
SMILES: Simplified Molecular Input Line Entry System
Turks use Kinase affinity binding affinity dataset	

Deep Learning for Computational Biology. 
Christof Angermueller, Tanel Pärnamaa, Leopold Parts, Oliver Stegle
LINK
The appeal of deep learning in computational biology is “the ability to derive predictive models without a need for strong assumptions about underlying mechanisms, which are frequently unknown or insufficiently defined.”
Machine Learning solves for relationships in the data without us having to know what they are beforehand
Convolutional architectures are well suited for multi‐ and high‐dimensional data, such as two‐dimensional images or abundant genomic data. 
Recurrent neural networks can capture long‐range dependencies in sequential data of varying lengths, such as text, protein or DNA sequences.
Article contains lots of advice on reducing variance, hyper-parameter tuning, etc.

A deep learning framework for improving protein interaction prediction using sequence properties.
Yi Guo,  Xiang Chen LINK
Why determining PPIs is important: biological activity is driven by a network of inter- and intra-cellular protein interactions.
“[The model,] iPPI integrates the amino acid properties and compositions of protein sequence into a unified prediction framework using a hybrid deep neural network.”
“The primary amino acid sequence remains the most complete type of data available for all proteins.”
“The systematic analysis of PPI interfaces has shown complementarity in shape and electrostatic properties of amino acids .”
Cluster amino acids into 7 groups based on AAIndex
AAIndex: database has 566 properties for the 20 amino acids; properties divided into 6 categories (alpha and turn propensities, beta propensity, composition, hydrophobicity, physicochemical properties)
Conjoint AAIndex Modules (CAM): cluster numbers replaced AAs and sliding window of three AAs computes frequencies of triads.
PPIScore(S) = logist(LSTM(ReLU(Dense(Encode(S)))))
Encode: Turns primary sequence into CAM
Dense&ReLU: Classic neural net node clamps negative values to zero and introduces non-linearity.
Long Short-Term Memory (LSTM): enable order dependence and learn adjacent/near-adjacent relationships.
Logistic Regression: computes interaction probabilities
Handles imbalanced class problem by bootstrapping and downsampling
Dropout regularization used to prevent overfitting
Used Random Search approach for hyper-parameter tuning including learning rate, layer size, regularization constant, output dimension of the layer, and dropout rate in the dense layer.



Predicting Protein-Protein Interactions from Primary Protein Sequences Using a Novel Multi-Scale Local Feature Representation Scheme and the Random Forest.
Zhu-Hong You, Keith C. C. Chan, Pengwei Hu LINK
Hypothesis: continuous amino acid segments with different segment lengths influence PPIs
Feature representation method represents multiscale continuous amino acid segments at the same time.
Use multiscale descriptor and random forest
Multiscale Descriptor (MSD): Given amino acids sequence GYYGCCGY
Reassign to cluster numbers: 12213312
Break into 4 sequences: 12 21 33 12
Tabulate all continuous sequences (1221, 122133, 2133, etc)
Generate descriptors for each sequence (AA compositions, AA transitions, and AA distributions.
Compile metrics into one input feature vector
Random Forest trained with feature vector

Predicting Protein-Protein Interactions from Matrix-Based Protein Sequence Using Convolution Neural Network and Feature-Selective Rotation Forest. 
Lei Wang, Hai-Feng Wang, San-Rong Liu, Xin Yan & Ke-Jian Song 
 
LINK
Combine Convolutional Neural Network with Feature-Selective Rotation Forest
“[the] protein sequence contains abundant information, it is also mixed with a lot of noise. In order to get a more precise representation, we use the deep learning CNN algorithm to extract its features.”
Position-Specific Scoring Matrix: Use an alignment database (SwissProt PSI-BLAST) to score each amino acid’s conserved-ness

An integration of deep learning with feature embedding for protein-protein interaction prediction.
Yu Yao, Xiuquan Du,Yanyu Diao, and Huaixu Zhu

An issue with many existing PPI predictors is the reliance on hand-designed features.
Representation Learning aims to automatically learn representations from raw data with hand-designed encoding.
Example of representation learning is Word2vec, a word embedding tool in NLP that discovers semantic relationships between words in the document
Paper proposes Res2vec for protein sequence feature generation
Residue representations in the form of 20 eigenvectors are generated using skip-gram Word2vec where a word is an individual residue
A protein sequence is replaced with a sequence of eigenvectors
Input is fixed to specific size and fed into Deep Neural Network (DNN)

Sequence-based prediction of protein protein interaction using a deep-learning algorithm. 
Tanlin Sun, Bo Zhou, Luhua Lai & Jianfeng Pei

Used stacked-encoder deep-learning algorithm to achieve 97% accuracy (cross-val) on PPI.
Only used primary structure amino acid sequence data.
Used two techniques for preprocessessing primary sequence: autocovariance (AC) and conjoint triad method (CT)
Autocovariance of amino acids at different positions relative to the positions of amino acids in the rest of the protein.
Conjoint triad method clusters amino acids based on dipole and side chain volumes and then replaced with their cluster number. A 3-aa window is slid across the sequence and the frequency of the three number combinations is calculated and turned into a representative vector
Stacked Autoencoder: “An autoencoder is an artificial neural network that applies an unsupervised learning algorithm which infers a function to construct hidden structures from unlabeled data.”
Predicting commercially available antiviral drugs that may act on the novel coronavirus (SARS-CoV-2) through a drug-target interaction deep learning model
Used previously trained MT-DTI model to determine affinity of antiviral drugs to coronavirus proteins
Uses both amino acid sequence and SMILE sequence as input
Uses BindingDB and DTC database for model training
Showed stronger performance than DeepDTA