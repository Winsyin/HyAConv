# Hypergraph Attention Convolutional Network for Predicting CircRNA-miRNA Associations
# Requirements
pytorch 1.9.0

torch-geometric 2.1.0

# File Annotation
hyg: constructed hypergraph

Before training the model, each file is unzipped to get the corresponding .pickle file

The following data files are available from https://github.com/Winsyin/MRHRL.

1. CMA: the raw data
2. 5fold_CV: the training set, validation set and the test set in the 5-fold cross validation experiment
3. miRNA_seq_similarity_kmer: miRNA sequence similarity matrix
4. circRNA_seq_similarity_circRNA2vec: circRNA sequence similarity matrix.

   When training the model, four files "part1-4" need to be merged into one file and named “circRNA_seq_similarity_circRNA2vec”


# How to train the HyAConv model
You can train the model of 5-fold cross-validation with a very simple way by the command blow:  

python main.py
