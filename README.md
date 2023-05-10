# Description
Experimentation around using ensemble classifiers to identify out of domain data.

Classifiers are trained on each cluster found in the dataset to determine membership or not of that cluster. The ensemble results for each new sample is compared to the ideal (a One Hot encoding) to determine whether the sample is in domain or not.

# Table of Contents
Check_Results.ipynb
	A notebook for investigating the results of the experiments.

Ensemble_Concept_Drift.py
	Script for running experiments. For each dataset and each label in the dataset the following files are made:
		dists.npy - for each test sample, the distance from OHE
		embedding.npy - for each training sample, the 2D UMAP embedding
		labels.npy - for each training sample, the HDBSCAN cluster label
		pred_vector.npy - for each training sample, the ensemble classifier output
		test_embedding.npy - for each test sample, the 2D UMAP embedding
		test_labels.npy - for each test sample, the HDBSCAN cluster label
		train_dists.npy - for each training sample, the distance from OHE
		train_pred_vector.npy - for each training sample, the ensemble classifier output
	
Mnist_Trial.ipynb
	First test notebook with proof of concept experimentation.
