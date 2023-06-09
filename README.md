# Description  
Experimentation around using ensemble classifiers to identify out of domain data.  

Classifiers are trained on each cluster found in the dataset to determine membership or not of that cluster. The ensemble results for each new sample is compared to the ideal (a One Hot encoding) to determine whether the sample is in domain or not.  

# Table of Contents  
1_Check_Results.ipynb  
  * A notebook for investigating the results of the experiments.  

2_Check_Results_All_Types.ipynb
  * Same as above, but for all model types (CNN, Forest, Logistic).

3_Investigate_Ensemble_Vectors.ipynb
  * Investigate how in domain vs out of domain ensemble vectors differ.

Ensemble_Concept_Drift_CNN.py  
  * Script for running experiments. For each dataset and each label in the dataset the following files are made, using CNN models:  
    * dists.npy - for each test sample, the distance from OHE
    * embedding.npy - for each training sample, the 2D UMAP embedding  
    * labels.npy - for each training sample, the HDBSCAN cluster label  
    * pred_vector.npy - for each training sample, the ensemble classifier output  
    * test_embedding.npy - for each test sample, the 2D UMAP embedding  
    * test_labels.npy - for each test sample, the HDBSCAN cluster label  
    * train_dists.npy - for each training sample, the distance from OHE  
    * train_pred_vector.npy - for each training sample, the ensemble classifier output  
	
Ensemble_Concept_Drift_No_CNN.py
  * Same as above, but using Random Forest and Logistic Regression models instead of CNNs.

Sum_Dist_From_OHE.py
  * Script for trying different "distance from OHE" functions. Implemented is a sum of all vector elements. For each dataset and label and model, makes the following:
    * dists_sum.npy - for each test sample, the sum of the ensemble vector.
    * train_dists_sum.npy - for each training sample, the sum of the ensemble vector.

Mnist_Trial.ipynb  
  * First test notebook with proof of concept experimentation.  
