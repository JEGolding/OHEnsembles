Experimentation around using ensemble classifiers to identify out of domain data.

Classifiers are trained on each cluster found in the dataset to determine membership or not of that cluster. The ensemble results for each new sample is compared to the ideal (a One Hot encoding) to determine whether the sample is in domain or not.