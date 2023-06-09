# General
import numpy as np
from os import mkdir, path

# def distance_from_one_hot(vector):
#     return min(np.sum(vector), 1.)

def distance_from_one_hot(vector):
    return np.abs(np.sum(vector))

append = '_abs'

for dataset in ['mnist','fashion_mnist','kmnist']:
    for rem in ['None']+list(range(10)):
            
        pred_vector       = np.load(f'{dataset}/{rem}/pred_vector.npy')
        train_pred_vector = np.load(f'{dataset}/{rem}/train_pred_vector.npy')
        
        dists = np.array([distance_from_one_hot(vector) for vector in pred_vector])
        train_dists = np.array([distance_from_one_hot(vector) for vector in train_pred_vector])

        print(f'{dataset}/{rem}/')
        if not path.exists(f'{dataset}/{rem}'):
            mkdir(f'{dataset}/{rem}')
            
        # OHE distances
        np.save(f'{dataset}/{rem}/dists{append}.npy', dists)
        np.save(f'{dataset}/{rem}/train_dists{append}.npy', train_dists)
        
        
        # Do the same for non-CNN results
        for model_typ in ['Forest','Logistic']:
            print(f'{dataset}/{model_typ}/{rem}/')
            pred_vector       = np.load(f'{dataset}/{model_typ}/{rem}/pred_vector.npy')
            train_pred_vector = np.load(f'{dataset}/{model_typ}/{rem}/train_pred_vector.npy')

            dists = np.array([distance_from_one_hot(vector) for vector in pred_vector])
            train_dists = np.array([distance_from_one_hot(vector) for vector in train_pred_vector])

            if not path.exists(f'{dataset}/{rem}'):
                mkdir(f'{dataset}/{rem}')

            # OHE distances
            np.save(f'{dataset}/{model_typ}/{rem}/dists{append}.npy', dists)
            np.save(f'{dataset}/{model_typ}/{rem}/train_dists{append}.npy', train_dists)