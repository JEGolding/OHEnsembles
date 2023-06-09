# General
import numpy as np
from os import mkdir, path, system

# Loading Data
import tensorflow_datasets as tfds

# Embedding
import umap

# Clustering
import hdbscan

# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def define_model(typ='Forest'):
    if typ == 'Forest':
        model = RandomForestClassifier(max_depth=4)
    elif typ == 'Logistic':
        model = LogisticRegression(random_state=42)
    return model

def distance_from_one_hot(vector):
    oh = np.zeros(vector.shape)
    oh[np.argmax(vector)] = 1.
    return np.sum(np.abs(oh-vector))

for dataset in ['mnist','fashion_mnist','kmnist']:
    if not path.exists(dataset):
        mkdir(f'{dataset}')
    
    print(f'Calculating dataset: {dataset}')
    ds, info = tfds.load(dataset, split=['train','test'], as_supervised=True, with_info=True)
    df = tfds.as_dataframe(ds[0], info)
    x_train = np.stack(df.image.values)/255
    y_train = np.stack(df.label.values)

    df = tfds.as_dataframe(ds[1], info)
    x_test = np.stack(df.image.values)/255
    y_test = np.stack(df.label.values)

    for rem in ['None']+list(range(10)):
        print(f'Removing category {rem} from {dataset}')
        if rem == 'None':
            x_train_rem = x_train
            y_train_rem = y_train
        else:
            x_train_rem = x_train[y_train!=rem]
            y_train_rem = y_train[y_train!=rem]

        reducer = umap.UMAP(
            n_neighbors=30,
            min_dist=0.0,
            n_components=2,
            random_state=42,
        )
        embedding = reducer.fit_transform(x_train_rem.reshape(x_train_rem.shape[0], 28*28))
        test_embedding = reducer.transform(x_test.reshape(x_test.shape[0], 28*28))
        print('Embeddings complete')

        clusterer = hdbscan.HDBSCAN(
            min_samples=10,
            min_cluster_size=500,
            prediction_data=True,
        )
        clusterer.fit(embedding)
        
        test_labels, strengths = hdbscan.approximate_predict(clusterer, test_embedding)
        
        clust_labels = set(clusterer.labels_) - set([-1])
        
        print(f'Clustering complete, {len(clust_labels)} clusters')

        for model_typ in ['Forest','Logistic']:
            models = dict()
            for clust_label in clust_labels:
                print(f'Training model {clust_label}')
                models[clust_label] = define_model(typ=model_typ)
                models[clust_label].fit(
                    x_train_rem.reshape(x_train_rem.shape[0], 28*28), 
                    np.array(clusterer.labels_ == clust_label).astype(int),
                )

            print('Running predictions')
            predicts = dict()
            for clust_label in clust_labels:
                if model_typ == 'Forest':
                    predicts[clust_label] = models[clust_label].predict(x_test.reshape(x_test.shape[0], 28*28))
                elif model_typ == 'Logistic':
                    predicts[clust_label] = models[clust_label].predict_proba(x_test.reshape(x_test.shape[0], 28*28))[:,1]

            pred_vector = [np.array([predicts[clust_label][i] for clust_label in clust_labels]) for i in range(x_test.shape[0])]
            dists = np.array([distance_from_one_hot(vector) for vector in pred_vector])

            train_predicts = dict()
            for clust_label in clust_labels:
                if model_typ == 'Forest':
                    train_predicts[clust_label] = models[clust_label].predict(x_train_rem.reshape(x_train_rem.shape[0], 28*28))
                elif model_typ == 'Logistic':
                    train_predicts[clust_label] = models[clust_label].predict_proba(x_train_rem.reshape(x_train_rem.shape[0], 28*28))[:,1]
                    
            train_pred_vector = [np.array([train_predicts[clust_label][i] for clust_label in clust_labels]) for i in range(x_train_rem.shape[0])]
            train_dists = np.array([distance_from_one_hot(vector) for vector in train_pred_vector])

            print('Saving outputs')
            if not path.exists(f'{dataset}/{model_typ}/{rem}'):
                mkdir(f'{dataset}/{model_typ}/{rem}')

            # UMAP and HDBSCAN results
            # Train
            np.save(f'{dataset}/{model_typ}/{rem}/embedding.npy', np.array(embedding))
            np.save(f'{dataset}/{model_typ}/{rem}/labels.npy', np.array(clusterer.labels_))

            # Test
            np.save(f'{dataset}/{model_typ}/{rem}/test_embedding.npy', np.array(test_embedding))
            np.save(f'{dataset}/{model_typ}/{rem}/test_labels.npy', np.array(test_labels))

            # Classifier results
            # Prediction vectors from classifiers
            np.save(f'{dataset}/{model_typ}/{rem}/pred_vector.npy', np.array(pred_vector))
            np.save(f'{dataset}/{model_typ}/{rem}/train_pred_vector.npy', np.array(train_pred_vector))

            # OHE distances
            np.save(f'{dataset}/{model_typ}/{rem}/dists.npy', dists)
            np.save(f'{dataset}/{model_typ}/{rem}/train_dists.npy', train_dists)
            
            
        system("printf '\a'")

system("printf '\a'")
system("printf '\a'")