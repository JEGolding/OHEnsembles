# General
import numpy as np
from os import mkdir, path

# Loading Data
import tensorflow_datasets as tfds

# Embedding
import umap

# Clustering
import hdbscan

# Models
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

VERBOSE=1

def define_model():
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy')
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

        models = dict()
        for clust_label in clust_labels:
            print(f'Training model {clust_label}')
            models[clust_label] = define_model()
            models[clust_label].fit(
                x_train_rem, 
                np.array(clusterer.labels_ == clust_label).astype(int),
                epochs=10,
                batch_size=128,
                verbose=VERBOSE,
            )

        print('Running predictions')
        predicts = dict()
        for clust_label in clust_labels:
            predicts[clust_label] = models[clust_label].predict(x_test, verbose=VERBOSE)
            
        pred_vector = [np.array([predicts[clust_label][i] for clust_label in clust_labels]) for i in range(x_test.shape[0])]
        dists = np.array([distance_from_one_hot(vector) for vector in pred_vector])

        train_predicts = dict()
        for clust_label in clust_labels:
            train_predicts[clust_label] = models[clust_label].predict(x_train_rem, verbose=VERBOSE)

        train_pred_vector = [np.array([train_predicts[clust_label][i] for clust_label in clust_labels]) for i in range(x_train_rem.shape[0])]
        train_dists = np.array([distance_from_one_hot(vector) for vector in train_pred_vector])

        print('Saving outputs')
        if not path.exists(f'{dataset}/CNN/{rem}'):
            mkdir(f'{dataset}/CNN/{rem}')
            
        # UMAP and HDBSCAN results
        # Train
        np.save(f'{dataset}/CNN/{rem}/embedding.npy', np.array(embedding))
        np.save(f'{dataset}/CNN/{rem}/labels.npy', np.array(clusterer.labels_))
        
        # Test
        np.save(f'{dataset}/CNN/{rem}/test_embedding.npy', np.array(test_embedding))
        np.save(f'{dataset}/CNN/{rem}/test_labels.npy', np.array(test_labels))
        
        # Classifier results
        # Prediction vectors from classifiers
        np.save(f'{dataset}/CNN/{rem}/pred_vector.npy', np.array(pred_vector))
        np.save(f'{dataset}/CNN/{rem}/train_pred_vector.npy', np.array(train_pred_vector))
        
        # OHE distances
        np.save(f'{dataset}/CNN/{rem}/dists.npy', dists)
        np.save(f'{dataset}/CNN/{rem}/train_dists.npy', train_dists)