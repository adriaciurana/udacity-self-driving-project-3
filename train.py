import os
import pandas as pd
import random
import pickle
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import cv2
import json

import tensorflow as tf
from model_lib import *
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

CURR_PATH = os.path.abspath(".")
FOLDER_DATASET_PATH = os.path.join(CURR_PATH, "dataset")
TRAIN_DATASET_PATH = os.path.join(FOLDER_DATASET_PATH, 'train_split.pickle')
VAL_DATASET_PATH = os.path.join(FOLDER_DATASET_PATH, 'val_split.pickle')
TEST_DATASET_PATH = os.path.join(FOLDER_DATASET_PATH, 'test_split.pickle')

"""
======================================
DATASET CREATION
======================================
"""
if not os.path.exists(TRAIN_DATASET_PATH) or not os.path.exists(VAL_DATASET_PATH) or not os.path.exists(TEST_DATASET_PATH):
    dataset = []
    correction = 0.2
    for d in os.listdir(FOLDER_DATASET_PATH):
        dataset_folder = os.path.join(FOLDER_DATASET_PATH, d)
        
        if os.path.isdir(dataset_folder):
            dataset_csv_path = os.path.join(dataset_folder, 'driving_log.csv')
            dataset_csv = pd.read_csv(dataset_csv_path, delimiter=',', names=['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed'])

            params_path = os.path.join(dataset_folder, 'params.json')

            if os.path.exists(params_path):
                with open(params_path, 'r') as f:
                    dataset_params = json.load(f)
            else:
                dataset_params = {}

            for camera, steering in [('center', 0), ('left', correction), ('right', -correction)]:
                for index, row in dataset_csv.iterrows():
                    if row[camera] == 0:
                        continue

                    dataset.append({
                        'image_path': os.path.join(d, "IMG", os.path.basename(row[camera].strip())),
                        'steering': row['steering'] + steering,
                        'params': dataset_params
                    })

    indices = list(range(len(dataset)))
    random.shuffle(indices)
    train_num_samples = int(.7 * len(dataset))
    val_num_samples = int(0.1 * len(dataset))
    test_num_samples = len(dataset) - train_num_samples - val_num_samples

    train_split = [dataset[idx] for idx in indices[:train_num_samples]]
    val_split = [dataset[idx] for idx in indices[train_num_samples:train_num_samples+val_num_samples]]
    test_split = [dataset[idx] for idx in indices[train_num_samples+val_num_samples:]]

    with open(TRAIN_DATASET_PATH, 'wb') as f:
        pickle.dump(train_split, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(VAL_DATASET_PATH, 'wb') as f:
        pickle.dump(val_split, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(TEST_DATASET_PATH, 'wb') as f:
        pickle.dump(test_split, f, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open(TRAIN_DATASET_PATH, 'rb') as f:
        train_split = pickle.load(f)

    with open(VAL_DATASET_PATH, 'rb') as f:
        val_split = pickle.load(f)

    with open(TEST_DATASET_PATH, 'rb') as f:
        test_split = pickle.load(f)

"""
======================================
DATASET AUGMENTATION
======================================
"""
SEQ = iaa.Sequential([
    iaa.GaussianBlur(sigma=(0.0, 3.0)),
    iaa.Multiply((0.9, 1.2)), 
    iaa.Affine(
        rotate=(-10, 10),
        scale=(0.5, 1),
        shear=(-16, 16),
        cval=(0, 255)
    ),  
])

def mirror(im, steering):
    return im[:, ::-1], - steering

"""
======================================
DATASET GENERATOR
======================================
"""
def generator(batch_size, dataset, do_aug=True):
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    batch_idx = 0
    X = np.empty(shape=(batch_size, IM_SHAPE[0], IM_SHAPE[1], 3))
    y = np.empty(shape=(batch_size, 1))
    while True:
        for idx in indices:
            data = dataset[idx]
            
            # Load image
            im = cv2.imread(os.path.join(FOLDER_DATASET_PATH, data['image_path']))[..., ::-1]
            steering = data['steering']

            # Image crop
            im = im[IM_CROP[0][0]:IM_CROP[0][1], IM_CROP[1][0]:IM_CROP[1][1]]

            # Do aug
            if do_aug:
                # Non modifies steering value
                im = SEQ.augment_images([im], )[0]

                # Modifies steering value
                if np.random.rand() > 0.5:
                    im, steering = mirror(im, steering)

            # Steering normalization
            if 'jitter' in data['params']:
                steering += (np.random.uniform(-data['params']['jitter'], data['params']['jitter']) / 50)
            steering = np.clip(steering, -1, 1)

            #print(steering )
            #steering = np.clip(steering, -90, 90) / 90 # Maximal angle
            steering = (steering + 1) / 2
            # Put into batch
            X[batch_idx] = im
            y[batch_idx] = steering

            batch_idx += 1
            if batch_idx % batch_size == 0:
                yield preprocess_input(X), y
                batch_idx = 0


"""
======================================
TIME TO TRAIN ;)
======================================
"""
if __name__ == '__main__':
    CHECKPOINT_PATH = os.path.join(CURR_PATH, "checkpoints")
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    LOGS_PATH = os.path.join(CURR_PATH, "logs")
    os.makedirs(LOGS_PATH, exist_ok=True)

    model = get_model(IM_SHAPE)
    model.compile(
        loss='binary_crossentropy',
        metrics=['mean_absolute_error'],
        optimizer='adam'
    )
    model.fit(
        generator(32, train_split, do_aug=True), 
        epochs=4000,
        steps_per_epoch=200, 
        callbacks=[
            ModelCheckpoint(CHECKPOINT_PATH + '/model.{epoch:02d}-{val_loss:.2f}.hdf5', save_best_only=True, period=5),
            TensorBoard(log_dir=LOGS_PATH, 
                write_graph=True
            )
        ],
        validation_steps=50,
        validation_data=generator(16, val_split, do_aug=True),
    )
