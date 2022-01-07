
import os
import pickle
from re import S
import joblib
import tensorflow as tf
from glob import glob, iglob
from model import GBRAS_Net
import numpy as np
from imageio import imread
from utils import load_images
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def extract_features(model, images, filename, override = False):
    if (os.path.exists(filename) and not override):
        return np.load(filename)
    folder, name = os.path.split(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)
    target_sub_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-3].output)
    feature_arr = None
    images  = np.rollaxis(images,1,4)
    total_count = np.math.ceil(len(images) / 32)
    for image_batch in tqdm(batch(images, n=32), total=total_count, desc='Evaluating batches for ' + filename):
        arr = image_batch
        features_batch = target_sub_model.predict(arr, batch_size=32)
        for f_batch in features_batch:
            if feature_arr is None:
                feature_arr = f_batch.flatten()
            else:
                feature_arr = np.vstack([feature_arr, f_batch.flatten()])
    np.save(filename, feature_arr)
    return feature_arr

if __name__ == "__main__":
    import gc
    dataset = 'VISION'
    alg = 'HUGO'
    level = '40'
    RANDOM_STATE = 137
    model = GBRAS_Net()
    cover_dir = r'E:\ml\notebook_train\dataset' + f'\{dataset}' + r'\cover\*.pgm'
    stego_dir = r'E:\ml\notebook_train\dataset' + f'\{dataset}\{alg}\{level}\*.pgm'
    path_to_search = r'E:\gbras\Steganalysis\logs\model_GBRAS-Net_' + f'{dataset}_{alg}-{level}bpp*\saved-model-010*.hdf5'
    print('Searching:\n\t' + path_to_search)
    weights_file = glob(path_to_search)[0]
    model.load_weights(weights_file)

    pca_model : PCA = None
    with open('pca_models/VISION.pkl', "rb") as f:
        pca_model = pickle.load(f)
    classifier : RandomForestClassifier = joblib.load('classifiers/rfc_VISION_HuGO_40.dump')


    alaska_folder = r'E:\ml\notebook_train\dataset\VISION\HUGO' # r'E:\ml\datasets\alaska\ALASKA_HUGO_1times\HUGO_1times'
    alaska_cover = cover_dir # r'E:\ml\datasets\alaska\cover\*.tif'
    cover_features_path = f'ft_train/VISION.npy'
    cover_images = load_images(alaska_cover, 5000)
    cover_features = extract_features(model, cover_images, cover_features_path)
    cover_components_path = 'cmp_train/VISION.npy'
    cover_components : np.ndarray = None
    if not os.path.exists(cover_components_path):
        cover_components = pca_model.transform(cover_features)
        np.save(cover_components_path, cover_components)
    else:
        cover_components = np.load(cover_components_path)
    del cover_images
    gc.collect()

    results = {
        'level': [],
        'total_err': [],
        'alpha_err': [],
        'beta_err': [],
        'f1_score': [],
        'mcc': []
    }
    levels = ['30', '50']
    for level in levels: # os.listdir(alaska_folder):
        images_pattern = os.path.join(alaska_folder, level, '*.pgm')
        features_path = f'ft_train/VISION_HUGO_{level}.npy'
        features : np.ndarray = None
        if not os.path.exists(features_path):
            stego_images = load_images(images_pattern, 5000)
            features = extract_features(model, stego_images, features_path)
            del stego_images
            gc.collect()
        else:
            features = np.load(features_path)

        components_path = f'cmp_train/VISION_HUGO_{level}.npy'
        components : np.ndarray = None
        if not os.path.exists(components_path):
            components = pca_model.transform(features)
            np.save(components_path, components)
        else:
            components = np.load(components_path)
        c_len = components.shape[0]
        X = np.vstack([components, cover_components])
        y = np.array([np.ones(c_len), np.zeros(c_len)]).flatten()
        np.random.seed(RANDOM_STATE)
        random_indices = np.random.permutation(c_len * 2)
        X = X[random_indices]
        y = y[random_indices]

        predictions = classifier.predict(X)
        from sklearn.metrics import f1_score, confusion_matrix, matthews_corrcoef
        y_true = y
        y_pred = predictions
        label_cover, label_stego = 0, 1
        f1 = f1_score(y_true, y_pred, average='binary')
        conf_matrix = confusion_matrix(y_true, y_pred, normalize='true')
        alpha_error = conf_matrix[label_cover, label_stego]
        beta_error = conf_matrix[label_stego, label_cover]
        total_error = np.array([alpha_error, beta_error]).mean()
        mcc = matthews_corrcoef(y_true, y_pred)

        print('Estimated metrics:')
        print('Total Error - {}'.format(total_error))
        print('Alpha Error - {}'.format(alpha_error))
        print('Beta Error - {}'.format(beta_error))
        print('F1 score - {}'.format(f1))
        print('MCC - {}'.format(mcc))

        results['level'].append(level)
        results['alpha_err'].append(alpha_error)
        results['beta_err'].append(beta_error)
        results['f1_score'].append(f1)
        results['mcc'].append(mcc)
        results['total_err'].append(total_error)
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv('VISION_40bpp_on_VISION.csv', index=False)

