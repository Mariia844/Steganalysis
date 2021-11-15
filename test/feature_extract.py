
import os
import pickle
import tensorflow as tf
from glob import glob, iglob
from model import GBRAS_Net
import numpy as np
from imageio import imread
from utils import load_images
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def model_feature_extractor():
    model = GBRAS_Net()
    cover_dir = r'E:\ml\notebook_train\dataset\VISION\cover\*.pgm'
    weights_dir = r'E:\gbras\Steganalysis\logs\model_GBRAS-Net_ALASKA_HUGO_04bpp_2021-11-11_14-53-18\saved-model-010-0.9785.hdf5'
    model.load_weights(weights_dir)
    # get submodels
    target_sub_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-3].output)
    # sub_models = [tf.keras.Model(inputs=model.input, outputs=curr_layer.output) for curr_layer in model.layers]
    feature_arr = None
    images = load_images(cover_dir)
    images  = np.rollaxis(images,1,4)
    total_count = np.math.ceil(len(images) / 32)
    for image_batch in tqdm(batch(images, n=32), total=total_count, desc='Evaluating batches'):
        arr = image_batch
        features_batch = target_sub_model.predict(arr, batch_size=32)
        for f_batch in features_batch:
            if feature_arr is None:
                feature_arr = f_batch.flatten()
            else:
                feature_arr = np.vstack([feature_arr, f_batch.flatten()])
    np.save('features_vision_40bpp.npy', feature_arr)

def get_pca(filename):
    from sklearn import decomposition
    matrix = np.load(filename)
    print(matrix.shape)
    num_features = matrix.shape[1]
    n_components = np.math.ceil(num_features / 2)
    pca = decomposition.PCA(n_components)
    components = pca.fit_transform(matrix)
    print(components.shape)
    filename = "pca_%d_comp_%s.npy" % (n_components, os.path.splitext(filename)[0])
    np.save(filename, components)

def send_message(text, obj = None):
    if ('SEND_AI_TELEGRAM_NOTIFICATIONS' in os.environ and 'AI_TG_TOKEN' in os.environ and 'AI_TG_CHAT' in os.environ):
        import telebot
        import json
        chat_id = os.environ['AI_TG_CHAT']
        bot = telebot.TeleBot(os.environ['AI_TG_TOKEN'])
        bot.send_message(chat_id=chat_id, text=text)
        if (obj != None):
            bot.send_message(chat_id, 'Object: ' + json.dumps(obj, sort_keys=True, indent=2))

def extract_features(model, images, filename):
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
def pca_train(features_file, n_components, model_file, random_state=137):
    from sklearn import decomposition
    import pickle

    model_folder = os.path.split(model_file)[0]


    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    features = np.load(features_file)
    pca = decomposition.PCA(n_components)
    X_train, X_test = train_test_split(features, test_size=0.5, train_size=0.5, random_state=random_state)
    print(f'Fitting model {model_file}')
    pca.fit(X_train)
    with open(model_file, "wb") as f:
        pickle.dump(pca, f)
def pca_evaluate(model, features, components_path):
    comp_folder = os.path.split(components_path)[0]
    if not os.path.exists(comp_folder):
        os.makedirs(comp_folder)
    components = model.transform(features)
    np.save(components_path, components)
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

    cover_features_file = f'features\{dataset}.npy'
    if not os.path.exists(cover_features_file):
        cover_images = load_images(cover_dir)
        print('Extracting cover features')

        extract_features(model, cover_images, cover_features_file)
        del cover_images
        gc.collect()
    stego_features_file = f'features\{dataset}_{alg}_{level}.npy'

    if not os.path.exists(stego_features_file):
        stego_images = load_images(stego_dir)
        print('Extracting stego features')
        
        extract_features(model, stego_images, stego_features_file)
        del stego_images
        gc.collect()

    pca_model_file = f'pca_models\{dataset}.pkl'
    if not os.path.exists(os.path.split(pca_model_file)[0]):
        print('Training PCA model and extract features')

        pca_train(cover_features_file, 500, pca_model_file, random_state=RANDOM_STATE)

        send_message('PCA training completed')

    cover_comp_file = f'components\{dataset}.npy'
    stego_comp_file = f'components\{dataset}_{alg}_{level}.npy'
    stego_c_folder, cover_c_folder = os.path.split(stego_comp_file)[0], os.path.split(cover_comp_file)[0]
    pca_model = None
    with open(pca_model_file, "rb") as f:
        pca_model = pickle.load(f)
    if not os.path.exists(stego_c_folder) or not os.path.exists(cover_c_folder):
        stego_features = np.load(stego_features_file)
        _, stego_features = train_test_split(stego_features, test_size=0.5, random_state=RANDOM_STATE)
        pca_evaluate(pca_model, stego_features, stego_comp_file)
        cover_features = np.load(cover_features_file)
        _, cover_features = train_test_split(cover_features, test_size=0.5, random_state=RANDOM_STATE)
        pca_evaluate(pca_model, cover_features, cover_comp_file)

    #TODO: Classifier training

    from sklearn.ensemble import RandomForestClassifier
    from joblib import dump, load

    cover_features = np.load(cover_comp_file)
    features = np.vstack([cover_features, np.load(stego_comp_file)])
    features_len = cover_features.shape[0]
    labels = np.array([np.zeros(features_len), np.ones(features_len)]).flatten()
    X_train, X_test, y_train, y_test = train_test_split(features, labels, train_size=0.7, test_size=0.3, shuffle=True, random_state=RANDOM_STATE)
    classifier : RandomForestClassifier = None
    classifier = RandomForestClassifier(max_depth=4, random_state=RANDOM_STATE)
    classifier.fit(X_train, y_train)

    

    predictions = classifier.predict(X_test)
    from sklearn.metrics import f1_score, confusion_matrix, matthews_corrcoef
    y_true = y_test
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
    if not os.path.exists('classifiers'):
        os.makedirs('classifiers')
    dump(classifier, f'classifiers/rfc_{dataset}_{alg}_{level}.dump')
    
