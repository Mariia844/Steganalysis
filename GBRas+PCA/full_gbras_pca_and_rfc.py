from operator import truediv
from full_gbras_test import *
import numpy as np
import tensorflow as tf
import utils, joblib
from tqdm import tqdm
from sklearn import decomposition
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 137

def get_path(path):
    folder = os.path.dirname(path)
    if not os.path.exists(folder):
        os.makedirs(folder)
    return path

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def extract_features(model, images, target_path):
    if (os.path.exists(target_path)):
        return np.load(target_path)
    target_sub_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-3].output)
    # sub_models = [tf.keras.Model(inputs=model.input, outputs=curr_layer.output) for curr_layer in model.layers]
    feature_arr = None
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
    np.save(target_path, feature_arr)
    return feature_arr

def fit_pca(model : decomposition.PCA, features : np.ndarray, components_file: str, model_file: str):
    need_train = True
    if os.path.exists(model_file):
        model = joblib.load(model_file)
        need_train = False
        if os.path.exists(components_file):
            return model, np.load(components_file)
    X_train, X_test = train_test_split(features, test_size=0.5, random_state=RANDOM_STATE)
    if need_train:
        model.fit(X_train)
        joblib.dump(model, model_file)
    components = model.transform(X_test)
    np.save(components_file, components)
    return model, components
def evaluate_pca(model: decomposition.PCA, features: np.ndarray, file_to_save: str):
    if os.path.exists(file_to_save):
        return np.load(file_to_save)
    cmp = model.transform(features)
    np.save(file_to_save, cmp)
    return cmp

def train_random_forest(cover_components, stego_components, file_to_save, stats_object, stats_file):
    stego_components = stego_components[:len(cover_components)]
    data = np.vstack([cover_components, stego_components])
    data_len = cover_components.shape[0]
    labels = np.array([np.zeros(data_len), np.ones(data_len)]).flatten()
    X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size=0.7, random_state=RANDOM_STATE)
    classifier = RandomForestClassifier(max_depth=4, random_state=RANDOM_STATE)
    if os.path.exists(file_to_save):
        classifier = joblib.load(file_to_save)
    else:
        classifier.fit(X_train, y_train)
        joblib.dump(classifier, file_to_save)
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
    
    results = {
        'alpha_error': [alpha_error],
        'beta_error': [beta_error],
        'total_error': [total_error],
        'f1_score': [f1],
        'mcc': [mcc]
    }
    stats_object.update(results)
    write_res_to_file(stats_file, stats_object)
if __name__ == "__main__":
    ## Extract features
    model = GBRAS_Net()
    components_divisions = test_config['pca_components_division']
    features_shape = model.layers[-3].output.shape.as_list()
    features_len = features_shape[1] * features_shape[2] * features_shape[3]

    override_components = False
    override_step = 0
    try:
        if test_config['override_components_division']['enabled']:
            override_step = ['override_components_division']['step']
            override_components = True
    except:
        pass
    if override_components:
        components_count = [i for i in range(features_len, 1, -override_step)]
    else:
        components_count = [np.math.ceil(features_len / div) for div in components_divisions]
    
    target_features_file = test_config['features_file']

    Xc_ = load_cache_or_images('vision')
    for alg, levels in test_config['weights_files'].items():
        for model_level, level_file in levels.items():
            print(f'Extracting features for trained model on {alg} {model_level}')
            model_test_file = rf'E:\gbras_after_article\test_results\vision_{alg}_{model_level}.csv'
            weights_path = glob.glob(os.path.join(level_file, '*.hdf5'))[0]
            model.load_weights(weights_path)
            algorithm = alg
            features_file_path = get_path(target_features_file.format(algorithm=algorithm, level=model_level))
            feature_arr = extract_features(model, Xc_, features_file_path)
            
            for component_count in components_count:
                print('\tProcessing %s components' % component_count)
                features_file = get_path(test_config['pca_features_file'].format(algorithm=algorithm, level=model_level, components_count=component_count))
                model_file = get_path(test_config['pca_model_file'].format(algorithm=algorithm, level=model_level, components_count=component_count))
                pca = decomposition.PCA(component_count)
                pca, cover_components = fit_pca(pca, feature_arr, features_file, model_file)
                for level in test_config['levels']:

                    print(f'\t{algorithm}_{level}:')
                    # Xc_ = load_cache_or_images('vision')
                    Xs_ = load_cache_or_images(f'vision_{algorithm}_{level}')
                    try:
                        _, Xs_ = train_test_split(Xs_, test_size=0.5, random_state=RANDOM_STATE)
                    except Exception as e:
                        print('Error', e)
                    stego_features_file = get_path(test_config['stego_features_file'].format(
                        algorithm=algorithm, 
                        level=model_level, 
                        components_count=component_count,
                        stego_level=level))
                    stego_features = extract_features(model, Xs_, target_path=stego_features_file)

                    stego_comp_file = get_path(test_config['pca_stego_features_file'].format(
                        algorithm=algorithm, 
                        level=model_level, 
                        components_count=component_count,
                        stego_level=level))
                    stego_components = evaluate_pca(pca, stego_features, stego_comp_file)
                    rfc_results_file = get_path(test_config['rfc_results_file'].format(
                        algorithm=algorithm, 
                        level=model_level))
                    rfc_model_file = get_path(test_config['rfc_model_file'].format(
                        algorithm=algorithm, 
                        level=model_level,
                        components_count=component_count,
                        stego_level=level))
                    rfc_results = {
                        'algorithm': algorithm,
                        'level': level,
                        'components': component_count
                    }
                    train_random_forest(
                        cover_components, 
                        stego_components,
                        rfc_model_file,
                        rfc_results,
                        rfc_results_file)
                    
                    # result_file = test_config['results_folder'].format(algorithm=algorithm, level=level)

                    # y_pred = model.predict(X_test, batch_size=4, verbose=1)

                    # y_true =  tf.argmax(y_test, axis=1).numpy()
                    # y_pred = tf.argmax(y_pred, axis=1).numpy()

                    # # estimate errors
                    # label_cover, label_stego = [0, 1]
                    # # alpha_error - false acceptance rate
                    # # beta_error - false rejection rate
                    # # total_error,  f1_score, mcc
                    # from sklearn.metrics import f1_score, confusion_matrix, matthews_corrcoef
                    # f1 = f1_score(y_true, y_pred, average='binary')
                    # conf_matrix = confusion_matrix(y_true, y_pred, normalize='true')
                    # alpha_error = conf_matrix[label_cover, label_stego]
                    # beta_error = conf_matrix[label_stego, label_cover]
                    # total_error = np.array([alpha_error, beta_error]).mean()
                    # mcc = matthews_corrcoef(y_true, y_pred)

                    # print('\t\tEstimated metrics:')
                    # print('\t\tTotal Error - {}'.format(total_error))
                    # print('\t\tAlpha Error - {}'.format(alpha_error))
                    # print('\t\tBeta Error - {}'.format(beta_error))
                    # print('\t\tF1 score - {}'.format(f1))
                    # print('\t\tMCC - {}'.format(mcc))
                    # model_test_results = {
                    #     'algorithm': [],
                    #     'level': [],
                    #     'total_error': [],
                    #     'alpha_error': [],
                    #     'beta_error': [],
                    #     'f1_score': [],
                    #     'mcc': []
                    # }
                    # model_test_results['algorithm'].append(algorithm)
                    # model_test_results['level'].append(level)
                    # model_test_results['total_error'].append(total_error)
                    # model_test_results['alpha_error'].append(alpha_error)
                    # model_test_results['beta_error'].append(beta_error)
                    # model_test_results['f1_score'].append(f1)
                    # model_test_results['mcc'].append(mcc)

                    # write_res_to_file(model_test_file, model_test_results)