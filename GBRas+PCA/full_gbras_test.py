import os, gc, glob

import pandas as pd
import utils
import numpy as np
from tensorflow.keras.utils import to_categorical
from math import ceil
from model import GBRAS_Net
import tensorflow as tf
model = GBRAS_Net()

TOTAL_IMAGES = 9977

base_path = r'E:\GBRAS+PCA extended by components'

test_config = {
    'alg': ['mipod'],
    'levels': [10, 20, 40],
    'weights_files': {
        # 'hugo': {
        #     # '10': r'E:\gbras\Steganalysis\logs\model_GBRAS-Net_VISION_HUGO-10bpp_2021-11-18_15-36-23',
        #     '20': f'{base_path}' + r'\trained_gbras_weights\hugo\20',
        #     '40': f'{base_path}' + r'\trained_gbras_weights\hugo\40',
        # },
        'mipod': {
            # '10': r'',
            '20': f'{base_path}' + r'\trained_gbras_weights\mipod\20',
            '40': f'{base_path}' + r'\trained_gbras_weights\mipod\40',
        }
    },
    'results_folder': f'{base_path}' + r'\trained_gbras_tests\vision_{algorithm}_{level}.csv',
    'features_file': f'{base_path}' + r'\trained_gbras_features\vision_{algorithm}_{level}_cover.npy',
    'stego_features_file': f'{base_path}' + r'\trained_gbras_features\vision_{algorithm}_{level}_stego_{stego_level}.npy',
    'pca_components_division': [2,4,8,16],
    'override_components_division': {
        'enabled': True,
        'step': 4
    },
    'pca_features_file': f'{base_path}' r'\trained_gbras_pca\vision_{algorithm}_{level}_{components_count}_cover_components.npy',
    'pca_stego_features_file': r'E:\gbras\trained_gbras_pca\vision_model_{algorithm}_{level}_{components_count}_components_stego_{stego_level}.npy',
    'pca_model_file': f'{base_path}' + r'\trained_gbras_pca\vision_{algorithm}_{level}_{components_count}_components.dump',
    'rfc_model_file': f'{base_path}' + r'\trained_gbras_rfc\vision_{algorithm}_{level}_{components_count}_components_stego_{stego_level}.dump',
    'rfc_results_file': f'{base_path}' + r'\trained_gbras_rfc\vision_{algorithm}_{level}.csv'
}


def load_cache_or_images(cache_name : str):
    try:
        cache_path = f'{cache_name}.npz'
        if (os.path.exists(cache_path)):
            return np.load(cache_path)['images']

        cover_folder = r'E:\ml\notebook_train\dataset\VISION\cover\*.pgm'
        stego_folders = {
            'mipod': r'E:\ml\datasets\vision\VISION_MiPOD_1times',
            'hugo': r'E:\ml\notebook_train\dataset\VISION\HUGO'
        }
        splitted = cache_name.split('_')
        last_position = splitted[-1]
        if (last_position.isnumeric()):
            alg = splitted[-2]
            stego_base_folder = stego_folders[alg]
            stego_folder = os.path.join(stego_base_folder, last_position, '*.pgm')
            images = utils.load_images(stego_folder, TOTAL_IMAGES)
            np.savez_compressed(cache_path, images=images)
        else:
            cover_name = cache_name
            assert cover_name == 'vision'
            images = utils.load_images(cover_folder, TOTAL_IMAGES)
            np.savez_compressed(cache_path, images=images)
        return images
    except Exception as e:
        print('Error while loading images npz')
        print(e)
def write_res_to_file(file, res):
    mode = 'w'
    header = True

    if os.path.exists(file):
        mode = 'a'
        header = False
    else:
        folder = os.path.dirname(file)
        if not os.path.exists(folder):
            os.makedirs(folder)
    df = pd.DataFrame(res)
    with open(file, mode) as f:
        df.to_csv(f, header=header, index=False)
    
if __name__ == "__main__":
    Xc_ = load_cache_or_images('vision')

    ds_len = len(Xc_)
    train_split = 0.4
    valid_split = 0.1
    test_split = 0.5

    train_idx = ceil(ds_len * train_split)
    valid_idx = ceil(ds_len * valid_split)
    test_idx = ceil(ds_len * test_split)
    train_bound_end = train_idx
    valid_bound_start = train_bound_end
    valid_bound_end = train_idx + valid_idx
    test_bound_start = valid_bound_end
    test_bound_end = valid_bound_end + test_idx

    # X_test  = np.concatenate([X_[test_bound_start:test_bound_end],X_[ds_len+test_bound_start:ds_len+test_bound_end]],axis=0)
    # y_test  = np.concatenate([Xt_[test_bound_start:test_bound_end],Xt_[ds_len+test_bound_start:ds_len+test_bound_end]],axis=0)
    # v = dict()
    # v.
    for alg, levels in test_config['weights_files'].items():
        for model_level, level_file in levels.items():
            print(f'Testing {alg} {model_level}')
            model_test_file = rf'E:\gbras\test_results\vision_{alg}_{model_level}.csv'
            weights_path = glob.glob(os.path.join(level_file, '*.hdf5'))[0]
            model.load_weights(weights_path)
            algorithm = alg
            for level in test_config['levels']:
                print(f'\t{algorithm}_{level}:')
                # Xc_ = load_cache_or_images('vision')
                Xs_ = load_cache_or_images(f'vision_{algorithm}_{level}')
                X_  = (np.vstack((Xc_, Xs_)))
                Xt_ = (np.hstack(([0]*len(Xc_), [1]*len(Xs_))))
                Xt_ = to_categorical(Xt_, 2)
                # print(Xt_.shape)
                X_  = np.rollaxis(X_,1,4)  #channel axis shifted to last axis
                X_test  = np.concatenate([X_[test_bound_start:test_bound_end],X_[ds_len+test_bound_start:ds_len+test_bound_end]],axis=0)
                y_test  = np.concatenate([Xt_[test_bound_start:test_bound_end],Xt_[ds_len+test_bound_start:ds_len+test_bound_end]],axis=0)
                del Xs_, X_, Xt_
                gc.collect()

                result_file = test_config['results_folder'].format(algorithm=algorithm, level=level)
                y_pred = model.predict(X_test, batch_size=4, verbose=1)

                y_true =  tf.argmax(y_test, axis=1).numpy()
                y_pred = tf.argmax(y_pred, axis=1).numpy()

                # estimate errors
                label_cover, label_stego = [0, 1]
                # alpha_error - false acceptance rate
                # beta_error - false rejection rate
                # total_error,  f1_score, mcc
                from sklearn.metrics import f1_score, confusion_matrix, matthews_corrcoef
                f1 = f1_score(y_true, y_pred, average='binary')
                conf_matrix = confusion_matrix(y_true, y_pred, normalize='true')
                alpha_error = conf_matrix[label_cover, label_stego]
                beta_error = conf_matrix[label_stego, label_cover]
                total_error = np.array([alpha_error, beta_error]).mean()
                mcc = matthews_corrcoef(y_true, y_pred)

                print('\t\tEstimated metrics:')
                print('\t\tTotal Error - {}'.format(total_error))
                print('\t\tAlpha Error - {}'.format(alpha_error))
                print('\t\tBeta Error - {}'.format(beta_error))
                print('\t\tF1 score - {}'.format(f1))
                print('\t\tMCC - {}'.format(mcc))
                model_test_results = {
                    'algorithm': [],
                    'level': [],
                    'total_error': [],
                    'alpha_error': [],
                    'beta_error': [],
                    'f1_score': [],
                    'mcc': []
                }
                model_test_results['algorithm'].append(algorithm)
                model_test_results['level'].append(level)
                model_test_results['total_error'].append(total_error)
                model_test_results['alpha_error'].append(alpha_error)
                model_test_results['beta_error'].append(beta_error)
                model_test_results['f1_score'].append(f1)
                model_test_results['mcc'].append(mcc)

                write_res_to_file(model_test_file, model_test_results)
            # for algorithm in test_config['alg']:
           
        



