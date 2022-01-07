import os
from utils import load_images
import numpy as np
from tensorflow.keras.utils import to_categorical
from math import ceil
from model import GBRAS_Net
import tensorflow as tf
model = GBRAS_Net()


# model.summary()


model_path = r'E:\gbras\Steganalysis\logs\model_GBRAS-Net_ALASKA_HUGO_04bpp_2021-11-11_14-53-18\saved-model-010-0.9785.hdf5'

model.load_weights(model_path)

pathc = r'E:\ml\notebook_train\dataset\VISION\cover'
paths = r'E:\ml\notebook_train\dataset\VISION\HUGO\40'

cache_name = r'.\cover_alaska.npy'




if (os.path.exists(f'.\{cache_name}')):
    print('Loading cached cover images')
    Xc_ = np.load(cache_name)
else:
    Xc_ = load_images(pathc+'\*.pgm') ##COVER IMAGES
    np.save(cache_name, Xc_)
stego_cache = r'.\alaska_hugo_40.npy'
if (os.path.exists(stego_cache)):
    print('Loading cached stego images')
    Xs_ = np.load(stego_cache)
else:
    Xs_ = load_images(paths+'\*.pgm') ##STEGO IMAGES
    np.save(stego_cache, Xs_)

X_  = (np.vstack((Xc_, Xs_)))
Xt_ = (np.hstack(([0]*len(Xc_), [1]*len(Xs_))))
Xt_ = to_categorical(Xt_, 2)
print(Xt_.shape)
X_  = np.rollaxis(X_,1,4)  #channel axis shifted to last axis

print("Total image data and labels",X_.shape,Xt_.shape)
#Cover hasta las 10000 ##Train hasta las 4000 ##Valid hasta de las 4000 a las 5000 ##Test de las 5000 a las 10000

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

print('DS len: {0}, train: 0:{1}, valid: {1}:{2}, test: {2}:{3}'.format(ds_len, train_bound_end, valid_bound_end, test_bound_end))


X_test  = np.concatenate([X_[test_bound_start:test_bound_end],X_[ds_len+test_bound_start:ds_len+test_bound_end]],axis=0)
y_test  = np.concatenate([Xt_[test_bound_start:test_bound_end],Xt_[ds_len+test_bound_start:ds_len+test_bound_end]],axis=0)
#Controled randomized data for training

print(X_test.shape)
print(y_test.shape)

# evaluate on test data
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

print('Estimated metrics:')
print('Total Error - {}'.format(total_error))
print('Alpha Error - {}'.format(alpha_error))
print('Beta Error - {}'.format(beta_error))
print('F1 score - {}'.format(f1))
print('MCC - {}'.format(mcc))