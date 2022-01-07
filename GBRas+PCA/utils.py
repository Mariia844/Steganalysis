import glob

import cv2
from tqdm import tqdm
import cv2
import numpy
from skimage.util.shape import view_as_blocks
n = 512
def load_images(path_pattern, take = None):
    files=glob.glob(path_pattern)
    X=[]
    files = sorted(files)
    if (take is not None):
        files = files[:take]
    for f in tqdm(files, f'Loading {path_pattern}'):
        I = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        patches = view_as_blocks(I, (n, n))
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                X.append( [ patches[i,j] ] )
    X=numpy.array(X)
    return X