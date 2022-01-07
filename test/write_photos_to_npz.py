import utils
import numpy as np
import os
def gbras_save_to_npz(images_pattern, target_path, count=None):
    dir = os.path.dirname(target_path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    images = utils.load_images(images_pattern, take=count)
    if os.path.exists(target_path):
        os.remove(target_path)
    np.savez_compressed(target_path, images=images)

if __name__ == "__main__":
    IMAGES_PATTERN = r'E:\ml\notebook_train\dataset\VISION\HUGO\10\*.pgm'
    TARGET_PATH = 'result/vision_hugo_10.npz'
    gbras_save_to_npz(IMAGES_PATTERN, TARGET_PATH, 9976)
    