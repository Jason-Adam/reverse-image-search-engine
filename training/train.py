import os
import pickle
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
extensions = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", "PNG"]
root_dir = "caltech101"


def extract_features(img_path: str, model):
    input_shape = (224, 224, 3)
    img = tf.keras.preprocessing.image.load_img(
        img_path, target_size=(input_shape[0], input_shape[1])
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    features = model.predict(preprocessed_img)
    flattened_features = features.flatten()
    normalized_features = flattened_features / np.linalg.norm(flattened_features)
    return normalized_features


def get_file_list(root_dir=root_dir) -> List[str]:
    file_list = []
    counter = 1
    for root, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if any(ext in filename for ext in extensions):
                file_list.append(os.path.join(root, filename))
                counter += 1

    print(f"found {len(file_list)} files")
    return file_list


def main():
    filenames = sorted(get_file_list())
    feature_list = []
    for i in range(len(filenames)):
        feature_list.append(extract_features(filenames[i], model))
        if (i + 1) % 50 == 0:
            print(f"processed {i + 1} images out of {len(filenames)}")

    print("saving features and filenames")
    pickle.dump(
        feature_list, open("training/data/features-caltech101-resnet.pkl", "wb")
    )
    pickle.dump(filenames, open("training/data/filenames-caltech101-resnet.pkl", "wb"))


if __name__ == "__main__":
    main()
