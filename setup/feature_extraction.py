from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
import numpy as np
import glob
import os


class FeatureExtractor:
    def __init__(self):
        base_model = VGG16(weights='imagenet')
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

    def extract(self, img):
        """
        Extract a deep feature with shape=(4096, ) from an input image
        """
        img = img.resize((224, 224))  # VGG must take a 224x224 img as an input
        img = img.convert('RGB')  # Make sure img is color
        x = image.img_to_array(img)  # To np.array. Height x Width x Channel. type=float32
        x = np.expand_dims(x, axis=0)  # (H, W, C)->(1, H, W, C), where the first elem is the number of img
        x = preprocess_input(x)  # Subtracting avg values for each pixel
        feature = self.model.predict(x)[0]  # (1, 4096) -> (4096, )
        return feature / np.linalg.norm(feature)  # Normalize

    def load(self, path="../app/static/data/images/"):
        from dimension_reduction import perform_pca_on_single_vector
        features = []
        names = []
        for img_path in sorted(glob.glob(path + "*.jpg")):
            try:
                feature = self.extract(img=image.load_img(img_path))
            except:
                continue
            # PCA
            # feature = perform_pca_on_single_vector(feature, 5, 512)
            img_name = os.path.split(img_path)[1]
            features.append(feature)
            names.append(img_name)
        features = np.array(features)
        return features, names
