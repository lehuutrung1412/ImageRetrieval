from feature_extraction import FeatureExtractor
from indexing import Index
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-path", required=True, help="Path to save features file")
args = vars(ap.parse_args())

feature = FeatureExtractor()
features, names = feature.load()
index = Index(name=args['path'])
index.set(feats=features, names=names)
