from flask import Flask, render_template, request, jsonify
from flask_ngrok import run_with_ngrok
import time
import numpy as np
from PIL import Image
from sklearn.neighbors import KDTree
from lshashpy3 import LSHash
import argparse

import sys
sys.path.append("..")
from setup.feature_extraction import FeatureExtractor
from setup.indexing import Index
from setup.dimension_reduction import perform_pca_on_single_vector

ap = argparse.ArgumentParser()
ap.add_argument("--large", choices=['kdtree', 'lsh'], required=False, help="Large scale method")
args = vars(ap.parse_args())

DATA_PATH = "/static/data/images/"
app = Flask(__name__)
run_with_ngrok(app)
extractor = FeatureExtractor()
features, names = Index(name='./static/data/features_pca.h5').get()

if args['large'] is not None:
    if args['large'] == 'kdtree':
        # Large scale with kd-tree
        features = KDTree(features)
    elif args['large'] == 'lsh':
        # Large scale with LSH
        lsh = LSHash(8, 2560)
        for i in range(len(features)):
            lsh.index(features[i], extra_data=names[i])


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            print('File was not found!')
        else:
            print('File was found!')
            img = request.files.get('file')

            start = time.time()
            img = Image.open(img.stream)

            query = extractor.extract(img)
            # PCA
            query = perform_pca_on_single_vector(query, 5, 512)

            if args['large'] is not None:
                if args['large'] == 'kdtree':
                    # Large scale search using kd-tree
                    query = np.expand_dims(query, axis=0)
                    dists, ids = features.query(query, k=30)
                    dists = np.squeeze(dists, axis=0)
                    ids = np.squeeze(ids, axis=0)
                    results = [(DATA_PATH + str(names[index_img], 'utf-8'), float(dists[i])) for i, index_img in enumerate(ids)]
                elif args['large'] == 'lsh':
                    # Large scale search using LSH
                    lsh_search = lsh.query(query, num_results=30)
                    print(lsh_search)
                    results = [(DATA_PATH + str(name, 'utf-8'), float(dist)) for ((vec, name), dist) in lsh_search]
            else:
                # Normal calculate euclid distance
                dists = np.linalg.norm(features - query, axis=1)
                ids = np.argsort(dists)[:30]
                results = [(DATA_PATH + str(names[index_img], 'utf-8'), float(dists[index_img])) for index_img in ids]

            time_span = time.time() - start
            return jsonify({'time': time_span, 'results': dict(results)})
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run()
