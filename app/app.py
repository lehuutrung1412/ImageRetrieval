from flask import Flask, render_template, request, jsonify
from flask_ngrok import run_with_ngrok
import time
import numpy as np
from PIL import Image
from sklearn.neighbors import KDTree
from lshashpy3 import LSHash
import argparse
import faiss

import sys
sys.path.append("..")
from setup.feature_extraction import FeatureExtractor
from setup.indexing import Index
from setup.dimension_reduction import perform_pca_on_single_vector

ap = argparse.ArgumentParser()
ap.add_argument("--large", choices=['kdtree', 'lsh', 'faiss'], required=False, help="Large scale method")
ap.add_argument("--feature", required=False, help="Features indexing file path")
ap.add_argument("--pca", required=False, help="Enable pca")
args = vars(ap.parse_args())

DATA_PATH = "/static/data/images/"
app = Flask(__name__)
run_with_ngrok(app)
extractor = FeatureExtractor()
index_path = './static/data/features_no_pca.h5'
if args['pca'] is not None:
    index_path = './static/data/features_pca.h5'
if args['feature'] is not None:
    index_path = args['feature']
features, names = Index(name=index_path).get()

if args['large'] is not None:
    if args['large'] == 'kdtree':
        # Large scale with kd-tree
        features = KDTree(features)
    elif args['large'] == 'lsh':
        # Large scale with LSH
        lsh = LSHash(8, features.shape[1], 2)
        for i in range(len(features)):
            lsh.index(features[i], extra_data=names[i])
    elif args['large'] == 'faiss':
        # Large scale with faiss
        index_flat = faiss.IndexFlatL2(features.shape[1])
        if faiss.get_num_gpus() > 0:
            # Using GPU
            res = faiss.StandardGpuResources()
            index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
        index_flat.train(features)
        index_flat.add(features)


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
            if args['pca'] is not None:
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
                elif args['large'] == 'faiss':
                    # Large scale search using faiss
                    query = np.expand_dims(query, axis=0)
                    dists, ids = index_flat.search(query, 30)
                    dists = np.squeeze(dists, axis=0)
                    ids = np.squeeze(ids, axis=0)
                    results = [(DATA_PATH + str(names[index_img], 'utf-8'), float(dists[i])) for i, index_img in
                               enumerate(ids)]
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
