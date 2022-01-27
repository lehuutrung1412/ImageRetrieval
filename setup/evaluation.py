from typing import List
import os
import numpy as np
from feature_extraction import FeatureExtractor
from indexing import Index
from dimension_reduction import perform_pca_on_single_vector
from PIL import Image
from sklearn.neighbors import KDTree
from lshashpy3 import LSHash
import argparse
import faiss

ap = argparse.ArgumentParser()
ap.add_argument("--large", choices=['kdtree', 'lsh', 'faiss'], required=False, help="Large scale method")
ap.add_argument("--top", required=True, help="Number of ranked lists element")
args = vars(ap.parse_args())

img_path = '../app/static/data/images/'
gt_path = '../app/static/data/gt_files/'

extractor = FeatureExtractor()
features, names = Index(name='../app/static/data/features_pca.h5').get()
if args['large'] is not None:
    if args['large'] == 'kdtree':
        # Large scale with kd-tree
        features = KDTree(features)
    elif args['large'] == 'lsh':
        # Large scale with LSH
        lsh = LSHash(8, 2560)
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


def load_list(file_name: str):
    file_name = os.path.join(gt_path, file_name)
    return [e.strip() for e in open(file_name, 'r').readlines()]


def compute_ap(pos: List[str], amb: List[str], ranked_list: List[str]):
    ap = 0
    num_relevant = 0
    for i, e in enumerate(ranked_list):
        if e in amb:
            continue
        if e in pos:
            num_relevant += 1
            ap += num_relevant / (i+1)
    return ap / num_relevant


def read_query(query_name):
    return load_list("%s_query.txt" % query_name)[0].split(' ')[0]


def get_ranked_lists(file_name):
    num = int(args['top'])
    img = Image.open(os.path.join(img_path, file_name + '.jpg'))

    query = extractor.extract(img)
    # PCA
    query = perform_pca_on_single_vector(query, 5, 512)

    if args['large'] is not None:
        if args['large'] == 'kdtree':
            # Large scale search using kd-tree
            query = np.expand_dims(query, axis=0)
            dists, ids = features.query(query, k=num)
            dists = np.squeeze(dists, axis=0)
            ids = np.squeeze(ids, axis=0)
            results = [str(names[index_img], 'utf-8').split(".")[0] for index_img in ids]
        elif args['large'] == 'lsh':
            # Large scale search using LSH
            lsh_search = lsh.query(query, num_results=num)
            results = [str(name, 'utf-8').split(".")[0] for ((vec, name), dist) in lsh_search]
        elif args['large'] == 'faiss':
            # Large scale search using faiss
            query = np.expand_dims(query, axis=0)
            dists, ids = index_flat.search(query, 30)
            dists = np.squeeze(dists, axis=0)
            ids = np.squeeze(ids, axis=0)
            results = [str(names[index_img], 'utf-8').split(".")[0] for index_img in ids]
    else:
        # Normal calculate euclid distance
        dists = np.linalg.norm(features - query, axis=1)
        ids = np.argsort(dists)[:num]
        results = [str(names[index_img], 'utf-8').split(".")[0] for index_img in ids]

    return results


def compute_map():
    queries = ["defense", "eiffel", "invalides", "louvre", "moulinrouge", "museedorsay",
               "notredame", "pantheon", "pompidou", "sacrecoeur", "triomphe"]
    aps = []
    for query in queries:
        for i in range(1, 6):
            file_name = query + '_' + str(i)
            print(file_name, end=" ")
            ranked_lists = get_ranked_lists(read_query(file_name))
            pos_set = list(set(load_list("%s_good.txt" % file_name) + load_list("%s_ok.txt" % file_name)))
            junk_set = load_list("%s_junk.txt" % file_name)
            ap = compute_ap(pos_set, junk_set, ranked_lists)
            print(ap)
            aps.append(ap)
    return np.mean(aps)


if __name__ == '__main__':
    print(compute_map())
