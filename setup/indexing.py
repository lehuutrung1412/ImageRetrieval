import h5py
import numpy as np


class Index:
    def __init__(self, name="../app/static/data/features.h5"):
        self.name = name

    def set(self, feats, names):
        with h5py.File(self.name, 'w') as h5f:
            h5f.create_dataset('features', data=feats)
            h5f.create_dataset('names', data=np.string_(names))

    def get(self):
        with h5py.File(self.name, 'r') as h5f:
            feats = h5f['features'][:]
            names = h5f['names'][:]
        return feats, names
