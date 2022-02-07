import json
import numpy as np


def parse_npz_files(data):
    features = data['features']
    edges = data['edges']
    target = data['target']

    return features, edges, target


def summarise_data(data):
    print(data.files)
    print(data.shape)

