import os

import numpy as np
import pandas as pd


def aggregate_sequence_embeddings(model_name, layer, dataset_name, base_dir=None):
    """
    Calculate aggregated embeddings for recordings

    Only aggregates within a single layer

    :param model_name: Name of model, e.g. hubert_base_ls960
    :param layer: Layer number
    :param dataset_name: Name of dataset to use
    :param base_dir: Base directory to use for loading and saving embeddings
    :return:
    """
    if base_dir is None:
        load_dir = os.path.join('embeddings', dataset_name, model_name, f'layer_{layer}')
    else:
        load_dir = os.path.join(base_dir, 'embeddings', dataset_name, model_name, f'layer_{layer}')
    len_path = os.path.join(load_dir, 'all_0_1.len')
    lens = pd.read_csv(len_path, header=None)

    embeddings_path = os.path.join(load_dir, 'all_0_1.npy')
    embeddings = np.load(embeddings_path)

    splits = np.cumsum(lens)[:-1]

    embeddings = np.split(embeddings, splits[0].tolist(), axis=0)

    sequence_aggregations = {
        'min': np.stack([np.min(e, axis=0) for e in embeddings]),
        'max': np.stack([np.max(e, axis=0) for e in embeddings]),
        'mean': np.stack([np.mean(e, axis=0) for e in embeddings]),
        'last': np.stack([e[-1, :] for e in embeddings]),
        'penultimate': np.stack([e[-2, :] for e in embeddings]),
        'first': np.stack([e[0, :] for e in embeddings]),
        'second': np.stack([e[1, :] for e in embeddings]),
        'q1': np.stack([e[int(len(e) * 0.25), :] for e in embeddings]),
        'q2': np.stack([e[int(len(e) * 0.5), :] for e in embeddings]),
        'q3': np.stack([e[int(len(e) * 0.75), :] for e in embeddings]),
    }

    del embeddings

    for agg_name, agg_data in sequence_aggregations.items():
        save_path = os.path.join(load_dir, f'{agg_name}.npy')
        np.save(save_path, agg_data)
