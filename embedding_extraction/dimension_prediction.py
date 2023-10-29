import gc
import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import trange

from utils import load_dataset_info

class EmbeddingDataset(Dataset):
    def __init__(self, dataset_name, dimension_name, model_name, num_layers, labels, lazy, device = None):
        self.dataset_name = dataset_name
        self.dimension_name = dimension_name
        self.model_name = model_name
        self.num_layers = num_layers
        self.use_labels = labels
        self.lazy = lazy
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        layer_dir = os.path.join('embeddings', dataset_name, model_name, f'layer_0')
        try:
            self.seq_lens = pd.read_csv(os.path.join(layer_dir, 'all_0_1.len'), header=None)[0].values
        except FileNotFoundError:
            layer_dir = os.path.join('/Volumes','Backup Plus', 'research','SpEAT', 'embeddings', 'embeddings', dataset_name,
                                     model_name, f'layer_0')
            self.seq_lens = pd.read_csv(os.path.join(layer_dir, 'all_0_1.len'), header=None)[0].values

        self.original_num_indexes = len(self.seq_lens)
        self.to_keep = get_rows_to_keep(self.dataset_name, np.arange(self.original_num_indexes))
        self.num_indexes = len(self.to_keep)
        self.end_idxs = np.cumsum(self.seq_lens)
        self.start_idxs = self.end_idxs - self.seq_lens
        self.load_embeddings(lazy)
        self.load_labels()



    def load_labels(self):
        if self.use_labels:
                self.labels, self.label_min, self.label_range = get_labels(self.dataset_name, self.dimension_name)
                self.labels = np.array(self.labels)
                self.labels = self.labels[self.to_keep]
                self.labels = (self.labels - self.label_min) / self.label_range

    def load_embeddings(self, lazy):
        mmap_mode = 'r' if lazy else None
        self.layers = []

        # Iterate through the layers
        for layer_num in range(self.num_layers):
            layer_dir = os.path.join('embeddings', self.dataset_name, self.model_name, f'layer_{layer_num}')
            try:
                # Read how long each sequence is.
                self.seq_lens = pd.read_csv(os.path.join(layer_dir, 'all_0_1.len'),header=None)[0].values
            except FileNotFoundError:
                layer_dir = os.path.join('/Volumes', 'Backup Plus', 'research', 'SpEAT', 'embeddings', layer_dir)
                self.seq_lens = pd.read_csv(os.path.join(layer_dir, 'all_0_1.len'), header=None)[0].values

            self.layers.append(np.load(os.path.join(layer_dir, f'all_0_1.npy'), mmap_mode=mmap_mode))

    def __len__(self):
        return self.num_indexes

    def __getitem__(self, original_idx):
        idx_in_original_data = self.to_keep[original_idx]
        start_idx_in_original_data = self.start_idxs[idx_in_original_data]
        end_idx_in_original_data = self.end_idxs[idx_in_original_data]
        stacked = np.stack([l[start_idx_in_original_data:end_idx_in_original_data] for l in self.layers])
        if self.use_labels:
            return stacked, self.labels[original_idx]
        else:
            return stacked



def get_labels(dataset_name, dimension_name):
    # Todo docstring
    if dataset_name == 'EU_Emotion_Stimulus_Set':
        labels = pd.read_csv(os.path.join('data','EU_Emotion_Stimulus_Set','voice_file_info.csv'))
        labels = labels[labels['file_exists']]
        labels = labels[dimension_name].tolist()
        min_score = 1
        max_score = 5
        return labels, min_score, max_score - min_score
    elif dataset_name == 'morgan_emotional_speech_set':
        labels = pd.read_csv(os.path.join('data','morgan_emotional_speech_set','clip_info.csv'))
        if dimension_name.title() not in labels.columns:
            raise ValueError(f'Dimension "{dimension_name}" not contained in dataset "{dataset_name}"')
        labels = labels[dimension_name.title()].tolist()
        min_score = 0
        max_score = 100
        return labels, min_score, max_score - min_score
    else:
        raise ValueError('Unknown dataset name')


def get_rows_to_keep(dataset_name, current_indexes):
    # Todo docstring
    if dataset_name == 'EU_Emotion_Stimulus_Set':
        rows_to_use = load_dataset_info(dataset_name, group_sizes=3).index.tolist()
        rows_to_use = [r for r in rows_to_use if r in current_indexes]
        return rows_to_use
    elif dataset_name == 'speech_accent_archive':
        rows_to_use = load_dataset_info('speech_accent_archive').index.tolist()
        rows_to_use = [r for r in rows_to_use if r in current_indexes]
        return rows_to_use
    elif dataset_name == os.path.join('audio_iats','pantos_perkins'):
        return current_indexes
    elif dataset_name == os.path.join('audio_iats','mitchell_et_al'):
        return current_indexes
    elif dataset_name == 'morgan_emotional_speech_set':
        return current_indexes
    elif dataset_name == os.path.join('audio_iats', 'romero_rivas_et_al'):
        return current_indexes
    elif dataset_name == os.path.join('speech_accent_archive', 'british_young_old'):
        return current_indexes
    elif dataset_name == os.path.join('speech_accent_archive', 'usa_young_old'):
        return current_indexes
    elif dataset_name == 'UASpeech':
        return current_indexes
    elif dataset_name == 'coraal_buckeye_joined':
        return current_indexes
    elif dataset_name == 'TORGO':
        return current_indexes
    elif dataset_name == 'human_synthesized':
        return current_indexes
    elif dataset_name == os.path.join('speech_accent_archive', 'male_female'):
        return current_indexes
    elif dataset_name == os.path.join('speech_accent_archive', 'us_korean'):
        return current_indexes
    else:
        raise ValueError



def collate_unequal_seqlens(batch):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = [torch.DoubleTensor(item[0]).to(device) for item in batch]
    target = [item[1] for item in batch]
    target = torch.DoubleTensor(target).to(device)
    return [data, target]


class DimensionPredictor(nn.Module):
    def __init__(self, nlayers, embedding_input_size, sequence_aggregation_method, intermediate_size=256, dtype=torch.double, categorical=False, device=None):
        super(DimensionPredictor, self).__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.nlayers = nlayers
        self.layer_sum_weights = nn.Parameter(torch.zeros(nlayers)).to(device)
        self.softmax = nn.Softmax(0).to(device)
        assert sequence_aggregation_method in ['min','max','mean']
        self.sequence_aggregation_method = sequence_aggregation_method
        self.device = device

        self.first_projection = nn.Linear(embedding_input_size, intermediate_size,dtype=dtype).to(device)
        output_dim = 2 if categorical else 1
        self.second_projection = nn.Linear(intermediate_size, output_dim, dtype=dtype).to(device)
        self.categorical = categorical


    def forward(self, x):

        # Take a weighted sum of the data
        weights = self.softmax(self.layer_sum_weights)
        averaged_over_layers = [(x_i.view(self.nlayers, -1) * weights[:, None]).view(x_i.shape).sum(axis=0) for x_i in x]

        # Project to a lower number of dimensions
        first_projections = [self.first_projection(x_i) for x_i in averaged_over_layers]

        # Take the aggregate of the projected vectors over each sequence
        if self.sequence_aggregation_method == 'mean':
            aggregated_over_sequence = torch.stack([x_i.mean(axis=0) for x_i in first_projections])
        elif self.sequence_aggregation_method == 'max':
            aggregated_over_sequence = torch.stack([x_i.max(axis=0)[0] for x_i in first_projections])
        elif self.sequence_aggregation_method == 'min':
            aggregated_over_sequence = torch.stack([x_i.min(axis=0)[0] for x_i in first_projections])
        else:
            raise ValueError

        # Project down to a single dimension, for use in predicting valence
        predictions = self.second_projection(aggregated_over_sequence)
        if self.categorical:
            predictions = torch.softmax(predictions, dim=1)
        else:
            predictions = predictions.flatten()

        return predictions

    def extract_embeddings(self, x):
        # Take a weighted sum of the data
        weights = self.softmax(self.layer_sum_weights)
        aggregated_over_sequence = []
        for i in trange(len(x), smoothing=0):
            x_i = torch.DoubleTensor(x[i]).to(weights.device)
            averaged_over_layers = (x_i.view(self.nlayers, -1) * weights[:, None]).view(x_i.shape).sum(axis=0)

            # Project to a lower number of dimensions
            first_projection = self.first_projection(averaged_over_layers)

            # Take the aggregate of the projected vectors over the sequence
            if self.sequence_aggregation_method == 'mean':
                aggregated_over_sequence.append(first_projection.mean(axis=0).cpu().detach().numpy())
            elif self.sequence_aggregation_method == 'max':
                aggregated_over_sequence.append(first_projection.max(axis=0)[0].cpu().detach().numpy())
            elif self.sequence_aggregation_method == 'min':
                aggregated_over_sequence.append(first_projection.min(axis=0)[0].cpu().detach().numpy())
            else:
                raise ValueError
            del x_i, averaged_over_layers, first_projection

        # Combine aggregated embeddings into a single tensor
        aggregated_over_sequence = np.stack(aggregated_over_sequence)


        return aggregated_over_sequence
