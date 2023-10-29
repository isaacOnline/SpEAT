import os

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances


def load_aggregated_sequence_embeddings(model_name, layer, dataset_name):
    """
    Load embeddings that have been aggregated at the sequence level

    :param model_name:
    :param layer:
    :param dataset_name:
    :return:
    """
    load_dir = os.path.join('embeddings', dataset_name, model_name, f'layer_{layer}')
    sequence_aggregations = {}
    for agg_method in ['min', 'max', 'mean', 'last']:
        sequence_aggregations[agg_method] = np.load(os.path.join(load_dir, f'{agg_method}.npy'))
    return sequence_aggregations


def aggregate_layer_embeddings(model_name, dataset_name):
    """
    Aggregate the embeddings for a dataset, first across sequences, then across layers

    :param model_name: Name of model to aggregate for, e.g. hubert_base_ls960
    :param dataset_name:
    :return:
    """
    load_dir = os.path.join('embeddings', dataset_name, model_name)
    n_layers = len([f for f in os.listdir(load_dir) if 'layer_' in f])

    layer_embeddings = []
    for layer in range(n_layers):
        layer_embeddings += [load_aggregated_sequence_embeddings(model_name, layer, dataset_name)]

    all_aggs = {}
    for seq_agg in ['min', 'max', 'mean', 'last']:
        relevant_sequence_aggregates = np.stack([e[seq_agg] for e in layer_embeddings])

        all_aggs[f'sequence_{seq_agg}_layer_min'] = np.stack(np.min(relevant_sequence_aggregates, axis=0))
        all_aggs[f'sequence_{seq_agg}_layer_max'] = np.stack(np.max(relevant_sequence_aggregates, axis=0))
        all_aggs[f'sequence_{seq_agg}_layer_mean'] = np.stack(np.mean(relevant_sequence_aggregates, axis=0))
        all_aggs[f'sequence_{seq_agg}_layer_last'] = np.stack(relevant_sequence_aggregates[-1])

    return all_aggs


def create_distance_data(model_name, dataset_name):
    """
    Measure the distance between embeddings of recordings in a dataset

    Uses cosine distance. Will first aggregate using either the min, max, mean, or last element of the sequence
    within each layer, then will aggregate over layers (also using the min, max, mean or last).

    :param model_name:
    :param dataset_name:
    :return:
    """
    aggregated_embeddings = aggregate_layer_embeddings(model_name, dataset_name)

    pairwise_distances = {k: cosine_distances(v, v) for k, v in aggregated_embeddings.items()}

    save_dir = os.path.join('plots', f'permanova_{dataset_name}', model_name, 'distances')
    os.makedirs(save_dir, exist_ok=True)

    for agg_method, data in pairwise_distances.items():
        save_path = os.path.join(save_dir, f'{agg_method}.npy')
        np.save(save_path, data)


def load_dataset_info(dataset_name, group_sizes=50):
    """#TODO: DOCSTRING
    :param dataset_name:
    """
    if dataset_name == 'speech_accent_archive':
        np.random.seed(96518933)
        saa_speaker_info = pd.read_csv(os.path.join('data', 'speech_accent_archive', 'speaker_list.csv'))
        saa_speaker_info = saa_speaker_info[
            saa_speaker_info['has_mp3'] & (saa_speaker_info['language'] != 'synthesized')].reset_index(drop=True)
        saa_speaker_info = get_high_low(saa_speaker_info, 'age', method='top_n', n=group_sizes).sort_index()
        return saa_speaker_info

    elif dataset_name == 'EU_Emotion_Stimulus_Set':
        # load details on audio clips
        voice_info_path = os.path.join('data', 'EU_Emotion_Stimulus_Set', 'voice_file_info.csv')
        voice_info = pd.read_csv(voice_info_path)

        # Filter out recordings that weren't included in files given to me
        voice_info = voice_info[voice_info['file_exists']].reset_index(drop=True)

        # Also, only look at sentences that are considered "semantically neutral" in the dataset, so that we
        # do not mix content with auditory features
        semantically_neutral_info = voice_info[voice_info['semantically_neutral'] == 'yes'].copy()

        np.random.seed(431137)
        # Find the most/least valenced/intense/aroused
        dimensions = ['valence', 'intensity', 'arousal']
        for dimension in dimensions:
            semantically_neutral_info = get_high_low(semantically_neutral_info, dimension, method='by_speaker',n=group_sizes)
        return semantically_neutral_info


    elif dataset_name == 'morgan_emotional_speech_set':
        # load details on audio clips
        clip_info_path = os.path.join('data', dataset_name, 'clip_info.csv')
        clip_info = pd.read_csv(clip_info_path)

        # Filter out recordings that weren't included in files given to me
        clip_info = clip_info[clip_info['file_exists']].reset_index(drop=True)

        # Create a single speaker_id column
        clip_info['speaker_id'] = clip_info['Gender'] + clip_info['Talker'].astype(str)

        clip_info = clip_info.rename(columns={'Valence':'valence','Arousal':'arousal'})

        np.random.seed(431137)
        # Find the most/least valenced/intense/aroused
        dimensions = ['valence', 'arousal']
        for dimension in dimensions:
            clip_info = get_high_low(clip_info, dimension, method='by_speaker',n=group_sizes)
        return clip_info

    elif dataset_name == os.path.join('audio_iats','pantos_perkins'):
        full_info = pd.read_csv(os.path.join('data','audio_iats','pantos_perkins','clip_info.csv'))
        return full_info

    elif dataset_name == os.path.join('audio_iats','mitchell_et_al'):
        full_info = pd.read_csv(os.path.join('data','audio_iats','mitchell_et_al','audio_info.csv'))
        return full_info

    elif dataset_name == os.path.join('audio_iats','romero_rivas_et_al'):
        clip_info = pd.read_csv(os.path.join('data','audio_iats','romero_rivas_et_al','clip_info.csv'))
        return clip_info

    elif dataset_name == os.path.join('speech_accent_archive','british_young_old'):
        clip_info = pd.read_csv(os.path.join('data','speech_accent_archive','british_young_old','clip_info.csv'))
        return clip_info

    elif dataset_name == os.path.join('speech_accent_archive','usa_young_old'):
        clip_info = pd.read_csv(os.path.join('data','speech_accent_archive','usa_young_old','clip_info.csv'))
        return clip_info

    elif dataset_name == 'UASpeech':
        full_info = pd.read_csv(os.path.join('data','UASpeech','clip_info.csv'))
        return full_info

    elif dataset_name == 'coraal_buckeye_joined':
        full_info = pd.read_csv(os.path.join('data','coraal_buckeye_joined','clip_info.csv'))
        return full_info

    elif dataset_name == os.path.join('speech_accent_archive', 'us_korean'):
        full_info = pd.read_csv(os.path.join('data','speech_accent_archive', 'us_korean','clip_info.csv'))
        return full_info

    elif dataset_name == os.path.join('speech_accent_archive', 'male_female'):
        full_info = pd.read_csv(os.path.join('data','speech_accent_archive', 'male_female','clip_info.csv'))
        return full_info

    elif dataset_name == 'human_synthesized':
        full_info = pd.read_csv(os.path.join('data','human_synthesized','clip_info.csv'))
        return full_info

    elif dataset_name == 'TORGO':
        full_info = pd.read_csv(os.path.join('data', 'TORGO', 'clip_info.csv'))
        return full_info
    else:
        raise ValueError



def get_high_low(dataset, column,method='top_n', n=None, cutoffs=None):
    # Todo: Docstring
    dataset = dataset.sample(n=len(dataset), replace=False)
    if method == 'top_n':
        assert n is not None and cutoffs is None
        dataset[f'{column}_rank'] = np.where(
            dataset[column].rank(ascending=False, method='first') <= n,
            f'high_{column}',
            np.where(
                dataset[column].rank(ascending=True, method='first') <= n,
                f'low_{column}',
                None
            ))
    elif method == 'cutoffs':
        assert n is None and cutoffs is not None
        dataset[f'{column}_rank'] = np.where(
            dataset[column] <= cutoffs[0],
            f'low_{column}',
            np.where(
                dataset[column] >= cutoffs[1],
                f'high_{column}',
                None
            ))

        # Make sure the sizes are balanced
        dataset = dataset.reset_index()
        min_size = dataset.groupby(f'{column}_rank').size().min()
        sampled = dataset.groupby(f'{column}_rank').sample(min_size).reset_index(drop=True)[['index',f'{column}_rank']]
        dataset = dataset.drop(columns=f'{column}_rank').merge(sampled, how='left',on='index')
    elif method == 'by_speaker':
        low = dataset.sort_values(['speaker_id',column]).groupby('speaker_id').head(n)
        low[f'low_{column}'] = f'low_{column}'
        high = dataset.sort_values(['speaker_id',column]).groupby('speaker_id').tail(n)
        high[f'high_{column}'] = f'high_{column}'
        dataset = dataset.merge(low[f'low_{column}'], how='left',left_index=True,right_index=True)
        dataset = dataset.merge(high[f'high_{column}'], how='left',left_index=True,right_index=True)
        dataset[f'{column}_rank'] = np.where(
            dataset[f'low_{column}'] == f'low_{column}',
            f'low_{column}',
            np.where(
                dataset[f'high_{column}'] == f'high_{column}',
                f'high_{column}',
                None
            )
        )
        dataset = dataset.drop(columns=[f'high_{column}', f'low_{column}'])

    else:
        raise ValueError(f'Unknown method: {method}')


    dataset = dataset.sort_index()

    return dataset
