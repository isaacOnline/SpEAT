import os
import logging
import sys
import warnings

import numpy as np
from scipy.special import comb
from tqdm import tqdm

from utils import load_dataset_info
from plots.eats.weat.weat.test import Test
from plots.eats.utils import check_if_se_already_run, write_eat_se_result


def perform_tests(model_name, attribute_dataset, dimension, timestamp, npermutations=10000, nlayers=13):

    seed = 75876873

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    model_log_path = os.path.join('plots', 'standard_error', 'dysarthric_speech.log')
    fh = logging.FileHandler(model_log_path)
    formatter = logging.Formatter(
        '%(asctime)s (%(levelname)s): %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info(f'For embedding extraction, Using model {model_name}, Emotion Dataset {attribute_dataset}, '
                f'Dimension {dimension}, Timestamp {timestamp}')

    clip_info = load_dataset_info('UASpeech')
    split_file_name = clip_info['file_name'].str.rsplit('_')
    clip_info['sample_info'] = split_file_name.str.get(1) + '_' + split_file_name.str.get(2) + '_' + split_file_name.str.get(3)
    clip_info['id'] = clip_info['Speaker'].str.extract('([MF][0-9]+)')
    clip_info['embd_index'] = range(len(clip_info))

    target_embeddings = np.stack(
        [np.load(os.path.join('embeddings', 'UASpeech', model_name, f'layer_{i}', f'mean.npy')) for i in
         range(nlayers)], axis=2).sum(axis=2)

    if attribute_dataset == 'EU_Emotion_Stimulus_Set':
        gs = 3
        n_attribute_embd = 54
    elif attribute_dataset == 'morgan_emotional_speech_set':
        gs = 10
        n_attribute_embd = 60
    else:
        raise ValueError

    attribute_embeddings = np.stack(
        [np.load(os.path.join('embeddings', 'morgan_emotional_speech_set', model_name, f'layer_{i}', f'mean.npy')) for i
         in range(nlayers)], axis=2).sum(axis=2)

    valence_info = load_dataset_info(attribute_dataset, group_sizes=gs).reset_index(drop=True)
    high_valence = valence_info[valence_info[f'{dimension}_rank'] == f'high_{dimension}']
    low_valence = valence_info[valence_info[f'{dimension}_rank'] == f'low_{dimension}']
    high_valence_embd = attribute_embeddings[high_valence.index]
    low_valence_embd = attribute_embeddings[low_valence.index]

    np.random.seed(seed)
    n_attribute_embd = len(high_valence_embd)
    attribute_dataset_name = attribute_dataset
    target_dataset_name = 'UASpeech Recordings'

    names = ['Non-Dysarthric', 'Dysarthric',
             'Pleasant', f'Unpleasant']


    with tqdm(total=npermutations * 5) as pbar:
        for num_words_per_speaker in range(1, 6):
            n_target_embd = num_words_per_speaker * 11
            if not check_if_se_already_run(f'({names[0]} vs. {names[1]}) vs. ({names[2]} vs. {names[3]})',
                                           'Nosek et al.',
                                           names, n_target_embd, n_attribute_embd, target_dataset_name,
                                           attribute_dataset_name,
                                           attribute_dataset, model_name):
                effect_sizes = []
                for _ in range(npermutations):
                    unique_matched_samples = clip_info[['sample_info', 'id']].drop_duplicates().copy()
                    samples_to_take = unique_matched_samples.groupby(['id']).sample(num_words_per_speaker, replace=True)
                    sampled_info = samples_to_take.merge(clip_info, on = ['sample_info','id'])

                    dysarthric_embeddings = target_embeddings[sampled_info[sampled_info['type'] == 'dysarthric']['embd_index'].tolist()]
                    non_dysarthric_embeddings = target_embeddings[sampled_info[sampled_info['type'] == 'non_dysarthric']['embd_index'].tolist()]

                    logger.disabled = True
                    test = Test(non_dysarthric_embeddings, dysarthric_embeddings, high_valence_embd, low_valence_embd,
                                names=names)
                    logger.disabled= False
                    effect_sizes.append(test.effect_size())
                    pbar.update()

                stdev = np.std(effect_sizes, ddof=1)
                var = np.var(effect_sizes, ddof=1)
                logger.info(f'For sample size {len(dysarthric_embeddings)}, stdev of effect sizes for resampled '
                            f'embeddings is {stdev}')
                write_eat_se_result(f'({names[0]} vs. {names[1]}) vs. ({names[2]} vs. {names[3]})',
                                     'Nosek et al.',
                                     names, n_target_embd, n_attribute_embd, target_dataset_name, attribute_dataset_name,
                                     attribute_dataset, model_name,
                                     npermutations=npermutations,stdev=stdev,var=var)


if __name__ =='__main__':
    to_test = [
        ('morgan_emotional_speech_set', 'mean', 'valence', 'wavlm_base', 13, '1662262874'),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'wavlm_base_plus', 13, '1662081826'),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'wavlm_large', 25, '1662162141'),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'wav2vec2_large_ll60k', 25, '1663291063'),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'wav2vec2_large_ls960', 25, '1663367964'),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'wav2vec2_base', 13, '1664237818'),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'hubert_base_ls960', 13, '1661565382'),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'hubert_large_ll60k', 25, '1664405708'),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'hubert_xtralarge_ll60k', 49, '1662751606'),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'whisper_base_encoder', 7, '1665595739'),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'whisper_base_decoder', 7, '1665603675'),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'whisper_base_en_encoder', 7, '1666242267'),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'whisper_base_en_decoder', 7, '1666246777'),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'whisper_small_encoder', 13, '1666236347'),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'whisper_small_decoder', 13, '1666246053'),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'whisper_small_en_encoder', 13, '1666231576'),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'whisper_small_en_decoder', 13, '1666245417'),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'whisper_medium_encoder', 25, '1666728028'),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'whisper_medium_decoder', 25, '1666738776'),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'whisper_medium_en_encoder', 25, '1666374320'),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'whisper_medium_en_decoder', 25, '1666383843'),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'whisper_large_encoder', 33, '1665849979'),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'whisper_large_decoder', 33, '1665973628'),
    ]
    for valence_model_dataset, _, dimension, model_name, nlayers, timestamp in to_test:
        perform_tests(model_name, valence_model_dataset, dimension, timestamp, nlayers=nlayers)
