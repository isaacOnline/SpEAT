import os
import logging
import sys

import numpy as np
from scipy.special import comb

from utils import load_dataset_info
from plots.eats.weat.weat.test import Test
from plots.eats.utils import check_if_already_run, write_eat_result, load_embeddings


def perform_tests(model_name, attribute_dataset, dimension, timestamp, npermutations=1000000, nlayers= 13, embedding_aggregation_method=('mean', 'mean')):

    seed = 12228324



    clip_info = load_dataset_info('TORGO')
    target_embeddings = load_embeddings('TORGO', model_name, nlayers, embedding_aggregation_method)

    dysarthric_embeddings = target_embeddings[clip_info[clip_info['dysarthric'] == 'dysarthric'].index]
    non_dysarthric_embeddings = target_embeddings[clip_info[clip_info['dysarthric'] == 'control'].index]

    names = ['Non-Dysarthric', 'Dysarthric',
             'Pleasant', f'Unpleasant']

    if attribute_dataset == 'EU_Emotion_Stimulus_Set':
        gs = 3
        n_attribute_embd = 54
    elif attribute_dataset == 'morgan_emotional_speech_set':
        gs = 10
        n_attribute_embd = 60
    else:
        raise ValueError

    attribute_embeddings = load_embeddings(attribute_dataset, model_name, nlayers, embedding_aggregation_method)
    valence_info = load_dataset_info(attribute_dataset, group_sizes=gs).reset_index(drop=True)
    high_valence = valence_info[valence_info[f'{dimension}_rank'] == f'high_{dimension}']
    low_valence = valence_info[valence_info[f'{dimension}_rank'] == f'low_{dimension}']
    high_valence_embd = attribute_embeddings[high_valence.index]
    low_valence_embd = attribute_embeddings[low_valence.index]



    n_target_embd = len(dysarthric_embeddings)
    n_attribute_embd = len(high_valence_embd)
    attribute_dataset_name = attribute_dataset
    target_dataset_name = 'TORGO Recordings'
    if not check_if_already_run(f'({names[0]} vs. {names[1]}) vs. ({names[2]} vs. {names[3]})',
                                'Nosek et al.',
                                names, n_target_embd, n_attribute_embd, target_dataset_name, attribute_dataset_name,
                                attribute_dataset, model_name, embedding_aggregation_method=embedding_aggregation_method):
        np.random.seed(seed)


        test = Test(non_dysarthric_embeddings, dysarthric_embeddings, high_valence_embd, low_valence_embd,
                    names=names)
        p = test.p(npermutations)


        write_eat_result(f'({names[0]} vs. {names[1]}) vs. ({names[2]} vs. {names[3]})',
                             'Nosek et al.',
                             names, n_target_embd, n_attribute_embd, target_dataset_name, attribute_dataset_name,
                             attribute_dataset, model_name,
                             npermutations=npermutations,speat_d=test.effect_size(),speat_p=p,
                             iat_d=1.05, embedding_aggregation_method=embedding_aggregation_method)


if __name__ =='__main__':
    to_test = [
        ('morgan_emotional_speech_set', 'mean', 'valence', 'wav2vec2_base', 13, '1664237818'),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'wav2vec2_large_ls960', 25, '1663367964'),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'wav2vec2_large_ll60k', 25, '1663291063'),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'hubert_base_ls960', 13, '1661565382'),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'hubert_large_ll60k', 25, '1664405708'),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'hubert_xtralarge_ll60k', 49, '1662751606'),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'wavlm_base', 13, '1662262874'),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'wavlm_base_plus', 13, '1662081826'),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'wavlm_large', 25, '1662162141'),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'whisper_base_encoder', 7, '1665595739'),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'whisper_base_en_encoder', 7, '1666242267'),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'whisper_small_encoder', 13, '1666236347'),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'whisper_small_en_encoder', 13, '1666231576'),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'whisper_medium_encoder', 25, '1666728028'),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'whisper_medium_en_encoder', 25, '1666374320'),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'whisper_large_encoder', 33, '1665849979'),
    ]
    for valence_model_dataset, _, dimension, model_name, nlayers, timestamp in to_test:
        for layer_agg in ['mean',
                          'first', 'second', 'q1', 'q2', 'q3', 'penultimate', 'last', 'min', 'max'
                          ]:
            for seq_agg in [
                'mean',
                'first', 'second', 'q1', 'q2', 'q3', 'penultimate', 'last', 'min', 'max'

            ]:
                perform_tests(model_name, valence_model_dataset, dimension, timestamp, 1000000, nlayers=nlayers,
                              embedding_aggregation_method=(layer_agg, seq_agg))
