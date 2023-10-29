import os
import logging

import numpy as np
from scipy.special import comb

from utils import load_dataset_info
from plots.eats.weat.weat.test import Test
from plots.eats.utils import check_if_already_run, write_eat_result, load_embeddings


def perform_tests(model_name, attribute_dataset, dimension, timestamp, nlayers=13, embedding_aggregation_method=('mean', 'mean')):
    names = ['American Accent', 'Foreign Accent', 'Pleasant Audio Clips',
             f'Unpleasant Audio Clips ({valence_model_dataset})']

    if attribute_dataset == 'EU_Emotion_Stimulus_Set':
        gs = 3
        n_attribute_embd = 54
    elif attribute_dataset == 'morgan_emotional_speech_set':
        gs = 10
        n_attribute_embd = 60
    else:
        raise ValueError

    target_embeddings = load_embeddings(os.path.join('audio_iats', 'pantos_perkins'), model_name, nlayers, embedding_aggregation_method)
    speaker_info = load_dataset_info(os.path.join('audio_iats', 'pantos_perkins'))
    american = speaker_info[speaker_info['category'] == 'american']
    foreign = speaker_info[speaker_info['category'] == 'foreign']
    american_embd = target_embeddings[american.index]
    foreign_embd = target_embeddings[foreign.index]

    n_target_embd = len(foreign_embd)
    target_dataset = ('Pantos & Perkins recordings')

    if not check_if_already_run('American vs. Korean English Accents',
                                'Pantos & Perkins',
                                names, n_target_embd, n_attribute_embd, target_dataset, attribute_dataset,
                                attribute_dataset, model_name, embedding_aggregation_method=embedding_aggregation_method):
        attribute_embeddings = load_embeddings(attribute_dataset, model_name, nlayers, embedding_aggregation_method)

        valence_info = load_dataset_info(attribute_dataset, group_sizes=gs).reset_index(drop=True)
        high_valence = valence_info[valence_info[f'{dimension}_rank'] == f'high_{dimension}']
        low_valence = valence_info[valence_info[f'{dimension}_rank'] == f'low_{dimension}']
        high_valence_embd = attribute_embeddings[high_valence.index]
        low_valence_embd = attribute_embeddings[low_valence.index]

        seed = 82518857
        np.random.seed(seed)

        npermutations = int(comb(16, 8))

        test = Test(american_embd, foreign_embd, high_valence_embd, low_valence_embd,
                    names=names)
        p = test.p(npermutations)

        print('U.S. VS. FOREIGN', model_name, p, test.effect_size())

        write_eat_result('American vs. Korean English Accents',
                         'Pantos & Perkins',
                         names, n_target_embd, n_attribute_embd, target_dataset, attribute_dataset,
                         attribute_dataset, model_name,
                         npermutations=npermutations, speat_d=test.effect_size(), speat_p=p,
                         iat_d=0.32, embedding_aggregation_method=embedding_aggregation_method)


if __name__ == '__main__':
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
                perform_tests(model_name, valence_model_dataset, dimension, timestamp, nlayers=nlayers,
                              embedding_aggregation_method=(layer_agg, seq_agg))
