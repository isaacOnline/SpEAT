import os
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import aggregate_layer_embeddings, load_dataset_info, get_high_low
from plots.eats.weat.weat.test import Test
from plots.eats.utils import check_if_already_run, write_eat_result, load_embeddings
from embedding_extraction.embedding_aggregation import aggregate_sequence_embeddings


def perform_tests(model_name, target_voices, attribute_dataset, dimension, timestamp, npermutations = 50000, nlayers=13,
                  embedding_aggregation_method=('mean', 'mean')):

    if target_voices == 'british':
        target_dataset_full_name = 'Speaker Accent Archive (Young/Old Native British English Speakers)'
        dataset_name = 'british_young_old'
    elif target_voices == 'american':
        target_dataset_full_name = 'Speaker Accent Archive (Young/Old Native American English Speakers)'
        dataset_name = 'usa_young_old'
    else:
        raise ValueError


    if attribute_dataset == 'EU_Emotion_Stimulus_Set':
        gs = 3
        n_attribute_embd = 54
    elif attribute_dataset == 'morgan_emotional_speech_set':
        gs = 10
        n_attribute_embd = 60
    else:
        raise ValueError

    target_embeddings = load_embeddings(os.path.join('speech_accent_archive', dataset_name), model_name, nlayers, embedding_aggregation_method)
    speaker_info = load_dataset_info(os.path.join('speech_accent_archive',dataset_name))
    high_age = speaker_info[speaker_info['age_rank'] == 'high_age']
    low_age = speaker_info[speaker_info['age_rank'] == 'low_age']
    high_age_embd = target_embeddings[high_age['index']]
    low_age_embd = target_embeddings[low_age['index']]

    n_target_embd = len(high_age_embd)


    names = ['Young','Old','Pleasant','Unpleasant']


    if not check_if_already_run('Young vs. Old Accents',
            'Nosek et al.',
            names, n_target_embd, n_attribute_embd, target_dataset_full_name, attribute_dataset,
            attribute_dataset, model_name,
                                embedding_aggregation_method=embedding_aggregation_method):
        attribute_embeddings = load_embeddings(valence_model_dataset, model_name, nlayers, embedding_aggregation_method)
        valence_info = load_dataset_info(attribute_dataset, group_sizes=gs).reset_index(drop=True)
        high_valence = valence_info[valence_info['valence_rank'] == 'high_valence']
        low_valence = valence_info[valence_info['valence_rank'] == 'low_valence']
        high_valence_embd = attribute_embeddings[high_valence.index]
        low_valence_embd = attribute_embeddings[low_valence.index]

        seed = 79331598
        np.random.seed(seed)


        test = Test(low_age_embd, high_age_embd, high_valence_embd, low_valence_embd,
        names=names)
        p = test.p(npermutations)


        write_eat_result('Young vs. Old Accents',
                         'Nosek et al.',
                         names, n_target_embd, n_attribute_embd, target_dataset_full_name, attribute_dataset,
                         attribute_dataset, model_name,
                         npermutations=npermutations, speat_d=test.effect_size(), speat_p=p,
                         iat_d=1.42, embedding_aggregation_method=embedding_aggregation_method)


if __name__ =='__main__':
    to_test = [
        # ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'wav2vec2_base', 13, '1664257024'),
        # ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'wav2vec2_large_ls960', 25, '1663482418'),
        # ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'wav2vec2_large_ll60k', 25, '1663446589'),
        # ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'hubert_base_ls960', 13, '1661404253'),
        # ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'hubert_large_ll60k', 25, '1664498938'),
        # ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'hubert_xtralarge_ll60k', 49, '1663008589'),
        # ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'wavlm_base', 13, '1662278003'),
        # ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'wavlm_base_plus', 13, '1662122581'),
        # ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'wavlm_large', 25, '1662267816'),
        # ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'whisper_large_encoder', 33, '1666059780'),
        # ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'whisper_base_en_encoder', 7, '1666255677'),
        # ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'whisper_base_encoder', 7, '1665605059'),
        # ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'whisper_medium_en_encoder', 25, '1666389810'),
        # ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'whisper_medium_encoder', 25, '1666803677'),
        # ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'whisper_small_en_encoder', 13, '1666247229'),
        # ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'whisper_small_encoder', 13, '1666251448'),
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
        if valence_model_dataset == 'morgan_emotional_speech_set':
            target_voices = ['american']
        elif valence_model_dataset == 'EU_Emotion_Stimulus_Set':
            target_voices = ['british']
        else:
            raise ValueError
        for voice_type in target_voices:
            for layer_agg in ['mean',
                              'first','second','q1','q2','q3','penultimate','last', 'min', 'max'
                              ]:
                for seq_agg in [
                    'mean',
                    'first', 'second', 'q1', 'q2', 'q3', 'penultimate', 'last', 'min', 'max'

                ]:
                    perform_tests(model_name, voice_type, valence_model_dataset, dimension, timestamp,  npermutations=1000000,
                                  nlayers=nlayers, embedding_aggregation_method=(layer_agg, seq_agg))
