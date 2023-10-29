import os
import logging

import numpy as np

from utils import load_dataset_info
from plots.eats.weat.weat.test import Test
from plots.eats.utils import check_if_already_run, write_eat_result, load_embeddings


def perform_tests(model_name, attribute_dataset, dimension, timestamp, nlayers=13, embedding_aggregation_method=('mean','mean')):

    names = ['British Accent', 'Foreign Accent','Pleasant Audio Clips',f'Unpleasant Audio Clips ({valence_model_dataset})']

    if attribute_dataset == 'EU_Emotion_Stimulus_Set':
        gs = 3
        n_attribute_embd = 54
    elif attribute_dataset == 'morgan_emotional_speech_set':
        gs = 10
        n_attribute_embd = 60
    else:
        raise ValueError

    target_embeddings = load_embeddings(os.path.join('audio_iats','romero_rivas_et_al'), model_name, nlayers, embedding_aggregation_method)

    speaker_info = load_dataset_info(os.path.join('audio_iats','romero_rivas_et_al'))
    british = speaker_info[(speaker_info['stimuli_type'] == 'neutral') & (speaker_info['speaker_type'] == 'native')]
    foreign = speaker_info[(speaker_info['stimuli_type'] == 'neutral') & (speaker_info['speaker_type'] == 'foreign')]
    british_embd = target_embeddings[british.index]
    foreign_embd = target_embeddings[foreign.index]

    n_target_embd = len(british_embd)
    target_dataset = 'Romero Rivas et al recordings'



    if not check_if_already_run('British vs. Spanish English Accents',
                                'Romero Rivas et al.',
                                names, n_target_embd, n_attribute_embd, target_dataset, attribute_dataset,
                                attribute_dataset, model_name, embedding_aggregation_method):

        attribute_embeddings = load_embeddings(attribute_dataset, model_name, nlayers, embedding_aggregation_method)

        valence_info = load_dataset_info(attribute_dataset, group_sizes=gs).reset_index(drop=True)
        high_valence = valence_info[valence_info[f'{dimension}_rank'] == f'high_{dimension}']
        low_valence = valence_info[valence_info[f'{dimension}_rank'] == f'low_{dimension}']
        high_valence_embd = attribute_embeddings[high_valence.index]
        low_valence_embd = attribute_embeddings[low_valence.index]


        seed = 88193464
        np.random.seed(seed)

        npermutations = 1000000

        test = Test(british_embd, foreign_embd, high_valence_embd, low_valence_embd,
                    names=names)
        p = test.p(npermutations)

        write_eat_result('British vs. Spanish English Accents',
                         'Romero Rivas et al.',
                         names, n_target_embd, n_attribute_embd, target_dataset, attribute_dataset,
                         attribute_dataset, model_name,
                         npermutations=npermutations, speat_d=test.effect_size(), speat_p=p,
                         iat_d=0.60-(-0.46), embedding_aggregation_method=embedding_aggregation_method)


if __name__ =='__main__':
    to_test = [
        ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'hubert_base_ls960', 13, '1661404253'),
        ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'wavlm_base', 13, '1662278003'),
        ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'wavlm_base_plus', 13, '1662122581'),
        ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'wavlm_large', 25, '1662267816'),
        ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'hubert_xtralarge_ll60k', 49, '1663008589'),
        ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'wav2vec2_large_ll60k', 25, '1663446589'),
        ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'wav2vec2_large_ls960', 25, '1663482418'),
        ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'wav2vec2_base', 13, '1664257024'),
        ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'hubert_large_ll60k', 25, '1664498938'),
        ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'whisper_large_encoder', 33, '1666059780'),
        ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'whisper_base_en_encoder', 7, '1666255677'),
        ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'whisper_base_encoder', 7, '1665605059'),
        ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'whisper_medium_en_encoder', 25, '1666389810'),
        ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'whisper_medium_encoder', 25, '1666803677'),
        ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'whisper_small_en_encoder', 13, '1666247229'),
        ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'whisper_small_encoder', 13, '1666251448'),
    ]
    for valence_model_dataset, _, dimension, model_name, nlayers, timestamp in to_test:
        for layer_agg in ['mean',
                          'first', 'second', 'q1', 'q2', 'q3', 'penultimate', 'last', 'min', 'max'
                          ]:
            for seq_agg in [
                'mean',
                'first', 'second', 'q1', 'q2', 'q3', 'penultimate', 'last', 'min', 'max'

            ]:
                perform_tests(model_name, valence_model_dataset, dimension, timestamp, nlayers=nlayers, embedding_aggregation_method=(layer_agg, seq_agg))
