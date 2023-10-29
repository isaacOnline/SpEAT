import os
import logging

import numpy as np
import pandas as pd

from utils import load_dataset_info
from plots.eats.weat.weat.test import Test
from plots.eats.utils import check_if_se_already_run, write_eat_se_result, load_embeddings


def perform_tests(model_name, attribute_dataset, dimension, timestamp, npermutations=10000, nlayers=13,
                  embedding_aggregation_method=('mean','mean')):
    logger = logging.getLogger()

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

    # Test 1
    target_embeddings = load_embeddings('human_synthesized', model_name, nlayers, embedding_aggregation_method)

    speaker_info = load_dataset_info('human_synthesized')

    for n_target_embd in range(2, len(speaker_info)//2 + 1, 2):
        speaker_info = load_dataset_info('human_synthesized')

        names = ['Human Voice', 'Synthesized Voice', f'Pleasant Audio Clips ({valence_model_dataset})',
                 f'Unpleasant Audio Clips ({valence_model_dataset})']

        target_dataset = 'Human Vs. Synthetic (SAA)'
        if not check_if_se_already_run('Human vs. Synthesized Speech',
                                       target_dataset,
                                       names, n_target_embd, n_attribute_embd, target_dataset, attribute_dataset,
                                       attribute_dataset, model_name):

            seed = 62300794
            np.random.seed(seed)

            effect_sizes= []
            logger.disabled = True

            for _ in range(npermutations):
                synthetic_samples = speaker_info[~speaker_info['human']].sample(n_target_embd, replace=True)
                human_samples = []
                for i, genders_info in pd.DataFrame(synthetic_samples['sex'].value_counts()).reset_index().iterrows():
                    human_speaker_info = speaker_info[speaker_info['human']]
                    gendered_info = human_speaker_info[human_speaker_info['sex'] == genders_info['index']]
                    human_samples += [gendered_info.sample(genders_info['sex'],replace=True)]
                human_samples = pd.concat(human_samples)

                human_embeddings = target_embeddings[human_samples['index'].values]
                synthesized_embeddings = target_embeddings[synthetic_samples['index'].values]

                test = Test(human_embeddings, synthesized_embeddings, high_valence_embd, low_valence_embd,
                            names=names)
                effect_sizes.append(test.effect_size())


            stdev = np.std(effect_sizes, ddof=1)
            var = np.var(effect_sizes, ddof=1)
            write_eat_se_result('Human vs. Synthesized Speech',
                            target_dataset,
                             names, n_target_embd, n_attribute_embd, target_dataset, attribute_dataset,
                             attribute_dataset, model_name,
                             npermutations=npermutations,var=var,stdev=stdev)

    # Test 2
    target_embeddings = load_embeddings(os.path.join('speech_accent_archive', 'male_female'), model_name, nlayers,
                                        embedding_aggregation_method)
    speaker_info = load_dataset_info(os.path.join('speech_accent_archive', 'male_female'))

    for n_target_embd in range(2, len(speaker_info) // 2 + 1, 2):

        target_dataset = 'Female vs. Male (SAA)'

        names = ['Female', 'Male', f'Pleasant Audio Clips ({valence_model_dataset})',
                 f'Unpleasant Audio Clips ({valence_model_dataset})']

        if not check_if_se_already_run('Female vs. Male Speech',
                                       target_dataset,
                                       names, n_target_embd, n_attribute_embd, target_dataset, attribute_dataset,
                                       attribute_dataset, model_name):
            seed = 14955756
            np.random.seed(seed)


            effect_sizes = []
            for _ in range(npermutations):
                sample_pairs = speaker_info['pair_id'].drop_duplicates().sample(n_target_embd, replace=True)
                sampled_info = speaker_info[speaker_info['pair_id'].isin(sample_pairs.values)]


                male = sampled_info[sampled_info['sex'] == 'male']
                female = sampled_info[sampled_info['sex'] == 'female']
                male_embeddings = target_embeddings[male['index']]
                female_embeddings = target_embeddings[female['index']]



                test = Test(female_embeddings, male_embeddings, high_valence_embd, low_valence_embd,
                        names=names)
                effect_sizes.append(test.effect_size())

            stdev = np.std(effect_sizes, ddof=1)
            var = np.var(effect_sizes, ddof=1)

            write_eat_se_result('Female vs. Male Speech',
                            'Female vs. Male (SAA)',
                             names, n_target_embd, n_attribute_embd, target_dataset, attribute_dataset,
                             attribute_dataset, model_name,
                             npermutations=npermutations,stdev=stdev,var=var)


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
        for layer_agg in ['mean']:
            for seq_agg in [
                'mean',
                # 'min',
                # 'max','last',
            ]:
                perform_tests(model_name, valence_model_dataset, dimension, timestamp, nlayers=nlayers,
                              embedding_aggregation_method=(layer_agg, seq_agg))
