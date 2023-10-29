import os
import logging

import numpy as np
from scipy.special import comb
from tqdm import tqdm

from utils import load_dataset_info
from plots.eats.weat.weat.test import Test
from plots.eats.utils import check_if_se_already_run, write_eat_se_result

def perform_tests(model_name, attribute_dataset, dimension, timestamp, npermutations=10000, nlayers=13):

    names = ['American Accent', 'Foreign Accent','Pleasant Audio Clips',f'Unpleasant Audio Clips ({valence_model_dataset})']

    if attribute_dataset == 'EU_Emotion_Stimulus_Set':
        gs = 3
        n_attribute_embd = 54
    elif attribute_dataset == 'morgan_emotional_speech_set':
        gs = 10
        n_attribute_embd = 60
    else:
        raise ValueError

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    model_log_path = os.path.join('plots', 'standard_error', 'pp_audio_iat.log')
    fh = logging.FileHandler(model_log_path)
    formatter = logging.Formatter(
        '%(asctime)s (%(levelname)s): %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    target_embeddings = np.stack(
        [np.load(
            os.path.join('embeddings', 'audio_iats', 'pantos_perkins', model_name, f'layer_{i}', f'mean.npy'))
            for i in
            range(nlayers)], axis=2).sum(axis=2)
    speaker_info = load_dataset_info(os.path.join('audio_iats','pantos_perkins'))

    speaker_info['type'] = speaker_info['clip_name'].str.split('_').str.get(0)
    speaker_info['spkr'] = speaker_info['clip_name'].str.split('_').str.get(1)
    speaker_info['sample'] = speaker_info['clip_name'].str.split('_').str.get(2)
    speaker_info['sample'] = speaker_info['sample'].str.replace('frequeX3n', 'frequX3n').str.replace('225x3n','225X3n')
    speaker_info['index'] = range(len(speaker_info))

    match_info = speaker_info[['spkr','sample']].drop_duplicates()

    attribute_embeddings = np.stack(
        [np.load(os.path.join('embeddings', attribute_dataset, model_name, f'layer_{i}', f'mean.npy')) for i
         in range(nlayers)], axis=2).sum(axis=2)
    valence_info = load_dataset_info(attribute_dataset, group_sizes=gs).reset_index(drop=True)
    high_valence = valence_info[valence_info[f'{dimension}_rank'] == f'high_{dimension}']
    low_valence = valence_info[valence_info[f'{dimension}_rank'] == f'low_{dimension}']
    high_valence_embd = attribute_embeddings[high_valence.index]
    low_valence_embd = attribute_embeddings[low_valence.index]

    seed = 82518857
    np.random.seed(seed)
    logger.info(f'Using model {model_name}, Emotion Dataset {valence_model_dataset}, '
                f'Dimension {dimension}, Timestamp {timestamp}')
    with tqdm(total = npermutations * 4) as pbar:

        for num_samples in range(1, 5):
            n_target_embd = num_samples * 4
            target_dataset = ('Pantos & Perkins recordings')

            if not check_if_se_already_run('American vs. Korean English Accents',
                                        'Pantos & Perkins',
                                        names, n_target_embd, n_attribute_embd, target_dataset, attribute_dataset,
                                        attribute_dataset, model_name):
                effect_sizes = []
                logger.disabled = True
                for i in range(npermutations):
                    sampled_matches = match_info.groupby('spkr').sample(num_samples, replace=True)
                    sampled_info = speaker_info[speaker_info['sample'].isin(sampled_matches['sample'].tolist())]

                    american = sampled_info[sampled_info['category'] == 'american']
                    foreign = sampled_info[sampled_info['category'] == 'foreign']
                    american_embd = target_embeddings[american['index']]
                    foreign_embd = target_embeddings[foreign['index']]

                    test = Test(american_embd, foreign_embd, high_valence_embd, low_valence_embd,
                                names=names)

                    effect_sizes.append(test.effect_size())
                    pbar.update()

                logger.disabled = False

                stdev = np.std(effect_sizes, ddof=1)
                var = np.var(effect_sizes, ddof=1)
                logger.info(f'For sample size {num_samples * 4}, stdev of effect sizes for resampled '
                            f'embeddings is {stdev}')
                write_eat_se_result('American vs. Korean English Accents',
                                 'Pantos & Perkins',
                                 names, n_target_embd, n_attribute_embd, target_dataset, attribute_dataset,
                                 attribute_dataset, model_name,
                                 npermutations=npermutations, stdev=stdev, var=var)



if __name__ =='__main__':
    to_test = [
        ('morgan_emotional_speech_set', 'mean', 'valence', 'hubert_base_ls960', 13, '1661565382'),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'wavlm_base', 13, '1662262874'),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'wavlm_base_plus', 13, '1662081826'),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'wavlm_large', 25, '1662162141'),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'hubert_xtralarge_ll60k', 49, '1662751606'),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'wav2vec2_large_ll60k', 25, '1663291063'),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'wav2vec2_large_ls960', 25, '1663367964'),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'wav2vec2_base', 13, '1664237818'),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'hubert_large_ll60k', 25, '1664405708'),
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
