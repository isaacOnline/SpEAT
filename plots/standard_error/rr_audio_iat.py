import os
import logging

import numpy as np
from tqdm import tqdm

from utils import load_dataset_info
from plots.eats.weat.weat.test import Test
from plots.eats.utils import write_eat_se_result, check_if_se_already_run


def perform_tests(model_name, attribute_dataset, dimension, timestamp, npermutations = 10000,nlayers=13):

    names = ['British Accent', 'Foreign Accent','Pleasant Audio Clips',f'Unpleasant Audio Clips ({valence_model_dataset})']

    if attribute_dataset == 'EU_Emotion_Stimulus_Set':
        gs = 3
        n_attribute_embd = 54
    elif attribute_dataset == 'morgan_emotional_speech_set':
        gs = 10
        n_attribute_embd = 60
    else:
        raise ValueError

    attribute_embeddings = np.stack(
        [np.load(os.path.join('embeddings', attribute_dataset, model_name, f'layer_{i}', f'mean.npy')) for i
         in range(nlayers)], axis=2).sum(axis=2)
    valence_info = load_dataset_info(attribute_dataset, group_sizes=gs).reset_index(drop=True)
    high_valence = valence_info[valence_info[f'{dimension}_rank'] == f'high_{dimension}']
    low_valence = valence_info[valence_info[f'{dimension}_rank'] == f'low_{dimension}']
    high_valence_embd = attribute_embeddings[high_valence.index]
    low_valence_embd = attribute_embeddings[low_valence.index]

    target_embeddings = np.stack(
        [np.load(
            os.path.join('embeddings', 'audio_iats','romero_rivas_et_al', model_name, f'layer_{i}', f'mean.npy'))
            for i in
            range(nlayers)], axis=2).sum(axis=2)
    speaker_info = load_dataset_info(os.path.join('audio_iats','romero_rivas_et_al'))
    speaker_info['embd_index'] = range(len(speaker_info))

    seed = 58214641
    np.random.seed(seed)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    model_log_path = os.path.join('plots', 'standard_error', 'rr_audio_iat.log')
    fh = logging.FileHandler(model_log_path)
    formatter = logging.Formatter(
        '%(asctime)s (%(levelname)s): %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info(f'Using model {model_name}, Emotion Dataset {valence_model_dataset}, '
                f'Dimension {dimension}, Timestamp {timestamp}')
    logger.info(f'Using targets: {names[:2]} (in order)')
    logger.info(f'Using attributes: {names[2:]} (in order)')

    logger.info(f'Using npermutations: {npermutations:,}')
    logger.info(f'Using seed: {seed}')

    words = speaker_info[['word', 'stimuli_type']].drop_duplicates()

    with tqdm(total = npermutations * 8) as pbar:
        for num_words_to_sample in range(1, 9):

            target_dataset = 'Romero Rivas et al recordings'
            n_target_embd = num_words_to_sample * 4

            if not check_if_se_already_run('British vs. Spanish English Accents',
                                           'Romero Rivas et al.',
                                            names, n_target_embd, n_attribute_embd, target_dataset, attribute_dataset,
                                            attribute_dataset, model_name):

                logger.disabled = True
                effect_sizes = []

                for _ in range(npermutations):
                    words_to_take = words.groupby('stimuli_type').sample(frac = num_words_to_sample/8, replace=True)
                    sampled_info = words_to_take.merge(speaker_info, on = ['word','stimuli_type'])

                    british = sampled_info[(sampled_info['stimuli_type'] == 'neutral') & (sampled_info['speaker_type'] == 'native')]
                    foreign = sampled_info[(sampled_info['stimuli_type'] == 'neutral') & (sampled_info['speaker_type'] == 'foreign')]
                    british_embd = target_embeddings[british['embd_index']]
                    foreign_embd = target_embeddings[foreign['embd_index']]


                    test = Test(british_embd, foreign_embd, high_valence_embd, low_valence_embd,
                                names=names)

                    effect_sizes.append(test.effect_size())
                    pbar.update()

                logger.disabled = False

                stdev = np.std(effect_sizes, ddof=1)
                var = np.var(effect_sizes, ddof=1)
                logger.info(f'For sample size {num_words_to_sample * 4}, stdev of effect sizes for resampled '
                            f'embeddings is {stdev}')
                write_eat_se_result('British vs. Spanish English Accents',
                                 'Romero Rivas et al.',
                                 names, n_target_embd, n_attribute_embd, target_dataset, attribute_dataset,
                                 attribute_dataset, model_name,
                                 npermutations=npermutations, stdev=stdev, var=var)


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
        ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'whisper_large_decoder', 33, '1666082369'),
        ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'whisper_base_decoder', 7, '1665611782'),
        ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'whisper_base_en_decoder', 7, '1666258986'),
        ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'whisper_base_en_encoder', 7, '1666255677'),
        ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'whisper_base_encoder', 7, '1665605059'),
        ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'whisper_medium_decoder', 25, '1666810158'),
        ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'whisper_medium_en_decoder', 25, '1666396216'),
        ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'whisper_medium_en_encoder', 25, '1666389810'),
        ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'whisper_medium_encoder', 25, '1666803677'),
        ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'whisper_small_decoder', 13, '1666258498'),
        ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'whisper_small_en_decoder', 13, '1666257882'),
        ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'whisper_small_en_encoder', 13, '1666247229'),
        ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'whisper_small_encoder', 13, '1666251448'),
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
