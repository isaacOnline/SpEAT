import os

import numpy as np
import pandas as pd

from preprocessing.downloading_utils import _convert_mp3_to_wav


def create_stimulus_info_frame():
    """
    Create a dataframe containing information on each file tested in the dataset. Also convert all files to .wav

    :return:
    """
    # Open XLSX with voice info
    file_info_name = 'uk_voices_survey_results.xlsx'
    file_info_path = os.path.join('data', 'EU_Emotion_Stimulus_Set', 'Supplementary Information', file_info_name)
    voices = pd.read_excel(file_info_path)

    # Clean stimulus_id to match file paths
    voices['original_id'] = voices['stimulus_id'].copy()

    # Incorporate speaker info
    voices['speaker_id'] = voices['original_id'].str[0]
    speaker_info = pd.read_csv(os.path.join('data','EU_Emotion_Stimulus_Set', 'speaker_info.csv')).drop(
        columns=['MeanCCR','Script']).rename(
        columns={'ActorCode':'speaker_id','Age':'speaker_age','Gender':'speaker_gender'})
    voices = voices.merge(speaker_info,on='speaker_id')

    # Some of the stimulus IDs in the supplementary information for the dataset contain an 'x' at the end, which I am
    # assuming means that they used the 'fixed_audio', which is volume adjusted
    voices['fixed_audio'] = voices['stimulus_id'].str[-1] == 'x'
    voices['stimulus_id'] = (
            pd.Series(np.where(voices['fixed_audio'], 'fix__', ''))
            + voices['stimulus_id'].str.replace('-', '').str.replace('(.{4})1', '\\1 1', regex=True)).str.replace(
        '(.*)x', '\\1', regex=True)
    voices['stimulus_dir'] = (
            'data' + os.sep
            + 'EU_Emotion_Stimulus_Set' + os.sep
            + pd.Series(np.where(voices['fixed_audio'], 'Fixed - amplified volume', 'Original')) + os.sep
            + voices['expressed_emotion']
            + np.where(voices['emotion_intensity'] == 'Low', ' - Low Intensity', '')

    )

    # convert to .wav
    id_corrected = []
    sample_lengths = []
    for i, (fdir, stimulus_id) in voices[['stimulus_dir', 'stimulus_id']].iterrows():
        # Some of the IDs in the supplementary info do not correspond to files, so this first checks whether this is the
        # case
        mp3_fp = os.path.join(fdir, stimulus_id + '.mp3')
        wav_fp = os.path.join(fdir, stimulus_id + '.wav')
        if os.path.exists(mp3_fp) or os.path.exists(wav_fp):
            sample_lengths += [_convert_mp3_to_wav(stimulus_id, fdir)]
            id_corrected += [False]
        else:
            # For the IDs that don't have files, some do have an alternate file, e.g. even though 'HV8C 1' doesn't
            # exist, 'HV8C' does. So here I am just using the alternate version
            if stimulus_id[-2:] == ' 1':
                alternate_id = stimulus_id[:-2]
            else:
                alternate_id = stimulus_id + ' 1'

            mp3_fp = os.path.join(fdir, alternate_id + '.mp3')
            wav_fp = os.path.join(fdir, alternate_id + '.wav')
            if os.path.exists(mp3_fp) or os.path.exists(wav_fp):
                id_corrected += [True]
                voices.at[i, 'stimulus_id'] = alternate_id
                sample_lengths += [_convert_mp3_to_wav(alternate_id, fdir)]
            else:
                # Some IDs don't even have an alternate version, which I record here
                id_corrected += [np.nan]
                sample_lengths += [np.nan]
                print('Could not find ' + os.path.join(fdir, stimulus_id + '.mp3'))

    voices['id_corrected'] = id_corrected
    voices['file_exists'] = ~voices['id_corrected'].isna()
    voices['sample_length'] = sample_lengths
    voices['path'] = np.where(voices['file_exists'],
                              voices['stimulus_dir'] + os.sep + voices['stimulus_id'] + '.wav',
                              np.nan)

    # Save info
    voices_save_path = os.path.join('data', 'EU_Emotion_Stimulus_Set', 'voice_file_info.csv')
    voices.to_csv(voices_save_path, index=False)

    # Save tsv for embedding extraction
    to_embed = voices[voices['file_exists']]
    tsv_info = [os.path.join('data', 'EU_Emotion_Stimulus_Set')] + (
            to_embed['path'].str.replace(os.path.join('data', 'EU_Emotion_Stimulus_Set') + r'[\\/]{0,2}', '',
                                         regex=True)
            + '\t'
            + to_embed['sample_length'].astype(int).astype(str)
    ).to_list()
    tsv_path = os.path.join(os.path.join('data', 'EU_Emotion_Stimulus_Set', 'all.tsv'))
    pd.Series(tsv_info).to_csv(tsv_path, index=False, header=False)


if __name__ == '__main__':
    create_stimulus_info_frame()
