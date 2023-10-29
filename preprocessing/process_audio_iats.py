import os
import re
import shutil

import pandas as pd
import numpy as np

from preprocessing.downloading_utils import _convert_wav_to_wav, _convert_mp3_to_wav

def process_pantos_perkins():
    base_dir = os.path.join('data', 'audio_iats', 'pantos_perkins')
    stimuli_names = pd.read_csv(os.path.join(base_dir, 'audio_stimuli_used.csv'), header=None)


    american_wav_files = [f for f in os.listdir(base_dir) if f[-4:] == '.wav' and f[:5] == 'AM_Dr']
    american_wav_files_used_in_study = []
    for f in american_wav_files:
        for s in stimuli_names[0].values:
            if re.search(f'(?:AM|KOR)_Dr[12]_{s}[xX]3n.wav', f) is not None:
                american_wav_files_used_in_study.append(f)
    assert len(american_wav_files_used_in_study) == 8

    korean_wav_files = [f for f in os.listdir(base_dir) if f[-4:] == '.wav' and f[:6] == 'KOR_Dr']
    korean_wav_files_used_in_study = []
    for f in korean_wav_files:
        for s in stimuli_names[0].values:
            if re.search(f'(?:AM|KOR)_Dr[12]_{s}[xX]3n.wav', f) is not None:
                korean_wav_files_used_in_study.append(f)
    assert len(korean_wav_files_used_in_study) == 8

    sample_lengths = []
    for stimuli in american_wav_files_used_in_study + korean_wav_files_used_in_study:
        sample_lengths.append(_convert_wav_to_wav(stimuli.replace('.wav',''), base_dir))

    clip_info = pd.DataFrame({
        'category': ['american'] * 8 + ['foreign'] * 8,
        'clip_name': american_wav_files_used_in_study + korean_wav_files_used_in_study,
        'sample_length': sample_lengths
    })
    clip_info.to_csv(os.path.join(base_dir, 'clip_info.csv'), index=False)


    tsv_info = [base_dir] + (clip_info['clip_name'].astype(str)
                             + '\t'
                             + clip_info['sample_length'].astype(int).astype(str)
                             ).to_list()
    tsv_fp = os.path.join(base_dir, f'all.tsv')
    pd.Series(tsv_info).to_csv(tsv_fp, index=False, header=False)


def process_mitchell_et_al():
    base_dir = os.path.join('data', 'audio_iats', 'mitchell_et_al')

    all_audio_files = pd.DataFrame({'file_name':[f for f in os.listdir(os.path.join(base_dir, 'test'))]})
    all_audio_files['human_or_synthesized'] = np.where(
        all_audio_files['file_name'].str[0] == 'h',
        'human',
        np.where(all_audio_files['file_name'].str[0] == 's','synthesized', np.nan)
    )
    all_audio_files['male_or_female'] = np.where(
        all_audio_files['file_name'].str[1] == 'm',
        'male',
        np.where(all_audio_files['file_name'].str[1] == 'f','female', np.nan)
    )

    sample_lengths = []
    for f in all_audio_files['file_name']:
        if f[-4:] == '.wav':
            sample_lengths.append(_convert_wav_to_wav(f[:-4], os.path.join(base_dir, 'test')))
        elif f[-4:] == '.mp3':
            sample_lengths.append(_convert_mp3_to_wav( f[:-4], os.path.join(base_dir, 'test')))
        else:
            raise ValueError
    all_audio_files['file_name'] = all_audio_files['file_name'].str.replace('.mp3','.wav',regex=False)
    all_audio_files['sample_length'] = sample_lengths
    all_audio_files.to_csv(os.path.join(base_dir, 'audio_info.csv'), index=False)

    tsv_info = [os.path.join(base_dir,'test')] + (all_audio_files['file_name'].astype(str)
                             + '\t'
                             + all_audio_files['sample_length'].astype(int).astype(str)
                             ).to_list()
    tsv_fp = os.path.join(base_dir, f'all.tsv')
    pd.Series(tsv_info).to_csv(tsv_fp, index=False, header=False)



def process_romero_rivas_et_al():
    base_dir = os.path.join('data','audio_iats','romero_rivas_et_al')
    stimuli_list_path = os.path.join(base_dir, 'IAT_Stimuli', 'stimuli_list.txt')
    stimuli = {}
    with open(stimuli_list_path, 'r') as f:
        for l in f.readlines():
            if l != '\n':
                stimuli_type, stimuli_list = l.split(': ')
                stimuli_list = stimuli_list.split(', ')
                stimuli_list = [s.replace('\n','') for s in stimuli_list]
                stimuli[stimuli_type] = stimuli_list

    clip_info = []
    processed_clip_dir = os.path.join(base_dir, 'all_clips')
    os.makedirs(processed_clip_dir, exist_ok=True)
    for background in ['noise','quiet']:
        clip_parent = 'IAT_Stimuli - noise' if background == 'noise' else 'IAT_Stimuli'
        clip_suffix = '_Noise' if background == 'noise' else ''
        clip_original_dir = os.path.join(base_dir, clip_parent)
        for stimuli_type, stimuli_list in stimuli.items():
            for s in stimuli_list:
                for speaker in ['foreign', 'native']:
                    clip_name = f'{s}_{speaker}{clip_suffix}'
                    clip_original_path = os.path.join(clip_original_dir, clip_name + '.wav')
                    clip_new_path = os.path.join(processed_clip_dir, clip_name + '.wav')
                    shutil.copy2(clip_original_path, clip_new_path)

                    sample_length = _convert_wav_to_wav(clip_name, processed_clip_dir, condense_channels=True)

                    clip_info.append(pd.DataFrame({
                        'speaker_type': [speaker],
                        'stimuli_type': stimuli_type,
                        'word':s,
                        'background_noise': background,
                        'original_path': clip_original_path,
                        'new_path': clip_new_path,
                        'clip_name': clip_name,
                        'sample_length': sample_length
                    }))

    clip_info = pd.concat(clip_info).reset_index(drop=True)
    clip_info.to_csv(os.path.join(base_dir, 'clip_info.csv'), index=False)

    tsv_info = [os.path.join(base_dir,'all_clips')] + (clip_info['clip_name'].astype(str)
                             + '.wav\t'
                             + clip_info['sample_length'].astype(int).astype(str)
                             ).to_list()
    tsv_fp = os.path.join(base_dir, f'all.tsv')
    pd.Series(tsv_info).to_csv(tsv_fp, index=False, header=False)





if __name__ == '__main__':
    process_pantos_perkins()
    process_mitchell_et_al()
    process_romero_rivas_et_al()
