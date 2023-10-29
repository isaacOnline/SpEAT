import os

import numpy as np
import pandas as pd

from preprocessing.downloading_utils import _convert_wav_to_wav

def process_set():
    base_dir = os.path.join('data','morgan_emotional_speech_set')
    metadata = pd.read_excel(os.path.join(base_dir,'Coded stimuli final.xlsx'))

    file_existance = []
    sample_lengths = []
    for code in metadata['code'].values:
        if os.path.exists(os.path.join(base_dir, f'{code}_SCR.wav')):
            file_existance.append(True)
            sample_lengths.append(_convert_wav_to_wav(f'{code}_SCR', base_dir))
        else:
            file_existance.append(False)
            sample_lengths.append(np.nan)

    metadata['file_exists'] = file_existance
    metadata['sample_length'] = sample_lengths

    metadata = metadata[metadata['file_exists']].reset_index(drop=True).copy()
    metadata['file_name'] = (metadata['code'] + '_SCR.wav')

    metadata.to_csv(os.path.join(base_dir, 'clip_info.csv'),index=False)
    tsv_info = [base_dir] + (metadata['file_name'].astype(str)
                             + '\t'
                             + metadata['sample_length'].astype(int).astype(str)
                             ).to_list()
    tsv_fp = os.path.join(base_dir, f'all.tsv')
    pd.Series(tsv_info).to_csv(tsv_fp, index=False, header=False)






if __name__ == '__main__':
    process_set()
