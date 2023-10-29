import os

import pandas as pd
import numpy as np



def sample_speakers():
    np.random.seed(8121490)
    all_metadata = []
    for region in ['ATL','DCB','LES','PRV','ROC','VLD']:
        region_dir = os.path.join('data', 'CORAAL', region)
        metadata_name = [n for n in os.listdir(region_dir) if 'metadata' in n][0]
        all_metadata.append(pd.read_csv(os.path.join(region_dir,metadata_name), sep='\t'))
    all_metadata = pd.concat(all_metadata)
    all_metadata['age_strata'] = np.where(
        all_metadata['Age'] < 30,
        'Y',
        np.where(all_metadata['Age'] > 40,
                 'O',
                 np.nan)
    )
    all_metadata = all_metadata[
        all_metadata['Gender'].isin(['Female', 'Male'])
        & all_metadata['age_strata'].isin(['Y','O'])
        & all_metadata['Interviewer.Gender'].isin(['Female', 'Male'])
        & (all_metadata['Primary.Spkr'] == 'yes')
    ]

    all_metadata = all_metadata[all_metadata['Age'] >= 18]
    return all_metadata

def process_text_file(text_fp, spk_id):
    transcript = pd.read_csv(text_fp, sep='\t')
    transcript['Spkr'] = np.where(transcript['Content'].str.contains('/RD-') | transcript['Content'].str.contains('[',regex=False),
                                  np.nan,
                                  transcript['Spkr'])
    transcript['new_speaker'] = pd.concat([
        pd.Series(True), transcript['Spkr'][1:].reset_index(drop=True) != transcript['Spkr'][:-1].reset_index(drop=True)
    ]).reset_index(drop=True)
    transcript['segment_id'] = transcript['new_speaker'].cumsum()
    segments = transcript.groupby('segment_id').agg({'Spkr':'first',
                                                     'StTime':'min','EnTime':'max',
                                                     'Content':lambda x: ' '.join(x)})
    segments['length'] = segments['EnTime'] - segments['StTime']
    segments = segments[segments['Spkr'] == spk_id]
    segments = segments.sort_values('length',ascending=False)
    return segments

def extract_audio(file_info, segments_per_speaker=2):

    # Can't match CORAAL with just the princville data, which is closest in time, as there aren't enough speakers in that dataset
    np.random.seed(9358285)
    num_segments = 0
    all_segments = []
    save_dir = os.path.join('data', 'CORAAL', 'processed')
    os.makedirs(save_dir, exist_ok=True)

    for spkr_id, spkr_data in file_info.groupby('CORAAL.Spkr'):
        this_speakers_segments = []
        for _, spk_file in spkr_data.iterrows():

            base_dir = os.path.join('data','CORAAL', spk_file['CORAAL.Sub'])
            all_dirs = [os.path.join(base_dir,n) for n in os.listdir(base_dir)
                        if os.path.isdir(os.path.join(base_dir, n))]

            possible_paths = []
            fname = spk_file['CORAAL.File'] + '.wav'
            for dir in all_dirs:
                if fname in os.listdir(dir):
                    possible_paths.append(os.path.join(dir,fname))
            if len(possible_paths) == 0:
                print(f'Not available: {fname}')
                continue
            elif len(possible_paths) > 1:
                raise ValueError('UNCLEAR FILE NAME')
            wav_fp = possible_paths[0]

            text_dir = [n for n in all_dirs if 'textfiles' in n][0]
            segments = process_text_file(
                text_fp=os.path.join(text_dir, spk_file['CORAAL.File'] + '.txt'),
                spk_id=spk_file['CORAAL.Spkr']
            )
            segments['original_file_name'] = spk_file['CORAAL.File'] + '.wav'
            segments['original_file_path'] = wav_fp
            segments = segments.rename({'Spkr':'speaker'})
            segments['speaker_gender'] = spk_file['Gender']
            segments['speaker_age_strata'] = spk_file['age_strata']
            segments['interviewer_gender'] = spk_file['Interviewer.Gender']
            this_speakers_segments.append(segments)
        if len(this_speakers_segments) == 0:
            continue
        this_speakers_segments = pd.concat(this_speakers_segments)
        this_speakers_segments = this_speakers_segments.sort_values('length',ascending=False).reset_index(drop=True)



        all_segments.append(this_speakers_segments)
    all_segments = pd.concat(all_segments).reset_index(drop=True)
    all_segments = all_segments[all_segments['length'] > 5]
    return all_segments


if __name__ == '__main__':
    file_info = sample_speakers()
    segments = extract_audio(file_info)

    segments.to_csv(os.path.join('data','CORAAL','full_clip_info.csv'), index=False)
