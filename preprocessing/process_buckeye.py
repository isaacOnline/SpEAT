import os

import pandas as pd


def read_transcript(transcript_fp):
    transcript = pd.read_csv(transcript_fp, sep=';', header=None, skiprows=9)
    transcript = transcript[0].str.split('\s+', regex=True)
    transcript = pd.DataFrame({'end_time': transcript.str[1].astype(float), 'content': transcript.str[3]})
    transcript['start_time'] = pd.concat([pd.Series([0]), transcript['end_time'][:-1]]).reset_index(drop=True)
    transcript['interuption'] = transcript['content'].str.contains('IVER') | transcript['content'].str.contains('<EXCLUDE-')
    transcript['segment_id'] = transcript['interuption'].cumsum()

    # Remove interviewer portion and extra noises
    transcript = transcript[~transcript['interuption']]
    transcript = transcript[~transcript['content'].isin(['<VOCNOISE>','<SIL>','<NOISE>'])]

    segments = transcript.groupby('segment_id').agg({'start_time': 'min', 'end_time':'max',
                                                     'content': lambda x: ' '.join(x)})

    segments.columns = ['start_time','end_time','content']
    segments['length'] = segments['end_time'] -  segments['start_time']
    segments = segments.sort_values('length',ascending=False).reset_index(drop=True)
    return segments

def extract_audio():
    metadata = pd.read_csv(os.path.join('data','buckeye','recording_metadata.csv'), sep=' ')
    audio_files = pd.DataFrame({'file_name': [n for n in os.listdir(os.path.join('data', 'buckeye'))
                                              if n[-4:] == '.wav']}).sort_values('file_name').reset_index(drop=True)
    audio_files['speaker'] = audio_files['file_name'].str.slice(0,3).str.upper()
    audio_files = audio_files.merge(metadata, on='speaker')
    save_dir = os.path.join('data','buckeye','processed')
    os.makedirs(save_dir, exist_ok=True)
    all_segments = []
    for _, speaker_data in audio_files.groupby('speaker'):
        this_speakers_segments = []

        for i, file in speaker_data.iterrows():
            transcript_fp = os.path.join('data','buckeye',file['file_name'].replace('.wav','.words'))
            segments = read_transcript(transcript_fp)

            segments['original_file_name'] = file['file_name']
            segments['original_file_path'] = os.path.join('data','buckeye',file['file_name'])
            segments['speaker'] = file['speaker']
            segments['speaker_gender'] = file['speaker_gender']
            segments['speaker_age_strata'] = file['speaker_age']
            segments['interviewer_gender'] = file['interviewer_gender']
            this_speakers_segments.append(segments)

        this_speakers_segments = pd.concat(this_speakers_segments)
        this_speakers_segments = this_speakers_segments.sort_values('length',ascending=False).reset_index(drop=True)


        all_segments.append(this_speakers_segments)
    all_segments = pd.concat(all_segments).reset_index(drop=True)
    all_segments = all_segments[all_segments['length'] > 5]
    return all_segments


if __name__ == '__main__':
    segments = extract_audio()
    segments.to_csv(os.path.join('data','buckeye','full_clip_info.csv'),index=False)

