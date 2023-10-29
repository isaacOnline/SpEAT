import os
import re
import shutil

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from psmpy import PsmPy

from preprocessing.downloading_utils import _convert_wav_to_wav

def read_session(session_dir):
    # Find head mic files
    head_mic_dir = os.path.join(session_dir, 'wav_headMic')
    if os.path.exists(head_mic_dir):
        all_head_mic_files = pd.DataFrame({'head_mic_fn': [f for f in os.listdir(head_mic_dir) if not f.startswith('.')]})
        all_head_mic_files['id'] = all_head_mic_files['head_mic_fn'].str.replace('.wav','',regex=False)
        all_head_mic_files['head_mic_fp'] = [os.path.join(head_mic_dir, f) for f in all_head_mic_files['head_mic_fn']]
        all_head_mic_files = all_head_mic_files.drop(columns='head_mic_fn')
    else:
        all_head_mic_files = pd.DataFrame({'id':[]})

    # Find array mic files
    array_mic_dir = os.path.join(session_dir, 'wav_arrayMic')
    if os.path.exists(array_mic_dir):
        all_array_mic_files = pd.DataFrame({'array_mic_fn': [f for f in os.listdir(array_mic_dir) if not f.startswith('.')]})
        all_array_mic_files['id'] = all_array_mic_files['array_mic_fn'].str.replace('.wav','',regex=False)
        all_array_mic_files['array_mic_fp'] = [os.path.join(array_mic_dir, f) for f in all_array_mic_files['array_mic_fn']]
        all_array_mic_files = all_array_mic_files.drop(columns='array_mic_fn')
    else:
        all_array_mic_files = pd.DataFrame({'id':[]})


    # Read prompts
    prompt_dir = os.path.join(session_dir, 'prompts')
    if os.path.exists(prompt_dir):
        all_prompt_paths = [f for f in os.listdir(prompt_dir) if not f.startswith('.')]
        prompt_ids = []
        prompt_texts = []
        num_words = []
        for prompt_path in all_prompt_paths:
            prompt_ids.append(prompt_path.replace('.txt', ''))
            with open(os.path.join(prompt_dir, prompt_path)) as f:
                prompt_text = f.read()[:-1]
            prompt_texts.append(prompt_text)

            # Want to filter out any directions that were not actually said when counting words
            words_read_out_loud = re.sub('\[.+($|\])', '', prompt_text)
            words_read_out_loud = words_read_out_loud.strip()
            num_words.append(len(words_read_out_loud.split(' ')))

        prompts = pd.DataFrame({'id':prompt_ids, 'text':prompt_texts, 'num_words':num_words})
    else:
        prompts = pd.DataFrame({'id':[]})


    full_info = pd.merge(all_array_mic_files, all_head_mic_files, on='id', how='outer')
    full_info = pd.merge(full_info, prompts, on='id', how='outer')

    speaker = os.path.basename(os.path.dirname(session_dir))
    full_info['speaker'] = speaker
    full_info['dysarthric'] = 'control' if 'C' in speaker else 'dysarthric'
    full_info['gender'] = 'female' if 'F' in speaker else 'male'
    full_info['session'] = os.path.basename(session_dir)

    return full_info

def read_speaker(speaker_dir):
    # Find all sessions
    sessions = [f for f in os.listdir(speaker_dir) if 'Session' in f and not f.startswith('.')]
    full_speaker_data = []
    for session in sessions:
        # load data for each session
        session_dir = os.path.join(speaker_dir, session)
        session_data = read_session(session_dir)
        full_speaker_data.append(session_data)

    full_speaker_data = pd.concat(full_speaker_data)
    return full_speaker_data


def load_torgo():
    gender_dirs = ['FC','F','MC','M']
    all_speaker_data = []
    for gender_dir in gender_dirs:
        speakers = os.listdir(os.path.join('data','TORGO',gender_dir))
        for speaker in speakers:
            speaker_dir = os.path.join('data','TORGO', gender_dir, speaker)
            all_speaker_data.append(read_speaker(speaker_dir))

    all_speaker_data = pd.concat(all_speaker_data)

    # Filter out any phrases that don't actually exist in the data
    all_speaker_data = all_speaker_data[~all_speaker_data['head_mic_fp'].isna()]
    return all_speaker_data

def move_file(from_fname, to_dir, from_dir, to_fname=None):
    if to_fname is None:
        to_fname = from_fname
    os.makedirs(to_dir, exist_ok=True)
    local_path = os.path.join(to_dir, to_fname)
    if not os.path.exists(local_path):
        shutil.copy(os.path.join(from_dir, from_fname),
                    local_path)

if __name__ == '__main__':
    torgo_data = load_torgo()
    # Filter to only use phrases
    phrases = torgo_data[torgo_data['num_words'] > 1]

    # Match phrases across conditions
    phrases_said_per_condition = phrases[['dysarthric','text']].drop_duplicates()
    repeated_across_conditions = phrases_said_per_condition['text'].value_counts() > 1
    repeated_across_conditions = repeated_across_conditions[repeated_across_conditions].index
    phrases = phrases[phrases['text'].isin(repeated_across_conditions)]
    phrases['uid'] = phrases['id'] + '_' + phrases['speaker'] + '_' + phrases['session']

    np.random.seed(258358)
    phrases['gender_bool'] = phrases['gender'] == 'male'
    phrases['dysarthric_bool'] = phrases['dysarthric'] == 'dysarthric'
    phrases = pd.concat(
        [phrases.reset_index(drop=True),
         pd.DataFrame(OneHotEncoder().fit_transform(phrases['text'].to_numpy().reshape(-1, 1)).toarray())
         ],axis=1)
    psm = PsmPy(phrases, treatment='dysarthric_bool', indx='uid', exclude=[
        'id', 'array_mic_fp', 'head_mic_fp', 'text', 'num_words', 'speaker', 'dysarthric',
        'gender', 'session', 'id', #'gender_bool',
    ])

    psm.logistic_ps(balance=True)

    psm.knn_matched(matcher='propensity_score', replacement=False, caliper=0.001)

    matches = psm.matched_ids[~psm.matched_ids['matched_ID'].isna()]

    all_matched_phrases = []
    for i, match in matches.iterrows():
    # check that phrases match across conditions
        phrase_1 = phrases[phrases['uid'] == match['uid']]['text'].values[0]
        phrase_2 = phrases[phrases['uid'] == match['matched_ID']]['text'].values[0]

        gender_1 = phrases[phrases['uid'] == match['uid']]['gender'].values[0]
        gender_2 = phrases[phrases['uid'] == match['matched_ID']]['gender'].values[0]
        match['gender'] = gender_1
        match['pair_id'] = i
        if phrase_1 == phrase_2 and gender_1 == gender_2:
            all_matched_phrases.append(match)

    complete_match_set = pd.DataFrame(all_matched_phrases)

    complete_match_set = complete_match_set.groupby('gender').sample(30, replace=False)

    matched_phrases = phrases[phrases['uid'].isin(complete_match_set['matched_ID']) | phrases['uid'].isin(complete_match_set['uid'])]
    matched_phrases = matched_phrases[['uid'] + torgo_data.columns.tolist()].drop(columns='array_mic_fp')

    # Read sample lengths and convert to correct audio format
    local_dir = os.path.join('data','TORGO','sampled')
    samples = []
    for _, sample in matched_phrases.iterrows():
        fp = sample['head_mic_fp']
        move_file(os.path.basename(fp), local_dir, os.path.dirname(fp), sample['uid'] + '.wav')
        sample['sample_length'] = _convert_wav_to_wav(sample['uid'],  local_dir, condense_channels=True)
        sample['pair_id'] = complete_match_set[(complete_match_set['uid'] == sample['uid']) | (complete_match_set['matched_ID'] == sample['uid'])]['pair_id'].iloc[0]
        samples.append(sample)
    matched_phrases = pd.DataFrame(samples)

    matched_phrases.to_csv('data/TORGO/clip_info.csv', index=False)

    tsv_info = [local_dir] + (matched_phrases['uid'].astype(str)
                                  + '.wav\t'
                                  + matched_phrases['sample_length'].astype(int).astype(str)
                                  ).to_list()
    pd.Series(tsv_info).to_csv(os.path.join('data','TORGO','all.tsv'), index=False, header=False)

