import os

import pandas as pd
import numpy as np

from preprocessing.downloading_utils import _convert_wav_to_wav

def read_speaker_list():
    # Read in dysarthric speaker list
    dysarthric_speakers = pd.read_excel('data/UASpeech/doc/speaker_wordlist.xls',sheet_name='Speaker',engine='xlrd',header=3)

    # Initial cleaning
    dysarthric_speakers['Speaker'] = dysarthric_speakers['Speaker'].str.strip()
    dysarthric_speakers['gender'] = dysarthric_speakers['Speaker'].str.slice(0,1)
    dysarthric_speakers['numeric_id'] = dysarthric_speakers['Speaker'].str.slice(start=1)

    # Filter down to only rows that actually contain data
    dysarthric_speakers = dysarthric_speakers[
        dysarthric_speakers['gender'].isin(['M','F'])
    ].reset_index(drop=True)

    # Read which dysarthric speakers we actually have audio data for, and filter table down to these
    available_speakers = os.listdir('data/UASpeech/audio/noisereduce')
    dysarthric_speakers = dysarthric_speakers[
        dysarthric_speakers['Speaker'].isin(available_speakers)
    ].reset_index(drop=True)

    # Create a dataset of all the dysarthric speakers we have access to data for
    non_dysarthric_speakers = pd.DataFrame({'Speaker':[s for s in available_speakers if 'CM' in s or 'CF' in s]})
    non_dysarthric_speakers['gender'] = non_dysarthric_speakers['Speaker'].str.slice(1,2)
    non_dysarthric_speakers['numeric_id'] = non_dysarthric_speakers['Speaker'].str.slice(start=2)

    # Filter down to only dysarthric speakers that have a match
    dysarthric_speakers = dysarthric_speakers[('C' + dysarthric_speakers['Speaker']).isin(non_dysarthric_speakers['Speaker'])
                            ].reset_index(drop=True)

    # Filter down to only non dysarthric speakers that have a match
    non_dysarthric_speakers = non_dysarthric_speakers[non_dysarthric_speakers['Speaker'].str.slice(start=1).isin(dysarthric_speakers['Speaker'])
                            ].reset_index(drop=True)

    # Get matched ages for non dysarthric speakers
    match_df = dysarthric_speakers[['Speaker','Age']].copy()
    match_df['Speaker'] = 'C' + match_df['Speaker']
    non_dysarthric_speakers = non_dysarthric_speakers.merge(match_df, on='Speaker')

    # Label dysarthric and nondysarthric
    dysarthric_speakers['type'] = 'dysarthric'
    non_dysarthric_speakers['type'] = 'non_dysarthric'

    # concatenate together into single df
    all_speakers = pd.concat([dysarthric_speakers, non_dysarthric_speakers]).reset_index(drop=True)

    return all_speakers


def get_base_data_dir():
    # We use the noisereduce audio (as opposed to original or normalized audio) as these achieve the lowest word error
    # rate, according to the readme
    return os.path.join('data','UASpeech','audio','noisereduce')


def sample_words(speakers, words_per_speaker = 5):
    # Read in stated word list
    words = pd.read_excel('data/UASpeech/doc/speaker_wordlist.xls',sheet_name='Word_filename',engine='xlrd')
    words = words.rename(columns=lambda x: x.lower().replace(' ','_'))

    # Select dysarthric speakers, who we'll sample within
    dysarthric_speakers = speakers[speakers['type'] == 'dysarthric']
    sampled_word_list = []

    # set seed
    np.random.seed(17601893)

    # Iterate through dysarthric speakers
    for i, speaker in dysarthric_speakers.iterrows():

        # Find their match
        matched_speaker = speakers[
            (speakers['numeric_id'] == speaker['numeric_id'])
            & (speakers['gender'] == speaker['gender'])
            & (speakers['type'] == 'non_dysarthric')
        ].iloc[0]

        # Sample more words for this than we'll need, as some files may not exist
        sampled_words = words.sample(words_per_speaker + 10, replace=False)

        # Sample a microphone
        sampled_words['microphone'] = np.random.randint(2,9, len(sampled_words))

        # Sample a block, if one has not been specified already
        sampled_words['block'] = np.where(
            sampled_words['file_name'].str.slice(0, 1) == 'B',
            sampled_words['file_name'].str.slice(0, 2),
            'B' + pd.Series(np.random.randint(1, 4, len(sampled_words)).astype(str))
        )

        # Extract word_id
        sampled_words['word_id'] = sampled_words['file_name'].str.replace('B[123]_', '', regex=True)

        # Iterate through words and check that files exist for both speakers
        both_files_exist = []
        for i, w_info in sampled_words.iterrows():
            both_files_exist.append(
                # Check that file exists for dysarthric speaker
                os.path.exists(
                    os.path.join(get_base_data_dir(), speaker['Speaker'],
                        speaker['Speaker'] + '_'
                        + w_info['block'] + '_'
                        + w_info['word_id'] + '_'
                        + 'M' + str(w_info['microphone']) + '.wav'
                    )
                ) and
                # Check that file exists for non dysarthric speaker
                os.path.exists(
                    os.path.join(get_base_data_dir(), matched_speaker['Speaker'],
                        matched_speaker['Speaker'] + '_'
                        + w_info['block'] + '_'
                        + w_info['word_id'] + '_'
                        + 'M' + str(w_info['microphone']) + '.wav'
                    )
                )
            )
        # Filter down to only words where both files exist, (and shrink down to the specified number of words)
        sampled_words = sampled_words[both_files_exist][:words_per_speaker]

        # Join to speaker info
        sampled_words['join_key'] = 1
        both_speakers = pd.DataFrame([speaker, matched_speaker])
        both_speakers['join_key'] = 1
        sampled_words = pd.merge(both_speakers, sampled_words, on = 'join_key')
        sampled_words = sampled_words.drop(columns='join_key')

        # Get file name for each word/speaker combo
        sampled_words['file_name'] = (
                sampled_words['Speaker'] + '_'
                + sampled_words['block'] + '_'
                + sampled_words['word_id'] + '_'
                + 'M' + sampled_words['microphone'].astype(str) + '.wav'
        )

        # Get file path
        sampled_words['file_path'] = sampled_words['Speaker'].str.cat(sampled_words['file_name'],
                                                                      sep=os.sep)


        # Append words for these two speakers to overall list of words
        sampled_word_list.append(sampled_words)

    # Join lists together
    sampled_word_list = pd.concat(sampled_word_list).reset_index(drop=True)

    return sampled_word_list


if __name__ == '__main__':
    speaker_list = read_speaker_list()
    word_list = sample_words(speaker_list)

    # Convert audio clips
    lengths = []
    for fp in word_list['file_path']:
        fp = os.path.join(get_base_data_dir(), fp)
        lengths.append(_convert_wav_to_wav(os.path.basename(fp).replace('.wav',''),
                                           os.path.dirname(fp),
                                           condense_channels=True))
    word_list['sample_length'] = lengths

    # Create clip_info.csv
    word_list.to_csv(os.path.join('data','UASpeech','clip_info.csv'), index=False)


    # Create all.tsv
    tsv_info = [get_base_data_dir()] + (word_list['file_path'].astype(str)
                                        + '\t'
                                        + word_list['sample_length'].astype(int).astype(str)
                                  ).to_list()
    pd.Series(tsv_info).to_csv(os.path.join('data','UASpeech','all.tsv'), index=False, header=False)




