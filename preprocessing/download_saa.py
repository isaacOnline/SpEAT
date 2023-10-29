import os
import re
import shutil
import time
import warnings
from urllib import request, parse

import bs4
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import pydub
import requests
from psmpy.plotting import *

from preprocessing.urls import sav_language_list
from preprocessing.downloading_utils import _convert_mp3_to_wav, ScrapingError, _convert_wav_to_wav
from utils import load_dataset_info, get_high_low


def get_language_list():
    """
    Download list of languages from the Speech Accent Archive

    The SAV is organized by native language of the speaker, and this goes through and downloads the full list of
    native languages. Not all of the listed languages will actually have speakers.

    If the language list has already been downloaded, then load from disk.

    :return:
    """
    save_fp = os.path.join(get_saa_dir(), 'language_list.csv')
    if os.path.exists(save_fp):
        return pd.read_csv(save_fp)
    # Load the page
    language_list_page = bs4.BeautifulSoup(request.urlopen(sav_language_list), 'lxml')

    # Find the language list tab
    language_tabs = language_list_page.find_all('ul', {'class': 'languagelist'})[0].children

    language_list = []
    for i in language_tabs:
        # For each language in the list, find its name
        language_name = i.text

        # Find the url for the language (the href only gives the subdir/query, so we have to do some editing to get the
        # full url)
        language_href = i.find_all('a', href=True)[0]['href']
        language_subdir, language_query = language_href.split('?')

        # Encode any characters that need to be url encoded, e.g. space -> %20
        language_query_terms = re.split(r'[=&]', language_query)
        language_query_terms_cleaned = [parse.quote(t) for t in language_query_terms]
        language_query_joiners = re.findall(r'[=&]', language_query) + ['']
        prev_language_query = ''.join([t + j for t, j in zip(language_query_terms, language_query_joiners)])
        if prev_language_query != language_query:
            raise ScrapingError(f'Error with cleaning language query on accent.gmu.edu; Please check code for '
                                f'language: {language_name}')
        language_query = ''.join([t + j for t, j in zip(language_query_terms_cleaned, language_query_joiners)])

        split_base_path = parse.urlsplit(sav_language_list)
        language_url = parse.urlunsplit(split_base_path._replace(path=f'/{language_subdir}', query=language_query))

        # Create a dataframe for the language
        language_list += [pd.DataFrame({'language': [language_name], 'url': [language_url]})]

    # Join all the individual dataframes together
    language_list = pd.concat(language_list).reset_index(drop=True)

    language_list.to_csv(save_fp, index=False)
    return language_list


def get_speaker_list(language_list):
    """
    Get the list of speakers from the archive

    Searches through language pages to compile the list. Some languages, like sa'a, will not have speakers represented
    here, as the archive does not have language pages for them (despite having their recordings)

    Load from disk if list has already been compiled

    :param language_list:
    :return:
    """
    save_fp = os.path.join(get_saa_dir(), 'speakers_from_language_pages.csv')
    if os.path.exists(save_fp):
        return pd.read_csv(save_fp)

    speaker_list = []
    for _, (language_name, language_url) in language_list.iterrows():
        time.sleep(np.random.uniform() * 5)
        language_page = bs4.BeautifulSoup(request.urlopen(language_url), 'lxml')
        language_speakers = language_page.find_all('div', {'class': 'content'})

        if len(language_speakers) < 1:
            continue
        if 'this language is not a native language of any of our speakers' in language_speakers[0].text:
            continue

        language_speakers = language_speakers[0].children
        for speaker in language_speakers:

            if re.match(r'There are \d+ result\(s\) for your search\.', speaker.text):
                continue

            # Find the url for the speaker (the href only gives the subdir/query, so we have to do some editing to get
            # the full url)
            speaker_href = speaker.find_all('a', href=True)[0]['href']
            speaker_subdir, speaker_query = speaker_href.split('?')
            split_base_path = parse.urlsplit(sav_language_list)
            speaker_url = parse.urlunsplit(split_base_path._replace(path=f'/{speaker_subdir}', query=speaker_query))

            speaker_id = re.search(r'.*speakerid\=(\d{1,})', speaker_href)
            if not speaker_id:
                raise ScrapingError(f'Speaker ID not found: {speaker.text}')
            speaker_id = int(speaker_id.group(1))

            speaker_split_text = speaker.text.split(', ')
            speaker_name, speaker_gender = speaker_split_text[:2]
            if len(speaker_split_text) > 2:
                speaker_country = speaker_split_text[-1]
            else:
                speaker_country = np.NaN

            if len(speaker_split_text) > 3:
                speaker_region = ', '.join(speaker_split_text[2:-1])
            else:
                speaker_region = np.NaN

            speaker_full_info = pd.DataFrame({
                'language': [language_name],
                'id': [speaker_id],
                'name': [speaker_name],
                'region': [speaker_region],
                'country': [speaker_country],
                'gender': [speaker_gender],
                'url': [speaker_url]
            })
            speaker_list += [speaker_full_info]

    speaker_list = pd.concat(speaker_list).reset_index(drop=True)
    speaker_list = speaker_list.drop_duplicates(subset=['id', 'region', 'country', 'gender', 'url']).reset_index(
        drop=True)
    if speaker_list['id'].value_counts().max() > 1:
        bad_ids = speaker_list['id'].value_counts()[speaker_list['id'].value_counts() > 1].tolist()
        raise ScrapingError(f'There has been a deduplication error: id(s) {bad_ids} have multiple values of region, '
                            f'country, gender, or url')
    speaker_list.to_csv(save_fp, index=False)
    return speaker_list


def download_speakers(speaker_list):
    """
    Iterate through language pages to find the full list of speakers

    :param speaker_list: Full list containing demographic information, as well as urls, for all speakers
    :return:
    """
    save_fp = os.path.join(get_saa_dir(), 'speaker_list.csv')
    if os.path.exists(save_fp):
        return pd.read_csv(save_fp)
    speaker_infos = []
    for _, (language, id, name, region, country, gender, speaker_url) in speaker_list.iterrows():
        time.sleep(np.random.uniform() * 5)
        speaker_page = bs4.BeautifulSoup(request.urlopen(speaker_url), 'lxml')
        bio_info = speaker_page.find_all('ul', {'class': 'bio'})
        speaker_info = pd.DataFrame(
            {'language': [language], 'id': id, 'name': name, 'region': region, 'country': country,
             'gender': gender, 'url': speaker_url})
        for info in bio_info[0].children:
            if isinstance(info, bs4.element.Tag):
                info_name, info_value = info.text.split(':')
                info_name = info_name.replace(' ', '_')
                speaker_info[info_name] = info_value

        mp3_relative_location = speaker_page.find_all('source', {'type': 'audio/mpeg'})[0]['src']
        split_base_path = parse.urlsplit(sav_language_list)
        speaker_info['mp3_url'] = parse.urlunsplit(split_base_path._replace(path=f'{mp3_relative_location}'))
        speaker_info['has_mp3'] = True
        speaker_infos += [speaker_info]
        download_mp3(id, speaker_info['mp3_url'].iloc[0])

    # There are some ids, e.g. 1675 that are not linked to on any language pages, likely because of a server error on
    # SAA's side - (it appears that the server can't create language pages for languages with apostrophes in them). This
    # goes through and finds/downloads such ids.
    speaker_info = pd.concat(speaker_infos).reset_index(drop=True)
    missing_ids = [i for i in range(1, len(speaker_info) + 1) if str(i) not in speaker_info['id'].astype(str).values]
    for id in missing_ids:
        time.sleep(np.random.uniform() * 5)
        speaker_url = f'http://accent.gmu.edu/browse_language.php?function=detail&speakerid={id}'
        speaker_page = bs4.BeautifulSoup(request.urlopen(speaker_url), 'lxml')
        bio_info = speaker_page.find_all('ul', {'class': 'bio'})
        speaker_info = pd.DataFrame(
            {'language': [np.nan], 'id': id, 'name': np.nan, 'region': np.nan, 'country': np.nan,
             'gender': np.nan, 'url': speaker_url})
        for info in bio_info[0].children:
            if isinstance(info, bs4.element.Tag):
                info_name, info_value = info.text.split(':')
                info_name = info_name.replace(' ', '_')
                speaker_info[info_name] = info_value

        mp3_relative_location = speaker_page.find_all('source', {'type': 'audio/mpeg'})[0]['src']

        # If an MP3 is not found for the speaker, we don't want to try and download it. This catches that condition.
        if mp3_relative_location != '/soundtracks/':
            split_base_path = parse.urlsplit(sav_language_list)
            speaker_info['mp3_url'] = parse.urlunsplit(split_base_path._replace(path=f'{mp3_relative_location}'))
            download_mp3(id, speaker_info['mp3_url'].iloc[0])
            speaker_info['has_mp3'] = True
        else:
            speaker_info['mp3_url'] = np.NaN
            speaker_info['has_mp3'] = False
        speaker_infos += [speaker_info]

    speaker_info = pd.concat(speaker_infos).reset_index(drop=True)
    speaker_info = clean_speaker_info(speaker_info)
    speaker_info.to_csv(save_fp, index=False)
    return speaker_info


def download_mp3(speaker_id, speaker_url):
    """
    Download mp3 from a url

    Saves to data/speech_accent_archive

    :param speaker_id: ID number for speaker. Will be used as filepath
    :param speaker_url: Url to download the speaker's recording from
    :return:
    """
    if not os.path.exists('data'):
        os.mkdir('data')
    saa_dir = get_saa_dir()
    if not os.path.exists(saa_dir):
        os.mkdir(saa_dir)
    fp = os.path.join(saa_dir, f'{speaker_id}.mp3')
    if not os.path.exists(fp):
        r = requests.get(speaker_url, stream=True)
        r.raw.decode_content = True
        with open(fp, 'wb') as f:
            shutil.copyfileobj(r.raw, f)


def clean_speaker_info(speaker_info):
    """
    Clean speaker information

    Change to correct datatypes, remove unneeded whitespace,  sort by id, etc.

    :param speaker_info: Speaker information to be cleaned
    :return:
    """
    # sort
    speaker_info = speaker_info.sort_values('id').reset_index(drop=True)

    # rename columns
    speaker_info = speaker_info.rename(columns={'other_language(s)': 'other_languages', 'age,_sex': 'age_and_sex'})

    # Remove text from html that shouldn't be here
    speaker_info['english_learning_method'] = speaker_info['english_learning_method'].str.replace(r'(\r\n|\r|\n)', ' ',
                                                                                                  regex=True)
    speaker_info['region'] = speaker_info['region'].str.replace(r'(\r\n|\r|\n)', ' ', regex=True)
    speaker_info['birth_place'] = speaker_info['birth_place'].str.replace(' (map)', '', regex=False)
    speaker_info['native_language'] = speaker_info['native_language'].str.replace(r'\n\([a-z]+\)', '', regex=True)
    speaker_info['length_of_english_residence'] = speaker_info['length_of_english_residence'].str.replace(' years', '',
                                                                                                          regex=False)

    # Split age and sex columns
    speaker_info['age'] = speaker_info['age_and_sex'].str.split(', ', expand=True)[0]
    speaker_info['sex'] = speaker_info['age_and_sex'].str.split(', ', expand=True)[1]
    speaker_info = speaker_info.drop(columns='age_and_sex')

    # Remove unneeded white space
    for col in speaker_info.columns:
        if speaker_info[col].dtype == object:
            speaker_info[col] = speaker_info[col].str.strip()

    # Convert numerics
    speaker_info = speaker_info.replace(r'^\s*$', np.NaN, regex=True)
    speaker_info['id'] = speaker_info['id'].astype(int)
    for col in ['age', 'age_of_english_onset', 'length_of_english_residence']:
        speaker_info[col] = speaker_info[col].astype(float)

    # Drop duplicate columns,
    speaker_info = speaker_info.drop(columns=['gender', 'region', 'country'])

    # Calculate how long the speaker has spoken english
    speaker_info['length_of_time_speaking_english'] = speaker_info['age'] - speaker_info['age_of_english_onset']

    return speaker_info


def convert_to_wav(speaker_info):
    """
    Convert audio files for a full list of speakers to mp3

    If any audio files are corrupted, this will delete them

    :param speaker_info: DataFrame with demographic and id info on the speakers to convert. Should have an 'id' column.
    :return: Updated speaker list with records of deleted mp3s
    """
    ids = speaker_info['id'][speaker_info['has_mp3']]
    for speaker_id in ids:
        try:
            speaker_info.at[
                speaker_info.index[speaker_info['id'] == speaker_id][0], 'sample_length'] = _convert_mp3_to_wav(
                speaker_id, os.path.join('data', 'speech_accent_archive'))
        except pydub.exceptions.CouldntDecodeError:
            warnings.warn(f'Speaker ID {speaker_id} is corrupted. Audio file has been deleted')
            speaker_info.at[speaker_info.index[speaker_info['id'] == speaker_id][0], 'has_mp3'] = False
            saa_dir = get_saa_dir()
            mp3_fp = os.path.join(saa_dir, f'{speaker_id}.mp3')
            os.remove(mp3_fp)
    save_fp = os.path.join(get_saa_dir(), 'speaker_list.csv')
    speaker_info.to_csv(save_fp, index=False)
    return speaker_info


def create_speaker_tsv(speaker_info, fp):
    """
    Create TSV file for use when extracting embeddings

    :param speaker_info: Dataframe of information on speakers
    :param fp: File path to save to
    :return:
    """
    speaker_info = speaker_info[speaker_info['has_mp3'] & (speaker_info['language'] != 'synthesized')]
    tsv_info = [os.path.dirname(fp)] + (speaker_info['id'].astype(str)
                                  + '.wav\t'
                                  + speaker_info['sample_length'].astype(int).astype(str)
                                  ).to_list()
    pd.Series(tsv_info).to_csv(fp, index=False, header=False)


def get_saa_dir():
    """
    Get the directory for all Speech Accent Archive Data

    :return:
    """
    return os.path.join('data', 'speech_accent_archive')


def move_file(fname, local_dir, backup_dir):
    local_path = os.path.join(local_dir, fname)
    if not os.path.exists(local_path):
        shutil.copy(os.path.join(backup_dir, fname), local_path)

def split_subsets():
    saa_speaker_info = load_dataset_info('speech_accent_archive', group_sizes=10)

    np.random.seed(64899955)
    native_british_english_speakers = saa_speaker_info[
        (saa_speaker_info['birth_place'].str.contains(', uk')
         | saa_speaker_info['birth_place'].str.contains(', ireland')
         | saa_speaker_info['birth_place'].str.contains('ireland,'))
    & (saa_speaker_info['native_language'] == 'english')
    ]
    native_british_english_speakers = get_high_low(native_british_english_speakers, 'age', method='top_n', n=25)
    min_sex_size = native_british_english_speakers.groupby(['age_rank','sex']).count()['age'].min()
    native_british_english_speakers = native_british_english_speakers.groupby(['age_rank','sex']).sample(min_sex_size)
    native_british_english_speakers = native_british_english_speakers.sort_values('id')

    british_dir = os.path.join(get_saa_dir(), 'british_young_old')
    os.makedirs(british_dir, exist_ok=True)
    native_british_english_speakers['index'] = range(len(native_british_english_speakers))
    native_british_english_speakers.to_csv(os.path.join(british_dir, 'clip_info.csv'), index=False)
    create_speaker_tsv(native_british_english_speakers, fp = os.path.join(british_dir, f'all.tsv'))


    # Split off data for Young/old test
    native_us_english_speakers = saa_speaker_info[
        saa_speaker_info['birth_place'].str.contains(', usa')
        & (saa_speaker_info['native_language'] == 'english')
    ]
    native_us_english_speakers = get_high_low(native_us_english_speakers, 'age', method='top_n', n=60)
    min_sex_size = native_us_english_speakers.groupby(['age_rank','sex']).count()['age'].min()
    native_us_english_speakers = native_us_english_speakers.groupby(['age_rank','sex']).sample(min_sex_size)
    native_us_english_speakers = native_us_english_speakers.sort_values('id')

    usa_dir = os.path.join(get_saa_dir(), 'usa_young_old')
    os.makedirs(usa_dir, exist_ok=True)
    native_us_english_speakers['index'] = range(len(native_us_english_speakers))
    native_us_english_speakers.to_csv(os.path.join(usa_dir, 'clip_info.csv'), index=False)
    create_speaker_tsv(native_us_english_speakers, fp = os.path.join(usa_dir, f'all.tsv'))

    # Split off data for US/Korean test
    native_us_english_speakers = saa_speaker_info[
        saa_speaker_info['birth_place'].str.contains(', usa')
        & (saa_speaker_info['native_language'] == 'english')
    ]
    korean_speakers = saa_speaker_info[
        saa_speaker_info['birth_place'].str.contains('korea')
        & (saa_speaker_info['native_language'] == 'korean')
    ]
    korean_and_us = pd.concat([native_us_english_speakers, korean_speakers])
    korean_and_us['native_language_bool'] = korean_and_us['native_language'] == 'korean'
    korean_and_us['sex_bool'] = korean_and_us['sex'] == 'male'
    psm = PsmPy(korean_and_us, treatment='native_language_bool', indx='id', exclude=[
        'language', 'name', 'url', 'birth_place', 'native_language',
       'other_languages', 'age_of_english_onset', 'english_learning_method',
       'english_residence', 'length_of_english_residence', 'mp3_url',
       'has_mp3', 'length_of_time_speaking_english', 'sample_length','sex',
       'age_rank', #'age', 'sex_bool'
    ]
    )

    np.random.seed(63182)
    psm.logistic_ps(balance=True)

    psm.knn_matched(matcher='propensity_score', replacement=False, caliper=0.001)

    matches = psm.matched_ids[~psm.matched_ids['matched_ID'].isna()]
    matches['pair_id'] = range(len(matches))

    korean_and_us = pd.concat([
        korean_and_us.merge(matches.drop(columns='matched_ID'), on='id'),
        korean_and_us.merge(matches.drop(columns='id').rename(columns={'matched_ID':'id'}), on='id')
    ])

    us_korean_dir = os.path.join(get_saa_dir(), 'us_korean')
    os.makedirs(us_korean_dir, exist_ok=True)
    korean_and_us['index'] = range(len(korean_and_us))
    korean_and_us.to_csv(os.path.join(us_korean_dir, 'clip_info.csv'), index=False)
    create_speaker_tsv(korean_and_us, fp = os.path.join(us_korean_dir, f'all.tsv'))

    for fname in korean_and_us['id'].values:
        move_file(f'{fname}.wav', 'data/speech_accent_archive/us_korean',
                                '/Volumes/Backup Plus/research/SpEAT/data/speech_accent_archive')


    # Split off data for human/synthetic test
    # Create dataframe of synthetic speaker info
    synthetic_dir= 'data/human_synthesized'

    synthesized_speaker_info = {}
    synthesized_speaker_info['path'] = [p for p in os.listdir(synthetic_dir) if p.endswith('.wav') and (re.match('\d+', p.replace('.wav','')) is None)]
    synthesized_speaker_info['sex'] = [p.split('_')[0] for p in synthesized_speaker_info['path']]
    speeds = []
    voices = []
    for p in synthesized_speaker_info['path']:
        if 'slow' in p:
            speeds += ['slow']
        elif 'fast' in p:
            speeds += ['fast']
        else:
            speeds += ['normal']
        voices += ['_'.join(p.split('_')[1:]).replace('_slow','').replace('_fast','').replace('.wav','')]

    synthesized_speaker_info['speed'] = speeds
    synthesized_speaker_info['voice'] = voices
    synthesized_speaker_info = pd.DataFrame(synthesized_speaker_info)
    sample_lengths = []
    for p in synthesized_speaker_info['path']:
        sample_lengths.append(_convert_wav_to_wav(p.replace('.wav',''), synthetic_dir, condense_channels=True))
    synthesized_speaker_info['sample_length'] = sample_lengths

    # join dfs
    synthesized_speaker_info['human'] = False
    synthesized_speaker_info = synthesized_speaker_info.rename(columns={'path':'id'})

    human_speaker_info = saa_speaker_info[
        saa_speaker_info['birth_place'].str.contains(', usa')
        & (saa_speaker_info['native_language'] == 'english')
    ].copy()
    human_speaker_info['human'] = True

    np.random.seed(69401751)
    human_samples = []
    for i, genders_info in pd.DataFrame(synthesized_speaker_info['sex'].value_counts()).reset_index().iterrows():
        human_samples += [human_speaker_info[human_speaker_info['sex'] == genders_info['index']].sample(genders_info['sex'], replace=False)]
    human_samples = pd.concat(human_samples)

    for fname in human_samples['id'].values:
        move_file(f'{fname}.wav', synthetic_dir,
                                '/Volumes/Backup Plus/research/SpEAT/data/speech_accent_archive')

    human_and_synthetic = pd.concat([human_samples, synthesized_speaker_info])

    human_and_synthetic['index'] = range(len(human_and_synthetic))
    human_and_synthetic.to_csv(os.path.join(synthetic_dir, 'clip_info.csv'), index=False)
    human_and_synthetic['fname'] = np.where(human_and_synthetic['human'], human_and_synthetic['id'].astype(str) + '.wav', human_and_synthetic['id'].astype(str))
    tsv_info = [synthetic_dir] + (
            human_and_synthetic['fname'].astype(str) + '\t'
                                  + human_and_synthetic['sample_length'].astype(int).astype(str)
                                  ).to_list()
    pd.Series(tsv_info).to_csv(os.path.join(synthetic_dir, f'all.tsv'), index=False, header=False)

    # Split off data for Male/Female Test
    us_english = saa_speaker_info[
        saa_speaker_info['birth_place'].str.contains(', usa')
        & (saa_speaker_info['native_language'] == 'english')
    ].copy()
    us_english['sex_bool'] = us_english['sex'] == 'male'
    psm = PsmPy(us_english, treatment='sex_bool', indx='id',
                exclude=['name', 'url', 'birth_place', 'language', 'native_language',
       'other_languages', 'age_of_english_onset', 'english_learning_method',
       'english_residence', 'length_of_english_residence', 'mp3_url', 'length_of_time_speaking_english',
       'sex', 'has_mp3', 'sample_length',
       'age_rank', # age,
    ])

    np.random.seed(63182)
    psm.logistic_ps(balance=True)

    psm.knn_matched(matcher='propensity_score', replacement=False, caliper=0.001)
    matches = psm.matched_ids[~psm.matched_ids['matched_ID'].isna()]

    # make sure they have same languages
    id_pairs_to_keep = []
    for m in matches.iterrows():
        lang_1 = saa_speaker_info.loc[saa_speaker_info['id'] == m[1]['id'], 'language'].values[0]
        lang_2 = saa_speaker_info.loc[saa_speaker_info['id'] == m[1]['matched_ID'], 'language'].values[0]
        age_1 = saa_speaker_info.loc[saa_speaker_info['id'] == m[1]['id'], 'age'].values[0]
        age_2 = saa_speaker_info.loc[saa_speaker_info['id'] == m[1]['matched_ID'], 'age'].values[0]
        if age_1 == age_2 and lang_1 == lang_2:
            id_pairs_to_keep.append((m[1]['id'], m[1]['matched_ID']))

    # sample to only 60 matches
    np.random.seed(82641920)
    id_pairs_to_keep = [id_pairs_to_keep[i] for i in np.random.choice(range(len(id_pairs_to_keep)),  60, replace=False)]
    ids_to_keep = [item for sublist in id_pairs_to_keep for item in sublist]
    matches = pd.DataFrame(id_pairs_to_keep,columns=['id','matched_ID'])
    matches['pair_id'] = range(len(matches))

    gender_matches = saa_speaker_info[saa_speaker_info['id'].isin(ids_to_keep)].copy()

    gender_matches = pd.concat([
        gender_matches.merge(matches.drop(columns='matched_ID'), on='id'),
        gender_matches.merge(matches.drop(columns='id').rename(columns={'matched_ID':'id'}), on='id')
    ])


    gender_dir = os.path.join(get_saa_dir(), 'male_female')
    os.makedirs(gender_dir, exist_ok=True)
    gender_matches['index'] = range(len(gender_matches))
    gender_matches.to_csv(os.path.join(gender_dir, 'clip_info.csv'), index=False)
    create_speaker_tsv(gender_matches, fp = os.path.join(gender_dir, f'all.tsv'))

    for fname in gender_matches['id'].values:
        move_file(f'{fname}.wav', 'data/speech_accent_archive/male_female',
                                '/Volumes/Backup Plus/research/SpEAT/data/speech_accent_archive')






if __name__ == '__main__':
    language_list = get_language_list()
    speaker_list = get_speaker_list(language_list)
    full_speaker_info = download_speakers(speaker_list)
    full_speaker_info = convert_to_wav(full_speaker_info)

    create_speaker_tsv(full_speaker_info, fp = os.path.join(get_saa_dir(), f'all.tsv'))

    split_subsets()