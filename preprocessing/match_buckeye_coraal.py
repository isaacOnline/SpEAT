import os
import logging
import csv
import urllib.request

import pydub
import pandas as pd
import numpy as np
# from psmpy.plotting import *
import scipy.stats
from scipy.stats import shapiro, kstest
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer as VaderSentimentIntensityAnalyzer
from transformers import pipeline

from preprocessing.downloading_utils import _convert_wav_to_wav
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from scipy.special import softmax



def get_full_matches():
    coraal = pd.read_csv(os.path.join('data','CORAAL','full_clip_info.csv'))
    coraal['dataset'] = 0
    coraal = coraal.rename(columns = {'Spkr':'speaker','StTime':'start_time','EnTime':'end_time','Content':'content'})
    buckeye = pd.read_csv(os.path.join('data','buckeye','full_clip_info.csv'))
    buckeye['dataset'] = 1
    joined = pd.concat([coraal,buckeye])
    joined = joined.reset_index()
    joined['index'] = joined['dataset'].astype(str) + '_' + joined['index'].astype(str)
    joined = joined.rename(columns={'index':'segment_id'})

    for var in ['speaker_gender', 'interviewer_gender']:
        joined[var] = (joined[var] == 'Female').astype(int)
    joined['speaker_age_strata'] = (joined['speaker_age_strata'] == 'Y').astype(int)

    np.random.seed(1710010)
    psm = PsmPy(joined, treatment='dataset', indx='segment_id', exclude=[
        'speaker', 'start_time', 'end_time', 'content',
        'original_file_name', 'original_file_path',
        # 'speaker_gender', 'speaker_age_strata', 'interviewer_gender',
    ])

    psm.logistic_ps(balance=True)

    psm.knn_matched(matcher='propensity_score', replacement=False, caliper=0.001)

    matches = psm.matched_ids[~psm.matched_ids['matched_ID'].isna()]
    return matches, joined

def filter_matches(matches, joined):

    # Due to the small sample size required for the test, we sample from the set of valid matches. We found that
    # taking samples from the raw matches did not lead to equal distributions in length between the AA/EA
    # speakers, so we elected to require that the segments be within 5 seconds in length of each other, be from
    # speaker of the same gender, and speakers of the same age strata. We finally take stratafied samples within these
    # age/gender, and perform a KS test to check that the clips are of similar length.
    np.random.seed(5616755)
    valid_matches = []
    all_tdiff = []
    tdiff = []
    for i, match in matches.iterrows():
        first = joined[joined['segment_id'] == match['segment_id']]
        second = joined[joined['segment_id'] == match['matched_ID']]
        all_tdiff.append(np.abs(first['length'].values[0] - second['length'].values[0]))
        if (not (match['segment_id'] is np.nan or match['matched_ID'] is np.nan)
            and (first['speaker_gender'].values[0] == second['speaker_gender'].values[0])
            and (first['speaker_age_strata'].values[0] == second['speaker_age_strata'].values[0])
            # and ((first['length'].values[0] - second['length'].values[0]) < 8)
            # and ((second['length'].values[0] - first['length'].values[0]) < 8)
        ):
            match = match.copy()
            match = pd.concat([match,
                              pd.Series([first['speaker_gender'].values[0], first['speaker_age_strata'].values[0],
                                         first['length'].values[0], second['length'].values[0]],
                                        index=['speaker_gender','speaker_age_strata','first_length','second_length'])])
            valid_matches.append(match)

    valid_matches = pd.DataFrame(np.stack(valid_matches),
                                 columns=['first_id','second_id', 'speaker_gender', 'speaker_age_strata',
                                          'first_length','second_length'])
    sampled_matches = valid_matches.groupby(['speaker_gender',
                                             'speaker_age_strata']).sample(15)

    full_match_info = joined[joined['segment_id'].isin(sampled_matches['first_id'])
                             | joined['segment_id'].isin(sampled_matches['second_id'])]

    save_dir = os.path.join('data', 'coraal_buckeye_joined')
    os.makedirs(save_dir, exist_ok=True)
    full_match_info.to_csv(os.path.join(save_dir,'sampled_matches.csv'),index=False)
    joined.to_csv(os.path.join(save_dir, 'all_audio_clips.csv'), index=False)

    full_match_info = full_match_info.reset_index(drop=True)
    return sampled_matches, full_match_info


def extract_matches(sampled_matches, full_matches):
    np.random.seed(401678)
    matches = full_matches.sample(frac=1, replace=False).reset_index(drop=True)


    fnames = []
    lengths = []
    save_dir = os.path.join('data', 'coraal_buckeye_joined')
    for i, segment_info in matches.iterrows():
        wav = pydub.AudioSegment.from_wav(segment_info['original_file_path'])
        segment = wav[int(segment_info['start_time'] * 1000):int(segment_info['end_time'] * 1000)]
        fnames.append(str(i) + '.wav')
        segment.export(os.path.join(save_dir, fnames[-1]), format='wav')
        lengths.append(_convert_wav_to_wav(fnames[-1].replace('.wav',''),save_dir,condense_channels=False))
    matches['file_name'] = fnames
    matches['sample_length'] = lengths
    matches['speaker_gender'] = np.where(matches['speaker_gender'] == 0, 'male', 'female')
    matches['interviewer_gender'] = np.where(matches['interviewer_gender'] == 0, 'male', 'female')
    matches['speaker_age'] = np.where(matches['speaker_age_strata'] == 0, 'old', 'young')
    matches['speaker_race'] = np.where(matches['dataset'] == 0, 'african_american', 'caucasian')
    matches = matches.drop(columns=['speaker_age_strata','dataset'])

    sampled_matches = sampled_matches.merge(
        matches[['segment_id','file_name']],left_on='first_id',right_on='segment_id'
    ).drop(columns='segment_id').rename(columns={'file_name':'first_file_name'})

    sampled_matches = sampled_matches.merge(
        matches[['segment_id','file_name']],left_on='second_id',right_on='segment_id'
    ).drop(columns='segment_id').rename(columns={'file_name':'second_file_name'})

    sampled_matches = pd.concat([
        sampled_matches[['first_id', 'second_file_name']].rename(
            columns={'first_id':'segment_id', 'second_file_name':'match_file_name'}),
        sampled_matches[['second_id', 'first_file_name']].rename(
            columns={'second_id':'segment_id', 'first_file_name':'match_file_name'}),
    ])

    matches = matches.merge(sampled_matches, on='segment_id')

    matches.to_csv(os.path.join(save_dir, 'clip_info.csv'),index=False)

    tsv_info = [save_dir] + (
            matches['file_name']
            + '\t'
            + matches['sample_length'].astype(int).astype(str)
    ).to_list()
    tsv_path = os.path.join('data','coraal_buckeye_joined', 'all.tsv')
    pd.Series(tsv_info).to_csv(tsv_path, index=False, header=False)



def validate_distributions():
    save_dir = os.path.join('data', 'coraal_buckeye_joined')
    matches = pd.read_csv(os.path.join(save_dir, 'clip_info.csv'))



    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    model_log_path = os.path.join(save_dir, 'equality_of_distributions.log')
    fh = logging.FileHandler(model_log_path)
    formatter = logging.Formatter(
        '%(asctime)s (%(levelname)s): %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    normality_test_result_1 = shapiro(matches[matches['speaker_race'] == 'african_american']['length'])
    logger.info('Shapiro-Wilk test for normality (for AA speakers) finds that distribution of audio clip length is not '
                 f'normal (p={normality_test_result_1.pvalue}, test statistic={normality_test_result_1.statistic})')
    normality_test_result_2 = shapiro(matches[matches['speaker_race'] == 'caucasian']['length'])
    logger.info('Shapiro-Wilk test for normality (for EA speakers) finds that distribution of audio clip length is not '
                 f'normal (p={normality_test_result_2.pvalue}, test statistic={normality_test_result_2.statistic})')

    kstresults = kstest(
        matches[matches['speaker_race'] == 'african_american']['length'],
        matches[matches['speaker_race'] == 'caucasian']['length']
    )

    logger.info(f'KS tests results suggest equality of distribution: Statistic: {kstresults.statistic},  Pvalue: {kstresults.pvalue}')


    matches['unified_file_name'] = np.where(
        matches['speaker_race'] == 'caucasian',
        matches['file_name'],
        matches['match_file_name']
    )
    pairs = []
    for fn, data in matches.groupby('unified_file_name'):
        data=data.copy()
        data = data.sort_values('speaker_race')
        pairs.append(data['sample_length'].values)

    pairs = pd.DataFrame(pairs)

    results = scipy.stats.ttest_rel(pairs[0], pairs[1])

    logger.info('We perform a matched pairs t-test to evaluate whether samples from european-american and '
                f'african-american speakers differ in length, and do not find evidence that they do.\n{results}')


def validate_sentiment():
    # PREPROCESS
    clip_info = pd.read_csv(os.path.join('data', 'coraal_buckeye_joined', 'clip_info.csv'))
    # Clean notes in transcripts
    clip_info['content'] = clip_info['content'].str.replace(r'\<VOCNOISE-?=?|EXT-?=?|LAUGH-?=?|HES-?=?|NOISE=?-?([^\>]*)\>', '\\1',regex=True)
    clip_info['content'] = clip_info['content'].str.replace(r'/unintelligible/', '',regex=True)
    clip_info['content'] = clip_info['content'].str.replace(r'/Inaudible/', '',regex=True, case=False)
    clip_info['content'] = clip_info['content'].str.replace(r'<clears throat>', '',regex=True, case=False)
    clip_info['content'] = clip_info['content'].str.replace(r'<um>', '',regex=True, case=False)
    clip_info['content'] = clip_info['content'].str.replace(r'<uh>', '',regex=True, case=False)

    clip_info['content'] = clip_info['content'].str.replace(r'\(pause [\d\.]*\)', '',regex=True)
    clip_info['content'] = clip_info['content'].str.replace(r'\(laughing\)', '',regex=True)
    clip_info['content'] = clip_info['content'].str.replace(r'\<UNKNOWN\>', '',regex=True)
    clip_info['content'] = clip_info['content'].str.replace(r'\<CUTOFF.*?\>', '',regex=True)
    clip_info['content'] = clip_info['content'].str.replace(r'\<laugh\>', '',regex=True)
    clip_info['content'] = clip_info['content'].str.replace(r'\<ts\>', '',regex=True)
    clip_info['content'] = clip_info['content'].str.replace(r'\(whispered\)', '',regex=True)
    clip_info['content'] = clip_info['content'].str.replace(r'\{E_TRANS\}', '',regex=True)
    clip_info['content'] = clip_info['content'].str.replace(r'\{B_TRANS\}', '',regex=True)
    clip_info['content'] = clip_info['content'].str.replace(r'\{B_TRANS\}', '',regex=True)
    clip_info['content'] = clip_info['content'].str.replace(r'\<.+=(.+)\>', '\\1',regex=True)
    clip_info['content'] = clip_info['content'].str.replace(r'[\<\>]', '',regex=True)
    clip_info['content'] = clip_info['content'].str.replace(r'_', ' ',regex=True)
    clip_info['content'] = clip_info['content'].str.replace(r'\s+', ' ',regex=True)
    clip_info['content'] = clip_info['content'].str.strip()


    # VADER
    clip_info['vader_compound'] = clip_info['content'].apply(lambda x: VaderSentimentIntensityAnalyzer().polarity_scores(x)['compound'])
    clip_info['vader_positive'] = clip_info['content'].apply(lambda x: VaderSentimentIntensityAnalyzer().polarity_scores(x)['pos'])
    clip_info['vader_negative'] = clip_info['content'].apply(lambda x: VaderSentimentIntensityAnalyzer().polarity_scores(x)['neg'])
    clip_info['vader_neutral'] = clip_info['content'].apply(lambda x: VaderSentimentIntensityAnalyzer().polarity_scores(x)['neu'])


    def score_text(text, tokenizer, label_map, model):
        encoded_input = tokenizer(text, return_tensors='pt')

        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        scores = {label_map[i]: float(score) for i, score in enumerate(scores)}
        return scores

    # MODEL 1
    m1_name = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(m1_name)
    config = AutoConfig.from_pretrained(m1_name)
    model = AutoModelForSequenceClassification.from_pretrained(m1_name)

    m1_scores = [score_text(c, tokenizer, config.id2label, model) for c in clip_info['content']]
    clip_info['m1_negative'] = [s['negative'] for s in m1_scores]
    clip_info['m1_neutral'] = [s['neutral'] for s in m1_scores]
    clip_info['m1_positive'] = [s['positive'] for s in m1_scores]

    # MODEL 2
    m2_name = f"cardiffnlp/twitter-xlm-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(m2_name)
    config = AutoConfig.from_pretrained(m2_name)
    model = AutoModelForSequenceClassification.from_pretrained(m2_name)

    m2_scores = [score_text(c, tokenizer, config.id2label, model) for c in clip_info['content']]
    clip_info['m2_negative'] = [s['negative'] for s in m2_scores]
    clip_info['m2_neutral'] = [s['neutral'] for s in m2_scores]
    clip_info['m2_positive'] = [s['positive'] for s in m2_scores]


    # MODEL 3
    task = 'sentiment'
    m3_name = f"cardiffnlp/twitter-roberta-base-{task}"
    tokenizer = AutoTokenizer.from_pretrained(m3_name)
    # download label mapping
    mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
    with urllib.request.urlopen(mapping_link) as f:
        html = f.read().decode('utf-8').split("\n")
        csvreader = csv.reader(html, delimiter='\t')
    labels = [row[1] for row in csvreader if len(row) > 1]

    model = AutoModelForSequenceClassification.from_pretrained(m3_name)
    m3_scores = [score_text(c, tokenizer, labels, model) for c in clip_info['content']]
    clip_info['m3_negative'] = [s['negative'] for s in m3_scores]
    clip_info['m3_neutral'] = [s['neutral'] for s in m3_scores]
    clip_info['m3_positive'] = [s['positive'] for s in m3_scores]

    # MODEL 4
    m4_name = "Seethal/sentiment_analysis_generic_dataset"
    tokenizer = AutoTokenizer.from_pretrained(m4_name)
    model = AutoModelForSequenceClassification.from_pretrained(m4_name)
    labels = ['negative', 'neutral', 'positive']
    m4_scores = [score_text(c, tokenizer, labels, model) for c in clip_info['content']]
    clip_info['m4_negative'] = [s['negative'] for s in m4_scores]
    clip_info['m4_neutral'] = [s['neutral'] for s in m4_scores]
    clip_info['m4_positive'] = [s['positive'] for s in m4_scores]


    for col in ['negative', 'neutral', 'positive', 'compound']:
        for model in ['vader', 'm1', 'm2', 'm3', 'm4']:
            col_name = f'{model}_{col}'
            if col_name not in clip_info.columns:
                continue
            clip_info[col_name] = clip_info[col_name].astype(float)




            aa_data = clip_info[clip_info['speaker_race'] == 'african_american'][col_name]
            ea_data = clip_info[clip_info['speaker_race'] == 'caucasian'][col_name]

            res = scipy.stats.ttest_ind(aa_data, ea_data, equal_var=False, axis=0)
            print(f'T-test for {model}, {col}: (p={res.pvalue}, test statistic={res.statistic})')






if __name__ == '__main__':
    matches, joined = get_full_matches()
    sampled_matches, full_matches = filter_matches(matches, joined)
    extract_matches(sampled_matches, full_matches)
    validate_distributions()
    validate_sentiment()
