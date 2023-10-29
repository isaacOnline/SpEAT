import seaborn as sns
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

def jitter(values,j):
    return values + np.random.normal(0,j,values.shape)

if __name__ == '__main__':
    # Load all results
    all_results = pd.read_csv(os.path.join('plots','standard_error','all_mean_mean_results.csv'))

    # Replace names for formatting in plots, and filter out data we didn't end up using
    all_results = all_results[
        ~(
            all_results['Target Dataset'].isin(['Speaker Accent Archive (Young/Old Native British English Speakers)',
                                                'Romero Rivas et al recordings'])
            & (all_results['Attribute Dataset'] == 'morgan_emotional_speech_set')
        )

        & ~all_results['Speech Model'].str.contains('_decoder')
    ].reset_index(drop=True)

    all_results['Test'] = np.where(
        (all_results['Authors'] == 'Pantos & Perkins'),
        'U.S. vs. Foreign (OLD)',
        np.where(
            all_results['Authors'] == 'SAA (U.S. vs. Korean Matches)',
            'U.S. vs. Foreign (NEW)',
            np.where(
                (all_results['Authors'] == 'Mitchell et al.') & (all_results['Test'] == 'Female vs. Male Speech'),
                'Female vs. Male (OLD)',
                np.where(
                    (all_results['Test'] == 'Female vs. Male Speech'),
                    'Female vs. Male (NEW)',
                    np.where(
                        (all_results['Authors'] == 'Mitchell et al.'),
                        'Human vs. Synthesized (OLD)',
                        np.where(
                            (all_results['Test'] == 'Human vs. Synthesized Speech'),
                            'Human vs. Synthesized (NEW)',
                            np.where(
                                (all_results['Target Dataset'] == 'TORGO Recordings'),
                                'Abled and Disabled (NEW)',
                                np.where(
                                    (all_results['Target Dataset'] == 'UASpeech Recordings'),
                                    'Abled and Disabled (OLD)',
                                    all_results['Test']
                                ))))))))

    all_results['Test'] = all_results['Test'].str.replace('Speech','')
    all_results['Test'] = all_results['Test'].str.replace('English Accents','')
    all_results['Test'] = all_results['Test'].str.replace('(European American vs. African American) vs. (Pleasant vs. Unpleasant)','EA vs. AA', regex=False)
    all_results['Test'] = all_results['Test'].str.replace('(Non-Dysarthric vs. Dysarthric) vs. (Pleasant vs. Unpleasant)','Non-Dysarthric vs. Dysarthric', regex=False)
    all_results['Test'] = np.where(
        (all_results['Target Dataset'] == 'Speaker Accent Archive (Young/Old Native American English Speakers)'),
        'Y vs. O (SAE)',
        np.where(
            (all_results['Target Dataset'] == 'Speaker Accent Archive (Young/Old Native British English Speakers)'),
            'Y vs. O (SBE)',
            all_results['Test']
        )
    )
    all_results['Test'] = all_results['Test'].str.strip()

    for c in ['X','Y','A','B']:
        all_results[c] = all_results[c].str.replace('Voice','')
        all_results[c] = all_results[c].str.replace('Audio Clips','')
        all_results[c] = all_results[c].str.replace('(morgan_emotional_speech_set)','',regex=False)
        all_results[c] = all_results[c].str.replace('(EU_Emotion_Stimulus_Set)','',regex=False)
        all_results[c] = all_results[c].str.strip()

    all_results = all_results.sort_values([
        'Authors','Test','Target Dataset','Attribute Dataset','Speech Model'
    ])


    all_results['Model Family'] = all_results['Speech Model'].map({
        'hubert_base_ls960': 'HuBERT',
        'hubert_large_ll60k': 'HuBERT',
        'hubert_xtralarge_ll60k': 'HuBERT',
        'wav2vec2_base': 'wav2vec 2.0',
        'wav2vec2_large_ll60k': 'wav2vec 2.0',
        'wav2vec2_large_ls960': 'wav2vec 2.0',
        'wavlm_base': 'WavLM',
        'wavlm_base_plus': 'WavLM',
        'wavlm_large': 'WavLM',
        'whisper_base_encoder': 'Whisper',
        'whisper_medium_encoder': 'Whisper',
        'whisper_small_encoder': 'Whisper',
        'whisper_base_en_encoder': 'Whisper',
        'whisper_medium_en_encoder': 'Whisper',
        'whisper_small_en_encoder': 'Whisper',
        'whisper_large_encoder': 'Whisper',
    })

    all_results['Number of Parameters'] = all_results['Speech Model'].map({
        'hubert_base_ls960': 90000000,
        'hubert_large_ll60k': 300000000,
        'hubert_xtralarge_ll60k': 1000000000,
        'wav2vec2_base': 95000000,
        'wav2vec2_large_ll60k': 317000000,
        'wav2vec2_large_ls960': 317000000,
        'wavlm_base': 94700000,
        'wavlm_base_plus': 94700000,
        'wavlm_large': 316620000,
        'whisper_base_encoder': 74000000,
        'whisper_base_en_encoder': 74000000,
        'whisper_small_encoder': 244000000,
        'whisper_small_en_encoder': 244000000,
        'whisper_medium_encoder': 769000000,
        'whisper_medium_en_encoder': 769000000,
        'whisper_large_encoder': 1550000000,
    }).astype(int)

    test_name_map = {
        'British vs. Spanish': 'British and Foreign',
        'Human vs. Synthesized (OLD)': 'Human and Synthesized (OLD)',
        'Human vs. Synthesized (NEW)': 'Human and Synthesized (NEW)',
        'U.S. vs. Foreign (OLD)': 'U.S. and Foreign (OLD)',
        'U.S. vs. Foreign (NEW)': 'U.S. and Foreign (NEW)',
        'Female vs. Male (OLD)': 'Female and Male (OLD)',
        'Female vs. Male (NEW)': 'Female and Male (NEW)',
        'EA vs. AA': 'EA and AA',
        'Non-Dysarthric vs. Dysarthric': 'Abled and Disabled',
        'Y vs. O (SAE)': 'Young and Old',
        'Y vs. O (SBE)': 'Young and Old (U.K.)',
    }

    all_results['Test'] = all_results['Test'].replace(test_name_map)

    all_results = all_results.rename(columns={'stdev':'Standard Error',
                                              'var':'Resampled d Variance',
                                              'Number of Target Embeddings per Group':'Number of Stimuli per Target Group'})

    all_results = all_results[~all_results['Test'].isin(['Y vs. O (SBE)', 'British vs. Spanish'])]
    plt.style.use(os.path.join('plots', 'default.mplstyle'))

    uk = ['British and Foreign', 'Young and Old (U.K.)']
    us = [t for t in all_results['Test'].unique() if t not in uk]
    us.sort()
    us_results = all_results[all_results['Test'].isin(us)].sort_values('Test').copy()
    us_results = us_results[~us_results['Test'].str.contains('(OLD)', regex=False)]
    us_results['Test'] = us_results['Test'].str.replace(' (NEW)', '', regex=False)



    plt.clf()
    matplotlib.rc('font', **{'size':8})
    f, (ax1) = plt.subplots(1,1,figsize=(6, 4.5))
    plt.rcParams['axes.axisbelow'] = True
    grid = sns.scatterplot(x=us_results['Number of Stimuli per Target Group'],y=all_results['Standard Error'],
                           alpha=0.2, marker='o',edgecolor=None)





    mns = us_results.groupby(['Number of Stimuli per Target Group'])['Standard Error'].max()
    plt.plot(mns, c='orange',linewidth=4)

    ax1.set_axisbelow(True)
    plt.grid(zorder=-100)
    grid.xaxis.set_major_locator(MultipleLocator(10))
    grid.yaxis.set_major_locator(MultipleLocator(0.25))
    grid.yaxis.set_minor_locator(MultipleLocator(0.2))
    plt.ylim(-0.01,1.01)

    plt.tight_layout()
    plt.savefig(os.path.join('plots','standard_error', 'images','se_without_hue.pdf'))



    plt.clf()
    grid = sns.jointplot(x=us_results['Number of Stimuli per Target Group'],y=us_results['Standard Error'], kind='scatter', hue=all_results['Test'])
    width = 6
    grid.fig.set_figwidth(width)
    grid.fig.set_figwidth(width * 1.6)
    plt.savefig(os.path.join('plots','standard_error', 'images','test.png'))

    plt.clf()
    grid = sns.jointplot(x=us_results['Number of Stimuli per Target Group'],y=us_results['Standard Error'], kind='scatter', hue=all_results['Model Family'])
    width = 6
    grid.fig.set_figwidth(width)
    grid.fig.set_figwidth(width * 1.6)
    plt.savefig(os.path.join('plots','standard_error', 'images','model_family.png'))


    plt.clf()
    grid = sns.jointplot(x=us_results['Number of Stimuli per Target Group'],y=us_results['Standard Error'], kind='scatter',  hue=all_results['Number of Parameters'])
    width = 6
    grid.fig.set_figwidth(width)
    grid.fig.set_figwidth(width * 1.6)
    plt.savefig(os.path.join('plots','standard_error', 'images','n_params.png'))

    plt.clf()
    grid = sns.jointplot(x=us_results['Number of Stimuli per Target Group'],y=us_results['Standard Error'], kind='scatter', hue=all_results['Attribute Dataset'])
    width = 6
    grid.fig.set_figwidth(width)
    grid.fig.set_figwidth(width * 1.6)
    plt.savefig(os.path.join('plots','standard_error', 'images','attribute_dataset.png'))