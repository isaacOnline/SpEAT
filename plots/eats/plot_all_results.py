
import math
import os
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_all_results(aggregation_method):
    all_results = pd.read_csv(os.path.join('plots','eats', 'test_results',f'all_{aggregation_method}_results.csv'))
    all_results = all_results.drop_duplicates()

    # Replace names for formatting in plots, and filter out data we didn't end up using
    all_results = all_results[
        & ~(
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

    for c in ['X', 'Y', 'A', 'B']:
        all_results[c] = all_results[c].str.replace('Voice', '')
        all_results[c] = all_results[c].str.replace('Audio Clips', '')
        all_results[c] = all_results[c].str.replace('(morgan_emotional_speech_set)', '', regex=False)
        all_results[c] = all_results[c].str.replace('(EU_Emotion_Stimulus_Set)', '', regex=False)
        all_results[c] = all_results[c].str.strip()

    all_results = all_results.sort_values([
        'Authors', 'Test', 'Target Dataset', 'Attribute Dataset', 'Speech Model'
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


    return all_results



def plot_overall(df, y_axis_col, aggregation_method, fname, include_black_lines=False, dims=None):
    plt.clf()
    df = df.copy()

    def jitter(y, jitter_size=0.25):
        return y + random.uniform(0, jitter_size * 2) - jitter_size


    speat_means = {k:v for k,v in df.groupby(y_axis_col)['SpEAT d'].mean().iteritems()}
    iat_means = {'Abled and Disabled': 0.45, 'Abled and Disabled (OLD)': 0.45, 'Abled and Disabled (NEW)': 0.45, 'Abled and Disabled (TORGO)': 0.45, 'Abled and Disabled (UASpeech)':.45,
                 'EA and AA': 0.37,
                 'Female and Male': 0.31, 'Female and Male (NEW)': 0.31, 'Female and Male (OLD)': 0.31, 'Female and Male (SAA)': 0.31, 'Female and Male (Mitchell et al.)': 0.31,
                 'Human and Synthesized': 0.33, 'Human and Synthesized (NEW)': 0.33, 'Human and Synthesized (OLD)': 0.33,'Human and Synthesized (SAA)': 0.33, 'Human and Synthesized (Mitchell et al.)': 0.33,
                 'U.S. and Foreign': 0.32, 'U.S. and Foreign (NEW)': 0.32, 'U.S. and Foreign (OLD)': 0.32, 'U.S. and Foreign (SAA)': 0.32, 'U.S. and Foreign (Pantos and Perkins)': 0.32,
                 'Young and Old': 0.49, 'Young and Old (U.K.)': 0.49, 'British and Foreign': None}
    type_ids = {k:(y_val, speat_means[k],iat_means[k]) for k,y_val in
                zip(df[y_axis_col].unique(),
                    range(len(df[y_axis_col].unique())-1, -1, -1))}

    df[y_axis_col] = df[y_axis_col].apply(lambda x: type_ids[x][0])
    df[y_axis_col] = df[y_axis_col].apply(lambda x: jitter(x))

    dims = (10, 2.3) if dims == None else dims
    fig, (ax) = plt.subplots(1, 1, figsize=dims)

    plt.axvline(x=0, color='black', zorder=-1000)

    plt.scatter(y=df[y_axis_col],
                marker='s',
                x=df['SpEAT d'],
                c='gray')



    for i, (mean_y, speat, iat) in enumerate(type_ids.values()):
        if include_black_lines and i % 2 != 0:
            plt.axhline(y=mean_y - 0.5, c='black', alpha=0.8)
        else:
            plt.axhline(y=mean_y - 0.5, c='gray',alpha=0.4)
        plt.scatter([speat], mean_y, marker='s', c='#D81B60',s=100,zorder=10)
        plt.scatter([iat], mean_y, marker='v', c='#1E88E5',s=100,zorder=10)


    # label y axis
    plt.yticks(range(len(type_ids)))
    labs = list(type_ids.keys())
    labs.reverse()
    plt.gca().set_yticklabels(labs)


    # add legend
    handles = [
        plt.scatter([-10], [-100], marker='s', c='gray', label='SpEAT $d$ for \nIndivid. Model', s=50),
        plt.scatter([-10], [-100], marker='s',  c='#D81B60', label='Mean SpEAT $d$', s=100),
               plt.scatter([-10], [-100], marker='v', c='#1E88E5', label='Mean IAT $D$', s=100)]
    ax.legend(handles=handles,
              # bbox_to_anchor=(0,0.85),
              loc='upper left')

    plt.xlim(-1.99, 1.99)
    plt.ylim(-0.5, len(type_ids)-0.5)

    plt.tight_layout()
    plt.savefig(os.path.join('plots', 'eats', 'images', f'{aggregation_method}_images', fname))


def plot_cat(df, y_axis_col, hue_col, aggregation_method, dataset_name=''):
    plt.clf()
    df = df.copy()

    def jitter(y, jitter_size=0.25):
        return y + random.uniform(0, jitter_size * 2) - jitter_size

    type_ids = {k:v for k,v in zip(df[y_axis_col].unique(), range(len(df[y_axis_col].unique())-1,-1,-1))}

    df[y_axis_col] = df[y_axis_col].apply(lambda x: type_ids[x])
    df[y_axis_col] = df[y_axis_col].apply(lambda x: jitter(x))

    colors = ['#332288','#88CCEE','#DDCC77','#882255','#117733','#E83C59','#88CCEE','#AA4499']
    markers = ['o','P','s','v','X', 'D']
    mmap = {k:[c,m] for k, c, m in zip(df[hue_col].unique(),
                                       colors[:len(df[hue_col].unique())],
                                       markers[:len(df[hue_col].unique())],
                                       )}

    fig, (ax) = plt.subplots(1, 1, figsize=(10, 2.3))

    for v, (c, m) in mmap.items():
        plt.scatter(y=df[y_axis_col][df[hue_col] == v],
                    x=df['SpEAT d'][df[hue_col] == v],
                    c=c,
                    marker = m)

    # add legend
    handles = [plt.Line2D([0], [0], marker=m, color='w', markerfacecolor=c, label=v, markersize=8) for v, (c,m) in
               mmap.items()]
    ax.legend(title=hue_col, handles=handles,
              # bbox_to_anchor=(0,0.85),
              loc='upper right')


    plt.yticks(range(len(type_ids)))
    labs = list(type_ids.keys())
    labs.reverse()
    plt.gca().set_yticklabels(labs)

    plt.axvline(x=0, color='black')

    for mean_y in range(len(type_ids) - 1):
        plt.axhline(y=mean_y+0.5,c='gray',alpha=0.2)


    plt.xlim(-1.99, 1.99)

    plt.tight_layout()
    plt.savefig(os.path.join('plots', 'eats', 'images', f'{aggregation_method}_images',
                             f'{dataset_name}{hue_col.lower().replace(" ","_")}.pdf'))


if __name__ == '__main__':
    all_all_results = {}
    for layer_agg in ['mean',
                      'first', 'second', 'q1', 'q2', 'q3', 'penultimate', 'last', 'min', 'max'
                      ]:
        for seq_agg in [
            'mean','min', 'max'
                          # 'first', 'second', 'q1', 'q2', 'q3', 'penultimate', 'last',
        ]:
            aggregation_method = f'{layer_agg}_{seq_agg}'

            os.makedirs(os.path.join('plots', 'eats', 'images', f'{aggregation_method}_images'), exist_ok=True)


            plt.style.use(os.path.join('plots', 'small_fonts.mplstyle'))


            all_results = load_all_results(aggregation_method)
            plt.clf()

            plt.style.use(os.path.join('plots', 'small_fonts.mplstyle'))

            iat_effect_sizes = pd.DataFrame({
                'Test': ['Abled and Disabled', 'EA and AA', 'Female and Male', 'Female and Male',
                          'Human and Synthesized', 'U.S. and Foreign', 'Young and Old'],
                'SpEAT d':[0.45, 0.37, 0.33, 0.02, 0.42, 0.32, 0.49]
            })

            # filter to U.K. results
            uk = ['British and Foreign', 'Young and Old (U.K.)']
            uk_results = all_results[all_results['Test'].isin(uk)].sort_values('Test').copy()
            plot_overall(uk_results, 'Test', aggregation_method, f'uk_results.pdf', dims=(10, 2))

            # Filter to only U.S. results

            us = [t for t in all_results['Test'].unique() if t not in uk]
            us.sort()
            us_results = all_results[all_results['Test'].isin(us)].sort_values('Test').copy()
            us_results = us_results[~us_results['Test'].str.contains('(OLD)', regex=False)]
            us_results['Test'] = us_results['Test'].str.replace(' (NEW)', '', regex=False)
            us_results['Significance'] = np.where(
                us_results['SpEAT p'] < (0.05 / len(us_results)),
                '**',
                np.where(
                    us_results['SpEAT p'] < (0.05),
                    '*',
                    '0')
            )
            us_results['significance_color'] = np.where(
                us_results['Significance'] == '**',
                1,
                np.where(
                    us_results['Significance'] == '*',
                    0.5,
                    0))
            all_all_results[aggregation_method] = us_results[['Speech Model', 'Test', 'SpEAT d', 'IAT d', 'significance_color']].copy()

            plt.clf()
            f, (ax1) = plt.subplots(1, 1, figsize=(10, 4))

            heat_map_data = us_results.pivot(index='Speech Model', columns='Test', values='significance_color').T
            heat_map_data.columns = heat_map_data.columns.str.replace('_encoder','')
            # Plot heatmap of significance values
            sns.heatmap(heat_map_data, cmap='Blues', cbar=False, linewidths=0.5, linecolor='black', ax=ax1)
            plt.xticks(rotation=40, ha='right')
            plt.legend(handles=[
                plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#08306B',
                           markeredgecolor='black',
                           label='Significant at 0.01\nAfter Correction', markersize=8),
                plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#6AAED6',
                           label='Significant at 0.01\nOnly',markeredgecolor='black',
                           markersize=8),
                plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#F7FBFF', label='Not Significant', markeredgecolor='black',
                           markersize=8),
            ],  # place outside figure along right edge of figure
                bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.xlabel('')
            plt.ylabel('')
            plt.tight_layout()

            # save figure
            f.savefig(
                os.path.join('plots', 'eats', 'images', f'{aggregation_method}_images', 'significance_heatmap.pdf'))

            plt.clf()
            plt.style.use(os.path.join('plots', 'small_fonts.mplstyle'))
            a = sns.scatterplot(x='SpEAT d', y='Test', data=us_results, hue='Number of Parameters', s=100)
            plt.xlim(-1.6, 1.6)
            plt.axvline(x=0, color='black')
            plt.tight_layout()
            plt.savefig(os.path.join('plots', 'eats', 'images', f'{aggregation_method}_images', 'number_of_parameters.pdf'))

            plot_cat(df=us_results, y_axis_col='Test', hue_col='Model Family', aggregation_method=aggregation_method)
            plot_cat(df=us_results, y_axis_col='Test', hue_col='Significance', aggregation_method=aggregation_method)
            plot_cat(df=us_results, y_axis_col='Test', hue_col='Number of Target Embeddings per Group',
                     aggregation_method=aggregation_method)

            plot_overall(us_results, 'Test', aggregation_method, f'overall_results.pdf')

            # Plot new vs. old results
            new_old_results = all_results[all_results['Test'].str.contains('NEW') | all_results['Test'].str.contains('OLD')].sort_values('Test').copy()

            new_old_results['Test'] = new_old_results['Test'].replace(
                {'Abled and Disabled (NEW)':'Abled and Disabled (TORGO)',
                'Abled and Disabled (OLD)':'Abled and Disabled (UASpeech)',
                'Female and Male (NEW)':'Female and Male (SAA)',
                'Female and Male (OLD)':'Female and Male (Mitchell et al.)',
                'Human and Synthesized (NEW)':'Human and Synthesized (SAA)',
                'Human and Synthesized (OLD)':'Human and Synthesized (Mitchell et al.)',
                'U.S. and Foreign (NEW)': 'U.S. and Foreign (SAA)',
                'U.S. and Foreign (OLD)': 'U.S. and Foreign (Pantos and Perkins)'
                 }
            )
            plt.clf()
            plt.style.use(os.path.join('plots', 'small_fonts.mplstyle'))
            a = sns.scatterplot(x='SpEAT d', y='Test', data=new_old_results, hue='Number of Parameters', s=100)
            plt.xlim(-1.6, 1.6)
            plt.axvline(x=0, color='black')
            plt.tight_layout()
            plt.savefig(os.path.join('plots', 'eats', 'images', f'{aggregation_method}_images', 'new_old_number_of_parameters.pdf'))

            plot_cat(df=new_old_results, y_axis_col='Test', hue_col='Model Family', aggregation_method=aggregation_method,
                     dataset_name='new_old_')
            plot_cat(df=new_old_results, y_axis_col='Test', hue_col='Number of Target Embeddings per Group',
                     aggregation_method=aggregation_method, dataset_name='new_old_')

            plot_overall(new_old_results, 'Test', aggregation_method, f'new_old_overall_results.pdf', include_black_lines=True)
            print(aggregation_method,
                  ((all_results['Test'].value_counts() == 16).all()
                   and len(new_old_results['Test'].value_counts()) == 8
                   and len(us_results['Test'].value_counts()) == 6))


    joined = all_all_results['mean_mean'].merge(all_all_results['max_mean'], on=['Speech Model', 'Test', 'IAT d'], suffixes=['_mean_mean', '_max_mean'])
    for k in sorted(all_all_results, key=lambda k: k.split('_')[1]):
        v = all_all_results[k]
        v = v.rename(columns={'SpEAT d': f'SpEAT d_{k}'})
        if k != 'mean_mean' and k != 'max_mean':
            joined = joined.merge(v, on=['Speech Model', 'Test', 'IAT d'])

    joined = joined[[c for c in joined.columns if'signif' not in c]].copy()
    means = (joined[['SpEAT d_mean_mean'] + joined.columns[4:].to_list()] > 0).mean()
    means = pd.DataFrame(means).reset_index()
    means.columns = ['agg', 'Percent Positive']
    means['agg'] = means['agg'].str.replace('SpEAT d_', '')
    means['Layer Aggregation'] = means['agg'].str.split('_').str[0]
    means['Sequence Aggregation'] = means['agg'].str.split('_').str[1]
    pivot = means.pivot(index='Layer Aggregation', columns='Sequence Aggregation', values='Percent Positive')

    pivot.loc[['mean','min', 'max', 'first', 'second', 'q1', 'q2', 'q3', 'penultimate', 'last']][['mean','min','max']]

    joined['Model Family'] = joined['Speech Model'].map({
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
    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    # Anova
    sm.stats.anova_lm(
        ols('SpEAT_d_mean_mean ~ Model_Family*Test', data=joined.rename(columns=lambda x: x.replace(' ', '_'))).fit(),
        typ=1)

    # We in fact find that considering all 30 aggregation strategies and 6 social groups, exactly 150 of 180 (83%)
    # average SpEAT $d$ values are in the same direction as the related IAT $D$.
    mns = joined.groupby('Test').mean().copy()
    for c in mns.columns:
        if mns[c].dtype != 'object':
            mns[c] = mns[c] > 0

    mns = mns[[c for c in mns.columns if 'SpEAT' in c]].copy()
    mns.sum().sum() / (30 * 6)

    # Across the 30 aggregation strategies, the standard deviation of the overall mean SpEAT $d$ is 0.14
    joined.mean().std()

    # whose average magnitude was 0.70 across all sets of stimuli and models
    joined['SpEAT d_mean_mean'].abs().mean()


# To evaluate the extent to which our results were affected by the embedding extraction method we use, we also test a variety of other methods
# for aggregating embeddings across layers and across sequences. We then compare the percent of SpEAT effect sizes that are
# positive out of the 96 tests we perform. Across layers, we aggregate using either the mean, min, or max---or by simply selecting embeddings from the first layer,
# second layer, first quartile layer, median layer, third quartile layer, penultimate layer, or last layer, and ignoring other layers. Across sequences, we
# also aggregate by using either the mean, min, or max---or by selecting the last set of embeddings in the sequence.
# While aggregating embeddings by selecting the last set of embeddings does not appear to consistently result in more positive effect sizes than
# negative, we note that all but one other methods of aggregating embeddings result in more positive effect sizes than negative. Other trends
# are visible in the table as well, for example that using the mean to aggregate across the sequence results in more positive effect sizes,
# or that taking embeddings from the first layer results in fewer positive effect sizes.

def plotting_fun(my_data, fname, test_name):
    plt.clf()
    sns.lineplot(data=my_data.iloc[2:], dashes=False, legend=False)
    plt.ylim(-2, 2)

    for i in [1, 2, 5, 9, 13, 16, 17]:
        plt.axvline(x=i, color='gray', alpha=0.5)

    plt.xticks([1, 2, 5, 9, 13, 16, 17], labels=['First','Second','Q1','Q2','Q3','Penultimate','Last'])

    # 45 degree tilt to x-axis labels
    for tick in plt.gca().get_xticklabels():
        tick.set_rotation(30)
        tick.set_ha('right')

    plt.title(test_name)
    plt.xlabel('Layer')
    plt.ylabel('SpEAT $d$')

    plt.tight_layout()
    plt.savefig(f'plots/eats/images/mean_mean_images/layers/{fname}.pdf')




    a = joined[['Speech Model', 'Test', 'SpEAT d_first_mean', 'SpEAT d_second_mean', 'SpEAT d_q1_mean',
            'SpEAT d_q2_mean', 'SpEAT d_q3_mean', 'SpEAT d_penultimate_mean',
            'SpEAT d_last_mean']].copy().T

    a.index=[
        'Speech Model', 'Test',
            1, 2, 5,
            9,
            13, 16, 17
    ]

    f, (ax1) = plt.subplots(1, 1, figsize=(10, 8))
    for tst in joined['Test'].unique():
        dat = (a.T[a.T['Test'] == tst]).T.copy()

        plotting_fun(dat, tst, tst)









