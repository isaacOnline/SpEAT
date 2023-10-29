import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import load_dataset_info
from plots.eats.plot_all_results import load_all_results

plt.style.use(os.path.join('plots', 'small_fonts.mplstyle'))

def cohen_d(x,y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

to_score = [
    ('TORGO', ['morgan_emotional_speech_set']),
    # ('UASpeech', ['morgan_emotional_speech_set']),

    ('human_synthesized', ['morgan_emotional_speech_set']),
    # (os.path.join('audio_iats', 'mitchell_et_al'), ['morgan_emotional_speech_set']),
    (os.path.join('speech_accent_archive', 'male_female'), ['morgan_emotional_speech_set']),

    # (os.path.join('audio_iats', 'pantos_perkins'), ['morgan_emotional_speech_set']),
    (os.path.join('speech_accent_archive', 'us_korean'), ['morgan_emotional_speech_set']),

    (os.path.join('speech_accent_archive', 'usa_young_old'), ['morgan_emotional_speech_set']),
    ('coraal_buckeye_joined', ['morgan_emotional_speech_set']),
]

JUST_MAIN_LR = False
if JUST_MAIN_LR:
    models = [
        ('morgan_emotional_speech_set', 'mean', 'valence', 'wavlm_base', 13, '1662262874', 1e-4),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'wavlm_base_plus', 13, '1662081826', 1e-4),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'wavlm_large', 25, '1662162141', 1e-4),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'wav2vec2_large_ll60k', 25, '1663291063', 1e-4),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'wav2vec2_large_ls960', 25, '1663367964', 1e-4),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'wav2vec2_base', 13, '1664237818', 1e-4),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'hubert_base_ls960', 13, '1661565382', 1e-4),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'hubert_large_ll60k', 25, '1664405708', 1e-4),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'hubert_xtralarge_ll60k', 49, '1662751606', 1e-4),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'whisper_base_encoder', 7, '1665595739', 1e-4),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'whisper_base_en_encoder', 7, '1666242267', 1e-4),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'whisper_small_encoder', 13, '1666236347', 1e-4),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'whisper_small_en_encoder', 13, '1666231576', 1e-4),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'whisper_medium_encoder', 25, '1666728028', 1e-4),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'whisper_medium_en_encoder', 25, '1666374320', 1e-4),
        ('morgan_emotional_speech_set', 'mean', 'valence', 'whisper_large_encoder', 33, '1665849979', 1e-4),
    ]
else:    
    models = pd.read_csv(os.path.join('dimension_models', 'all_results.csv'))
    models = models[
        (models['learning_rate'].isin([1e-3, 1e-4, 1e-5]))
        & (models['sequence_aggregation'] == 'mean')
        & (models['dimension_name'].isin(['pleasure', 'valence']))
        ].reset_index(drop=True).sort_values(['dataset_name', 'model_name'])
    models.columns = models.columns.str.replace('_', ' ').str.title()
    models = models.rename(columns={'Best R2': '$R^2$', 'Best Mae': 'MAE', 'Best Rmse': 'RMSE'})
    
    models['Dataset Name'] = models['Dataset Name'].replace(
        {'EU_Emotion_Stimulus_Set': 'EUESS',
         'morgan_emotional_speech_set': 'MESS'}
    )
    models = models[models['Dataset Name'] == 'MESS']
    models = models[~models['Model Name'].str.contains('_decoder')]
    
    nlayers_dict = {
        'wavlm_base': 13,
        'wavlm_base_plus': 13,
        'wavlm_large': 25,
        'wav2vec2_large_ll60k': 25,
        'wav2vec2_large_ls960': 25,
        'wav2vec2_base': 13,
        'hubert_base_ls960': 13,
        'hubert_large_ll60k': 25,
        'hubert_xtralarge_ll60k': 49,
        'whisper_base_encoder': 7,
        'whisper_base_en_encoder': 7,
        'whisper_small_encoder': 13,
        'whisper_small_en_encoder': 13,
        'whisper_medium_encoder': 25,
        'whisper_medium_en_encoder': 25,
        'whisper_large_encoder': 33,
    }
    models['nlayers'] = models['Model Name'].apply(lambda x: nlayers_dict[x])

def create_predictions_dataframe(test, group, X_Y, dataset, valence_dataset_name, model_name, predictions, lr):
    if len(group) == len(X_Y) == len(predictions):
        df = pd.DataFrame({
                    'Test': test,
                    'Group': group,
                    'Target': X_Y,
                    'Target Dataset': dataset,
                    'Dataset Used for Embedding Extraction':valence_dataset_name,
                    'Speech Model':model_name,
                    'Valence': predictions,
                    'Learning Rate': lr
                })
    else:
        raise ValueError('Length of lists must be the same.')
    return df


def get_speaker_info(dataset, which=None):
    ds_info = load_dataset_info(dataset)
    if dataset == 'speech_accent_archive/usa_young_old':
        test = 'Young and Old'
        group = np.where(ds_info['age_rank'] == 'low_age', 'Y', 'O')
        X_Y = np.where(ds_info['age_rank'] == 'low_age', 'X', 'Y')
    elif dataset == 'speech_accent_archive/british_young_old':
        test = 'Y and O (SBE)'
        group = np.where(ds_info['age_rank'] == 'low_age', 'Y', 'O')
        X_Y = np.where(ds_info['age_rank'] == 'low_age', 'X', 'Y')
    elif dataset == 'audio_iats/romero_rivas_et_al':
        test = 'British and Spanish'
        group = np.where(ds_info['speaker_type'] == 'native', 'British', 'Spanish')
        X_Y = np.where(ds_info['speaker_type'] == 'native', 'X', 'Y')
    elif dataset == 'audio_iats/romero_rivas_et_al':
        test = 'British and Spanish'
        group = np.where(ds_info['speaker_type'] == 'native', 'British', 'Spanish')
        X_Y = np.where(ds_info['speaker_type'] == 'native', 'X', 'Y')
    elif dataset == 'audio_iats/pantos_perkins':
        test = 'U.S. and Foreign (OLD)'
        group = np.where(ds_info['category'] == 'american', 'American', 'Korean')
        X_Y = np.where(ds_info['category'] == 'american', 'X', 'Y')
    elif dataset == 'audio_iats/mitchell_et_al' and which == 'female_male':
        test = 'Female and Male (OLD)'
        group = np.where(ds_info['male_or_female'] == 'female', 'Female', 'Male')
        X_Y = np.where(ds_info['male_or_female'] == 'female', 'X', 'Y')
    elif dataset == 'audio_iats/mitchell_et_al' and which == 'human_synthesized':
        test = 'Human and Synthesized (OLD)'
        group = np.where(ds_info['human_or_synthesized'] == 'human', 'Human', 'Synthesized')
        X_Y = np.where(ds_info['human_or_synthesized'] == 'human', 'X', 'Y')
    elif dataset == 'UASpeech':
        test = 'Abled and Disabled (OLD)'
        group = np.where(ds_info['type'] == 'non_dysarthric', 'Non-Dysarthric', 'Dysarthric')
        X_Y = np.where(ds_info['type'] == 'non_dysarthric', 'X', 'Y')
    elif dataset == 'coraal_buckeye_joined':
        test = 'EA and AA'
        group = np.where(ds_info['speaker_race'] == 'caucasian', 'European American', 'African American')
        X_Y = np.where(ds_info['speaker_race'] == 'caucasian', 'X', 'Y')
    elif dataset == 'TORGO':
        test = 'Abled and Disabled (NEW)'
        group = np.where(ds_info['dysarthric'] == 'control', 'Non-Dysarthric', 'Dysarthric')
        X_Y = np.where(ds_info['dysarthric'] == 'control', 'X', 'Y')
    elif dataset == 'human_synthesized':
        test = 'Human and Synthesized (NEW)'
        group = np.where(ds_info['human'], 'Human', 'Synthesized')
        X_Y = np.where(ds_info['human'], 'X', 'Y')
    elif dataset == os.path.join('speech_accent_archive', 'male_female'):
        ds_info = load_dataset_info(dataset)
        test = 'Female and Male (NEW)'
        group = np.where(ds_info['sex'] == 'female', 'Female','Male')
        X_Y = np.where(ds_info['sex'] == 'female', 'X', 'Y')
    elif dataset == os.path.join('speech_accent_archive', 'us_korean'):
        test = 'U.S. and Foreign (NEW)'
        group = np.where(ds_info['language'] == 'english', 'American', 'Korean')
        X_Y = np.where(ds_info['language'] == 'english', 'X', 'Y')
    else:
        raise ValueError

    return test, group, X_Y

# Create big dataframe with all predictions
all_predictions = []
for dataset, valence_models_to_use in to_score:
    if type(models) == pd.DataFrame:
        models['valence_dataset_name'] = 'morgan_emotional_speech_set'
        models['sequence_aggregation_method'] = 'mean'
        models['dimension_name'] = 'valence'
        models = models[[
            'valence_dataset_name', 'sequence_aggregation_method', 'dimension_name',
            'Model Name','nlayers','Start Time', 'Learning Rate']].values.tolist()
        
    for valence_dataset_name, sequence_aggregation_method, dimension_name, model_name, nlayers, timestamp, lr in models:
        if valence_dataset_name in valence_models_to_use:
            run_id = f'{valence_dataset_name}_{dimension_name}_{model_name}_{timestamp}'
            save_dir = os.path.join('embeddings', dataset, 'valence_predictions')
            save_path = os.path.join(save_dir, f'{run_id}.npy')
            try:
                predictions = np.load(save_path)
            except FileNotFoundError:
                save_dir = os.path.join('/Volumes','Backup Plus','research','SpEAT','embeddings', dataset, 'valence_predictions')
                save_path = os.path.join(save_dir, f'{run_id}.npy')
                predictions = np.load(save_path)

            if dataset != 'audio_iats/mitchell_et_al':
                test, group, X_Y = get_speaker_info(dataset)
                all_predictions.append(
                    create_predictions_dataframe(
                        test, group, X_Y, dataset, valence_dataset_name, model_name, predictions, lr
                    ))
            else:
                test, group, X_Y = get_speaker_info(dataset, 'human_synthesized')
                all_predictions.append(
                    create_predictions_dataframe(
                        test, group, X_Y, dataset, valence_dataset_name, model_name, predictions, lr
                    ))

                test, group, X_Y = get_speaker_info(dataset, 'female_male')
                all_predictions.append(
                    create_predictions_dataframe(
                        test, group, X_Y, dataset, valence_dataset_name, model_name, predictions, lr
                    ))

all_predictions = pd.concat(all_predictions)

all_predictions = all_predictions[
    ~all_predictions['Speech Model'].str.contains('_decoder')
    ].reset_index(drop=True)


all_predictions['Model Family'] = all_predictions['Speech Model'].map({
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

all_predictions['Number of Parameters'] = all_predictions['Speech Model'].map({
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



group_map = {'Y':'Young',
             'O':'Old',
             'Foreign':'Foreign Accent',
             'Spanish':'Foreign Accent',
             'British':'British Accent',
             'American':'American Accent',
             'Korean':'Foreign Accent'}

all_predictions['Group'] = all_predictions['Group'].replace(group_map)

def plot_cat(relevant_results_with_differences, cat_var = None, remove_legend=False):
    plt.close()
    plt.clf()
    relevant_results_with_differences = relevant_results_with_differences.rename(columns={
        'SpEAT d': 'SpEAT $d$\n(Pre-Trained Model)'}
    )
    if cat_var:
        order = relevant_results_with_differences[cat_var].unique()
        order.sort()

        ax = sns.scatterplot(x='SpEAT $d$\n(Pre-Trained Model)', y='Difference in Mean Valence\n(Downstream Model)',
                         data=relevant_results_with_differences,
                         hue=cat_var, hue_order=order, style=cat_var,
                         s=100)
    else:
        ax = sns.scatterplot(x='SpEAT $d$\n(Pre-Trained Model)', y='Difference in Mean Valence\n(Downstream Model)',
                         data=relevant_results_with_differences,
                         s=100)
    if remove_legend:
        plt.legend([],[], frameon=False)
    plt.xticks(np.arange(-1.5, 2, 0.5))
    plt.xlim(-1.75, 1.75)
    plt.axvline(0, color='gray')
    plt.axhline(0, color='gray')
    plt.tight_layout()

    cat_var = '' if not cat_var else f'_{cat_var.lower().replace(" ", "_")}'
    plt.savefig(os.path.join('plots', 'valence_by_group', f'valence_speat{cat_var}.pdf'))


def plot_overall(df, y_axis_col, aggregation_method, fname):
    plt.clf()
    df = df.copy()

    def jitter(y, jitter_size=0.25):
        return y + np.random.uniform(0, jitter_size * 2) - jitter_size


    speat_means = {k:v for k,v in df.groupby(y_axis_col)['SpEAT d'].mean().iteritems()}
    ser_means = {k:v for k,v in df.groupby(y_axis_col)['Cohen\'s d'].mean().iteritems()}
    type_ids = {k:(y_val, speat_means[k], ser_means[k]) for k,y_val in
                zip(df[y_axis_col].unique(),
                    range(len(df[y_axis_col].unique())-1, -1, -1))}

    df[y_axis_col] = df[y_axis_col].apply(lambda x: type_ids[x][0])
    df[y_axis_col] = df[y_axis_col].apply(lambda x: jitter(x))


    fig, (ax) = plt.subplots(1, 1, figsize=(10, 2.3))


    plt.scatter(y=df[y_axis_col],
                x=df['Cohen\'s d'],
                c='gray')


    plt.axvline(x=0, color='black')

    for mean_y, speat, ser in type_ids.values():
        plt.axhline(y=mean_y-0.5,c='gray',alpha=0.2)
        plt.scatter([speat], mean_y, marker='s', c='#D81B60',s=100,zorder=10)
        plt.scatter([ser], mean_y, marker='o', c='#FFC107',s=100,zorder=10)


    # label y axis
    plt.yticks(range(len(type_ids)))
    labs = list(type_ids.keys())
    labs.reverse()
    plt.gca().set_yticklabels(labs)


    # add legend
    handles = [
        plt.scatter([-10], [-100], marker='o', c='gray', label='Cohen\'s d\nfor Individ.\nDownstream\nModel', s=50),
        plt.scatter([-10], [-100], marker='s',  c='#D81B60', label='Mean SpEAT $d$\nfor Upstream\nModels', s=100),
               plt.scatter([-10], [-100],  marker='o', c='#FFC107', label='Mean Cohen\'s $d$\nfor Downstream\nModels', s=100)]
    ax.legend(handles=handles,
              # bbox_to_anchor=(0,0.85),
              loc='upper left')

    plt.xlim(-2.75, 2.75)
    plt.ylim(-0.5, len(type_ids)-0.5)

    plt.tight_layout()
    plt.savefig(os.path.join('plots', 'eats', 'images', f'{aggregation_method}_images', fname))



for ds in ['morgan_emotional_speech_set']:
    to_use = all_predictions[all_predictions['Dataset Used for Embedding Extraction'] == ds]

    plt.clf()

    order = to_use['Test'].unique()
    order.sort()
    sns.catplot(x='Valence',y='Test',hue='Target',data=to_use,kind='box', order=order, aspect=1.6,hue_order=['X', 'Y'])

    plt.savefig(os.path.join('plots','valence_by_group',f'{ds}.pdf'))

    all_results = load_all_results('mean_mean')
    uk = ['British and Foreign', 'Young and Old (U.K.)']
    us = [t for t in all_results['Test'].unique() if t not in uk]
    us.sort()
    us_results = all_results[all_results['Test'].isin(us)].sort_values('Test').copy()


    relevant_results_with_differences = []
    for i, result in us_results.iterrows():
        X_predictions = all_predictions[
            (all_predictions['Group'] == result['X'])
            & (all_predictions['Speech Model'] == result['Speech Model'])
            & (all_predictions['Test'] == result['Test'])
        ]
        Y_predictions = all_predictions[
            (all_predictions['Group'] == result['Y'])
            & (all_predictions['Speech Model'] == result['Speech Model'])
            & (all_predictions['Test'] == result['Test'])
        ]
        dif = None
        dif = X_predictions['Valence'].mean() - Y_predictions['Valence'].mean()
        div = X_predictions['Valence'].mean() / Y_predictions['Valence'].mean()
        coh = cohen_d(X_predictions['Valence'], Y_predictions['Valence'])



        relevant_results_with_differences.append(pd.concat([result.copy(), pd.Series([dif, div, coh], index=['Difference in Mean Valence\n(Downstream Model)', 'Div Mean Valences', 'Cohen\'s d'])]))

    relevant_results_with_differences = pd.DataFrame(relevant_results_with_differences)


    new_results = relevant_results_with_differences[~relevant_results_with_differences['Test'].str.contains('(OLD)', regex=False)].copy()
    new_results['Test'] = new_results['Test'].str.replace(' (NEW)', '', regex=False)

    new_results['same_direction'] = (((new_results['SpEAT d'] > 0) & (new_results['Cohen\'s d'] > 0)) | (
                (new_results['SpEAT d'] < 0) & (new_results['Cohen\'s d'] < 0)))

    plot_overall(new_results, 'Test', 'mean_mean', 'downstream_comparison.pdf')


    #  For context, the average $R^2$ for predicting valence in the MESS was 0.87.
    models['$R^2$'].mean()

    performances = models.groupby(['Model Name'])['$R^2$'].agg(['min','max','mean'])
    performances['dif'] = performances['max'] - performances['min']

    nr = nr.sort_values(['Test', 'Speech Model'])
    new_results = new_results.sort_values(['Test', 'Speech Model'])

    (nr['Cohen\'s d'] - new_results['Cohen\'s d']).abs().mean()

    (nr.groupby('Test')['Cohen\'s d'].mean() - new_results.groupby('Test')['Cohen\'s d'].mean()).abs().mean()

    new_results['Cohen\'s d'].abs().mean()
    nr['Cohen\'s d'].abs().mean()