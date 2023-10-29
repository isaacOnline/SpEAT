import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.extend([PROJECT_ROOT])

import numpy as np
import pandas as pd
import torch

from embedding_extraction.dimension_prediction import EmbeddingDataset, DimensionPredictor, get_labels


if __name__ == '__main__':

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

    to_score = [
    ('TORGO', ['morgan_emotional_speech_set']),
    ('human_synthesized', ['morgan_emotional_speech_set']),
    (os.path.join('speech_accent_archive', 'male_female'), ['morgan_emotional_speech_set']),
    (os.path.join('speech_accent_archive', 'us_korean'), ['morgan_emotional_speech_set']),
    (os.path.join('speech_accent_archive', 'usa_young_old'), ['morgan_emotional_speech_set']),
        # (os.path.join('speech_accent_archive', 'usa_young_old'), ['morgan_emotional_speech_set']),
        # (os.path.join('speech_accent_archive', 'british_young_old'), ['EU_Emotion_Stimulus_Set','morgan_emotional_speech_set']),
        # (os.path.join('audio_iats', 'romero_rivas_et_al'), ['EU_Emotion_Stimulus_Set','morgan_emotional_speech_set']),
        # (os.path.join('audio_iats', 'pantos_perkins'), ['morgan_emotional_speech_set']),
        # (os.path.join('audio_iats', 'mitchell_et_al'), ['morgan_emotional_speech_set']),
        # ('UASpeech', ['morgan_emotional_speech_set']),
        ('coraal_buckeye_joined', ['morgan_emotional_speech_set']),
    ]
    # shuffle models
    models = models.sample(frac=1,replace=False).reset_index(drop=True)
    for dataset_to_score_name, valence_models_to_use in to_score:
        for _, (model_name, nlayers, timestamp) in models[['Model Name','nlayers', 'Start Time']].iterrows():
            valence_dataset_name, sequence_aggregation_method, dimension_name = 'morgan_emotional_speech_set', 'mean', 'valence'
            if valence_dataset_name in valence_models_to_use:
                run_id = f'{valence_dataset_name}_{dimension_name}_{model_name}_{timestamp}'
                save_dir = os.path.join('embeddings', dataset_to_score_name, 'valence_predictions')
                save_path = os.path.join(save_dir, f'{run_id}.npy')

                if os.path.exists(save_path):
                    continue

                print(f'{model_name}\n{dataset_to_score_name}\n{valence_dataset_name}\n')
                # Since the sequence aggregation method isn't saved with the .pt file, and can't be validated by pytorch, validate
                # that we're using the correct aggregation method by looking at the model's log
                log_path = os.path.join('dimension_models','logs',f'{run_id}.log')
                try:
                    with open(log_path) as f:
                        log = f.readlines()
                except FileNotFoundError:
                    print(f'Could not find log for {run_id}. Skipping.')
                    continue
                assert f'Using sequence aggregation method: {sequence_aggregation_method}\n' in log

                device = 'cpu'

                try:
                    dataset_to_score = EmbeddingDataset(dataset_to_score_name, dimension_name, model_name, nlayers, False,
                                                        lazy=True, device=device)
                except FileNotFoundError:
                    print(f'Could not find embeddings for {dataset_to_score_name}, {model_name}, {nlayers}. (Run id: {run_id}) Skipping.')
                    continue
                n_input_embd = dataset_to_score[0].shape[-1]

                # Load model
                categorical = False
                model = DimensionPredictor(nlayers, n_input_embd, sequence_aggregation_method, categorical=categorical, device=device)
                model = model.to(device)
                try:
                    loaded = torch.load(os.path.join('dimension_models','model_objects',f'{run_id}_BEST.pt'))
                except RuntimeError:
                    loaded = torch.load(os.path.join('dimension_models','model_objects',f'{run_id}_BEST.pt'), map_location=torch.device('cpu'))

                # Unloading by hand as model.load_state_dict seems to have trouble with mismatched devices
                for k, v in loaded.items():
                    key_tree = k.split('.')
                    current_v = model
                    for attr in key_tree:
                        parent = current_v
                        current_v = current_v.__getattr__(attr)
                    is_param = type(current_v) == torch.nn.Parameter
                    new_v = torch.nn.Parameter(v.to(device)) if is_param else v.to(device)
                    parent.__setattr__(attr, new_v)
                model.eval()

                predictions = np.concatenate([
                    model(
                        [
                            torch.DoubleTensor(X_i).to(device)
                        ]
                    ).cpu().detach().numpy()
                    for X_i in dataset_to_score])

                # Convert back to original scale
                _, label_min, label_range = get_labels(valence_dataset_name, dimension_name)


                predictions = predictions * label_range + label_min

                # Save embeddings
                os.makedirs(save_dir, exist_ok=True)
                np.save(save_path, predictions)
                print(f'Saved predictions to {save_path}')

