import os

import numpy as np
import torch

from embedding_extraction.dimension_prediction import EmbeddingDataset, DimensionPredictor


if __name__ == '__main__':
    to_score = [
        # (os.path.join('speech_accent_archive', 'usa_young_old'), ['morgan_emotional_speech_set']),
        # (os.path.join('speech_accent_archive', 'british_young_old'), ['EU_Emotion_Stimulus_Set','morgan_emotional_speech_set']),
        # (os.path.join('audio_iats', 'romero_rivas_et_al'), ['EU_Emotion_Stimulus_Set','morgan_emotional_speech_set']),
        # (os.path.join('audio_iats', 'pantos_perkins'), ['morgan_emotional_speech_set']),
        # (os.path.join('audio_iats', 'mitchell_et_al'), ['morgan_emotional_speech_set']),
        # ('UASpeech', ['morgan_emotional_speech_set']),
        ('coraal_buckeye_joined', ['morgan_emotional_speech_set']),
        # ('EU_Emotion_Stimulus_Set', ['EU_Emotion_Stimulus_Set']),
        ('morgan_emotional_speech_set', ['morgan_emotional_speech_set']),
    ]
    for dataset_to_score_name, valence_models_to_use in to_score:
        models = [
            # ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'wavlm_base', 13, '1662278003'),
            # ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'wavlm_base_plus', 13, '1662122581'),
            # ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'wavlm_large', 25, '1662267816'),
            # ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'hubert_base_ls960', 13, '1661404253'),
            # ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'hubert_large_ll60k', 25, '1664498938'),
            # ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'hubert_xtralarge_ll60k', 49, '1663008589'),
            # ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'wav2vec2_large_ll60k', 25, '1663446589'),
            # ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'wav2vec2_large_ls960', 25, '1663482418'),
            # ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'wav2vec2_base', 13, '1664257024'),
            ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'whisper_base_encoder', 7, '1665605059'),
            ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'whisper_base_decoder', 7, '1665611782'),
            ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'whisper_base_en_encoder', 7, '1666255677'),
            ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'whisper_base_en_decoder', 7, '1666258986'),
            ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'whisper_small_encoder', 13, '1666251448'),
            ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'whisper_small_decoder', 13, '1666258498'),
            ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'whisper_small_en_encoder', 13, '1666247229'),
            ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'whisper_small_en_decoder', 13, '1666257882'),
            ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'whisper_medium_encoder', 25, '1666803677'),
            ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'whisper_medium_decoder', 25, '1666810158'),
            ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'whisper_medium_en_encoder', 25, '1666389810'),
            ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'whisper_medium_en_decoder', 25, '1666396216'),
            ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'whisper_large_encoder', 33, '1666059780'),
            ('EU_Emotion_Stimulus_Set', 'mean', 'valence', 'whisper_large_decoder', 33, '1666082369'),
            # ('mean', 'pleasure', 'hubert_base_ls960', 13, '1662008402'),
            # ('mean', 'pleasure', 'wavlm_base', 13, '1662367401'),
            # ('mean', 'pleasure', 'wavlm_base_plus', 13, '1662127512'),
            # ('morgan_emotional_speech_set', 'mean', 'valence', 'wavlm_base', 13, '1662262874'),
            # ('morgan_emotional_speech_set', 'mean', 'valence', 'wavlm_base_plus', 13, '1662081826'),
            # ('morgan_emotional_speech_set', 'mean', 'valence', 'wavlm_large', 25, '1662162141'),
            # ('morgan_emotional_speech_set', 'mean', 'valence', 'wav2vec2_large_ll60k', 25, '1663291063'),
            # ('morgan_emotional_speech_set', 'mean', 'valence', 'wav2vec2_large_ls960', 25, '1663367964'),
            # ('morgan_emotional_speech_set', 'mean', 'valence', 'wav2vec2_base', 13, '1664237818'),
            # ('morgan_emotional_speech_set', 'mean', 'valence', 'hubert_base_ls960', 13, '1661565382'),
            # ('morgan_emotional_speech_set', 'mean', 'valence', 'hubert_large_ll60k', 25, '1664405708'),
            # ('morgan_emotional_speech_set', 'mean', 'valence', 'hubert_xtralarge_ll60k', 49, '1662751606'),
            ('morgan_emotional_speech_set', 'mean', 'valence', 'whisper_base_encoder', 7, '1665595739'),
            ('morgan_emotional_speech_set', 'mean', 'valence', 'whisper_base_decoder', 7, '1665603675'),
            ('morgan_emotional_speech_set', 'mean', 'valence', 'whisper_base_en_encoder', 7, '1666242267'),
            ('morgan_emotional_speech_set', 'mean', 'valence', 'whisper_base_en_decoder', 7, '1666246777'),
            ('morgan_emotional_speech_set', 'mean', 'valence', 'whisper_small_encoder', 13, '1666236347'),
            ('morgan_emotional_speech_set', 'mean', 'valence', 'whisper_small_decoder', 13, '1666246053'),
            ('morgan_emotional_speech_set', 'mean', 'valence', 'whisper_small_en_encoder', 13, '1666231576'),
            ('morgan_emotional_speech_set', 'mean', 'valence', 'whisper_small_en_decoder', 13, '1666245417'),
            ('morgan_emotional_speech_set', 'mean', 'valence', 'whisper_medium_encoder', 25, '1666728028'),
            ('morgan_emotional_speech_set', 'mean', 'valence', 'whisper_medium_decoder', 25, '1666738776'),
            ('morgan_emotional_speech_set', 'mean', 'valence', 'whisper_medium_en_encoder', 25, '1666374320'),
            ('morgan_emotional_speech_set', 'mean', 'valence', 'whisper_medium_en_decoder', 25, '1666383843'),
            ('morgan_emotional_speech_set', 'mean', 'valence', 'whisper_large_encoder', 33, '1665849979'),
            ('morgan_emotional_speech_set', 'mean', 'valence', 'whisper_large_decoder', 33, '1665973628'),
        ]
        for valence_dataset_name, sequence_aggregation_method, dimension_name, model_name, nlayers, timestamp in models:
            if valence_dataset_name in valence_models_to_use:
                run_id = f'{valence_dataset_name}_{dimension_name}_{model_name}_{timestamp}'
                save_dir = os.path.join('embeddings', dataset_to_score_name, 'model_aggregated')
                save_path = os.path.join(save_dir, f'{run_id}.npy')

                if os.path.exists(save_path):
                    continue

                print(f'{model_name}\n{dataset_to_score_name}\n{valence_dataset_name}\n')
                # Since the sequence aggregation method isn't saved with the .pt file, and can't be validated by pytorch, validate
                # that we're using the correct aggregation method by looking at the model's log
                log_path = os.path.join('dimension_models','logs',f'{run_id}.log')
                with open(log_path) as f:
                    log = f.readlines()
                assert f'Using sequence aggregation method: {sequence_aggregation_method}\n' in log

                device = 'cpu'

                dataset_to_score = EmbeddingDataset(dataset_to_score_name, dimension_name, model_name, nlayers, False,
                                                    lazy=True, device=device)
                n_input_embd = dataset_to_score[0].shape[-1]

                # Load model
                categorical = False
                model = DimensionPredictor(nlayers, n_input_embd, sequence_aggregation_method, categorical=categorical, device=device)
                model = model.to(device)
                loaded = torch.load(os.path.join('dimension_models','model_objects',f'{run_id}_BEST.pt'))
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


                # Extract embeddings
                embeddings = model.extract_embeddings(dataset_to_score)

                # Save embeddings
                os.makedirs(save_dir, exist_ok=True)
                np.save(save_path, embeddings)



