import os
import warnings

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys

sys.path.extend([PROJECT_ROOT])
import whisper

from shutil import copy2, rmtree
import torch

import soundfile as sf
import pandas as pd
from tqdm import tqdm
from npy_append_array import NpyAppendArray
from tempfile import NamedTemporaryFile

from embedding_extraction.embedding_aggregation import aggregate_sequence_embeddings




def extract_whisper_features(whisper_version, dataset_name, tsv_file_name, nlayers, only_store_aggregated):
    # Check if embeddings have already been extracted for this model
    whisper_encoder_name = 'whisper_' + whisper_version.replace('.', '_') + '_encoder'
    whisper_decoder_name = 'whisper_' + whisper_version.replace('.', '_') + '_decoder'
    encoder_embedding_dir = os.path.join('embeddings', dataset_name, whisper_encoder_name)
    decoder_embedding_dir = os.path.join('embeddings', dataset_name, whisper_decoder_name)
    if os.path.exists(os.path.join(encoder_embedding_dir, '.SUCCESS')) and os.path.exists(
            os.path.join(decoder_embedding_dir, '.SUCCESS')):
        return
    if os.path.exists(encoder_embedding_dir):
        rmtree(encoder_embedding_dir)
    if os.path.exists(decoder_embedding_dir):
        rmtree(decoder_embedding_dir)

    print(f'Model Version: whisper_{whisper_version}\nDataset: {dataset_name}')

    # Load model
    model = whisper.load_model(whisper_version, device=None,
                               download_root=os.path.join(PROJECT_ROOT, '.whisper_models'))

    # Load names of files that need to be added
    tsv_fp = os.path.join('data', dataset_name, f'{tsv_file_name}.tsv')
    wav_info = pd.read_csv(tsv_fp, sep='\t')
    data_root = wav_info.columns[0]
    wav_names = wav_info.index.tolist()

    # Open NpyAppendArrays to save the layer embeddings to
    encoder_layer_save_paths = [os.path.join(encoder_embedding_dir, f'layer_{i}', 'all_0_1.npy') for i in
                                range(nlayers)]
    for dir in [os.path.dirname(f) for f in encoder_layer_save_paths]:
        os.makedirs(dir, exist_ok=False)
    encoder_layer_arrays = [NpyAppendArray(fp) for fp in encoder_layer_save_paths]

    decoder_layer_save_paths = [os.path.join(decoder_embedding_dir, f'layer_{i}', 'all_0_1.npy') for i in
                                range(nlayers)]
    for dir in [os.path.dirname(f) for f in decoder_layer_save_paths]:
        os.makedirs(dir, exist_ok=False)
    decoder_layer_arrays = [NpyAppendArray(fp) for fp in decoder_layer_save_paths]

    # Find path of seq len file to save to
    encoder_seq_len_save_path = os.path.join(encoder_embedding_dir, f'layer_0', 'all_0_1.len')
    decoder_seq_len_save_path = os.path.join(decoder_embedding_dir, f'layer_0', 'all_0_1.len')

    # Open the seq len files
    with open(encoder_seq_len_save_path, "w") as enc_seq_len_f:
        with open(decoder_seq_len_save_path, "w") as dec_seq_len_f:
            # Iterate through wav files
            for wav_name in tqdm(wav_names, smoothing=0):
                # Read wav
                wav_fp = os.path.join(data_root, wav_name)
                wav_data, frame_rate = sf.read(wav_fp)
                assert frame_rate == 16000
                # If wav is stereo, convert to mono (HuBERT converts to mono by taking the mean of the channels, so I did that same)
                if wav_data.ndim == 2:
                    wav_data = wav_data.mean(axis=-1)

                # Reshape
                # wav_data = wav_data.reshape(1, wav_data.shape[0])

                tmp_path = NamedTemporaryFile()
                sf.write(tmp_path.name, wav_data, frame_rate, format='wav')

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    torch.manual_seed(775320)
                    no_speech_threshold = 0.9999 if dataset_name != 'TORGO' else None # Set high no speech threshold so model will output embeddings even for non speech. Didn't realize this could be set to None until doing TORGO.
                    logprob_threshold = -1.0 if dataset_name != 'TORGO' else None
                    compression_ratio_threshold = 2.4 if dataset_name != 'TORGO' else None
                    force = False if dataset_name != 'TORGO' else True
                    result = model.transcribe(tmp_path.name, no_speech_threshold=no_speech_threshold,
                                              logprob_threshold=logprob_threshold,
                                              compression_ratio_threshold=compression_ratio_threshold,
                                              force_extraction=force)
                encoder_embeddings = np.concatenate([segment['encoder_embeddings'] for segment in result['segments']],
                                                    axis=2).squeeze(0)
                decoder_embeddings = np.concatenate([segment['decoder_embeddings'] for segment in result['segments']],
                                                    axis=2).squeeze(0)
                tmp_path.close()

                enc_seq_len_f.write(str(encoder_embeddings.shape[1]) + '\n')
                dec_seq_len_f.write(str(decoder_embeddings.shape[1]) + '\n')

                # Save to layers
                for layer in range(nlayers):
                    encoder_layer_arrays[layer].append(np.ascontiguousarray(encoder_embeddings[layer]))
                    decoder_layer_arrays[layer].append(np.ascontiguousarray(decoder_embeddings[layer]))

    # Close NpyAppendArrays
    for arr in encoder_layer_arrays:
        arr.close()
        del arr
    for arr in decoder_layer_arrays:
        arr.close()
        del arr

    # Copy seqlens to other layer dirs
    for i in range(nlayers):
        if i != 0:
            copy2(encoder_seq_len_save_path, encoder_seq_len_save_path.replace('layer_0', f'layer_{i}'))
            copy2(decoder_seq_len_save_path, decoder_seq_len_save_path.replace('layer_0', f'layer_{i}'))

        # perform aggregation
        # aggregate_sequence_embeddings(model_name=whisper_decoder_name, layer=i, dataset_name=dataset_name)
        # aggregate_sequence_embeddings(model_name=whisper_encoder_name, layer=i, dataset_name=dataset_name)

        # Delete raw embeddings, if we're only storing the aggregated ones
        if only_store_aggregated:
            enc_embeddings_path = os.path.join(encoder_embedding_dir, f'layer_{i}', 'all_0_1.npy')
            os.remove(enc_embeddings_path)
            dec_embeddings_path = os.path.join(decoder_embedding_dir, f'layer_{i}', 'all_0_1.npy')
            os.remove(dec_embeddings_path)

    # Save success files
    with open(os.path.join(encoder_embedding_dir, '.SUCCESS'), 'w') as f:
        os.utime(os.path.join(encoder_embedding_dir, '.SUCCESS'), None)
    with open(os.path.join(decoder_embedding_dir, '.SUCCESS'), 'w') as f:
        os.utime(os.path.join(decoder_embedding_dir, '.SUCCESS'), None)


if __name__ == "__main__":
    for whisper_version, nlayers, only_store_aggregated in [
        ('large', 33, False),
        ('medium.en', 25, False),
        ('medium', 25, False),
        ('small.en', 13, False),
        ('small', 13, False),
        ('base.en', 7, False),
        ('base', 7, False),
    ]:

        datasets = [
            'TORGO',
            'human_synthesized',
            os.path.join('speech_accent_archive', 'male_female'),
            os.path.join('speech_accent_archive', 'us_korean'),
            os.path.join('speech_accent_archive','british_young_old'),
            os.path.join('speech_accent_archive','usa_young_old'),
            os.path.join('audio_iats','romero_rivas_et_al'),
            os.path.join('audio_iats','mitchell_et_al'),
            os.path.join('audio_iats','pantos_perkins'),
            'morgan_emotional_speech_set',
            'EU_Emotion_Stimulus_Set',
            'UASpeech',
            'coraal_buckeye_joined',
        ]
        for dataset_name in datasets:
            extract_whisper_features(
                whisper_version,
                dataset_name,
                'all',
                nlayers,
                only_store_aggregated
            )
