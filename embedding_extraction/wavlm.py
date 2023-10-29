import gc
import importlib
from shutil import copy2, rmtree

import pandas as pd
import torch
import soundfile as sf
from tqdm import tqdm
from npy_append_array import NpyAppendArray

import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(PROJECT_ROOT)
import sys
sys.path.extend([PROJECT_ROOT])
from embedding_extraction.embedding_aggregation import aggregate_sequence_embeddings

sys.path.extend([os.path.join('unilm','wavlm')])
from WavLM import WavLM, WavLMConfig

# load the pre-trained checkpoints


def extract_wavlm_features(wavlm_version, dataset_name, tsv_file_name, nlayers, only_store_aggregated):
    # Check if embeddings have already been extracted for this model
    embedding_dir = os.path.join('embeddings',dataset_name, wavlm_version)
    if os.path.exists(embedding_dir):
        if os.path.exists(os.path.join(embedding_dir, '.SUCCESS')):
            return
        else:
            rmtree(embedding_dir)

    print(f'Model Version: {wavlm_version}\nDataset: {dataset_name}')

    # Load model
    ckpt = os.path.join('models', f'{wavlm_version}.pt', )
    checkpoint = torch.load(ckpt)
    cfg = WavLMConfig(checkpoint['cfg'])
    model = WavLM(cfg)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Load names of files that need to be added
    tsv_fp = os.path.join('data', dataset_name, f'{tsv_file_name}.tsv')
    wav_info = pd.read_csv(tsv_fp,sep='\t')
    data_root = wav_info.columns[0]
    wav_names = wav_info.index.tolist()

    # Open NpyAppendArrays to save the layer embeddings to
    layer_save_paths = [os.path.join(embedding_dir, f'layer_{i}', 'all_0_1.npy') for i in range(nlayers)]
    for dir in [os.path.dirname(f) for f in layer_save_paths]:
        os.makedirs(dir, exist_ok=False)
    layer_arrays = [NpyAppendArray(fp) for fp in layer_save_paths]

    # Find path of seq len file to save to
    seq_len_save_path = os.path.join(embedding_dir, f'layer_0', 'all_0_1.len')

    # Open the seq len file
    with open(seq_len_save_path, "w") as seq_len_f:
        # Iterate through wav files
        for wav_name in tqdm(wav_names, smoothing = 0):
            # Read wav
            wav_fp = os.path.join(data_root, wav_name)
            wav_data, frame_rate = sf.read(wav_fp)
            assert frame_rate == 16000
            # If wav is stereo, convert to mono (HuBERT converts to mono by taking the mean of the channels, so I did that same)
            if wav_data.ndim == 2:
                wav_data = wav_data.mean(axis=-1)

            # Convert to pytorch, change to correct shape
            wav_data = wav_data.reshape(1, wav_data.shape[0])
            wav_data_torch = torch.from_numpy(wav_data).float()

            # Extract features
            with torch.no_grad():
                rep, layer_results = model.extract_features(wav_data_torch, output_layer=model.cfg.encoder_layers,
                                                            ret_layer_results=True)[0]

            seq_len_f.write(str(rep.shape[1]) + '\n')

            # Save to layers
            for layer in range(nlayers):
                layer_arrays[layer].append(layer_results[layer][0].transpose(0, 1).detach().numpy().squeeze())



    # Close NpyAppendArrays
    for arr in layer_arrays:
        arr.close()
        del arr
    del layer_arrays

    # Copy seqlens to other layer dirs
    for i in range(nlayers):
        if i != 0:
            copy2(seq_len_save_path, seq_len_save_path.replace('layer_0', f'layer_{i}'))

        # perform aggregation
        aggregate_sequence_embeddings(model_name=wavlm_version, layer=i, dataset_name=dataset_name)

        # Delete raw embeddings, if we're only storing the aggregated ones
        if only_store_aggregated:
            embeddings_path = os.path.join(embedding_dir, f'layer_{i}', 'all_0_1.npy')
            os.remove(embeddings_path)

    # Create .SUCCESS file to indicate that embeddings have been extracted
    with open(os.path.join(embedding_dir, '.SUCCESS'), 'a'):
        os.utime(os.path.join(embedding_dir, '.SUCCESS'), None)


if __name__ == "__main__":
    for wavlm_version, nlayers, only_store_aggregated in [
        ('wavlm_large', 25, False),
        ('wavlm_base', 13, False),
        ('wavlm_base_plus', 13, False),
    ]:

        datasets = [
            'TORGO',
            'human_synthesized',
            os.path.join('speech_accent_archive', 'male_female'),
            os.path.join('speech_accent_archive', 'us_korean'),
            os.path.join('speech_accent_archive', 'british_young_old'),
            os.path.join('speech_accent_archive', 'usa_young_old'),
            os.path.join('audio_iats','romero_rivas_et_al'),
            os.path.join('audio_iats','mitchell_et_al'),
            os.path.join('audio_iats','pantos_perkins'),
            'morgan_emotional_speech_set',
            'EU_Emotion_Stimulus_Set',
            'UASpeech',
            'coraal_buckeye_joined',
        ]
        for dataset_name in datasets:
            extract_wavlm_features(
                wavlm_version,
                dataset_name,
                'all',
                nlayers,
                only_store_aggregated
            )




