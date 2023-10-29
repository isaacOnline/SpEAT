# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the fairseq directory.


# (File adapted from fairseq/examples/hubert/simple_kmeans/dump_hubert_feature.py)

import logging
import shutil

import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(PROJECT_ROOT)
import sys
sys.path.extend([PROJECT_ROOT])
from embedding_extraction.embedding_aggregation import aggregate_sequence_embeddings

sys.path.extend(['fairseq'])

from npy_append_array import NpyAppendArray
import torch
import torch.nn.functional as F
import tqdm

import fairseq
from examples.hubert.simple_kmeans.feature_utils import get_path_iterator
from fairseq.data.audio.audio_utils import get_features_or_waveform

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_wav2vec2_feature")


class Wav2vec2FeatureReader(object):
    def __init__(self, ckpt_path, max_chunk=1600000):
        (
            model,
            cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.model = model[0].eval()
        self.task = task
        self.max_chunk = max_chunk
        logger.info(f"TASK CONFIG:\n{self.task.cfg}")
        logger.info(f" max_chunk = {self.max_chunk}")

    def read_audio(self, path, ref_len=None):
        wav = get_features_or_waveform(path, need_waveform=True, use_sample_rate=self.task.cfg.sample_rate)
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            logging.warning(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav

    def get_feats(self, path, ref_len=None):
        x = self.read_audio(path, ref_len=ref_len)
        with torch.no_grad():
            x = torch.from_numpy(x).float()
            if self.task.cfg.normalize:
                x = F.layer_norm(x, x.shape)
            x = x.view(1, -1)

            n_transformer_layers = len(self.model.encoder.layers)
            all_feats = [[] for _ in range(n_transformer_layers + 1)]
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start: start + self.max_chunk]
                if type(self.model) == fairseq.models.wav2vec.wav2vec2.Wav2Vec2Model:
                    results = self.model.extract_features(
                        source=x_chunk,
                        padding_mask=None,
                        mask=False
                    )
                    all_feats[0].append(self.model.post_extract_proj(results['features']))
                    # Taking 'x' from the layer_results, as this is how feature extraction is done for HuBERT
                    for l in range(n_transformer_layers):
                        all_feats[l+1].append(results['layer_results'][l][0].transpose(0,1))
        return [torch.cat(l, 1).squeeze(0) for l in all_feats]

def dump_features(reader, generator, num, split, nshard, rank, feat_dir):
    iterator = generator()

    feat_paths = []
    leng_paths = []
    os.makedirs(feat_dir, exist_ok=True)
    for l in range(nlayers):
        layer_dir = os.path.join(f"{feat_dir}",f"layer_{l}/")
        os.makedirs(layer_dir, exist_ok=True)
        feat_paths.append(os.path.join(layer_dir, f"{split}_{rank}_{nshard}.npy"))
        leng_paths.append(f"{feat_dir}/layer_{l}/{split}_{rank}_{nshard}.len")

        if os.path.exists(feat_paths[-1]):
            os.remove(feat_paths[-1])

    feat_arrays = [NpyAppendArray(fp) for fp in feat_paths]
    with open(leng_paths[0], "w") as leng_f:
        for path, nsample in tqdm.tqdm(iterator, total=num, smoothing=0):
            feats = reader.get_feats(path, nsample)
            for l in range(nlayers):
                feat_arrays[l].append(feats[l].cpu().numpy())
            leng_f.write(f"{len(feats[0])}\n")
    for l in range(nlayers):
        feat_arrays[l].close()
        if l != 0:
            shutil.copy2(leng_paths[0], leng_paths[l])
    logger.info("finished successfully")

def main(tsv_dir, split, ckpt_path, nshard, rank, feat_dir, max_chunk):
    reader = Wav2vec2FeatureReader(ckpt_path, max_chunk=max_chunk)
    generator, num = get_path_iterator(os.path.join(tsv_dir, f'{split}.tsv'), nshard, rank)
    dump_features(reader, generator, num, split, nshard, rank, feat_dir)


if __name__ == "__main__":
    for wav2vec2_version, nlayers, only_store_aggregated in [
        ('wav2vec2_large_ll60k', 25, False),
        ('wav2vec2_large_ls960', 25, False),
        ('wav2vec2_base', 13, False),
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
                save_dir = os.path.join('embeddings', dataset_name, wav2vec2_version)
                if os.path.exists(save_dir):
                    if not os.path.exists(os.path.join(save_dir, '.SUCCESS')):
                        shutil.rmtree(save_dir)
                if not os.path.exists(save_dir):
                    print(f'Model Version: {wav2vec2_version}\nDataset: {dataset_name}\n(All layers)')
                    args = {
                        'tsv_dir': os.path.join('data', dataset_name),
                        'split': 'all',
                        'ckpt_path': os.path.join('models', f'{wav2vec2_version}.pt'),
                        'nshard': 1,
                        'rank': 0,
                        'feat_dir': save_dir,
                        'max_chunk': 1600000
                    }
                    main(**args)
                    for i in range(nlayers):
                        aggregate_sequence_embeddings(model_name=wav2vec2_version, layer=i, dataset_name=dataset_name)
                    if only_store_aggregated:
                        embeddings_path = os.path.join(save_dir,
                                                       'all_0_1.npy')
                        os.remove(embeddings_path)
                    # Create a .SUCCESS file to indicate that the embeddings have been extracted
                    with open(os.path.join(save_dir, '.SUCCESS'), 'a'):
                        os.utime(os.path.join(save_dir, '.SUCCESS'), None)
