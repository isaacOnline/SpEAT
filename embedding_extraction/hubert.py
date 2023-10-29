# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the fairseq directory.


# (File adapted from fairseq/examples/hubert/simple_kmeans/dump_hubert_feature.py)

import logging
import shutil

import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.extend([PROJECT_ROOT])

from embedding_extraction.embedding_aggregation import aggregate_sequence_embeddings

sys.path.extend(['fairseq'])

import torch
import torch.nn.functional as F

import fairseq
from examples.hubert.simple_kmeans.feature_utils import get_path_iterator, dump_feature
from fairseq.data.audio.audio_utils import get_features_or_waveform

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_hubert_feature")


class HubertFeatureReader(object):
    def __init__(self, ckpt_path, layer, max_chunk=1600000):
        (
            model,
            cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.model = model[0].eval()
        self.task = task
        self.layer = layer
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

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start: start + self.max_chunk]
                if type(self.model) == fairseq.models.hubert.hubert.HubertModel:
                    feat_chunk, _ = self.model.extract_features(
                        source=x_chunk,
                        padding_mask=None,
                        mask=False,
                        output_layer=self.layer,
                    )
                feat.append(feat_chunk)
        return torch.cat(feat, 1).squeeze(0)


def main(tsv_dir, split, ckpt_path, layer, nshard, rank, feat_dir, max_chunk):
    reader = HubertFeatureReader(ckpt_path, layer, max_chunk)
    generator, num = get_path_iterator(os.path.join(tsv_dir, f'{split}.tsv'), nshard, rank)
    dump_feature(reader, generator, num, split, nshard, rank, feat_dir)


if __name__ == "__main__":
    for hubert_version, nlayers, only_store_aggregated in [
        ('hubert_xtralarge_ll60k', 49, False),
        ('hubert_large_ll60k', 25, False),
        ('hubert_base_ls960', 13, False),
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
            for i in range(nlayers):
                layer_dir = os.path.join('embeddings', dataset_name, hubert_version, f'layer_{i}')
                # Check if embeddings have already been extracted for this model
                if os.path.exists(layer_dir):
                    if not os.path.exists(os.path.join(layer_dir, '.SUCCESS')):
                        shutil.rmtree(layer_dir)
                if not os.path.exists(layer_dir):
                    print(f'Model Version: {hubert_version}\nDataset: {dataset_name}\nLayer: {i}')
                    args = {
                        'tsv_dir': os.path.join('data', dataset_name),
                        'split': 'all',
                        'ckpt_path': os.path.join('models', f'{hubert_version}.pt'),
                        'layer': i,
                        'nshard': 1,
                        'rank': 0,
                        'feat_dir': layer_dir,
                        'max_chunk': 1600000
                    }
                    main(**args)
                    aggregate_sequence_embeddings(model_name=hubert_version, layer=i, dataset_name=dataset_name)
                    if only_store_aggregated:
                        embeddings_path = os.path.join(layer_dir,
                                                       'all_0_1.npy')
                        os.remove(embeddings_path)

                    # Create a .SUCCESS file to indicate that the embeddings have been extracted
                    with open(os.path.join(layer_dir, '.SUCCESS'), 'a'):
                        os.utime(os.path.join(layer_dir, '.SUCCESS'), None)

