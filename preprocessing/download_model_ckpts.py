import os
import re
import shutil
import warnings

import requests

from preprocessing.urls import hubert_model_ckpts, wavlm_model_ckpts


def download_single_model(model_name, model_url):
    """
    Download a model checkpoint from a url, and save to the models directory

    :param model_name: Name of model to be downloaded. Will be converted into a file name
    :param model_url: Url to download checkpoint from
    :return:
    """
    standardized_model_name = model_name.replace('+', '_plus')
    standardized_model_name = re.sub(r'[^a-z0-9]', '_', standardized_model_name.lower()).strip('_')
    model_save_path = f'models/{standardized_model_name}.pt'
    if os.path.exists(model_save_path):
        warnings.warn(f'A file already exists for model {model_name}. Skipping download. Please delete the file if you '
                      f'would like it to be redownloaded.')
        return
    r = requests.get(model_url, stream=True)
    r.raw.decode_content = True
    with open(model_save_path, 'wb') as f:
        shutil.copyfileobj(r.raw, f)


if __name__ == '__main__':
    for model_name, model_url in wavlm_model_ckpts.items():
        download_single_model(model_name, model_url)
    for model_name, model_url in hubert_model_ckpts.items():
        download_single_model(model_name, model_url)
