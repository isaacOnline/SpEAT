import os
from time import time
import logging

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Subset
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, f1_score
from tqdm import tqdm

from embedding_extraction.dimension_prediction import EmbeddingDataset, collate_unequal_seqlens, DimensionPredictor
from utils import load_dataset_info

def train(dataloader, model, loss_fn, optimizer, remaining_num_batches):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = [X_i.to(device) for X_i in X], y.to(device)

        # Compute prediction error
        pred = model(X)

        optimizer.zero_grad()
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()

        if batch % 5 == 0:
            loss, current = loss.item(), batch * len(X)
            logger.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        remaining_num_batches -= 1
        if remaining_num_batches == 0:
            break
    return remaining_num_batches



def test_continuous(dataloader, model):
    model.eval()
    all_preds = []
    all_y = []
    with torch.no_grad():
        for X, y in dataloader:
            all_y.extend(y.cpu().detach().numpy()*dataloader.dataset.dataset.label_range + dataloader.dataset.dataset.label_min)
            X, y = [X_i.to(device) for X_i in X], y.to(device)
            pred = model(X)
            all_preds.extend(pred.cpu().detach().numpy()*dataloader.dataset.dataset.label_range + dataloader.dataset.dataset.label_min)
    mae = mean_absolute_error(all_y, all_preds)
    mse = mean_squared_error(all_y, all_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_y, all_preds)
    logger.info(f"Test Error: \nMSE: {mse:>8f}\nRMSE: {rmse:>8f}\n"
                f"MAE: {np.sqrt(mae):>8f}\nR^2: {r2}")
    return r2, mse, rmse, mae


def test_categorical(dataloader, model):
    model.eval()
    all_preds = []
    all_y = []
    with torch.no_grad():
        for X, y in dataloader:
            all_y.extend(y.cpu().detach().numpy()*dataloader.dataset.dataset.label_range + dataloader.dataset.dataset.label_min)
            X, y = [X_i.to(device) for X_i in X], y.to(device)
            pred = model(X)
            all_preds.extend(pred.cpu().detach().numpy()*dataloader.dataset.dataset.label_range + dataloader.dataset.dataset.label_min)
    all_y = [int(i[0] < 1) for i in all_y]
    all_preds = [int(i[0] < 1) for i in all_preds]

    acc = accuracy_score(all_y, all_preds)
    f1 = f1_score(all_y, all_preds)
    logger.info(f"Test Error: \nAccuracy: {acc:>8f}\nF1: {f1:>8f}")
    return acc, f1, None, None

def get_results_path():
    results_path = os.path.join('dimension_models', 'all_results.csv')
    return results_path

def load_all_results():
    if os.path.exists(get_results_path()):
        all_results = pd.read_csv(get_results_path())
    else:
        all_results = pd.DataFrame({})
    return all_results


def run_already_completed(model_name, dataset_name, optimizer_name,
                               learning_rate, sequence_aggregation, dimension_name,
                               total_num_batches):
    all_results = load_all_results()
    if len(all_results) == 0:
        return False
    relevant_results = all_results[
        (all_results['model_name'] == model_name)
        & (all_results['dataset_name'] == dataset_name)
        & (all_results['optimizer_name'] == optimizer_name)
        & (all_results['learning_rate'] == learning_rate)
        & (all_results['sequence_aggregation'] == sequence_aggregation)
        & (all_results['dimension_name'] == dimension_name)
        & (all_results['total_num_batches'] == total_num_batches)
    ]
    return len(relevant_results) == 1


def save_results(model_name, dataset_name, optimizer_name, learning_rate, sequence_aggregation, dimension_name,
                 total_num_batches, start_time, best_r2, best_mse, best_rmse, best_mae, best_weights, best_model_path,
                 final_model_path):
    all_results = load_all_results()
    new_result = pd.DataFrame({
        'model_name': [model_name],
        'dataset_name': dataset_name,
        'optimizer_name': optimizer_name,
        'learning_rate': learning_rate,
        'sequence_aggregation': sequence_aggregation,
        'dimension_name': dimension_name,
        'total_num_batches': total_num_batches,
        'start_time': start_time,
        'best_r2': best_r2,
        'best_mse': best_mse,
        'best_rmse': best_rmse,
        'best_mae': best_mae,
        'best_weights': [best_weights],
        'best_model_path': best_model_path,
        'final_model_path': final_model_path,
    })
    pd.concat([all_results, new_result]).to_csv(get_results_path(), index=False)


def test_train_split(dataset, dataset_name):
    torch.manual_seed(93935486)
    np.random.seed(93935486)
    test_dataset, train_dataset = random_split(dataset, [int(len(dataset) / 8), len(dataset) - int(len(dataset) / 8)])
    return test_dataset, train_dataset


if __name__ == '__main__':
    for learning_rate in [
        1e-4,
        1e-5,
        1e-3,
    ]:
        datasets_and_dimensions = [
            ('morgan_emotional_speech_set', ['valence',
                                             ]),
            ('EU_Emotion_Stimulus_Set', ['valence',
                                         ]),
        ]
        for dataset_name, dimensions in datasets_and_dimensions:
            models = [
                ('wav2vec2_base', 13),
                ('wav2vec2_large_ll60k', 25),
                ('wav2vec2_large_ls960', 25),

                ('wavlm_base_plus', 13),
                ('wavlm_base', 13),
                ('wavlm_large', 25),

                ('hubert_xtralarge_ll60k', 49),
                ('hubert_large_ll60k', 25),
                ('hubert_base_ls960', 13),

                ('whisper_large_encoder', 33),
                ('whisper_medium_en_encoder', 25),
                ('whisper_medium_encoder', 25),
                ('whisper_small_en_encoder', 13),
                ('whisper_small_encoder', 13),
                ('whisper_base_en_encoder', 7),
                ('whisper_base_encoder', 7),

                ('whisper_large_decoder', 33),
                ('whisper_medium_en_decoder', 25),
                ('whisper_medium_decoder', 25),
                ('whisper_small_en_decoder', 13),
                ('whisper_small_decoder', 13),
                ('whisper_base_en_decoder', 7),
                ('whisper_base_decoder', 7),
            ]
            for model_name, nlayers in models:
                optimizer_name = 'adam'
                TOTAL_NUM_BATCHES = 20000
                model_obj_dir = os.path.join(f'dimension_models', 'model_objects')
                os.makedirs(model_obj_dir, exist_ok=True)
                for sequence_aggregation in [
                    'mean',
                    # 'max',
                    # 'min'
                ]:
                    for dimension_name in dimensions:
                        if not run_already_completed(model_name, dataset_name, optimizer_name,
                                       learning_rate, sequence_aggregation, dimension_name,
                                       TOTAL_NUM_BATCHES):
                            print(f'{dataset_name}\n{learning_rate}\n{model_name}\n{dimension_name}')

                            start_time = int(time())

                            run_id = f'{dataset_name}_{dimension_name}_{model_name}_{start_time}'

                            logger = logging.getLogger(run_id)
                            logger.setLevel(logging.INFO)
                            model_log_path = os.path.join('dimension_models','logs',run_id + '.log')
                            os.makedirs(os.path.dirname(model_log_path), exist_ok=True)
                            fh = logging.FileHandler(model_log_path)
                            logger.addHandler(fh)

                            logger.info(f'Using learning rate: {learning_rate}')
                            logger.info(f'Using sequence aggregation method: {sequence_aggregation}')
                            logger.info(f'Using optimizer: {optimizer_name}')


                            dataset = EmbeddingDataset(dataset_name, dimension_name, model_name, nlayers, labels=True, lazy=True)


                            test_dataset, train_dataset = test_train_split(dataset, dataset_name)
                            n_input_embd = train_dataset[0][0].shape[-1]


                            # Create data loaders.
                            batch_size = 32
                            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_unequal_seqlens)
                            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_unequal_seqlens)


                            device = "cuda" if torch.cuda.is_available() else "cpu"
                            logger.info(f"Using {device} device")





                            model = DimensionPredictor(nlayers, n_input_embd,
                                                       sequence_aggregation_method=sequence_aggregation,
                                                       categorical=False).to(device)
                            logger.info(model)

                            loss_fn = nn.MSELoss()
                            test_fn = test_continuous
                            if optimizer_name == 'adam':
                                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                            elif optimizer_name == 'adamw':
                                optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
                            else:
                                raise ValueError

                            remaining_num_batches = TOTAL_NUM_BATCHES
                            prev_total_batches = TOTAL_NUM_BATCHES
                            best_e1 = -np.Inf
                            best_weights = None
                            with tqdm(total=TOTAL_NUM_BATCHES, smoothing=0) as pbar:
                                while remaining_num_batches > 0:
                                    remaining_num_batches = train(train_dataloader, model, loss_fn, optimizer, remaining_num_batches)
                                    e1, e2, e3, e4 = test_fn(test_dataloader, model)
                                    if e1 > best_e1:
                                        best_e1, best_e2, best_e3, best_e4 = e1, e2, e3, e4
                                        best_weights = model.softmax(model.layer_sum_weights).cpu().detach().numpy()
                                        best_model_path = os.path.join(model_obj_dir, f'{run_id}_BEST.pt')
                                        torch.save(model.state_dict(), best_model_path)
                                        logger.info(f'Saved PyTorch Model State to {best_model_path}')
                                    batches_completed = prev_total_batches - remaining_num_batches
                                    prev_total_batches = remaining_num_batches
                                    pbar.update(batches_completed)


                            logger.info("Done!")
                            logger.info(f'Best layer weights found were: {best_weights}')

                            final_model_path = os.path.join(model_obj_dir, f'{run_id}_FINAL.pt')
                            torch.save(model.state_dict(), final_model_path)
                            logger.info(f'Layer Weights: {model.softmax(model.layer_sum_weights)}')
                            logger.info(f'Layer Weights (Unnormalized): {model.layer_sum_weights}')
                            logger.info(f'Saved PyTorch Model State to {final_model_path}')

                            save_results(model_name, dataset_name, optimizer_name, learning_rate, sequence_aggregation,
                                         dimension_name, TOTAL_NUM_BATCHES, start_time, best_e1, best_e2, best_e3, best_e4,
                                         best_weights, best_model_path, final_model_path)
