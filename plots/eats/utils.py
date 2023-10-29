import os
import numpy as np
import pandas as pd

from embedding_extraction.embedding_aggregation import aggregate_sequence_embeddings


def get_results_path(embedding_aggregation_method, se=False):
    embedding_aggregation_method = '_'.join(embedding_aggregation_method)
    if se:
        return os.path.join('plots','standard_error',f'all_{embedding_aggregation_method}_results.csv')
    else:
        return os.path.join('plots','eats','test_results', f'all_{embedding_aggregation_method}_results.csv')

def check_if_already_run(test, authors, names, n_target_embd, n_attribute_embd, target_dataset, attribute_dataset,
                     valence_model, speech_model, embedding_aggregation_method):

    if not os.path.exists(get_results_path(embedding_aggregation_method)):
        return False
    all_results = pd.read_csv(get_results_path(embedding_aggregation_method))
    relevant_results = all_results[
        (all_results['Test'] == test)
        &(all_results['Authors'] == authors)
        &(all_results['X'] == names[0])
        &(all_results['Y'] == names[1])
        &(all_results['A'] == names[2])
        &(all_results['B'] == names[3])
        &(all_results['Number of Target Embeddings per Group'] == n_target_embd)
        &(all_results['Number of Attribute Embeddings per Group'] == n_attribute_embd)
        &(all_results['Target Dataset'] == target_dataset)
        &(all_results['Attribute Dataset'] == attribute_dataset)
        &(all_results['Dataset Used for Embedding Extraction'] == valence_model)
        &(all_results['Speech Model'] == speech_model)
        ]
    return len(relevant_results) > 0

def write_eat_result(test, authors, names, n_target_embd, n_attribute_embd, target_dataset, attribute_dataset,
                     valence_model, speech_model, npermutations, speat_d, speat_p, iat_d, embedding_aggregation_method='mean'):
    if os.path.exists(get_results_path(embedding_aggregation_method)):
        prior_results = pd.read_csv(get_results_path(embedding_aggregation_method))
    else:
        prior_results = pd.DataFrame({})

    new_results = pd.DataFrame({
        'Test': [test],
        'Authors': authors,
        'X': names[0],
        'Y': names[1],
        'A': names[2],
        'B': names[3],
        'Number of Target Embeddings per Group': n_target_embd,
        'Number of Attribute Embeddings per Group': n_attribute_embd,
        'Target Dataset': target_dataset,
        'Attribute Dataset': attribute_dataset,
        'Dataset Used for Embedding Extraction': valence_model,
        'Speech Model': speech_model,
        'Number of Permutations': npermutations,
        'SpEAT d': speat_d,
        'SpEAT p': speat_p,
        'IAT d': iat_d
    })

    all_results = pd.concat([prior_results, new_results])
    all_results.to_csv(get_results_path(embedding_aggregation_method), index=False)



def check_if_se_already_run(test, authors, names, n_target_embd, n_attribute_embd, target_dataset, attribute_dataset,
                     valence_model, speech_model):

    if not os.path.exists(get_results_path(embedding_aggregation_method=('mean','mean'), se=True)):
        return False
    all_results = pd.read_csv(get_results_path(embedding_aggregation_method=('mean','mean'), se=True))
    relevant_results = all_results[
        (all_results['Test'] == test)
        &(all_results['Authors'] == authors)
        &(all_results['X'] == names[0])
        &(all_results['Y'] == names[1])
        &(all_results['A'] == names[2])
        &(all_results['B'] == names[3])
        &(all_results['Number of Target Embeddings per Group'] == n_target_embd)
        &(all_results['Number of Attribute Embeddings per Group'] == n_attribute_embd)
        &(all_results['Target Dataset'] == target_dataset)
        &(all_results['Attribute Dataset'] == attribute_dataset)
        &(all_results['Dataset Used for Embedding Extraction'] == valence_model)
        &(all_results['Speech Model'] == speech_model)
        ]
    return len(relevant_results) > 0


def write_eat_se_result(test, authors, names, n_target_embd, n_attribute_embd, target_dataset, attribute_dataset,
                        valence_model, speech_model, npermutations, stdev,var):
    if os.path.exists(get_results_path(embedding_aggregation_method=('mean','mean'), se=True)):
        prior_results = pd.read_csv(get_results_path(embedding_aggregation_method=('mean','mean'), se=True))
    else:
        prior_results = pd.DataFrame({})

    new_results = pd.DataFrame({
        'Test': [test],
        'Authors': authors,
        'X': names[0],
        'Y': names[1],
        'A': names[2],
        'B': names[3],
        'Number of Target Embeddings per Group': n_target_embd,
        'Number of Attribute Embeddings per Group': n_attribute_embd,
        'Target Dataset': target_dataset,
        'Attribute Dataset': attribute_dataset,
        'Dataset Used for Embedding Extraction': valence_model,
        'Speech Model': speech_model,
        'Number of Permutations': npermutations,
        'stdev': stdev,
        'var': var,
    })

    all_results = pd.concat([prior_results, new_results])
    all_results.to_csv(get_results_path(embedding_aggregation_method=('mean','mean'), se=True), index=False)


def load_embeddings(dataset, model_name, nlayers, aggregation_method):
    layer_agg = aggregation_method[0]
    seq_agg = aggregation_method[1]
    backup_dir = '/Volumes/Backup Plus/research/SpEAT/embeddings'

    # Make sure embeddings are aggregated, and that they exist either on backup drive or main drive
    if os.path.exists(os.path.join('embeddings', dataset, model_name, f'layer_0')):
        if os.path.exists(os.path.join('embeddings', dataset, model_name, f'layer_0', f'{seq_agg}.npy')):
            base_dir = ''
        else:
            try:
                for layer in range(nlayers):
                    aggregate_sequence_embeddings(model_name=model_name, layer=layer, dataset_name=dataset,
                                                  base_dir='')
                base_dir = ''
            except FileNotFoundError:
                if os.path.exists(os.path.join(backup_dir, 'embeddings', dataset, model_name, f'layer_0')):
                    if os.path.exists(os.path.join(backup_dir, 'embeddings', dataset, model_name, f'layer_0', f'{seq_agg}.npy')):
                        base_dir = backup_dir
                    else:
                        try:
                            for layer in range(nlayers):
                                aggregate_sequence_embeddings(model_name=model_name, layer=layer, dataset_name=dataset,
                                                              base_dir=backup_dir)
                            base_dir = backup_dir
                        except FileNotFoundError:
                            raise ValueError('Embeddings not found')

    elif os.path.exists(os.path.join(backup_dir, 'embeddings', dataset, model_name, f'layer_0')):
        if os.path.exists(os.path.join(backup_dir, 'embeddings', dataset, model_name, f'layer_0', f'{seq_agg}.npy')):
            base_dir = backup_dir
        else:
            try:
                for layer in range(nlayers):
                    aggregate_sequence_embeddings(model_name=model_name, layer=layer, dataset_name=dataset,
                                                  base_dir=backup_dir)
                base_dir = backup_dir
            except FileNotFoundError:
                raise ValueError('Embeddings not found')
    else:
        raise ValueError('Embeddings not found')

    # Load embeddings
    if layer_agg == 'mean':
        embeddings = np.stack(
            [np.load(
                os.path.join(base_dir, 'embeddings', dataset, model_name, f'layer_{i}', f'{seq_agg}.npy'))
                for i in
                range(nlayers)], axis=2).mean(axis=2)
    elif layer_agg == 'max':
        embeddings = np.stack(
            [np.load(
                os.path.join(base_dir, 'embeddings', dataset, model_name, f'layer_{i}', f'{seq_agg}.npy'))
                for i in
                range(nlayers)], axis=2).max(axis=2)
    elif layer_agg == 'min':
        embeddings = np.stack(
            [np.load(
                os.path.join(base_dir, 'embeddings', dataset, model_name, f'layer_{i}', f'{seq_agg}.npy'))
                for i in
                range(nlayers)], axis=2).min(axis=2)
    elif layer_agg in ['first', 'second', 'q1', 'q2', 'q3', 'penultimate', 'last']:
        if layer_agg == 'first':
            layer_num = 0
        elif layer_agg == 'second':
            layer_num = 1
        elif layer_agg == 'q1':
            layer_num = nlayers // 4
        elif layer_agg == 'q2':
            layer_num = nlayers // 2
        elif layer_agg == 'q3':
            layer_num = 3 * nlayers // 4
        elif layer_agg == 'penultimate':
            layer_num = nlayers - 2
        elif layer_agg == 'last':
            layer_num = nlayers - 1
        else:
            raise ValueError(f'Aggregation method {aggregation_method} not recognized.')
        embeddings = np.load(os.path.join(base_dir, 'embeddings', dataset, model_name, f'layer_{layer_num}', f'{seq_agg}.npy'))
    else:
        raise ValueError(f'Aggregation method {aggregation_method} not recognized.')

    return embeddings