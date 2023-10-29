# Pre-trained Speech Processing Models Contain Human-Like Biases that Propagate to Speech Emotion Recognition
This repository contains code for the paper *Pre-trained Speech Processing Models Contain Human-Like Biases that 
Propagate to Speech Emotion Recognition*, which appears in Findings of EMNLP 2023. Please create an issue and tag me
(@isaacOnline) if you have any questions.

## Python Environments
The python packages necessary to run the majority of code in this repo are listed in `mac_env.yml` and `unix_env.yml`, 
which specify the environments we used for running experiments on either mac or ubuntu machines, respectively. When 
preprocessing data with propensity score matching, we used `psmpy`, and because of package conflicts, created a separate
environment (`psmpy_env.yml`) for exclusively that purpose.

## Input Data
Data used for this project comes from a variety of sources, most of which we are not able to re-distribute. We have 
included information about the files in our data directory (e.g. the names of specific clips we used). Links to the
datasets are below. 
* The data in `audio_iats/mitchell_et_al` comes from the paper *[Does Social Desirability Bias Favor Humans? 
Explicit–Implicit Evaluations Of Synthesized Speech Support A New HCI Model Of Impression 
Management](https://doi.org/10.1016/j.chb.2010.09.002)*
* The data in `audio_iats/pantos_perkins` comes from the paper *[Measuring Implicit and Explicit Attitudes Toward Foreign 
Accented Speech](https://doi.org/10.1177/0261927X12463005)*
* The data in `audio_iats/romero_rivas_et_al` comes from the paper *[Accentism on Trial: Categorization/Stereotyping and
Implicit Biases Predict Harsher Sentences for Foreign-Accented Defendants](https://doi.org/10.1177/0261927X211022785)*
* The data in `CORAAL` comes from the *[Corpus of Regional African American Language](https://oraal.uoregon.edu/coraal)*
We used all CORAAL components that were recorded after the year 2000 and available in October of 2022.
* The data in `EU_Emotion_Stimulus_Set` comes from *[The EU-Emotion Stimulus Set: A validation 
study](https://doi.org/10.3758/s13428-015-0601-4)*
* The data in `MESS` comes from the paper *[Categorical and Dimensional Ratings of Emotional Speech: Behavioral Findings 
From the Morgan Emotional Speech Set](https://doi.org/10.1044/2019_JSLHR-S-19-0144)*
* The data in `speech_accent_archive` can be downloaded using the file `downloading/download_saa.py`
* The data in `TORGO` comes from *[The TORGO database of acoustic and articulatory speech from speakers with 
dysarthria](https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html)*
* The data in `UASpeech` comes from  *[Dysarthric Speech Database for Universal Access 
Research](http://www.isle.illinois.edu/sst/data/UASpeech/)*
* The data in `buckeye` comes from *[The Buckeye Corpus](https://buckeyecorpus.osu.edu/)*

After acquiring these datasets and placing them in the `data` directory, you will need to run the scripts in the 
`preprocessing` directory. These scripts will clean the datasets and create necessary metadata that will be used for 
extracting embeddings later. The `preprocessing/process_buckeye.py` and `preprocessing/process_coraal.py` scripts need 
to be run before `preprocessing/match_buckeye_coraal.py`, but other than this the scripts do not need to be 
run in a particular order. Some of these scripts will need to be run using the environment you create with 
`psmpy_env.yml`.

If you would like to extract embeddings for a new dataset, you will need to create an `all.tsv` file, examples of which 
can be seen in the data directory. This file contains a header listing the directory where wav files for the dataset can 
be founded, followed by relative paths to wav files in the dataset from this directory. Each wav file will need to be 
accompanied by its sequence length. You can use the functions in `downloading_utils.py` to find this sequence length, 
as well as to ensure the audio clips have a uniform number of channels.

## Speech Models
We use models from the HuBERT, wav2vec 2.0, WavLM, and Whisper model families. To download the relevant HuBERT and WavLM
checkpoints, you may be able to use the file `downloading/download_model_ckpts.py` (depending on whether the links we 
used are still working). This file uses urls defined in `downloading/urls.py` 
which may need to be updated in the future. As of publication, the wav2vec 2.0 models we used are available [here](https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/README.md).
We use the `Wav2Vec 2.0 Base—No finetuning`, `Wav2Vec 2.0 Large—No finetuning`, and `Wav2Vec 2.0 Large (LV-60)—No 
finetuning` checkpoints. The Whisper models will be automatically downloaded when extracting embeddings.

## Extracting Embeddings
Scripts for extracting embeddings are available in the `embedding_extraction` directory (`extract_whisper.py`, 
`hubert.py`, `wav2vec2.py`, and `wavlm.py`). If you want to extract embeddings for a new dataset, you can add the dataset
to these files. Embedding extraction was generally the most time consuming part of running this project. When extracting
embeddings for Whisper, you'll need to make sure you're using the `extract-embeddings` branch of my Whisper fork.

## Carrying Out SpEATs and Other Experiments
Once embeddings have been extracted, you can run the scripts in `plots/eats` to carry out the embedding 
association tests. These will save the SpEAT *d*s and *p*-values to results to files in `plots/eats/test_results` (the 
result files from our experiments are currently stored there). A script used for creating some of the plots in the paper 
is available at `plots/eats/plot_all_results.py`. To estimate the standard error of the SpEAT *d*s, there are scripts in
`plots/standard_error`. The results from our standard error estimation is in `plots/standard_error/all_mean_mean_results.csv`.
To train downstream SER models, you can use the file `embedding_extraction/train_emotion_model.py`. Weights of the SER
models we trained are in `dimension_models/model_objects`. You can use them to predict valence in the input datasets using 
`embedding_extraction/predict_valence.py`. 