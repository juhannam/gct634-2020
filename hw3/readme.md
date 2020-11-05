# HW3 : Automatic Polyphonic Piano Transcription

Automatic music transcription (AMT) refers to an automated process that converts musical signals into a piano roll. Polyphonic piano transcription is a specific AMT task for piano music.  
Because of sequential aspect of piano transcription, recurrent neural network(RNN) module is commonly used for the task.   
In this homework, your task is re-implement two RNN-based transcription models. The goals of the homeworks are as follows:

* Experiencing the deep learning process with sequential data.
* Getting familiar with RNN architectures.

## Dataset [[download link]](https://drive.google.com/file/d/185czlGZGXdDu8lFCnpe5nLpyZtKuao_p/view?usp=sharing)
We will use subset of [MAESTRO](https://magenta.tensorflow.org/datasets/maestro) dataset for this homework. The dataset contains performances of classical pieces, played by junior pianists. The audiofile and corresponding midi files are given. The midi files are recorded through special piano that can capture note timings and speeds.
We will convert the midi files into pianoroll format, and we will train our network to estimate that pianoroll from the audio, in supervised way.
We arbitary selected 100 / 20 / 50 (train/valid/test) performances from original dataset for this homework.

Once you downloaded the dataset, unzip and move the dataset to the homework folder. *'data'* folder is expected to be located inside the *'hw3'* folder.

```
$ pwd
{YOUR_DIRECTORY}/gct-2020/hw3
$ unzip maestro_small.zip
...
$ ls data
2004  2006  2008  2009  2011  2013  2014  2015  2017  2018  data.json
```

We provide dataloader to process the dataset (dataset.py). It will segement the audio and midis into specified length (when *sequence_length* is given), or precess whole audio (when *sequence_length = None*), and convert midis into pianoroll format (frame roll and onset roll). Details are given in notebook (notebooks/dataset.ipynb).

## Training simple CNN based model

## Question 1: implement LSTM based model, consists of 3-layers of bi-directional LSTM.

## Question 2: implement CNN-RNN(CRNN) model
Work in progress

## Question 3: implement Onsets-and-Frames model, which have interconnection between onsets and frames.
Work in progress

## Deliverables
Work in progress

## Note

## Credit
Many of codes are borrowed from Onsets-and-Frames implementation of Jongwook Kim. Implemented by Taegyun Kwon @ MAClab