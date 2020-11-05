import os
import math
import json
from abc import abstractmethod
from glob import glob

import numpy as np
import torch as th
import soundfile
from torch.utils.data import Dataset
from tqdm import tqdm
import pretty_midi

from constants import MIN_MIDI, MAX_MIDI, SAMPLE_RATE, HOP_SIZE


def allocate_batch(batch, device):
    for key in batch.keys():
        if key != 'path':
            batch[key] = batch[key].to(device)
    return batch


class PianoSampleDataset(Dataset):
    def __init__(self, path, groups=None, sample_length=16000*5, hop_size=HOP_SIZE, seed=1257, random_sample=True):
        self.path = path
        self.groups = groups if groups is not None else self.available_groups()
        assert all(group in self.available_groups() for group in self.groups)
        self.sample_length = sample_length // hop_size * hop_size if sample_length is not None else None
        self.random = np.random.RandomState(seed)
        self.random_sample = random_sample
        self.hop_size = hop_size

        self.file_list = dict()
        self.data = []

        print('Loading %d group%s of %s at %s' % (len(groups), 's'[:len(groups) - 1], self.__class__.__name__, path))
        for group in groups:
            self.file_list[group] = self.files(group)
            for input_files in tqdm(self.file_list[group], desc='Loading group %s' % group):
                self.data.append(self.load(*input_files))

    def __getitem__(self, index):
        data = self.data[index]

        audio = data['audio']
        audio_path = data['path']
        frames = (data['frame'] >= 1)
        onsets = (data['onset'] >= 1)

        frame_len = frames.shape[0]
        if self.sample_length is not None:
            n_steps = self.sample_length // self.hop_size

            if self.random_sample:
                step_begin = self.random.randint(frame_len - n_steps)
                step_end = step_begin + n_steps
            else:
                step_begin = 0
                step_end = n_steps

            begin = step_begin * self.hop_size
            end = begin + self.sample_length
            
            audio_seg = audio[begin:end]
            frame_seg = frames[step_begin:step_end]
            onset_seg = onsets[step_begin:step_end]
            
            result = dict(path=data['path'])
            result['audio'] = audio_seg.float().div_(32768.0)
            result['frame'] = frame_seg.float()
            result['onset'] = onset_seg.float()
        else:
            result = dict(path=data['path'])
            result['audio'] = audio.float().div_(32768.0)
            result['frame'] = frames.float()
            result['onset'] = onsets.float()
        return result

    def __len__(self):
        return len(self.data)

    @classmethod
    @abstractmethod
    def available_groups(cls):
        """return the names of all available groups"""
        raise NotImplementedError

    @abstractmethod
    def files(self, group):
        """return the list of input files (audio_filename, tsv_filename) for this group"""
        raise NotImplementedError

    def load(self, audio_path, midi_path):
        """
        load an audio track and the corresponding labels

        """
        audio, sr = soundfile.read(audio_path, dtype='int16')
        assert sr == SAMPLE_RATE
        fs = sr / self.hop_size

        audio = th.ShortTensor(audio)
        audio_length = len(audio)

        n_keys = MAX_MIDI - MIN_MIDI + 1
        mel_length = audio_length // self.hop_size + 1

        midi = pretty_midi.PrettyMIDI(midi_path)
        midi_length_sec = midi.get_end_time()
        frame_length = np.min((int(midi_length_sec * fs) , mel_length))

        audio = audio[:frame_length*self.hop_size]
        frame = midi.get_piano_roll(fs=fs)
        onset = np.zeros_like(frame)
        for inst in midi.instruments:
            for note in inst.notes:
                onset[note.pitch, int(note.start * fs)] = 1

        frame = th.from_numpy(frame[21:108 + 1].T)  # to shape (times x 88 pitch)
        onset = th.from_numpy(onset[21:108 + 1].T)
        data = dict(path=audio_path, audio=audio, frame=frame, onset=onset)
        return data


class MAESTRO_small(PianoSampleDataset):
    def __init__(self, path='data', groups=None, sequence_length=None, hop_size=512, seed=42, random_sample=True):
        super().__init__(path, groups if groups is not None else ['train'], sequence_length, hop_size, seed, random_sample)

    @classmethod
    def available_groups(cls):
        return ['train', 'validation', 'test', 'debug']

    def files(self, group):
        metadata = json.load(open(os.path.join(self.path, 'data.json')))

        if group == 'debug':
            files = sorted([(os.path.join(self.path, row['audio_filename'].replace('.wav', '.flac')),
                             os.path.join(self.path, row['midi_filename'])) for row in metadata if
                            row['split'] == 'train'])
            files = files[:10]
        else:
            files = sorted([(os.path.join(self.path, row['audio_filename'].replace('.wav', '.flac')),
                            os.path.join(self.path, row['midi_filename'])) for row in metadata if row['split'] == group])
            files = [(audio if os.path.exists(audio) else audio.replace('.flac', '.wav'), midi) for audio, midi in files]

        return files
