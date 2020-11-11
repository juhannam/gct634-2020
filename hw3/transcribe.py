from pathlib import Path
import soundfile
import subprocess
import tempfile
import argparse
import math

import numpy as np
import torch as th
import torch.nn.functional as F
import pretty_midi
from mir_eval.util import midi_to_hz, hz_to_midi

from model import Transciber
from evaluate import extract_notes, save_midi
from constants import HOP_SIZE, N_FFT, SAMPLE_RATE, MIN_MIDI


def load_audio(audiofile):
    try:
        audio, sr = soundfile.read(audiofile)
        if audio.shape[1] != 1 or sr != 16000:
            raise TypeError
    except:
        path_audio = Path(audiofile)
        filetype = path_audio.suffix
        assert filetype in ['.mp3', '.ogg', '.flac', '.wav', '.m4a', '.mp4'], filetype
        with tempfile.TemporaryDirectory() as tempdir:
            tempwav = Path(tempdir) / (path_audio.stem + '_temp' + '.flac')
            command = ['ffmpeg', '-i', audiofile, '-af', 'aformat=s16:16000', '-ac', '1', tempwav] 
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            audio, sr = soundfile.read(tempwav)
    return audio


def transcribe(audio, model, args, save_name, max_len):
    print(f'save_path: {save_name}')
    audio = audio[:max_len*SAMPLE_RATE]
    t_audio = th.tensor(audio).to(th.float).cuda()
    pad_len = math.ceil(len(t_audio) / HOP_SIZE) * HOP_SIZE - len(t_audio)
    t_audio = th.unsqueeze(F.pad(t_audio, (0, pad_len)), 0)

    frame_logit, onset_logit = model(t_audio)
    onset = th.sigmoid(onset_logit[0])
    frame = th.sigmoid(frame_logit[0])

    p_est, i_est = extract_notes(onset, frame)

    scaling = HOP_SIZE / SAMPLE_RATE

    i_est = (i_est * scaling).reshape(-1, 2)
    p_est = np.array([midi_to_hz(MIN_MIDI + pitch) for pitch in p_est])

    numpy_filename = Path(save_name).parent / (Path(save_name).stem + '.npz')
    np.savez(save_name, onset=onset.cpu().numpy(), frame=frame.cpu().numpy())


    midi_filename = Path(save_name).parent / (Path(save_name).stem + '.midi')
    save_midi(midi_filename, p_est, i_est, [64] * len(p_est))

    wav_filename = Path(save_name).parent / (Path(save_name).stem + '.wav')
    midi_file = pretty_midi.PrettyMIDI(str(midi_filename))
    synth_audio = midi_file.fluidsynth(fs=16000)
    soundfile.write(wav_filename, synth_audio, 16000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', type=str)
    parser.add_argument('audio_file', type=str)
    parser.add_argument('--max_len', default=30, type=int)
    parser.add_argument('--rep_type', default='base')
    parser.add_argument('--save_path', default=None)
    args = parser.parse_args()
    with th.no_grad():
        model_state_path = args.model_file
        ckp = th.load(model_state_path, map_location='cpu')
        model = Transciber(ckp['cnn_unit'], ckp['fc_unit'])

        model.load_state_dict(ckp['model_state_dict'])
        model.eval()
        model = model.cuda()

        audio = load_audio(args.audio_file)

        if args.save_path is None:
            save_path = Path(args.model_file).parent / (Path(args.audio_file).stem + '_transcribed')

        transcribe(audio, model, args, save_path, args.max_len)