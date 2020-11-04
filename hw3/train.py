import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch as th
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Transciber
from dataset import MAESTRO_small
from evaluate import evaluate
from constants import HOP_SIZE

def cycle(iterable):
    while True:
        for item in iterable:
            yield item

def train(logdir, batch_size, iterations, sequence_length, learning_rate, weight_decay, cnn_unit, fc_unit, debug=False):
    if logdir is None:
        logdir = Path('runs') / ('exp_' + datetime.now().strftime('%y%m%d-%H%M%S'))
    Path(logdir).mkdir(parents=True, exist_ok=True)

    if sequence_length % HOP_SIZE != 0:
        adj_length = sequence_length // HOP_SIZE * HOP_SIZE
        print(f'sequence_length: {sequence_length} is not divide by {HOP_SIZE}.\n \
                adjusted into : {adj_length}')
        sequence_length = adj_length

    if debug:
        dataset = MAESTRO_small(groups=['debug'], sequence_length=sequence_length, hop_size=HOP_SIZE, random_sample=True)
        iterations = 100
    else:
        dataset = MAESTRO_small(groups=['train'], sequence_length=sequence_length, hop_size=HOP_SIZE, random_sample=True)
    loader = DataLoader(dataset, batch_size, shuffle=True)

    device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')


    model = Transciber(cnn_unit=cnn_unit, fc_unit=fc_unit)
    optimizer = th.optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.98)
    criterion = nn.BCEWithLogitsLoss()

    model = model.to(device)

    loop = tqdm(range(iterations))
    
    for step, batch in zip(loop, cycle(loader)):
        optimizer.zero_grad()

        frame_logit, onset_logit = model(batch['audio'].to(device))
        frame_loss = criterion(frame_logit, batch['frame'].to(device))
        onset_loss = criterion(onset_logit, batch['onset'].to(device))
        loss = onset_loss + frame_loss

        loss.mean().backward()

        for parameter in model.parameters():
            clip_grad_norm_([parameter], 3.0)

        optimizer.step()
        scheduler.step()
        loop.set_postfix_str("loss: {:.3e}".format(loss.mean()))

    th.save({'model_state_dict': model.state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'step' : step,
            'cnn_unit' : cnn_unit,
            'fc_unit' : fc_unit
            },
            Path(logdir) / f'model-{step}.pt')
    
    results = evaluate(model, device)

    with open(Path(logdir) / 'results.txt', 'w') as f:
      for key, values in results.items():
          if key.startswith('metric/'):
              _, category, name = key.split('/')
              metric_string = f'{category:>32} {name:26}: {np.mean(values):.3f} +- {np.std(values):.3f}'
              print(metric_string)
              f.write(metric_string + '\n')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', default=None, type=str)
    parser.add_argument('-v', '--sequence_length', default=102400, type=int)
    parser.add_argument('-lr', '--learning_rate', default=6e-4, type=float)
    parser.add_argument('-b', '--batch_size', default=16, type=int)
    parser.add_argument('-i', '--iterations', default=10000, type=int)
    parser.add_argument('-wd', '--weight_decay', default=0)
    parser.add_argument('-cnn', '--cnn_unit', default=48, type=int)
    parser.add_argument('-fc', '--fc_unit', default=256, type=int)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    train(**vars(parser.parse_args()))