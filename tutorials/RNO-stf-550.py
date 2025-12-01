"""
This program predicts the stochastic forcing using trained DeepONet
The psd of stochastic forcing is also saved
This prediction is for Retau-550
The only difference from that of Retau 180 is the coarsed sampled inputs of branch net input
"""

import os
import sys
import numpy as np
import torch
import toml
import matplotlib.pyplot as plt

path = '/home/wuct/todo/RNO-main/'  # absolute path, in which there is PyChannelResolvent
sys.path.append(os.path.abspath(path))
from PyChannelResolvent import PyResolvent as PyR
from PyChannelResolvent import DeepResolvent as DP
from PyChannelResolvent import NeuralOperator as NO

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('GPU avail.' if torch.cuda.is_available() else 'CPU avail.')
model = NO.load_NeuralOperator(f'{path}/DeepONet/model.pth',
                               toml.load(f'{path}/DeepONet/NeuO-config.toml'),
                               device=device)
nut_, U_ = np.loadtxt(f'{path}/RAS/results-dns-Retau-550.csv',
                      delimiter=',',
                      skiprows=1,
                      usecols=[1, 2],
                      unpack=True)
wave_id_samples = [(1, 5), (3, 4), (4, 4), (4, 5), (4, 6), (5, 5), (5, 6),
                   (6, 6)]

for (kxid, kzid) in wave_id_samples:
    dat = PyR.DNSVEL(10150, 550, 257, 576, 576, 1024, 5.e-2, kxid, kzid, 0.5,
                     1.0, f'{path}/DNS/Retau-550')
    print('wavenumber', dat.kx, dat.kz)
    input = DP.make_input(nut_[:dat.Ny - 1:2] / (dat.nu * dat.Retau),
                          U_[:dat.Ny - 1:2], dat.dist, np.array([dat.kx]),
                          np.array([dat.kz]), dat.omega)
    output = NO.NeuralOperator_prediction(model, input, device=device)
    output[output < 0] = 0
    output = output.reshape(dat.Nt_window, dat.Ny, 3, order='C')
    output /= output.max()

    result_dir = f'./results/Retau-550'
    os.makedirs(result_dir, exist_ok=True)

    np.save(f'{result_dir}/stf-psd-normed-kx-{kxid:02d}-kz-{kzid:02d}', output)
