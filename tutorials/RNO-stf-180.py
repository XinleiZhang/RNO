"""
This program predicts the stochastic forcing using trained DeepONet
The psd of stochastic forcing is also saved
This prediction is for Retau-180
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
nut_, U_ = np.loadtxt(f'{path}/RAS/results-dns-Retau-180.csv',
                      delimiter=',',
                      skiprows=1,
                      usecols=[1, 2],
                      unpack=True)
wave_id_samples = [(2, 9), (2, 10), (2, 11), (6, 9), (7, 9), (8, 8), (8, 10),
                   (8, 12), (9, 9), (9, 11), (10, 10), (10, 12), (11, 11),
                   (12, 12)]

for (kxid, kzid) in wave_id_samples:
    dat = PyR.DNSVEL(2800, 180, 129, 384, 384, 2048, 5.e-2, kxid, kzid, 0.25,
                     0.5, f'{path}/DNS/Retau-180')
    print('wavenumber', dat.kx, dat.kz)
    input = DP.make_input(nut_ / (dat.nu * dat.Retau), U_, dat.dist,
                          np.array([dat.kx]), np.array([dat.kz]), dat.omega)
    output = NO.NeuralOperator_prediction(model, input, device=device)
    output[output < 0] = 0
    output = output.reshape(dat.Nt_window, dat.Ny, 3, order='C')
    output /= output.max()

    result_dir = f'./results/Retau-180'
    os.makedirs(result_dir, exist_ok=True)

    np.save(f'{result_dir}/stf-psd-normed-kx-{kxid:02d}-kz-{kzid:02d}', output)
