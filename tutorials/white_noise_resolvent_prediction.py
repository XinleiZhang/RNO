import os
import sys
import numpy as np
import matplotlib.pyplot as plt

path = '/home/wuct/todo/RNO-main/'  # absolute path, in which there is PyChannelResolvent
sys.path.append(os.path.abspath(path))
from PyChannelResolvent import PyResolvent as PyR

if __name__ == '__main__':
    nut_dns, U_dns = np.loadtxt('../RAS/results-dns-Retau-180.csv',
                                delimiter=',',
                                skiprows=1,
                                usecols=(1, 2),
                                unpack=True)
    res = PyR.Resolvent(Re=2800, Retau=180, Ny=129)

    domega = 2 * np.pi / (2048 * 0.05)
    omega = np.arange(-1024, 1023) * domega
    kx = -2
    kz = -4
    vel_psd = np.zeros((len(omega), res.Ny, 3))
    for i in range(len(omega)):
        print(i, omega[i])
        R = res.operator_R(kx, kz, omega[i], U_dns, nut_dns)
        vel_psd[i, :, :] = np.real(np.einsum('ij,ji->i', R,
                                             R.conj().T)).reshape(res.Ny,
                                                                  3,
                                                                  order='f')

    fig, ax = plt.subplots(1, 3)
    X, Y = np.meshgrid(omega / np.abs(kx), res.y, indexing='ij')
    for i in range(3):
        ax[i].contourf(X, Y, vel_psd[:, :, i])
        ax[i].set(xlim=(-1, 2), ylim=(-1, 1))
    plt.savefig('./white-noise-resolvent-prediction.png')

