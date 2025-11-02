import numpy as np
import h5py
from scipy.fft import fftfreq, fftshift, fft, ifft
import PyChannelResolvent.WelchFFT as WF
import os
import gc
import time
import sys

# This class deals with channel flow in spectral space

# def Integrate_coeff(Ny):
#     coeff = np.zeros(Ny)
#     for i in range(Ny):
#         coeff[i] = (1 - np.power(-1, i)) / (1 - i * i)
#     return coeff


def derivative_2(signal, dt, axis=0):
    # This function calculates dy/dt
    # y is periodic, and uniformly recorded

    shape = np.shape(signal)
    N = shape[axis]

    signal_fft = fft(signal, norm='backward', axis=axis)
    # create frequency

    if signal_fft.ndim == 1:
        freq = np.power(1j * 2 * np.pi * fftfreq(N, d=dt), 2)
    else:
        slicer = [np.newaxis] * signal_fft.ndim
        slicer[axis] = slice(None)
        freq = np.power((1j * 2 * np.pi * fftfreq(N, d=dt))[tuple(slicer)], 2)

    derivative_fft = freq * signal_fft
    derivative_time = ifft(derivative_fft, norm='backward', axis=axis)

    return derivative_time.real


def grad_y_op(Ny):
    import numpy as np
    yc = np.cos(np.linspace(0, np.pi, Ny))
    # print(yc)
    Dy = np.zeros((Ny, Ny))
    for i in range(Ny):
        for j in range(Ny):
            if i != j:
                if (i % (Ny - 1) == 0) and (j % (Ny - 1) != 0):
                    c = 2.0
                elif (i % (Ny - 1) != 0) and (j % (Ny - 1) == 0):
                    c = 0.5
                else:
                    c = 1.0
                Dy[i, j] = c * (-1)**(i + j) / (yc[i] - yc[j])
            if (i == j and i != 0 and i != Ny - 1):
                Dy[i, j] = -yc[i] / (2.0 * (1.0 - yc[i]**2))
    Dy[0, 0] = (2.0 * (Ny - 1)**2 + 1.0) / 6.0
    Dy[Ny - 1, Ny - 1] = -Dy[0, 0]

    return Dy


def Integrate_basis(Ny):
    n = np.arange(Ny)
    n[1] = 3
    coeff = (1 + np.power(-1, n)) / (1 - n * n)
    return coeff


def Integrate(Ny, phi):
    IB = Integrate_basis(Ny)
    phi_cheby = np.dot(p2s_y_op(Ny), phi)
    integral_phi = np.dot(phi_cheby, IB).real
    return integral_phi


def Integrate_y(arr, axis=0):
    from scipy.integrate import simpson
    if axis >= arr.ndim or axis < -arr.ndim:
        raise ValueError(
            f'error: axis {axis} is out of bounds for array with {arr.ndim} dimensions'
        )
    N = np.shape(arr)[axis]
    y = np.cos(np.linspace(0, np.pi, N))
    arr_integral = -simpson(arr, y, axis=axis)
    return arr_integral


def Dy_spectral(Ny):
    # {Steven A. Orszag, 1971, Accurate solution of the Orr-Sommerfeld stability equation, J. Fluid. Mech.}
    import numpy as np
    # Ny is the total number of Chebyshev points
    Op = np.zeros((Ny, Ny))
    N = Ny - 1
    for n in range(N + 1):
        c = 1
        if (n == 0): c = 2
        for p in range(N + 1):
            if (p > n and (p + n - 1) % 2 == 0): Op[n, p] = 2 * p / c
    return Op


def grad_y_physical(Ny, var_p):
    var_cheby = np.dot(p2s_y_op(Ny), var_p)
    Dy = Dy_spectral(Ny)
    dvar_dy_cheby = Dy @ var_cheby
    dvar_dy = np.dot(s2p_y_op(Ny), dvar_dy_cheby)
    return dvar_dy


def grad_y2_physical(Ny, var_p):
    var_cheby = np.dot(p2s_y_op(Ny), var_p)
    Dy = Dy_spectral(Ny)
    ddvar_dy2_cheby = Dy @ (Dy @ var_cheby)
    ddvar_dy2 = np.dot(s2p_y_op(Ny), ddvar_dy2_cheby)
    return ddvar_dy2


def grad_y3_physical(Ny, var_p):
    var_cheby = np.dot(p2s_y_op(Ny), var_p)
    Dy = Dy_spectral(Ny)
    ddvar_dy3_cheby = Dy @ (Dy @ (Dy @ var_cheby))
    ddvar_dy3 = np.dot(s2p_y_op(Ny), ddvar_dy3_cheby)
    return ddvar_dy3


def grad_y4_physical(Ny, var_p):
    var_cheby = np.dot(p2s_y_op(Ny), var_p)
    Dy = Dy_spectral(Ny)
    ddvar_dy4_cheby = Dy @ (Dy @ (Dy @ (Dy @ var_cheby)))
    ddvar_dy4 = np.dot(s2p_y_op(Ny), ddvar_dy4_cheby)
    return ddvar_dy4


def s2p_y_op(Ny):
    import numpy as np
    Op = np.zeros((Ny, Ny))
    for i in range(Ny):
        for j in range(Ny):
            Op[i, j] = np.cos(np.pi * i * j / (Ny - 1))
    return Op


def p2s_y_op(Ny):
    import numpy as np
    Op = np.linalg.inv(s2p_y_op(Ny))
    return Op


def s2p(phi_):
    import numpy as np
    from scipy.fft import fft

    [Nz, Ny, NxH] = np.shape(phi_)
    Nx = NxH * 2
    NzH = int(Nz / 2)

    S2P_y = s2p_y_op(Ny)
    phi = np.zeros((Nz, Ny, Nx), dtype=np.complex64)

    phi[:, :, 0:NxH] = phi_

    # transform Chebyshev
    phi = np.einsum('ij,zjk->zik', S2P_y, phi)

    # FFT in z direction
    phi[NzH, :, :] = 0.0
    fft(phi, axis=0, overwrite_x=True, norm='backward')

    # FFT in x direction
    phi[:, :, Nx - 1:NxH:-1] = np.conjugate(phi[:, :, 1:NxH])
    fft(phi, axis=2, overwrite_x=True, norm='backward')

    phi = np.real(phi)  # finished transform
    return phi


def p2s(phi_):
    import numpy as np
    import scipy.fft
    [Nz, Ny, Nx] = np.shape(phi_)
    p2s_y = p2s_y_op(Ny)
    phi = np.zeros((Nz, Ny, Nx), dtype=np.complex128)

    phi.real = phi_

    # transform Chebyshev
    phi = np.einsum('ij,zjk->zik', p2s_y, phi)

    # FFT in x direction
    scipy.fft.ifft(phi, axis=2, overwrite_x=True, norm='backward')
    # FFT in z direction
    scipy.fft.ifft(phi, axis=0, overwrite_x=True, norm='backward')

    phi = phi[:, :, 0:int(Nx / 2)]
    return phi


def Cess_eddy_viscosity(y, Re_tau):
    # This function calculates the Cess eddy viscosity
    kappa = 0.426
    A = 25.4
    yplus = Re_tau * (1 - np.abs(y))
    h = kappa * Re_tau / 3 * (1 - y * y) * (1 + 2 * y * y) * (
        1 - np.exp(-yplus / A))
    return 0.5 * np.sqrt(1 + h * h) - 0.5


def Cess_eddy_viscosity_dy(y, Re_tau):
    # This function calculates the derivative of Cess eddy viscosity
    kappa = 0.426
    A = 25.4
    yplus = Re_tau * (1 - np.abs(y))
    nu_t = Cess_eddy_viscosity(y, Re_tau)
    a = 1 - y * y
    b = 1 + 2 * y * y
    c = 1 - np.exp(-yplus / A)
    h = kappa * Re_tau / 3 * a * b * c
    dh = -2 * y * b * c
    dh += a * 4 * y * c
    dh -= a * b * np.exp(-yplus / A) * Re_tau / A * np.sign(y)
    dh *= (kappa * Re_tau / 3)
    return h * dh / (2 * nu_t + 1) * 0.5


def Integral_scales(psd, freq):
    if np.shape(psd)[0] != np.size(freq):
        raise ValueError(
            "Dimension of psd along freq_axis must match the size of freq.")
    # dfreq = freq[1] - freq[0]
    Nfreq = np.size(freq)
    dt = np.pi / freq[-1]
    tau = fftshift(fftfreq(Nfreq, d=1 / (dt * Nfreq)))

    # calculate time correlation
    R_corr = np.abs(fftshift(fft(psd, norm='backward', axis=0), axes=0))
    R_corr /= np.take(R_corr, indices=Nfreq // 2, axis=0)

    # calculate integration time
    T = np.sum(R_corr, axis=0) * dt * 0.5

    return tau, R_corr, T


def Taylor_microscale_defi(psd, freq, freq_axis=0):
    if np.shape(psd)[freq_axis] != np.size(freq):
        raise ValueError(
            "Dimension of psd along freq_axis must match the size of freq.")
    Nfreq = np.size(freq)
    dt = np.pi / freq[-1]
    tau = fftshift(fftfreq(Nfreq, d=1 / (dt * Nfreq)))

    # calculate time correlation
    R_corr = np.abs(fftshift(fft(psd, norm='backward', axis=0), axes=0))
    R_corr /= np.take(R_corr, indices=Nfreq // 2, axis=0)

    # calculate second derive at tau=0
    dtau = tau[1] - tau[0]
    print(freq[Nfreq // 2])

    Taylor = np.zeros((np.shape(psd)[1], np.shape(psd)[2]))
    for j in range(np.shape(psd)[1]):
        for k in range(np.shape(psd)[2]):
            R_n2 = R_corr[Nfreq // 2 - 2, j, k]
            R_n1 = R_corr[Nfreq // 2 - 1, j, k]
            R0 = R_corr[Nfreq // 2, j, k]
            R_p1 = R_corr[Nfreq // 2 + 1, j, k]
            R_p2 = R_corr[Nfreq // 2 + 2, j, k]
            RD2 = (-1.0 / 12.0 * R_n2 + 4.0 / 3.0 * R_n1 - 2.5 * R0 +
                   4.0 / 3.0 * R_p1 - 1.0 / 12.0 * R_p2) / np.power(dtau, 2)
            Taylor[j, k] = np.sqrt(-2 / RD2)
    return Taylor


def Taylor_microscale(psd, freq, freq_axis=0):
    if np.shape(psd)[freq_axis] != np.size(freq):
        raise ValueError(
            "Dimension of psd along freq_axis must match the size of freq.")
    dfreq = freq[1] - freq[0]
    psd_sum = np.sum(psd, axis=freq_axis) * dfreq
    # normalize PSD along the specified frequency axis
    slice_freq = [
        np.newaxis if i != freq_axis else slice(None) for i in range(psd.ndim)
    ]
    slice_sum = [
        np.newaxis if i == freq_axis else slice(None) for i in range(psd.ndim)
    ]
    psd_normed = psd / psd_sum[tuple(slice_sum)]

    # compute first and second momentum
    freq_1m = np.sum(psd_normed * freq[tuple(slice_freq)],
                     axis=freq_axis) * dfreq
    freq_2m = np.sum(psd_normed * np.power(freq[tuple(slice_freq)], 2),
                     axis=freq_axis) * dfreq
    result = np.sqrt(2 / (freq_2m - np.power(freq_1m, 2)))
    return np.real(result)


def convection_velocity(psd, freq, kx_, freq_axis=0):
    if np.shape(psd)[freq_axis] != np.size(freq):
        raise ValueError(
            "Dimension of psd along freq_axis must match the size of freq.")
    dfreq = freq[1] - freq[0]
    psd_sum = np.sum(psd, axis=freq_axis) * dfreq

    # normalize PSD along the specified frequency axis
    slice_freq = [
        np.newaxis if i != freq_axis else slice(None) for i in range(psd.ndim)
    ]
    slice_sum = [
        np.newaxis if i == freq_axis else slice(None) for i in range(psd.ndim)
    ]
    psd_normed = psd / psd_sum[tuple(slice_sum)]

    # compute first momentum
    freq_1m = np.sum(psd_normed * freq[tuple(slice_freq)],
                     axis=freq_axis) * dfreq
    result = freq_1m / (-kx_)
    return np.real(result)


def YangZX_FFT_kz_shift(arr, axis):
    N = arr.shape[axis]
    mid_index = N // 2

    temp = np.empty_like(arr, dtype=arr.dtype)

    slices = [slice(None)] * arr.ndim

    slices[axis] = slice(mid_index + 1, N)
    temp[tuple(slices)] = np.flip(arr[tuple(slices)], axis=axis)

    slices[axis] = slice(0, mid_index + 1)
    temp[tuple(slices)] = np.flip(arr[tuple(slices)], axis=axis)

    np.copyto(arr, temp)


class CFR:

    def __init__(self,
                 Re,
                 Retau,
                 Nkx,
                 Nkz,
                 Ny,
                 Nomega,
                 dkx,
                 dkz,
                 domega,
                 caseDir=''):
        self.var_dict = {}
        self.Re = Re
        self.Retau = Retau
        self.Nkx = Nkx
        self.Nkz = Nkz
        self.Ny = Ny
        self.Nomega = Nomega
        self.dkx = dkx
        self.dkz = dkz
        self.domega = domega
        self.caseDir = caseDir

        self.kx = np.arange(Nkx) * dkx
        self.kz = np.arange(Nkz) * dkz
        self.omega = (np.arange(Nomega) - Nomega // 2) * domega
        self.y = np.cos(np.linspace(0, np.pi, Ny))

    def add_var(self, name_, value_):
        self.var_dict[name_] = value_

    def getvar(self, name_):
        return self.var_dict.get(name_, f'key {name_} not found')

    def yplus(self):
        return self.Retau * (1 - np.abs(self.y))

    def utau(self):
        return self.Retau / self.Re

    def load_tke_u_kxkz(self):
        data = np.loadtxt(
            os.path.join(self.caseDir, 'results-post', 'summation-uu.dat')).T
        data = np.nan_to_num(data, nan=0)
        self.add_var('uu_kxkz', data)

    def get_gamma(self, field_dns):
        self.load_tke_u_kxkz()

        # then we load results of DNS
        IB = Integrate_basis(field_dns.Ny)
        uu_cheby = np.einsum('ij,zjk->zik', p2s_y_op(field_dns.Ny),
                             field_dns.getvar('UU'))
        uu_kxkz_dns = np.sum(uu_cheby * IB[np.newaxis, :, np.newaxis], axis=1)
        # YangZX_FFT_kz_shift(uu_kxkz_dns, axis=0)

        # calculate coefficient gamma
        kx_s = np.argmin(np.abs(field_dns.plt_kx - self.kx[0]))
        kx_e = np.argmin(np.abs(field_dns.plt_kx - self.kx[-1]))
        kz_s = np.argmin(np.abs(field_dns.plt_kz - self.kz[0]))
        kz_e = np.argmin(np.abs(field_dns.plt_kz - self.kz[-1]))
        uu_kxkz_dns_subset = uu_kxkz_dns[kz_s:kz_e + 1, kx_s:kx_e + 1]
        uu_kxkz = self.getvar('uu_kxkz')
        gamma = np.sqrt(
            (np.sum(uu_kxkz_dns_subset) * field_dns.dkx * field_dns.dkz) /
            (np.sum(uu_kxkz) * self.dkx * self.dkz))
        self.add_var('gamma', gamma)

    def save_NET(self):
        NyH = self.Ny // 2
        KX, YP_x = np.meshgrid(self.kx[1:], self.yplus()[1:NyH])
        KZ, YP_z = np.meshgrid(self.kz[1:], self.yplus()[1:NyH])
        LX = 2 * np.pi / KX * self.Retau
        LZ = 2 * np.pi / KZ * self.Retau

        net_x = np.zeros((self.Ny // 2 - 1, self.Nkz - 1))
        net_z = np.zeros((self.Ny // 2 - 1, self.Nkx - 1))
        for i in range(self.Nkx - 1):
            for j in range(self.Nkz - 1):
                filename = os.path.join(
                    self.caseDir, 'results-tke',
                    f'kx-{self.kx[i+1]:6.2f}-kz-{self.kz[j+1]:6.2f}-budget.dat'
                )
                if os.path.exists(filename):
                    profile = np.loadtxt(filename, skiprows=1)
                net_x[:, i] += (profile[1:self.Ny // 2, 2] +
                                profile[1:self.Ny // 2, 3]) * self.kx[i]

        for i in range(self.Nkx - 1):
            for j in range(self.Nkz - 1):
                filename = os.path.join(
                    self.caseDir, 'results-tke',
                    f'kx-{self.kx[i+1]:6.2f}-kz-{self.kz[j+1]:6.2f}-budget.dat'
                )
                if os.path.exists(filename):
                    profile = np.loadtxt(filename, skiprows=1)
                net_z[:, j] += (profile[1:self.Ny // 2, 2] +
                                profile[1:self.Ny // 2, 3]) * self.kz[j]

        filePath = os.path.join(self.caseDir, 'NET_X.h5')
        print(filePath)
        if os.path.exists(filePath):
            os.remove(filePath)
        with h5py.File(filePath, 'w') as f:
            f.create_dataset('LX', data=LX)
            f.create_dataset('YP', data=YP_x)
            f.create_dataset('NET_X', data=net_x)

        filePath = os.path.join(self.caseDir, 'NET_Z.h5')
        # print(filePath)
        if os.path.exists(filePath):
            os.remove(filePath)
        with h5py.File(filePath, 'w') as f:
            f.create_dataset('LZ', data=LZ)
            f.create_dataset('YP', data=YP_z)
            f.create_dataset('NET_Z', data=net_z)

    def save_NET_new(self):
        Ny = self.Ny
        NyH = self.Ny // 2
        KX, YP_x = np.meshgrid(self.kx[1:], self.yplus()[1:NyH])
        KZ, YP_z = np.meshgrid(self.kz[1:], self.yplus()[1:NyH])
        LX = 2 * np.pi / KX * self.Retau
        LZ = 2 * np.pi / KZ * self.Retau

        # Initialize net arrays
        net_x = np.zeros((NyH - 1, self.Nkz - 1))
        net_z = np.zeros((NyH - 1, self.Nkx - 1))

        for i, kx_val in enumerate(self.kx[1:]):
            for j, kz_val in enumerate(self.kz[1:]):
                filename = os.path.join(
                    self.caseDir, 'results-tke',
                    f'kx-{kx_val:6.2f}-kz-{kz_val:6.2f}-budget.dat')

                if os.path.exists(filename):
                    profile = np.loadtxt(filename, skiprows=1)
                    data_slice = profile[:, 2] + profile[:, 3]
                    data_slice = 0.5 * (data_slice + np.flip(data_slice))

                    net_x[:, i] += data_slice[1:NyH] * kx_val
                    net_z[:, j] += data_slice[1:NyH] * kz_val

        # Write net_x and net_z to .h5 files
        for name, data, YP, L, file_suffix in [
            ('NET_X', net_x, YP_x, LX, 'NET_X.h5'),
            ('NET_Z', net_z, YP_z, LZ, 'NET_Z.h5')
        ]:
            filePath = os.path.join(self.caseDir, file_suffix)
            # print(filePath)
            if os.path.exists(filePath):
                os.remove(filePath)

            with h5py.File(filePath, 'w') as f:
                f.create_dataset('L' + name[-1], data=L)  # 'LX' or 'LZ'
                f.create_dataset('YP', data=YP)
                f.create_dataset(name, data=data)

    def load_NET(self):

        def load_file(var_name, filepath):
            if os.path.exists(filepath):
                data = h5py.File(filepath, 'r')
                # print(var_name, 'has', data.keys())
                self.add_var(var_name, data)
                # with h5py.File(file_path, 'r') as net_file:
                #     self.add_var(var_name, net_file)
                # print(net_file.keys())

        # Load both NET_X and NET_Z
        load_file('NET_X', os.path.join(self.caseDir, 'NET_X.h5'))
        load_file('NET_Z', os.path.join(self.caseDir, 'NET_Z.h5'))

    @classmethod
    def init_from_case(cls, caseDir):
        filename = os.path.join(caseDir, 'set-up.ini')
        with open(filename, 'r') as file:
            lines = file.readlines()

        # extract relevant parameters by manually parsing the file
        Re = None
        Retau = None
        Nkx, Nkz, Ny, Nt = None, None, None, None
        dkx, dkz, domega = None, None, None

        for i, line in enumerate(lines):
            if "Friction Reynolds number" in line:
                Retau = float(lines[i + 1].replace('d', ''))  # Ignore 'd'
            elif "Karman constant" in line:
                kappa = float(lines[i + 1].replace('d', ''))  # Ignore 'd'
            elif "Wall constants" in line:
                B_wall, B1_wall = map(lambda x: float(x.replace('d', '')),
                                      lines[i + 1].split())
            elif "Discretization size in x,z,y,t" in line:
                sizes = lines[i + 1].split()
                Nkx, Nkz, Ny, Nomega = map(int, sizes)
            elif "Resolution in wave and frequency" in line:
                resolutions = lines[i + 1].split()
                dkx, dkz, domega = map(lambda x: float(x.replace('d', '')),
                                       resolutions)
                domega = 2 * np.pi / domega

        Re = Retau * ((-1.0 + np.log(Retau)) / kappa + B_wall + B1_wall)

        return cls(Re, Retau, Nkx, Nkz, Ny, Nomega, dkx, dkz, domega, caseDir)


class CFS:

    def __init__(self,
                 Lx,
                 Lz,
                 Re,
                 Retau,
                 Nx,
                 Ny,
                 Nz,
                 dt_sample,
                 Nt_window,
                 caseDir=''):
        self.var_dict = {}
        self.Lx = Lx
        self.Lz = Lz
        self.Re = Re
        self.Retau = Retau
        self.Nx = Nx  # number of waves in x directions
        self.Ny = Ny  # number of waves in z direction
        self.Nz = Nz  # number of Chebyshev points in y direction (points of -1,1 are included)
        NzH = int(Nz / 2)
        NxH = int(Nx / 2)
        self.NxH = NxH
        self.NzH = NzH
        self.NyH = Ny // 2 + 1
        self.dkx = 2 * np.pi / Lx
        self.dkz = 2 * np.pi / Lz
        self.dt_sample = dt_sample
        self.Nt_window = Nt_window
        self.domega = 2 * np.pi / (Nt_window * dt_sample)
        self.caseDir = caseDir

        # wave number equipped on spectral space
        kx_ = np.linspace(0, -NxH + 1, NxH) * 2 * np.pi / Lx
        kz_ = np.roll(np.linspace(NzH - 1, -NzH, Nz),
                      -(NzH - 1)) * 2 * np.pi / Lz
        self.kx = np.copy(kx_)
        self.kz = np.copy(kz_)
        self.y = np.cos(np.linspace(0, np.pi, Ny))
        self.x = np.arange(Nx) * Lx / Nx
        self.z = np.arange(Nz) * Lz / Nz

        self.plt_kx = np.abs(self.kx)
        YangZX_FFT_kz_shift(kz_, axis=0)
        self.plt_kz = kz_
        self.omega = fftshift(fftfreq(Nt_window, d=dt_sample / (2 * np.pi)))

    def add_var(self, name_, value_):
        self.var_dict[name_] = value_

    def getvar(self, name_):
        return self.var_dict.get(name_, f'key {name_} not found')

    def get_shape(self):
        return [self.Nz, self.Ny, self.Nx]

    def yplus(self):
        return self.Retau * (1 - np.abs(self.y))

    def utau(self):
        return self.Retau / self.Re

    def add_VelocityInstant_frdir(self, path):
        with h5py.File(path, 'r') as data:
            # dims are in [Z,Y,X] order
            U_ = data['U'][:, :, ::2] + 1j * data['U'][:, :, 1::2]
            V_ = data['V'][:, :, ::2] + 1j * data['V'][:, :, 1::2]
            W_ = data['W'][:, :, ::2] + 1j * data['W'][:, :, 1::2]
            # P_ = data['P'][:, :, ::2] + 1j * data['P'][:, :, 1::2]
        self.add_var('U', U_)
        self.add_var('V', V_)
        self.add_var('W', W_)
        # self.add_var('P', P_)

    def add_Pressure_frdir(self, path):
        with h5py.File(path, 'r') as data:
            # dims are in [Z,Y,X] order
            P_TOTAL = data['P-TOTAL'][:, :, ::2] + 1j * data['P-TOTAL'][:, :,
                                                                        1::2]
            P_RAPID = data['P-RAPID'][:, :, ::2] + 1j * data['P-RAPID'][:, :,
                                                                        1::2]
            P_SLOWW = data['P-SLOWW'][:, :, ::2] + 1j * data['P-SLOWW'][:, :,
                                                                        1::2]
            P_STOKE = data['P-STOKE'][:, :, ::2] + 1j * data['P-STOKE'][:, :,
                                                                        1::2]
            # print(np.shape(P_))
        self.add_var('P-TOTAL', P_TOTAL)
        self.add_var('P-RAPID', P_RAPID)
        self.add_var('P-SLOWW', P_SLOWW)
        self.add_var('P-STOKE', P_STOKE)

    def add_Var_TimeXZ_frdir(self,
                             path,
                             load_var_name,
                             given_var_name,
                             shift_kz=False,
                             kz_axis=1):
        with h5py.File(path, 'r') as data:
            print(data.keys())
            U_ = data[load_var_name][:, :, ::
                                     2] + 1j * data[load_var_name][:, :, 1::2]
            # U_[:, 0, 0] = 0
            # U_[:, self.Nz // 2, 0] = 0
            if (shift_kz):
                YangZX_FFT_kz_shift(U_, axis=kz_axis)
        self.add_var(given_var_name, U_)

    def add_Force_TimeY_frdir(self, caseDir, kx_id, kz_id, ITB, ITE):
        data = np.load(
            os.path.join(caseDir, 'PROBE_FORCE',
                         f'F-{ITB}-{ITE}-kx-{kx_id}-kz-{kz_id}.npy'))
        assert data.shape[1] == self.Ny, 'Y points mismatch'
        data_transformed = np.einsum('ij,zjk->zik', s2p_y_op(self.Ny), data)

        FNL, FED = data_transformed[0:3072, :, :3], data_transformed[0:3072, :,
                                                                     3:6]
        FST = FNL - FED

        print('Force laoded', np.shape(FST))

        self.add_var(f'NLF-kx-{kx_id}-kz-{kz_id}', FNL)
        self.add_var(f'EDF-kx-{kx_id}-kz-{kz_id}', FED)
        self.add_var(f'STF-kx-{kx_id}-kz-{kz_id}', FST)

    def add_Velocity_TimeY_frdir(self, caseDir, kx_id, kz_id, ITB, ITE):
        data = np.load(
            os.path.join(caseDir, 'PROBE_VELOCITY',
                         f'VEL-{ITB}-{ITE}-kx-{kx_id}-kz-{kz_id}.npy'))
        assert data.shape[1] == self.Ny, 'Y points mismatch'
        data_transformed = np.einsum('ij,zjk->zik', s2p_y_op(self.Ny), data)

        print('Velocity laoded', np.shape(data_transformed))

        self.add_var(f'VEL-kx-{kx_id}-kz-{kz_id}', data_transformed)

    def add_Reynoldstress_frdir(self, path):
        with h5py.File(path, 'r') as data:
            # data is of type real
            uu_ = np.array(data['UU']) / (self.dkx * self.dkz)
            uv_ = np.array(data['UV']) / (self.dkx * self.dkz)
            uw_ = np.array(data['UW']) / (self.dkx * self.dkz)
            vv_ = np.array(data['VV']) / (self.dkx * self.dkz)
            vw_ = np.array(data['VW']) / (self.dkx * self.dkz)
            ww_ = np.array(data['WW']) / (self.dkx * self.dkz)
            uu_, vv_, ww_ = [
                0.5 * (var + np.flip(var, axis=1)) for var in [uu_, vv_, ww_]
            ]
            uv_ = 0.5 * (uv_ - np.flip(uv_))
            YangZX_FFT_kz_shift(uu_, axis=0)
            YangZX_FFT_kz_shift(uv_, axis=0)
            YangZX_FFT_kz_shift(uw_, axis=0)
            YangZX_FFT_kz_shift(vv_, axis=0)
            YangZX_FFT_kz_shift(vw_, axis=0)
            YangZX_FFT_kz_shift(ww_, axis=0)
            self.add_var('UU', uu_)
            self.add_var('UV', uv_)
            self.add_var('UW', uw_)
            self.add_var('VV', vv_)
            self.add_var('VW', vw_)
            self.add_var('WW', ww_)

    def add_Pressure_psd_frdir(self, path):
        with h5py.File(path, 'r') as data:
            # data is of type real
            pp_ = np.array(data['P-COV'])
            pp_ = 0.5 * (pp_ + np.flip(pp_, axis=1))
            YangZX_FFT_kz_shift(pp_, axis=0)
            self.add_var('PP', pp_)

    def add_CovForce_frdir(self, path):
        with h5py.File(path, 'r') as data:
            for name_ in [
                    'NLFXFX', 'NLFXFY', 'NLFXFZ', 'NLFYFY', 'NLFYFZ', 'NLFZFZ',
                    'EDFXFX', 'EDFXFY', 'EDFXFZ', 'EDFYFY', 'EDFYFZ', 'EDFZFZ',
                    'STFXFX', 'STFXFY', 'STFXFZ', 'STFYFY', 'STFYFZ', 'STFZFZ'
            ]:
                data_ = np.array(data[name_])
                data_[0, :, 0] = 0
                YangZX_FFT_kz_shift(data_, axis=0)
                if name_ in {'NLFXFY', 'EDFXFY', 'STFXFY'}:
                    data_ = 0.5 * (data_ - np.flip(data_, axis=1))
                else:
                    data_ = 0.5 * (data_ + np.flip(data_, axis=1))
                self.add_var(name_, data_)

    def add_NonlinearForceInstant_frdir(self, path):

        with h5py.File(path, 'r') as data:
            NLFX = data['NLFX'][:, :, ::2] + 1j * data['NLFX'][:, :, 1::2]
            NLFY = data['NLFY'][:, :, ::2] + 1j * data['NLFY'][:, :, 1::2]
            NLFZ = data['NLFZ'][:, :, ::2] + 1j * data['NLFZ'][:, :, 1::2]
            NLFX[0, :, 0] = 0
            NLFY[0, :, 0] = 0
            NLFZ[0, :, 0] = 0

        YangZX_FFT_kz_shift(NLFX, axis=0)
        YangZX_FFT_kz_shift(NLFY, axis=0)
        YangZX_FFT_kz_shift(NLFZ, axis=0)
        self.add_var('NLFX', NLFX)
        self.add_var('NLFY', NLFY)
        self.add_var('NLFZ', NLFZ)

    def add_EddyViscForceInstant_frdir(self, path):

        with h5py.File(path, 'r') as data:
            EDFX = data['EDFX'][:, :, ::2] + 1j * data['EDFX'][:, :, 1::2]
            EDFY = data['EDFY'][:, :, ::2] + 1j * data['EDFY'][:, :, 1::2]
            EDFZ = data['EDFZ'][:, :, ::2] + 1j * data['EDFZ'][:, :, 1::2]

        self.add_var('EDFX', EDFX)
        self.add_var('EDFY', EDFY)
        self.add_var('EDFZ', EDFZ)

    def add_TKEBudget_frdir(self, path):
        with h5py.File(path, 'r') as data:
            TKE_PRODUCT = np.array(data['TKE_PRODUCT'])
            TKE_VISDISS = np.array(data['TKE_VISDISS'])
            TKE_VISTRAN = np.array(data['TKE_VISTRAN'])
            TKE_PGDTRAN = np.array(data['TKE_PGDTRAN'])
            TKE_NLETRAN = np.array(data['TKE_NLETRAN'])

            TKE_PRODUCT = 0.5 * (TKE_PRODUCT + np.flip(TKE_PRODUCT, axis=1))
            TKE_VISDISS = 0.5 * (TKE_VISDISS + np.flip(TKE_VISDISS, axis=1))
            TKE_VISTRAN = 0.5 * (TKE_VISTRAN + np.flip(TKE_VISTRAN, axis=1))
            TKE_PGDTRAN = 0.5 * (TKE_PGDTRAN + np.flip(TKE_PGDTRAN, axis=1))
            TKE_NLETRAN = 0.5 * (TKE_NLETRAN + np.flip(TKE_NLETRAN, axis=1))
            YangZX_FFT_kz_shift(TKE_PRODUCT, axis=0)
            YangZX_FFT_kz_shift(TKE_VISDISS, axis=0)
            YangZX_FFT_kz_shift(TKE_VISTRAN, axis=0)
            YangZX_FFT_kz_shift(TKE_PGDTRAN, axis=0)
            YangZX_FFT_kz_shift(TKE_NLETRAN, axis=0)

            self.add_var('TKE_PRODUCT', TKE_PRODUCT)
            self.add_var('TKE_VISDISS', TKE_VISDISS)
            self.add_var('TKE_VISTRAN', TKE_VISTRAN)
            self.add_var('TKE_PGDTRAN', TKE_PGDTRAN)
            self.add_var('TKE_NLETRAN', TKE_NLETRAN)

    def add_Lambda_frdir(self, path):
        with h5py.File(path, 'r') as data:
            DVELDY = np.array(data['DVELDY'])
            SWEEPY = np.array(data['SWEEPY'])
            DVELDY = 0.5 * (DVELDY + np.flip(DVELDY, axis=1))
            SWEEPY = 0.5 * (SWEEPY + np.flip(SWEEPY, axis=1))
            YangZX_FFT_kz_shift(DVELDY, axis=0)
            YangZX_FFT_kz_shift(SWEEPY, axis=0)
            vv = self.getvar('vv')
            lambda2 = vv[np.newaxis, :, np.newaxis] * DVELDY / SWEEPY
            lambda2 = np.nan_to_num(lambda2, nan=0)
            self.add_var('lambda', np.sqrt(lambda2))

    def get_product(self, u_, v_):
        import scipy.fft
        [Nz, Ny, Nx] = self.get_shape()
        NzH = int(Nz / 2)
        NxH = int(Nx / 2)
        Mx = int(Nx / 2 * 3)
        Mz = int(Nz / 2 * 3)

        S2P_y = s2p_y_op(Ny)

        u_ = np.einsum('ij,zjk->zik', S2P_y, u_)
        v_ = np.einsum('ij,zjk->zik', S2P_y, v_)

        # set the Nyquist mode to zero
        u_[NzH, :, :] = 0
        v_[NzH, :, :] = 0

        # initialize larger arrays for dealiasing
        u_dea = np.zeros((Mz, Ny, Mx), dtype=np.complex128)
        v_dea = np.zeros((Mz, Ny, Mx), dtype=np.complex128)

        # copy data into the dealiasing arrays
        u_dea[0:NzH, :, 0:NxH] = u_[0:NzH, :, :]
        u_dea[Nz:, :, 0:NxH] = u_[NzH:, :, :]

        v_dea[0:NzH, :, 0:NxH] = v_[0:NzH, :, :]
        v_dea[Nz:, :, 0:NxH] = v_[NzH:, :, :]

        # perform FFT in z direction
        scipy.fft.fft(u_dea, axis=0, overwrite_x=True, norm='backward')
        scipy.fft.fft(v_dea, axis=0, overwrite_x=True, norm='backward')

        # set u_dea,v_dea in x direction and do FFT in x direction
        u_dea[:, :, Mx - 1:Nx:-1] = np.conjugate(u_dea[:, :, 1:NxH])
        v_dea[:, :, Mx - 1:Nx:-1] = np.conjugate(v_dea[:, :, 1:NxH])
        scipy.fft.fft(u_dea, axis=2, overwrite_x=True, norm='backward')
        scipy.fft.fft(v_dea, axis=2, overwrite_x=True, norm='backward')

        # multiple in physical space
        # print(u_dea[0, 5, :].real)
        uv_dea = u_dea * v_dea
        uv_dea.imag = 0

        # return to spectral space
        scipy.fft.ifft(uv_dea, axis=2, overwrite_x=True, norm='backward')
        scipy.fft.ifft(uv_dea, axis=0, overwrite_x=True, norm='backward')
        P2S_y = p2s_y_op(self.Ny)
        uv_dea = np.einsum('ij,zjk->zik', P2S_y, uv_dea)

        uv = np.zeros((Nz, Ny, NxH), dtype=np.complex128)
        uv[0:NzH, :, :] = uv_dea[0:NzH, :, 0:NxH]
        uv[NzH:, :, :] = uv_dea[Nz:, :, 0:NxH]
        return uv

    def get_nl_force(self):
        UU = self.get_product(self.get_U(), self.get_U())
        UV = self.get_product(self.get_U(), self.get_V())
        UW = self.get_product(self.get_U(), self.get_W())
        VV = self.get_product(self.get_V(), self.get_V())
        VW = self.get_product(self.get_V(), self.get_W())
        WW = self.get_product(self.get_W(), self.get_W())

        f1 = self.DX(UU) + self.DY(UV) + self.DZ(UW)
        f2 = self.DX(UV) + self.DY(VV) + self.DZ(VW)
        f3 = self.DX(UW) + self.DY(VW) + self.DZ(WW)

        return [f1, f2, f3]

    def DX(self, var_):
        return self.kx[np.newaxis, np.newaxis, :] * var_ * 1j

    def DY(self, var_):
        Dy_s = Dy_spectral(self.Ny)
        return np.einsum('ij,zjk->zik', Dy_s, var_)

    def DZ(self, var_):
        return self.kz[:, np.newaxis, np.newaxis] * var_ * 1j

    def div(self, f1_, f2_, f3_):
        return self.DX(f1_) + self.DY(f2_) + self.DZ(f3_)

    def delete_var(self, var_name_):
        # Ensure var_name_ is a list; if not, convert it to a list
        if not isinstance(var_name_, list):
            var_name_ = [var_name_]

        for name in var_name_:
            if name in self.var_dict:
                del self.var_dict[name]
                print(f'Variable {name} deleted')
            else:
                print(f'Variable {name} not found')

    def get_TimeFFT(self, var_name, time_axis):
        import PyChannelResolvent.WelchFFT as WF
        var = self.getvar(var_name)
        _, var_hat = WF.Welch_FFT(var,
                                  dt=self.dt_sample,
                                  window='Hann',
                                  seg_size=self.Nt_window,
                                  overlap=0.5,
                                  axis=time_axis)
        del (var)
        gc.collect()
        self.add_var(f'{var_name}-freqmode', var_hat)

    def get_WaveFreqSpectra(self, var_name, time_axis):
        import PyChannelResolvent.WelchFFT as WF
        var = self.getvar(var_name)
        _, var_hat = WF.Welch_FFT(var,
                                  dt=self.dt_sample,
                                  window='Hann',
                                  seg_size=self.Nt_window,
                                  overlap=0.5,
                                  axis=time_axis)

        del (var)
        gc.collect()
        # print(np.shape(var_hat))
        freq_spectra = np.mean(np.abs(var_hat * np.conjugate(var_hat)), axis=0)

        # print(np.shape(freq_spectra))
        self.add_var(f'{var_name}-freqspectra', freq_spectra)


class CFP:

    def __init__(self, Lx, Lz, Re, Nx, Ny, Nz):
        self.var_dict = {}
        self.Lx = Lx
        self.Lz = Lz
        self.Re = Re
        self.Nx = Nx  # number of cells in x direction
        self.Ny = Ny  # number of points in y direction
        self.Nz = Nz  # number of cells in z direction

        # mesh axes in physical space
        dx = Lx / Nx
        dz = Lz / Nz
        self.x = np.linspace(0, Lx - dx, Nx)
        self.z = np.linspace(0, Lz - dz, Nz)
        self.y = np.cos(np.linspace(0, np.pi, Ny))

        # 2d mesh grid in physical space
        [Z_y, X_y] = np.meshgrid(self.z, self.x, indexing='ij')
        [Z_x, Y_x] = np.meshgrid(self.z, self.y, indexing='ij')
        [Y_z, X_z] = np.meshgrid(self.y, self.x, indexing='ij')
        self.X_y = X_y
        self.X_z = X_z
        self.Y_x = Y_x
        self.Y_z = Y_z
        self.Z_x = Z_x
        self.Z_y = Z_y

    def add_var(self, name_, value_):
        self.var_dict[name_] = value_

    def getvar(self, name_):
        return self.var_dict.get(name_, f'key {name_} not found')

    def get_shape(self):
        return [self.Nz, self.Ny, self.Nx]

    def get_product(self, u_, v_):
        return u_ * v_

    def add_Vel_frdir(self, path):
        with h5py.File(path, 'r') as data:
            U_data = np.array(data['U'])
            V_data = np.array(data['V'])
            W_data = np.array(data['W'])
        self.add_var('U', U_data)
        self.add_var('V', V_data)
        self.add_var('W', W_data)


class VecSignal:

    def __init__(self, dkx, dkz, Ny, dt, Nt, seg_size=1024, overlap=0.5):
        self.signal_dict = {}
        self.dkx = dkx
        self.dkz = dkz
        self.Ny = Ny
        self.dt_sample = dt
        self.Nt = Nt
        self.seg_size = seg_size
        self.overlap = overlap
        self.y = np.cos(np.linspace(0, np.pi, self.Ny))
        self.S2P = s2p_y_op(self.Ny)
        self.freq_win = fftshift(fftfreq(self.seg_size, d=dt / (2 * np.pi)))
        self.time_lag = fftshift(
            fftfreq(self.seg_size, d=1 / (dt * self.seg_size)))
        self.dt_lag = np.abs(self.time_lag[1] - self.time_lag[0])

    def getvar(self, name_):
        return self.signal_dict.get(name_, f'key {name_} not found')

    def add_signal(self, name_, value_):
        self.signal_dict[name_] = value_

    def load_F_frdir(self, caseDir, kx_id, kz_id, ITB, ITE):
        data = np.load(
            os.path.join(caseDir, 'PROBE_FORCE',
                         f'F-{ITB}-{ITE}-kx-{kx_id}-kz-{kz_id}.npy'))
        assert data.shape[0] == self.Nt, 'signal length mismatch'
        assert data.shape[1] == self.Ny, 'Y points mismatch'
        data_transformed = np.einsum('ij,zjk->zik', self.S2P, data)

        FNL, FED = data_transformed[:, :, :3], data_transformed[:, :, 3:6]
        FST = FNL - FED

        self.add_signal(f'FNL_kx_{kx_id}_kz_{kz_id}', FNL)
        self.add_signal(f'FED_kx_{kx_id}_kz_{kz_id}', FED)
        self.add_signal(f'FST_kx_{kx_id}_kz_{kz_id}', FST)

        FNL_WINS, FED_WINS, FST_WINS = [
            WF.Welch_FFT(signal,
                         dt=self.dt_sample,
                         window='Hann',
                         seg_size=self.seg_size,
                         overlap=self.overlap,
                         axis=0)[1] for signal in (FNL, FED, FST)
        ]

        self.add_signal(f'FNL_WINS_kx_{kx_id}_kz_{kz_id}', FNL_WINS)
        self.add_signal(f'FED_WINS_kx_{kx_id}_kz_{kz_id}', FED_WINS)
        self.add_signal(f'FST_WINS_kx_{kx_id}_kz_{kz_id}', FST_WINS)

    def load_V_frdir(self, caseDir, kx_id, kz_id, ITB, ITE):
        data = np.load(
            os.path.join(caseDir, 'PROBE_VELOCITY',
                         f'VEL-{ITB}-{ITE}-kx-{kx_id}-kz-{kz_id}.npy'))
        assert data.shape[0] == self.Nt, 'signal length mismatch'
        assert data.shape[1] == self.Ny, 'Y points mismatch'
        data_transformed = np.einsum('ij,zjk->zik', self.S2P, data)

        self.add_signal(f'VEL_kx_{kx_id}_kz_{kz_id}', data_transformed)

        _, VEL_WINS = WF.Welch_FFT(data_transformed,
                                   dt=self.dt_sample,
                                   window='Hann',
                                   seg_size=self.seg_size,
                                   overlap=self.overlap,
                                   axis=0)

        self.add_signal(f'VEL_WINS_kx_{kx_id}_kz_{kz_id}', VEL_WINS)
