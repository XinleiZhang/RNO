import numpy as np
import matplotlib.pyplot as plt


def random_nut(num_funcs_, poly_amp_range_, poly_degree_, y_, nu_, Retau_,
               nut_base_):
    coeffs = np.random.uniform(-np.abs(poly_amp_range_),
                               np.abs(poly_amp_range_),
                               (num_funcs_, poly_degree_))
    coeffs[:, 1::2] = 0
    if poly_degree_ % 2 == 0:
        coeffs[:, -2] = -np.sum(coeffs[:, :poly_degree_ - 2], axis=1)
    else:
        coeffs[:, -1] = -np.sum(coeffs[:, :poly_degree_ - 1], axis=1)

    nut = []
    for i in np.arange(num_funcs_):
        nut.append(
            np.polynomial.Chebyshev(coeffs[i, :])(y_) *
            np.cos(np.pi / 2 * y_) * nu_ * Retau_ + nut_base_)
    nut = np.array(nut)
    return nut


def sample_frequency(nut_,
                     U_,
                     stf_psd_,
                     omega_,
                     dist_,
                     kx_,
                     kz_,
                     Ny_step=1,
                     Ny_start=0,
                     eps=5.e-3,
                     nut_scale=1,
                     U_scale=1,
                     stf_scale=1,
                     omega_scale=1,
                     y_scale=1,
                     k_scale=1):
    from rdp import rdp
    [func_space_size, _] = np.shape(nut_)
    freq_size = np.size(omega_)
    Ny = np.size(dist_)
    print(f'function space size {func_space_size}')
    print(f'frequency size {freq_size}')
    print(f'Ny {Ny}')

    # check input data shape
    if np.shape(nut_) != np.shape(U_):
        raise ValueError('nut_ and U_ should have the same shape')
    if np.shape(nut_)[1] != Ny:
        raise ValueError('nut_ axis 1 should besame size as Ny')
    if np.shape(stf_psd_)[0] != func_space_size:
        raise ValueError('stf_psd_ axis 0 should be size as func_space_size')
    if np.shape(stf_psd_)[1] != freq_size:
        raise ValueError('stf_psd_ axis 1 should be size as freq_size')
    if np.shape(stf_psd_)[2] != (Ny * 3):
        raise ValueError('stf_psd_ axis 2 should be size as 3*Ny')

    NyH = Ny // 2
    Ny0 = 0
    Ny1 = Ny
    Ny2 = Ny * 2
    Ny3 = Ny * 3
    id_sample_y = np.arange(Ny_start, NyH + Ny_step, Ny_step)
    print(id_sample_y, np.size(id_sample_y))

    stfx_psd = stf_psd_[:, :, Ny0:Ny1]
    stfy_psd = stf_psd_[:, :, Ny1:Ny2]
    stfz_psd = stf_psd_[:, :, Ny2:Ny3]
    stfx_psd = 0.5 * (stfx_psd + np.flip(stfx_psd, axis=2))
    stfy_psd = 0.5 * (stfy_psd + np.flip(stfy_psd, axis=2))
    stfz_psd = 0.5 * (stfz_psd + np.flip(stfz_psd, axis=2))

    # find the characteristic y position
    domega = omega_[1] - omega_[0]
    stfx_psd_sum = np.sum(stfx_psd, axis=1) * domega
    stfx_psd_nrm = stfx_psd / stfx_psd_sum[:, np.newaxis, :]
    stfx_psd_nrm_std = np.std(stfx_psd_nrm[0, :, :], axis=0)
    id_y_1 = np.argmin(stfx_psd_nrm_std)
    id_y_2 = np.argmax(stfx_psd_nrm_std)

    # determine the sampling frequency
    phi_1 = np.column_stack((omega_, stfx_psd_nrm[0, :, id_y_1]))
    phi_2 = np.column_stack((omega_, stfx_psd_nrm[0, :, id_y_2]))

    mask_1 = rdp(phi_1, epsilon=eps, algo='iter', return_mask=True)
    mask_2 = rdp(phi_2, epsilon=eps, algo='iter', return_mask=True)
    mask = mask_1 | mask_2
    indices = np.linspace(0, np.where(mask)[0][1], 5, dtype=int)[1:-1]
    mask[indices] = True
    indices = np.linspace(np.where(mask)[0][-2], freq_size - 1, 5,
                          dtype=int)[1:-1]
    mask[indices] = True

    id_sample_freq = np.where(mask)[0]
    print(f'number of freq sample {np.count_nonzero(mask)}')

    fig, ax = plt.subplots(1, 2)
    for i in np.arange(2):
        ax[i].plot(omega_, stfz_psd[0, :, 20] / stf_scale)
        ax[i].plot(omega_[id_sample_freq],
                   stfz_psd[0, id_sample_freq, 20] / stf_scale, '.')
    ax[1].set(xlim=(-2, 6))

    # collect data
    data_inputs = []
    data_target = []
    for i in np.arange(func_space_size):
        for j in id_sample_y:
            for t in id_sample_freq:
                trajectory = np.concatenate(
                    (U_[i, id_sample_y] / U_scale,
                     nut_[i, id_sample_y] / nut_scale,
                     np.array([
                         kx_ / k_scale, kz_ / k_scale, dist_[j] / y_scale,
                         omega_[t] / omega_scale
                     ])))
                data_inputs.append(trajectory)
                data_target.append([
                    stfx_psd[i, t, j] / stf_scale,
                    stfy_psd[i, t, j] / stf_scale,
                    stfz_psd[i, t, j] / stf_scale
                ])
    data_inputs = np.array(data_inputs)
    data_target = np.array(data_target)
    print(f'inputs data shape {np.shape(data_inputs)}')
    print(f'target data shape {np.shape(data_target)}')

    return data_inputs, data_target, id_sample_freq


def collect_data(nut_, U_, stf_psd_, omega_, yplus_, kx_, kz_):
    [func_space_size, _] = np.shape(nut_)
    freq_size = np.size(omega_)
    Ny = np.size(yplus_)
    print(f'function space size {func_space_size}')
    print(f'frequency size {freq_size}')
    print(f'Ny {Ny}')

    # check input data shape
    if np.shape(nut_) != np.shape(U_):
        raise ValueError('nut_ and U_ should have the same shape')
    if np.shape(nut_)[1] != Ny:
        raise ValueError('nut_ axis 1 should besame size as Ny')
    if np.shape(stf_psd_)[0] != func_space_size:
        raise ValueError('stf_psd_ axis 0 should be size as func_space_size')
    if np.shape(stf_psd_)[1] != freq_size:
        raise ValueError('stf_psd_ axis 1 should be size as freq_size')
    if np.shape(stf_psd_)[2] != (Ny * 3):
        raise ValueError('stf_psd_ axis 2 should be size as 3*Ny')

    NyH = Ny // 2 + 1
    Ny0 = 0
    Ny1 = Ny
    Ny2 = Ny * 2
    Ny3 = Ny * 3

    stfx_psd = stf_psd_[:, :, Ny0:Ny1]
    stfy_psd = stf_psd_[:, :, Ny1:Ny2]
    stfz_psd = stf_psd_[:, :, Ny2:Ny3]
    stfx_psd = 0.5 * (stfx_psd + np.flip(stfx_psd, axis=2))
    stfy_psd = 0.5 * (stfy_psd + np.flip(stfy_psd, axis=2))
    stfz_psd = 0.5 * (stfz_psd + np.flip(stfz_psd, axis=2))

    # collect data
    data_inputs = []
    data_target = []
    for i in np.arange(func_space_size):
        for t in np.arange(freq_size):
            for j in np.arange(Ny):
                trajectory = np.concatenate(
                    (U_[i, :NyH], nut_[i, :NyH],
                     np.array([kx_, kz_, yplus_[j], omega_[t]])))
                data_inputs.append(trajectory)
                data_target.append(
                    [stfx_psd[i, t, j], stfy_psd[i, t, j], stfz_psd[i, t, j]])
    data_inputs = np.array(data_inputs)
    data_target = np.array(data_target)
    print(f'inputs data shape {np.shape(data_inputs)}')
    print(f'target data shape {np.shape(data_target)}')

    return data_inputs, data_target


def make_input(nut_, U_, yplus_, kx_, kz_, omega_):
    # check size:
    # nut_.size = Ny
    # U_.size = Ny
    # yplus_.size = Ny
    # kx_.size= NKX or scalar
    # kz_.size= NKZ or scalar
    # omega_.size= NOMEGA or scalar

    if nut_.size != U_.size:
        raise ValueError(f"U_ and nut_ should have the same size")

    # Ensure kx_ is either a scalar or an array of size NKX
    if isinstance(kx_, np.ndarray):
        NKX = kx_.size
    elif np.isscalar(kx_):
        NKX = 1  # Scalar case
        kx_ = np.array([kx_])  # Convert to array for consistency
    else:
        raise ValueError("kx_ must be either a scalar or a 1D array")

    # Ensure kz_ is either a scalar or an array of size NKZ
    if isinstance(kz_, np.ndarray):
        NKZ = kz_.size
    elif np.isscalar(kz_):
        NKZ = 1  # Scalar case
        kz_ = np.array([kz_])  # Convert to array for consistency
    else:
        raise ValueError("kz_ must be either a scalar or a 1D array")

    # Ensure omega_ is either a scalar or an array of size NOMEGA
    if isinstance(omega_, np.ndarray):
        NOMEGA = omega_.size
    elif np.isscalar(omega_):
        NOMEGA = 1  # Scalar case
        omega_ = np.array([omega_])  # Convert to array for consistency
    else:
        raise ValueError("omega_ must be either a scalar or a 1D array")

    Ny = len(yplus_)
    NyH = Ny // 2 + 1
    N_nut = len(nut_)
    NH_nut = N_nut // 2 + 1
    inputs = []
    for i in range(NKX):
        for j in range(NKZ):
            for t in range(NOMEGA):
                for k in range(Ny):
                    trajectory = np.concatenate(
                        (U_[:NH_nut], nut_[:NH_nut],
                         np.array([kx_[i], kz_[j], yplus_[k], omega_[t]])))
                    inputs.append(trajectory)
    return np.array(inputs)
