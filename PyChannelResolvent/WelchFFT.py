def segment_arrays(signal_original, seg_size=512, overlap=0.75, axis=0):
    import numpy as np

    step = int(seg_size * (1 - overlap))
    segments = []

    # loop through the array and create segments
    if signal_original.ndim == 1:
        for i in range(0, signal_original.shape[0] - seg_size + 1, step):
            segments.append(signal_original[i:i + seg_size])
    else:
        for i in range(0, signal_original.shape[axis] - seg_size + 1, step):
            slicer = [slice(None)] * signal_original.ndim
            slicer[axis] = slice(i, i + seg_size)
            segments.append(signal_original[tuple(slicer)])

    return np.array(segments)


def window_Hann(N):
    import numpy as np
    x = np.array(range(N)) * np.pi / N
    return np.sin(x) * np.sin(x)


def window_Hann_integ(N):
    import numpy as np
    w = window_Hann(N)
    return np.sum(np.power(w, 2)) / N


def window_Tukey(N, alpha=0.5):
    import numpy as np
    n = np.arange(N)
    window = np.ones(N)

    if alpha > 0:
        first_condition = n < (alpha * (N - 1) / 2)
        window[first_condition] = 0.5 * (1 + np.cos(np.pi *
                                                    (2 * n[first_condition] /
                                                     (alpha * (N - 1)) - 1)))

    if alpha > 0:
        last_condition = n >= ((1 - alpha / 2) * (N - 1))
        window[last_condition] = 0.5 * (1 +
                                        np.cos(np.pi *
                                               (2 * n[last_condition] /
                                                (alpha *
                                                 (N - 1)) - 2 / alpha + 1)))

    return window


import numpy as np


def window_flat_top(M):
    """
    Generate a flat-top window of size M.
    
    Parameters:
    M (int): The size of the window.
    
    Returns:
    numpy.ndarray: Flat-top window of size M.
    """
    if M < 1:
        return np.array([])

    # Coefficients for the flat-top window
    a0 = 1.0
    a1 = 1.93
    a2 = 1.29
    a3 = 0.388
    a4 = 0.028

    n = np.arange(0, M)
    x = np.pi * n / (M - 1)
    w = a0 - a1 * np.cos(2 * x) + a2 * np.cos(4 * x) - a3 * np.cos(
        6 * x) + a4 * np.cos(8 * x)

    return w


def Welch_FFT(x, dt, window='HANN', seg_size=512, overlap=0.75, axis=0):
    from scipy.fft import fft, fftshift, fftfreq
    import numpy as np

    # determine window function
    if (window == 'HANN' or window == 'Hann'):
        window_func = window_Hann(seg_size)
        window_inte = window_Hann_integ(seg_size)
    elif (window == 'TUKEY' or window == 'Tukey'):
        window_func = window_Tukey(seg_size)
    elif (window == 'Flat-top'):
        window_func = window_flat_top(seg_size)
    else:
        print('no window function')
        window_func = np.ones(seg_size)
        window_inte = 1.0

    # piece signals into segments
    segments = segment_arrays(x, seg_size, overlap, axis)

    # applying window on each segments
    if x.ndim == 1:
        # for 1d signals, simply apply the window function
        segments = segments * window_func[np.newaxis, :]
    else:
        # for multi-dimensional arrays, apply windowing on the specified axis
        slicer = [np.newaxis] * x.ndim
        slicer[axis] = slice(None)
        window_func = window_func[tuple(slicer)]
        segments = segments * window_func

    fft_axis = axis + 1
    segments_f = fftshift(fft(segments, axis=fft_axis, norm='forward'),
                          axes=fft_axis)
    frequency = fftshift(fftfreq(seg_size, d=dt / (2 * np.pi)))
    return frequency, segments_f / np.sqrt(window_inte)


def derivative(signal, dt, axis=0):
    # This function calculates dy/dt
    # y is periodic, and uniformly recorded
    from scipy.fft import fft, ifft, fftfreq
    import numpy as np

    shape = np.shape(signal)
    N = shape[axis]

    signal_fft = fft(signal, norm='backward', axis=axis)
    # create frequency

    if signal_fft.ndim == 1:
        freq = 1j * 2 * np.pi * fftfreq(N, d=dt)
    else:
        slicer = [np.newaxis] * signal_fft.ndim
        slicer[axis] = slice(None)
        freq = (1j * 2 * np.pi * fftfreq(N, d=dt))[tuple(slicer)]

    derivative_fft = freq * signal_fft
    derivative_time = ifft(derivative_fft, norm='backward', axis=axis)

    return derivative_time.real


def derivative_2(signal, dt, axis=0):
    # This function calculates dy/dt
    # y is periodic, and uniformly recorded
    from scipy.fft import fft, ifft, fftfreq
    import numpy as np

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
