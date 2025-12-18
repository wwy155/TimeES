import numpy as np
from scipy.ndimage import gaussian_filter1d
def estimate_acf_single_signal(x, max_lag=None, smooth_sigma=2.0):
    """
    Estimate non-stationary ACF R(t, tau) from a single signal x.
    
    Args:
        x: (T,) real-valued signal
        max_lag: maximum |tau| to consider (default: T//4)
        smooth_sigma: Gaussian smoothing std in time direction (in samples)
    
    Returns:
        R: (T, 2*max_lag+1) array, R[t, lag_idx] = R(t, tau)
           where tau = -max_lag, ..., 0, ..., max_lag
    """
    T = len(x)
    if max_lag is None:
        max_lag = min(T // 4, 100)  # avoid too large
    
    # Initialize ACF array
    R = np.zeros((T, 2 * max_lag + 1))
    taus = np.arange(-max_lag, max_lag + 1)
    
    # Compute raw ACF: R_raw(t, tau) = x[t] * x[t+tau]
    for i, tau in enumerate(taus):
        if tau >= 0:
            valid_t = np.arange(0, T - tau)
            R[valid_t, i] = x[valid_t] * x[valid_t + tau]
        else:
            valid_t = np.arange(-tau, T)
            R[valid_t, i] = x[valid_t] * x[valid_t + tau]
    
    # Smooth in time direction to reduce noise (critical!)
    if smooth_sigma > 0:
        for i in range(R.shape[1]):
            R[:, i] = gaussian_filter1d(R[:, i], sigma=smooth_sigma, mode='nearest')
    
    return R, taus

def compute_pseudo_spectrum_from_acf(R, taus, dt=1.0):
    """
    Compute pseudo-spectrum S~(t, omega) = F_tau{ R(t, tau) } / (2*pi)
    following Benowitz et al. (2015), Eq. (29).
    
    Args:
        R: (T, 2*L+1) estimated ACF
        taus: (2*L+1,) array of lags [-L, ..., L]
        dt: sampling interval (default=1.0)
    
    Returns:
        S_tilde: (T, M) real-valued pseudo-spectrum (M = 2*L+1)
        omega: (M,) angular frequencies
    """
    T, M = R.shape
    dtau = dt  # assume tau sampled same as time
    
    # Shift so that zero-lag is at index 0 for FFT (standard DFT assumes tau=0 at start)
    R_shifted = np.fft.ifftshift(R, axes=1)  # moves zero-lag to first column
    
    # FFT over lag axis
    S_complex = np.fft.fft(R_shifted, axis=1)
    
    # Shift back so that zero-frequency is centered
    S_complex = np.fft.fftshift(S_complex, axes=1)
    
    # Scale by dtau / (2*pi) to approximate continuous FT
    S_tilde = (dtau / (2 * np.pi)) * S_complex.real  # should be real
    
    # Force non-negative (numerical errors may cause small negatives)
    S_tilde = np.maximum(S_tilde, 0.0)
    
    # Frequency vector (angular frequency)
    omega = np.fft.fftshift(np.fft.fftfreq(M, d=dtau)) * 2 * np.pi
    
    return S_tilde, omega

# ----------------------------
# Main function
# ----------------------------
def get_initial_spectrum_benowitz(x, max_lag=None, smooth_sigma=2.0, dt=1.0):
    """
    Given a single signal x, compute initial evolutionary spectrum S(t, omega)
    using Benowitz's pseudo-spectrum method.
    
    Returns:
        S_init: (T, M) initial spectrum
        omega: (M,) angular frequencies
        R_est: (T, M) estimated ACF (for debugging)
    """
    R_est, taus = estimate_acf_single_signal(x, max_lag=max_lag, smooth_sigma=smooth_sigma)
    S_init, omega = compute_pseudo_spectrum_from_acf(R_est, taus, dt=dt)
    return S_init, omega, R_est


def get_initial_spectrum_benowitz_targetM(x, max_lag=None, smooth_sigma=2.0, dt=1.0, M_target=None):
    """
    Compute initial evolutionary spectrum with optional target frequency resolution.

    Args:
        x: (T,) signal
        max_lag: max lag for ACF estimation
        smooth_sigma: smoothing in time
        dt: sampling interval
        M_target: desired number of frequency bins (e.g., 128, 256)

    Returns:
        S_init: (T, M_target) or (T, 2*max_lag+1)
        omega: (M_target,) or (2*max_lag+1,)
        R_est: (T, 2*max_lag+1) — note: R_est lag dim unchanged
    """
    R_est, taus = estimate_acf_single_signal(x, max_lag=max_lag, smooth_sigma=smooth_sigma)
    S_init, omega = compute_pseudo_spectrum_from_acf_targetM(R_est, taus, dt=dt, M_target=M_target)
    return S_init, omega, R_est

def compute_pseudo_spectrum_from_acf_targetM(R, taus, dt=1.0, M_target=None):
    """
    Compute pseudo-spectrum S~(t, omega) with optionally specified number of frequency bins.

    Args:
        R: (T, 2*L+1) estimated ACF
        taus: (2*L+1,) array of lags [-L, ..., L]
        dt: sampling interval (default=1.0)
        M_target: desired number of frequency bins (optional). 
                  If None, use M = len(taus).

    Returns:
        S_tilde: (T, M_target or M) real-valued pseudo-spectrum
        omega: (M_target or M,) angular frequencies
    """
    T, M_orig = R.shape
    dtau = dt

    if M_target is None:
        M_target = M_orig

    # Step 1: Shift R so zero-lag is at index 0 (for proper FFT alignment)
    R_shifted = np.fft.ifftshift(R, axes=1)  # shape [T, M_orig], zero-lag at column 0

    # Step 2: Zero-pad or truncate in lag dimension to length M_target
    if M_target >= M_orig:
        # Pad with zeros on both sides? But R_shifted has zero-lag at start.
        # We pad at the END (which corresponds to high |tau|), which is correct.
        R_padded = np.zeros((T, M_target), dtype=R.dtype)
        R_padded[:, :M_orig] = R_shifted
        # The rest remains zero → implicit assumption R(t, tau)=0 for |tau| > max_lag
    else:
        # Truncate (not recommended unless necessary)
        R_padded = R_shifted[:, :M_target]

    # Step 3: FFT over lag axis
    S_complex = np.fft.fft(R_padded, axis=1)

    # Step 4: Shift so that zero-frequency is centered
    S_complex = np.fft.fftshift(S_complex, axes=1)

    # Step 5: Scale by dtau / (2*pi)
    S_tilde = (dtau / (2 * np.pi)) * S_complex.real
    S_tilde = np.maximum(S_tilde, 0.0)  # enforce non-negative

    # Step 6: Frequency vector
    omega = np.fft.fftshift(np.fft.fftfreq(M_target, d=dtau)) * 2 * np.pi

    return S_tilde, omega
