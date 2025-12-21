import numpy as np
from scipy.signal import stft, get_window

def get_initial_amplitude_stft(
    x,
    fs=1.0,
    nperseg=None,
    noverlap=None,
    M_target=None,
    window='boxcar',
    return_stft=False
):
    """
    Estimate initial complex evolutionary amplitude A(t, omega) using STFT.
    
    Matches the synthesis model:
        x[n] ≈ (1/√M) Σ_k A(n, ω_k) e^{i ω_k n}
    
    Args:
        x: (T,) real-valued signal
        fs: sampling frequency (default=1.0 → dt=1)
        nperseg: STFT segment length (default: min(256, T))
        noverlap: overlap points (default: nperseg // 2)
        M_target: target number of frequency bins (optional)
        window: window type ('hann', 'boxcar', etc.)
        return_stft: if True, also return interpolated STFT on full time grid
    
    Returns:
        A_init: (T, M) complex array — initial A(t, ω)
        omega: (M,) angular frequencies in rad/sample, ordered from -π to π (or -fs/2 to fs/2)
        stft_full (optional): (T, M) interpolated STFT (same as A_init before √M scaling)
    """
    T = len(x)
    dt = 1.0 / fs

    if nperseg is None:
        nperseg = min(256, T)
    if noverlap is None:
        noverlap = nperseg // 8

    # --- Step 1: Compute STFT ---
    f, t_stft, Zxx = stft(
        x,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        padded=True,
        boundary='zeros',
        return_onesided=False,
        
    )
    # Zxx shape: (n_freq, n_time_frames)
    # f: [0, df, 2df, ..., fs/2, -fs/2+df, ..., -df]  ← scipy's convention
    # --- Step 1.5: Convert one-sided STFT to two-sided ---
    # N_full = nperseg  # full DFT length
    # N_pos = Zxx_one_sided.shape[0]  # = N_full//2 + 1

    # if N_full % 2 == 0:
    #     # Even
    #     Zxx_full = np.zeros((N_full, Zxx_one_sided.shape[1]), dtype=complex)
    #     Zxx_full[0] = Zxx_one_sided[0]          # DC
    #     Zxx_full[1:N_full//2] = Zxx_one_sided[1:-1]  # pos freqs
    #     Zxx_full[N_full//2] = Zxx_one_sided[-1]      # Nyquist
    #     Zxx_full[N_full//2+1:] = np.conj(Zxx_one_sided[1:-1][::-1])  # neg freqs
    # else:
    #     # Odd
    #     Zxx_full = np.zeros((N_full, Zxx_one_sided.shape[1]), dtype=complex)
    #     Zxx_full[0] = Zxx_one_sided[0]
    #     Zxx_full[1:(N_full+1)//2] = Zxx_one_sided[1:]
    #     Zxx_full[(N_full+1)//2:] = np.conj(Zxx_one_sided[1:][::-1])

    # Zxx = Zxx_full
    # M_orig = N_full

    # --- Step 2: Normalize by window energy ---
    if window == 'boxcar':
        win_norm = np.sqrt(nperseg)
    else:
        win_array = get_window(window, nperseg)
        win_norm = np.linalg.norm(win_array)

    Zxx = Zxx / win_norm  # Now Zxx approximates "local Fourier coefficients" scaled by 1/||w||_2

    # --- Step 3: Adjust frequency bins to M_target ---
    M_orig = Zxx.shape[0]
    if M_target is not None:
        if M_target > M_orig:
            # Zero-pad in frequency domain (centered)
            Zxx_padded = np.zeros((M_target, Zxx.shape[1]), dtype=Zxx.dtype)
            start = (M_target - M_orig) // 2
            Zxx_padded[start:start + M_orig, :] = Zxx
            Zxx = Zxx_padded
            M = M_target
        elif M_target < M_orig:
            # Center crop
            start = (M_orig - M_target) // 2
            Zxx = Zxx[start:start + M_target, :]
            M = M_target
        else:
            M = M_orig
    else:
        M = M_orig

    # --- Step 4: Reorder frequencies to [-π, ..., 0, ..., π) i.e., fftshift order ---
    # scipy.stft returns: [0, +ve freqs..., Nyquist?, -ve freqs...]
    # We want: negative freqs first, then positive → use fftshift
    
    Zxx = np.fft.fftshift(Zxx, axes=0)  # Now Zxx[0] = most negative freq, Zxx[M//2] ≈ 0
    A_interp = np.sqrt(M) * Zxx

    # Corresponding angular frequencies (rad/sample)
    # Note: np.fft.fftfreq(M, d=dt) gives [0, 1, ..., M/2-1, -M/2, ..., -1] * (2π/(M*dt)) ?
    # But we want consistent with fftshift ordering.
    freqs_hz = np.fft.fftshift(np.fft.fftfreq(M, d=dt))  # in Hz, centered at 0
    omega = 2 * np.pi * freqs_hz  # rad/sample
    # --- Step 5: Scale to match synthesis model x[n] = (1/√M) Σ A e^{iωn} ---
    # From derivation: A ≈ √M * (normalized STFT)
    A_complex = Zxx  # shape: (M, n_frames)

    # --- Step 6: Interpolate to full time grid T ---
    t_original = np.arange(T) * dt  # (T,)
    t_stft_sec = t_stft               # (n_frames,)

    A_interp = np.zeros((T, M), dtype=complex)
    for k in range(M):
        # Interpolate each frequency bin over time
        A_interp[:, k] = np.interp(
            t_original,
            t_stft_sec,
            A_complex[k, :],
            left=0.0,
            right=0.0
        )
    # Now A_interp is (T, M), with omega[k] corresponding to column k
    # and omega ordered from -π/dt to +π/dt (i.e., -π*fs to +π*fs if dt=1/fs)

    if return_stft:
        # Return the interpolated normalized STFT (before √M scaling)
        stft_interp = A_interp / np.sqrt(M)
        return A_interp, omega, stft_interp
    else:
        return A_interp, omega
import torch
import torch.nn.functional as F

def get_initial_amplitude_stft_torch_nopad(
    x,
    n_fft=None,
    hop_length=None,
    fs=1.0,
    window='boxcar',
    return_stft=False,
    device=None
):
    """
    Estimate initial complex evolutionary amplitude A(t, omega) using torch.stft.
    
    Matches synthesis model: x[n] ≈ (1/√M) Σ_k A(n, ω_k) e^{i ω_k n}
    
    Each STFT frame is held constant over its hop interval (zero-order hold in time).
    
    Args:
        x: (T,) real tensor, dtype=torch.float32 or float64
        n_fft: FFT size (default = min(256, T))
        hop_length: hop size (default = n_fft // 8)
        fs: sampling frequency (default=1.0 → dt=1)
        window: 'boxcar', 'hann', 'hamming'
        return_stft: if True, also return unscaled STFT
        device: torch device (optional)

    Returns:
        A_interp: (T, M) complex tensor — initial A(t, ω), M = n_fft
        omega: (M,) angular frequencies in rad/sample, ordered from -π to π
        stft_interp (optional): same as A_interp before √M scaling
    """
    assert x.ndim == 1, "Input x must be 1D"
    T = x.shape[0]
    dt = 1.0 / fs
    dtype = x.dtype
    if device is None:
        device = x.device

    if n_fft is None:
        n_fft = min(256, T)
    if hop_length is None:
        hop_length = n_fft // 8

    # --- Create window ---
    if window == 'boxcar':
        win = torch.ones(n_fft, dtype=dtype, device=device)
    elif window == 'hann':
        win = torch.hann_window(n_fft, periodic=True, dtype=dtype, device=device)
    elif window == 'hamming':
        win = torch.hamming_window(n_fft, periodic=True, dtype=dtype, device=device)
    else:
        raise ValueError(f"Unsupported window: {window}")

    # win_norm = torch.norm(win).item()

    # --- Compute two-sided STFT (no center padding for clean time alignment) ---
    Zxx = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=win,
        center=False,
        normalized=True,
        onesided=False,
        return_complex=True
    )  # shape: (n_fft, n_frames)

    # Normalize by window energy
    # Zxx = Zxx / win_norm

    # M is fixed by n_fft
    M = n_fft

    # Reorder frequencies to [-π, ..., π) via fftshift
    Zxx = torch.fft.fftshift(Zxx, dim=0)  # (M, n_frames)

    # Scale to match synthesis model: x[n] ≈ (1/√M) Σ A e^{iωn}
    scale = torch.sqrt(torch.tensor(M, dtype=dtype, device=device))
    # A_complex = scale * Zxx  # (M, n_frames)
    A_complex =  Zxx  # (M, n_frames)

    # --- Zero-order hold in time (no interpolation) ---
    n_frames = A_complex.shape[1]
    A_interp = torch.zeros(T, M, dtype=torch.complex64 if dtype == torch.float32 else torch.complex128, device=device)

    for j in range(n_frames):
        start_t = j * hop_length
        end_t = min(start_t + hop_length, T)
        if start_t >= T:
            break
        A_interp[start_t:end_t, :] = A_complex[:, j].unsqueeze(0)

    # --- Angular frequencies (rad/sample) ---
    freqs_hz = torch.fft.fftshift(torch.fft.fftfreq(M, d=dt)).to(device=device, dtype=dtype)
    omega = 2 * torch.pi * freqs_hz  # (M,)

    if return_stft:
        stft_interp = A_interp / scale
        return A_interp, omega, stft_interp
    else:
        return A_interp, omega




def get_initial_amplitude_stft_torch(
    x,
    n_fft=None,
    hop_length=None,
    fs=1.0,
    window='boxcar',
    return_stft=False,
    device=None
):
    """
    Estimate initial complex evolutionary amplitude A(t, omega) using torch.stft with center=False.
    
    We manually pad the signal on the right so that all original time points [0, T)
    are covered by at least one STFT frame.

    Synthesis model assumed: x[n] ≈ (1/√M) Σ_k A(n, ω_k) e^{i ω_k n}
    BUT in this version, we DO NOT apply √M scaling — A = normalized STFT directly,
    matching your preference.

    Args:
        x: (T,) real tensor, dtype=torch.float32 or float64
        n_fft: FFT size (default = min(256, T))
        hop_length: hop size (default = n_fft // 8)
        fs: sampling frequency (default=1.0 → dt=1)
        window: 'boxcar', 'hann', 'hamming'
        return_stft: if True, also return unscaled STFT (same as A here)
        device: torch device (optional)

    Returns:
        A_interp: (T, M) complex tensor — initial A(t, ω), M = n_fft
        omega: (M,) angular frequencies in rad/sample, ordered from -π to π
        stft_interp (optional): same as A_interp
    """
    assert x.ndim == 1, "Input x must be 1D"
    T = x.shape[0]
    dt = 1.0 / fs
    dtype = x.dtype
    if device is None:
        device = x.device

    if n_fft is None:
        n_fft = min(256, T)
    if hop_length is None:
        hop_length = n_fft // 8

    # --- Handle padding for center=False ---
    if T < n_fft:
        pad_right = n_fft - T
        x_pad = F.pad(x, (0, pad_right))
        T_pad = n_fft
    else:
        # Compute how much to pad so that (T_pad - n_fft) % hop_length == 0
        remainder = (T - n_fft) % hop_length
        if remainder == 0:
            x_pad = x
            T_pad = T
        else:
            pad_right = hop_length - remainder
            x_pad = F.pad(x, (0, pad_right))
            T_pad = T + pad_right

    # --- Create window ---
    if window == 'boxcar':
        win = torch.ones(n_fft, dtype=dtype, device=device)
    elif window == 'hann':
        win = torch.hann_window(n_fft, periodic=True, dtype=dtype, device=device)
    elif window == 'hamming':
        win = torch.hamming_window(n_fft, periodic=True, dtype=dtype, device=device)
    else:
        raise ValueError(f"Unsupported window: {window}")

    # --- STFT with center=False ---
    Zxx = torch.stft(
        x_pad,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=win,
        center=False,          # ← no implicit padding
        normalized=True,      # we normalize manually
        onesided=False,
        return_complex=True
    )  # shape: (n_fft, n_frames)

    M = n_fft

    # Reorder frequencies to [-π, ..., π)
    Zxx = torch.fft.fftshift(Zxx, dim=0)  # (M, n_frames)

    # No √M scaling — as per your design
    A_complex = Zxx  # (M, n_frames)

    # --- Zero-order hold: time t belongs to frame j = t // hop_length ---
    n_frames = A_complex.shape[1]

    # For t in [0, T), frame index is j = t // hop_length
    # Since we padded to ensure coverage, j will always be < n_frames
    t_indices = torch.arange(T, device=device)
    frame_indices = t_indices // hop_length

    # Safety clamp (should not be needed, but robust)
    frame_indices = torch.clamp(frame_indices, max=n_frames - 1)

    # Assign: A_interp[t, :] = A_complex[:, frame_indices[t]]
    A_interp = A_complex[:, frame_indices].t()  # (T, M)

    # --- Angular frequencies ---
    freqs_hz = torch.fft.fftshift(torch.fft.fftfreq(M, d=dt)).to(device=device, dtype=dtype)
    omega = 2 * torch.pi * freqs_hz  # (M,)

    if return_stft:
        return A_interp, omega, A_interp
    else:
        return A_interp, omega

def get_initial_amplitude_right_stft_torch(
    x,
    n_fft=None,
    hop_length=None,
    fs=1.0,
    window='boxcar',
    return_stft=False,
    device=None,
    phase_remodulate=True,
):
    assert x.ndim == 1, "Input x must be 1D"
    T = x.shape[0]
    dt = 1.0 / fs
    dtype = x.dtype
    if device is None:
        device = x.device

    if n_fft is None:
        n_fft = min(256, T)
    if hop_length is None:
        hop_length = n_fft // 8

    # Left pad so t=0 can be the last sample of first frame
    pad_left = n_fft - 1
    x_left_padded = F.pad(x, (pad_left, 0))
    T_orig = x.shape[0]

    # Compute the maximum frame index we will need
    max_t = T_orig - 1
    max_frame_idx = (max_t + pad_left) // hop_length   # this is the largest j we will index

    # Minimum signal length required to compute frame `max_frame_idx`
    min_len_required = max_frame_idx * hop_length + n_fft

    # Right-pad if necessary
    current_len = x_left_padded.shape[0]
    if current_len < min_len_required:
        pad_right = min_len_required - current_len
        x_pad = F.pad(x_left_padded, (0, pad_right))
    else:
        x_pad = x_left_padded


    # --- Create window ---
    if window == 'boxcar':
        win = torch.ones(n_fft, dtype=dtype, device=device)
    elif window == 'hann':
        win = torch.hann_window(n_fft, periodic=True, dtype=dtype, device=device)
    elif window == 'hamming':
        win = torch.hamming_window(n_fft, periodic=True, dtype=dtype, device=device)
    else:
        raise ValueError(f"Unsupported window: {window}")

    # --- STFT with center=False ---
    Zxx = torch.stft(
        x_pad,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=win,
        center=False,
        normalized=True,
        onesided=False,
        return_complex=True
    )  # shape: (n_fft, n_frames)

    # Reorder frequencies to [-π, π)
    Zxx = torch.fft.fftshift(Zxx, dim=0)  # (M, n_frames)
    M = n_fft
    A_complex = Zxx  # no √M scaling

    # --- Map each t to its frame ---
    t_indices = torch.arange(T, device=device)
    frame_indices = (t_indices + pad_left) // hop_length  # (T,)
    # Now guaranteed: frame_indices.max() < A_complex.shape[1]
    A_interp = A_complex[:, frame_indices].t()  # (T, M)

    # --- Angular frequencies ---
    freqs_hz = torch.fft.fftshift(torch.fft.fftfreq(M, d=dt)).to(device=device, dtype=dtype)
    omega = 2 * torch.pi * freqs_hz  # (M,)

    if phase_remodulate:
        # phase re-modulation
        j_vals = frame_indices  # shape (T,)
        t_start = j_vals * hop_length - (n_fft - 1)  # shape (T,)

        # omega is (M,) → angular frequencies
        # We need phase = exp(-1j * omega * t_start) → shape (T, M)
        phase_factor = torch.exp(-1j * t_start.unsqueeze(1) * omega.unsqueeze(0))  # (T, M)
        # Apply phase correction
        A_interp = A_interp * phase_factor




    if return_stft:
        return A_interp, omega, A_interp
    else:
        return A_interp, omega




