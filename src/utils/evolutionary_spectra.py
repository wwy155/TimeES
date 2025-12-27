
import torch
import torch.nn as nn
import torch.nn.functional as F


def construct_hermitian_spectrum(A_half: torch.Tensor, M: int) -> torch.Tensor:
    r"""
    Construct a full Hermitian-symmetric complex spectrum of length M
    from the first half (including DC and Nyquist if M is even).

    Input:
        A_half: Tensor of shape (..., K), where
                K = M // 2 + 1   ← frequency dimension must be the LAST dimension
                - A_half[..., 0]       : DC component
                - A_half[..., 1:-1]    : positive frequencies (1 to M//2 - 1)
                - A_half[..., -1]      : Nyquist frequency (only if M is even)

        M: int, total number of frequency bins (FFT size)

    Output:
        A_full: Tensor of shape (..., M) satisfying Hermitian symmetry,
                so that the inverse DFT yields a real-valued signal.
    """
    *dims, K = A_half.shape
    expected_K = M // 2 + 1
    assert K == expected_K, f"Expected A_half last dim = {expected_K}, got {K}"

    if M % 2 == 0:
        # Even M: [DC, f1, ..., f_{M/2-1}, Nyquist]
        if K > 2:
            # Extract positive frequencies excluding DC and Nyquist: indices 1 to -2
            pos_freqs = A_half[..., 1:-1]               # shape: (..., M//2 - 1)
            neg_freqs = torch.conj(pos_freqs.flip(-1))  # reverse order and conjugate
        else:
            # M=2 → K=2 → no middle frequencies
            neg_freqs = torch.empty(*dims, 0, dtype=A_half.dtype, device=A_half.device)
        A_full = torch.cat([A_half, neg_freqs], dim=-1)
    else:
        # Odd M: [DC, f1, ..., f_{(M-1)/2}]
        if K > 1:
            pos_freqs = A_half[..., 1:]                 # all except DC
            neg_freqs = torch.conj(pos_freqs.flip(-1))
        else:
            # M=1 → K=1 → only DC
            neg_freqs = torch.empty(*dims, 0, dtype=A_half.dtype, device=A_half.device)
        A_full = torch.cat([A_half, neg_freqs], dim=-1)

    # Final sanity check
    assert A_full.shape[-1] == M, f"Output length {A_full.shape[-1]} != M={M}"
    return A_full


def synthesize_signal_on_subband(A_vals, selected_omegas, M, t_index):
    """
    Synthesize real-valued signal from complex amplitudes on a subset of frequencies.
    
    Args:
        A_vals: [..., T, K], complex64
        selected_omegas: [B, K], angular frequencies (rad/sample)
        t_index: [B, T], time indices (long or float)

    Returns:
        x: [..., T], real
    """
    device = A_vals.device
    phase = torch.exp(1j * torch.einsum('...t,...k->...tk', t_index.float(), selected_omegas.to(device)))
    # phase = torch.exp(1j * t_index.unsqueeze(-1) * selected_omegas)
    integrand = A_vals * phase
    x_complex = (1.0 / torch.sqrt(torch.tensor(M, dtype=torch.float32, device=device))) * \
                torch.sum(integrand, dim=-1)
    # x_complex =  torch.sum(integrand, dim=-1)
    return x_complex.real



def select_frequencies_by_energy_ratio(energy_per_bin, ratio=0.96):
    """
    Select the smallest set of frequency bins that contain at least `ratio` of total energy.
    Bins are selected by descending energy (greedy optimal for L2 energy).
    Returned indices are sorted in ascending order for indexing convenience.

    Args:
        energy_per_bin (Tensor): [M], non-negative energy per frequency bin
        ratio (float): in (0, 1]

    Returns:
        freq_indices (LongTensor): selected frequency indices, sorted ascending
    """
    assert 0 < ratio <= 1.0
    energy_per_bin = energy_per_bin.float()
    total_energy = energy_per_bin.sum()

    if total_energy == 0:
        # All zero: return all or none? Usually return all to avoid empty.
        return torch.arange(energy_per_bin.shape[0], device=energy_per_bin.device)
    # Sort by energy descending
    sorted_vals, sorted_idx = torch.sort(energy_per_bin, descending=True)

    # Cumulative sum of top energies
    cumsum = torch.cumsum(sorted_vals, dim=0)
    threshold = ratio * total_energy

    # Find minimal k such that cumsum[k-1] >= threshold
    k = torch.searchsorted(cumsum, threshold, right=True).item() + 1
    k = min(k, len(sorted_idx))  # safety

    # Get top-k indices and sort them ascending (for consistent indexing)
    selected_unsorted = sorted_idx[:k]
    selected_sorted = torch.sort(selected_unsorted).values

    return selected_sorted
