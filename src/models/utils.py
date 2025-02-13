import torch as T


def calculate_signal_efficiency(
    scores: T.Tensor,
    labels: T.Tensor,
    rejection_rate: float = 0.999,
) -> T.Tensor:
    """Calculate the signal efficiency at a given background rejection rate."""
    # Pull out the background scores and sort them
    bkg_mask = labels == 0
    bkg_scores = T.sort(scores[bkg_mask]).values

    # Find the score that is at the desired rejection rate to get the threshold
    rej_index = int(bkg_scores.shape[0] * rejection_rate)
    threshold = bkg_scores[rej_index]

    # Calculate the signal efficiency at this threshold
    sig_mask = labels == 1
    sig_scores = scores[sig_mask]
    if sig_mask.sum() == 0:  # If the signal is empty, return 0
        return T.tensor(0.0, device=scores.device)
    return (sig_scores > threshold).float().mean()
