import numpy as np
from scipy.stats import entropy


def find_threshold(L, mask, x_frac) -> float:
    """Calculate c such that x_frac of the array is less than c.

    Parameters
    ----------
    L : Array
        The array where the cutoff is to be found
    mask : Array,
        Mask that returns L[mask] the part of the original array over which it is desired to calculate the threshold.
    x_frac : float
        Of the area that is lass than or equal to c.

    returns c (type=L.dtype)
    """
    max_x = mask.sum()
    x = int(np.round(x_frac * max_x))
    L_sorted = np.sort(L[mask.astype(bool)])
    return L_sorted[x]


def calculate_jsd(hist1, hist2) -> float:
    """Calculate the Jensen-Shannon divergence between two histograms.

    Parameters
    ----------
    hist1 : Array
        First histogram
    hist2 : Array
        Second histogram

    Returns
    -------
    float
        Jensen-Shannon divergence between the two histograms.
    """
    # Normalise histograms ###
    # hist1 = hist1 / hist1.sum()
    # hist2 = hist2 / hist2.sum()
    avg_hist = 0.5 * (hist1 + hist2)
    return 0.5 * (entropy(hist1, avg_hist) + entropy(hist2, avg_hist))


def get_r50_jsd(
    preds,
    labels,
    mass,
) -> tuple:
    """Calculate the R50 and JSD metrics for the predictions and labels and return them
    as a tuple.

    Parameters
    ----------
    preds : Array or List
        Predictions
    labels : Array or List
        Labels
    mass : Array or List
        Invariant mass of both signal and background samples
    """
    # Make sure predictions, labels and mass are numpy arrays ###
    preds = np.array(preds)
    labels = np.array(labels)
    mass = np.array(mass)

    # Get threshold for 50% signal efficiency ###
    threshold = find_threshold(preds, (labels == 1), 0.5)

    # Calculate R50
    R50 = 1 / ((preds[labels == 0] > threshold).sum() / (labels == 0).sum())

    # Calculate JSD between background samples that pass and fail selection ###
    # Get the two samples ###
    pass_sample = mass[(preds >= threshold) & (labels == 0)]
    fail_sample = mass[(preds < threshold) & (labels == 0)]
    # get number of pass and fail samples ###
    n_pass = len(pass_sample)
    # Get the histograms ###
    hist_pass, bins = np.histogram(pass_sample, bins=50, density=True)
    hist_fail, _ = np.histogram(fail_sample, bins=bins, density=True)
    # Mask out zeros ###
    mx = (hist_pass > 0) & (hist_fail > 0)
    hist_pass = hist_pass[mx]
    hist_fail = hist_fail[mx]

    # Calculate JSD ###
    JSD = calculate_jsd(hist_pass, hist_fail)

    return R50, JSD, n_pass


def idealised_jsd(data, n_pass, num_rand_samples=10) -> float:
    """Calculate the idealised JSD for mJJ distribution based on the output of
    get_r50_jsd.

    Parameters
    ----------
    data : Array or List
        Dataset with signal and background
    n_pass : int
        Number of events that pass selection from get_r50_jsd. n_fail = data.shape[0] - n_pass
    num_rand_samples : int
        Number of random samples to take to calculate the idealised JSD.

    Returns
    -------
    float
        The idealised JSD between the two samples.
    """
    # Get the number of events in the data ###
    ideal_JSD = []
    for n in np.arange(num_rand_samples):
        # create a shuffled copy of the original data ###
        sampling_data = data.sample(frac=1)
        # Randomly sample n_pass and n_fail events from the data without replacement ###
        data_pass = sampling_data[:n_pass]
        data_fail = sampling_data[n_pass:]
        # Get the histograms ###
        hist_pass, bins = np.histogram(data_pass["mJJ"], bins=50, density=True)
        hist_fail, _ = np.histogram(data_fail["mJJ"], bins=bins, density=True)
        # Mask out zeros ###
        mx = (hist_pass > 0) & (hist_fail > 0)
        hist_pass = hist_pass[mx]
        hist_fail = hist_fail[mx]
        # Calculate JSD ###
        JSD = calculate_jsd(hist_pass, hist_fail)
        ideal_JSD.append(1 / JSD)
    return np.mean(ideal_JSD), np.std(ideal_JSD)
