"""
Metrics for uncertainty quantification. 
"""

import numpy as np


def calibration_curve(outputs, labels, num_bins=20):
    confidences = np.max(outputs, 1)
    step = (confidences.shape[0] + num_bins - 1) // num_bins
    bins = np.sort(confidences)[::step]
    if confidences.shape[0] % step != 1:
        bins = np.concatenate((bins, [np.max(confidences)]))
    predictions = np.argmax(outputs, 1)
    bin_lowers = bins[:-1]
    bin_uppers = bins[1:]

    accuracies = predictions == labels

    xs = []
    ys = []
    zs = []

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = (confidences > bin_lower) * (confidences < bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            xs.append(avg_confidence_in_bin)
            ys.append(accuracy_in_bin)
            zs.append(prop_in_bin)
    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)

    out = {
        'confidence': xs,
        'accuracy': ys,
        'p': zs,
        'ece': ece,
    }
    return out


def ece(outputs, labels, num_bins=20):
    return calibration_curve(outputs, labels, num_bins=num_bins)['ece']


def brier(probs, labels):
    one_hot = np.zeros((labels.shape[0], labels.max() + 1))
    one_hot[np.arange(labels.shape[0]), labels] = 1

    return np.mean(np.sum((probs - one_hot) ** 2, 1), 0)


def labels_convert(labels, dataset, other):
    # shift labels between CIFAR10 and STL10
    if dataset == 'STL10' and other == 'CIFAR10':
        new_labels = np.copy(labels)
        new_labels[labels == 1] = 2
        new_labels[labels == 2] = 1
        new_labels[labels == 7] = 6
        new_labels[labels == 6] = 7
        return new_labels
    elif dataset == 'CIFAR10' and other == 'STL10':
        new_labels = np.copy(labels)
        new_labels[labels == 1] = 2
        new_labels[labels == 2] = 1
        new_labels[labels == 7] = 6
        new_labels[labels == 6] = 7
        return new_labels
    else:
        return labels
