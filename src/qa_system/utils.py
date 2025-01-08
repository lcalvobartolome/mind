import pathlib
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize
import numpy as np
import pickle
from itertools import islice
import gzip
from prompter import Prompter
import sys
import os
import pathlib
import numpy as np
import json
import matplotlib.pyplot as plt
from kneed import KneeLocator
from scipy.ndimage import uniform_filter1d
from scipy import sparse

def get_doc_top_tpcs(doc_distr, topn=2):
    sorted_tpc_indices = np.argsort(doc_distr)[::-1]
    top = sorted_tpc_indices[:topn].tolist()
    top_weight = [(k, doc_distr[k]) for k in top]
    return top_weight

def get_doc_main_topc(doc_distr):
    sorted_tpc_indices = np.argsort(doc_distr)[::-1]
    top = sorted_tpc_indices[:1][0]
    return top

def thrFig(
        thetas32,
        topics=None,
        max_docs=1000,
        poly_degree=3,
        smoothing_window=5,
        do_knee=True,
        n_steps=1000,
        figsize=(10, 6),
        fontsize=12,
        output_fpath=None,
    ):
    significant_docs = {}
    all_elbows = []
    
    # use colorbrewer Set2 colors
    colors = plt.cm.Dark2(np.linspace(0, 1, thetas32.shape[1]))
    n_docs = thetas32.shape[0]
    print(max_docs)
    max_docs = n_docs
    plt.figure(figsize=figsize)

    lines = []
    for k in range(len(thetas32.T)):
        theta_k = np.sort(thetas32[:, k])
        theta_over_th = theta_k[-max_docs:]
        step = max(1, int(np.round(len(theta_over_th) / n_steps)))
        y_values = theta_over_th[::step]
        x_values = np.arange(n_docs-max_docs, n_docs)[::step]

        # Apply smoothing
        x_values_smooth = uniform_filter1d(x_values, size=smoothing_window)

        label = None
        if topics is not None:
            label = topics[k]
        line, = plt.plot(x_values_smooth, y_values, color=colors[k], label=label)
        lines.append(line)
        
        if do_knee:
            # Using KneeLocator to find the elbow point
            allvalues = np.sort(thetas32[:, k].flatten())
            step = int(np.round(len(allvalues) / 1000))
            theta_values = allvalues[::step]
            idx_values = (100 / len(allvalues)) * np.arange(0, len(allvalues))[::step]
            
            # Apply smoothing
            idx_values_smooth = uniform_filter1d(idx_values, size=smoothing_window)

            kneedle = KneeLocator(theta_values, idx_values_smooth, curve='concave', direction='increasing', interp_method='polynomial', polynomial_degree=poly_degree)
            elbow = kneedle.elbow
            if elbow is not None:
                all_elbows.append(elbow)

                # Filter document indices based on the elbow point (keeping values above the elbow)
                significant_docs[k] = np.where(thetas32[:, k] >= elbow)[0]

        if elbow:
            # plot elbow in same color, smaller linewidth
            plt.plot([n_docs - max_docs, n_docs], [elbow, elbow], color=colors[k], linestyle='--', linewidth=1)

    # add legend where this series is named with the kth topic, do not assign to the 
    # elbow line
    if topics is not None:
        plt.legend(handles=lines, loc='upper left', fontsize=fontsize-1)

    # Add axis labels
    plt.xlabel('Document Index', fontsize=fontsize)
    plt.ylabel('Theta — P(k | d)', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    if output_fpath:
        # make bounding box extremely tight
        plt.savefig(output_fpath, bbox_inches='tight', pad_inches=0)

    plt.show()

    return significant_docs, all_elbows