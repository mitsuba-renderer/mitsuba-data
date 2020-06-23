import sys
import numpy as np

import random

import matplotlib
matplotlib.use('svg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import mitsuba
mitsuba.set_variant("scalar_rgb")

from mitsuba.core import xml

def plot_samples(sampler, filename):

    sample_count = sampler.sample_count()

    sampler.seed(0)
    samples = []
    for s in range(sample_count):
        sampler.prepare_wavefront()
        samples.append(sampler.next_2d())

    xx = [samples[s][0] for s in range(sample_count)]
    yy = [samples[s][1] for s in range(sample_count)]

    fig, axes = plt.subplots(nrows=2, ncols=2,
                             figsize=(10, 10),
                             gridspec_kw=dict(width_ratios=[0.9, 0.1], height_ratios=[0.1, 0.9]))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    # Plot 2D samples

    axes[1][0].scatter(xx, yy)
    axes[1][0].set_xlim([0.0, 1.0])
    axes[1][0].set_ylim([0.0, 1.0])

    axes[1][0].xaxis.set_major_locator(MultipleLocator(0.2))
    axes[1][0].xaxis.set_minor_locator(MultipleLocator(0.02))
    axes[1][0].yaxis.set_major_locator(MultipleLocator(0.2))
    axes[1][0].yaxis.set_minor_locator(MultipleLocator(0.02))

    # Plot 1D projections on Y axis

    axes[1][1].vlines(0.0, 0.0, 1.0)
    proj = yy #random.sample(yy, 512)
    axes[1][1].plot(np.zeros(np.shape(proj)), proj, '_', ms=20)

    axes[1][1].set_xlim(0.0, 1.0)
    axes[1][1].set_ylim(0.0, 1.0)
    axes[1][1].axis('off')

    # Plot 1D projections on X axis

    axes[0][0].hlines(0.0, 0.0, 1)
    proj = xx #random.sample(xx, 512)
    axes[0][0].plot(proj, np.zeros(np.shape(proj)), '|', ms=20)

    axes[0][0].set_xlim(0.0, 1.0)
    axes[0][0].set_ylim(0.0, 1.0)
    axes[0][0].axis('off')

    # Make the 4th plot invisible

    axes[0][1].axis('off')

    fig.savefig("%s.svg" % filename)


sample_count = 1024
sampler_dicts = {
    "independent" : {
        "type" : "independent",
        "sample_count" : sample_count,
    },
    "stratified" : {
        "type" : "stratified",
        "sample_count" : sample_count,
    },
    "multijitter" : {
        "type" : "multijitter",
        "sample_count" : sample_count,
    },
    "ldsampler" : {
        "type" : "ldsampler",
        "sample_count" : sample_count,
    },
    "bose" : {
        "type" : "bose",
        "sample_count" : sample_count,
    },
    "bush" : {
        "type" : "bush",
        "sample_count" : sample_count,
    },
}

for name, sampler_dict in sampler_dicts.items():
    sampler = xml.load_dict(sampler_dict)
    plot_samples(sampler, "%s_samples" % name)
