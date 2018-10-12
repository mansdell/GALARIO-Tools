""" Code for fitting two Gaussians (i.e., a binary pair) to ALMA continuum visibilities using GALARIO """

# ======================== Import Packages ==========================

from __future__ import (division, print_function, absolute_import, unicode_literals)

import os, pdb, sys
import numpy as np
import argparse
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from emcee import EnsembleSampler
import corner
from galario import arcsec, deg
import galario.double as g_double
import time


# ====================== Parse Arguments ==================

### example command-line input
# python galario_binary.py 'input/uvtable.txt' 1.33e-3 10.0 0.06 30.0 140.0 -0.20 -0.76 10.0 0.05 40.0 150.0 1.17 -0.31 5000 40 --outdir 'output'

### input parameters
parser = argparse.ArgumentParser()
parser.add_argument("uvtable", help="name of UV table to fit")
parser.add_argument("wavelength", help="wavelength of data [m]", type=float)

### primary star parameters
parser.add_argument("f0_a", help="flux normalization [mJy]", type=float)
parser.add_argument("sigma_a", help="FWHM of the gaussian [arcsec]", type=float)
parser.add_argument("incl_a", help="inclination [deg]", type=float)
parser.add_argument("pa_a", help="position angle [deg]", type=float)
parser.add_argument("dra_a", help="right ascension offset [arcsec]", type=float)
parser.add_argument("ddec_a", help="declination offset [arcsec]", type=float)

### secondary star parameters
parser.add_argument("f0_b", help="flux normalization [mJy]", type=float)
parser.add_argument("sigma_b", help="FWHM of the gaussian [arcsec]", type=float)
parser.add_argument("incl_b", help="inclination [deg]", type=float)
parser.add_argument("pa_b", help="position angle [deg]", type=float)
parser.add_argument("dra_b", help="right ascension offset [arcsec]", type=float)
parser.add_argument("ddec_b", help="declination offset [arcsec]", type=float)

### emcee parameters
parser.add_argument("nsteps", help="number of steps to run mcmc", type=int)
parser.add_argument("nthreads", help="number of emcee threads to use", type=int)
parser.add_argument("--outdir", help="output directory; otherwise outputs to current directory")
parser.add_argument("--restart", help="restart from existing snapshops; input filename of snapshot")
args = parser.parse_args()


# ===================== Define Functions ===================

def lnpriorfn(p, par_ranges):

    """ Uniform prior probability function """

    for i in range(len(p)):
        if p[i] < par_ranges[i][0] or p[i] > par_ranges[i][1]:
            return -np.inf

    jacob = -p[0]

    return jacob


def lnpostfn(p, p_ranges, rmin, dr, nr, nxy, dxy, u, v, re, im, w):

    """ Log of posterior probability function """

    ### apply prior
    lnprior = lnpriorfn(p, p_ranges)
    if not np.isfinite(lnprior):
        return -np.inf

    ### unpack the parameters
    f0_a, sigma_a, inc_a, pa_a, dra_a, ddec_a, f0_b, sigma_b, inc_b, pa_b, dra_b, ddec_b = p

    ### convert to correct units
    f0_a = 10.**f0_a
    f0_b = 10.**f0_b
    sigma_a *= arcsec
    sigma_b *= arcsec
    inc_a *= deg
    inc_b *= deg
    pa_a *= deg
    pa_b *= deg
    dra_a *= arcsec
    dra_b *= arcsec
    ddec_a *= arcsec
    ddec_b *= arcsec

    ### get gaussian profile
    f_a = GaussianProfile(f0_a, sigma_a, rmin, dr, nr)
    f_b = GaussianProfile(f0_b, sigma_b, rmin, dr, nr)

    ### calculate chi-squared
    chi2_a = g_double.chi2Profile(f_a, rmin, dr, nxy, dxy, u, v, re, im, w, inc=inc_a, PA=pa_a, dRA=dra_a, dDec=ddec_a)
    chi2_b = g_double.chi2Profile(f_b, rmin, dr, nxy, dxy, u, v, re, im, w, inc=inc_b, PA=pa_b, dRA=dra_b, dDec=ddec_b)

    ### return likelihood
    return (-0.5 * chi2_a) + (-0.5 * chi2_b) + lnprior


def GaussianProfile(f0, sigma, rmin, dr, nr):

    """ Gaussian brightness profile. """

    ### calculate radial grid
    r = np.linspace(rmin, rmin + dr * nr, nr, endpoint=False)

    ### calculate gaussian profile
    f = f0 * np.exp(-0.5 * (r / sigma)**2)

    ### return gaussian profile
    return f


# ========================== Code ==========================

### read in visibilities
print("\nReading in UV table: " + args.uvtable)
U, V, Re, Im, W = np.loadtxt(args.uvtable, unpack=True)
U, V = np.ascontiguousarray(U), np.ascontiguousarray(V)
U /= args.wavelength
V /= args.wavelength

### get image dimensions
print("\nImage size: ")
Nxy, Dxy = g_double.get_image_size(U, V, verbose=True, f_max=2.1, f_min=2.0)
Rmin, dR, nR = 1e-4 * arcsec, 0.01 * arcsec, 2000

### get initial guesses and ranges of gaussian fit parameters
p0 = [args.f0_a, args.sigma_a, args.incl_a, args.pa_a, args.dra_a, args.ddec_a, args.f0_b, args.sigma_b, args.incl_b, args.pa_b, args.dra_b, args.ddec_b]
p_ranges = [[0.1, 100.], [0.01, 5.], [0., 90.], [0., 180.], [-2., 2.], [-2., 2.], [0.1, 100.], [0.01, 5.], [0., 90.], [0., 180.], [-2., 2.], [-2., 2.]]


### setup mcmc
ndim = len(p0)
nwalkers = ndim * 10
nsteps = int(args.nsteps / 10)
tsteps = args.nsteps
nthreads = args.nthreads
print("\nEmcee setup:")
print("   Steps = " + str(tsteps))
print("   Walkers = " + str(nwalkers))
print("   Threads = " + str(nthreads))

### set sampler and initial positions of walkers
sampler = EnsembleSampler(nwalkers, ndim, lnpostfn, args=[p_ranges, Rmin, dR, nR, Nxy, Dxy, U, V, Re, Im, W], threads=nthreads)
if args.restart:
    pos = np.load(args.restart)[:, -1, :]
    print("Restarting from " + args.restart)
else:
    pos = [p0 + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]

### set output directory
if args.outdir:
    outdir = args.outdir
else:
    outdir = '.'

### set labels for plotting
labels = [r"$f_{0a}$", r"$\sigma_a$", r"incl$_a$", r"PA$_a$", r"$\Delta$RA$_a$", r"$\Delta$Dec$_a$",
          r"$f_{0b}$", r"$\sigma_b$", r"incl$_b$", r"PA$_b$", r"$\Delta$RA$_b$", r"$\Delta$Dec$_b$"]

### do mcmc fit
print("\nStarting fit...\n")
start = time.time()
prob, state = None, None
for j in range(nsteps, tsteps + 1, nsteps):

    ### get last 500 samples
    pos, prob, state = sampler.run_mcmc(pos, nsteps, rstate0=state, lnprob0=prob)
    samples = sampler.chain[:, -500:, :].reshape((-1, ndim))

    ### plot corner plot ever nsteps
    fig = corner.corner(samples, labels=labels, show_titles=True, quantiles=[0.16, 0.50, 0.84], label_kwargs={'labelpad': 20, 'fontsize': 0}, fontsize=8)
    fig.savefig(os.path.join(outdir, "corner_{}.png".format(j)))
    plt.close('all')

    ### output walkers every nsteps
    np.save(os.path.join(outdir, "chain_{}".format(j)), sampler.chain)
    print("{0} steps completed: chain saved in chain_{0}.npy - corner plot saved in triangle_{0}".format(j))

### plot final chains
samples_final = np.load(os.path.join(outdir, "chain_{}.npy".format(j)))
fig, axes = plt.subplots(len(labels), 1, figsize=(9, len(labels) * 2 + 1))
for walker in samples_final:
    for i, (ax, lab) in enumerate(zip(axes, labels)):
        ax.plot(walker[:, i])
        ax.set_xlabel("step number", fontsize=14)
        ax.set_ylabel(lab, fontsize=14)
fig.savefig(os.path.join(args.outdir, "chain_{}.png".format(j)), bbox='tight', rastersize=True, dpi=100)
plt.close('all')

### print out elapsed time
print("\n Total time: " + str(int((time.time() - start) / 60)) + " min")
