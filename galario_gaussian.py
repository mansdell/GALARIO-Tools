""" Code for fitting Gaussians to ALMA continuum visibilities using GALARIO """

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
# python galario_gaussian.py 'EPIC_204211116/uvtable_EPIC204211116.txt' 1.33e-3 10.0 0.10 70.0 40.0 -0.1 -0.6 2000 50 --outdir "output"

parser = argparse.ArgumentParser()
parser.add_argument("uvtable", help="name of UV table to fit")
parser.add_argument("wavelength", help="wavelength of data [m]", type=float)
parser.add_argument("f0", help="flux normalization [mJy]", type=float)
parser.add_argument("sigma", help="FWHM of the gaussian [arcsec]", type=float)
parser.add_argument("incl", help="inclination [deg]", type=float)
parser.add_argument("pa", help="position angle [deg]", type=float)
parser.add_argument("dra", help="right ascension offset [arcsec]", type=float)
parser.add_argument("ddec", help="declination offset [arcsec]", type=float)
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
    f0, sigma, inc, pa, dra, ddec = p

    ### convert to correct units
    f0 = 10.**f0
    sigma *= arcsec
    inc *= deg
    pa *= deg
    dra *= arcsec
    ddec *= arcsec

    ### get gaussian profile
    f = GaussianProfile(f0, sigma, rmin, dr, nr)

    ### calculate chi-squared
    chi2 = g_double.chi2Profile(f, rmin, dr, nxy, dxy, u, v, re, im, w,
                                inc=inc, PA=pa, dRA=dra, dDec=ddec)

    ### return likelihood
    return -0.5 * chi2 + lnprior


def GaussianProfile(f0, sigma, rmin, dr, nr):

    """ Gaussian brightness profile. """

    ### calculate radial grid
    R = np.linspace(rmin, rmin + dr * nr, nr, endpoint=False)

    ### return gaussian profile
    return f0 * np.exp(-0.5 * (R / sigma)**2)


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
p0 = [args.f0, args.sigma, args.incl, args.pa, args.dra, args.ddec]
p_ranges = [[0.1, 100.0], [0.01, 1.0], [0., 90.], [-180., 180.], [-2.0, 2.0], [-2.0, 2.0]]

### setup mcmc
ndim = len(p0)
nwalkers = ndim * 6
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

### do mcmc fit
print("\nStarting fit...\n")
start = time.time()
prob, state = None, None
for j in range(nsteps, tsteps + 1, nsteps):

    ### get last 500 samples
    pos, prob, state = sampler.run_mcmc(pos, nsteps, rstate0=state, lnprob0=prob)
    samples = sampler.chain[:, -500:, :].reshape((-1, ndim))

    ### plot corner plot ever nsteps
    fig = corner.corner(samples, labels=[r"$f_0$", r"$\sigma$", r"$incl$", r"PA", r"$\Delta$RA", r"$\Delta$Dec"], show_titles=True, quantiles=[0.16, 0.50, 0.84], label_kwargs={'labelpad': 20, 'fontsize': 0}, fontsize=8)
    fig.savefig(os.path.join(outdir, "corner_{}.png".format(j)))
    plt.close('all')

    ### output walkers every nsteps
    np.save(os.path.join(outdir, "chain_{}".format(j)), sampler.chain)
    print("{0} steps completed: chain saved in chain_{0}.npy - corner plot saved in triangle_{0}".format(j))

### plot final chains
samples_final = np.load(os.path.join(outdir, "chain_{}.npy".format(j)))
labels=[r"$f_0$", r"$\sigma$", r"$incl$", r"PA", r"$\Delta$RA", r"$\Delta$Dec"]
fig, [axf0, axSig, axInc, axPA, axRA, axDec] = plt.subplots(len(labels), 1, figsize=(9, len(labels) * 2 + 1))
axes = [axf0, axSig, axInc, axPA, axRA, axDec]
for walker in samples_final:
    for i, (ax, lab) in enumerate(zip(axes, labels)):
        ax.plot(walker[:, i])
        ax.set_xlabel("step number", fontsize=14)
        ax.set_ylabel(lab, fontsize=14)
fig.savefig(os.path.join(args.outdir, "chain_{}.pdf".format(j)), bbox='tight', rastersize=True, dpi=100)
plt.close('all')

### print out elapsed time
print("\n Total time: " + str(int((time.time() - start) / 60)) + " min")
