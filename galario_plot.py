
""" 
Code for plotting results of galario_gaussian.py or galario_binary.py
Used in Ansdell et al. 2019 (Dipper Disc Inclinations, in prep)
"""


# ======================== Import Packages ================

from __future__ import (division, print_function, absolute_import, unicode_literals)

import os, pdb, sys
import numpy as np
import argparse

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

from galario import arcsec, deg
import galario.double as g_double
from uvplot import UVTable
import corner


# ====================== Parse Arguments ==================

### example command-line input
# python galario_plot.py 'EPIC_203824153' 'EPIC_203824153/uvtable_EPIC_203824153.txt' 'EPIC_203824153/chain_5000.npy' 1.3e-3 'EPIC_203824153' --uvbin 30e3
# python galario_plot.py 'EPIC_203770559' 'EPIC_203770559/uvtable_EPIC_203770559.txt' 'EPIC_203770559/chain_10000.npy' 1.3e-3 'EPIC_203770559' --binary True
# python galario_plot.py 'EPIC_204630363' 'EPIC_204630363/uvtable_EPIC_204630363.txt' 'EPIC_204630363/chain_10000.npy' 1.3e-3 'EPIC_204630363' --cavity True

parser = argparse.ArgumentParser()
parser.add_argument("targname", help="name target")
parser.add_argument("uvtable", help="name of UV table to fit")
parser.add_argument("mcmcfile", help="name of mcmc results file")
parser.add_argument("wavelength", help="wavelength of data [m]", type=float)
parser.add_argument("outdir", help="name of directory to output things")
parser.add_argument("--binary", help="set if binary system", type=bool, default=False)
parser.add_argument("--cavity", help="set if binary system", type=bool, default=False)
parser.add_argument("--uvbin", help="size of uvbin for plotting data + model", type=float)
parser.add_argument("--chain_recall", help="number of steps backward to use for plotting", type=int)
parser.add_argument("--incl", help="manually set inclination", type=float)
parser.add_argument("--pa", help="manually set position angle", type=float)
parser.add_argument("--dra", help="manually set ra offset", type=float)
parser.add_argument("--ddec", help="manually set dec offset", type=float)
parser.add_argument("--sigma", help="manually set FWHM", type=float)
parser.add_argument("--f0", help="manually set flux normalization", type=float)
parser.add_argument("--r_in", help="set if inner cavity", type=float)
args = parser.parse_args()


# ===================== Define Functions ==================

def GaussianProfile(f0, sigma, rmin, dr, nr):

    """ Gaussian brightness profile. """

    ### calculate radial grid
    r = np.linspace(rmin, rmin + dr * nr, nr, endpoint=False)

    ### calculate gaussian profile
    f = f0 * np.exp(-0.5 * (r / sigma)**2)

    ### return gaussian profile
    return f, r


# ========================== Code =========================

### READ UV TABLE AND GET VISIBILITIES
print("\nReading in UV table: " + args.uvtable)
u, v, re, im, w = np.loadtxt(args.uvtable, unpack=True)
u, v = np.ascontiguousarray(u), np.ascontiguousarray(v)
u /= args.wavelength
v /= args.wavelength

### GET IMAGE SIZE
print("\nImage size: ")
nxy, dxy = g_double.get_image_size(u, v, verbose=True, f_max=2.1, f_min=2.0)
rmin, dr, nr = 1e-4 * arcsec, 0.001 * arcsec, 20000

### CHECK FOR EMCEE RESULTS FILE
if os.path.isfile(args.mcmcfile) == False:
    print("File for plotting not found!")
    sys.exit()

### GET EMCEE RESULTS
if args.chain_recall:
    chain_recall = args.chain_recall
else:
    chain_recall = 500
samples_all = np.load(args.mcmcfile)
nwalkers, nsteps, ndim = samples_all.shape
samples = samples_all[:, -chain_recall:, :].reshape((-1, ndim))

### SET LABELS FOR PLOTTING
if args.cavity == True:
    labels = [r"$f_0$", r"$\sigma$", r"$r_{in}$", r"incl", r"PA", r"$\Delta$RA", r"$\Delta$Dec"]
elif args.binary == True:
    labels = [r"$f_{0a}$", r"$\sigma_a$", r"incl$_a$", r"PA$_a$", r"$\Delta$RA$_a$", r"$\Delta$Dec$_a$",
              r"$f_{0b}$", r"$\sigma_b$", r"incl$_b$", r"PA$_b$", r"$\Delta$RA$_b$", r"$\Delta$Dec$_b$"]
else:
    labels = [r"$f_0$", r"$\sigma$", r"incl", r"PA", r"$\Delta$RA", r"$\Delta$Dec"]

### PLOT CORNER PLOT
print("\nPlotting corner plot...")
fig = corner.corner(samples, labels=labels, show_titles=True, quantiles=[0.16, 0.50, 0.84], label_kwargs={'labelpad': 20, 'fontsize': 0}, fontsize=8)
fig.savefig(os.path.join(args.outdir, args.targname + '_corner.png'), rastersize=True, dpi=100)

### PLOT CHAINS
print("\nPlotting chains plot...")
fig, axes = plt.subplots(len(labels), 1, figsize=(9, len(labels) * 2 + 1))
for walker in samples_all:
    for i, (ax, lab) in enumerate(zip(axes, labels)):
        ax.plot(walker[:, i])
        ax.set_xlabel('step number', fontsize=14)
        ax.set_ylabel(lab, fontsize=14)
fig.savefig(os.path.join(args.outdir, args.targname + '_chains.png'), bbox='tight', rastersize=True, dpi=100)
plt.close('all')

### GET BEST FIT RESULTS
bestfit = [np.percentile(samples[:, i], 50) for i in range(ndim)]
if args.cavity == True:
    f0, sigma, r_in, inc, pa, dra, ddec = bestfit
    r_in *= arcsec
elif args.binary == True:
    f0_a, sigma_a, inc_a, pa_a, dra_a, ddec_a, f0_b, sigma_b, inc_b, pa_b, dra_b, ddec_b = bestfit
else:
    f0, sigma, inc, pa, dra, ddec = bestfit
    # f0, sigma, inc, pa, dra, ddec, f0_b, sigma_b, inc_b, pa_b, dra_b, ddec_b = bestfit
print("\nBESTFIT", bestfit)

### REAPLCE WITH MANUAL VALUES?
if args.incl:
    inc = args.incl
    print("\n using manual inclination")
if args.pa:
    pa = args.pa
    print("\n using manual position angle")
if args.dra:
    dra = args.dra
    print("\n using manual ra offset")
if args.ddec:
    ddec = args.ddec
    print("\n using manual dec offset")
if args.sigma:
    sigma = args.sigma
    print("\n using manual sigma")
if args.f0:
    f0 = args.f0
    print("\n using manual f0")
if args.r_in:
    r_in = args.r_in
    r_in *= arcsec
    print("\n using manual r_in")

### CREATE BEST-FIT MODEL
print("\nSaving best-fit model as uvtable_mod.txt")
if args.binary == False:
    f0 = 10.**f0
    sigma *= arcsec
    inc *= deg
    pa *= deg
    dra *= arcsec
    ddec *= arcsec
    f, r = GaussianProfile(f0, sigma, rmin, dr, nr)
    if args.cavity == True:
        f[r < r_in] *= 0
    vis_mod = g_double.sampleProfile(f, rmin, dr, nxy, dxy, np.ascontiguousarray(u), np.ascontiguousarray(v), inc=inc, PA=pa, dRA=dra, dDec=ddec)
else:
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
    f_a = GaussianProfile(f0_a, sigma_a, rmin, dr, nr)
    f_b = GaussianProfile(f0_b, sigma_b, rmin, dr, nr)
    vis_mod_a = g_double.sampleProfile(f_a[0], rmin, dr, nxy, dxy, np.ascontiguousarray(u), np.ascontiguousarray(v), inc=inc_a, PA=pa_a, dRA=dra_a, dDec=ddec_a)   
    vis_mod_b = g_double.sampleProfile(f_b[0], rmin, dr, nxy, dxy, np.ascontiguousarray(u), np.ascontiguousarray(v), inc=inc_b, PA=pa_b, dRA=dra_b, dDec=ddec_b)
    vis_mod = vis_mod_a + vis_mod_b 
np.savetxt(os.path.join(args.outdir, args.targname + '_uvtable_mod.txt'), np.stack([u * args.wavelength, v * args.wavelength, vis_mod.real, vis_mod.imag, w], axis=-1))


### GET VISIBILITIES OF DATA OF DATA
print("\nCreating UV plot of best-fit model...")
uv = UVTable(filename=args.uvtable, wle=args.wavelength)
if args.binary == False:
    uv.apply_phase(-dra, -ddec)
    uv.deproject(inc, pa)
else:
    uv.apply_phase(-dra_a, -ddec_a,)
    uv.deproject(inc_a, pa_a,)

### GET VISIBILITIES OF BEST-FIT MODEL
uv_mod = UVTable(filename=os.path.join(args.outdir, args.targname + '_uvtable_mod.txt'), wle=args.wavelength)
if args.binary == False:
    uv_mod.apply_phase(-dra, -ddec)
    uv_mod.deproject(inc, pa)
else:
    uv_mod.apply_phase(-dra_a, -ddec_a)
    uv_mod.deproject(inc_a, pa_a)

### PLOT DATA + MODEL
if args.uvbin:
    uvbin_size = args.uvbin
else:
    uvbin_size = 30.e3
axes = uv.plot(linestyle='.', color='k', label='Data', uvbin_size=uvbin_size)
uv_mod.plot(axes=axes, linestyle='-', color='r', label='Model', yerr=False, uvbin_size=uvbin_size)
axes[0].figure.savefig(os.path.join(args.outdir, args.targname + '_uvplot.png'), rasterisized=True, dpi=400)

### SAVE DATA + MODEL
np.savetxt(os.path.join(args.outdir, args.targname + '_visfit.txt'), np.stack([uv.bin_re, uv.bin_re_err, uv.bin_uvdist, uv_mod.bin_re, uv_mod.bin_re_err, uv_mod.bin_uvdist], axis=-1))
