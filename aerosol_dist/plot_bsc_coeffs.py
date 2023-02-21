# Plots backscatter coefficients as a function of humidity for various aerosol ensembles.
# Coefficients are calculated using MOPSMAP software, which utilize Mie theory
# for calculation of optical properties of particles
from netCDF4 import Dataset 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

fig, ax = plt.subplots()
filenames = glob.glob('aerosol_dist/all/*.txt')
lgnd = [fn.split("\\")[1] for fn in filenames]
lgnd = [fn.split("-")[0] for fn in lgnd]
for fn in filenames:
    data = np.genfromtxt(fn, dtype=float)
    RH = data[0]
    bsc = data[1]
    bsc0 = bsc[0]
    print(bsc)
    ax.plot(bsc/bsc0, RH)
ax.legend(lgnd)
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(1))
ax.grid(which='both')
ax.set_title('Normalized backscatter coefficient aaf. of relative humidity', fontsize=24)
ax.set_ylabel('Relative humidity [%]', fontsize=20)
ax.set_xlabel('Backscatter coefficient', fontsize=20)

fig1, ax1 = plt.subplots()
for fn in filenames:
    data = np.genfromtxt(fn, dtype=float)
    RH = data[0]
    bsc = data[1]
    ax1.plot(bsc, RH)
ax1.legend(lgnd, prop={'size': 14})
#ax1.xaxis.set_major_locator(MultipleLocator(5))
#ax1.xaxis.set_minor_locator(MultipleLocator(1))
ax1.yaxis.set_major_locator(MultipleLocator(5))
ax1.yaxis.set_minor_locator(MultipleLocator(1))
ax1.grid(which='both')
ax1.set_title('Backscatter coefficient aaf. of relative humidity', fontsize=24)
ax1.set_ylabel('Relative humidity [%]', fontsize=20)
ax1.set_xlabel('Backscatter coefficient', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.show()