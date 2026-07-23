import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import math
import h5py
import argparse
import os

from arepy.read_write import binary_read as rsnap
from arepy.read_write import binary_write as wsnap
from arepy.utility import cgs_constants as cgs
from arepy.utility import snap_utility as sutil
from arepy.utility import snap_utility as snut

import matplotlib.colors as mc
from matplotlib.colors import LogNorm
from scipy.interpolate import NearestNDInterpolator
from numpy import uint32, uint64, float64, float32
import math

from numpy import sqrt as sqrt
from numpy import cov as cov
from numpy import sum as npsum
from numpy import abs as npabs
from numpy import column_stack as colstack
from numpy import vstack as vstack
from numpy import transpose as transpose
from numpy import arange as arange
from numpy import mean as mean

from scipy import signal
from scipy import ndimage
from scipy.stats import norm
import scipy.stats as stats
from statistics import mean, stdev
import yt
import sys
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 

from scipy import ndimage
from scipy.ndimage import gaussian_filter

from tqdm import tqdm
import time

import yt
import matplotlib.pyplot as plt
import numpy as np
import pickle
from astropy import units as u, constants as c

#arepo conversions
ulength = 3.0856e20       # [cm]
umass = 1.991e33          # [g]
uvel = 1.0e5              # [cm/s]
utime = ulength/uvel
udensity = umass/ulength/ulength/ulength
uenergy= umass*uvel*uvel
ucolumn = umass/ulength/ulength
uMyr=utime/(60.*60.*24.*365.25*1.e6)
uparsec=ulength/3.0856e18
uyear=utime/(365.25*24.*60.*60.)
umag = umass**0.5 / ulength**1.5 * uvel
mp = 1.6726231e-24
kb = 1.3806485e-16
ABHE = 0.1
xHe=0.1

# ---- Units
pc = c.pc.cgs.value
kB  = c.k_B.cgs.value
Msun = c.M_sun.cgs.value
G = c.G.cgs.value
Myr = u.Myr.in_units("s")
kpc = 1e3*pc

########### GENERAL STUFF  ##########

#make the images bigger
fig_size = plt.rcParams["figure.figsize"]
fig_size[0]=10
fig_size[1]=10
plt.rcParams["figure.figsize"] = fig_size

def read_snapshot_hdf5(filename, verbose=False):
    """
    Read data from an HDF5 snapshot file.

    Args:
        filename (str): Name of HDF5 file to be read
        verbose (bool, optional): Print additional info. Default is False.

    Returns:
        dict: A dictionary containing data and information from the HDF5 file, with keys:
            - 'header' for header information
            - 'params' for simulation parameters
            - 'config' for simulation configuration
            - 'data' for data from 'PartType0'
            - 'sink_data' for sink data (if present)
    """
    if verbose:
        print("Reading ", filename)
    header = {}
    data = {}
    halo_data = {}
    disk_data = {}
    star_data = {}
    sink_data = {}
    params = {}
    config = {}
    stars_present = False
    sinks_present = False
    with h5py.File(filename,'r') as f:

        for item in f['Header'].attrs:
            header[item] = f['Header'].attrs[item]

        for item in f['Parameters'].attrs:
            params[item] = f['Parameters'].attrs[item]

        for item in f['Config'].attrs:
            config[item] = f['Config'].attrs[item]

        for item in f['PartType0'].keys():
            data[item] = f['PartType0'][item][:]

        if "PartType1" in f.keys():
            if verbose:
                print(len(f["PartType1"]["Coordinates"]), "halo")
            halo_present = True
            for item in f['PartType1'].keys():
                halo_data[item] = f['PartType1'][item][:]

        if "PartType2" in f.keys():
            if verbose:
                print(len(f["PartType2"]["Coordinates"]), "disk present")
            disk_present = True
            for item in f['PartType2'].keys():
                disk_data[item] = f['PartType2'][item][:]

        if "PartType4" in f.keys():
            if verbose:
                print(len(f["PartType4"]["Coordinates"]), "stars present")
            stars_present = True
            for item in f['PartType4'].keys():
                star_data[item] = f['PartType4'][item][:]
        
        if "PartType5" in f.keys():
            if verbose:
                print(len(f["PartType5"]["Coordinates"]), "sinks present")
            sinks_present = True
            for item in f['PartType5'].keys():
                sink_data[item] = f['PartType5'][item][:]
                
    output = {
        'header': header,
        'params': params,
        'config': config,
        'data': data,
        'halo_data': halo_data if halo_present else None,
        'disk_data': disk_data if disk_present else None,
        'star_data': star_data if stars_present else None,
        'sink_data': sink_data if sinks_present else None
    }
    return output

def write_snapshot_hdf5(filename, output):
    """Write modified data back to HDF5 file"""
    with h5py.File(filename, 'r+') as file:  # 'r+' mode for read/write
        # Overwrite PartType0 datasets
        for key in output['data']:
            if key in file['PartType0']:
                file['PartType0'][key][:] = output['data'][key]
                
def parse_args():
    p = argparse.ArgumentParser(description="Run HRO pipeline for one snapshot.")
    p.add_argument("--snapshot", type=int, required=True, help="Snapshot number, e.g. 1356")
    p.add_argument("--base-snap", type=str, default="/cosma8/data/dp058/dc-whit3/Lyon/LSD_model/snapshots/",
                   help="Base path containing MHD_LSD_XXXX.hdf5")
    p.add_argument("--base-out", type=str, default="/cosma8/data/dp058/dc-whit3/Lyon/LSD_model/grids/",
                   help="Output directory for cube hdf5 and hro npz")
    p.add_argument("--level", type=int, default=10)
    p.add_argument("--plot", action="store_true", help="Enable plots (default off)")
    return p.parse_args()                
                
args = parse_args()
number = args.snapshot
filenum = str(number).zfill(3)

base_SAT = args.base_snap

file_path = os.path.join(base_SAT, f"MHD_LSD_{filenum}.hdf5")
f = file_path

level = args.level

rsnap.io_flags['mc_tracer']=False
rsnap.io_flags['time_steps']=True
rsnap.io_flags['sgchem']=True
rsnap.io_flags['variable_metallicity']=False
rsnap.io_flags['MHD']=True
rsnap.io_flags['potential']=True
rsnap.io_flags['sgchem_NL99']=True
rsnap.io_flags['Grav_acc']=False

output = read_snapshot_hdf5(f)
data = output['data']
header = output['header']
star_data = output['star_data']
time = header['Time']*uMyr
ID = data['ParticleIDs']

halo_data = output['halo_data']
print('Number of stars = ', star_data['Masses'].shape)    #3129
print('star part ID max', np.max(star_data['PID']))
print('data part ID max', np.max(data['ParticleIDs']))
print('data part shape', data['ParticleIDs'].shape)

# ---- PARAMETERS (set these directly in the cell)

fields = ["magnetic_field_x", "magnetic_field_y", "magnetic_field_z","density", "H2", "HI"]
plot_chk = False

# ---- Load snapshot
ds = yt.load(file_path)

# ---- Add magnetic field vector split if not already there
if not (("gas", "magnetic_field_x") in ds.derived_field_list):
    def _magnetic_field_x(field, data):
        return data["MagneticField"][:, 0]
    def _magnetic_field_y(field, data):
        return data["MagneticField"][:, 1]
    def _magnetic_field_z(field, data):
        return data["MagneticField"][:, 2]
    ds.add_field(("gas", "magnetic_field_x"), function=_magnetic_field_x,
                 units="code_magnetic", sampling_type="local")
    ds.add_field(("gas", "magnetic_field_y"), function=_magnetic_field_y,
                 units="code_magnetic", sampling_type="local")
    ds.add_field(("gas", "magnetic_field_z"), function=_magnetic_field_z,
                 units="code_magnetic", sampling_type="local")

#add chemistry densities
if not (("gas", "H2") in ds.derived_field_list):
    def _H2(field, data):
        return data['ChemicalAbundances'][:, 0]
    def _HI(field, data):
        return data['ChemicalAbundances'][:, 1]
    ds.add_field(("gas", "H2"), function=_H2,
                 sampling_type="local")
    ds.add_field(("gas", "HI"), function=_HI,
                 sampling_type="local")

# ---- Cube geometry setup
N_tot = 2 ** level
x0 = 0; y0 = 0; z0 = 0   # adjust for your target center if needed

ctr_kpc = np.array([50.0, 50.0, 50.0])#ds.domain_center.in_units("kpc").v
ctr_kpc[0] += x0
ctr_kpc[1] += y0
ctr_kpc[2] += z0

win_kpc = np.array([2,2,2]) # edit for your window size
win_frac = win_kpc / ds.domain_width.in_units("kpc").v
N_win = np.asarray(np.round(win_frac * N_tot), dtype=np.int32)
left_kpc  = ctr_kpc - 0.5 * win_kpc
right_kpc = ctr_kpc + 0.5 * win_kpc
left = left_kpc * kpc / ds.parameters["UnitLength_in_cm"]
dx = win_kpc / N_win

# Center and left/right as before
ctr_kpc = np.array([50.0, 50.0, 50.0])
left_kpc  = ctr_kpc - 0.5 * win_kpc
left = left_kpc * kpc / ds.parameters["UnitLength_in_cm"]

# ---- Extract cube
cube = ds.covering_grid(level=level, left_edge=left, dims=N_win)

print("Window (kpc):", win_kpc)
print("Win_frac:", win_frac)
print("Grid shape:", N_win)
print("Voxel size :", dx)
print("ctr of domain  : ", ctr_kpc)
print("level of cut   : ", level)
print("effective Ntot : ", N_tot)
print("window in kpc  : ", win_kpc)
print("effective Nwin : ", N_win)
print("left corner kpc: ", left_kpc)
print("delta x        : ", dx)
print("Simulation domain extent in kpc:", ds.domain_left_edge.in_units('kpc').d, ds.domain_right_edge.in_units('kpc').d)
print("Cube left/right edge (kpc):", left_kpc, right_kpc)
print("Cube center (kpc):", ctr_kpc)
print("Difference from domain center (kpc):", left_kpc - ds.domain_center.in_units('kpc').d)

# ---- Store grid lines (coordinate bins)
data = {}
for i, ix in zip([0, 1, 2], ["x", "y", "z"]):
    data[ix+"glob"]   = np.linspace(left_kpc[i]+0.5*dx[i], right_kpc[i]-0.5*dx[i], N_win[i])
    data[ix+"g_bnds"] = np.linspace(left_kpc[i], right_kpc[i], N_win[i]+1)
    data[ix+"ctr"]    = np.linspace(-0.5*win_kpc[i]+0.5*dx[i], 0.5*win_kpc[i]-0.5*dx[i], N_win[i])
    data[ix+"c_bnds"] = np.linspace(-0.5*win_kpc[i], 0.5*win_kpc[i], N_win[i]+1)

# ---- Extract fields
#for f in fields:
#    print(f)
#    print(cube["gas", f].shape)
    #data[f] = cube["gas", f]
#    data[f] = np.asarray(cube["gas", f].d, dtype=np.float32)
    
# ---- Optional: plot a few slices as a check
if plot_chk:
    extent = [data["xc_bnds"][0],data["xc_bnds"][-1],data["yc_bnds"][0],data["yc_bnds"][-1]]
    fig, ax = plt.subplots(figsize=(5*len(fields), 6), ncols=len(fields))
    for i, f in enumerate(fields):
        # Slices through the middle of the box
        im = ax[i].imshow(np.log10(data[f][:,:,data[f].shape[2]//2]), extent=extent)
        ax[i].set_xlabel("x (kpc)")
        cbar = plt.colorbar(im, ax=ax[i], orientation="horizontal", location="top", label="log "+str(f))
    fig.tight_layout()
    plt.show()

total_z = data['density'].shape[2]   # should be 512
crop_z = data['density'].shape[2]//2   #4
print('total_z:', total_z)
print('crop_z:', crop_z)
start = (total_z - crop_z) // 2      # 192
end = start + crop_z                 # 320
print("Central z-cells:", start, "to", end-1)

print(start, end)                # Should be 192, 320 for 128 centered in 512

print('data[density].shape:', data['density'].shape)

#for f in fields:
#    data[f] = data[f][:, :, start:end]

for f in fields:
    #arr = cube["gas", f].d[:, :, start:end]
    arr = np.asarray(cube["gas", f], dtype=np.float32)[:, :, start:end]
    data[f] = arr#np.asarray(arr, dtype=np.float32)

#print('data[density].shape after crop:', data['density'].shape)
#print('data[magnetic_field_x].shape after crop:', data['magnetic_field_x'].shape)

#print('opening rho')
#rho = data['density']
#print('loaded rho')
#print("rho type:", type(rho))
#print("rho shape:", rho.shape)
#print("rho dtype:", rho.dtype)
#print("xHe type:", type(xHe))

#data['density'] = np.asarray(data['density'])
#data['magnetic_field_x'] = np.asarray(data['magnetic_field_x'])
#data['magnetic_field_y'] = np.asarray(data['magnetic_field_y'])
#data['magnetic_field_z'] = np.asarray(data['magnetic_field_z'])
#data['H2'] = np.asarray(data['H2'])
#data['HI'] = np.asarray(data['HI'])
rho = data['density']
print('loaded rho')
print('converting to yn')
yn = rho/((1. + 4.0 * xHe) * mp)
print('converted')
nH2 = data['H2'] * yn
nHI = data['HI'] * yn
print('opening mag field x')
Bx = data["magnetic_field_x"]#[:, 0]
print('opened')
print('opening y')
By = data["magnetic_field_y"]#[:, 1]
print('opened')
print('opening z')
Bz = data["magnetic_field_z"]#[:, 2]
print('opened')

#yn  = yn.astype(np.float32, copy=False)
#print('yn done')
#nH2  = nH2.astype(np.float32, copy=False)
#print('yn done')
#nHI  = nHI.astype(np.float32, copy=False)
#print('yn done')
#Bx  = Bx.astype(np.float32, copy=False)
#print('Bx done')
#By  = By.astype(np.float32, copy=False)
#print('By done')
#Bz  = Bz.astype(np.float32, copy=False)
#print('Bx done')



cube_out = os.path.join(args.base_out, f"{filenum}_cube_data_float32.hdf5")
with h5py.File(cube_out, "w") as hf:
    hf.create_dataset("yn", data=yn)   # no gzip
    hf.create_dataset("nH2", data=nH2)
    hf.create_dataset("nHI", data=nHI)
    hf.create_dataset("Bx", data=Bx)
    hf.create_dataset("By", data=By)
    hf.create_dataset("Bz", data=Bz)
