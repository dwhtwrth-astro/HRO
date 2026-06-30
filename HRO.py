import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import math
import h5py

from arepy.read_write import binary_read as rsnap
from arepy.read_write import binary_write as wsnap
from arepy.utility import cgs_constants as cgs
from arepy.utility import snap_utility as sutil
#import h5py
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
#from arepy.utility import snap_utility as snut
#from arepy.utility.ut_utility import *

import yt
import matplotlib.pyplot as plt
import numpy as np
import pickle
from astropy import units as u, constants as c

#arepo conversions
ulength = 3.0856e20       # [cm]
umass = 1.991e33          # [g]
uvel = 1.0e5              # [cm/s]
udensity = umass/ulength/ulength/ulength   # g/cm^3
umag = umass**0.5 / ulength**1.5 * uvel    # Gauss

mp = 1.6726231e-24        # g
xHe = 0.1                 # atomic fraction of He

# ---- Units
pc = c.pc.cgs.value
kB  = c.k_B.cgs.value
Msun = c.M_sun.cgs.value
G = c.G.cgs.value
Myr = u.Myr.in_units("s")
kpc = 1e3*pc


########### GENERAL STUFF  ##########

#internal arepo units in cgs
ulength = 3.0856e20
umass = 1.991e33
uvel = 1.0e5

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

#make the images bigger
fig_size = plt.rcParams["figure.figsize"]
fig_size[0]=10
fig_size[1]=10
plt.rcParams["figure.figsize"] = fig_size

def JeansMass(cs,yn):
    # In solar masses. cs sound speed in kms^-1, yn in cm^-3
    JM = 2.* (cs / 0.2)**3. * (yn / 1000.)**-0.5
    return JM

def tff(yn):
    # In years. yn in cm^-3
    t = 2.e6 * (yn/1000.)**-0.5
    return t
    
def JeansLength(cs,yn):    #also known as jeans radius
    #In pc
    JL = 0.4 * (cs/0.2) * (yn/1000.)**-0.5
    return JL

def molweight(H2,Hp):
    # Molecular weight given molecular and ionised hydrogen abundance
    ABHE=0.1
    mw=(1.0 + 4.0*ABHE) / (1.0 + ABHE - H2 + Hp)
    return mw

def soundspeed(T,mw):
    # In kms^-1
    mp=1.6726e-24
    kboltz=1.38066e-16
    cs=1.e-5 * np.sqrt(kboltz * T / (mw*mp) )
    return cs

def critdensity(length,cs):
    yncrit = 1.e3*(length/(2.0*cs))**-2 
    return yncrit


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
                
                
#base_SAT = '/cosma8/data/dp058/dc-whit3/MEXICO_MHD/Double_Mass/pNbody_test/sne_test/snapshots/'
base_SAT = '/cosma8/data/dp058/dc-whit3/Lyon/LSD_model/snapshots/'
base = '/cosma8/data/dp058/dc-whit3/Lyon/LSD_model/'
#base_SAT = '/cosma8/data/dp058/dc-whit3/Lyon/LSD_jeans_from_start/snapshots/'
# ---- PARAMETERS (set these directly in the cell)
#file_path = "/cosma8/data/dp058/dc-whit3/Lyon/LSD_model/snapshots/MHD_LSD_3702.hdf5" # adapt as needed

#peaks
#number = 1437
#number = 1836
#number = 2636
#troughs
#number = 1945
#number = 2286
#number = 2487
#number = 3447
#quiescent
#number = 1356
#number = 2405
#inc
#number = 1525
#number = 2165
#number = 2792
#dec
#number = 1576
#number = 2066
number = 3070


filenum = str(number).zfill(3)
file_path = '/cosma8/data/dp058/dc-whit3/Lyon/LSD_model/snapshots/MHD_LSD_' + filenum + '.hdf5'

level = 10
fields = ["magnetic_field_x", "magnetic_field_y", "magnetic_field_z","density"]#, "mass"]
plot_chk = True # or False

filenum = str(number).zfill(3)
f = base_SAT + 'MHD_LSD_' + filenum + '.hdf5'

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

import yt
import matplotlib.pyplot as plt
import numpy as np
import pickle
from astropy import units as u, constants as c

#arepo conversions
ulength = 3.0856e20       # [cm]
umass = 1.991e33          # [g]
uvel = 1.0e5              # [cm/s]
udensity = umass/ulength/ulength/ulength   # g/cm^3
umag = umass**0.5 / ulength**1.5 * uvel    # Gauss

mp = 1.6726231e-24        # g
xHe = 0.1                 # atomic fraction of He

# ---- Units
pc = c.pc.cgs.value
kB  = c.k_B.cgs.value
Msun = c.M_sun.cgs.value
G = c.G.cgs.value
Myr = u.Myr.in_units("s")
kpc = 1e3*pc

# ---- PARAMETERS (set these directly in the cell)
#file_path = "/cosma8/data/dp058/dc-whit3/Lyon/LSD_model/snapshots/MHD_LSD_3702.hdf5" # adapt as needed
level = 10
fields = ["magnetic_field_x", "magnetic_field_y", "magnetic_field_z","density"]#, "mass"]
plot_chk = True # or False

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

#print(ds.derived_field_list)

# ---- Cube geometry setup
N_tot = 2 ** level
x0 = 0; y0 = 0; z0 = 0   # adjust for your target center if needed

ctr_kpc = np.array([50.0, 50.0, 50.0])#ds.domain_center.in_units("kpc").v
ctr_kpc[0] += x0
ctr_kpc[1] += y0
ctr_kpc[2] += z0

win_kpc = np.array([2,2,2])#4, 4, 4]) # edit for your window size
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
for f in fields:
    print(f)
    print(cube["gas", f].shape)
    data[f] = cube["gas", f]

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
    #plt.show()

# ---- Save output as needed
#with open('/cosma8/data/dp058/dc-whit3/Lyon/LSD_model/grids/uniform-grid-lvl%i.pkl' % level, 'wb') as handle:
#    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

total_z = data['density'].shape[2]   # should be 512
crop_z = data['density'].shape[2]//2   #4
print('total_z:', total_z)
print('crop_z:', crop_z)
start = (total_z - crop_z) // 2      # 192
end = start + crop_z                 # 320
print("Central z-cells:", start, "to", end-1)

#z_total = 512
#z_crop = 128

print(start, end)                # Should be 192, 320 for 128 centered in 512

print('data[density].shape:', data['density'].shape)

for f in fields:
    data[f] = data[f][:, :, start:end]

print('data[density].shape after crop:', data['density'].shape)
print('data[magnetic_field_x].shape after crop:', data['magnetic_field_x'].shape)

print('opening rho')
rho = data['density']
print('loaded rho')
print("rho type:", type(rho))
print("rho shape:", rho.shape)
print("rho dtype:", rho.dtype)
print("xHe type:", type(xHe))
print("xHe:", xHe)
print("mp:", mp)
data['density'] = np.asarray(data['density'])
data['magnetic_field_x'] = np.asarray(data['magnetic_field_x'])
data['magnetic_field_y'] = np.asarray(data['magnetic_field_y'])
data['magnetic_field_z'] = np.asarray(data['magnetic_field_z'])
rho = data['density']
print('loaded rho')
print("rho type:", type(rho))
print("rho shape:", rho.shape)
print("rho dtype:", rho.dtype)
print('converting to yn')
yn = rho/((1. + 4.0 * xHe) * mp)
print('converted')
print('opening mag field x')
Bx = data["magnetic_field_x"]#[:, 0]
print('opened')
print('opening y')
By = data["magnetic_field_y"]#[:, 1]
print('opened')
print('opening z')
Bz = data["magnetic_field_z"]#[:, 2]
print('opened')

print(yn.shape)

yn  = yn.astype(np.float32, copy=False)
print('yn done')
#rho = rho.astype(np.float32, copy=False)
#print('rho done')
Bx  = Bx.astype(np.float32, copy=False)
print('Bx done')
By  = By.astype(np.float32, copy=False)
print('By done')
Bz  = Bz.astype(np.float32, copy=False)
print('Bx done')

with h5py.File("/cosma8/data/dp058/dc-whit3/Lyon/LSD_model/grids/test_cube_data_float32.hdf5", "w") as hf:
    hf.create_dataset("yn", data=yn, compression="gzip")
    #hf.create_dataset("rho", data=rho, compression="gzip")
    hf.create_dataset("Bx", data=Bx, compression="gzip")
    hf.create_dataset("By", data=By, compression="gzip")
    hf.create_dataset("Bz", data=Bz, compression="gzip")
print("Saved all arrays and scalars to my_cube_data_float32.h5")

#peaks
#number1 = 1437
#number2 = 1836
#number3 = 2636
#troughs
#number1 = 1945
#number2 = 2286
#number3 = 3447
#quiescent
number1 = 1356
number2 = 2405
#inc
#number1 = 1525
#number2 = 2165
#number3 = 2792
#dec
#number1 = 1576
#number2 = 2066
#number3 = 3070

filenum1 = str(number1).zfill(3)
f1 = base + 'grids/' + filenum1 + '_cube_data_float32.hdf5'
with h5py.File(f1, "r") as hf:
    yn_1356 = hf["yn"][:]
    Bx_1356 = hf["Bx"][:]
    By_1356 = hf["By"][:]
    Bz_1356 = hf["Bz"][:]
print("Loaded data from file:", f1)

filenum2 = str(number2).zfill(3)
f2 = base + 'grids/' + filenum2 + '_cube_data_float32.hdf5'
with h5py.File(f2, "r") as hf:
    yn_2405 = hf["yn"][:]
    Bx_2405 = hf["Bx"][:]
    By_2405 = hf["By"][:]
    Bz_2405 = hf["Bz"][:]
print("Loaded data from file:", f2)

filenum3 = str(number3).zfill(3)
f3 = base + 'grids/' + filenum3 + '_cube_data_float32.hdf5'
with h5py.File(f3, "r") as hf:
    yn_3070 = hf["yn"][:]
    Bx_3070 = hf["Bx"][:]
    By_3070 = hf["By"][:]
    Bz_3070 = hf["Bz"][:]
print("Loaded data from file:", f3)

import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 

#from scipy import ndimage
from scipy.ndimage import gaussian_filter

from tqdm import tqdm

# ========================================================================================================================
def roangles3D(dens, Bx, By, Bz, mode='nearest', pxksz=2.56):#10.24):#5.12):#2.56):
    """
    Calculates the cosine of the relative orientation angles between the density gradient and the magnetic field in 3D. 
    
        Parameters
        ----------
        dens : numpy.ndarray. 
               Density field.

        Bx, By, Bz : float or numpy.ndarray
            	     Components of the 3D magnetic field vector.
    
        Returns
        -------
        cosphi: numpy.ndarray containing the cosine of the relative orientation angles between the density gradient and the magnetic field.
    
        Notes
        -----
    	...
    
        References
        ----------
        .. [1] Soler et al 2013 ...
    
        Examples
        --------
    	...
    """

    assert np.shape(dens) == np.shape(Bx), "Dimensions of dens and Bx must match"
    t0 = time.time()
    print("Step Bx done in", time.time()-t0, "sec")
    assert np.shape(dens) == np.shape(By), "Dimensions of dens and By must match" 
    t0 = time.time()
    print("Step By done in", time.time()-t0, "sec")
    assert np.shape(dens) == np.shape(Bz), "Dimensions of dens and Bz must match"  
    t0 = time.time()
    print("Step Bz done in", time.time()-t0, "sec")
    print(np.shape(dens),np.shape(Bx),np.shape(By),np.shape(Bz))
    #gx=grad[1]; gy=grad[0]; gz=grad[2];
    #gx=ndimage.filters.gaussian_filter(dens, [pxksz, pxksz, pxksz], order=[0,0,1], mode=mode)
    #gy=ndimage.filters.gaussian_filter(dens, [pxksz, pxksz, pxksz], order=[0,1,0], mode=mode)
    #gz=ndimage.filters.gaussian_filter(dens, [pxksz, pxksz, pxksz], order=[1,0,0], mode=mode)   

    gx = gaussian_filter(dens, [pxksz, pxksz, pxksz], order=[0,0,1], mode=mode)
    gy = gaussian_filter(dens, [pxksz, pxksz, pxksz], order=[0,1,0], mode=mode)
    gz = gaussian_filter(dens, [pxksz, pxksz, pxksz], order=[1,0,0], mode=mode)
    t0 = time.time()
    print("Step X done in", time.time()-t0, "sec")
    
    normgrad=np.sqrt(gx*gx+gy*gy+gz*gz)
    normb   =np.sqrt(Bx*Bx+By*By+Bz*Bz)
    t0 = time.time()
    print("Step normb done in", time.time()-t0, "sec")
    #zerograd=(normgrad==0.).nonzero()	
    #zerob   =(normb   ==0.).nonzero()
    #print("Step zerob done in", time.time()-t0, "sec")    
    cross=np.sqrt((gy*Bz-gz*By)**2+(gx*Bz-gz*Bx)**2+(gx*By-gy*Bx)**2)
    dot  =gx*Bx+gy*By+gz*Bz	
    print("Step dot done in", time.time()-t0, "sec")    
    cosphi=dot/(normgrad*normb)   
    cosphi[(normgrad == 0.).nonzero()]=np.nan
    cosphi[(normb    == 0.).nonzero()]=np.nan
    
    return cosphi


# ============================================================================================================
def equibins(dens, steps=10, mind=None):
   # Compute bins with equal number of voxels 
   #
   # INPUTS
   # dens       - input cube 
   # steps      - number of bins
   # mind       - minimun values of the input array  
   #
   # OUTPUTS
   # bincentres - 

   if (mind is None):
      mind=np.nanmin(dens)
 
   sz=np.size(dens)
   hist, bin_edges = np.histogram(dens[(dens > mind).nonzero()], bins=sz)
   bin_centre     =0.5*(bin_edges[0:np.size(bin_edges)-1]+bin_edges[1:np.size(bin_edges)])
   print('done bin_centre in equibins')    
   chist=np.cumsum(hist)
   pitch=np.max(chist)/float(steps)
   hsteps=pitch*np.arange(0,steps+1,1)
   dsteps=np.zeros(steps+1)
   print('start dsteps loop in equibins')
   for i in range(0, np.size(dsteps)-1):	                
      good=np.logical_and(chist>hsteps[i],chist<hsteps[i+1]).nonzero()
      dsteps[i]=np.min(bin_centre[good])

   dsteps[np.size(dsteps)-1]=np.max(dens)
   print('done dsteps in equibins')
   return dsteps

# ============================================================================================================
def roparameterhist(cosphi, hsize=15, ppwin=0.25, w=None):

   # Calculate the relative orientation parameter $\xi$, Eq. 13 in Soler et al. 2013
   #
   # INPUTS
   # cosphi     - relative orientation angles in 3D, its range should be [-1.0,1.0]
   # hsize      - size of the histogram used for the calculation of the relative orientation parameter       was 15
   # ppwin      - histogram range for the definitions of parallel (- ppwin < cos(phi) < ppwin) or  
   #              perpendicular (-1 < cos(phi) < -1.+ppwin && 1-ppwin < cos(phi) < 1.)
   # OUTPUTS
   # xi          - relative orientation parameter  

   if (w is None):
      w=np.ones_like(cosphi)
   else:
      assert np.shape(cosphi) == np.shape(w), "Dimensions of cosphi and w must match"  
  
   hist, bin_edges = np.histogram(cosphi, bins=hsize, range=(-1.,1.), weights=w)
   bin_centres=0.5*(bin_edges[0:np.size(bin_edges)-1]+bin_edges[1:np.size(bin_edges)])  
 
   xi, s_xi = roparameter(bin_centres, hist, ppwin=ppwin)
   print('done roparameterhist')
   return xi, s_xi

# ===================================================================================================
def roparameter(cosphi, hist, ppwin=0.25):
    # Calculate the relative orientation parameter $\xi$ as defined in Planck intermediate results. XXXV. A&A 586A (2016) 138P.
    #
    # INPUTS
    # cosphi      - histogram bin centres 
    # hist        - histogram counts
    # ppwin       - range for the definitions of parallel (0 < phi < s_phi and 180-s_phi < phi < 180) or 
    #               perpendicular (90-s_phi < phi < 90+s_phi)
    # OUTPUTS
    # xi          - relative orientation parameter
  
    perp=(np.abs(cosphi)>1.-ppwin).nonzero()
    para=(np.abs(cosphi)<ppwin).nonzero()
  
    temp=float(np.sum(hist[para])+np.sum(hist[perp]))
    if (temp > 0.):
       xi=float(np.sum(hist[para])-np.sum(hist[perp]))/float(np.sum(hist[para])+np.sum(hist[perp]))
       s_xi=2.*np.sqrt((np.sum(hist[para])*np.std(hist[perp]))**2+(np.sum(hist[perp])*np.std(hist[para]))**2)/(np.sum(hist[para])+np.sum(hist[perp]))**2
    else:
       xi=np.nan
       s_xi=np.nan
    print('done roparameter')
    return xi, s_xi

# ============================================================================================================
#def hro3D(dens, Bx, By, Bz, steps=10, dsteps=None, hsize=21, mind=None, outh=[0,4,9], pxksz=2.56, label=r'$n$', weights=None):
#def hro3D(dens, Bx, By, Bz, steps=10, dsteps=None, hsize=21, mind=None, outh=[0,4,9], pxksz=2.56, label=r'$n$', weights=None,savefile=None, make_plots=False):
def hro3D(dens, Bx, By, Bz, steps=10, hsize=21, mind=None, outh=[0,4,9], pxksz=2.56, label=r'$n$', weights=None,savefile=None, make_plots=False):

    # Calculate the relative orientation parameter $\xi$, Eq. 13 in Soler et al. 2013
   #
   # INPUTS
   # cosphi     - relative orientation angles in 3D, its range should be [-1.0,1.0]
   # hsize      - size of the histogram used for the calculation of the relative orientation parameter     was 21
   # ppwin      - histogram range for the definitions of parallel (- ppwin < cos(phi) < ppwin) or  
   #              perpendicular (-1 < cos(phi) < -1.+ppwin && 1-ppwin < cos(phi) < 1.)
   # OUTPUTS
   # xi          - relative orientation parameter 
   print('start')
   if (weights is None):
      # Account for the correlation introduced by the kernel size
      weights=(1./pxksz**2)*np.ones_like(dens)
   else:
      assert np.shape(dens) == np.shape(weights), "Dimensions of cosphi and w must match"

   cosphi = roangles3D(dens, Bx, By, Bz, pxksz=pxksz)
   print('done cosphi')
   dsteps = equibins(dens, steps=steps, mind=mind)
   print('done dsteps')
   hros   =np.zeros([steps,hsize])
   cdens  =np.zeros(steps)
   xi     =np.zeros(steps)
   s_xi   =np.zeros(steps)
   meancos=np.zeros(steps)
   scube = 0.*dens

   for i in tqdm(range(0, np.size(dsteps)-1)):
      good=np.logical_and(dens>dsteps[i],dens<dsteps[i+1]).nonzero()
      hist, bin_edges=np.histogram(cosphi[good], bins=hsize, range=(-1.,1.), weights=weights[good])	
      bin_centre=0.5*(bin_edges[0:np.size(bin_edges)-1]+bin_edges[1:np.size(bin_edges)])
      hros[i,:]=hist
      scube[good]=i
      cdens[i]=np.mean([dsteps[i],dsteps[i+1]])
      xi[i], s_xi[i]=roparameter(bin_centre, hist)
      meancos[i]=np.mean(weights[good]*cosphi[good]) 
   print('done meancos')
    
   if savefile is not None:
       np.savez(
           savefile,
           hros=hros,
           abins=cdens,
           xi=xi,
           s_xi=s_xi,
           meancos=meancos,
           dsteps=dsteps,
           bin_centre=bin_centre,
           hsize=hsize,
           pxksz=pxksz
       )
       
   if make_plots:
       outsteps = np.size(outh)
       color = iter(cm.cool(np.linspace(0, 1, outsteps)))

       fig = plt.figure(figsize=(6.0,6.0))
       plt.rc('font', size=10)
       ax1=plt.subplot(111)
       for i in range(0, outsteps):
          c = next(color)
          lower = manual_dsteps[outh[i]]
          upper = manual_dsteps[outh[i] + 1]
          labeltext = f"{lower:.2f} < n < {upper:.2f}"
          counts = hros[outh[i], :].astype(float)
          dx = bin_centre[1] - bin_centre[0]
          norm_counts = counts / (counts.sum() * dx)
          #print(counts.sum(), counts.min(), counts.max())
          ax1.plot(bin_centre, norm_counts, '-', linewidth=2, c=c, label=labeltext)
          #c = next(color)
          #labeltext = str(np.round(dsteps[outh[i]],2))+' < '+label+' < '+str(np.round(dsteps[outh[i]+1],2))
          #ax1.plot(bin_centre, hros[outh[i],:], '-', linewidth=2, c=c, label=labeltext)
       ax1.set_xlabel(r'cos($\phi$)')
       ax1.set_ylabel('Counts')
       ax1.tick_params(axis='y', labelrotation=90)
       ax1.legend()
       plt.grid()
       plt.tight_layout()
       #plt.savefig('/cosma8/data/dp058/dc-whit3/Lyon/LSD_model/grids/3070-HRO.png')
       plt.show()
       
       fig = plt.figure(figsize=(8.0,4.0))
       plt.rc('font', size=10)
       ax1=plt.subplot(111)
       ax1.semilogx(cdens, xi, 'o-', linewidth=2, color='blue')
       ax1.tick_params(axis='y', labelrotation=90)
       ax1.axhline(y=0., c='k', ls='--')
       ax1.set_xlabel(r'log$_{10}$ ($n_{\rm H}/$cm$^{-3}$)')
       ax1.set_ylabel(r'$\xi$')
       ax1.set_ylim(-0.2,0.5)
       plt.grid()
       plt.tight_layout()
       #plt.savefig('/cosma8/data/dp058/dc-whit3/Lyon/LSD_model/grids/3070-ROvsLogNH.png')
       plt.show()

       fig = plt.figure(figsize=(8.0,4.0))
       plt.rc('font', size=10)
       ax1=plt.subplot(111)
       ax1.semilogx(cdens, meancos, 'o-', linewidth=2, color='blue')
       ax1.tick_params(axis='y', labelrotation=90)
       ax1.axhline(y=0., c='k', ls='--')
       ax1.set_xlabel(r'log$_{10}$ ($n_{\rm H}/$cm$^{-3}$)')
       ax1.set_ylabel(r'$\left<\cos\theta\right>$')
       ax1.set_yticks([-0.005, -0.0025, 0, 0.0025, 0.005])
       plt.grid()
       plt.tight_layout()
       #plt.savefig('/cosma8/data/dp058/dc-whit3/Lyon/LSD_model/grids/3070-cosphivsLogNH.png')
       plt.show()
  
   #return hros, cdens, xi
   #return {'hros': hros, 'abins': cdens, 'xi': xi, 's_xi': s_xi, 'meancos': meancos}
   return {'hros': hros,'abins': cdens,'xi': xi,'s_xi': s_xi,'meancos': meancos,'dsteps': dsteps,'bin_centre': bin_centre}

# Testing the hro3D. should go in the test script or something.
#def main(args=None):
#
#    	if res.prnt:
#		print('Primes: {0}'.format(primes))
#
#from astropy.convolution import Gaussian2DKernel
#g2D=Gaussian2DKernel(10)
#sz=np.shape(g2D)
#dens=np.dstack([g2D]*sz[0])
#
#grad=np.gradient(dens, edge_order=2)
##Bx=grad[1]
##By=grad[0]
##Bz=grad[2]
#Bx=np.random.uniform(low=-1., high=1., size=np.shape(dens))
#By=np.random.uniform(low=-1., high=1., size=np.shape(dens))
##Bz=np.random.uniform(low=-1., high=1., size=np.shape(dens))
##Bx=0.*dens
##By=0.*dens
#Bz=0.*dens
#rho = ag[("PartType0","Density")].d.astype(np.float32)
#Bx  = ag[("PartType0","MagneticField_x_raw")].d.astype(np.float32)
#By  = ag[("PartType0","MagneticField_y_raw")].d.astype(np.float32)
#Bz  = ag[("PartType0","MagneticField_z_raw")].d.astype(np.float32)

with h5py.File("/cosma8/data/dp058/dc-whit3/Lyon/LSD_model/grids/test_cube_data_float32.hdf5", "r") as hf:
    yn_test  = hf["yn"][:]
    #rho = hf["rho"][:]
    Bx_test  = hf["Bx"][:]
    By_test  = hf["By"][:]
    Bz_test  = hf["Bz"][:]
print("Loaded data from file.")

#hros, cdens, zeta = hro3D(rho, Bx, By, Bz, mind=np.mean(rho))
manual_dsteps = np.array([0.1, 1, 10, 100, 500])#, 800])#, 5000])
#gy, gx, gz = np.gradient(yn_test)
outh = [0,4,9]
gy, gx, gz = np.gradient(yn_test)
print("About to call hro3D")
#results = hro3D(yn, Bx, By, Bz, mind=np.mean(yn), dsteps=manual_dsteps, outh=None)
#results = hro3D(yn_test, Bx_test, By_test, Bz_test, mind=np.mean(yn_test), dsteps=manual_dsteps, outh=None, savefile="hro_data_test.npz", make_plots=True)
#results = hro3D(yn_test, gy, gx, gz, mind=np.mean(yn_test), dsteps=manual_dsteps, outh=outh, savefile="hro_data_test.npz", make_plots=True)
results = hro3D(yn_test, gy, gx, gz, mind=np.mean(yn_test), outh=outh, savefile="hro_data_test.npz", make_plots=True)


hros   = results['hros']
cdens  = results['abins']
zeta   = results['xi']
bin_centre = results['bin_centre']
