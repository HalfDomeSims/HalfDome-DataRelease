import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from nbodykit.lab import BigFileCatalog
from nbodykit.transform import ConcatenateSources, CartesianToEquatorial
import healpy as hp
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
matplotlib.rcParams.update({'font.size':20})

from mpi4py import MPI
root = 0
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# FIXME make a_i, a_f, res, m_or_h free params.
# FIXME the offsets are wrong. Either make parallel more like we do for healpix maps, or save the corrected offsets now. https://github.com/atokiwaipmu/LensingSSC/blob/c3af7fd3e5454b38ed850c2cdd72ba4ae3ff1f60/src/kappa_preprocessor.py#L13
a_i,a_f=0.5,0.51

catname = ['/global/cfs/cdirs/cmb/data/halfdome/stampede2_3750Mpch_6144cube/final_res/3750Mpc_2048node_seed_100/rfof_proc131072_nc6144_size3750_nsteps60lin_ldr0_rcvfalse_fstnone_pnf2_lnf2_s100_dhf1.0000_tiled0.20_fll_elllim_10000_npix_8192_rfofkdt_8/usmesh']

def read_range(cat, amin, amax, length_cut=None):
    """ Read a portion of the lightcone between two red shift ranges
        The lightcone from FastPM is sorted in Aemit and an index is built.
        So we make use of that.
        CrowCanyon is z > 0; We paste the mirror image to form a full sky.
    """
    edges = cat.attrs['aemitIndex.edges']    # FIXME this is different to the aemitIndex for the Healpix cats. Are these bugged?
    offsets = cat.attrs['aemitIndex.offset']
    start, end = edges.searchsorted([amin, amax])
    if cat.comm.rank == 0:
        cat.logger.info("Range of index is %d to %d" %(( start + 1, end + 1)))
    start = offsets[start + 1]
    end = offsets[end + 1]
    cat =  cat.query_range(start, end)
    if cat.csize > 0:
        if length_cut is not None:
            length = cat['Length'].compute()
            ra, dec = CartesianToEquatorial(cat['Position'][length>320], frame='icrs')
        else:
            ra, dec = CartesianToEquatorial(cat['Position'], frame='icrs')
    else:
        ra = 0
        dec = 0
    return ra, dec


"""This reads the map with and prepare a healpix map"""
for m_or_h in ['h']:
    if m_or_h == 'h':
        length_cut = 320
        cat = BigFileCatalog(catname,dataset='RFOF', comm=comm)
    else:
        length_cut = None
        cat = BigFileCatalog(catname,dataset='1', comm=comm)
        
    ra, dec = read_range(cat,a_i,a_f, length_cut)
    Ra = ra.compute() #read the right ascension of lightcone
    Dec = dec.compute() #read the declination of lightcone
    theta = 0.5 * np.pi - np.deg2rad(Dec) #from declination to theta
    phi = np.deg2rad(Ra) #from right ascension to phi
    NSIDE = 64
    pix_num = hp.pixelfunc.ang2pix(NSIDE, theta, phi) #obtain the pixel number for a certain {theta,phi}
    NPIX = hp.nside2npix(NSIDE) #total pixel number based on NSIDE
    
    fig = plt.subplots(figsize=(12,7),dpi=150)
    [num_lc, bins_lc, patches_lc] = plt.hist(pix_num,range=(0,NPIX),bins=NPIX) #number count of pixel number

    num_lc = comm.reduce(num_lc, op=MPI.SUM, root=root)

    if rank == root:
        dir_data = '/pscratch/sd/a/abayer/fastpm/halfdome/maps/'
        os.makedirs(dir_data, exist_ok=1)
        
        halo1 = num_lc/num_lc.mean()-1
        
        #fig, ax= plt.subplots(1,1,figsize=(10, 5), dpi=300,facecolor='w')
        plt.figure()
        if m_or_h == 'h': title = 'Halo Density'
        else: title = 'Matter Density (subsampled)'
        hp.mollview(halo1,title=title,cmap='RdBu_r',nest=False,hold=True,min=-0.1,max=0.1) #$\delta_h$
        plt.savefig(dir_data+'1map_%s_ai%.2f.png'%(m_or_h,a_i))
        np.save(dir_data+'1map_%s_ai%.2f.npy'%(m_or_h,a_i), halo1)
        
        halo1_smoo = hp.sphtfunc.smoothing(halo1,fwhm=0.1,nest=False)
        
        #fig, ax= plt.subplots(1,1,figsize=(10, 5), dpi=300,facecolor='w')
        plt.figure()
        hp.mollview(halo1_smoo,title=title,cmap='RdBu_r',nest=False,hold=True,min=-0.1,max=0.1) #$\delta_h$
        plt.savefig(dir_data+'1map_%s_smooth_ai%.2f.png'%(m_or_h,a_i))
        np.save(dir_data+'1map_%s_smooth_ai%.2f.npy'%(m_or_h,a_i), halo1_smoo)
