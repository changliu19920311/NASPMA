"""
Copy of LMRlite.ipynb as an example of pseudoproxy experiments.
The Pseudo proxy experiment
1. read configuration
2. read proxies
3. read the sample poor for truth vector and prior
4. build the noise and pseudoproxies
5. assimilate
"""


# put the directory path to your LMR repository here
import sys
sys.path.append("/Users/jjyan/Desktop/PPE_NA_1")

# import the necessary modules for PPE
import LMR_lite_utils as LMRlite
import LMR_utils
import LMR_config
import numpy as np
import os,sys
from time import time
from random import sample, seed, choices
import importlib
importlib.reload(LMRlite)
import netCDF4
import itertools
import random

def red_noise_generator(truth_vector,SNR,mean,ntimes,ranseed):
    """
    purporse: producing the red noise based on the truth vector
    
    Input:
        truth_vector: the truth value
        SNR: signal to noise value
        mean: the mean value for producing white noise
        ranseed(optional): seed the random number generator for repeatability
    
    Output: 
        white_noise: the derived white noise
    """
    #var_truth_vector: the variance of the truth vector on time (nlat*nlon)
    var_truth_vector=np.var(truth_vector, axis=1)
    print(var_truth_vector.shape)

    #var_noise: the variance of the white noise (nlat*nlon)
    var_noise=var_truth_vector/(SNR**2)
    print(var_noise.shape)

    std = np.sqrt(var_noise) 
    #white_noise: the white noise added to the truth vector (nlatnlon,Ny)
    red_noise= np.copy(truth_vector) 
    auto_cor=0.32
    if ranseed != None:
        np.random.seed(ranseed)
    for t in range(ntimes):
        for i in range(len(std)):
            if t == 0:
                red_noise[i,t] = np.random.normal(mean, std[i], size=1)
            else:
                red_noise[i,t] =auto_cor*red_noise[i,t-1] + std[i]*np.random.normal(loc =0, scale=1, size=1)*np.sqrt(1-auto_cor) 
    return red_noise,var_noise


def white_noise_generator(truth_vector,SNR,mean,ntimes,ranseed):
    """
    purporse: producing the white noise based on the truth vector
    
    Input:
        truth_vector: the truth value
        SNR: signal to noise value
        mean: the mean value for producing white noise
        ranseed(optional): seed the random number generator for repeatability
    
    Output: 
        white_noise: the derived white noise
    """
    #var_truth_vector: the variance of the truth vector on time (nlat*nlon)
    var_truth_vector=np.var(truth_vector, axis=1)
    print(var_truth_vector.shape)

    #var_noise: the variance of the white noise (nlat*nlon)
    var_noise=var_truth_vector/(SNR**2)
    print(var_noise.shape)

    std = np.sqrt(var_noise) 
    #white_noise: the white noise added to the truth vector (nlatnlon,Ny)
    white_noise= np.copy(truth_vector) 
    
    if ranseed != None:
        np.random.seed(ranseed)
        
    for i in range(len(std)):
        white_noise[i,:] = np.random.normal(mean, std[i], size=ntimes)
    
    return white_noise,var_noise


def out_netcdf_data(filename,grid,ntimes,prior,recon_times,Xam_pp_ensemble,truth_latlon,Xbm_pp_ensemble):
    """
    purporse: output the data to the netcdf files
    
    Input:
    filename: the name of output file
    grid: provided the latitude and longitude information
    ntimes: the number of reconstructed time
    recon_times: reconstructed time
    Xam_pp_latlon: analysis data
    truth_latlon: the truth vector
    Xbm_pp_latlon: the prior state 
    
    Output: 
        the netcdf file 
    """
    if os.path.exists(filename):
        os.remove(filename)
    ncfile = netCDF4.Dataset(filename,mode='w',format='NETCDF4_CLASSIC') 
    nlat=grid.nlat
    nlon=grid.nlon
    nite = len(prior)
    ite_dim = ncfile.createDimension('ite', nite) # latitude axis
    lat_dim = ncfile.createDimension('lat', nlat) # latitude axis
    lon_dim = ncfile.createDimension('lon', nlon) # longitude axis
    time_dim = ncfile.createDimension('time', ntimes) # unlimited axis (can be appended to).

    ncfile.title='The results of Pseudo Peoxy Experiments'
    ite = ncfile.createVariable('ite', np.float32, ('ite',))
    ite.long_name = 'prior'
    lat = ncfile.createVariable('lat', np.float32, ('lat',))
    lat.units = 'degrees_north'
    lat.long_name = 'latitude'
    lon = ncfile.createVariable('lon', np.float32, ('lon',))
    lon.units = 'degrees_east'
    lon.long_name = 'longitude'
    time = ncfile.createVariable('time', np.float64, ('time',))
    time.units = 'Year'
    time.long_name = 'time'
    pr_sfc_Amon_analysis = ncfile.createVariable('pr_sfc_Amon_analysis',np.float64,('ite','lat','lon','time')) # note: unlimited dimension is leftmost
    pr_sfc_Amon_analysis.units = 'C' # degrees Kelvin
    pr_sfc_Amon_analysis.standard_name = 'analysis_surface_precipitatio_anomaly' # this is a CF standard name
    pr_sfc_Amon_truth = ncfile.createVariable('pr_sfc_Amon_truth',np.float64,('lat','lon','time')) # note: unlimited dimension is leftmost
    pr_sfc_Amon_truth.units = 'C' # degrees Kelvin
    pr_sfc_Amon_truth.standard_name = 'truth_surface_precipitation_anomaly' # this is a CF standard name
    pr_sfc_Amon_prior = ncfile.createVariable('pr_sfc_Amon_prior',np.float64,('ite','lat','lon')) # note: unlimited dimension is leftmost
    pr_sfc_Amon_prior.units = 'C' # degrees Kelvin
    pr_sfc_Amon_prior.standard_name = 'prior_surface_precipitatio_anomaly' # this is a CF standard name
    ite[:] = prior
    lat[:] = grid.lat[:,0]
    lon[:] = grid.lon[0,:]
    time[:] = recon_times
    pr_sfc_Amon_analysis[:,:,:,:] = Xam_pp_ensemble # Appends data along unlimited dimension
    pr_sfc_Amon_truth[:,:,:] = truth_latlon
    pr_sfc_Amon_prior[:,:,:] = Xbm_pp_ensemble
    ncfile.close()

def analysis_state_generator(cfg,prox_manager,grid,nproxies,ntimes,nens,Xb_one_new,truth_latlon,white_noise_latlon,Xb_sampler,var_noise_latlon):
    """
    Purpose: assimilating the psudoproxy
    
    Inputs:
           cfg: the configure for PPE
           prox_manager: contain the proxy information
           grid: contain the latitude and longitude information about prior
           nproxies:  the number of proxy
           ntimes: the reconstructed time
           nens: the number of the members for prior
           Xb_one_new: the prior state
           truth_latlon: the true vector
           noise_latlon: the noise
           Xb_sampler: the sample to retrive 
           var_noise_latlon: the variation of noise
    Output:
           Xam_pp: the analysis state (nlat*nlon,Ny)
          """
    """Step 6 calculationg the pseudoproxy,the estimate of proxy, and the error of the pseudoproxy
   vY: the pseudoproxy (np,Ny)
   vYe: the estimate of the proxy from the prior state (np,Nm)
   vR:  the error of the psudoproxy (np)
   Ye_assim_coords： the coordinated information about vYe (np,2)
   Xb_one_coords : the coordinated information about prior (nlat*nlon,2)
   Xam_pp: he analysis state (nlat*nlon,Ny)
   Xap_pp: the perturbation of analysis state (nlat*nlon,Nm)
   loc : localization
    """

    # loop over proxies
    vY = np.zeros([nproxies,ntimes])      
    vYe = np.zeros([nproxies,nens])
    vR = [nproxies]
    k = -1
    Ye_assim_coords= np.zeros([nproxies,2])
    print("nproxies",nproxies)
    # options to filter to a single proxy group, or single proxy
    prox_type='Seasonal tree ring'
    locRad = cfg.core.loc_rad
    Xb_one_coords = np.zeros(shape=[grid.nlat*grid.nlon, 2])
    Xb_one_coords[:, 0] = grid.lat.flatten()
    Xb_one_coords[:, 1] = grid.lon.flatten()

    for proxy_idx, Y in enumerate(prox_manager.sites_assim_proxy_objs()):
        k=k+1
        tmp = grid.lat[:,0]-Y.lat
        itlat = np.argmin(np.abs(tmp))
        tmp = grid.lon[0,:]-Y.lon
        itlon = np.argmin(np.abs(tmp))
        pseudoproxy = (truth_latlon[itlat,itlon,:] + white_noise_latlon[itlat,itlon,:]).tolist()
        vY[k,:]=pseudoproxy
        vR.append(var_noise_latlon[itlat,itlon])
        
        tmp = grid.lat[:,0]-Y.lat
        itlat = np.argmin(np.abs(tmp))
        tmp = grid.lon[0,:]-Y.lon
        itlon = np.argmin(np.abs(tmp))
        vYe[k,:] = Xb_sampler[itlat,itlon,:]
        Ye_assim_coords[k, :] = np.asarray([grid.lat[itlat,0], grid.lon[0,itlon]], dtype=np.float64)

    core = cfg.core
    Xam_pp=np.zeros([grid.nlat*grid.nlon,ntimes]) 
    # -------------------------------------
    # Loop over years of the reconstruction
    # -------------------------------------
    for yr_idx, t in enumerate(range(core.recon_period[0], core.recon_period[1]+1, core.recon_timescale)):
            print(yr_idx)
            start_yr = int(t-core.recon_timescale//2)
            end_yr = int(t+core.recon_timescale//2)
            if start_yr == end_yr:
                time_str = 'year: '+str(t)
            else:
                time_str = 'time period (yrs): ['+str(start_yr)+','+str(end_yr)+']'
            print('\n==== Working on ' + time_str)
        
            index=[]  # index contains the index of proxies that having values on the reconstructed year
            
            for proxy_idx, Y in enumerate(prox_manager.sites_assim_proxy_objs()):
                if core.recon_timescale > 1:
                    if (Y.start_yr > start_yr) & (Y.end_yr <= end_yr):
                        index.append(proxy_idx)
                else:
                    if (Y.start_yr >= start_yr) & (Y.end_yr <= end_yr):
                        index.append(proxy_idx)

            index=np.array(index)
            vR = np.array(vR)
            # assimilating
            locRad = cfg.core.loc_rad
            loc = cov_localization(locRad, Y, np.append(Xb_one_coords, Ye_assim_coords[index,:], axis=0))
            xam_pp,xap_pp= Kalman_ESRF(cfg,vY[index,yr_idx],vR[index],vYe[index,:],Xb_one_new,Ye_assim_coords,loc)
            Xam_pp[:,yr_idx]=xam_pp
           # filename = core.datadir_output+truth_var+"_"+cfile+"_"+str(i)+"_"+str(t)+".nc"
           # out_Xap_netcdf_data(filename,grid,xap_pp)
            
    return Xam_pp


def Kalman_ESRF(cfg,vY,vR,vYe,Xb_in, Ye_assim_coords,loc):
   
    """
    Purpose: applying the ESRF method to every reconstructed time
    
    Inputs:
           cfg: the configure for PPE
           vY: the pseudoproxy (np,Ny)
           vYe: the estimate of the proxy from the prior state (np,Nm)
           vR:  the error of the psudoproxy (np)
           Ye_assim_coords： the coordinated information about vYe (np,2)
           Xb_in: the prior
           loc: the localization

    Output:
           Xam: the analysis state mean (nlat*nlon)
           Xap: the analysis state perturbation (nlat*nlon,Ny)
          """ # number of state variables
    nx = Xb_in.shape[0]
    # augmented state vector with Ye appended
    Xb = np.append(Xb_in, vYe, axis=0)


    nobs = len(vY)
    for k in range(nobs):
        #if np.mod(k,100)==0: 

        obvalue = vY[k]
        ob_err = vR[k]
        Ye = Xb[nx+k,:]
        Xa,Xap= enkf_update_array(Xb, obvalue, Ye, ob_err, loc, inflate=None)   
        Xb = Xa
      
    # ensemble mean and perturbations
    xam = Xa[0:nx,:].mean(axis=1)
    return xam,Xap[0:nx,:]

def cov_localization(locRad, Y, X_coords):
    """

    Originator: R. Tardif, 
                Dept. Atmos. Sciences, Univ. of Washington
    -----------------------------------------------------------------
     Inputs:
        locRad : Localization radius (distance in km beyond which cov are forced to zero)
             Y : Proxy object, needed to get ob site lat/lon (to calculate distances w.r.t. grid pts
      X_coords : Array containing geographic location information of state vector elements

     Output:
        covLoc : Localization vector (weights) applied to ensemble covariance estimates.
                 Dims = (Nx x 1), with Nx the dimension of the state vector.

     Note: Uses the Gaspari-Cohn localization function.

    """

    # declare the localization array, filled with ones to start with (as in no localization)
    stateVectDim, nbdimcoord = X_coords.shape
    print("X_coords",X_coords)
    covLoc = np.ones(shape=[stateVectDim],dtype=np.float64)

    # Mask to identify elements of state vector that are "localizeable"
    # i.e. fields with (lat,lon)
    localizeable = covLoc == 1. # Initialize as True
    
    # array of distances between state vector elements & proxy site
    # initialized as zeros: this is important!
    dists = np.zeros(shape=[stateVectDim])

    # geographic location of proxy site
    site_lat = Y.lat
    site_lon = Y.lon
    # geographic locations of elements of state vector
    X_lon = X_coords[:,1]
    X_lat = X_coords[:,0]

    # calculate distances for elements tagged as "localizeable". 
    dists[localizeable] = np.array(LMR_utils.haversine(site_lon, site_lat,
                                                       X_lon[localizeable],
                                                       X_lat[localizeable]),dtype=np.float64)

    # those not "localizeable" are assigned with a disdtance of "nan"
    # so these elements will not be included in the indexing
    # according to distances (see below)
    dists[~localizeable] = np.nan
    
    # Some transformation to variables used in calculating localization weights
    hlr = 0.5*locRad; # work with half the localization radius
    r = dists/hlr;
    
    # indexing w.r.t. distances
    ind_inner = np.where(dists <= hlr)    # closest
    ind_outer = np.where(dists >  hlr)    # close
    ind_out   = np.where(dists >  2.*hlr) # out

    # Gaspari-Cohn function
    # for pts within 1/2 of localization radius
    covLoc[ind_inner] = (((-0.25*r[ind_inner]+0.5)*r[ind_inner]+0.625)* \
                         r[ind_inner]-(5.0/3.0))*(r[ind_inner]**2)+1.0
    # for pts between 1/2 and one localization radius
    covLoc[ind_outer] = ((((r[ind_outer]/12. - 0.5) * r[ind_outer] + 0.625) * \
                          r[ind_outer] + 5.0/3.0) * r[ind_outer] - 5.0) * \
                          r[ind_outer] + 4.0 - 2.0/(3.0*r[ind_outer])
    # Impose zero for pts outside of localization radius
    covLoc[ind_out] = 0.0

    # prevent negative values: calc. above may produce tiny negative
    # values for distances very near the localization radius
    # TODO: revisit calculations to minimize round-off errors
    covLoc[covLoc < 0.0] = 0.0    
    return covLoc

def enkf_update_array(Xb, obvalue, Ye, ob_err, loc=None, inflate=None):
    """
    Function to do the ensemble square-root filter (EnSRF) update
    (ref: Whitaker and Hamill, Mon. Wea. Rev., 2002)

    Originator: G. J. Hakim, with code borrowed from L. Madaus
                Dept. Atmos. Sciences, Univ. of Washington

    Revisions:

    1 September 2017: 
                    - changed varye = np.var(Ye) to varye = np.var(Ye,ddof=1) 
                    for an unbiased calculation of the variance. 
                    (G. Hakim - U. Washington)
    
    -----------------------------------------------------------------
     Inputs:
          Xb: background ensemble estimates of state (Nx x Nens) 
     obvalue: proxy value
          Ye: background ensemble estimate of the proxy (Nens x 1)
      ob_err: proxy error variance
         loc: localization vector (Nx x 1) [optional]
     inflate: scalar inflation factor [optional]
    """

    # Get ensemble size from passed array: Xb has dims [state vect.,ens. members]
    Nens = Xb.shape[1]
    # ensemble mean background and perturbations
    xbm = np.mean(Xb,axis=1)
    Xbp = np.subtract(Xb,xbm[:,None])  # "None" means replicate in this dimension

    # ensemble mean and variance of the background estimate of the proxy 
    mye   = np.mean(Ye)
    varye = np.var(Ye,ddof=1)

    # lowercase ye has ensemble-mean removed 
    ye = np.subtract(Ye, mye)

    # innovation
    try:
        innov = obvalue - mye
    except:
        print('innovation error. obvalue = ' + str(obvalue) + ' mye = ' + str(mye))
        print('returning Xb unchanged...')
        return Xb
    
    # innovation variance (denominator of serial Kalman gain)
    kdenom = (varye + ob_err)

    # numerator of serial Kalman gain (cov(x,Hx))
    kcov = np.dot(Xbp,np.transpose(ye)) / (Nens-1)
    # Option to inflate the covariances by a certain factor
    #if inflate is not None:
    #    kcov = inflate * kcov # This implementation is not correct. To be revised later.

    # Option to localize the gain
    if loc is not None:
        kcov = np.multiply(kcov,loc) 
   
    # Kalman gain
    kmat = np.divide(kcov, kdenom)

    # update ensemble mean
    xam = xbm + np.multiply(kmat,innov)

    # update the ensemble members using the square-root approach
    beta = 1./(1. + np.sqrt(ob_err/(varye+ob_err)))
    kmat = np.multiply(beta,kmat)
    ye   = np.array(ye)[np.newaxis]
    kmat = np.array(kmat)[np.newaxis]
    Xap  = Xbp - np.dot(kmat.T, ye)

    # full state
    Xa = np.add(xam[:,None], Xap)

    # if masked array, making sure that fill_value = nan in the new array 
    if np.ma.isMaskedArray(Xa): np.ma.set_fill_value(Xa, np.nan)

    
    # Return the full state
    return Xa,Xap


"""The main code started here"""
"""set pseudoproxy parameters here"""

# set variable that will be sampled for pseudoproxies
truth_var = 'pr_sfc_Amon'

# set the signal to noise value
SNR=0.5

# set seed for random
prior=[5,10,20,50,100,200,300,500] 

# set the type of noise 
noise= 'white_noise' # red_noise or white_noise

# the number of samples that used to drive true vector
nsample = 100  

'''Step1 read configuration'''
cfile = 'config_season_GPCC_WARM_prior_number.yml'

'''Iter'''
iter = 20

""" end the parameter setting"""
yaml_file = os.path.join(LMR_config.SRC_DIR,cfile)
cfg = LMRlite.load_config(yaml_file)

'''Step2 read proxies''' 
prox_manager = LMRlite.load_proxies(cfg)
nproxies = len(prox_manager.ind_assim)

'''Step3 read the sample poor for truth vector and prior''' 
iter_range = LMR_config.wrapper.iter_range
MCiters = range(iter_range[0], iter_range[1]+1)
param_iterables = [MCiters]
nite=len(prior)

#loading 200 memebers as the sample poor for truth vector and prior
X, Xb_one = LMRlite.load_prior(cfg)#Xb_one: the vector selected from model (Nlat*Nlon*Nvar,Nm)
Xbp = Xb_one - Xb_one.mean(axis=1,keepdims=True)#Xbp: the anomaly of Xb_one (Nlat*Nlon*Nvar,Nm)


# check if config is set to regrid the prior
if cfg.prior.regrid_method:
    print('regridding prior...')
    # this function over-writes X, even if return is given a different name
    [X,Xb_one_new] = LMRlite.prior_regrid(cfg,X,Xb_one,verbose=False)
else:
    X.trunc_state_info = X.full_state_info
    
Xb_one = Xb_one_new # Xb_one: applying regridding method to previous Xb_one to reduce the size. (nlat*nlon,Nm)
grid = LMRlite.Grid(X)
p = np.random.permutation(Xb_one.shape[1])
# From the sample poor, selecting the first 100 members as the truth vector and the rest 100 memebers 
# as the prior ensemble state vector
truth_vector_sample = Xb_one[:,p[:nsample]]# the sample poor to retrive truth vector (nlat*nlon,nsample)
# resampling the 100 true vector to the reconstruction years

core = cfg.core
recon_times = np.arange(core.recon_period[0], core.recon_period[1]+1,core.recon_timescale)
ntimes, = recon_times.shape
ind_ens = choices(list(range(nsample)), k=ntimes)   
truth_vector= truth_vector_sample[:,ind_ens] #the truth vector (nlat*nlon,Ny)
    #Step5 build the white noise
mean = 0

if noise == 'white_noise':
    add_noise,var_noise = white_noise_generator(truth_vector,SNR,mean,ntimes,5)
elif noise == 'red_noise':
    add_noise,var_noise = red_noise_generator(truth_vector,SNR,mean,ntimes,5)
else:
    print('The type of noise cannot be found. reset the parameter of noise ... Exiting!')
    raise SystemExit(1)
 # convert the resolution to the lat/lon grid
truth_latlon = np.reshape(truth_vector,[grid.nlat,grid.nlon,ntimes]) #convert the resolution of truth_vector to the 
                                                                    #grid of lat lon (nlat,nlon,Ny)
add_noise_latlon =  np.reshape(add_noise,[grid.nlat,grid.nlon,ntimes])#convert the resolution of white_noise to the
                                                                   #grid of lat lon (nlat,nlon,Ny)
var_noise_latlon=np.reshape(var_noise,[grid.nlat,grid.nlon])#convert the resolution of var_noise to the 
                                                            #grid of lat lon (nlat,nlon)
print(truth_latlon.shape)
nproxies = len(prox_manager.ind_assim)
print(nproxies)
Xam_pp_ensemble=np.zeros([nite,grid.nlat,grid.nlon,ntimes])
Xbm_pp_ensemble=np.zeros([nite,grid.nlat,grid.nlon])
for iter_num in range(0,iter):
    print('iter_num',iter_num)
    itr_str = 'r{:d}'.format(iter_num)       
    p_2 = np.random.permutation(p[nsample:])
    for i in range(len(prior)):
        Xb_one_new = Xb_one[:,p_2[:prior[i]]] # the priors. it is required that the truth_vector and 
                                                      # the priors doesn't have the same timeseries
                                                      # the prior state (nlat*nlon,Nm-nsample)
        Xbm_pp = Xb_one_new.mean(axis=1) #the mean prior state of Nm-nsample memebers (nlat*nlon)
        Xbm_pp_latlon=np.reshape(Xbm_pp,[grid.nlat,grid.nlon]) #the mean prior state of Nm-nsample memebers (nalt,nlon)

        # the sample poor to retrive vYe
        Xb_sampler = np.reshape(Xb_one_new,[grid.nlat,grid.nlon,prior[i]])
        # assimilating 
        nens=grid.nens-nsample
        Xam_pp= analysis_state_generator(cfg,prox_manager,grid,nproxies,ntimes,prior[i],Xb_one_new,truth_latlon,add_noise_latlon,Xb_sampler,var_noise_latlon)#the analysis state (nlat*nlon,Ny)   
        Xam_pp_latlon=np.reshape(Xam_pp,[grid.nlat,grid.nlon,ntimes]) #the analysis sta at the lat/lon grid (nlat,nlon,Ny)
        print(Xam_pp_latlon.shape)
        Xam_pp_ensemble[i,:,:,:]=Xam_pp_latlon
        Xbm_pp_ensemble[i,:,:]=Xbm_pp_latlon
    filename=core.datadir_output+truth_var+"_"+cfile+"_"+itr_str+".nc"
    out_netcdf_data(filename,grid,ntimes,prior,recon_times,Xam_pp_ensemble,truth_latlon,Xbm_pp_ensemble)




