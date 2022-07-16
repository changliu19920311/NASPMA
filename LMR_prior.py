"""
Module: LMR_prior.
"""
import numpy as np
from random import sample, seed
from copy import deepcopy

# -------------------------------------------------------------------------------
# *** Prior source assignment  --------------------------------------------------
# -------------------------------------------------------------------------------
# All logic for prior object assignment
def prior_assignment(iprior):
    if iprior == 'generic':
        prior_object = prior_generic()
    elif iprior == 'ccsm4_last_millenium':
        prior_object = prior_ccsm4_last_millenium()
    return prior_object


# -------------------------------------------------------------------------------
# *** Master class for model data as prior --------------------------------------
# -------------------------------------------------------------------------------
class prior_master(object):
    '''
    This is the master class for the prior data. Inherent to create classes for each prior source.
    '''

    # Populate the prior ensemble from gridded model/analysis data
    def populate_ensemble(self,prior_source, prior_cfg):

        # Load prior data from file(s) - multiple state variables
        self.read_prior()        
        
        Nens_max = len(self.prior_dict[list(self.prior_dict.keys())[0]]['years'])
        print('Nens_max',Nens_max)
        if self.Nens and self.Nens > Nens_max:
            raise SystemExit('ERROR in populate_ensemble! Specified ensemble size too large for available nb of states. '
            'Max allowed with current configuration: %d' %Nens_max)

        
        nbvars = len(self.statevars)
        # Check consistency between specified state variables and uploaded dictionary
        if len(list(self.prior_dict.keys())) != nbvars:
            raise SystemExit('Problem with load of prior state variables. Exiting!')

        
        # Defining content of state vector => dictionary: state_vect_content
        # NOTE: now assumes that dims of state variables are (lat,lon) only !!!
        state_vect_info = {}
        
        # Loop over state variables
        Nx = 0
        timedim = []
        for var in list(self.prior_dict.keys()):

            vartype = self.prior_dict[var]['vartype']

            dct = {}
            timedim.append(len(self.prior_dict[var]['years']))

            spacecoords = self.prior_dict[var]['spacecoords']

            if '2D' in vartype:

                dim1, dim2 = spacecoords

                # How are these defined? Check dims of arrays
                if len(self.prior_dict[var][dim1].shape) == 2 and len(self.prior_dict[var][dim2].shape) == 2:
                    # we have a field defined on an irregular lat/lon grid, requiring lat & lon
                    # each be defined with a 2d array
                    ndim1 = self.prior_dict[var][dim1].shape[0]
                    ndim2 = self.prior_dict[var][dim1].shape[1]
                    
                elif len(self.prior_dict[var][dim1].shape) == 1 and len(self.prior_dict[var][dim2].shape) == 1:
                    # regular lat/lon array : lat and lon can be defined with 1d arrays
                    ndim1 = len(self.prior_dict[var][dim1])
                    ndim2 = len(self.prior_dict[var][dim2])

                else:
                    raise SystemExit('ERROR in populate_ensemble: Unrecognized info on spatial dimensions. Exiting!')

                ndimtot = ndim1*ndim2
                
                dct['pos'] = (Nx,Nx+(ndimtot)-1)
                dct['spacecoords'] = spacecoords
                dct['spacedims'] = (ndim1,ndim2)

                if 'lat' in spacecoords and 'lon' in spacecoords:
                    dct['vartype'] = '2D:horizontal'
                elif 'lat' in spacecoords and 'lev' in spacecoords:
                    dct['vartype'] = '2D:meridional_vertical'
                else:
                    raise SystemExit('ERROR in populate_ensemble: Unrecognized info on spatial dimensions. Exiting!')
                
            elif vartype == '1D:meridional':
                dim1,  = spacecoords
                ndim1 = self.prior_dict[var][dim1].shape[0]
                ndimtot = ndim1
                dct['pos'] = (Nx,Nx+(ndimtot)-1)
                dct['spacecoords'] = spacecoords
                dct['spacedims'] = (ndim1,)
                dct['vartype'] = vartype
                
            else:
                # variable is simple time series'
                ndimtot = 1
                dct['pos'] = (Nx,Nx+(ndimtot)-1)
                dct['spacecoords'] = None
                dct['spacedims'] = None
                dct['vartype'] = '0D:time series'
                
            # assign to master dictionary
            state_vect_info[var] = dct

            # determining length of state vector
            Nx = Nx + (ndimtot)

        # Looped through all state variables, now a summary:
        print(' ')
        print('State vector information:')
        print('Nx =', Nx)
        print('state_vect_info=', state_vect_info)
        
        # time dimension consistent across variables?
        if all(x == timedim[0] for x in timedim):
            ntime = timedim[0]
        else:
            raise SystemExit('ERROR im populate_ensemble: time dimension not consistent across all state variables. Exiting!')
      

        # If Nens is None, use all of prior with no random sampling
        if self.Nens is None:
            take_sample = False
            self.Nens = ntime
        else:
            take_sample = True
        print('ntime',ntime)
        # Array that will contain the prior ensemble (state vector)
        Xb = np.zeros(shape=[Nx,self.Nens]) # no time dimension now...
        # ***NOTE: Following code assumes that data for a given year are located at same array time index across all state variables

        if take_sample:
            print('Random selection of', str(self.Nens), 'ensemble members')
            # Populate prior ensemble from randomly sampled states
            seed(prior_cfg.seed)
            ind_ens = sample(list(range(ntime)), self.Nens)
            print('seed',prior_cfg.seed)
            print('ind_ens',ind_ens)
        else:
            print('Using entire consecutive years in prior dataset.')
            ind_ens = list(range(ntime))

        self.prior_sample_indices = ind_ens

        # To keep spatial coords of gridpoints (needed geo. information)
        Xb_coords = np.empty(shape=[Nx,2]) # 2 is max nb of spatial dim a variable can take
        Xb_coords[:,:] = np.NAN # initialize with Nan's

        for var in list(self.prior_dict.keys()):

            vartype = self.prior_dict[var]['vartype']
            
            indstart = state_vect_info[var]['pos'][0]
            indend   = state_vect_info[var]['pos'][1]

            if '2D' in vartype:
                    # Loop over ensemble members
                    for i in range(0,self.Nens):
                        Xb[indstart:indend+1,i] = self.prior_dict[var]['value'][ind_ens[i],:,:].flatten()

                    # get the name of the spatial coordinates for state variable 'var'
                    coordname1, coordname2 = state_vect_info[var]['spacecoords']
                    # load in the coord values from data dictionary 
                    coord1 = self.prior_dict[var][coordname1]
                    coord2 = self.prior_dict[var][coordname2]

                    # check how coords are defined:
                    # 1d (regular lat/lon grid) or 2d (irregular lat/lon grid)
                    if len(coord1.shape) == 1 and len(coord2.shape) == 1:
                        ndim1 = coord1.shape[0]
                        ndim2 = coord2.shape[0]
                        X_coord1 =  np.array([coord1,]*ndim2).transpose()
                        X_coord2 =  np.array([coord2,]*ndim1)
                    elif len(coord1.shape) == 2 and len(coord2.shape) == 2:
                        ndim1, ndim2 = coord1.shape
                        X_coord1 =  coord1
                        X_coord2 =  coord2

                    Xb_coords[indstart:indend+1,0] = X_coord1.flatten()
                    Xb_coords[indstart:indend+1,1] = X_coord2.flatten()

                    # Some cleanup
                    del coord1
                    del coord2
                    del X_coord1
                    del X_coord2

            elif vartype == '1D:meridional':
                # Loop over ensemble members
                for i in range(0,self.Nens):
                    Xb[indstart:indend+1,i] = self.prior_dict[var]['value'][ind_ens[i],:].flatten()

                    # get the name of the spatial coordinate for state variable 'var'
                    coordname1, = state_vect_info[var]['spacecoords']
                    # load in the coord values from data dictionary 
                    X_coord1 = self.prior_dict[var][coordname1]

                    Xb_coords[indstart:indend+1,0] = X_coord1.flatten()

                    # Some cleanup
                    del X_coord1                    

            elif vartype == '0D:time series':
                # Loop over ensemble members
                for i in range(0,self.Nens):
                    Xb[indstart:indend+1,i] = self.prior_dict[var]['value'][ind_ens[i]].flatten()
                
            else:
                raise SystemExit('ERROR im populate_ensemble: variable of unrecognized spatial dimensions. Exiting!')



            """
            # RT dev ... ... ...
            # if anom_reference period is defined, re-centering sample around a mean of zero ? ...
            
            print self.prior_dict[var]['climo'].shape
            print self.prior_dict[var]['value'][ind_ens[0]].shape


            print np.mean(Xb[indstart:indend+1,:], axis=1)

            sample_mean = np.mean(Xb[indstart:indend+1,:], axis=1)
            
            Xb[indstart:indend+1,:] = Xb[indstart:indend+1,:] - sample_mean[:,np.newaxis]

            print np.mean(Xb[indstart:indend+1,:], axis=1)
            
            exit(1)
            # RT dev ... ... ...
            """
        
        # Returning state vector Xb as masked array, if it contains
        # at least one invalid value

        if np.any(np.isnan(Xb)):
            # Returning state vector Xb as masked array
            Xb_res = np.ma.masked_invalid(Xb)

            # Set fill_value to np.nan
            np.ma.set_fill_value(Xb_res, np.nan)
        
            # array indices of masked & valid elements
            inds_mask = np.nonzero(Xb_res.mask)
            inds_valid = np.nonzero(~Xb_res.mask)
        else:
            Xb_res = Xb
        
        # Assign return variables
        self.ens = Xb_res
        self.coords = Xb_coords
        self.full_state_info = state_vect_info
        
        return


# -------------------------------------------------------------------------------
# Classes for specific model/simulation -----------------------------------------
# -------------------------------------------------------------------------------


# class for generic object
class prior_generic(prior_master):
    pass

# class for GISTEMP gridded surface temperature dataset
class prior_gistemp(prior_master):
    pass

# class for BerkeleyEarth gridded surface temperature dataset
class prior_BerkeleyEarth(prior_master):
    pass

# class for the CCSM4 Last Millennium simulation
class prior_ccsm4_last_millenium(prior_master):

    def read_prior(self):
    
        from load_gridded_data import read_gridded_data_CMIP5_model
        self.prior_dict = read_gridded_data_CMIP5_model(self.prior_datadir,
                                                        self.prior_datafile,
                                                        self.statevars,
                                                        self.avgInterval,
                                                        self.detrend,
                                                        self.anom_reference,
                                                        self.statevars_info)
        return

