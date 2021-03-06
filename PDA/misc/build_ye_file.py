import sys
import numpy as np
import os
import yaml
import timeit

sys.path.append('../')

import LMR_prior
import LMR_proxy_pandas_rework
import LMR_config
from LMR_utils import create_precalc_ye_filename



if not LMR_config.LEGACY_CONFIG:
    if len(sys.argv) > 1:
        yaml_file = sys.argv[1]
    else:
        yaml_file = os.path.join(LMR_config.SRC_DIR, 'config_cool_season_pre.yml')

    try:
        print('Loading configuration: {}'.format(yaml_file))
        f = open(yaml_file, 'r')
        yml_dict = yaml.load(f)
        update_result = LMR_config.update_config_class_yaml(yml_dict,
                                                            LMR_config)

        # Check that all yml params match value in LMR_config
        if update_result:
            raise SystemExit(
                'Extra or mismatching values found in the configuration yaml'
                ' file.  Please fix or remove them.\n  Residual parameters:\n '
                '{}'.format(update_result))

    except IOError as e:
        raise SystemExit(
            ('Could not locate {}.  If use of legacy LMR_config usage is '
             'desired then please change LEGACY_CONFIG to True'
             'in LMR_wrapper.py.').format(yaml_file))

cfg = LMR_config.Config()

masterstarttime = timeit.default_timer()


print ('Starting Ye precalculation using prior data from '
       '{}'.format(cfg.prior.prior_source))


#  Load the proxy information
cfg.psm.linear.psm_r_crit = 0.0

# from LMR_config, figure out all the psm types the user wants to use
proxy_database = cfg.proxies.use_from[0]
proxy_class = LMR_proxy_pandas_rework.get_proxy_class(proxy_database)
print('proxy_database',proxy_database)
if proxy_database == 'LMRdb':
    proxy_cfg = cfg.proxies.LMRdb
else:
    print('ERROR in specification of proxy database.')
    raise SystemExit()

proxy_types = proxy_cfg.proxy_psm_type.keys()
psm_keys = [proxy_cfg.proxy_psm_type[p] for p in proxy_types]
unique_psm_keys = list(set(psm_keys))
# A quick check of availability of calibrated PSMs

if 'linear' in unique_psm_keys:
    if not os.path.exists(cfg.psm.linear.pre_calib_datafile):
        print ('*** linear PSM: Cannot find file of pre-calibrated PSMs:'
               ' \n {}'.format(cfg.psm.linear.pre_calib_datafile))
        print ('Perform calibration "on-the-fly" and calculate Ye values?'
               ' \nThis will take longer and PSM calibration parameters will not be stored in a file...')
        userinput = input('Continue (y/n)? ')
        if userinput == 'y' or userinput == 'yes':
            print('ok...continuing...')
        else:
            print('Exiting! Use the PSMbuild facility to generate the appropriate calibrated PSMs')
            raise SystemExit
        
if 'bilinear' in unique_psm_keys:
    if not os.path.exists(cfg.psm.bilinear.pre_calib_datafile):
        print ('*** bilinear PSM: Cannot find file of pre-calibrated PSMs:'
               ' \n {}'.format(cfg.psm.bilinear.pre_calib_datafile))
        print ('Perform calibration "on-the-fly" and calculate Ye values?'
               ' \nThis will take longer and PSM calibration parameters will not be stored in a file...')
        userinput = input('Continue (y/n)? ')
        if userinput == 'y' or userinput == 'yes':
            print('ok...continuing...')
        else:
            print('Exiting! Use the PSMbuild facility to generate the appropriate calibrated PSMs')
            raise SystemExit
# Finished checking ...

# Loop over all psm types found in the configuration
for psm_key in unique_psm_keys:

    print('Loading psm information for psm type: {} ...'.format(psm_key))
    
    # re-assign current psm type to all proxy records
    # TODO: Could think of implementing filter to restrict to relevant proxy records only
    for p in proxy_types: proxy_cfg.proxy_psm_type[p] = psm_key
    
    pre_calib_file = None
    # Define the psm-dependent required state variables
    if psm_key == 'linear':
        print('psm_key',psm_key)
        statevars = cfg.psm.linear.psm_required_variables
        psm_avg = cfg.psm.avgPeriod
        pre_calib_file =  cfg.psm.linear.pre_calib_datafile

    elif psm_key == 'bilinear':
        statevars = cfg.psm.bilinear.psm_required_variables
        psm_avg = cfg.psm.avgPeriod
        pre_calib_file =  cfg.psm.bilinear.pre_calib_datafile

    else:
        print('Exiting! You must use the PSMbuild facility to generate the appropriate calibrated PSMs')
        raise SystemExit

    if pre_calib_file:
        if isinstance(pre_calib_file, (list,tuple)):
            print('  from files: {}\n'
                  '         and: {}'.format(pre_calib_file[0],pre_calib_file[1]))
        else:
            print('  from file: {}'.format(pre_calib_file))

        
    # loading all available proxy objects
    proxy_objects = proxy_class.load_all_annual_no_filtering(cfg)
    print('proxy_objects',proxy_objects)
    # Number of proxy objects (will be a dim of ye_out array)
    num_proxy = len(proxy_objects)
    print('Calculating ye values for {:d} proxies'.format(num_proxy))

        
    # Define required temporal averaging
    if psm_avg == 'annual':
        # calendar year as the only seasonality vector for all proxies
        annual = [1,2,3,4,5,6,7,8,9,10,11,12]

        # assign annual seasonality attribute to all proxy objects
        # (override metadata of the proxy record)
        if psm_key == 'bilinear':
            # seasonality in fom of tuple
            season_unique = [(annual,annual)]
            for pobj in proxy_objects: pobj.seasonality = (annual,annual)
        else:
            # seasonality in form of list
            season_unique = [annual]
            for pobj in proxy_objects: pobj.seasonality = annual

        base_time_interval = 'annual'

    elif psm_avg == 'season':
        # map out all possible seasonality vectors that will have to be considered
        season_vects = []
        # Which seasonality to use? from proxy metadata or derived from psm calibration?
        # Attribute exists?
        if hasattr(cfg.psm,'season_source'):
            # which option is activated?
            if cfg.psm.season_source == 'psm_calib':
                for pobj in proxy_objects: season_vects.append(pobj.psm_obj.seasonality)
            elif cfg.psm.season_source == 'proxy_metadata':
                for pobj in proxy_objects: season_vects.append(pobj.seasonality)
            else:
                print('ERROR: Unrecognized value of *season_source* attribute')
                print('       in psm configuration.')
                raise SystemExit()
        else:
            # attribute does not exist in config., revert to proxy metadata
            for pobj in proxy_objects: season_vects.append(pobj.seasonality)
        
        season_unique = []
        for item in season_vects:
            if item not in season_unique:season_unique.append(item)
        base_time_interval = 'annual'

    elif  psm_avg == 'multiyear':
        season_unique = [cfg.prior.avgInterval['multiyear']]
        base_time_interval = 'multiyear'        

    else:
        print('ERROR in specification of averaging period.')
        raise SystemExit()        

    
    # Loop over seasonality definitions found in the proxy set
    firstloop = True
    for season in season_unique:

        if base_time_interval == 'annual':
            print('Calculating estimates for proxies with seasonality: '
                  '{}'.format(season))
        elif base_time_interval == 'multiyear':
            print('Calculating estimates for proxies with multiyear averaging: '
                  '{}'.format(season))

            
        # Create prior source object
        X = LMR_prior.prior_assignment(cfg.prior.prior_source)
        X.prior_datadir = cfg.prior.datadir_prior
        X.prior_datafile = cfg.prior.datafile_prior
        X.anom_reference = cfg.prior.anom_reference
        X.detrend = cfg.prior.detrend
        X.avgInterval = cfg.prior.avgInterval
        X.Nens = None  # None => Load entire prior
        X.statevars = statevars
        X.statevars_info = cfg.prior.state_variables_info

        
        # Load the prior data, averaged over interval corresponding
        # to current "season" (i.e. proxy seasonality)
        #X.avgInterval = season
        X.avgInterval = {base_time_interval: season} # new definition

        X.populate_ensemble(cfg.prior.prior_source, cfg.prior)
        
        statedim = X.ens.shape[0]
        ntottime = X.ens.shape[1]

        
        # Calculate the Ye values
        # -----------------------
        if firstloop:
            # Declare array of ye values if first time in loop
            ye_out = np.zeros((num_proxy, ntottime))
            # initialize with nan
            ye_out[:] = np.nan
            firstloop = False
        
        # loop over proxies
        for i, pobj in enumerate(proxy_objects):

            if base_time_interval == 'annual':
                # Restrict to proxy records with seasonality
                # corresponding to current "season" loop variable
                if cfg.psm.season_source == 'proxy_metadata':
                    if pobj.seasonality == season:
                        print('{:10d} (...of {:d})'.format(i, num_proxy), pobj.id)
                        ye_out[i] = pobj.psm(X.ens, X.full_state_info, X.coords)
                elif cfg.psm.season_source == 'psm_calib':
                    if pobj.psm_obj.seasonality == season:
                        print('{:10d} (...of {:d})'.format(i, num_proxy), pobj.id)
                        ye_out[i] = pobj.psm(X.ens, X.full_state_info, X.coords)
            else:
                print('{:10d} (...of {:d})'.format(i, num_proxy), pobj.id)
                ye_out[i] = pobj.psm(X.ens, X.full_state_info, X.coords)


    elapsed = timeit.default_timer() - masterstarttime
    print('\nElapsed time:', elapsed, ' secs')

    # Create a mapping for each proxy id to an index of the array
    pid_map = {pobj.id: idx for idx, pobj in enumerate(proxy_objects)} 

    # Create filename for current experiment
    out_dir = os.path.join(cfg.core.lmr_path, 'ye_precalc_files')

    # TODO: fix key usage
    vkind = X.statevars[list(X.statevars.keys())[0]]
    out_fname = create_precalc_ye_filename(cfg,psm_key,vkind)
    
    assert len(out_fname) <= 255, 'Filename is too long...'

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Write precalculated ye file
    out_full = os.path.join(out_dir, out_fname)
    print('Writing precalculated ye file: {}'.format(out_full))
    np.savez(out_full,
             pid_index_map=pid_map,
             ye_vals=ye_out)


elapsedtot = timeit.default_timer() - masterstarttime
print('------------------ ')
print('Total elapsed time:', elapsedtot/60.0 , ' mins')

