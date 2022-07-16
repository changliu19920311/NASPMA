"""
Module: LMR_proxy_preprocess.py
Purpose: Convert the proxy data in their native format to  "pickled" DataFrames

"""
import glob
import os
import os.path
import numpy as np
import pandas as pd
import time as clock
from copy import deepcopy
from scipy import stats
import string
import re
import six
import ast
from os.path import join
import pickle as pickle
import gzip
import calendar

# =========================================================================================

class EmptyError(Exception):
    print(Exception)

# =========================================================================================
# ---------------------------------------- MAIN -------------------------------------------
# =========================================================================================
def main():

    # ********************************************************************************
    # Section for User-defined options: begin
    # --- *** --- *** --- *** --- *** --- *** --- *** --- *** --- *** --- *** ---
    proxy_data_source = 'LMRdb'     # proxies from PAGES2k phase 2 (2017) + # "in-house" collection in NCDC-templated files
    include_NCDC = True
    include_PAGES2kphase2 = True
    include_NASPA_cool = True
    include_NASPA_warm = True
    #PAGES2kphase2file = 'PAGES2k_v2.0.0_tempOnly.pklz' # compressed version of the file
    PAGES2kphase2file = 'PAGES2k_v2.0.0_tempOnly.pckl'
    NASPAwarmfile = 'Warm_season_crns_NASPA.xlsx'
    NASPAcoolfile = 'Cool_season_crns_NASPA.xlsx'
    LMRdb_dbversion = 'v1.0.0'

    # File containing info on duplicates in proxy records
    infoDuplicates = 'Proxy_Duplicates_PAGES2kv2_NCDC_LMR'+LMRdb_dbversion+'.xlsx'

    # Specify the type of year to use for data averaging. "calendar year" (Jan-Dec)
    # or "tropical year" (Apr-Mar)
    year_type = "calendar year"

    eliminate_duplicates = True

    # datadir: directory where the original proxy datafiles are located
    datadir = '/Users/jjyan/Desktop/NA_reconstrution/code/PDA/data/data/proxies' 
    outdir = '/Users/jjyan/Desktop/NA_reconstrution/code/PDA/data/data/proxies' 

    #
    # Section for User-defined options: end
    # ***************************************************************

    main_begin_time = clock.time()

    # first checking that input and output directories exist on disk
    if not os.path.isdir(datadir):
        print('ERROR: Directory <<datadir>> does not exist. Please revise your'
              ' entry for this user-defined parameter.')
        raise SystemExit(1)
    else:
        # check that datadir ends with '/' -> expected thereafter
        if not datadir[-1] == '/':
            datadir = datadir+'/'

    if not os.path.isdir(outdir):
        print('ERROR: Directory <<outdir>> does not exist. Please revise your'
              ' entry for this user-defined parameter.')
        raise SystemExit(1)
    else:
        # check that outdir ends with '/' -> expected thereafter
        if not outdir[-1] == '/':
            outdir = outdir+'/'

    if  proxy_data_source == 'LMRdb':
        # ============================================================================
        # LMRdb proxy data -----------------------------------------------------------
        # ============================================================================

        datadir = datadir+'LMRdb/ToPandas_'+LMRdb_dbversion+'/'
        print('datadir',datadir)
        infoDuplicates = datadir+infoDuplicates

        # Some checks
        if not os.path.isdir(datadir):
            print('ERROR: Directory % is not found. Directory structure'
                  ' <<datadir>>/LMRdb/ToPandas_vX.Y.Z is expected.'
                  ' Please revise your set-up.' %datadir)
            raise SystemExit(1)

        if eliminate_duplicates and not os.path.isfile(infoDuplicates):
            print('ERROR: eliminate_duplicates parameter set to True but'
                  ' required file %s not found! Please rectify.' %infoDuplicates)
            raise SystemExit(1)


        meta_outfile = outdir + 'LMRdb_'+LMRdb_dbversion+'_Metadata.df.pckl'
        data_outfile = outdir + 'LMRdb_'+LMRdb_dbversion+'_Proxies.df.pckl'


        # Specify all proxy types & associated proxy measurements to look for & extract from the data files
        # This is to take into account all the possible different names found in the PAGES2kv2 and NCDC data files.
        proxy_def = \
            {
#old             'Tree Rings_WidthPages'                : ['TRW','ERW','LRW'],\
                'Tree Rings_WidthPages2'               : ['trsgi'],\
                'Tree Rings_WidthBreit'                : ['trsgi'],\
                'Tree Rings_WoodDensity'               : ['max_d','min_d','early_d','earl_d','late_d','MXD','density'],\
                'Tree Rings_Isotopes'                  : ['d18O'],\
                'Corals and Sclerosponges_d18O'        : ['d18O','delta18O','d18o','d18O_stk','d18O_int','d18O_norm','d18o_avg','d18o_ave','dO18','d18O_4'],\
                'Corals and Sclerosponges_SrCa'        : ['Sr/Ca','Sr_Ca','Sr/Ca_norm','Sr/Ca_anom','Sr/Ca_int'],\
                'Corals and Sclerosponges_Rates'       : ['ext','calc','calcification','calcification rate', 'composite'],\
                'Ice Cores_d18O'                       : ['d18O','delta18O','delta18o','d18o','d18o_int','d18O_int','d18O_norm','d18o_norm','dO18','d18O_anom'],\
                'Ice Cores_dD'                         : ['deltaD','delD','dD'],\
                'Ice Cores_Accumulation'               : ['accum','accumu'],\
                'Ice Cores_MeltFeature'                : ['MFP','melt'],\
                'Lake Cores_Varve'                     : ['varve', 'varve_thickness', 'varve thickness', 'thickness'],\
                'Lake Cores_BioMarkers'                : ['Uk37', 'TEX86', 'tex86'],\
                'Lake Cores_GeoChem'                   : ['Sr/Ca', 'Mg/Ca','Cl_cont'],\
                'Lake Cores_Misc'                      : ['RABD660_670','X_radiograph_dark_layer','massacum'],\
                'Marine Cores_d18O'                    : ['d18O'],\
                'Marine Cores_tex86'                   : ['tex86'],\
                'Marine Cores_uk37'                    : ['uk37','UK37'],\
                'Speleothems_d18O'                     : ['d18O'],\
                'Bivalve_d18O'                         : ['d18O'],\
                'Seasonal Tree Rings'               : ['trsgi'],\
                'Seasonal Tree Rings'               : ['trsgi'],\
            }


        # --- data from LMR's NCDC-templated files
        if include_NCDC:
            ncdc_dict = ncdc_txt_to_dict(datadir, proxy_def, year_type)
        else:
            ncdc_dict = []

        # --- PAGES2k phase2 (2017) data
        if include_PAGES2kphase2:
            pages2kv2_dict = pages2kv2_pickle_to_dict(datadir, PAGES2kphase2file, proxy_def, year_type)
        else:
            pages2kv2_dict = []

        if include_NASPA_warm:
            NASPAwarm_dict = STR_WARM_xcel_to_dict(datadir, NASPAwarmfile)
        else:
            NASPAwarm_dict = []

        if include_NASPA_cool:
            NASPAcool_dict = STR_COOL_xcel_to_dict(datadir, NASPAcoolfile)
        else:
            NASPAcool_dict = []
            
        # --- Merge datasets, scrub duplicates and write metadata & data to file
        merge_dicts_to_dataframes(proxy_def, ncdc_dict, pages2kv2_dict,NASPAwarm_dict,NASPAcool_dict,meta_outfile, data_outfile, infoDuplicates, eliminate_duplicates)

    else:
        raise SystemExit('ERROR: Unkown proxy data source! Exiting!')


    elapsed_time = clock.time() - main_begin_time
    print('Build of integrated proxy database completed in %s mins' %str(elapsed_time/60.))


# =========================================================================================
# ------------------------------------- END OF MAIN ---------------------------------------
# =========================================================================================


# =========================================================================================
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

def STR_WARM_xcel_to_dict(datadir, STR_WARM_file):
    """
    Read proxy data from collection of sesonal tree ring data excel and store the data in
    a python dictionary.

    """

    valid_frac = 0.5
     # ===============================================================================
    # Upload proxy data from excel files
    # ===============================================================================

    begin_time = clock.time()

    infile = os.path.join(datadir, STR_WARM_file)
    if os.path.isfile(infile):
        print('Data from STR_WARM:')
        print(' Uploading data from %s ...' %infile)
    else:
        raise SystemExit(('ERROR: Option to include STR WARM proxies enabled'
                         ' but corresponding data file could not be found!'
                         ' Please place file {} in directory {}').format(STR_WARM_file,datadir))

    # read the metadata
    meta_sheet_name = 'Metadata'
    metadata = pd.read_excel(infile, meta_sheet_name)
    proxy_index = metadata.iloc[:,0]
    print(proxy_index)
    lat = metadata.iloc[:,3]
    lon = metadata.iloc[:,4]
    start_yr = metadata.iloc[:,1]
    end_yr = metadata.iloc[:,2]
    # read the DW  data
    sheet_name = 'trsgi'
    data = pd.read_excel(infile,sheet_name)
    years = data.iloc[: , 0].copy()
    DWdata = data.iloc[: , 1:].copy()
    fillval = -9999
    DWdata[DWdata == fillval] = np.NAN

# Summary of the uploaded data
    nbsites = len(proxy_index)
    print('nbsites',nbsites)
    proxy_dict_STR_WARM = {}

    for counter in range(0,nbsites):
        data_annual=DWdata.iloc[:,counter]

        # Give each record a unique descriptive name
        proxy_name = proxy_index[counter]
        proxy_dict_STR_WARM[proxy_name] = {}
                # metadata
        proxy_dict_STR_WARM[proxy_name]['Archive']              = 'Seasonal Tree Rings'
        proxy_dict_STR_WARM[proxy_name]['Measurement']          = 'trsgi'
        proxy_dict_STR_WARM[proxy_name]['SiteName']             = proxy_index[counter]
        proxy_dict_STR_WARM[proxy_name]['StudyName']            = ['pub_title']
        proxy_dict_STR_WARM[proxy_name]['Investigators']        = ['pub_author']
        proxy_dict_STR_WARM[proxy_name]['Location']             = ['North America']
        proxy_dict_STR_WARM[proxy_name]['Resolution (yr)']      = 1
        proxy_dict_STR_WARM[proxy_name]['Lat']                  = lat[counter]
        proxy_dict_STR_WARM[proxy_name]['Lon']                  = lon[counter]
        proxy_dict_STR_WARM[proxy_name]['Elevation']            = ['None']
        proxy_dict_STR_WARM[proxy_name]['YearRange']            = (int('%.0f' %years[0]),int('%.0f' %years[nbsites]))
        proxy_dict_STR_WARM[proxy_name]['Databases']            = 'NASPA'
        proxy_dict_STR_WARM[proxy_name]['Seasonality']          = [5,6,7]
        proxy_dict_STR_WARM[proxy_name]['climateVariable']      = ['precipitation']
        proxy_dict_STR_WARM[proxy_name]['Realm']                = ['surface precipitation']
        proxy_dict_STR_WARM[proxy_name]['climateVariableDirec'] = ['positive']
        proxy_dict_STR_WARM[proxy_name]['Years']                = years
        proxy_dict_STR_WARM[proxy_name]['Data']                 = data_annual
        proxy_dict_STR_WARM[proxy_name]['Youngest (C.E.)']      =   end_yr
        proxy_dict_STR_WARM[proxy_name]['Oldest (C.E.)']        =   start_yr

    # ===============================================================================
    # Produce a summary of uploaded proxy data &
    # generate integrated database in pandas DataFrame format
    # ===============================================================================

    # Summary
    print('----------------------------------------------------------------------')
    print(' STR_WARM SUMMARY: ')
    print('  Total nb of records found in file           : %d' % nbsites)
    print('  Number of proxy chronologies included in df : %d' %(len(proxy_dict_STR_WARM)))
    print('  ------------------------------------------------------')
    print(' ')

    elapsed_time = clock.time() - begin_time
    print('STR WARM data extraction completed in %s secs' %str(elapsed_time))
    return proxy_dict_STR_WARM
   

def STR_COOL_xcel_to_dict(datadir, STR_COOL_file):
    """
    Read proxy data from collection of sesonal tree ring data excel and store the data in
    a python dictionary.
    """

    valid_frac = 0.5
     # ===============================================================================
    # Upload proxy data from excel files
    # ===============================================================================

    begin_time = clock.time()

    infile = os.path.join(datadir, STR_COOL_file)
    if os.path.isfile(infile):
        print('Data from STR_COOL:')
        print(' Uploading data from %s ...' %infile)
    else:
        raise SystemExit(('ERROR: Option to include STR COOL proxies enabled'
                         ' but corresponding data file could not be found!'
                         ' Please place file {} in directory {}').format(STR_COOL_file,datadir))

    # read the metadata
    meta_sheet_name = 'Metadata'
    metadata = pd.read_excel(infile, meta_sheet_name)
    proxy_index = metadata.iloc[:,0]
    lat = metadata.iloc[:,3]
    lon = metadata.iloc[:,4]
    start_yr = metadata.iloc[:,1]
    end_yr = metadata.iloc[:,2]

    # read the DW  data
    sheet_name = 'trsgi'
    data = pd.read_excel(infile,sheet_name)
    years = data.iloc[: , 0].copy()
    DWdata = data.iloc[: , 1:].copy()
    fillval = -9999
    DWdata[DWdata == fillval] = np.NAN

# Summary of the uploaded data
    nbsites = len(proxy_index)
    proxy_dict_STR_COOL = {}

    for counter in range(0,nbsites):
        data_annual=DWdata.iloc[:,counter]

        proxy_name = "STR_COOL_"+proxy_index[counter]
        proxy_dict_STR_COOL[proxy_name] = {}
                # metadata
        proxy_dict_STR_COOL[proxy_name]['Archive']              = 'Seasonal Tree Rings'
        proxy_dict_STR_COOL[proxy_name]['Measurement']          = 'trsgi'
        proxy_dict_STR_COOL[proxy_name]['SiteName']             = proxy_index[counter]
        proxy_dict_STR_COOL[proxy_name]['StudyName']            = ['pub_title']
        proxy_dict_STR_COOL[proxy_name]['Investigators']        = ['pub_author']
        proxy_dict_STR_COOL[proxy_name]['Location']             = ['North America']
        proxy_dict_STR_COOL[proxy_name]['Resolution (yr)']      = 1
        proxy_dict_STR_COOL[proxy_name]['Lat']                  = lat[counter]
        proxy_dict_STR_COOL[proxy_name]['Lon']                  = lon[counter]
        proxy_dict_STR_COOL[proxy_name]['Elevation']            = ['None']
        proxy_dict_STR_COOL[proxy_name]['YearRange']            = (int('%.0f' %years[0]),int('%.0f' %years[nbsites]))
        proxy_dict_STR_COOL[proxy_name]['Databases']            = 'NASPA'
        proxy_dict_STR_COOL[proxy_name]['Seasonality']          = [-12,1,2,3,4]
        proxy_dict_STR_COOL[proxy_name]['climateVariable']      = ['precipitation']
        proxy_dict_STR_COOL[proxy_name]['Realm']                = ['surface precipitation']
        proxy_dict_STR_COOL[proxy_name]['climateVariableDirec'] = ['positive']
                # data
        proxy_dict_STR_COOL[proxy_name]['Years']                = years
        proxy_dict_STR_COOL[proxy_name]['Data']                 = data_annual
        proxy_dict_STR_COOL[proxy_name]['Youngest (C.E.)']      =   end_yr
        proxy_dict_STR_COOL[proxy_name]['Oldest (C.E.)']        =   start_yr

    # ===============================================================================
    # Produce a summary of uploaded proxy data &
    # generate integrated database in pandas DataFrame format
    # ===============================================================================

    print('----------------------------------------------------------------------')
    print(' STR_COOL SUMMARY: ')
    print('  Total nb of records found in file           : %d' % nbsites)
    print('  Number of proxy chronologies included in df : %d' %(len(proxy_dict_STR_COOL)))
    print('  ------------------------------------------------------')
    print(' ')

    elapsed_time = clock.time() - begin_time
    print('STR COOL data extraction completed in %s secs' %str(elapsed_time))

    return proxy_dict_STR_COOL
   
# =========================================================================================
def compute_annual_means(time_raw,data_raw,valid_frac,year_type):
    """
    Computes annual-means from raw data.
    Inputs:
        time_raw   : Original time axis
        data_raw   : Original data
        valid_frac : The fraction of sub-annual data necessary to create annual mean.  Otherwise NaN.
        year_type  : "calendar year" (Jan-Dec) or "tropical year" (Apr-Mar)

    Outputs: time_annual, data_annual

    """

    # Check if dealing with multiple chronologies in one data stream (for NCDC files)
    array_shape = data_raw.shape
    if len(array_shape) == 2:
        nbtimes, nbvalid = data_raw.shape
    elif len(array_shape) == 1:
        nbtimes, = data_raw.shape
        nbvalid = 1
    else:
        raise SystemExit('ERROR in compute_annual_means: Unrecognized shape of data input array.')


    time_between_records = np.diff(time_raw, n=1)

    # Temporal resolution of the data, calculated as the mode of time difference.
    time_resolution = abs(stats.mode(time_between_records)[0][0])

    # check if time_resolution = 0.0 !!! sometimes adjacent records are tagged at same time ...
    if time_resolution == 0.0:
        print('***WARNING! Found adjacent records with same times!')
        inderr = np.where(time_between_records == 0.0)
        time_between_records = np.delete(time_between_records,inderr)
        time_resolution = abs(stats.mode(time_between_records)[0][0])

    max_nb_per_year = int(1.0/time_resolution)

    if time_resolution <=1.0:
        proxy_resolution = int(1.0) # coarse-graining to annual
    else:
        proxy_resolution = int(time_resolution)


    # Get rounded integer values of all years present in record.
    years_all = [int(np.floor(time_raw[k])) for k in range(0,len(time_raw))]
    years = list(set(years_all)) # 'set' is used to get unique values in list
    years = sorted(years) # sort the list

    years = np.insert(years,0,years[0]-1) # M. Erb

    # bounds, for calendar year : [years_beg,years_end[
    years_beg = np.asarray(years,dtype=np.float64) # inclusive lower bound
    years_end = years_beg + 1.                     # exclusive upper bound

    # If some of the time values are floats (sub-annual resolution)
    # and year_type is tropical_year, adjust the years to cover the
    # tropical year (Apr-Mar).
    if np.equal(np.mod(time_raw,1),0).all() == False and year_type == 'tropical year':
        print("Tropical year averaging...")

        # modify bounds defining the "year"
        for i, yr in enumerate(years):
            # beginning of interval
            if calendar.isleap(yr):
                years_beg[i] = float(yr)+((31+29+31)/float(366))
            else:
                years_beg[i] = float(yr)+((31+28+31)/float(365))
            # end of interval
            if calendar.isleap(yr+1):
                years_end[i] = float(yr+1)+((31+29+31)/float(366))
            else:
                years_end[i] = float(yr+1)+((31+28+31)/float(365))

    time_annual = np.asarray(years,dtype=np.float64)
    data_annual = np.zeros(shape=[len(years),nbvalid], dtype=np.float64)
    # fill with NaNs for default values
    data_annual[:] = np.NAN

    # Calculate the mean of all data points with the same year.
    for i in range(len(years)):
        ind = [j for j, year in enumerate(time_raw) if (year >= years_beg[i]) and (year < years_end[i])]
        nbdat = len(ind)

        # TODO: check nb of non-NaN values !!!!! ... ... ... ... ... ...

        if time_resolution <= 1.0:
            frac = float(nbdat)/float(max_nb_per_year)
            if frac > valid_frac:
                data_annual[i,:] = np.nanmean(data_raw[ind],axis=0)
        else:
            if nbdat > 1:
                print('***WARNING! Found multiple records in same year in data with multiyear resolution!')
                print('   year= %d %d' %(years[i], nbdat))
            # Note: this calculates the mean if multiple entries found
            data_annual[i,:] = np.nanmean(data_raw[ind],axis=0)


    # check and modify time_annual array to reflect only the valid data present in the annual record
    # for correct tagging of "Oldest" and "Youngest" data
    indok = np.where(np.isfinite(data_annual))[0]
    keep = np.arange(indok[0],indok[-1]+1,1)

    return time_annual[keep], data_annual[keep,:], proxy_resolution


# ===================================================================================
# For PAGES2k phase 2 (2017) proxy data ---------------------------------------------
# ===================================================================================

def pages2kv2_pickle_to_dict(datadir, pages2kv2_file, proxy_def, year_type):
    """

    Takes in a Pages2k pickle (pklz) file and converts it to python dictionary  storage.

    """

    valid_frac = 0.5

    # ===============================================================================
    # Upload proxy data from Pages2k v2 pickle file
    # ===============================================================================

    begin_time = clock.time()

    # Open the pickle file containing the Pages2k data, if it exists in target directory
    infile = os.path.join(datadir, pages2kv2_file)
    if os.path.isfile(infile):
        print('Data from PAGES2k phase 2:')
        print(' Uploading data from %s ...' %infile)
        try:
            # try to read as a straight pckl file
            pages2k_data = pd.read_pickle(infile)
            # f = open(infile,'rb')
            # pages2k_data = pickle.load(f)
            # f.close()
        except:
            # failed to read so try as a compressed pckl (pklz) file
            try:
                f = gzip.open(infile,'rb')
                pages2k_data = pickle.load(f)
                f.close()
            except:
                raise SystemExit(('ERROR: Could not read the PAGES2kv2 proxy file {}'
                         ' as a regular or compressed pickle file. Unrecognized format!').format(pages2kv2_file))
    else:
        raise SystemExit(('ERROR: Option to include PAGES2kv2 proxies enabled'
                         ' but corresponding data file could not be found!'
                         ' Please place file {} in directory {}').format(pages2kv2_file,datadir))


    # Summary of the uploaded data
    nbsites = len(pages2k_data)

    proxy_dict_pagesv2 = {}
    tot = []
    nb = []
    for counter in range(0,nbsites):

        # Give each record a unique descriptive name
        pages2k_data[counter]['siteID'] = "PAGES2kv2_"+pages2k_data[counter]['dataSetName']+\
                                          "_"+pages2k_data[counter]['paleoData_pages2kID']+\
                                          ":"+pages2k_data[counter]['paleoData_variableName']
        nb.append(pages2k_data[counter]['siteID'])

        print(' Processing %s/%s : %s' %(str(counter+1), str(len(pages2k_data)), pages2k_data[counter]['paleoData_pages2kID']))

        # Look for publication title & authors
        if 'NEEDS A TITLE' not in pages2k_data[counter]['pub1_title']:
            pages2k_data[counter]['pub_title'] = pages2k_data[counter]['pub1_title']
            pages2k_data[counter]['pub_author'] = pages2k_data[counter]['pub1_author']
        else:
            if 'NEEDS A TITLE' not in pages2k_data[counter]['pub2_title']:
                pages2k_data[counter]['pub_title'] = pages2k_data[counter]['pub2_title']
                pages2k_data[counter]['pub_author'] = pages2k_data[counter]['pub2_author']
            else:
                pages2k_data[counter]['pub_title'] = 'Unknown'
                pages2k_data[counter]['pub_author'] = 'Unknown'


        # If the time axis goes backwards (i.e. newer to older), reverse it.
        if pages2k_data[counter]['year'][-1] - pages2k_data[counter]['year'][-2] < 0:
            pages2k_data[counter]['year'].reverse()
            pages2k_data[counter]['paleoData_values'].reverse()

        # If subannual, average up to annual --------------------------------------------------------
        time_raw = np.array(pages2k_data[counter]['year'],dtype=np.float)
        data_raw = np.array(pages2k_data[counter]['paleoData_values'],dtype=np.float)

        # Remove values where either time or data is nan.
        nan_indices = np.isnan(time_raw)+np.isnan(data_raw)
        time_raw = time_raw[~nan_indices]
        data_raw = data_raw[~nan_indices]

        # Use the following function to make annual-means.
        # Inputs: time_raw, data_raw, valid_frac, year_type.  Outputs: time_annual, data_annual
        time_annual, data_annual, proxy_resolution = compute_annual_means(time_raw,data_raw,valid_frac,year_type)
        data_annual = np.squeeze(data_annual)

    

        # Write the annual data to the dictionary, so they can use written to
        # the data file outside of this loop.
        pages2k_data[counter]['time_annual'] = time_annual
        pages2k_data[counter]['data_annual'] = data_annual

        # Rename the proxy types in the same convention as the LMR's NCDC dataset.
        # Proxy types not renamed, except capitalizing 1st letter: bivalve, borehole, documents, hybrid
        if (pages2k_data[counter]['archiveType'] == 'coral') or (pages2k_data[counter]['archiveType'] == 'sclerosponge'):
            pages2k_data[counter]['archiveType'] = 'Corals and Sclerosponges'
        elif pages2k_data[counter]['archiveType'] == 'glacier ice':
            pages2k_data[counter]['archiveType'] = 'Ice Cores'
        elif pages2k_data[counter]['archiveType'] == 'lake sediment':
            pages2k_data[counter]['archiveType'] = 'Lake Cores'
        elif pages2k_data[counter]['archiveType'] == 'marine sediment':
            pages2k_data[counter]['archiveType'] = 'Marine Cores'
        elif pages2k_data[counter]['archiveType'] == 'speleothem':
            pages2k_data[counter]['archiveType'] = 'Speleothems'
        elif pages2k_data[counter]['archiveType'] == 'tree':
            pages2k_data[counter]['archiveType'] = 'Tree Rings'
        elif pages2k_data[counter]['archiveType'] == 'bivalve':
            pages2k_data[counter]['archiveType'] = 'Bivalve'
        elif pages2k_data[counter]['archiveType'] == 'borehole':
            pages2k_data[counter]['archiveType'] = 'Borehole'
        elif pages2k_data[counter]['archiveType'] == 'documents':
            pages2k_data[counter]['archiveType'] = 'Documents'
        elif pages2k_data[counter]['archiveType'] == 'hybrid':
            pages2k_data[counter]['archiveType'] = 'Hybrid'

        # Rename some of the the proxy measurements to be more standard.
        if (pages2k_data[counter]['archiveType'] == 'Ice Cores') and (pages2k_data[counter]['paleoData_variableName'] == 'd18O1'):
            pages2k_data[counter]['paleoData_variableName'] = 'd18O'
        elif (pages2k_data[counter]['archiveType'] == 'Tree Rings') and (pages2k_data[counter]['paleoData_variableName'] == 'temperature1'):
            pages2k_data[counter]['paleoData_variableName'] = 'temperature'
        elif (pages2k_data[counter]['archiveType'] == 'Lake Cores') and (pages2k_data[counter]['paleoData_variableName'] == 'temperature1'):
            pages2k_data[counter]['paleoData_variableName'] = 'temperature'
        elif (pages2k_data[counter]['archiveType'] == 'Lake Cores') and (pages2k_data[counter]['paleoData_variableName'] == 'temperature3'):
            pages2k_data[counter]['paleoData_variableName'] = 'temperature'

        # Not all records have data for elevation.  In these cases, set elevation to nan.
        if 'geo_meanElev' not in pages2k_data[counter]:
            pages2k_data[counter]['geo_meanElev'] = np.nan

        # Ensure lon is in [0,360] domain
        if pages2k_data[counter]['geo_meanLon'] < 0.0:
            pages2k_data[counter]['geo_meanLon'] = 360 + pages2k_data[counter]['geo_meanLon']

        # Determine the seasonality of the record.
        # Seasonal names were mapped the three-month climatological seasons.
        # 'early summer' was mapped to the first two months of summer only.  Is this right????????????
        # 'growing season' was mapped to summer.
        season_orig = pages2k_data[counter]['climateInterpretation_seasonality']
        if any(char.isdigit() for char in season_orig):
            pages2k_data_seasonality = list(map(int,season_orig.split(' ')))
        elif season_orig == 'annual':
            if year_type == 'tropical year': pages2k_data_seasonality = [4,5,6,7,8,9,10,11,12,13,14,15]
            else: pages2k_data_seasonality = [1,2,3,4,5,6,7,8,9,10,11,12]
        elif season_orig == 'summer':
            if pages2k_data[counter]['geo_meanLat'] >= 0: pages2k_data_seasonality = [6,7,8]
            else: pages2k_data_seasonality = [-12,1,2]
        elif season_orig == 'winter':
            if pages2k_data[counter]['geo_meanLat'] >= 0: pages2k_data_seasonality = [-12,1,2]
            else: pages2k_data_seasonality = [6,7,8]
        elif season_orig == 'winter/spring':
            if pages2k_data[counter]['geo_meanLat'] >= 0: pages2k_data_seasonality = [-12,1,2,3,4,5]
            else: pages2k_data_seasonality = [6,7,8,9,10,11]
        elif season_orig == 'early summer':
            if pages2k_data[counter]['geo_meanLat'] >= 0: pages2k_data_seasonality = [6,7]
            else: pages2k_data_seasonality = [-12,1]
        elif season_orig == 'growing season':
            if pages2k_data[counter]['geo_meanLat'] >= 0: pages2k_data_seasonality = [6,7,8]
            else: pages2k_data_seasonality = [-12,1,2]
        else:
            if year_type == 'tropical year': pages2k_data_seasonality = [4,5,6,7,8,9,10,11,12,13,14,15]
            else: pages2k_data_seasonality = [1,2,3,4,5,6,7,8,9,10,11,12]

        # If the year type is "tropical", change all records tagged as "annual" to be tropical-year.
        if year_type == 'tropical year' and pages2k_data_seasonality == [1,2,3,4,5,6,7,8,9,10,11,12]:
            pages2k_data_seasonality = [4,5,6,7,8,9,10,11,12,13,14,15]


        # Some code to fix two erroneous seasonality metadata found in the PAGES2kv2 file:
        # The data in the file itself should be fixed, but error dealt with here in the mean time.
        if pages2k_data_seasonality == [6,7,2008]:
            pages2k_data_seasonality = [6,7,8]
        elif pages2k_data_seasonality == [7,8,2009]:
            pages2k_data_seasonality = [7,8,9]

        # Spell out the name of the interpretation variable.
        if pages2k_data[counter]['climateInterpretation_variable'] == 'T':
            pages2k_data[counter]['climateInterpretation_variable'] = 'temperature'

        tot.append(len(nb))

        # ----------------------------------------------------------------------
        # Filter the records which correspond to the proxy types & measurements
        # specified in proxy_def dictionary. For records retained, transfer
        # a subset of the available information to elements used in used in
        # theLMR proxy database.
        # ----------------------------------------------------------------------

        for key in sorted(proxy_def.keys()):
            proxy_archive = key.split('_')[0]

            if pages2k_data[counter]['archiveType'] == proxy_archive \
               and pages2k_data[counter]['paleoData_variableName'] in proxy_def[key]:

                proxy_name = pages2k_data[counter]['siteID']
                proxy_dict_pagesv2[proxy_name] = {}
                # metadata
                proxy_dict_pagesv2[proxy_name]['Archive']              = pages2k_data[counter]['archiveType']
                proxy_dict_pagesv2[proxy_name]['Measurement']          = pages2k_data[counter]['paleoData_variableName']
                proxy_dict_pagesv2[proxy_name]['SiteName']             = pages2k_data[counter]['geo_siteName']
                proxy_dict_pagesv2[proxy_name]['StudyName']            = pages2k_data[counter]['pub_title']
                proxy_dict_pagesv2[proxy_name]['Investigators']        = pages2k_data[counter]['pub_author']
                proxy_dict_pagesv2[proxy_name]['Location']             = pages2k_data[counter]['geo_pages2kRegion']
                proxy_dict_pagesv2[proxy_name]['Resolution (yr)']      = proxy_resolution
                proxy_dict_pagesv2[proxy_name]['Lat']                  = pages2k_data[counter]['geo_meanLat']
                proxy_dict_pagesv2[proxy_name]['Lon']                  = pages2k_data[counter]['geo_meanLon']
                proxy_dict_pagesv2[proxy_name]['Elevation']            = pages2k_data[counter]['geo_meanElev']
                proxy_dict_pagesv2[proxy_name]['YearRange']            = (int('%.0f' %pages2k_data[counter]['time_annual'][0]),int('%.0f' %pages2k_data[counter]['time_annual'][-1]))
                proxy_dict_pagesv2[proxy_name]['Databases']            = ['PAGES2kv2']
                proxy_dict_pagesv2[proxy_name]['Seasonality']          = pages2k_data_seasonality
                proxy_dict_pagesv2[proxy_name]['climateVariable']      = pages2k_data[counter]['climateInterpretation_variable']
                proxy_dict_pagesv2[proxy_name]['Realm']                = pages2k_data[counter]['climateInterpretation_variableDetail']
                proxy_dict_pagesv2[proxy_name]['climateVariableDirec'] = pages2k_data[counter]['climateInterpretation_interpDirection']
                # data
                proxy_dict_pagesv2[proxy_name]['Years']                = pages2k_data[counter]['time_annual']
                proxy_dict_pagesv2[proxy_name]['Data']                 = pages2k_data[counter]['data_annual']

    nbtot = sum(tot)


    print('----------------------------------------------------------------------')
    print(' PAGES2kv2 SUMMARY: ')
    print('  Total nb of records found in file           : %d' %nbsites)
    print('  Number of proxy chronologies included in df : %d' %(len(proxy_dict_pagesv2)))
    print('  ------------------------------------------------------')
    print(' ')

    tot = []
    for key in sorted(proxy_def.keys()):
        proxy_archive = key.split('_')[0]
        proxy_measurement =  proxy_def[key]

        # change the associated between proxy type and proxy measurement for Breitenmoser tree ring data
        if key == 'Tree Rings_WidthBreit':
            proxy_measurement = [item+'_breit' for item in proxy_measurement]

        nb = []
        for siteID in list(proxy_dict_pagesv2.keys()):
            if proxy_dict_pagesv2[siteID]['Archive'] == proxy_archive and proxy_dict_pagesv2[siteID]['Measurement'] in proxy_measurement:
                nb.append(siteID)

        print(('   %s : %d' %('{:40}'.format(key), len(nb))))
        tot.append(len(nb))
    nbtot = sum(tot)
    print('  ------------------------------------------------------')
    print(('   %s : %d' %('{:40}'.format('Total:'), nbtot)))
    print('----------------------------------------------------------------------')
    print(' ')


    elapsed_time = clock.time() - begin_time
    print('PAGES2k phase2 data extraction completed in %s secs' %str(elapsed_time))

    return proxy_dict_pagesv2


# ===================================================================================
# For NCDC-templated proxy data files -----------------------------------------------
# ===================================================================================

def contains_blankspace(str):
    return True in [c in str for c in string.whitespace]

# ===================================================================================
def colonReader(string, fCon, fCon_low, end):
    '''This function seeks a specified string (or list of strings) within
    the transcribed file fCon (lowercase version fCon_low) until a specified
    character (typically end of the line) is found.x
    If a list of strings is provided, make sure they encompass all possibilities

    From Julien Emile-Geay (Univ. of Southern California)
    '''

    if isinstance(string, str):
        lstr = string + ': ' # append the annoying stuff
        Index = fCon_low.find(lstr)
        Len = len(lstr)

        if Index != -1:
            endlIndex = fCon_low[Index:].find(end)
            rstring = fCon[Index+Len:Index+endlIndex]  # returned string
            if rstring[-1:] == '\r':  # strip the '\r' character if it appears
                rstring = rstring[:-1]
            return rstring.strip()
        else:
            return ""
    else:
        num_str = len(string)
        rstring = "" # initialize returned string

        for k in range(0,num_str):  # loop over possible strings
            lstr = string[k] + ': ' # append the annoying stuff
            Index = fCon_low.find(lstr)
            Len = len(lstr)
            if Index != -1:
                endlIndex = fCon_low[Index:].find(end)
                rstring = fCon[Index+Len:Index+endlIndex]
                if rstring[-1:] == '\r':  # strip the '\r' character if it appears
                    rstring = rstring[:-1]

        if rstring == "":
            return ""
        else:
            return rstring.strip()


# ===================================================================================

def read_proxy_data_NCDCtxt(site, proxy_def, year_type=None):
#====================================================================================
# Purpose: Reads data from a selected site (chronology) in NCDC proxy dataset
#
# Input   :
#      - site        : Full name of proxy data file, including the directory where
#                      file is located.
#      - proxy_def   : Dictionary containing information on proxy types to look for
#                      and associated characteristics, such as possible proxy
#                      measurement labels for the specific proxy type
#                      (ex. ['d18O','d18o','d18o_stk','d18o_int','d18o_norm']
#                      for delta 18 oxygen isotope measurements)
#
# Returns :
#      - id      : Site id read from the data file
#      - lat/lon : latitude & longitude of the site
#      - alt     : Elevation of the site
#      - time    : Array containing the time of uploaded data
#      - value   : Array of uploaded proxy data
#
#====================================================================================


    # Possible header definitions of time in data files ...
    time_defs = ['age', 'age_int', 'year', \
                 'y_ad','Age_AD','age_AD','age_AD_ass','age_AD_int','Midpt_year','AD',\
                 'age_yb1950','yb_1950','yrb_1950',\
                 'kyb_1950',\
                 'yb_1989','age_yb1989',\
                 'yb_2000','yr_b2k','yb_2k','ky_b2k','kyb_2000','kyb_2k','kab2k','ka_b2k','kyr_b2k',\
                 'ky_BP','kyr_BP','ka_BP','age_kaBP','yr_BP','calyr_BP','Age(yrBP)','age_calBP','cal yr BP']

    filename = site

    valid_frac = 0.5


    if os.path.isfile(filename):
        print(' ')
        print('File: %s' % filename)

        # Define root string for filename
        file_s   = filename.replace(" ", '_')  # strip all whitespaces if present
        fileroot = '_'.join(file_s.split('.')[:-1])

        # Open the file and port content to a string object

        # Changed assumed encoding to UTF-8, anything not readable replaced with
        # a '?' --AP Jan 2018
        filein      = open(filename, encoding='utf-8', errors='replace')
        fileContent = filein.read()
        fileContent_low = fileContent.lower()

        # Initialize empty dictionary
        d = {}

        # Assign default values to some metadata
        d['ElevationUnit'] = 'm'
        d['TimeUnit']      = 'y_ad'

        # note: 8240/2030 ASCII code for "permil"

        # ===========================================================================
        # ===========================================================================
        # Extract metadata from file ------------------------------------------------
        # ===========================================================================
        # ===========================================================================
        try:
            # 'Archive' is the proxy type
            archive_tag               = colonReader('archive', fileContent, fileContent_low, '\n')
            # to match definitions of records from original NCDC-templeated files and those
            # provided by J. Tierney (U. of Arizona)
            if archive_tag == 'Paleoceanography': archive_tag = 'Marine Cores'
            d['Archive']              = archive_tag
            # Other info
            study_name                = colonReader('study_name', fileContent, fileContent_low, '\n')
            d['Title']                = study_name
            investigators             = colonReader('investigators', fileContent, fileContent_low, '\n')
            investigators.replace(';',' and') # take out the ; so that turtle doesn't freak out.
            d['Investigators']        = investigators
            d['PubDOI']               = colonReader('doi', fileContent, fileContent_low, '\n')

            # ===========================================================================
            # Extract information from the "Site_Information" section of the file -------
            # ===========================================================================
            # Find beginning of block
            sline_begin = fileContent.find('# Site_Information:')
            if sline_begin == -1:
                sline_begin = fileContent.find('# Site_Information')
            if sline_begin == -1:
                sline_begin = fileContent.find('# Site Information')
            # Find end of block
            sline_end = fileContent.find('# Data_Collection:')
            if sline_end == -1:
                sline_end = fileContent.find('# Data_Collection\n')
            if sline_end == -1:
                sline_end = fileContent.find('# Data_Collection\n')
            if sline_end == -1:
                sline_end = fileContent.find('# Data_Collection \n')
            if sline_end == -1:
                sline_end = fileContent.find('# Data_Collection  \n')
            if sline_end == -1:
                sline_end = fileContent.find('# Data_Collection   \n')
            if sline_end == -1:
                sline_end = fileContent.find('# Data_Collection    \n')
            if sline_end == -1:
                sline_end = fileContent.find('# Data Collection\n')
            if sline_end == -1:
                sline_end = fileContent.find('# Data Collection \n')
            if sline_end == -1:
                sline_end = fileContent.find('# Data Collection  \n')
            if sline_end == -1:
                sline_end = fileContent.find('# Data Collection   \n')
            if sline_end == -1:
                sline_end = fileContent.find('# Data Collection    \n')

            SiteInfo = fileContent[sline_begin:sline_end]
            SiteInfo_low = SiteInfo.lower()
            d['SiteName'] = colonReader('site_name', SiteInfo, SiteInfo_low, '\n')
            d['Location'] = colonReader('location', SiteInfo, SiteInfo_low, '\n')

            # get lat/lon info
            try:
                str_lst = ['northernmost_latitude', 'northernmost latitude'] # documented instances of this field property
                d['NorthernmostLatitude'] = float(colonReader(str_lst, SiteInfo, SiteInfo_low, '\n'))
                str_lst = ['southernmost_latitude', 'southernmost latitude'] # documented instances of this field property
                d['SouthernmostLatitude'] = float(colonReader(str_lst, SiteInfo, SiteInfo_low, '\n'))
                str_lst = ['easternmost_longitude', 'easternmost longitude'] # documented instances of this field property
                d['EasternmostLongitude'] = float(colonReader(str_lst, SiteInfo, SiteInfo_low, '\n'))
                str_lst = ['westernmost_longitude', 'westernmost longitude'] # documented instances of this field property
                d['WesternmostLongitude'] = float(colonReader(str_lst, SiteInfo, SiteInfo_low, '\n'))
            except (EmptyError,TypeError,ValueError) as err:
                print('*** %s' % err.args)
                print('*** WARNING ***: Valid values of lat/lon were not found! Skipping proxy record...')
                return (None, None)

            # get elevation info
            elev = colonReader('elevation', SiteInfo, SiteInfo_low, '\n')
            if 'nan' not in elev and len(elev)>0:
                elev_s = elev.split(' ')
                # is elevation negative (depth)?
                if '-' in elev_s[0] or d['Archive'] == 'Marine Cores':
                    negative = True
                    sign = '-'
                else:
                    negative = False
                    sign = ''
                # is there a decimal in elev_s?
                if '.' in elev_s[0]:
                    elev_s_split = elev_s[0].split('.')
                    elev_s_int = ''.join(c for c in elev_s_split[0] if c.isdigit())
                    elev_s_dec = ''.join(c for c in elev_s_split[1] if c.isdigit())
                    d['Elevation'] = float(sign+elev_s_int+'.'+elev_s_dec)
                else:
                    d['Elevation'] = float(sign+''.join(c for c in elev_s[0] if c.isdigit())) # to only keep digits ...

            else:
                d['Elevation'] = float('NaN')


            # ===========================================================================
            # Extract information from the "Data_Collection" section of the file --------
            # ===========================================================================
            # Find beginning of block
            sline_begin = fileContent.find('# Data_Collection:')
            if sline_begin == -1:
                sline_begin = fileContent.find('# Data_Collection')
            if sline_begin == -1:
                sline_begin = fileContent.find('# Data_Collection\n')

            # Find end of block
            sline_end = fileContent.find('# Variables:')
            if sline_end == -1:
                sline_end = fileContent.find('# Variables\n')
            if sline_end == -1:
                sline_end = fileContent.find('# Variables \n')
            if sline_end == -1:
                sline_end = fileContent.find('# Variables')
            if sline_end == -1:
                sline_end = fileContent.find('# Variables ')

            DataColl = fileContent[sline_begin:sline_end]
            DataColl_low = DataColl.lower()

            d['CollectionName'] = colonReader('collection_name', DataColl, DataColl_low, '\n')
            if not d['CollectionName']: d['CollectionName'] = filename.split('/')[-1].rstrip('.txt')

            EarliestYearStr   = colonReader('earliest_year', DataColl, DataColl_low, '\n')
            MostRecentYearStr = colonReader('most_recent_year', DataColl, DataColl_low, '\n')
            d['EarliestYear'] = None
            d['MostRecentYear'] = None
            if EarliestYearStr: d['EarliestYear'] = float(EarliestYearStr)
            if EarliestYearStr: d['MostRecentYear'] = float(MostRecentYearStr)

            d['TimeUnit'] = colonReader('time_unit', DataColl, DataColl_low, '\n')
            if not d['TimeUnit']: d['TimeUnit'] = colonReader('time unit', DataColl, DataColl_low, '\n')
            if d['TimeUnit'] not in time_defs:
                print('***Time_Unit *%s* not in recognized time definitions! Exiting!' %d['TimeUnit'])
                return (None, None)

            # Get Notes: information, if it exists
            notes = colonReader('notes', DataColl, DataColl_low, '\n')
            if notes: # not empty

                # database info is in form {"database":db1}{"database":db2} ...
                # extract fields that are in {}. This produces a list.
                jsdata = re.findall('\{.*?\}',notes)
                bad_chars = '{}"'
                jsdata = [item.translate(str.maketrans("", "", bad_chars)) for item in jsdata]

                # Look for database information
                # -----------------------------
                # item in jsdata list with database info?

                # TODO: ... think about using try/except instead ...
                dbinfo = None
                jsdata_db = [item for i, item in enumerate(jsdata) if 'database:' in item]
                if jsdata_db:
                    db_lst = re.sub('database:', '', jsdata_db[0]).split(',')
                    if len(db_lst) > 1:
                        dbinfo = [item.split(':')[1] for item in db_lst]
                    else:
                        dbinfo = db_lst

                # check if some db info exists
                if dbinfo:
                    d['Databases'] = dbinfo
                else:
                    # Set to default value if not found.
                    #d['Databases'] = None
                    d['Databases'] = ['LMR']


                # Look for information on "climate interpretation" of proxy record
                # ----------------------------------------------------------------

                # Initialize metadata to be extracted
                seasonality = [1,2,3,4,5,6,7,8,9,10,11,12] # annual (calendar)
                climateVariable = None
                climateVariableRealm = None
                climateVariableDirec = None

                jsdata_clim = [item for i, item in enumerate(jsdata) if 'climateInterpretation:' in item]
                if jsdata_clim:
                    clim_lst = re.sub('climateInterpretation:', '', jsdata_clim[0])
                    clim_lst = clim_lst.replace('[','(').replace(']',')')
                    tmp =  re.split(r',\s*(?![^()]*\))',clim_lst)
                    clim_elements = [item.replace('(','[').replace(')',']') for item in tmp]

                    seasonality          = [item.split(':')[1] for item in clim_elements if 'seasonality:' in item][0]
                    climateVariable      = [item.split(':')[1] for item in clim_elements if 'climateVariable:' in item][0]
                    climateVariableRealm = [item.split(':')[1] for item in clim_elements if 'climateVariableDetail:' in item][0]
                    climateVariableDirec = [item.split(':')[1] for item in clim_elements if 'interpDirection:' in item][0]

                    if len(seasonality) == 0: seasonality = [1,2,3,4,5,6,7,8,9,10,11,12]
                    if len(climateVariable) == 0: climateVariable = None
                    if len(climateVariableRealm) == 0: climateVariableRealm = None
                    if len(climateVariableDirec) == 0: climateVariableDirec = None

                    # Some translation...
                    if climateVariable == 'T': climateVariable = 'temperature'
                    if climateVariable == 'M': climateVariable = 'moisture'

                # test whether seasonality is a string or already a list
                # if a string, convert to list
                if type(seasonality) is not list:
                    if isinstance(seasonality,six.string_types):
                        seasonality = ast.literal_eval(seasonality)
                    else:
                        print('Problem with seasonality metadata! Exiting!')
                        SystemExit(1)

                d['Seasonality'] = seasonality
                d['climateVariable'] =  climateVariable
                d['climateVariableRealm'] = climateVariableRealm
                d['climateVariableDirec'] = climateVariableDirec

                # Look for information about duplicate proxy records
                # --------------------------------------------------
                dup_lst = []
                jsdata_dup = [item for i, item in enumerate(jsdata) if 'duplicate:' in item]
                if jsdata_dup:
                    tmp = re.sub('duplicate:', '', jsdata_dup[0]).split(',')
                    if len(tmp) > 1:
                        dup_lst = [item.split(':')[1].rstrip('.txt') for item in tmp]
                    else:
                        dup_lst = [item.rstrip('.txt') for item in tmp]

                d['Duplicates'] = dup_lst



                """
                # Old code that worked for NCDC v0.0.0

                # Look for information on relation to temperature
                # -----------------------------------------------
                clim_temp_relation = [item.split(':')[1] for item in jsdata if item.split(':')[0] == 'relationship']
                if clim_temp_relation:
                    d['Relation_to_temp'] = clim_temp_relation[0]
                else:
                    d['Relation_to_temp'] = None

                # Look for information on the nature of sensitivity of the proxy data
                # (i.e. temperature or moisture or etc.)
                # -------------------------------------------------------------------
                clim_sensitivity = [item.split(':')[1] for item in jsdata if item.split(':')[0] == 'sensitivity']
                if clim_sensitivity:
                    d['Sensitivity'] = clim_sensitivity[0]
                else:
                    d['Sensitivity'] = None

                """

                d['Relation_to_temp'] = None
                d['Sensitivity'] = None



            else:
                # Default values if not found.
                #d['Databases'] = None
                d['Databases'] = ['LMR']

                d['Seasonality'] = [1,2,3,4,5,6,7,8,9,10,11,12]
                d['climateVariable'] =  None
                d['climateVariableRealm'] = None
                d['climateVariableDirec'] = None

                d['Duplicates'] = []

                d['Relation_to_temp'] = None
                d['Sensitivity'] = None

            # If the year type is "tropical", change all annual records to the tropical-year mean.
            if year_type == 'tropical year' and d['Seasonality'] == [1,2,3,4,5,6,7,8,9,10,11,12]:
                d['Seasonality'] = [4,5,6,7,8,9,10,11,12,13,14,15]

        except EmptyError as e:
            print(e)
            return (None, None)

        # ===========================================================================
        # ===========================================================================
        # Extract the data from file ------------------------------------------------
        # ===========================================================================
        # ===========================================================================

        # ===========================================================================
        # Extract information from the "Variables" section of the file --------------
        # ===========================================================================

        # Find beginning of block
        sline_begin = fileContent.find('# Variables:')
        if sline_begin == -1:
            sline_begin = fileContent.find('# Variables')
        # Find end of block
        sline_end = fileContent.find('# Data:')
        if sline_end == -1:
            sline_end = fileContent.find('# Data\n')

        VarDesc = fileContent[sline_begin:sline_end].splitlines()
        nvar = 0 # counter for variable number
        for line in VarDesc:  # handle all the NCDC convention changes
            # (TODO: more clever/general exception handling)
            if line and line[0] != '' and line[0] != ' ' and line[0:2] != '#-' and line[0:2] != '# ' and line != '#':
                nvar = nvar + 1
                line2 = line.replace('\t',',') # clean up
                sp_line = line2.split(',')     # split line along commas
                if len(sp_line) < 9:
                    continue
                else:
                    d['DataColumn' + format(nvar, '02') + '_ShortName']   = sp_line[0].strip('#').strip(' ')
                    d['DataColumn' + format(nvar, '02') + '_LongName']    = sp_line[1]
                    d['DataColumn' + format(nvar, '02') + '_Material']    = sp_line[2]
                    d['DataColumn' + format(nvar, '02') + '_Uncertainty'] = sp_line[3]
                    d['DataColumn' + format(nvar, '02') + '_Units']       = sp_line[4]
                    d['DataColumn' + format(nvar, '02') + '_Seasonality'] = sp_line[5]
                    d['DataColumn' + format(nvar, '02') + '_Archive']     = sp_line[6]
                    d['DataColumn' + format(nvar, '02') + '_Detail']      = sp_line[7]
                    d['DataColumn' + format(nvar, '02') + '_Method']      = sp_line[8]
                    d['DataColumn' + format(nvar, '02') + '_CharOrNum']   = sp_line[9].strip(' ')

        print('Site ID: %s Archive: %s' %(d['CollectionName'], d['Archive']))


        # Cross-reference "ShortName" entries with possible proxy measurements specified in proxy_def dictionary
        proxy_types_all = list(proxy_def.keys())

        # Restrict to those matching d['Archive']
        proxy_types_keep = [s for s in proxy_types_all if d['Archive'] in s or d['Archive'] in s.lower()]

        # Which columns contain the important data (time & proxy values) to be extracted?
        # Referencing variables (time/age & proxy data) with data column IDsx
        # Time/age
        TimeColumn_ided = False
        for ivar in range(nvar):
            if d['DataColumn' + format(ivar+1, '02') + '_ShortName'] in time_defs:
                TimeColumn_ided = True
                TimeColumn_id = ivar
        if TimeColumn_ided:
            print('  Time/Age data in data column: %d' %TimeColumn_id)
        else:
            print(' ')


        # Proxy data
        # Dictionary containing info on proxy type and column ID where to find the data
        DataColumns_ided = False
        proxy_types_in_file = {}
        for ivar in range(nvar):
            proxy_types = [s for s in proxy_types_keep if d['DataColumn' + format(ivar+1, '02') + '_ShortName'] in proxy_def[s]]
            if proxy_types: # if non-empty list
                # Crude logic to distinguish between PAGES2kv2 vs Breitenmoser Tree Rings data at proxy type level
                if len(proxy_types) > 1 and [item for item in proxy_types if 'Tree Rings' in item ]:
                    if 'Breitenmoser' in d['Investigators'].split(',')[0]:
                        treetag = '_WidthBreit'
                    else:
                        treetag = '_WidthPages2'

                    ind = [i for i, s in enumerate(proxy_types) if s.endswith(treetag)][0]
                    proxy_types_in_file[proxy_types[ind]] = (d['DataColumn' + format(ivar+1, '02') + '_ShortName'], ivar)
                else:
                    proxy_types_in_file[proxy_types[0]] = (d['DataColumn' + format(ivar+1, '02') + '_ShortName'], ivar)


        dkeys = list(proxy_types_in_file.keys())
        nbvalid = len(dkeys)
        if nbvalid > 0:
            DataColumns_ided = True
            print('  Found %d valid proxy variables:' %nbvalid)
            for i in range(nbvalid):
                print(' %d : %s %s' %(i,dkeys[i],proxy_types_in_file[dkeys[i]]))

        # Check status of what has been found in the data file
        # If nothing found, just return (exit function by returning None)
        if not TimeColumn_ided or not DataColumns_ided:
            print('*** WARNING *** Valid data was not found in file!')
            return (None, None)



        # -- Checking time/age definition --
        tdef = d['TimeUnit']

        # Crude sanity checks on make-up of tdef string
        if contains_blankspace(tdef):
            tdef = tdef.replace(' ', '_')
        tdef_parsed = tdef.split('_')
        if len(tdef_parsed) != 2:
            tdef_parsed = tdef.split('_')
            if tdef_parsed[0] == 'cal' and tdef_parsed[1] == 'yr':
                tdef = tdef_parsed[0]+tdef_parsed[1]+'_'+tdef_parsed[2]
                tdef_parsed = tdef.split('_')
            else:
                print('*** WARNING *** Unrecognized time definition. Skipping proxy record...')
                return (None, None)


        # ===========================================================================
        # Extract the numerical data from the "Data" section of the file ------------
        # ===========================================================================

        # Find line number at beginning of data block
        sline = fileContent.find('# Data:')
        if sline == -1:
            sline = fileContent.find('# Data\n')
        fileContent_datalines = fileContent[sline:].splitlines()

        # Look for missing value info
        missing_info_line= [line for line in fileContent_datalines if 'missing value' in line.lower()]

        if len(missing_info_line) > 0:
            missing_info = missing_info_line[0].split(':')[-1].replace(' ', '')
            if len(missing_info) > 0:
                missing_values = np.array([float(missing_info)])
            else:
                # Line present but no value found
                missing_values = np.array([-999.0, np.nan])
        else:
            # Line not found
            missing_values = np.array([-999.0, np.nan])


        # Find where the actual data begin
        start_line_index = 0
        line_nb = 0
        for line in fileContent_datalines:  # skip lines without actual data
            if not line or line[0]=='#' or line[0] == ' ':
                start_line_index += 1
            else:
                start_line_index2 = line_nb
                break

            line_nb +=1


        # Extract column descriptions (headers) of the data matrix
        DataColumn_headers = fileContent_datalines[start_line_index].splitlines()[0].split('\t')
        # Strip possible blanks in column headers
        DataColumn_headers = [item.strip() for item in  DataColumn_headers]
        nc = len(DataColumn_headers)

        # ---------------------
        # -- Now the data !! --
        # ---------------------
        inds_to_extract = []
        for dkey in dkeys:
            inds_to_extract.append(proxy_types_in_file[dkey][1])

        # from start of data block to end, in a list
        datalist = fileContent_datalines[start_line_index+1:]
        # Strip any empty lines
        datalist = [x for x in datalist if x]
        nbdata = len(datalist)

        # into numpy arrays
        time_raw = np.zeros(shape=[nbdata])
        data_raw = np.zeros(shape=[nbdata,nbvalid])
        # fill with NaNs for default values
        data_raw[:] = np.NAN

        for i in range(nbdata):
            tmp = datalist[i].split('\t')
            # any empty element replaced by NANs
            tmp = ['NAN' if x == '' else x for x in tmp]
            time_raw[i]   = tmp[TimeColumn_id]
            # strip possible "()" in data before conversion to float
            # not sure why these are found sometimes ... sigh...
            tmp = [tmp[j].replace('(','') for j in range(len(tmp))]
            tmp = [tmp[j].replace(')','') for j in range(len(tmp))]
            data_raw[i,:] = [float(tmp[j]) for j in inds_to_extract]

        # -- Double check data validity --
        # (time/age in particular as some records have entries w/ undefined age)
        # Eliminate entries for which time/age is not defined (tagged as missing)
        mask = np.in1d(time_raw, missing_values, invert=True)
        time_raw = time_raw[mask]
        data_raw = data_raw[mask,:]

        # Making sure remaining entries in data array with missing values are converted to NaN.
        ntime, ncols = data_raw.shape
        for c in range(ncols):
            data_raw[np.in1d(data_raw[:,c], missing_values), c] = np.NAN



        # --- Modify "time" array into "years CE" if not already ---

        # Here, tdef_parsed *should* have the expected structure
        if len(tdef_parsed) == 2 and tdef_parsed[0] and tdef_parsed[1]:

            if tdef_parsed[0] == 'yb' and is_number(tdef_parsed[1]):
                time_raw = float(tdef_parsed[1]) - time_raw
            elif tdef_parsed[0] == 'kyb' and is_number(tdef_parsed[1]):
                time_raw = float(tdef_parsed[1]) - 1000.0*time_raw
            elif tdef_parsed[0] == 'calyr' and tdef_parsed[1] == 'BP':
                time_raw = 1950.0 - time_raw
            elif tdef_parsed[0] == 'kyr' and tdef_parsed[1] == 'BP':
                time_raw = 1950.0 - 1000.*time_raw
            elif tdef_parsed[0] == 'kyr' and tdef_parsed[1] == 'b2k':
                time_raw = 2000.0 - 1000.*time_raw
            elif tdef_parsed[0] == 'y' and tdef_parsed[1] == 'ad':
                pass # do nothing, time already in years_AD
            else:
                print('*** WARNING *** Unrecognized time definition. Skipping proxy record...')
                return (None, None)
        else:
            print('*** WARNING *** Unexpected time definition. Skipping proxy record...')
            return (None, None)


        # Making sure the tagged earliest and most recent years of the record are consistent with the data,
        # already transformed in year CE, common to all records before inclusion in the pandas DF.
        d['EarliestYear']   = np.min(time_raw)
        d['MostRecentYear'] = np.max(time_raw)

        # Initial range in years for which data is available
        yearRange = (int('%.0f' % d['EarliestYear']),int('%.0f' %d['MostRecentYear']))


        # proxy identifier and geo location
        id  = d['CollectionName']
        alt = d['Elevation']

        # Something crude in assignement of lat/lon:
        if d['NorthernmostLatitude'] != d['SouthernmostLatitude']:
            lat = (d['NorthernmostLatitude'] + d['SouthernmostLatitude'])/2.0
        else:
            lat = d['NorthernmostLatitude']
        if d['EasternmostLongitude'] != d['WesternmostLongitude']:
            lon = (d['EasternmostLongitude'] + d['WesternmostLongitude'])/2.0
        else:
            lon = d['EasternmostLongitude']

        # Ensure lon is in [0,360] domain
        if lon < 0.0:
            lon = 360 + lon


        # If subannual, average up to annual --------------------------------------------------------
        time_annual, data_annual, proxy_resolution = compute_annual_means(time_raw,data_raw,valid_frac,year_type)


        # update to yearRange given availability of annual data
        yearRange = (int('%.0f' %time_annual[0]),int('%.0f' %time_annual[-1]))


        # Define and fill list of dictionaries to be returned by function
        returned_list = []
        duplicate_list = []
        for k in range(len(dkeys)):
            key = dkeys[k]

            ind = proxy_types_in_file[key][1]
            proxy_units = d['DataColumn' + format(ind+1, '02') + '_Units']
            proxy_archive = key.split('_')[0]
            proxy_measurement = key.split('_')[1]
            proxy_measurement = d['DataColumn' + format(ind+1, '02') + '_ShortName']

            if key == 'Tree Rings_WidthBreit': proxy_measurement = proxy_measurement + '_breit'
            proxy_name = d['CollectionName']+':'+proxy_measurement

            proxydata_dict = {}
            proxydata_dict[proxy_name] = {}

            if d['Archive'] != proxy_archive: d['Archive'] = proxy_archive
            proxydata_dict[proxy_name]['Archive']              = d['Archive']
            proxydata_dict[proxy_name]['SiteName']             = d['SiteName']
            proxydata_dict[proxy_name]['StudyName']            = d['Title']
            proxydata_dict[proxy_name]['Investigators']        = d['Investigators']
            proxydata_dict[proxy_name]['Location']             = d['Location']
            proxydata_dict[proxy_name]['Resolution (yr)']      = proxy_resolution
            proxydata_dict[proxy_name]['Lat']                  = lat
            proxydata_dict[proxy_name]['Lon']                  = lon
            proxydata_dict[proxy_name]['Elevation']            = alt
            proxydata_dict[proxy_name]['YearRange']            = yearRange
            proxydata_dict[proxy_name]['Measurement']          = proxy_measurement
            proxydata_dict[proxy_name]['DataUnits']            = proxy_units
            proxydata_dict[proxy_name]['Databases']            = d['Databases']

            proxydata_dict[proxy_name]['Seasonality']          = d['Seasonality']
            proxydata_dict[proxy_name]['climateVariable']      = d['climateVariable']
            proxydata_dict[proxy_name]['Realm']                = d['climateVariableRealm']
            proxydata_dict[proxy_name]['climateVariableDirec'] = d['climateVariableDirec']

            # *** for v.0.1.0:
            #proxydata_dict[proxy_name]['Relation_to_temp'] = d['Relation_to_temp']
            #proxydata_dict[proxy_name]['Sensitivity']      = d['Sensitivity']

            proxydata_dict[proxy_name]['Years']            = time_annual
            proxydata_dict[proxy_name]['Data']             = data_annual[:, k]



            if d['Duplicates']:
                duplicate_list.extend(d['Duplicates'])

            # append to list of dictionaries
            returned_list.append(proxydata_dict)

    else:
        print('***File NOT FOUND: %s' % filename)
        returned_list = []
        duplicate_list = []

    return returned_list, duplicate_list

# =========================================================================================

def ncdc_txt_to_dict(datadir, proxy_def, year_type):
    """
    Read proxy data from collection of NCDC-templated text files and store the data in
    a python dictionary.

    """

    # ===============================================================================
    # Upload proxy data from NCDC-formatted text files
    # ===============================================================================

    begin_time = clock.time()

    print('Data from LMR NCDC-templated text files:')

    valid_frac = 0.5

    # List filenames im the data directory (dirname)
    # files is a python list contining file names to be read
    sites_data = glob.glob(datadir+"/*.txt")
    nbsites = len(sites_data)

    if nbsites == 0:
        print('ERROR: NCDC-templated proxy data files not found in directory:'
              ' %s. Please revise your user-defined parameters or directory/'
              ' data set-up.' %datadir)
        raise SystemExit(1)

    # Master dictionary containing all proxy chronologies extracted from the data files.
    proxy_dict_ncdc = {}
    dupelist = []
    # Loop over files
    nbsites_valid = 0
    for file_site in sites_data:

        proxy_list, duplicate_list = read_proxy_data_NCDCtxt(file_site,proxy_def,year_type)

        if proxy_list: # if returned list is not empty
            # extract data from list and populate the master proxy dictionary
            for item in proxy_list:
                proxy_name = list(item.keys())[0]
                # test if dict element already exists
                if proxy_name in list(proxy_dict_ncdc.keys()):
                    dupelist.append(proxy_name)
                else:
                    proxy_dict_ncdc[proxy_name] = item[proxy_name]
            nbsites_valid = nbsites_valid + 1
        else: # returned list is empty, just move to next site
            pass

    # ===============================================================================
    # Produce a summary of uploaded proxy data &
    # generate integrated database in pandas DataFrame format
    # ===============================================================================

    # Summary
    nbchronol = len(proxy_dict_ncdc)
    print(' ')
    print(' ')
    print('----------------------------------------------------------------------')
    print(' NCDC SUMMARY: ')
    print('  Total nb of files found & queried           : %d' % nbsites)
    print('  Total nb of files with valid data           : %d' % nbsites_valid)
    print('  Number of proxy chronologies included in df : %d' % nbchronol)
    print('  ------------------------------------------------------')
    print(' ')

    tot = []
    for key in sorted(proxy_def.keys()):
        proxy_archive = key.split('_')[0]
        proxy_measurement =  proxy_def[key]

        # change the association between proxy type and proxy measurement for Breitenmoser tree ring data
        if key == 'Tree Rings_WidthBreit':
            proxy_measurement = [item+'_breit' for item in proxy_measurement]

        nb = []
        for siteID in list(proxy_dict_ncdc.keys()):
            if proxy_dict_ncdc[siteID]['Archive'] == proxy_archive and proxy_dict_ncdc[siteID]['Measurement'] in proxy_measurement:
                nb.append(siteID)


        print('   %s : %d' %('{:40}'.format(key), len(nb)))
        tot.append(len(nb))
    nbtot = sum(tot)
    print('  ------------------------------------------------------')
    print('   %s : %d' %('{:40}'.format('Total:'), nbtot))
    print('----------------------------------------------------------------------')
    print(' ')

    if dupelist:
        print('***WARNING***: Proxy records with these names were found multiple times:')
        print(dupelist)

    elapsed_time = clock.time() - begin_time
    print('NCDC data extraction completed in %s secs' %str(elapsed_time))

    return proxy_dict_ncdc


# =========================================================================================

def merge_dicts_to_dataframes(proxy_def, ncdc_dict, pages2kv2_dict,NASPAwarm_dict,\
                              NASPAcool_dict,meta_outfile,data_outfile,duplicates_file, eliminate_duplicates):
    """
    Merges four dictionaries containing proxy metadata and data from three data source into one,
    and writes out metadata and data into pickled pandas DataFrames.

     Chang Liu, Univ. of Arkansas, June 2019

    It is based on merge_dicts_to_dataframes function written by
    Robert (Univ. of Washington)
    """
    

    if len(ncdc_dict) > 0:
        merged_dict = deepcopy(ncdc_dict)
        if len(pages2kv2_dict) > 0:
            merged_dict.update(pages2kv2_dict)
            if len(NASPAwarm_dict) > 0:
                merged_dict.update(NASPAwarm_dict)
            if len(NASPAcool_dict) > 0:
                merged_dict.update(NASPAcool_dict)  
    elif len(NASPAcool_dict) > 0:
        merged_dict.update(NASPAcool_dict)     
    elif len(pages2kv2_dict) > 0:
        merged_dict.update(pages2kv2_dict)
    elif len(NASPAwarm_dict) > 0:
        merged_dict.update(NASPAwarm_dict)
    elif len(NASPAcool_dict) > 0:
        merged_dict.update(NASPAcool_dict)      
    else:
        raise SystemExit('No dataset has been selected for inclusion in the proxy database!')

    totchronol = len(merged_dict)

    dupecount = 0
    if eliminate_duplicates:
        print(' ')
        print('Checking list of duplicate/bad records:')

        # load info on duplicate records
        dupes = pd.read_excel(duplicates_file,'ProxyDuplicates')

        # numpy array containing names of proxy records to eliminate
        toflush = dupes['Record_To_Eliminate'].values

        for siteID in list(merged_dict.keys()):
            if siteID in toflush:
                try:
                    del merged_dict[siteID]
                    print(' -- deleting: %s' % siteID)
                    dupecount += 1
                except KeyError:
                    print(' -- not found: %s' % siteID)
                    pass

    print(' ')
    print('----------------------------------------------------------------------')
    print(' FINAL SUMMARY: ')
    print('  Total number of merged proxy chronologies   : %d' %totchronol)
    print('  Total number of eliminated chronologies     : %d' %dupecount)
    print('  Number of proxy chronologies included in df : %d' %len(merged_dict))
    print('  ------------------------------------------------------')

    tot = []
    for key in sorted(proxy_def.keys()):
        proxy_archive = key.split('_')[0]
        proxy_measurement = proxy_def[key]

        # change the association between proxy type and proxy measurement for Breitenmoser tree ring data
        if key == 'Tree Rings_WidthBreit':
            proxy_measurement = [item+'_breit' for item in proxy_measurement]

        nb = []
        for siteID in list(merged_dict.keys()):
            if merged_dict[siteID]['Archive'] == proxy_archive and merged_dict[siteID]['Measurement'] in proxy_measurement:
                nb.append(siteID)

        print('   %s : %d' %('{:40}'.format(key), len(nb)))
        tot.append(len(nb))
    nbtot = sum(tot)
    print('  ------------------------------------------------------')
    print('   %s : %d' %('{:40}'.format('Total:'), nbtot))
    print('----------------------------------------------------------------------')
    print(' ')

    # ---------------------------------------------------------------------
    # Preparing pandas DataFrames containing merged proxy metadata and data
    # and output to pickle files
    # ---------------------------------------------------------------------

    # Loop over proxy types specified in *main*
    counter = 0
    # Build up pandas DataFrame
    metadf  = pd.DataFrame()

#    headers = ['Proxy ID','Site name','Lat (N)','Lon (E)','Elev','Archive type','Proxy measurement','Resolution (yr)',\
#               'Oldest (C.E.)','Youngest (C.E.)','Location','climateVariable','Realm','Relation_to_climateVariable',\
#               'Seasonality', 'Databases']

    headers = ['Proxy ID','Study name','Investigators','Site name','Lat (N)','Lon (E)','Elev','Archive type','Proxy measurement',\
               'Resolution (yr)','Oldest (C.E.)','Youngest (C.E.)','Location','climateVariable','Realm','Relation_to_climateVariable',\
               'Seasonality', 'Databases']


    for key in sorted(proxy_def.keys()):
        proxy_archive = key.split('_')[0]

        # change the associated between proxy type and proxy measurement for Breitenmoser tree ring data
        if key == 'Tree Rings_WidthBreit':
            proxy_def[key] = [item+'_breit' for item in proxy_def[key]]

        for siteID in list(merged_dict.keys()):
            if merged_dict[siteID]['Archive'] == proxy_archive and merged_dict[siteID]['Measurement'] in proxy_def[key]:

                frame  = pd.DataFrame({'a':siteID, 'b':merged_dict[siteID]['StudyName'], 'c':merged_dict[siteID]['Investigators'], \
                                       'd':merged_dict[siteID]['SiteName'], 'e':merged_dict[siteID]['Lat'], 'f':merged_dict[siteID]['Lon'], \
                                       'g':merged_dict[siteID]['Elevation'], 'h':merged_dict[siteID]['Archive'], 'i':merged_dict[siteID]['Measurement'], \
                                       'j':merged_dict[siteID]['Resolution (yr)'], 'k':merged_dict[siteID]['YearRange'][0], \
                                       'l':merged_dict[siteID]['YearRange'][1], 'm':merged_dict[siteID]['Location'], \
                                       'n':merged_dict[siteID]['climateVariable'], 'o':merged_dict[siteID]['Realm'], \
                                       'p':merged_dict[siteID]['climateVariableDirec'], \
                                       'q':None, 'r':None}, index=[counter])
                # To get seasonality & databases *lists* into columns 'o' and 'p' of DataFrame
                # To be deprecated - frame.set_value(counter,'q',merged_dict[siteID]['Seasonality'])
                # To be deprecated - frame.set_value(counter,'r',merged_dict[siteID]['Databases'])
                frame.at[counter,'q'] = merged_dict[siteID]['Seasonality']
                frame.at[counter,'r'] = merged_dict[siteID]['Databases']


                # Append to main DataFrame
                metadf = metadf.append(frame)

                counter = counter + 1

    # Redefine column headers
    metadf.columns = headers

    # Write metadata to file
    print('Now writing metadata to file: %s' %meta_outfile)
    metadf.to_pickle(meta_outfile)

    # -----------------------------------------------------
    # Build the proxy **data** DataFrame and output to file
    # -----------------------------------------------------
    print(' ')
    print('Now creating & loading the data in the pandas DataFrame...')
    print(' ')

    counter = 0
    for siteID in list(merged_dict.keys()):

        years = merged_dict[siteID]['Years']
        data = merged_dict[siteID]['Data']
        [nbdata,] = years.shape

        # Load data in numpy array
        frame_data = np.zeros(shape=[nbdata,2])
        frame_data[:,0] = years
        frame_data[:,1] = data

        if counter == 0:
            # Build up pandas DataFrame
            header = ['Proxy ID', siteID]
            df = pd.DataFrame({'a':frame_data[:,0], 'b':frame_data[:,1]})
            df.columns = header
        else:
            frame = pd.DataFrame({'Proxy ID':frame_data[:,0], siteID:frame_data[:,1]})
            df = df.merge(frame, how='outer', on='Proxy ID')

        counter = counter + 1

    # Fix DataFrame index and column name
    col0 = df.columns[0]
    df.set_index(col0, drop=True, inplace=True)
    df.index.name = 'Year C.E.'
    df.sort_index(inplace=True)

    # Write data to file
    print('Now writing to file:', data_outfile)
    df = df.to_sparse()
    df.to_pickle(data_outfile)
    print('Done!')

# =========================================================================================
# =========================================================================================
if __name__ == "__main__":
    main()
