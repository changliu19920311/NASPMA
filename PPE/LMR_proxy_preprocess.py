"""
Module: LMR_proxy_preprocess.py

Purpose: Takes proxy data in their native format (e.g. .pckl file for PAGES2k or collection of
          NCDC-templated .txt files) and generates Pandas DataFrames stored in pickle files
          containing metadata and actual data from proxy records. The "pickled" DataFrames
          are used as input by the Last Millennium Reanalysis software.
          Currently, the data is stored as *annual averages* for original records with
          subannual data.

 Originator : Robert Tardif | Dept. of Atmospheric Sciences, Univ. of Washington
                            | January 2016
              (Based on code written by Andre Perkins (U. of Washington) to handle
              PAGES(2013) proxies)

  Revisions :
            - Addition of proxy types corresponding to "deep-times" proxy records, which are
              being included in the NCDC-templated proxy collection.
              [R. Tardif, U. of Washington, March 2017]
            - Addition of recognized time/age definitions used in "deep-times" proxy records
              and improved conversion of time/age data to year CE (convention used in LMR).
              [R. Tardif, U. of Washington, March 2017]
            - Improved detection & treatment of missing data, now using tags found
              (or not) in each data file.
              [R. Tardif, U. of Washington, March 2017]
            - Added functionalities related to the merging of proxies coming from two
              sources (PAGES2k phase 2 data contained in a single compressed pickle file
              and "in-house" collections contained in NCDC-templated text files).
              The possibility to "gaussianize" records and to calculate annual averages
              on "tropical year" (Apr to Mar) or calendar year have also been implemented.
              [R. Tardif, U. of Washington, Michael Erb, USC, May 2017]
            - Renamed the proxy databases to less-confusing convention.
              'pages' renamed as 'PAGES2kv1' and 'NCDC' renamed as 'LMRdb'
              [R. Tardif, U. of Washington, Sept 2017]
            - Add the China Documentary data 
              [Chang Liu, U. of Arkansas, June 2019]
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

# LMR imports
from LMR_utils import gaussianize

# =========================================================================================

class EmptyError(Exception):
    print(Exception)

# =========================================================================================
# ---------------------------------------- MAIN -------------------------------------------
# =========================================================================================
def main():

    # ********************************************************************************
    # Section for User-defined options: begin
    #

    #proxy_data_source = 'PAGES2Kv1' # proxies from PAGES2k phase 1 (2013)

    # --- *** --- *** --- *** --- *** --- *** --- *** --- *** --- *** --- *** ---
    #proxy_data_source = 'LMRdb'     # proxies from PAGES2k phase 2 (2017) + # "in-house" collection in NCDC-templated files
    proxy_data_source = 'Warm_season'   # Warm season treering                          
    #proxy_data_source = 'Cool_season'   # Warm season treering     
    # This option transforms all data to a Gaussian distribution.  It should only be used for
    # linear regressions, not physically-based PSMs.
    gaussianize_data = False

    # Specify the type of year to use for data averaging. "calendar year" (Jan-Dec)
    # or "tropical year" (Apr-Mar)
    year_type = "calendar year"
    #year_type = "tropical year"

    eliminate_duplicates = False



    # --- *** --- *** --- *** --- *** --- *** --- *** --- *** --- *** --- *** ---

    # datadir: directory where the original proxy datafiles are located
    datadir = '/Users/chang/Desktop/PPE_NA_1/data/data/proxies' #'/home/katabatic/wperkins/data/LMR/data/proxies/'

    # outdir: directory where the proxy database files will be created
    #         The piece before /data/proxies should correspond to your "lmr_path" set in LMR_config.py
    outdir = '/Users/chang/Desktop/PPE_NA_1/data/data/proxies' #'/home/katabatic/wperkins/data/LMR/data/proxies/'

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


    if proxy_data_source == 'Warm_season':
        
        take_average_out = False
        datadir = datadir+'NASPA/'
        STR_WARMfile = datadir + 'Warm_season_crns_NASPA.xlsx'
        meta_outfile = outdir + 'Warm_season_Metadata.df.pckl'
        data_outfile = outdir + 'Warm_season_Proxies.df.pckl'
        STR_WARM_dict = STR_WARM_xcel_to_dict(datadir, STR_WARMfile, gaussianize_data)
        proxy_def = \
        {'Seasonal Tree Rings'               : ['trsgi'],}
        convert_dicts_to_dataframes(proxy_def, STR_WARM_dict, meta_outfile, data_outfile)
    elif proxy_data_source == 'Cool_season':
        datadir = datadir+'NASPA/'
        STR_COOLfile = datadir + 'Cool_season_crns_NASPA.xlsx'
        meta_outfile = outdir + 'Cool_season_Metadata.df.pckl'
        data_outfile = outdir + 'Cool_season_Proxies.df.pckl'
        STR_COOL_dict = STR_COOL_xcel_to_dict(datadir, STR_COOLfile, gaussianize_data)
        proxy_def = \
        {'Seasonal Tree Rings'               : ['trsgi'],}
        convert_dicts_to_dataframes(proxy_def, STR_COOL_dict, meta_outfile, data_outfile)

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



    # ===================================================================================
# For DTDA project proxy data -------------------------------------------------------
# ===================================================================================

def STR_WARM_xcel_to_dataframes(filename, metaout, dataout, take_average_out):
    """
    Takes in Pages2K CSV and converts it to dataframe storage.  This increases
    size on disk due to the joining along the time index (lots of null values).

    Makes it easier to query and grab data for the proxy experiments.

    :param filename:
    :param metaout:
    :param dataout:
    :return:

    Author: Robert Tardif, Univ. of Washington

    Based on pages_xcel_to_dataframes function written by
    Andre Perkins (Univ. of Washington)

    """

    meta_sheet_name = 'Metadata'
    metadata = pd.read_excel(filename, meta_sheet_name)

    # add a "Databases" column and set to LMR
    metadata.loc[:,'Databases'] = '[NASPA]'
    metadata.loc[:,'Seasonality'] = '[5,6,7]'
    metadata.loc[:,'Elev'] = 0.0
    metadata.loc[:,'Resolution (yr)'] = 1
    metadata.loc[:,'Oldest (C.E.)'] = metadata.iloc[: , 2].copy()
    metadata.loc[:,'Youngest (C.E.)'] = metadata.iloc[: , 3].copy()
    nbrecords = len(metadata)
    metadata['Archive type'] = 'tree ring'

    # write metadata to file
    metadata.to_pickle(metaout)
    sheet = "proxy"
    df = pd.read_excel(filename, sheet)
    age = (df[pdata.columns[0]][1:]).astype('float').round()
    df[df.columns[0]][1:] = age
    newcol0 = df.columns[0]
    if newcol0 == 'Year C.E.' or newcol0 == 'Year CE':
        # do nothing
        pass
    elif newcol0 == 'Year BP':
        newcol0 = 'Year C.E.'
        df[col0][1:] = 1950. - df[col0][1:]
    else:
        print('Unrecognized time definition...')
        raise SystemExit()



    if take_average_out:
        # copy of dataframe
        df_tmp = df_f
        # fill dataframe with new values where temporal averages over proxy records are subtracted
        df_f = df_tmp.sub(df_tmp.mean(axis=0), axis=1)


    # TODO: make sure year index is consecutive
    #write data to file
    df_f.to_pickle(dataout)


def STR_COOL_xcel_to_dataframes(filename, metaout, dataout, take_average_out):
    """
    Takes in Pages2K CSV and converts it to dataframe storage.  This increases
    size on disk due to the joining along the time index (lots of null values).

    Makes it easier to query and grab data for the proxy experiments.

    :param filename:
    :param metaout:
    :param dataout:
    :return:

    Author: Robert Tardif, Univ. of Washington

    Based on pages_xcel_to_dataframes function written by
    Andre Perkins (Univ. of Washington)

    """

    meta_sheet_name = 'Metadata'
    metadata = pd.read_excel(filename, meta_sheet_name)

    # add a "Databases" column and set to LMR
    metadata.loc[:,'Databases'] = '[NASPA]'
    metadata.loc[:,'Seasonality'] = '[-12,1,2,3,4]'
    metadata.loc[:,'Elev'] = 0.0
    metadata.loc[:,'Resolution (yr)'] = 1
    metadata.loc[:,'Oldest (C.E.)'] = metadata.iloc[: , 2].copy()
    metadata.loc[:,'Youngest (C.E.)'] = metadata.iloc[: , 3].copy()
    nbrecords = len(metadata)
    metadata['Archive type'] = 'tree ring'

    # write metadata to file
    metadata.to_pickle(metaout)
    sheet = "proxy"
    df = pd.read_excel(filename, sheet)
    age = (df[pdata.columns[0]][1:]).astype('float').round()
    df[df.columns[0]][1:] = age
    newcol0 = df.columns[0]
    if newcol0 == 'Year C.E.' or newcol0 == 'Year CE':
        # do nothing
        pass
    elif newcol0 == 'Year BP':
        newcol0 = 'Year C.E.'
        df[col0][1:] = 1950. - df[col0][1:]
    else:
        print('Unrecognized time definition...')
        raise SystemExit()



    if take_average_out:
        # copy of dataframe
        df_tmp = df_f
        # fill dataframe with new values where temporal averages over proxy records are subtracted
        df_f = df_tmp.sub(df_tmp.mean(axis=0), axis=1)


    # TODO: make sure year index is consecutive
    #write data to file
    df_f.to_pickle(dataout)




def STR_WARM_xcel_to_dict(datadir, STR_WARM_file, gaussianize_data):
    """
    Read proxy data from collection of sesonal tree ring data excel and store the data in
    a python dictionary.


    :param datadir   :
    :param proxy_def :
    :param metaout   :
    :param dataout   :
    :return:


    Author: Chang Liu, Univ. of Arkansas

    """

    valid_frac = 0.5
     # ===============================================================================
    # Upload proxy data from excel files
    # ===============================================================================

    begin_time = clock.time()
    # Open the excel file containing the China DW data, if it exists in target directory
    infile = os.path.join(datadir, STR_WARM_file)
    if os.path.isfile(infile):
        print('Data from STR_WARM:')
        print(' Uploading data from %s ...' %infile)
    else:
        raise SystemExit(('ERROR: Option to include ChinaDW proxies enabled'
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
        # If gaussianize_data is set to true, transform the proxy data to Gaussian.
        # This option should only be used when using regressions, not physically-based PSMs.
        if gaussianize_data == True:
            data_annual = gaussianize(data_annual)
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
                # data
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
   

def STR_COOL_xcel_to_dict(datadir, STR_COOL_file, gaussianize_data):
    """
    Read proxy data from collection of sesonal tree ring data excel and store the data in
    a python dictionary.


    :param datadir   :
    :param proxy_def :
    :param metaout   :
    :param dataout   :
    :return:


    Author: Chang Liu, Univ. of Arkansas

    """

    valid_frac = 0.5
     # ===============================================================================
    # Upload proxy data from excel files
    # ===============================================================================

    begin_time = clock.time()
    # Open the excel file containing the China DW data, if it exists in target directory
    infile = os.path.join(datadir, STR_COOL_file)
    if os.path.isfile(infile):
        print('Data from STR_COOL:')
        print(' Uploading data from %s ...' %infile)
    else:
        raise SystemExit(('ERROR: Option to include ChinaDW proxies enabled'
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
        # If gaussianize_data is set to true, transform the proxy data to Gaussian.
        # This option should only be used when using regressions, not physically-based PSMs.
        if gaussianize_data == True:
            data_annual = gaussianize(data_annual)
        # Give each record a unique descriptive name
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

    # Summary

    print('----------------------------------------------------------------------')
    print(' STR_COOL SUMMARY: ')
    print('  Total nb of records found in file           : %d' % nbsites)
    print('  Number of proxy chronologies included in df : %d' %(len(proxy_dict_STR_COOL)))
    print('  ------------------------------------------------------')
    print(' ')



    elapsed_time = clock.time() - begin_time
    print('STR COOL data extraction completed in %s secs' %str(elapsed_time))

    return proxy_dict_STR_COOL
   


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



# =========================================================================================

def convert_dicts_to_dataframes(proxy_def, STR_WARM_dict, meta_outfile, data_outfile):
    """
    Merges three dictionaries containing proxy metadata and data from three data sources
    (PAGES2k phase 2, NCDC-templated proxy data files, and China DW index) into one,
    and writes out metadata and data into pickled pandas DataFrames.

    Originator: Chang Liu, Univ. of Arkansas, June 2019

    It is based on merge_dicts_to_dataframes function written by
    Robert (Univ. of Washington)
    """


    if len(STR_WARM_dict) > 0:
        merged_dict = deepcopy(STR_WARM_dict)
    else:
        raise SystemExit('No dataset has been selected for inclusion in the proxy database!')

    totchronol = len(merged_dict)
    print(merged_dict.keys())
    tot = []
    for key in sorted(proxy_def.keys()):
        proxy_archive = key.split('_')[0]
        proxy_measurement = proxy_def[key]
        print("proxy_archive",proxy_archive)
        print(proxy_measurement)
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
   
        for siteID in list(merged_dict.keys()):

            if merged_dict[siteID]['Archive'] == proxy_archive and merged_dict[siteID]['Measurement'] in proxy_def[key]:

                frame  = pd.DataFrame({'a':siteID, 'b':merged_dict[siteID]['StudyName'], 'c':merged_dict[siteID]['Investigators'], \
                                       'd':merged_dict[siteID]['SiteName'], 'e':merged_dict[siteID]['Lat'], 'f':merged_dict[siteID]['Lon'], \
                                       'g':merged_dict[siteID]['Elevation'], 'h':merged_dict[siteID]['Archive'], 'i':merged_dict[siteID]['Measurement'], \
                                       'j':merged_dict[siteID]['Resolution (yr)'], 'k':merged_dict[siteID]['Oldest (C.E.)'], \
                                       'l':merged_dict[siteID]['Youngest (C.E.)'], 'm':merged_dict[siteID]['Location'], \
                                       'n':merged_dict[siteID]['climateVariable'], 'o':merged_dict[siteID]['Realm'], \
                                       'p':merged_dict[siteID]['climateVariableDirec'], \
                                       'q':None, 'r':None}, index=[counter])
  
                frame.at[counter,'q'] = merged_dict[siteID]['Seasonality']
                frame.at[counter,'r'] = merged_dict[siteID]['Databases']


                # Append to main DataFrame
                metadf = metadf.append(frame)

                counter = counter + 1

    # Redefine column headers
    print(metadf)
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
