; evaluating the reconstucted monthly data=================
 load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
 load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
 load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
 load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"

;================set the parameters=====================
Season1=(/"WARM","COOL"/)
Season =(/"warm","cool"/)
Season2=(/"Warm","Cool"/)
year1=1901
year2=2000
Year=ispan(year1,year2,1)
ntime = dimsizes(Year)
latS   =   (/10,30,24,40/)
latN   =   (/55,55,40,55/)
lonL   =   (/-135,-125,-100,-100/)
lonR   =   (/-60,-100,-70,-70/)
region = (/"NA","Western US","Southeastern US","Northeast NA"/)
pr_std_LMR = new((/dimsizes(latS),ntime/),"float")
do ns=0,dimsizes(Season1)-1
    ;======PDA_NASPA data==========
    diri1="/Users/chang/Desktop/NA_reconstruction/data/pda/"+Season(ns)+"_season/pre/full_ensemble/"
    files = systemfunc("ls "+diri1+"prate_MCruns_ensemble*.nc")
    pr_std= new((/dimsizes(files),dimsizes(region),ntime/),"float")
    do nf = 0,1;dimsizes(files)-1
        f1  = addfile(files(nf),"r")
        year=ispan(1900,2000,1)
        iyyyy=ind(year.ge.year1.and.year.le.year2)
        prate=f1->prate(iyyyy,0,:,:,:)
        delete(year)
        delete(iyyyy)
        printVarSummary(prate)
        prate&lon=prate&lon-360
        lat_LMR= prate&lat
        lon_LMR= prate&lon
        a    = addfile("$NCARG_ROOT/lib/ncarg/data/cdf/landsea.nc","r")
        lsdata = a->LSMASK
        lsm  = landsea_mask(lsdata,lat_LMR,lon_LMR)
        prate(time|:,members|:,lat|:,lon|:) = mask(prate(time|:,members|:,lat|:,lon|:),lsm.eq.0,False)
        prate(time|:,members|:,lat|:,lon|:) = mask(prate(time|:,members|:,lat|:,lon|:),lsm.eq.2,False)
        prate(time|:,members|:,lat|:,lon|:) = mask(prate(time|:,members|:,lat|:,lon|:),lsm.eq.4,False)
        delete(lsm)
        delete(a)
        delete(lsdata)
        delete(lat_LMR)
        delete(lon_LMR)
        do nr=0,dimsizes(latS)-1
            pr_LMR = prate(:,{latS(nr):latN(nr)},{lonL(nr):lonR(nr)},:)
    	    lat_LMR= pr_LMR&lat
    	    lon_LMR= pr_LMR&lon
            rad    = 4.0*atan(1.0)/180.0
            re     = 6371220.0
            rr     = re*rad
            clat   = cos(lat_LMR*rad)
            pr_LMR_ave = wgt_areaave_Wrap(pr_LMR(time|:,members|:,lat|:,lon|:), clat, 1.0, 0)*86400*30
            printVarSummary(pr_LMR_ave)
            pr_std(nf,nr,:) = dim_stddev_n_Wrap(pr_LMR_ave,1)  
            delete(rad)
            delete(re)
            delete(rr)
            delete(clat)
            delete(pr_LMR)
            delete(lat_LMR)
            delete(lon_LMR)
            delete(pr_LMR_ave)
        end do 
        delete(prate)
    end do
    pr_std_LMR = dim_avg_n_Wrap(pr_std,0)
    delete(pr_std)
    pr_std_LMR@units="mm/month"
    pr_std_LMR!0 = "region"
    pr_std_LMR!1 = "Year"
    pr_std_LMR&Year = Year
    setfileoption("nc","Format","NetCDF4") 
    system("rm "+Season2(ns)+"_timeseries_std.nc")
    fout=addfile(Season2(ns)+"_timeseries_std.nc","c")
    setfileoption("nc","DefineMode",True)
    fout->pr_std_LMR=pr_std_LMR
end do  





