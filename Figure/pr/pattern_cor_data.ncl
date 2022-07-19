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
latS   =   (/10,30,24,40,15/)
latN   =   (/55,55,40,55,25/)
lonL   =   (/-135,-125,-100,-100,-110/)
lonR   =   (/-60,-100,-70,-70,-85/)
pr_std_LMR = new((/dimsizes(latS),ntime/),"float")
do ns=0,dimsizes(Season1)-1
;==========read the data==========
    diri3="/Users/chang/Desktop/LMR_Asia/data/data/analyses/GPCC/"
    name3="GPCC_precip.mon.flux.1x1.v6.nc"
    f3  = addfile(diri3+name3,"r")
      time=f3->time
      utc_date = cd_calendar(time, 0)
      year   = tointeger(utc_date(:,0))    ; Convert to integer for
      month  = tointeger(utc_date(:,1))
      iyyyy=ind(year.ge.year1.and.year.le.year2)
      precip=f3->precip(iyyyy,::-1,:)
      precip=lonFlip(precip) 
      precip:=precip(:,{latS:latN},{lonL:lonR})
      month:=month(iyyyy)
      year:=year(iyyyy)
      delete(iyyyy)
      Pre = new((/dimsizes(Year),dimsizes(precip(0,:,0)),dimsizes(precip(0,0,:))/),"float")
      if(Season1(ns).eq."WARM") then
        do ny=0,dimsizes(Year)-1
          iyyyy=ind(year.eq.Year(ny).and.month.ge.5.and.month.le.7)
          Pre(ny,:,:)=dim_avg_n_Wrap(precip(iyyyy,:,:),0)
          delete(iyyyy)
        end do
      end if
      if(Season1(ns).eq."COOL") then
        do ny=0,dimsizes(Year)-1
          if(ny.eq.0) then
            iyyyy=ind(year.eq.Year(ny).and.month.le.4)
          else
            iyyyy=ind((year.eq.Year(ny).and.month.le.4).or.(year.eq.Year(ny-1).and.month.eq.12))
          end if
          Pre(ny,:,:)=dim_avg_n_Wrap(precip(iyyyy,:,:),0)
          delete(iyyyy)
        end do
      end if
      lat_GPCC= Pre&lat
      lon_GPCC= Pre&lon
      Pre=Pre-conform_dims((/dimsizes(Pre)/),dim_avg_n_Wrap(Pre,0),(/1,2/))
      delete(precip)
      delete(time)
      delete(year)
      delete(month)

    ;======PDA_NASPA data==========
    diri1="/Users/chang/Desktop/NA_reconstruction/data/pda/"+Season(ns)+"_season/pre/full_ensemble/"
    files = systemfunc("ls "+diri1+"prate_MCruns_ensemble*.nc")
    pr_std= new((/dimsizes(files),ntime/),"float")
    do nf = 0,1;dimsizes(files)-1
    	f1  = addfile(files(nf),"r")
    	year=ispan(1900,2000,1)
    	iyyyy=ind(year.ge.year1.and.year.le.year2)
    	prate=f1->prate(iyyyy,0,:,:,:)
    	delete(year)
    	delete(iyyyy)
    	printVarSummary(prate)
    	prate&lon=prate&lon-360
        pr_LMR = prate(:,{latS:latN},{lonL:lonR},:)
	    lat_LMR= pr_LMR&lat
	    lon_LMR= pr_LMR&lon
	    delete(prate)

	    a    = addfile("$NCARG_ROOT/lib/ncarg/data/cdf/landsea.nc","r")
	    lsdata = a->LSMASK
	    lsm  = landsea_mask(lsdata,lat_LMR,lon_LMR)
	    pr_LMR(time|:,members|:,lat|:,lon|:) = mask(pr_LMR(time|:,members|:,lat|:,lon|:),lsm.eq.0,False)
	    pr_LMR(time|:,members|:,lat|:,lon|:) = mask(pr_LMR(time|:,members|:,lat|:,lon|:),lsm.eq.2,False)
	    pr_LMR(time|:,members|:,lat|:,lon|:) = mask(pr_LMR(time|:,members|:,lat|:,lon|:),lsm.eq.4,False)
	    delete(lsm)
	    delete(a)
	    delete(lsdata)

        if(nf.eq.0) then
            Pre_regrid = linint2_Wrap (lon_GPCC,lat_GPCC,Pre,True, lon_LMR,lat_LMR, 0)
            Pre_regrid!0="time"
            Pre_regrid = mask( Pre_regrid,lsm.eq.0,False)
            Pre_regrid = mask( Pre_regrid,lsm.eq.2,False)
            Pre_regrid = mask( Pre_regrid,lsm.eq.4,False)
        end if   

        rad    = 4.0*atan(1.0)/180.0
        re     = 6371220.0
        rr     = re*rad
        ilat   =ind(lat_LMR.ge.latS.and.lat_LMR.le.latN)
        clat   = cos(lat_LMR(ilat)*rad)
        rc = new((/ntime,200/),"float")
        do nm=0,199
            rc(:,nm) = pattern_cor(pr_LMR(time|:,lat|:,lon|:,members|nm),Pre_regrid(time|:,{lat|latS:latN},{lon|lonL:lonR}),clat, 0)
        end do
	      pr_std(nf,:) = dim_stddev_n_Wrap(pr_LMR_ave,1)  
        delete(rad)
        delete(re)
        delete(rr)
        delete(clat)
        delete(pr_LMR)
    end do 
  	pr_std_LMR(nr,:) = dim_avg_n_Wrap(pr_std,0)
  end do
 setfileoption("nc","Format","NetCDF4") 
 system("rm "+Season2(ns)+"_pattern_cor_std.nc")
 fout=addfile(Season2(ns)+"_pattern_cor_std.nc","c")
 setfileoption("nc","DefineMode",True)
 fout->pr_std_LMR=pr_std_LMR
  





