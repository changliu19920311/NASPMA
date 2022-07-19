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
PSM = (/"linear"/)
latS   =   (/10,30,25,40/)
latN   =   (/55,55,40,55/)
lonL   =   (/-135,-130,-100,-100/)
lonR   =   (/-60,-100,-70,-70/)
region = (/"NA","Western US","Southeastern US","Northeast NA"/)
data=(/"GPCC"/)
name = (/"GPCC_precip.mon.flux.1x1.v6.nc"/)
p=new(4,"float")
ccr=p
ci_0=p
ci_1=p
do ns=0,dimsizes(Season1)-1
  diri1="//Users/jjyan/Desktop/NA_reconstrution/data/"+Season(ns)+"_season/pre/"
  file1 = "prate_MCruns_ensemble_mean_800_2000.nc"
  do nr=0,dimsizes(region)-1
    ;======PDA_NASPA data==========
     f1  = addfile(diri1+file1,"r")
  year=ispan(800,2000,1)
  iyyyy=ind(year.ge.year1.and.year.le.year2)
  prate=f1->prate(iyyyy,0,:,:)
  delete(year)
  delete(iyyyy)
  printVarSummary(prate)
  prate&lon=prate&lon-360
    pr_LMR = prate(:,{latS(nr):latN(nr)},{lonL(nr):lonR(nr)})
    lat_LMR= pr_LMR&lat
    lon_LMR= pr_LMR&lon
    delete(prate)

    a    = addfile("$NCARG_ROOT/lib/ncarg/data/cdf/landsea.nc","r")
    lsdata = a->LSMASK
    lsm  = landsea_mask(lsdata,lat_LMR,lon_LMR)
    pr_LMR(time|:,lat|:,lon|:) = mask(pr_LMR(time|:,lat|:,lon|:),lsm.eq.0,False)
    pr_LMR(time|:,lat|:,lon|:) = mask(pr_LMR(time|:,lat|:,lon|:),lsm.eq.2,False)
    pr_LMR(time|:,lat|:,lon|:) = mask(pr_LMR(time|:,lat|:,lon|:),lsm.eq.4,False)
    delete(lsm)
    delete(a)
    delete(lsdata)

    rad    = 4.0*atan(1.0)/180.0
    re     = 6371220.0
    rr     = re*rad
    clat   = cos(lat_LMR*rad)
    pr_avg_LMR = wgt_areaave_Wrap(pr_LMR(time|:,lat|:,lon|:), clat, 1.0, 0)
    printVarSummary(pr_avg_LMR)
    pr_avg_LMR =  pr_avg_LMR*86400
    delete(rad)
    delete(re)
    delete(rr)
    delete(clat)
    delete(pr_LMR) 


     diri3="/Users/jjyan/Desktop/NA_reconstrution/code/PDA/data/data/analyses/"+data(0)+"/"
    name3=name(0)

    ;==========read the GPCC data， calculate the cool season average ==========
    f3  = addfile(diri3+name3,"r")
    time=f3->time
    utc_date = cd_calendar(time, 0)
    year   = tointeger(utc_date(:,0))    ; Convert to integer for
    month  = tointeger(utc_date(:,1))
    iyyyy=ind(year.ge.year1.and.year.le.year2)

      precip=f3->precip(iyyyy,::-1,:)
      precip=lonFlip(precip) 

    precip:=precip(:,{latS(nr):latN(nr)},{lonL(nr):lonR(nr)})
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
    delete(precip)
    delete(time)
    delete(year)
    delete(month)
    delete(utc_date)

      ;========all the data are regrid to teh LMR grid=======
      lat_GPCC= Pre&lat
      lon_GPCC= Pre&lon
      Pre := linint2_Wrap(lon_GPCC,lat_GPCC,Pre, False, lon_LMR,lat_LMR, 0)
    

      ;==========adjust the climatology====================
      GPCC_clm=conform_dims(dimsizes(Pre),dim_avg_n_Wrap(Pre,0),(/1,2/))
      Pre=Pre-GPCC_clm
      delete(GPCC_clm)

      a    = addfile("$NCARG_ROOT/lib/ncarg/data/cdf/landsea.nc","r")
      lsdata = a->LSMASK
      lsm  = landsea_mask(lsdata,Pre&lat,Pre&lon)
      Pre= mask(Pre,lsm.eq.2,False)
      Pre = mask(Pre,lsm.eq.0,False)
      delete(lsm)
      delete(a)
      delete(lsdata)

      rad    = 4.0*atan(1.0)/180.0
      re     = 6371220.0
      rr     = re*rad
      clat   := cos(lat_LMR*rad)
      pr_avg_GPCC = wgt_areaave(Pre, clat, 1.0, 0)
      pr_avg_GPCC = pr_avg_GPCC*86400
      delete(Pre)
      delete(lat_GPCC)
      delete(lon_GPCC)
      delete(rad)
      delete(re)
      delete(rr)
      delete(clat)
      delete(lat_LMR)
      delete(lon_LMR)

    n=dimsizes(pr_avg_LMR)
    df=n-2
    ccr(nr) = escorc(pr_avg_LMR,pr_avg_GPCC) 
    t    = ccr(nr)*sqrt((n-2)/(1-ccr(nr)^2))      
    p(nr) = student_t(t, df)
   


  pr_clm=conform_dims((/dimsizes(pr_avg_LMR)/),dim_avg_n_Wrap(pr_avg_LMR,0),(/0/))
  aa=(pr_avg_LMR-pr_clm)^2
  tt_1=dim_sum_n_Wrap(aa,0)
  delete(aa)
  aa=(pr_avg_GPCC-pr_avg_LMR)^2
  tt_2=dim_sum_n_Wrap(aa,0)
  delete(aa)
  ci_0(nr)=1-tt_2/tt_1

  pr_clm=conform_dims((/dimsizes(pr_avg_GPCC)/),dim_avg_n_Wrap(pr_avg_GPCC,0),(/0/))
  aa=(pr_avg_GPCC-pr_clm)^2
  tt_1=dim_sum_n_Wrap(aa,0)
  delete(aa)
  aa=(pr_avg_LMR-pr_avg_GPCC)^2
  tt_2=dim_sum_n_Wrap(aa,0)
  delete(aa)
  ci_1(nr)=1-tt_2/tt_1
  end do

  
 
    write_table("correlation_"+Season2(ns)+".txt", "w", [/"         NA","Western US","Southeastern US","Northeast NA"/], "%s     %s     %s     %s     ")
    write_table("correlation_"+Season2(ns)+".txt", "a", [/ccr(0),ccr(1),ccr(2),ccr(3)/],"%7.3f %7.3f %7.3f %7.3f")
    write_table("correlation_"+Season2(ns)+".txt", "a", [/p(0),p(1),p(2),p(3)/],"%7.3f %7.3f %7.3f %7.3f")

   write_table("CE_"+Season2(ns)+".txt", "w", [/"         NA","Western US","Southeastern US","Northeast NA"/], "%s     %s     %s     %s     ")
    write_table("CE_"+Season2(ns)+".txt", "a", [/ci_0(0),ci_0(1),ci_0(2),ci_0(3)/],"%7.3f %7.3f %7.3f %7.3f")
    write_table("CE_"+Season2(ns)+".txt", "a", [/ci_1(0),ci_1(1),ci_1(2),ci_1(3)/],"%7.3f %7.3f %7.3f %7.3f")

end do
exit












