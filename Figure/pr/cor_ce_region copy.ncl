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
PSM = (/"linear","bilinear"/)
latS   =   (/10,30,24,40,15/)
latN   =   (/55,55,40,55,25/)
lonL   =   (/-135,-125,-100,-100,-110/)
lonR   =   (/-60,-100,-70,-70,-85/)
region = (/"NA","West","South","Northeast"/)
data=(/"GPCC","CRU","WM"/)
name = (/"GPCC_precip.mon.flux.1x1.v6.nc","cru_ts4.01.1901.2016.pre.dat.nc","precip.mon.total.v501.nc"/)
pr_avg_data=new((/dimsizes(data),dimsizes(Year)/),"float")
Colors = (/"red","blue","purple"/)
wks = gsn_open_wks("pdf" ,"timeseries")      ; send graphics to PNG file
gsn_define_colormap(wks,"amwg_blueyellowred") 
plot=new((/dimsizes(Season1),dimsizes(region)/),"graphic")
do ns=0,dimsizes(Season1)-1
  do nr=0,dimsizes(region)-1
    ;======PDA_NASPA data==========
    diri1="/Users/chang/Desktop/NA_reconstruction/data/pda/"+Season(ns)+"_season/pre/"
    name1="prate_MCruns_ensemble_mean_1900_2000.nc"
    f1  = addfile(diri1+name1,"r")
    year=ispan(1900,2000,1)
    precip=f1->prate
    precip:=dim_avg_n_Wrap(precip,1)
    iyyyy=ind(year.ge.year1.and.year.le.year2)
    precip&lon=precip&lon-360
    printVarSummary(precip)
    Pre_LMR_NASPA = precip(iyyyy,{latS(nr):latN(nr)},{lonL(nr):lonR(nr)})
    delete(precip)
    delete(iyyyy)
    delete(year)
    do nd=0,dimsizes(data)-1
      ;=================Berkeley Earth Temperature===========
      diri3="/Users/chang/Desktop/LMR_Asia/data/data/analyses/"+data(nd)+"/"
      name3=name(nd)

      ;==========read the GPCC data， calculate the cool season average ==========
      f3  = addfile(diri3+name3,"r")
      time=f3->time
      utc_date = cd_calendar(time, 0)
      year   = tointeger(utc_date(:,0))    ; Convert to integer for
      month  = tointeger(utc_date(:,1))
      iyyyy=ind(year.ge.year1.and.year.le.year2)
      if(nd.eq.0) then
        precip=f3->precip(iyyyy,::-1,:)
        precip=lonFlip(precip) 
      end if
      if (nd.eq.1) then
        precip=f3->pre(iyyyy,:,:)
      end if
      if(nd.eq.2) then
        precip=f3->precip(iyyyy,:,:)
        precip=lonFlip(precip) 
        precip=precip*10/(86400*30)
      end if
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

      ;========all the data are regrid to teh gpcc grid=======
      lat_GPCC= Pre&lat
      lon_GPCC= Pre&lon
      printVarSummary(Pre_LMR_NASPA)
      lat_LMR= Pre_LMR_NASPA&lat
      lon_LMR= Pre_LMR_NASPA&lon
      Pre_LMR_NASPA_regrid := linint2_Wrap(lon_LMR,lat_LMR,Pre_LMR_NASPA, False, lon_GPCC,lat_GPCC, 0)
      if(nd.eq.1) then
        Pre =Pre/(86400*30.0)
      end if 

      ;==========adjust the climatology====================
      GPCC_clm=conform_dims(dimsizes(Pre),dim_avg_n_Wrap(Pre,0),(/1,2/))
      LMR_NASPA_clm=conform_dims((/dimsizes(Pre_LMR_NASPA_regrid)/),dim_avg_n_Wrap(Pre_LMR_NASPA_regrid,0),(/1,2/))
      Pre_LMR_NASPA_regrid=Pre_LMR_NASPA_regrid-LMR_NASPA_clm
      Pre=Pre-GPCC_clm
      delete(GPCC_clm)
      delete(LMR_NASPA_clm)

      ;========calculate the metrics =======================
      a    = addfile("$NCARG_ROOT/lib/ncarg/data/cdf/landsea.nc","r")
      lsdata = a->LSMASK
      lsm  = landsea_mask(lsdata,Pre&lat,Pre&lon)
      Pre= mask(Pre,lsm.eq.2,False)
      Pre = mask(Pre,lsm.eq.0,False)
      delete(lsm)

      ;========calculate the metrics =======================

      rad    = 4.0*atan(1.0)/180.0
      re     = 6371220.0
      rr     = re*rad
      clat   := cos(lat_GPCC*rad)
      pr_avg_LMR = wgt_areaave(Pre_LMR_NASPA_regrid(time|:,lat|:,lon|:), clat, 1.0, 0)
      printVarSummary(Pre)
      pr_avg_GPCC = wgt_areaave(Pre, clat, 1.0, 0)
      pr_avg_GPCC = pr_avg_GPCC*86400
      pr_avg_LMR = pr_avg_LMR *86400
      pr_avg_data(nd,:) = pr_avg_GPCC
      delete(Pre)
      delete(lat_GPCC)
      delete(lon_GPCC)
      delete(lat_LMR)
      delete(lon_LMR)
    end do
    delete(Pre_LMR_NASPA)     
    delete(Pre_LMR_NASPA_regrid)

    ;=================output the correlation================
    n=dimsizes(pr_avg_LMR)
    df=n-2
    ccr_0 = escorc(pr_avg_LMR,pr_avg_data(0,:)) 
    t    = ccr_0*sqrt((n-2)/(1-ccr_0^2))      
    p_0 = student_t(t, df)
    ccr_1 = escorc(pr_avg_LMR,pr_avg_data(1,:))
    t    = ccr_0*sqrt((n-2)/(1-ccr_1^2))      
    p_1 = student_t(t, df) 
    ccr_2 = escorc(pr_avg_LMR,pr_avg_data(2,:))
    t    = ccr_0*sqrt((n-2)/(1-ccr_2^2))      
    p_2 = student_t(t, df)  
    ccr_3 = escorc(pr_avg_data(0,:),pr_avg_data(1,:)) 
    t    = ccr_0*sqrt((n-2)/(1-ccr_3^2))      
    p_3 = student_t(t, df) 
    ccr_4 = escorc(pr_avg_data(0,:),pr_avg_data(2,:)) 
    t    = ccr_0*sqrt((n-2)/(1-ccr_4^2))      
    p_4 = student_t(t, df) 
    ccr_5 = escorc(pr_avg_data(1,:),pr_avg_data(2,:)) 
    t    = ccr_0*sqrt((n-2)/(1-ccr_5^2))      
    p_5 = student_t(t, df) 
    aa=(/"LMR","GPCC","CRU","WM"/)
    bb=(/1.0,ccr_0,ccr_1,ccr_2/)
    cc=(/ccr_0,1.0,ccr_3,ccr_4/)
    dd=(/ccr_1,ccr_3,1.0,ccr_5/)
    ee=(/ccr_2,ccr_4,ccr_5,1.0/)
    if(nr.eq.0) then 
      write_table("correlation_"+Season2(ns)+".txt", "w", [/region(nr)/], "  %s")
    else
      write_table("correlation_"+Season2(ns)+".txt", "a", [/region(nr)/], "  %s")
    end if
    write_table("correlation_"+Season2(ns)+".txt", "a", [/"         LMR","GPCC","CRU","WM"/], "%s     %s     %s     %s     %s ")
    write_table("correlation_"+Season2(ns)+".txt", "a", [/aa,bb,cc,dd,ee/],"%5s %7.3f %7.3f %7.3f %7.3f")


    bb=(/1.0,p_0,p_1,p_2/)
    cc=(/p_0,1.0,p_3,p_4/)
    dd=(/p_1,p_3,1.0,p_5/)
    ee=(/p_2,p_4,p_5,1.0/)
    if(nr.eq.0) then 
      write_table("correlation_"+Season2(ns)+"_p.txt", "w", [/region(nr)/], "  %s")
    else
      write_table("correlation_"+Season2(ns)+"_p.txt", "a", [/region(nr)/], "  %s")
    end if
    write_table("correlation_"+Season2(ns)+"_p.txt", "a", [/"         LMR","GPCC","CRU","WM"/], "%s     %s     %s     %s     %s ")
    write_table("correlation_"+Season2(ns)+"_p.txt", "a", [/aa,bb,cc,dd,ee/],"%5s %7.3f %7.3f %7.3f %7.3f")
    delete(aa)

   ;=================output the CE================

  pr_clm=conform_dims((/dimsizes(pr_avg_LMR)/),dim_avg_n_Wrap(pr_avg_LMR,0),(/0/))
  aa=(pr_avg_LMR-pr_clm)^2
  tt_1=dim_sum_n_Wrap(aa,0)
  delete(aa)
  aa=(pr_avg_data(0,:)-pr_avg_LMR)^2
  tt_2=dim_sum_n_Wrap(aa,0)
  delete(aa)
  ci_0=1-tt_2/tt_1

  aa=(pr_avg_data(1,:)-pr_avg_LMR)^2
  tt_2=dim_sum_n_Wrap(aa,0)
  delete(aa)
  ci_1=1-tt_2/tt_1

  aa=(pr_avg_data(2,:)-pr_avg_LMR)^2
  tt_2=dim_sum_n_Wrap(aa,0)
  delete(aa)
  ci_2=1-tt_2/tt_1
  delete(pr_clm)

  pr_clm=conform_dims((/dimsizes(pr_avg_data(0,:))/),dim_avg_n_Wrap(pr_avg_data(0,:),0),(/0/))
  aa=(pr_avg_data(0,:)-pr_clm)^2
  tt_1=dim_sum_n_Wrap(aa,0)
  delete(aa)
  aa=(pr_avg_LMR-pr_avg_data(0,:))^2
  tt_2=dim_sum_n_Wrap(aa,0)
  delete(aa)
  ci_3=1-tt_2/tt_1
  
  aa=(pr_avg_data(1,:)-pr_avg_data(0,:))^2
  tt_2=dim_sum_n_Wrap(aa,0)
  delete(aa)
  ci_4=1-tt_2/tt_1

  aa=(pr_avg_data(2,:)-pr_avg_data(0,:))^2
  tt_2=dim_sum_n_Wrap(aa,0)
  delete(aa)
  ci_5=1-tt_2/tt_1
  delete(pr_clm)

  pr_clm=conform_dims((/dimsizes(pr_avg_data(1,:))/),dim_avg_n_Wrap(pr_avg_data(1,:),0),(/0/))
  aa=(pr_avg_data(1,:)-pr_clm)^2
  tt_1=dim_sum_n_Wrap(aa,0)
  delete(aa)
  aa=(pr_avg_LMR-pr_avg_data(1,:))^2
  tt_2=dim_sum_n_Wrap(aa,0)
  delete(aa)
  ci_6=1-tt_2/tt_1

  aa=(pr_avg_data(0,:)-pr_avg_data(1,:))^2
  tt_2=dim_sum_n_Wrap(aa,0)
  delete(aa)
  ci_7=1-tt_2/tt_1

  aa=(pr_avg_data(2,:)-pr_avg_data(1,:))^2
  tt_2=dim_sum_n_Wrap(aa,0)
  delete(aa)
  ci_8=1-tt_2/tt_1
  delete(pr_clm)

  pr_clm=conform_dims((/dimsizes(pr_avg_data(2,:))/),dim_avg_n_Wrap(pr_avg_data(2,:),0),(/0/))
  aa=(pr_avg_data(2,:)-pr_clm)^2
  tt_1=dim_sum_n_Wrap(aa,0)
  delete(aa)
  aa=(pr_avg_LMR-pr_avg_data(2,:))^2
  tt_2=dim_sum_n_Wrap(aa,0)
  delete(aa)
  ci_9=1-tt_2/tt_1

  aa=(pr_avg_data(0,:)-pr_avg_data(2,:))^2
  tt_2=dim_sum_n_Wrap(aa,0)
  delete(aa)
  ci_10=1-tt_2/tt_1

  aa=(pr_avg_data(1,:)-pr_avg_data(2,:))^2
  tt_2=dim_sum_n_Wrap(aa,0)
  delete(aa)
  ci_11=1-tt_2/tt_1

  aa=(/"LMR","GPCC","CRU","WM"/)
  bb=(/1.0,ci_3,ci_6,ci_9/)
  cc=(/ci_0,1.0,ci_7,ci_10/)
  dd=(/ci_1,ci_4,1.0,ci_11/)
  ee=(/ci_2,ci_5,ci_8,1.0/)
  if(nr.eq.0) then 
    write_table("CE_"+Season2(ns)+".txt", "w", [/region(nr)/], "  %s")
  else
    write_table("CE_"+Season2(ns)+".txt", "a", [/region(nr)/], "  %s")
  end if
  write_table("CE_"+Season2(ns)+".txt", "a", [/"         LMR","GPCC","CRU","WM"/], "%s     %s     %s     %s     %s ")
  write_table("CE_"+Season2(ns)+".txt", "a", [/aa,bb,cc,dd,ee/],"%5s %7.3f %7.3f %7.3f %7.3f")
  end do
end do
exit












