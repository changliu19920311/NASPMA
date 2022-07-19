; evaluating the reconstucted monthly data=================
 load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
 load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
 load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
 load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"

;================set the parameters=====================
Season1=(/"WARM","COOL"/)
Season2=(/"Warm","Cool"/)
year1=1901
year2=1927
Year=ispan(year1,year2,1)
name = (/"20CR_NCEP_ERA5_JRA55_5000"/)
pri_num=(/"353"/)
latS   =   15.25
latN   =   54.75
lonL   =   -132.75
lonR   =   -58.25
plot_CE_dif = new(dimsizes(name),"graphic")
plot_rr_dif = new(dimsizes(name),"graphic")
plot_CE = new(dimsizes(name),"graphic")
plot_rr = new(dimsizes(name),"graphic")

wks = gsn_open_wks("pdf" ,"evaluation_pr_5000")      ; send graphics to PNG file
gsn_define_colormap(wks,"amwg_blueyellowred") 

do ns=0,dimsizes(Season1)-1
  ;======PDA_NASPA data==========
  diri1="/Users/chang/Desktop/LMR_NA/output/test_recon_season_GPCC_"+Season1(ns)+"_5000/"  
  name1="prate_MCruns_ensemble_mean.nc"
  ;=======proxy reconstructed data==========
  diri2="/Users/chang/Desktop/NASPA_Song/NASPA_recons_stats_XYfile/"
  name2="NASPA_"+Season2(ns)+".nc"
  ;=================GPCC===========
  diri3="/Users/chang/Desktop/LMR_test/data/data/analyses/GPCC/"
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
  delete(precip)
  delete(time)
  delete(year)
  delete(month)

  ;=================read the NASPA data==================== 
    f2  = addfile(diri2+name2,"r")
    year=ispan(0,2016,1)
    precip=f2->Pre
    iyyyy=ind(year.ge.year1.and.year.le.year2)
    Pre_NASPA = precip(iyyyy,:,:)
    delete(precip)
    delete(iyyyy)
    delete(year)
  ;=================read the LMR NASPA data==================== 
    f1  = addfile(diri1+name1,"r")
    year=ispan(1900,2000,1)
    precip=f1->prate
    precip:=dim_avg_n_Wrap(precip,1)
    iyyyy=ind(year.ge.year1.and.year.le.year2)
    precip&lon=precip&lon-360
    ;printVarSummary(precip)
    Pre_LMR_NASPA = precip(iyyyy,{latS:latN},{lonL:lonR})
    delete(precip)
    delete(iyyyy)
    delete(year)

;========all the data are regrid to teh gpcc grid=======
  lat_GPCC= Pre&lat
  lon_GPCC= Pre&lon
  lat_NASPA= Pre_NASPA&lat
  lon_NASPA= Pre_NASPA&lon
  lat_LMR= Pre_LMR_NASPA&lat
  lon_LMR= Pre_LMR_NASPA&lon

  Pre_NASPA_regrid := linint2_Wrap(lon_NASPA,lat_NASPA,Pre_NASPA, False,lon_GPCC,lat_GPCC, 0)
  Pre_LMR_NASPA_regrid := linint2_Wrap(lon_LMR,lat_LMR,Pre_LMR_NASPA, False, lon_GPCC,lat_GPCC, 0)

  Pre_LMR_regrid=new((/dimsizes(name),dimsizes(Pre_NASPA_regrid(:,0,0)),dimsizes(Pre_NASPA_regrid(0,:,0)),dimsizes(Pre_NASPA_regrid(0,0,:))/),"float")
  do na=0,dimsizes(name)-1
    diri="/Users/chang/Desktop/LMR_NA/output/test_recon_season_GPCC_"+Season1(ns)+"_"+name(na)+"/"
    Name="prate_MCruns_ensemble_mean.nc"
    f  = addfile(diri+Name,"r")
    year=ispan(1900,2000,1)
    precip=f->prate

    precip:=dim_avg_n_Wrap(precip,1)
    iyyyy=ind(year.ge.year1.and.year.le.year2)
    precip&lon=precip&lon-360
    ;printVarSummary(precip)
    Pre_LMR = precip(iyyyy,{latS:latN},{lonL:lonR})
    delete(precip)
    delete(iyyyy)
    delete(year)
    Pre_LMR_reg := linint2_Wrap(Pre_LMR&lon,Pre_LMR&lat,Pre_LMR, False, lon_GPCC,lat_GPCC, 0)
    Pre_LMR_regrid(na,:,:,:)=Pre_LMR_reg
    delete(Pre_LMR)
  end do

if(Season1(ns).eq."COOL") then
  Pre_LMR_NASPA_regrid=Pre_LMR_NASPA_regrid*86400*150
  Pre_LMR_regrid=Pre_LMR_regrid*86400*150
  Pre=Pre*86400*150
else
  Pre_LMR_NASPA_regrid=Pre_LMR_NASPA_regrid*86400*90
  Pre_LMR_regrid=Pre_LMR_regrid*86400*90
  Pre=Pre*86400*90
end if

;==========adjust the climatology====================
GPCC_clm=conform_dims(dimsizes(Pre),dim_avg_n_Wrap(Pre,0),(/1,2/))
NASPA_clm=conform_dims((/dimsizes(Pre_NASPA_regrid)/),dim_avg_n_Wrap(Pre_NASPA_regrid,0),(/1,2/))
LMR_NASPA_clm=conform_dims((/dimsizes(Pre_LMR_NASPA_regrid)/),dim_avg_n_Wrap(Pre_LMR_NASPA_regrid,0),(/1,2/))
LMR_regrid_clm=conform_dims((/dimsizes(Pre_LMR_regrid)/),dim_avg_n_Wrap(Pre_LMR_regrid,1),(/0,2,3/))

Pre_NASPA_regrid=Pre_NASPA_regrid-NASPA_clm
Pre=Pre-GPCC_clm
Pre_LMR_NASPA_regrid=Pre_LMR_NASPA_regrid-LMR_NASPA_clm
Pre_LMR_regrid=Pre_LMR_regrid-LMR_regrid_clm
Pre&lat@units="degrees_north"
Pre&lon@units="degrees_east"
printVarSummary(Pre&lat)
;========calculate the metrics =======================

    ;=======coefficient of efficiency=========

    Pre_GPCC_mean=conform_dims(dimsizes(Pre),dim_avg_n_Wrap(Pre,0),(/1,2/))
    aa=(Pre-Pre_GPCC_mean)*(Pre-Pre_GPCC_mean)
    tt_1=dim_sum_n_Wrap(aa,0)
    aa=(Pre_NASPA_regrid-Pre)*(Pre_NASPA_regrid-Pre)
    tt_2=dim_sum_n_Wrap(aa,0)
    delete(aa)
    CE_NASPA=1-tt_2/tt_1
    copy_VarCoords(Pre(0,:,:),CE_NASPA)    

    aa=(Pre_LMR_NASPA_regrid- Pre)^2
    tt_2=dim_sum_n_Wrap(aa,0)
    CE_LMR_NASPA=1-tt_2/tt_1
    copy_VarCoords(Pre(0,:,:),CE_LMR_NASPA)
    delete(aa)   
    CE_NASPA_dif=CE_LMR_NASPA-CE_NASPA ;CE_LMR_NASPA ;
    copy_VarCoords(Pre(0,:,:),CE_NASPA_dif) 
    pos_num:=num(CE_NASPA_dif.gt.0)
    neg_num:=num(CE_NASPA_dif.lt.0)
    CE_NASPA_dif_percent =tofloat(pos_num)/tofloat(pos_num+neg_num)*100

    CE_dif_percent=new(dimsizes(name),"float")
    CE_LMR = new((/dimsizes(name),dimsizes(CE_NASPA(:,0)),dimsizes(CE_NASPA(0,:))/),"float")
    CE_dif = new((/dimsizes(name),dimsizes(CE_NASPA(:,0)),dimsizes(CE_NASPA(0,:))/),"float")

    do na=0,dimsizes(name)-1
      aa=(Pre_LMR_regrid(na,:,:,:)- Pre)^2
      tt_2=dim_sum_n_Wrap(aa,0)
      CE_LMR(na,:,:)=1-tt_2/tt_1
      delete(aa)
      CE_dif(na,:,:)=CE_LMR(na,:,:)-CE_NASPA ;CE_LMR_NASPA ;
      pos_num:=num(CE_dif(na,:,:).gt.0)
      neg_num:=num(CE_dif(na,:,:).lt.0)
      CE_dif_percent(na) = tofloat(pos_num)/tofloat(pos_num+neg_num)*100
    end do
    copy_VarCoords(Pre(0,:,:),CE_LMR(0,:,:))  
    copy_VarCoords(Pre(0,:,:),CE_dif(0,:,:))  
        ;=========correlation=============================== 
    n=dimsizes(Pre(:,0,0))
    df=n-2
    rr_LMR_NASPA   = escorc_n(Pre,Pre_LMR_NASPA_regrid,0,0) 
    t    = rr_LMR_NASPA*sqrt((n-2)/(1-rr_LMR_NASPA^2))      
    p_rr_LMR_NASPA = student_t(t, df)
    delete(t)
    copy_VarCoords(Pre(0,:,:),rr_LMR_NASPA) 
    copy_VarCoords(Pre(0,:,:),p_rr_LMR_NASPA) 

    rr_NASPA   = escorc_n(Pre,Pre_NASPA_regrid,0,0) 
    t    = rr_NASPA*sqrt((n-2)/(1-rr_NASPA^2))      
    p_rr_NASPA = student_t(t, df)
    delete(t)
    copy_VarCoords(Pre(0,:,:),rr_NASPA) 
    copy_VarCoords(Pre(0,:,:),p_rr_NASPA) 
    rr_NASPA_dif=rr_LMR_NASPA-rr_NASPA ;rr_LMR_NASPA;
    copy_VarCoords(Pre(0,:,:),rr_NASPA_dif) 
    pos_num:=num(rr_NASPA_dif.gt.0)
    neg_num:=num(rr_NASPA_dif.lt.0)
    rr_NASPA_dif_percent = tofloat(pos_num)/tofloat(pos_num+neg_num)*100

    rr_dif_percent=new(dimsizes(name),"float")
    rr_LMR = new((/dimsizes(name),dimsizes(rr_NASPA(:,0)),dimsizes(rr_NASPA(0,:))/),"float")
    rr_dif = new((/dimsizes(name),dimsizes(rr_NASPA(:,0)),dimsizes(rr_NASPA(0,:))/),"float")
    do na=0,dimsizes(name)-1
      rr_LMR(na,:,:)  = escorc_n(Pre,Pre_LMR_regrid(na,:,:,:),0,0) 
      t    = rr_LMR(na,:,:)*sqrt((n-2)/(1-rr_LMR(na,:,:)^2))      
      p_rr_LMR = student_t(t, df)
      delete(t)

      rr_dif(na,:,:)=rr_LMR(na,:,:)-rr_NASPA ;rr_LMR_NASPA;
      copy_VarCoords(Pre(0,:,:),rr_dif(na,:,:)) 
      pos_num:=num(rr_dif(na,:,:).gt.0)
      neg_num:=num(rr_dif(na,:,:).lt.0)
      rr_dif_percent(na) = tofloat(pos_num)/tofloat(pos_num+neg_num)*100
    end do
      copy_VarCoords(Pre(0,:,:),rr_LMR(0,:,:)) 
      copy_VarCoords(Pre(0,:,:),rr_dif(0,:,:)) 
rr_LMR&lat@units="degrees_north"
rr_LMR&lon@units="degrees_east"
rr_dif&lat@units="degrees_north"
rr_dif&lon@units="degrees_east"
CE_LMR&lat@units="degrees_north"
CE_LMR&lon@units="degrees_east"
CE_dif&lat@units="degrees_north"
CE_dif&lon@units="degrees_east"

;========Ploting===============
  
  res                       = True     ; plot mods desired
  res@cnFillOn              = True     ; turn on color fill
  res@cnLinesOn             = False    ; turn of contour lines
  res@gsnDraw = False 
  res@gsnFrame = False 
  res@mpFillOn             = False              
  res@cnFillOn              = True     ; turn on color fill
  res@cnLinesOn             = False    ; turn of contour lines
  res@cnLevelSelectionMode = "ManualLevels"  

  res@cnMinLevelValF       = -1.                ; set min contour level
  res@cnMaxLevelValF       =  1.                ; set max contour level
  res@cnLevelSpacingF      =  0.05  
  res@gsnAddCyclic         = False        ; plotted data are not cyclic
     ; turn off map fill
  res@mpMinLatF            = latS         ; zoom in on map
  res@mpMaxLatF            = latN
  res@mpMinLonF            = lonL
  res@mpMaxLonF            = lonR      
  res@lbLabelBarOn        = False
  res@cnFillMode = "RasterFill"
  res@gsnLeftString ="Cor,NASPA"
  plot_rr_NASPA=gsn_csm_contour_map(wks,rr_NASPA,res)
  res@gsnLeftString ="Cor,PDA (ccsm4)"
  plot_rr_LMR_NASPA=gsn_csm_contour_map(wks,rr_LMR_NASPA,res)

  do na=0,dimsizes(name)-1
    res@gsnLeftString ="Cor,PDA ("+name(na)+":"+pri_num(na)+")"
    plot_rr(na)=gsn_csm_contour_map(wks,rr_LMR(na,:,:),res)
  end do

  res@gsnLeftString ="Cor,Dif (ccsm4)"
  res@gsnRightString ="Positive grids = "+sprintf("%5.2f", rr_NASPA_dif_percent)+"%"
  plot_rr_NASPA_DIF=gsn_csm_contour_map(wks,rr_NASPA_dif,res)

  do na=0,dimsizes(name)-1
    res@gsnLeftString ="Cor,Dif ("+name(na)+":"+pri_num(na)+")"
    res@gsnRightString ="Positive grids = "+sprintf("%5.2f", rr_dif_percent(na))+"%"
    plot_rr_dif(na)=gsn_csm_contour_map(wks,rr_dif(na,:,:),res)
  end do
  delete(res@gsnRightString)
  
  res@gsnLeftString ="CE,NASPA"
  plot_CE_NASPA=gsn_csm_contour_map(wks,CE_NASPA,res)
  res@gsnLeftString ="CE,PDA (ccsm4)"
  plot_CE_LMR_NASPA=gsn_csm_contour_map(wks,CE_LMR_NASPA,res)
  do na=0,dimsizes(name)-1
    res@gsnLeftString ="CE,PDA ("+name(na)+":"+pri_num(na)+")"
    plot_CE(na)=gsn_csm_contour_map(wks,CE_LMR(na,:,:),res)
  end do

  res@gsnLeftString ="CE,Dif (ccsm4)"
  res@gsnRightString ="Positive grids = "+sprintf("%5.2f", CE_NASPA_dif_percent)+"%"
  plot_CE_NASPA_DIF=gsn_csm_contour_map(wks,CE_NASPA_dif,res)

  do na=0,dimsizes(name)-1
    res@gsnLeftString ="CE,Dif ("+name(na)+":"+pri_num(na)+")"
    res@gsnRightString ="Positive grids = "+sprintf("%5.2f", CE_dif_percent(na))+"%"
    plot_CE_dif(na)=gsn_csm_contour_map(wks,CE_dif(na,:,:),res)
  end do
  delete(res@gsnRightString)
  delete(res)

  resP                     = True         ; modify the panel plot
  resP@gsnPanelTop   = 0.85   
  resP@gsnPanelBottom   = 0.15
  resP@gsnPanelYWhiteSpacePercent = 5
  resP@gsnPanelXWhiteSpacePercent = 5
  resP@lbLabelFontHeightF  = 0.007  
  resP@gsnPanelLabelBar    = True       ; common label bar
  resP@lbOrientation       = "horizontal" ; vertical label bar
  resP@pmLabelBarWidthF    = 0.6      ; default is shorter
  resP@pmLabelBarHeightF   = 0.055        ; default is taller
  resP@lbLabelFontHeightF  = 0.008       ; make labels smaller ( default=0.02 )
  resP@gsnPanelMainString = Season2(ns)+" Season"
  gsn_panel(wks,(/plot_rr_LMR_NASPA,plot_rr_NASPA_DIF,plot_rr(0),plot_rr_dif(0)/),(/2,2/),resP) 
  gsn_panel(wks,(/plot_CE_LMR_NASPA,plot_CE_NASPA_DIF,plot_CE(0),plot_CE_dif(0)/),(/2,2/),resP) 
  delete(resP)
end do
exit












