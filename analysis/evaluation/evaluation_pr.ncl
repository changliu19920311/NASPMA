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

latS   =   15.25
latN   =   54.75
lonL   =   -132.75
lonR   =   -58.25

wks = gsn_open_wks("pdf" ,"evaluation_pr")      ; send graphics to PNG file
gsn_define_colormap(wks,"amwg_blueyellowred") 

do ns=0,dimsizes(Season1)-1
  ;======PDA_NASPA data==========
  diri1="/Users/chang/Desktop/LMR_NA/output/output_NASPA/test_recon_season_GPCC_"+Season1(ns)+"/"
  name1="prate_MCruns_ensemble_mean.nc"
  ;=======proxy reconstructed data==========
  diri2="/Users/chang/Desktop/NASPA_Song/NASPA_recons_stats_XYfile/"
  name2="NASPA_"+Season2(ns)+".nc"
  ;=================Berkeley Earth Temperature===========
  diri3="/Users/chang/Desktop/LMR_test/data/data/analyses/GPCC/"
  name3="GPCC_precip.mon.flux.1x1.v6.nc"
  ;======PDA_global data==========
  diri4="/Users/chang/Desktop/LMR_NA/output/output_global/test_recon_season_GPCC_"+Season1(ns)+"/"
  name4="prate_MCruns_ensemble_mean.nc"
  ;======PDA_NASPA+other types proxy data==========
  diri5="/Users/chang/Desktop/LMR_NA/output/output_NASPA2/test_recon_season_GPCC_"+Season1(ns)+"/"
  name5="prate_MCruns_ensemble_mean.nc"

  ;==========read the GPCC data， calculate the cool season average ==========
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

  ;=======read the LMR_GLOBAL data===================
    f4  = addfile(diri4+name4,"r")
    year=ispan(1900,2000,1)
    precip=f4->prate
    precip:=dim_avg_n_Wrap(precip,1)
    iyyyy=ind(year.ge.year1.and.year.le.year2)
    precip&lon=precip&lon-360
    ;printVarSummary(precip)
    Pre_LMR_Global = precip(iyyyy,{latS:latN},{lonL:lonR})
    delete(precip)
    delete(iyyyy)
    delete(year)

  ;=======read the LMR_NASPA2 data===================
    f5  = addfile(diri5+name4,"r")
    year=ispan(1900,2000,1)
    precip=f5->prate
    precip:=dim_avg_n_Wrap(precip,1)
    iyyyy=ind(year.ge.year1.and.year.le.year2)
    precip&lon=precip&lon-360
    ;printVarSummary(precip)
    Pre_LMR_NASPA2 = precip(iyyyy,{latS:latN},{lonL:lonR})
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
  Pre_LMR_Global_regrid := linint2_Wrap(lon_LMR,lat_LMR,Pre_LMR_Global, False, lon_GPCC,lat_GPCC, 0)
  Pre_LMR_NASPA2_regrid := linint2_Wrap(lon_LMR,lat_LMR,Pre_LMR_NASPA2, False, lon_GPCC,lat_GPCC, 0)

if(Season1(ns).eq."COOL") then
  Pre_LMR_NASPA_regrid=Pre_LMR_NASPA_regrid*86400*150
  Pre_LMR_Global_regrid=Pre_LMR_Global_regrid*86400*150
  Pre_LMR_NASPA2_regrid=Pre_LMR_NASPA2_regrid*86400*150
  Pre=Pre*86400*150
end if
if(Season1(ns).eq."WARM") then
  Pre_LMR_NASPA_regrid=Pre_LMR_NASPA_regrid*86400*90
  Pre_LMR_NASPA2_regrid=Pre_LMR_NASPA2_regrid*86400*90
  Pre_LMR_Global_regrid=Pre_LMR_Global_regrid*86400*90
  Pre=Pre*86400*90
end if

;==========adjust the climatology====================
GPCC_clm=conform_dims(dimsizes(Pre),dim_avg_n_Wrap(Pre,0),(/1,2/))
LMR_Global_clm=conform_dims((/dimsizes(Pre_LMR_Global_regrid)/),dim_avg_n_Wrap(Pre_LMR_Global_regrid,0),(/1,2/))
LMR_NASPA_clm=conform_dims((/dimsizes(Pre_LMR_NASPA_regrid)/),dim_avg_n_Wrap(Pre_LMR_NASPA_regrid,0),(/1,2/))
LMR_NASPA2_clm=conform_dims((/dimsizes(Pre_LMR_NASPA2_regrid)/),dim_avg_n_Wrap(Pre_LMR_NASPA2_regrid,0),(/1,2/))
NASPA_clm=conform_dims((/dimsizes(Pre_NASPA_regrid)/),dim_avg_n_Wrap(Pre_NASPA_regrid,0),(/1,2/))
;Pre_LMR_regrid=Pre_LMR_regrid/(LMR_clm+0.0001)*GPCC_clm

Pre_NASPA_regrid=Pre_NASPA_regrid-NASPA_clm
Pre_LMR_NASPA2_regrid=Pre_LMR_NASPA2_regrid-LMR_NASPA2_clm
Pre_LMR_NASPA_regrid=Pre_LMR_NASPA_regrid-LMR_NASPA_clm
Pre_LMR_Global_regrid=Pre_LMR_Global_regrid-LMR_Global_clm
Pre=Pre-GPCC_clm
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

    aa=(Pre_LMR_Global_regrid- Pre)^2
    tt_2=dim_sum_n_Wrap(aa,0)
    CE_LMR_Global=1-tt_2/tt_1
    copy_VarCoords(Pre(0,:,:),CE_LMR_Global)  
    delete(aa)

    aa=(Pre_LMR_NASPA2_regrid- Pre)^2
    tt_2=dim_sum_n_Wrap(aa,0)
    CE_LMR_NASPA2=1-tt_2/tt_1
    copy_VarCoords(Pre(0,:,:),CE_LMR_NASPA2)
    delete(aa)   

    CE_NASPA_dif=CE_LMR_NASPA-CE_NASPA
    copy_VarCoords(Pre(0,:,:),CE_NASPA_dif) 
    pos_num:=num(CE_NASPA_dif.gt.0)
    neg_num:=num(CE_NASPA_dif.lt.0)
    CE_NASPA_dif_percent = tofloat(pos_num)/tofloat(pos_num+neg_num)*100
    
    CE_NASPA2_dif=CE_LMR_NASPA2-CE_NASPA
    copy_VarCoords(Pre(0,:,:),CE_NASPA2_dif) 
    pos_num:=num(CE_NASPA2_dif.gt.0)
    neg_num:=num(CE_NASPA2_dif.lt.0)
    CE_NASPA2_dif_percent = tofloat(pos_num)/tofloat(pos_num+neg_num)*100

    CE_NASPA2_dif2=CE_LMR_NASPA2-CE_LMR_NASPA
    copy_VarCoords(Pre(0,:,:),CE_NASPA2_dif2) 
    pos_num:=num(CE_NASPA2_dif2.gt.0)
    neg_num:=num(CE_NASPA2_dif2.lt.0)
    CE_NASPA2_dif2_percent = tofloat(pos_num)/tofloat(pos_num+neg_num)*100

    CE_Global_dif=CE_LMR_Global-CE_NASPA
    copy_VarCoords(Pre(0,:,:),CE_Global_dif) 
    pos_num:=num(CE_Global_dif.gt.0)
    neg_num:=num(CE_Global_dif.lt.0)
    CE_Global_dif_percent = tofloat(pos_num)/tofloat(pos_num+neg_num)*100

    CE_Global_dif2=CE_LMR_Global-CE_LMR_NASPA
    copy_VarCoords(Pre(0,:,:),CE_Global_dif2) 
    pos_num:=num(CE_Global_dif2.gt.0)
    neg_num:=num(CE_Global_dif2.lt.0)
    CE_Global_dif2_percent = tofloat(pos_num)/tofloat(pos_num+neg_num)*100

    CE_Global_dif3=CE_LMR_Global-CE_LMR_NASPA2
    copy_VarCoords(Pre(0,:,:),CE_Global_dif3) 
    pos_num:=num(CE_Global_dif3.gt.0)
    neg_num:=num(CE_Global_dif3.lt.0)
    CE_Global_dif3_percent = tofloat(pos_num)/tofloat(pos_num+neg_num)*100


        ;=========correlation=============================== 
    n=dimsizes(Pre(:,0,0))
    df=n-2
    rr_LMR_NASPA   = escorc_n(Pre,Pre_LMR_NASPA_regrid,0,0) 
    t    = rr_LMR_NASPA*sqrt((n-2)/(1-rr_LMR_NASPA^2))      
    p_rr_LMR_NASPA = student_t(t, df)
    delete(t)
    copy_VarCoords(Pre(0,:,:),rr_LMR_NASPA) 
    copy_VarCoords(Pre(0,:,:),p_rr_LMR_NASPA) 

    rr_LMR_Global  = escorc_n(Pre,Pre_LMR_Global_regrid,0,0) 
    t    = rr_LMR_Global*sqrt((n-2)/(1-rr_LMR_Global^2))      
    p_rr_LMR_Global = student_t(t, df)
    delete(t)
    copy_VarCoords(Pre(0,:,:),rr_LMR_Global) 
    copy_VarCoords(Pre(0,:,:),p_rr_LMR_Global) 

    rr_NASPA   = escorc_n(Pre,Pre_NASPA_regrid,0,0) 
    t    = rr_NASPA*sqrt((n-2)/(1-rr_NASPA^2))      
    p_rr_NASPA = student_t(t, df)
    delete(t)
    copy_VarCoords(Pre(0,:,:),rr_NASPA) 
    copy_VarCoords(Pre(0,:,:),p_rr_NASPA) 

    rr_LMR_NASPA2   = escorc_n(Pre,Pre_LMR_NASPA2_regrid,0,0) 
    t    = rr_LMR_NASPA2*sqrt((n-2)/(1-rr_LMR_NASPA2^2))      
    p_rr_LMR_NASPA2 = student_t(t, df)
    delete(t)
    copy_VarCoords(Pre(0,:,:),rr_LMR_NASPA2) 
    copy_VarCoords(Pre(0,:,:),p_rr_LMR_NASPA2) 

    rr_NASPA_dif=rr_LMR_NASPA-rr_NASPA
    copy_VarCoords(Pre(0,:,:),rr_NASPA_dif) 
    pos_num:=num(rr_NASPA_dif.gt.0)
    neg_num:=num(rr_NASPA_dif.lt.0)
    rr_NASPA_dif_percent = tofloat(pos_num)/tofloat(pos_num+neg_num)*100

    rr_LMR_NASPA2_dif=rr_LMR_NASPA2-rr_NASPA
    copy_VarCoords(Pre(0,:,:),rr_LMR_NASPA2_dif) 
    pos_num:=num(rr_LMR_NASPA2_dif.gt.0)
    neg_num:=num(rr_LMR_NASPA2_dif.lt.0)
    rr_NASPA2_dif_percent = tofloat(pos_num)/tofloat(pos_num+neg_num)*100

    rr_LMR_NASPA2_dif2=rr_LMR_NASPA2-rr_LMR_NASPA
    copy_VarCoords(Pre(0,:,:),rr_LMR_NASPA2_dif2) 
    pos_num:=num(rr_LMR_NASPA2_dif2.gt.0)
    neg_num:=num(rr_LMR_NASPA2_dif2.lt.0)
    rr_NASPA2_dif2_percent = tofloat(pos_num)/tofloat(pos_num+neg_num)*100


    rr_Global_dif=rr_LMR_Global-rr_NASPA
    copy_VarCoords(Pre(0,:,:),rr_Global_dif) 
    pos_num:=num(rr_Global_dif.gt.0)
    neg_num:=num(rr_Global_dif.lt.0)
    rr_Global_dif_percent = tofloat(pos_num)/tofloat(pos_num+neg_num)*100

    rr_Global_dif2=rr_LMR_Global-rr_LMR_NASPA
    copy_VarCoords(Pre(0,:,:),rr_Global_dif2) 
    pos_num:=num(rr_Global_dif2.gt.0)
    neg_num:=num(rr_Global_dif2.lt.0)
    rr_Global_dif2_percent = tofloat(pos_num)/tofloat(pos_num+neg_num)*100

    rr_Global_dif3=rr_LMR_Global-rr_LMR_NASPA2
    copy_VarCoords(Pre(0,:,:),rr_Global_dif3) 
    pos_num:=num(rr_Global_dif3.gt.0)
    neg_num:=num(rr_Global_dif3.lt.0)
    rr_Global_dif3_percent = tofloat(pos_num)/tofloat(pos_num+neg_num)*100

print((/CE_NASPA_dif_percent,CE_NASPA2_dif_percent,CE_NASPA2_dif2_percent,CE_Global_dif_percent,CE_Global_dif2_percent,CE_Global_dif3_percent,rr_NASPA_dif_percent,rr_NASPA2_dif_percent,rr_NASPA2_dif2_percent,rr_Global_dif_percent,rr_Global_dif2_percent,rr_Global_dif3_percent/))
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
  res@gsnLeftString ="Cor,PDA (NA tree ring)"
  plot_rr_LMR_NASPA=gsn_csm_contour_map(wks,rr_LMR_NASPA,res)
  res@gsnLeftString ="Cor,PDA (NASPA tree ring + others)"
  plot_rr_LMR_NASPA2=gsn_csm_contour_map(wks,rr_LMR_NASPA2,res)
  res@gsnLeftString ="Cor,PDA (Global proxies)"
  plot_rr_LMR_Global=gsn_csm_contour_map(wks,rr_LMR_Global,res)
  res@gsnLeftString ="Cor,Dif (NA tree ting)"
  plot_rr_NASPA_DIF=gsn_csm_contour_map(wks,rr_NASPA_dif,res)
  res@gsnLeftString ="Cor,Dif (NASPA tree ring + others)"
  plot_rr_NASPA2_DIF=gsn_csm_contour_map(wks,rr_LMR_NASPA2_dif,res)
  res@gsnLeftString ="Cor,Dif2 (NASPA tree ring + others)"
  plot_rr_NASPA2_DIF2=gsn_csm_contour_map(wks,rr_LMR_NASPA2_dif2,res)
  res@gsnLeftString ="Cor,Dif (Global proxies)"
  plot_rr_Global_DIF=gsn_csm_contour_map(wks,rr_Global_dif,res)
  res@gsnLeftString ="Cor,Dif2 (Global proxies)"
  plot_rr_Global_DIF2=gsn_csm_contour_map(wks,rr_Global_dif2,res)
  res@gsnLeftString ="Cor,Dif3 (Global proxies)"
  plot_rr_Global_DIF3=gsn_csm_contour_map(wks,rr_Global_dif3,res)


  res@gsnLeftString ="CE,NASPA"
  plot_CE_NASPA=gsn_csm_contour_map(wks,CE_NASPA,res)
  res@gsnLeftString ="CE,PDA (NA tree ring)"
  plot_CE_LMR_NASPA=gsn_csm_contour_map(wks,CE_LMR_NASPA,res)
  res@gsnLeftString ="CE,PDA (NA tree ring + others)"
  plot_CE_LMR_NASPA2=gsn_csm_contour_map(wks,CE_LMR_NASPA2,res)
  res@gsnLeftString ="CE,PDA (Global proxies)"
  plot_CE_LMR_Global=gsn_csm_contour_map(wks,CE_LMR_Global,res)
  res@gsnLeftString ="CE,Dif (NA tree ting)"
  plot_CE_NASPA_DIF=gsn_csm_contour_map(wks,CE_NASPA_dif,res)
  res@gsnLeftString ="CE,Dif (NA tree ting + others)"
  plot_CE_NASPA2_DIF=gsn_csm_contour_map(wks,CE_NASPA2_dif,res)
  res@gsnLeftString ="CE,Dif2 (NA tree ting + others)"
  plot_CE_NASPA2_DIF2=gsn_csm_contour_map(wks,CE_NASPA2_dif2,res)
  res@gsnLeftString ="CE,Dif (Global proxies)"
  plot_CE_Global_DIF=gsn_csm_contour_map(wks,CE_Global_dif,res)
  res@gsnLeftString ="CE,Dif2 (Global proxies)"
  plot_CE_Global_DIF2=gsn_csm_contour_map(wks,CE_Global_dif2,res)
  res@gsnLeftString ="CE,Dif3 (Global proxies)"
  plot_CE_Global_DIF3=gsn_csm_contour_map(wks,CE_Global_dif3,res)

  res2 = True
  res2@gsnDraw = False
  res2@gsnFrame = False
  res2@cnFillOn = True
  res2@cnLinesOn = False
  res2@cnLineLabelsOn = False
  res2@cnInfoLabelOn = False
  res2@lbLabelBarOn = False
  res2@cnMonoFillPattern = False
  res2@cnLevelSelectionMode = "ExplicitLevels"
  res2@cnLevels = (/0.05/)                       ;; set to significance level
  res2@cnFillPatterns = (/11,-1/)
  res2@cnFillColors = (/"black","white"/)
  res2@gsnLeftString = ""
  ;plot0 = gsn_csm_contour(wks,p_rr_NASPA,res2)
  ;plot1 = gsn_csm_contour(wks,p_rr_LMR_NASPA,res2)
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
  resP@gsnPanelRowSpec = True 
  resP@gsnPanelCenter  = False
  gsn_panel(wks,(/plot_rr_NASPA,plot_CE_NASPA,plot_rr_LMR_NASPA,plot_CE_LMR_NASPA,plot_rr_NASPA_DIF,plot_CE_NASPA_DIF,plot_rr_LMR_NASPA2,plot_CE_LMR_NASPA2,plot_rr_NASPA2_DIF,plot_CE_NASPA2_DIF,plot_rr_LMR_Global,plot_CE_LMR_Global,plot_rr_Global_DIF,plot_CE_Global_DIF/),(/2,4,4,4/),resP) 
  gsn_panel(wks,(/plot_rr_NASPA2_DIF2,plot_CE_NASPA2_DIF2,plot_rr_Global_DIF2,plot_CE_Global_DIF2,plot_rr_Global_DIF3,plot_CE_Global_DIF3/),(/2,2,2/),resP) 
  delete(resP)
end do
exit












