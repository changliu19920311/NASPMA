; evaluating the reconstucted monthly data=================
 load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
 load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
 load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
 load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"

;================set the parameters=====================
LatS=-60
LatN=60
LonW=-180
LonE=180
hgt_sel=500
year1=1901
year2= 2000
Year=ispan(year1,year2,1)
Season =(/"warm","cool"/) ;(/"warm","cool"/)
Season1=(/"Warm","Cool"/)
wks = gsn_open_wks("pdf" ,"evaluation")      ; send graphics to PNG file
gsn_define_colormap(wks,"amwg_blueyellowred") 
plot_rr_hgt = new(2,"graphic")
plot_CE_hgt = new(2,"graphic")
plot_rr_sst = new(2,"graphic")
plot_CE_sst = new(2,"graphic")
do ns=0,dimsizes(Season1)-1
print(ns)
  ;=======LMR_hgt==========
  diri1="//Users/jjyan/Desktop/NA_reconstrution/data/"+Season(ns)+"_season/hgt/"
  name1="hgt500_MCruns_ensemble_mean.nc"

  ;=================20CR hgt========
  diri2="/Users/jjyan/Desktop/LMR_test/data/data/analyses/20CR-V3/"
  name2="hgt.mon.mean.nc"

  ;=======LMR_tos==========
  diri3="/Users/jjyan/Desktop/NA_reconstrution/data/"+Season(ns)+"_season/sst/"
  name3="sst_MCruns_ensemble_mean.nc"

  ;=================HadISST========
  diri4="/Users/jjyan/Desktop/LMR_test/data/data/analyses/HadISST/"
  name4="HadISST_sst.nc"

  ;==========read the data==========

   ;=================read the LMR hgt data==================== 
  f1  = addfile(diri1+name1,"r")
  year=ispan(800,2000,1)
  hgt_lmr=f1->hgt500
  printVarSummary(hgt_lmr)
  hgt_lmr:=dim_avg_n_Wrap(hgt_lmr,1)
  hgt_lmr=lonFlip(hgt_lmr) 
  iyyyy=ind(year.ge.year1.and.year.le.year2)
  printVarSummary(hgt_lmr)
  hgt_lmr:= hgt_lmr(iyyyy,:,:)
  delete(iyyyy)
  delete(year)

  ;============resd LMR sst==============
  f3  = addfile(diri3+name3,"r")
  year=ispan(800,2000,1)
  sst_lmr=f3->sst
  sst_lmr:=dim_avg_n_Wrap(sst_lmr,1)
  sst_lmr=lonFlip(sst_lmr) 
  iyyyy=ind(year.ge.year1.and.year.le.year2)
  printVarSummary(sst_lmr)
  sst_lmr:= sst_lmr(iyyyy,:,:)
  delete(iyyyy)
  delete(year)

  ;==========read the 20CR hgt data ==========
  f2  = addfile(diri2+name2,"r")
  time=f2->time
  utc_date = cd_calendar(time, 0)
  year   = tointeger(utc_date(:,0))    ; Convert to integer for
  month  = tointeger(utc_date(:,1))
  levels=f2->level
  iyy=ind(levels.eq.500)
  hgt=f2->hgt(:,iyy,:,:)
  hgt=lonFlip(hgt) 
  printVarSummary(hgt)
  delete(iyy)
  HGT = new((/dimsizes(Year),dimsizes(hgt(0,:,0)),dimsizes(hgt(0,0,:))/),"float")
  if(Season(ns).eq."warm") then
    do ny=0,dimsizes(Year)-1
      iyyyy=ind(year.eq.Year(ny).and.month.ge.5.and.month.le.7)
      HGT(ny,:,:)=dim_avg_n_Wrap(hgt(iyyyy,:,:),0)
      delete(iyyyy)
    end do
  end if
  if(Season(ns).eq."cool") then
    do ny=0,dimsizes(Year)-1
      if(ny.eq.0) then
        iyyyy=ind(year.eq.Year(ny).and.month.le.4)
      else
        iyyyy=ind((year.eq.Year(ny).and.month.le.4).or.(year.eq.Year(ny-1).and.month.eq.12))
      end if
      HGT(ny,:,:)=dim_avg_n_Wrap(hgt(iyyyy,:,:),0)
      delete(iyyyy)
    end do
  end if

  delete(time)
  delete(year)
  delete(month)
  delete(utc_date)

  ;==============read HadISST=======================
  f4  = addfile(diri4+name4,"r")
  time=f4->time
  utc_date = cd_calendar(time, 0)
  year   = tointeger(utc_date(:,0))    ; Convert to integer for
  month   = tointeger(utc_date(:,1))    ; Convert to integer for 
  sst=f4->sst(:,::-1,:)
  printVarSummary(sst)
  sst = where(sst.le.-100,sst@_FillValue,sst)
  SST = new((/dimsizes(Year),dimsizes(sst(0,:,0)),dimsizes(sst(0,0,:))/),"float")
  if(Season(ns).eq."warm") then
    do ny=0,dimsizes(Year)-1
      iyyyy=ind(year.eq.Year(ny).and.month.ge.5.and.month.le.7)
      SST(ny,:,:)=dim_avg_n_Wrap(sst(iyyyy,:,:),0)
      delete(iyyyy)
    end do
  end if
  if(Season(ns).eq."cool") then
    do ny=0,dimsizes(Year)-1
      if(ny.eq.0) then
        iyyyy=ind(year.eq.Year(ny).and.month.le.4)
      else
        iyyyy=ind((year.eq.Year(ny).and.month.le.4).or.(year.eq.Year(ny-1).and.month.eq.12))
      end if
      SST(ny,:,:)=dim_avg_n_Wrap(sst(iyyyy,:,:),0)
      delete(iyyyy)
    end do
  end if

  delete(time)
  delete(year)
  delete(month)
  delete(utc_date)
  
  ;========all the data are regrid to teh LMR grid=======
  lat_hgt = hgt&lat
  lon_hgt = hgt&lon
  lat_sst = sst&latitude
  lon_sst = sst&longitude
  lat_lmr= hgt_lmr&lat
  lon_lmr= hgt_lmr&lon
  hgt_regrid := linint2_Wrap(lon_hgt,lat_hgt,HGT, False, lon_lmr,lat_lmr, 0)
  sst_regrid := linint2_Wrap(lon_sst,lat_sst,SST, False, lon_lmr,lat_lmr, 0)
  delete(SST)
  delete(HGT)
  ;==========adjust the climatology====================
  hgt_clm=conform_dims(dimsizes(hgt_regrid),dim_avg_n_Wrap(hgt_regrid,0),(/1,2/))
  hgt_regrid = hgt_regrid - hgt_clm
  sst_clm=conform_dims(dimsizes(sst_regrid),dim_avg_n_Wrap(sst_regrid,0),(/1,2/))
  sst_regrid = sst_regrid - sst_clm

  hgt_lmr_clm=conform_dims(dimsizes(hgt_lmr),dim_avg_n_Wrap(hgt_lmr,0),(/1,2/))
  hgt_lmr =hgt_lmr -hgt_lmr_clm
  sst_lmr_clm=conform_dims(dimsizes(sst_lmr),dim_avg_n_Wrap(sst_lmr,0),(/1,2/))
  sst_lmr =sst_lmr-sst_lmr_clm

  ;=======coefficient of efficiency=========
  hgt_regrid_mean=conform_dims(dimsizes(hgt_regrid),dim_avg_n_Wrap(hgt_regrid,0),(/1,2/))
  aa=(hgt_regrid-hgt_regrid_mean)*(hgt_regrid-hgt_regrid_mean)
  tt_1=dim_sum_n_Wrap(aa,0)
  aa=(hgt_regrid-hgt_lmr)^2
  tt_2=dim_sum_n_Wrap(aa,0)
  CE_hgt=1-tt_2/tt_1
  copy_VarCoords(hgt_lmr(0,:,:),CE_hgt)
  delete(aa)
  delete(tt_1) 
  delete(tt_2) 

  sst_regrid_mean=conform_dims(dimsizes(sst_regrid),dim_avg_n_Wrap(sst_regrid,0),(/1,2/))
  aa=(sst_regrid-sst_regrid_mean)*(sst_regrid-sst_regrid_mean)
  tt_1=dim_sum_n_Wrap(aa,0)
  aa=(sst_regrid-sst_lmr)^2
  tt_2=dim_sum_n_Wrap(aa,0)
  tt_1=where(tt_1.eq.0,tt_1@_FillValue,tt_1)
  CE_sst=1-tt_2/tt_1
  copy_VarCoords(sst_lmr(0,:,:),CE_sst)
  delete(aa)
  delete(tt_1) 
  delete(tt_2) 


  ;=========correlation=============================== 
  n=dimsizes(hgt_regrid(:,0,0))
  df=n-2
  rr_hgt   = escorc_n(hgt_regrid,hgt_lmr,0,0) 

  t    = rr_hgt*sqrt((n-2)/(1-rr_hgt^2))      
  p_rr_hgt = student_t(t, df)
  delete(t)
  copy_VarCoords(hgt_lmr(0,:,:),rr_hgt) 
  copy_VarCoords(hgt_lmr(0,:,:),p_rr_hgt) 
  P_rr = ndtooned(p_rr_hgt)
    igood = ind(.not.ismissing(P_rr))
    p = P_rr(igood)
    qsort(p)
    cv = p
    index_i = p
    index_i = index_i@_FillValue
    do i=0,dimsizes(p)-1
      cv(i) =tofloat((i+1))/tofloat(dimsizes(p))*0.05
      if(p(i).lt.cv(i))
          index_i(i) = 1
      end if
    end do
    i_cr =ind(index_i.eq.1) 

    print(i_cr)
    p_cr_hgt = p(max(i_cr))
    delete(p)
    delete(cv)
    delete(index_i)
    delete(i_cr)
    delete(igood)
    delete(P_rr)

  rr_sst   = escorc_n(sst_regrid,sst_lmr,0,0) 
  t    = rr_sst*sqrt((n-2)/(1-rr_sst^2))      
  p_rr_sst = student_t(t, df)
  delete(t)
  copy_VarCoords(sst_lmr(0,:,:),rr_sst) 
  copy_VarCoords(sst_lmr(0,:,:),p_rr_sst) 
  
  P_rr = ndtooned(p_rr_sst)
    igood = ind(.not.ismissing(P_rr))
    p = P_rr(igood)
    qsort(p)
    cv = p
    index_i = p
    index_i = index_i@_FillValue
    do i=0,dimsizes(p)-1
      cv(i) =tofloat((i+1))/tofloat(dimsizes(p))*0.05
      if(p(i).lt.cv(i))
          index_i(i) = 1
      end if
    end do
    i_cr =ind(index_i.eq.1) 
    p_cr_sst = p(max(i_cr))
    delete(p)
    delete(cv)
    delete(index_i)
    delete(i_cr)
    delete(igood)
    delete(P_rr)
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
  res@cnMinLevelValF       = -0.7                ; set min contour level
  res@cnMaxLevelValF       =  0.7                ; set max contour level
  res@cnLevelSpacingF      =  0.1  
  res@lbLabelBarOn        = False
  res@gsnLeftStringFontHeightF = 0.025
  res@cnLineLabelsOn = False
  res@gsnRightStringFontHeightF = 0.025
  res@gsnRightString =Season1(ns) + " season"
  res@gsnRightStringFontHeightF = 0.025
  if(ns.eq.0)
  res@gsnLeftString ="(a) r"
  else
  res@gsnLeftString ="(c) r"
  end if

  plot_rr_sst(ns) =gsn_csm_contour_map(wks,rr_sst,res)
  plot_rr_hgt(ns) =gsn_csm_contour_map(wks,rr_hgt,res)
  if(ns.eq.0)
  res@gsnLeftString ="(b) CE"
  else
  res@gsnLeftString ="(d) CE"
  end if
  plot_CE_hgt(ns)=gsn_csm_contour_map(wks,CE_hgt,res) 
  plot_CE_sst(ns)=gsn_csm_contour_map(wks,CE_sst,res)


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
    res2@cnLevels = p_cr_sst                     ;; set to significance level
    res2@cnFillPatterns = (/11,-1/)
    res2@cnFillColors = (/"black","white"/)
    res2@gsnLeftString = ""
    plot0 = gsn_csm_contour(wks,p_rr_sst,res2)
    res2@cnLevels = p_cr_hgt
    plot1 = gsn_csm_contour(wks,p_rr_hgt,res2)
    overlay(plot_rr_sst(ns),plot0)
    overlay(plot_rr_hgt(ns),plot1)
    delete(plot0)
    delete(plot1) 

 
  delete(hgt_lmr)
  delete(sst_lmr)
end do
 ;============Panel plot========================================
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
  gsn_panel(wks,(/plot_rr_sst(0),plot_CE_sst(0),plot_rr_sst(1),plot_CE_sst(1)/),(/2,2/),resP)
  gsn_panel(wks,(/plot_rr_hgt(0),plot_CE_hgt(0),plot_rr_hgt(1),plot_CE_hgt(1)/),(/2,2/),resP) 
exit






