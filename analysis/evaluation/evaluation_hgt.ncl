; evaluating the reconstucted monthly data=================
 load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
 load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
 load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
 load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"

;================set the parameters=====================
Season1=(/"WARM","COOL"/)
Season2=(/"Warm","Cool"/)
hgt_sel=(/500,700/)
year1=1901
year2=2000
Year=ispan(year1,year2,1)
latS   =   15.25
latN   =   54.75
lonL   =   -132.75
lonR   =  -58.25

wks = gsn_open_wks("pdf" ,"evaluation_hgt")      ; send graphics to PNG file
gsn_define_colormap(wks,"amwg_blueyellowred") 
plot_rr=new((/2,2/),"graphic")
plot_CE=new((/2,2/),"graphic")
plot_cor=new((/2,2/),"graphic")

do ns=0,dimsizes(Season1)-1
  do nh=0,dimsizes(Season2)-1
  ;======LMR data==========
  diri1="/Users/chang/Desktop/LMR_NA/output/output_NASPA/test_recon_season_GPCC_"+Season1(ns)+"/"
  name1="hgt"+tostring(hgt_sel(nh))+"_MCruns_ensemble_mean.nc"

  ;======CCSM4 hgt500 data==========
  diri2="/Users/chang/Desktop/LMR_NA/output/output_NASPA/test_recon_season_GPCC_"+Season1(ns)+"/"
  name2="clm_zg_"+tostring(hgt_sel(nh))+"hpa_Amon_CCSM4_past1000_085001-185012.nc"

  ;=================Berkeley Earth Temperature
  diri3="/Users/chang/Desktop/LMR_test/data/data/analyses/20CR_V2/"
  name3="hgt.mon.mean.nc"

  ;==========read the 20CR_V2data,calculate the cool season average ==========

  f3  = addfile(diri3+name3,"r")
  time=f3->time
  utc_date = cd_calendar(time, 0)
  year   = tointeger(utc_date(:,0))    ; Convert to integer for
  month  = tointeger(utc_date(:,1))
  iyyyy=ind(year.ge.year1.and.year.le.year2)
  levels=f3->level
  iyy=ind(levels.eq.hgt_sel(nh))
  var=f3->hgt(iyyyy,iyy,::-1,:)
  printVarSummary(var)
  delete(iyy)
  var=lonFlip(var) 
  var:=var(:,{latS:latN},{lonL:lonR})
  month:=month(iyyyy)
  year:=year(iyyyy)
  delete(iyyyy)
  Var= new((/dimsizes(Year),dimsizes(var(0,:,0)),dimsizes(var(0,0,:))/),"float")
  if(Season1(ns).eq."WARM") then
    do ny=0,dimsizes(Year)-1
      iyyyy=ind(year.eq.Year(ny).and.month.ge.5.and.month.le.7)
      Var(ny,:,:)=dim_avg_n_Wrap(var(iyyyy,:,:),0)
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
      Var(ny,:,:)=dim_avg_n_Wrap(var(iyyyy,:,:),0)
      delete(iyyyy)
    end do
  end if
  delete(var)
  delete(time)
  delete(year)
  delete(month)

  
  ;=================read the LMR data==================== 

  f1  = addfile(diri1+name1,"r")
  year=ispan(1900,2000,1)
  iyyyy=ind(year.ge.year1.and.year.le.year2)
  if(nh.eq.1) then
  var=f1->hgt700(iyyyy,0,:,:)
  end if
  if(nh.eq.0) then
  var=f1->hgt500(iyyyy,0,:,:)
  end if
  var&lon=var&lon-360
  Var_LMR = var(:,{latS:latN},{lonL:lonR})
  delete(var)
  delete(iyyyy)
  delete(year)

  ;=================read the LMR climatology data==================== 
  f2  = addfile(diri2+name2,"r")
  zg_clm=f2->var_clm
  zg_clm=lonFlip(zg_clm) 
  printVarSummary(zg_clm)
  zg_clm:=zg_clm({latS:latN},{lonL:lonR})
  zg_clm_copy=conform_dims(dimsizes(Var_LMR),zg_clm,(/1,2/))
  Var_LMR =Var_LMR +zg_clm_copy
  delete(zg_clm)
  delete(zg_clm_copy)

;========all the data are regrid to teh gpcc grid=======
  lat_GPCC= Var&lat
  lon_GPCC= Var&lon
  lat_LMR= Var_LMR&lat
  lon_LMR= Var_LMR&lon
  Var_regrid := linint2_Wrap(lon_GPCC,lat_GPCC,Var, False,lon_LMR,lat_LMR, 0)

;==========adjust the climatology====================
Obs_clm=conform_dims(dimsizes(Var_regrid),dim_avg_n_Wrap(Var_regrid,0),(/1,2/))
LMR_clm=conform_dims((/dimsizes( Var_LMR )/),dim_avg_n_Wrap(Var_LMR,0),(/1,2/))
Var_LMR = Var_LMR -LMR_clm
Var_regrid=Var_regrid-Obs_clm

;========calculate the metrics =======================

    ;=======coefficient of efficiency=========

    Var_mean=conform_dims(dimsizes(Var_regrid),dim_avg_n_Wrap(Var_regrid,0),(/1,2/))
    aa=(Var_regrid-Var_mean)*(Var_regrid-Var_mean)
    tt_1=dim_sum_n_Wrap(aa,0)
    aa=(Var_LMR -Var_regrid)*(Var_LMR -Var_regrid)
    tt_2=dim_sum_n_Wrap(aa,0)
    delete(aa)
    CE=1-tt_2/tt_1
    copy_VarCoords(Var_regrid(0,:,:),CE)    

    ;=========correlation=============================== 
    n=dimsizes(Var_regrid(:,0,0))
    df=n-2
    rr   = escorc_n(Var_regrid,Var_LMR,0,0) 
    t    = rr*sqrt((n-2)/(1-rr^2))      
    p_rr = student_t(t, df)
    delete(t)
    copy_VarCoords(Var_regrid(0,:,:),rr) 
    copy_VarCoords(Var_regrid(0,:,:),p_rr) 

    ;===========pattern correlation===========  
    rad    = 4.0*atan(1.0)/180.0
    re     = 6371220.0
    ilat=ind(lat_LMR.ge.latS.and.lat_LMR.le.latN)
    clat   = cos(lat_LMR(ilat)*rad)
    rc = pattern_cor(Var_regrid,Var_LMR,clat, 0) 
    printVarSummary(rc)
    

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
  res@gsnLeftStringFontHeightF = 0.025
  res@gsnLeftString ="Correlation"
  res@gsnRightStringFontHeightF = 0.025
  plot_rr(ns,nh)=gsn_csm_contour_map(wks,rr,res)
  res@gsnLeftString ="CE"
  res@gsnRightStringFontHeightF = 0.025
  plot_CE(ns,nh)=gsn_csm_contour_map(wks,CE,res)

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
  ;plot0 = gsn_csm_contour(wks,p_rr,res2)

  delete(res)

    res                    = True                    ; plot mods desired
    res@gsnFrame=False
    res@gsnDraw=False
    res@trXMaxF   =  2000
    res@trXMinF   =  1900
    res@tmXBMode          = "Manual"
    res@tmXBMinorPerMajor = 2
    res@tmXBTickEndF      = 2000
    res@tmXBTickSpacingF  = 20
    res@tmXBTickStartF    = 1900
    res@trYMaxF   =  1.0
    res@trYMinF  =0.
    res@gsnLeftString=Season2(ns)+" (Pattern correlation)"
    res@gsnLeftStringFontHeightF =0.022
    res@tmXTOn= False
    res@tmYROn= False
    res@xyLineThicknessF   = 2.                      ; line thickness
    res@vpWidthF=0.7
    res@vpHeightF=0.3
    res@tiXAxisString  ="Year"
    res@xyLineColors       = "Red"
    plot_cor(ns,nh)  = gsn_csm_xy (wks,Year,rc,res)
    delete(res)
end do
end do

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
  resP@gsnPanelMainString = "Warm Season"
  gsn_panel(wks,(/plot_rr(0,0),plot_CE(0,0),plot_rr(0,1),plot_CE(0,1)/),(/2,2/),resP) 
  resP@gsnPanelMainString = "Cool Season"
  gsn_panel(wks,(/plot_rr(1,0),plot_CE(1,0),plot_rr(1,1),plot_CE(1,1)/),(/2,2/),resP) 
  delete(resP)
exit












