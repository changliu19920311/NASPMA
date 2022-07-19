;======plot the cor and ce between NASPMA/NASPA seasonal precipitation and GPCC data=================
 load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
 load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
 load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
 load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"

;================set the parameters=====================
Season1=(/"WARM","COOL"/)
Season =(/"warm","cool"/)
Season2=(/"Warm","Cool"/)
year1=1901
year2=1927
Year=ispan(year1,year2,1)
model = (/"GPCC"/)
names=(/"GPCC_precip.mon.flux.1x1.v6.nc"/)
variables = (/"precip"/)
latS   =   15.25
latN   =   54.75
lonL   =   -132.75
lonR   =   -58.25

wks = gsn_open_wks("pdf" ,"evaluation")      ; send graphics to PNG file
gsn_define_colormap(wks,"amwg_blueyellowred") 
plot_rr_NASPA=new(dimsizes(model),"graphic")
plot_rr_LMR_NASPA=new(dimsizes(model),"graphic")
plot_rr_NASPA_DIF=new(dimsizes(model),"graphic")
plot_CE_NASPA=new(dimsizes(model),"graphic")
plot_CE_LMR_NASPA=new(dimsizes(model),"graphic")
plot_CE_NASPA_DIF=new(dimsizes(model),"graphic")

do ns=0,dimsizes(Season1)-1
  ;======NASM_NASPA data==========
  diri1="//Users/jjyan/Desktop/NA_reconstrution/data/"+Season(ns)+"_season/pre/"
  name1="prate_MCruns_ensemble_mean_800_2000.nc"
  ;=======proxy reconstructed data==========
  diri2="/Users/jjyan/Desktop/NA_reconstrution/code/PDA/data/data/proxies/NASPA/"
  name2="NASPA_"+Season2(ns)+".nc"
 
  ;=================read the NASPA data==================== 
  f2  = addfile(diri2+name2,"r")
  year=ispan(0,2016,1)
  precip=f2->Pre
  iyyyy=ind(year.ge.year1.and.year.le.year2)
  Pre_NASPA = precip(iyyyy,:,:)
  delete(precip)
  delete(iyyyy)
  delete(year)
  ;=================read NASPMA data==================== 
  f1  = addfile(diri1+name1,"r")
  year=ispan(800,2000,1)
  precip=f1->prate
  precip:=dim_avg_n_Wrap(precip,1)
  iyyyy=ind(year.ge.year1.and.year.le.year2)
  precip&lon=precip&lon-360
  ;printVarSummary(precip)
  Pre_LMR_NASPA = precip(iyyyy,{latS:latN},{lonL:lonR})
  delete(precip)
  delete(iyyyy)
  delete(year)

  do nm=0,dimsizes(names)-1
    diri3="/Users/jjyan/Desktop/NA_reconstrution/code/PDA/data/data/analyses/"+model(nm)+"/"
    name3=names(nm)
    ;==========read the GPCC data==========
    f3  = addfile(diri3+name3,"r")
    print(f3)
    time=f3->time
    utc_date := cd_calendar(time, 0)
    year   := tointeger(utc_date(:,0))    ; Convert to integer for
    month  := tointeger(utc_date(:,1))
    iyyyy=ind(year.ge.year1.and.year.le.year2)
    lat=f3->lat
    lon=f3->lon
    precip := f3->$variables(nm)$
    if(lat(0).le.lat(1)) then
      precip:=precip(iyyyy,::-1,:)
    else
      precip:=precip(iyyyy,:,:)
    end if 
    if(max(lon).ge.200) then
      precip=lonFlip(precip) 
    end if
    precip:=precip(:,{latS:latN},{lonL:lonR})
    month:=month(iyyyy)
    year:=year(iyyyy)
    delete(iyyyy)
    delete(lat)
    delete(lon)
    Pre := new((/dimsizes(Year),dimsizes(precip(0,:,0)),dimsizes(precip(0,0,:))/),"float")
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
    copy_VarCoords(precip(0,:,:),Pre(0,:,:))  
    delete(precip)
    delete(time)
    delete(year)
    delete(month)

  ;========all the data are regrid to teh gpcc grid=======
    lat_GPCC:= Pre&lat
    lon_GPCC:= Pre&lon
    lat_NASPA= Pre_NASPA&lat
    lon_NASPA= Pre_NASPA&lon
    lat_LMR= Pre_LMR_NASPA&lat
    lon_LMR= Pre_LMR_NASPA&lon

    Pre_NASPA_regrid := linint2_Wrap(lon_NASPA,lat_NASPA,Pre_NASPA, False,lon_GPCC,lat_GPCC, 0)
    Pre_LMR_NASPA_regrid := linint2_Wrap(lon_LMR,lat_LMR,Pre_LMR_NASPA, False, lon_GPCC,lat_GPCC, 0)

    if(Season1(ns).eq."COOL") then
      Pre_LMR_NASPA_regrid=Pre_LMR_NASPA_regrid*86400*150
      Pre=Pre*86400*150
    end if
    if(Season1(ns).eq."WARM") then
      Pre_LMR_NASPA_regrid=Pre_LMR_NASPA_regrid*86400*90
      Pre=Pre*86400*90
    end if

    ;==========adjust the climatology====================
    GPCC_clm:=conform_dims(dimsizes(Pre),dim_avg_n_Wrap(Pre,0),(/1,2/))
    LMR_NASPA_clm:=conform_dims((/dimsizes(Pre_LMR_NASPA_regrid)/),dim_avg_n_Wrap(Pre_LMR_NASPA_regrid,0),(/1,2/))
    NASPA_clm:=conform_dims((/dimsizes(Pre_NASPA_regrid)/),dim_avg_n_Wrap(Pre_NASPA_regrid,0),(/1,2/))
    Pre_NASPA_regrid:=Pre_NASPA_regrid-NASPA_clm
    Pre_LMR_NASPA_regrid:=Pre_LMR_NASPA_regrid-LMR_NASPA_clm
    Pre=Pre-GPCC_clm
    ;========calculate the metrics =======================

    ;=======coefficient of efficiency=========

    Pre_GPCC_mean:=conform_dims(dimsizes(Pre),dim_avg_n_Wrap(Pre,0),(/1,2/))
    aa=(Pre-Pre_GPCC_mean)*(Pre-Pre_GPCC_mean)
    tt_1:=dim_sum_n_Wrap(aa,0)
    aa=(Pre_NASPA_regrid-Pre)*(Pre_NASPA_regrid-Pre)
    tt_2:=dim_sum_n_Wrap(aa,0)
    delete(aa)
    tt_1=where(tt_1.eq.0,tt_1@_FillValue,tt_1)
    CE_NASPA:=1-tt_2/tt_1
    copy_VarCoords(Pre(0,:,:),CE_NASPA)    

    aa=(Pre_LMR_NASPA_regrid- Pre)^2
    tt_2=dim_sum_n_Wrap(aa,0)
    CE_LMR_NASPA:=1-tt_2/tt_1
    copy_VarCoords(Pre(0,:,:),CE_LMR_NASPA)
    delete(aa)   

    CE_NASPA_dif:=CE_LMR_NASPA-CE_NASPA
    copy_VarCoords(Pre(0,:,:),CE_NASPA_dif) 
    pos_num:=num(CE_NASPA_dif.gt.0)
    neg_num:=num(CE_NASPA_dif.lt.0)
    CE_NASPA_dif_percent = tofloat(pos_num)/tofloat(pos_num+neg_num)*100

    ;=========correlation=============================== 
    n=dimsizes(Pre(:,0,0))
    df=n-2
    rr_LMR_NASPA  := escorc_n(Pre,Pre_LMR_NASPA_regrid,0,0) 
    t    = rr_LMR_NASPA*sqrt((n-2)/(1-rr_LMR_NASPA^2))      
    p_rr_LMR_NASPA := student_t(t, df)
    delete(t)
    copy_VarCoords(Pre(0,:,:),rr_LMR_NASPA) 
    copy_VarCoords(Pre(0,:,:),p_rr_LMR_NASPA) 

    P_rr = ndtooned(p_rr_LMR_NASPA)
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
    p_cr_LMR_NASPA = p(max(i_cr))
    delete(p)
    delete(cv)
    delete(index_i)
    delete(i_cr)
    delete(igood)
    delete(P_rr)

    rr_NASPA   := escorc_n(Pre,Pre_NASPA_regrid,0,0) 
    t    = rr_NASPA*sqrt((n-2)/(1-rr_NASPA^2))      
    p_rr_NASPA := student_t(t, df)
    delete(t)
    copy_VarCoords(Pre(0,:,:),rr_NASPA) 
    copy_VarCoords(Pre(0,:,:),p_rr_NASPA) 

    P_rr = ndtooned(p_rr_NASPA)
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
    p_cr_NASPA = p(max(i_cr))
    delete(p)
    delete(cv)
    delete(index_i)
    delete(i_cr)
    delete(igood)
    delete(P_rr)

    rr_NASPA_dif:=rr_LMR_NASPA-rr_NASPA
    copy_VarCoords(Pre(0,:,:),rr_NASPA_dif) 
    pos_num:=num(rr_NASPA_dif.gt.0)
    neg_num:=num(rr_NASPA_dif.lt.0)
    rr_NASPA_dif_percent = tofloat(pos_num)/tofloat(pos_num+neg_num)*100

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
    res@cnLineLabelsOn   = False
    res@cnMinLevelValF       = -0.7                ; set min contour level
    res@cnMaxLevelValF       =  0.7               ; set max contour level
    res@cnLevelSpacingF      =  0.1  
    res@gsnAddCyclic         = False        ; plotted data are not cyclic
    res@mpMinLatF            = latS         ; zoom in on map
    res@mpMaxLatF            = latN
    res@mpMinLonF            = lonL
    res@mpMaxLonF            = lonR      
    res@lbLabelBarOn        = False
    res@gsnLeftString ="(b) r (NASPA)"
    plot_rr_NASPA(nm)=gsn_csm_contour_map(wks,rr_NASPA,res)
    res@gsnLeftString ="(a) r (NASPMA)"
    plot_rr_LMR_NASPA(nm)=gsn_csm_contour_map(wks,rr_LMR_NASPA,res)
    res@gsnLeftString ="(c) r,Dif (NASPMA-NASPA)"
    plot_rr_NASPA_DIF(nm)=gsn_csm_contour_map(wks,rr_NASPA_dif,res)
    res@gsnLeftString ="(e) CE (NASPA)"
    plot_CE_NASPA(nm)=gsn_csm_contour_map(wks,CE_NASPA,res)
    res@gsnLeftString ="(d) CE (NASPMA)"
    plot_CE_LMR_NASPA(nm)=gsn_csm_contour_map(wks,CE_LMR_NASPA,res)
    res@gsnLeftString ="(f) CE (NASPMA-NASPA)"
    plot_CE_NASPA_DIF(nm)=gsn_csm_contour_map(wks,CE_NASPA_dif,res)
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

                     ;; set to significance level
    res2@cnFillPatterns = (/11,-1/)
    res2@cnFillColors = (/"black","white"/)
    res2@gsnLeftString = ""
    res2@cnLevels = p_cr_LMR_NASPA
    plot0 = gsn_csm_contour(wks,p_rr_LMR_NASPA,res2)
    res2@cnLevels = p_cr_NASPA
    plot1 = gsn_csm_contour(wks,p_rr_NASPA,res2)
    overlay(plot_rr_LMR_NASPA(nm),plot0)
    overlay(plot_rr_NASPA(nm),plot1)
    delete(plot0)
    delete(plot1) 
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
;resP@gsnPanelMainString = Season2(ns)+" Season"
gsn_panel(wks,(/plot_rr_LMR_NASPA(0),plot_rr_NASPA(0),plot_rr_NASPA_DIF(0),plot_CE_LMR_NASPA(0),plot_CE_NASPA(0),plot_CE_NASPA_DIF(0)/),(/2,3/),resP) 
  end do
exit












