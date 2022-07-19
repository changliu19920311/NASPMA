; evaluating the reconstucted monthly data=================
 load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
 load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
 load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
 load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"

;================set the parameters=====================
Season1=(/"WARM","COOL"/)
Season2=(/"Warm","Cool"/)
hgt_sel=(/200/)
year1=1950
year2=2000
Year=ispan(year1,year2,1)
latS   =   15.25
latN   =   54.75
lonL   =   -132.75
lonR   =  -58.25

wks = gsn_open_wks("pdf" ,"jet_stream")      ; send graphics to PNG file
gsn_define_colormap(wks,"amwg_blueyellowred") 
plot_0=new((/2/),"graphic")
plot_1=new((/2/),"graphic")


do ns=0,dimsizes(Season1)-1
  ;======LMR data==========
  diri1="/Users/chang/Desktop/LMR_NA/output/test_recon_season_GPCC_"+Season1(ns)+"/"
  name1="ua_MCruns_ensemble_mean.nc"

  ;======CCSM4 ua data==========
  diri2="/Users/chang/Desktop/LMR_NA/output/test_recon_season_GPCC_"+Season1(ns)+"/"
  name2="clm_ua_200hpa_Amon_CCSM4_past1000_085001-185012.nc"

  ;=================Berkeley Earth Temperature
  diri3="/Users/chang/Desktop/LMR_test/data/data/analyses/NCAR/"
  name3="uwnd.mon.mean.nc"

  ;==========read the 20CR_V2data,calculate the cool season average ==========

  f3  = addfile(diri3+name3,"r")
  time=f3->time
  print(f3)
  utc_date = cd_calendar(time, 0)
  year   = tointeger(utc_date(:,0))    ; Convert to integer for
  month  = tointeger(utc_date(:,1))
  iyyyy=ind(year.ge.year1.and.year.le.year2)
  levels=f3->level
  iyy=ind(levels.eq.hgt_sel)
  var=f3->uwnd(iyyyy,iyy,::-1,:)
  printVarSummary(var)
  delete(iyy)
  var=lonFlip(var) 
  var:=var(:,{latS:latN},{lonL:lonR})
  month:=month(iyyyy)
  year:=year(iyyyy)
  delete(iyyyy)
  Var= new((/dimsizes(Year),dimsizes(var(0,:,0)),dimsizes(var(0,0,:))/),"float")
  print(year)
  if(ns.eq.0) then
    do ny=0,dimsizes(Year)-1
      iyyyy=ind(year.eq.Year(ny).and.month.ge.5.and.month.le.7)
      Var(ny,:,:)=dim_avg_n_Wrap(var(iyyyy,:,:),0)
      delete(iyyyy)
    end do
  end if
  if(ns.eq.1) then
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
  var=f1->ua(iyyyy,0,:,:)
  var&lon=var&lon-360
  Var_LMR = var(:,{latS:latN},{lonL:lonR})
  delete(var)
  delete(iyyyy)
  delete(year)

  ;=================read the LMR climatology data==================== 
  f2  = addfile(diri2+name2,"r")
  print(f2)
  ua_clm=f2->var_clm
  ua_clm=lonFlip(ua_clm) 
  printVarSummary(ua_clm)
  ua_clm:=ua_clm({latS:latN},{lonL:lonR})
  ua_clm_copy=conform_dims(dimsizes(Var_LMR),ua_clm,(/1,2/))
  Var_LMR =Var_LMR +ua_clm_copy
  delete(ua_clm)
  delete(ua_clm_copy)

;========all the data are regrid to teh gpcc grid=======
  lat_GPCC= Var&lat
  lon_GPCC= Var&lon
  lat_LMR= Var_LMR&lat
  lon_LMR= Var_LMR&lon
  Var_regrid := linint2_Wrap(lon_GPCC,lat_GPCC,Var, False,lon_LMR,lat_LMR, 0)



;========calculate the location of jet stream=======================

Var_LMR_ave=dim_avg_n_Wrap(Var_LMR,0)
Var_regrid_ave=dim_avg_n_Wrap(Var_regrid,0)

Var_LMR_ave = runave_n(Var_LMR_ave, 9,0,0)
Var_LMR_ave = runave_n(Var_LMR_ave, 9,0,1)
Var_regrid_ave = runave_n(Var_regrid_ave, 9,0,0)
Var_regrid_ave = runave_n(Var_regrid_ave, 9,0,1)
do ny=0,dimsizes(lon_LMR)-1
    Var_LMR_ave(:,ny)=where( Var_LMR_ave(:,ny).eq.max( Var_LMR_ave(:,ny)),1001,10)
    Var_regrid_ave(:,ny)=where(Var_regrid_ave(:,ny).eq.max(Var_regrid_ave(:,ny)),1001,10)
end do



  res                       = True     ; plot mods desired
  res@gsnDraw  = False                          ; don't draw
  res@gsnFrame = False 
  res@cnFillOn              = False  
  res@cnLinesOn             = True    ; turn of contour lines
  res@gsnAddCyclic          = False    ; data already has cyclic point
  res@cnLevelSelectionMode = "ExplicitLevels"
  res@cnLevels = (/1000/)   
  res@gsnAddCyclic         = False        ; plotted data are not cyclic
  res@mpMinLatF            = latS         ; zoom in on map
  res@mpMaxLatF            = latN
  res@mpMinLonF            = lonL
  res@mpMaxLonF            = lonR  
  res@lbLabelBarOn        = False  
  res@cnLineLabelsOn         =False
  res@cnLineColor="Black"
  res@cnSmoothingOn=True
  res@cnSmoothingTensionF             = -0.001 ; turn of contour lines
  res@cnSmoothingDistanceF       = 0.01
  res@gsnLeftString = Season2(ns)+" season: LMR"
  plot_0(ns) = gsn_csm_contour_map_ce(wks,Var_LMR_ave, res)
  res@gsnLeftString          = Season2(ns)+" season: NCEP"
  res@pmTickMarkDisplayMode = "Always"; 
  plot_1(ns) = gsn_csm_contour_map_ce(wks, Var_regrid_ave, res)

end do
  resP                    = True                 ; modify the panel plot
  gsn_panel(wks,(/plot_0(0),plot_1(0),plot_0(1),plot_1(1)/),(/2,2/),resP)               ; now

exit












