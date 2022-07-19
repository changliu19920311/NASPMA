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
left_1=(/"(a)","(c)","(e)","(g)"/)
left_2=(/"(b)","(d)","(f)","(h)"/)
region = (/"NA","Western US","Southeastern US","Northeast NA"/)
data=(/"GPCC"/)
name = (/"GPCC_precip.mon.flux.1x1.v6.nc"/)
Colors = (/"red","blue","purple"/)
wks = gsn_open_wks("pdf" ,"timeseries_GPCC")      ; send graphics to PNG file
gsn_define_colormap(wks,"amwg_blueyellowred") 
plot=new((/dimsizes(Season1),dimsizes(region)/),"graphic")
pr_avg_data=new((/dimsizes(data),dimsizes(Year)/),"float")
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

    nd=0
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
    delete(Pre_LMR_NASPA)     
    delete( Pre_LMR_NASPA_regrid)
    ;========Ploting===============
    res                    = True                    ; plot mods desired
    res@gsnFrame=False
    res@gsnDraw=False
    res@trXMaxF   =  2000
    res@trXMinF   =  1900
    res@tmXBMode          = "Manual"
    res@tmXBMinorPerMajor = 1
    res@tmXBTickEndF      = 2000
    res@tmXBTickSpacingF  = 10
    res@tmXBTickStartF    = 1900
    res@trYMaxF   =  1.5
    res@trYMinF  =-1
    if(ns.eq.0) then
      res@gsnLeftString = left_1(nr)+" "+Season2(ns)+" season"
    else
      res@gsnLeftString = left_2(nr)+" "+Season2(ns)+" season"
    end if
    res@gsnRightString = region(nr)
    ;res@gsnLeftStringFontHeightF =0.022
    res@tmXTOn= False
    res@tmYROn= False
    res@xyLineThicknessF   = 2.                      ; line thickness
    res@tiXAxisString  ="Year"
    res@tiYAxisString  ="Pre (mm/day)" ;
    res@vpWidthF=0.7
    res@vpHeightF=0.3
    res@xyDashPattern  = 0
        res@xyLineColors       := (/"Red","black"/)
    if(nr.eq.0) then
    res@pmLegendDisplayMode    = "Always"            ; turn on legend
    res@pmLegendSide           = "Top"               ; Change location of 
    res@pmLegendParallelPosF   = .1                 ; move units right
    res@pmLegendOrthogonalPosF = -0.38                ; move units down
    res@pmLegendWidthF         = 0.06                ; Change width and
    res@pmLegendHeightF        = 0.08               ; height of legend.
    res@lgPerimOn              = False               ; turn off box around
    res@lgLabelFontHeightF     = .015                ; label font height
    res@xyExplicitLegendLabels:= (/"PDA","GPCC"/)   

    end if
    plot(ns,nr)  = gsn_csm_xy (wks,Year,(/pr_avg_LMR,pr_avg_data(0,:)/),res) 
    delete(res)
  end do
end do

resP                     = True         ; modify the panel plot
resP@gsnPanelYWhiteSpacePercent = 5
resP@gsnPanelXWhiteSpacePercent = 5
gsn_panel(wks,(/plot(0,0),plot(1,0),plot(0,1),plot(1,1),plot(0,2),plot(1,2),plot(0,3),plot(1,3)/),(/4,2/),resP) 

exit












