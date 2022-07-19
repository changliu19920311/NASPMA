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
PSM = (/"linear","bilinear"/)
latS   =   (/10,30,24,40,15/)
latN   =   (/55,55,40,55,25/)
lonL   =   (/-135,-125,-100,-100,-110/)
lonR   =   (/-60,-100,-70,-70,-85/)
left_1=(/"(a)","(c)","(e)","(g)"/)
left_2=(/"(b)","(d)","(f)","(h)"/)
region = (/"NA","Western US","Southeastern US","Northeast NA"/)
data=(/"GPCC","CRU","WM"/)
name = (/"GPCC_precip.mon.flux.1x1.v6.nc","cru_ts4.01.1901.2016.pre.dat.nc","precip.mon.total.v501.nc"/)
Colors = (/"red","blue","purple"/)
wks = gsn_open_wks("pdf" ,"timeseries_std")      ; send graphics to PNG file
gsn_define_colormap(wks,"amwg_blueyellowred") 
plot=new((/dimsizes(Season1),dimsizes(region)/),"graphic")
dummy=new((/dimsizes(Season1),dimsizes(region)/),"graphic")
pr_avg_data=new((/dimsizes(data),dimsizes(Year)/),"float")
do ns=0,dimsizes(Season1)-1
  do nr=0,dimsizes(region)-1
    ;======PDA_NASPA data==========
    diri1="/Users/chang/Desktop/NA_reconstruction/data/pda/"+Season(ns)+"_season/pre/"
    name1="prate_MCruns_ensemble_mean_1900_2000.nc"
    f1  = addfile(diri1+name1,"r")
    year=ispan(1900,2000,1)
    iyyyy=ind(year.ge.year1.and.year.le.year2)
    prate=f1->prate(iyyyy,:,:,:)
    prate&lon=prate&lon-360
    pr_LMR = dim_avg_n_Wrap(prate(:,:,{latS(nr):latN(nr)},{lonL(nr):lonR(nr)}),1)
    pr_LMR = pr_LMR-conform_dims((/dimsizes(pr_LMR)/),dim_avg_n_Wrap(pr_LMR,0),(/1,2/))
    delete(prate)
    delete(iyyyy)
    delete(year)

    diri2="/Users/chang/Desktop/NA_reconstruction/data/pda/"+Season(ns)+"_season/pre/1900_2000/"
    name2="prate_MCruns_ensemble_spread.nc"
    f2  = addfile(diri2+name2,"r")
    year=ispan(1900,2000,1)
    iyyyy=ind(year.ge.year1.and.year.le.year2)
    prate=f2->prate(iyyyy,:,:,:)
    prate&lon=prate&lon-360
    pr_LMR_std = dim_avg_n_Wrap(prate(:,:,{latS(nr):latN(nr)},{lonL(nr):lonR(nr)}),1)
    delete(prate)
    delete(iyyyy)
    delete(year) 
    lat_LMR= pr_LMR&lat
    lon_LMR= pr_LMR&lon

    a    = addfile("$NCARG_ROOT/lib/ncarg/data/cdf/landsea.nc","r")
    lsdata = a->LSMASK
    lsm  = landsea_mask(lsdata,lat_LMR,lon_LMR)
    pr_LMR= mask( pr_LMR,lsm.eq.0,False)
    pr_LMR = mask( pr_LMR,lsm.eq.2,False)
    pr_LMR= mask( pr_LMR,lsm.eq.4,False); 
    pr_LMR_std= mask( pr_LMR_std,lsm.eq.0,False)
    pr_LMR_std = mask( pr_LMR_std,lsm.eq.2,False)
    pr_LMR_std= mask( pr_LMR_std,lsm.eq.4,False); 
    delete(lsm)

    rad    = 4.0*atan(1.0)/180.0
    re     = 6371220.0
    rr     = re*rad
    clat   := cos(lat_LMR*rad)
    pr_avg_LMR = wgt_areaave(pr_LMR, clat, 1.0, 0)
    pr_std_LMR = wgt_areaave(pr_LMR_std, clat, 1.0, 0)  
    pr_avg_LMR =  pr_avg_LMR*86400
    pr_std_LMR =  pr_std_LMR*86400

    xp    = new( (/2*ntime/), float )
    yp    = new( (/2*ntime/), float )
    do k=0,ntime-1
      yp(k)           = pr_avg_LMR(k) + pr_std_LMR(k)
      xp(k)           = Year(k)
      xp(2*ntime-1-k) = Year(k)
      yp(2*ntime-1-k) = pr_avg_LMR(k) - pr_std_LMR(k)
    end do

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

      ;========all the data are regrid to teh LMR grid=======
      lat_GPCC= Pre&lat
      lon_GPCC= Pre&lon
      printVarSummary(pr_LMR)
      Pre := linint2_Wrap(lon_GPCC,lat_GPCC,Pre, False, lon_LMR,lat_LMR, 0)
      if(nd.eq.1) then
        Pre =Pre/(86400*30.0)
      end if 

      ;==========adjust the climatology====================
      GPCC_clm=conform_dims(dimsizes(Pre),dim_avg_n_Wrap(Pre,0),(/1,2/))
      Pre=Pre-GPCC_clm
      delete(GPCC_clm)

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
      clat   := cos(lat_LMR*rad)
      pr_avg_GPCC = wgt_areaave(Pre, clat, 1.0, 0)
      pr_avg_GPCC = pr_avg_GPCC*86400
      pr_avg_data(nd,:) = pr_avg_GPCC
      delete(Pre)
      delete(lat_GPCC)
      delete(lon_GPCC)
    end do

    delete(pr_LMR)     
    delete(pr_LMR_std)
    delete(lat_LMR)
    delete(lon_LMR)

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
        res@xyLineColors       := (/"Red","black","blue","purple"/)
    if(nr.eq.0) then
    res@pmLegendDisplayMode    = "Always"            ; turn on legend
    res@pmLegendSide           = "Top"               ; Change location of 
    res@pmLegendParallelPosF   = .1                 ; move units right
    res@pmLegendOrthogonalPosF = -0.38                ; move units down
    res@pmLegendWidthF         = 0.06                ; Change width and
    res@pmLegendHeightF        = 0.08               ; height of legend.
    res@lgPerimOn              = False               ; turn off box around
    res@lgLabelFontHeightF     = .015                ; label font height
    res@xyExplicitLegendLabels:= (/"PDA","GPCC","CRU","WM"/)   
    end if
    plot(ns,nr)  = gsn_csm_xy (wks,Year,(/pr_avg_LMR,pr_avg_data(0,:),pr_avg_data(1,:),pr_avg_data(2,:)/),res) 
    delete(res)

    gsres                   = True                        ; poly res
    gsres@tfPolyDrawOrder   = "Predraw"                   ; draw this first
    gsres@gsFillColor       = "pink"                ; color chosen
    gsres@gsFillOpacityF = 0.7
    dummy(ns,nr) = gsn_add_polygon (wks,plot(ns,nr),xp,yp,gsres)
  end do
end do

resP                     = True         ; modify the panel plot
resP@gsnPanelYWhiteSpacePercent = 5
resP@gsnPanelXWhiteSpacePercent = 5
gsn_panel(wks,(/plot(0,0),plot(1,0),plot(0,1),plot(1,1),plot(0,2),plot(1,2),plot(0,3),plot(1,3)/),(/4,2/),resP) 

exit










