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
left_1=(/"(a)","(c)","(e)","(g)"/)
left_2=(/"(b)","(d)","(f)","(h)"/)
region = (/"NA","Western US","Southeastern US","Northeast NA"/)
data=(/"GPCC"/)
name = (/"GPCC_precip.mon.flux.1x1.v6.nc"/)
wks = gsn_open_wks("pdf" ,"timeseries_std_full")      ; send graphics to PNG file
gsn_define_colormap(wks,"amwg_blueyellowred") 
plot=new((/dimsizes(Season1),dimsizes(region)/),"graphic")
dummy=new((/dimsizes(Season1),dimsizes(region)/),"graphic")
pr_avg_data=new((/dimsizes(data),dimsizes(Year)/),"float")
do ns=0,dimsizes(Season1)-1
  diri1="//Users/jjyan/Desktop/NA_reconstrution/data/"+Season(ns)+"_season/pre/"
  file1 = "prate_MCruns_ensemble_mean_800_2000.nc"
  file2 = "timeseries_std.nc"
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


    f2  = addfile(diri1+file2,"r")
	year=ispan(1900,2000,1)
	iyyyy=ind(year.ge.year1.and.year.le.year2)
	prate=f2->pr_std_LMR(nr,iyyyy)
	delete(year)
	delete(iyyyy)
	printVarSummary(prate)
  pr_std_LMR =  prate/30.0
  delete(prate)

    xp    = new( (/2*ntime/), float )
    yp    = new( (/2*ntime/), float )
    do k=0,ntime-1
      yp(k)           = pr_avg_LMR(k) + pr_std_LMR(k)
      xp(k)           = Year(k)
      xp(2*ntime-1-k) = Year(k)
      yp(2*ntime-1-k) = pr_avg_LMR(k) - pr_std_LMR(k)
    end do


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
    res@trYMaxF   =  1.6
    res@trYMinF  =-1.2
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
    res@xyExplicitLegendLabels:= (/"NASPMA","GPCC"/)   
    end if
    plot(ns,nr)  = gsn_csm_xy (wks,Year,(/pr_avg_LMR,pr_avg_GPCC/),res) 
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










