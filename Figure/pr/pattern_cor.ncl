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

latS   =   10
latN   =   55
lonL   =   -135
lonR   =   -60

wks = gsn_open_wks("png" ,"pattern_cor")      ; send graphics to PNG file
gsn_define_colormap(wks,"amwg_blueyellowred")  
plot_1=new(2,"graphic")
plot_2=new(2,"graphic")
do ns=0,dimsizes(Season1)-1
	;======PDA_NASPA data==========
    diri1="//Users/jjyan/Desktop/NA_reconstrution/data/"+Season(ns)+"_season/pre/"
    name1="prate_MCruns_ensemble_mean_800_2000.nc"
	;=================GPCC data========
    diri3="/Users/jjyan/Desktop/NA_reconstrution/code/PDA/data/data/analyses/GPCC/"
	name3="GPCC_precip.mon.flux.1x1.v6.nc"

	;==========read the data==========
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
	  lat_GPCC= Pre&lat
      lon_GPCC= Pre&lon
      Pre=Pre-conform_dims((/dimsizes(Pre)/),dim_avg_n_Wrap(Pre,0),(/1,2/))
	  delete(precip)
	  delete(time)
	  delete(year)
	  delete(month)

	
	f1  = addfile(diri1+name1,"r")
    year=ispan(800,2000,1)
    precip=f1->prate
    precip:=dim_avg_n_Wrap(precip,1)
    iyyyy=ind(year.ge.year1.and.year.le.year2)
    precip&lon=precip&lon-360
    ;printVarSummary(precip)
    pr_LMR = precip(iyyyy,{latS:latN},{lonL:lonR})
    lat_LMR=pr_LMR&lat
	lon_LMR=pr_LMR&lon
	pr_LMR=pr_LMR-conform_dims((/dimsizes(pr_LMR)/),dim_avg_n_Wrap(pr_LMR,0),(/1,2/))
    delete(precip)
    delete(iyyyy)
    delete(year)

	;==========regridding=======================
	Pre_regrid = linint2_Wrap (lon_GPCC,lat_GPCC,Pre,True, lon_LMR,lat_LMR, 0)
	Pre_regrid!0="time"
	 a    = addfile("$NCARG_ROOT/lib/ncarg/data/cdf/landsea.nc","r")
	lsdata = a->LSMASK
	lsm  = landsea_mask(lsdata,lat_LMR,lon_LMR)
	pr_LMR= mask( pr_LMR,lsm.eq.0,False)
	pr_LMR = mask( pr_LMR,lsm.eq.2,False)
	pr_LMR= mask( pr_LMR,lsm.eq.4,False); 
	Pre_regrid= mask( Pre_regrid,lsm.eq.0,False)
	Pre_regrid = mask( Pre_regrid,lsm.eq.2,False)
	Pre_regrid= mask( Pre_regrid,lsm.eq.4,False);   

	;=========correlation========================
	rad    = 4.0*atan(1.0)/180.0
	re     = 6371220.0
	rr     = re*rad
	ilat   =ind(lat_LMR.ge.latS.and.lat_LMR.le.latN)
	clat   = cos(lat_LMR(ilat)*rad)
	rc = pattern_cor(pr_LMR(time|:,{lat|latS:latN},{lon|lonL:lonR}),Pre_regrid(time|:,{lat|latS:latN},{lon|lonL:lonR}),clat, 0)
	delete(clat)
	delete(ilat)
	;========Ploting===============
	res                    = True                    ; plot mods desired
	res@gsnDraw   = False
	res@gsnFrame   = False
	res@trXMaxF   = 2000
	res@trXMinF   =  1900
	res@trYMaxF   =  0.8
	res@trYMinF   =  -0.2
	if(ns.eq.0) then
	  res@gsnLeftString = " (a) Warm season"
	else
	  res@gsnLeftString = " (c) Cool season"
	end if
	res@gsnLeftStringFontHeightF =0.022
	res@tiYAxisString          = "Pattern correlation"
	res@tiXAxisString          = "Year"
	res@tmXTOn= False
	res@tmYROn= False
	res@xyLineThicknessF   = 2.                      ; line thickness
	res@vpWidthF=0.7
	res@vpHeightF=0.35
	res@xyLineColors       = "Black"              
	plot_1(ns)  = gsn_csm_xy (wks,ispan(year1,year2,1),rc,res)   ;  create plot
	delete(res)


	x_sta=fspan(-0.3,0.7,21)
	y_sta=new(dimsizes(x_sta)-1,"integer")
	do nx=0,dimsizes(x_sta)-2
	    y_sta(nx)= dim_num(rc.ge.x_sta(nx).and.rc.lt.x_sta(nx+1))
	end do 
	res  = True
	res@trXMaxF   = 21
	res@trXMinF   =  1
	res@trYMaxF   =  24
	res@trYMinF   =  0
	res@gsnDraw   = False
	res@gsnFrame   = False
	res@tmXTOn= False
	res@tmYROn= False
	res@xyLineThicknessF       = 2
	res@vpHeightF= 0.35                   ; change aspect ratio of plot
	res@vpWidthF = 0.7   
	res@tiYAxisString          = "Counts"
	res@tiXAxisString          = "Pattern correlation"
	res@gsnXYBarChart            = True              ; Create bar plot
	res@tmXBMode        = "Explicit"  
	res@tmXBLabelDeltaF = 0.01
	res@tmXBValues      = (/1,3,5,7,9,11,13,15,17,19,21/)
	res@tmXBLabels      = (/"-0.3","-0.2","-0.1","0.0","0.1","0.2","0.3","0.4","0.5","0.6","0.7"/)
	res@gsnXYBarChartColors = "blue"
	if(ns.eq.0) then
		res@gsnLeftString = " (b) Warm season"
	else
		res@gsnLeftString = " (d) Cool season"
	end if
	plot_2(ns) = gsn_csm_xy (wks, ispan(1,dimsizes(y_sta),1), y_sta, res)
	delete(res)
end do
resP                    = True                 ; modify the panel plot
resP@gsnPanelYWhiteSpacePercent = 5    ; set main title
resP@gsnPanelLeft = 0.06
gsn_panel(wks,(/plot_1(0),plot_2(0),plot_1(1),plot_2(1)/),(/2,2/),resP)   
exit






