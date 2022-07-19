; =======plot the uncertainty of reconstruction data=================
 load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
 load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
 load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
 load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"

;================set the parameters=====================
Season =(/"warm","cool"/)
Season2=(/"Warm","Cool"/)
year1=1
year2=2000
Year=ispan(year1,year2,1)
ntime = dimsizes(Year)
region = (/"NA","Western US","Southeastern US","Northeast NA"/)
latS   =   (/10,30,24,40,15/)
latN   =   (/55,55,40,55,25/)
lonL   =   (/-135,-130,-100,-100,-110/)
lonR   =   (/-60,-100,-70,-70,-85/)
left_1=(/"(a)","(b)","(c)","(d)"/)
period=(/"MCA","LIA","PD"/)
MCA_str=1000
MCA_end=1300
LIA_str=1400
LIA_end=1700
PD_str=1850
PD_end=2000

wks = gsn_open_wks("pdf" ,"uncertainty")      ; send graphics to PNG file
gsn_define_colormap(wks,"amwg_blueyellowred") 
plot=new((/dimsizes(Season2),4/),"graphic")
pr_std = new((/ntime,dimsizes(region)/),"float")

;=========read the CCSM4 data to calculate the climatology ====================================
diri1="/Users/jjyan/Desktop/NA_reconstrution/code/PDA/data/data/model/ccsm4_last_millenium/"
name1="pr_sfc_Amon_CCSM4_past1000_085001-185012.nc"
f1  = addfile(diri1+name1,"r")
pr=f1->pr
time=f1->time
utc_date = cd_calendar(time, 0)
year   = tointeger(utc_date(:,0))    ; Convert to integer for
month  = tointeger(utc_date(:,1))
pr=lonFlip(pr)
printVarSummary(pr)
pr_warm = dim_avg_n((pr(4::12,:,:)+pr(5::12,:,:)+pr(6::12,:,:))/3.0,0)    
pr_warm_clm = conform_dims((/ntime,dimsizes(pr(0,:,0)),dimsizes(pr(0,0,:))/),pr_warm,(/1,2/))    
pr_cool = dim_avg_n((pr(0::12,:,:)+pr(1::12,:,:)+pr(2::12,:,:)+pr(3::12,:,:)+pr(11::12,:,:))/5.0,0)    
pr_cool_clm = conform_dims((/ntime,dimsizes(pr(0,:,0)),dimsizes(pr(0,0,:))/),pr_cool,(/1,2/))
copy_VarMeta(pr(0,:,:),pr_warm_clm(0,:,:))
copy_VarMeta(pr(0,:,:),pr_cool_clm(0,:,:))
printVarSummary(pr_warm_clm)
delete(pr)
delete(time)
delete(year)
delete(month)
delete(utc_date)


do ns=0,dimsizes(Season2)-1
    ;=======read the total number of available proxy===
    data = asciiread("/Users/jjyan/Desktop/NA_reconstrution/code/PDA/data/data/proxies/NASPA/"+ Season2(ns)+" Season.txt",(/1201/),"float")
    ;==========read the reconstructed seasonal precipitation====
    diri2="//Users/jjyan/Desktop/NA_reconstrution/data/"+Season(ns)+"_season/pre/"
    name2="prate_MCruns_ensemble_spread_800_2000.nc"
    f2  = addfile(diri2+name2,"r")
    prate=f2->prate
    prate&lon=prate&lon-360
    lat_LMR= prate&lat
    lon_LMR= prate&lon
    a    = addfile("$NCARG_ROOT/lib/ncarg/data/cdf/landsea.nc","r")
    lsdata = a->LSMASK
    lsm  = landsea_mask(lsdata,lat_LMR,lon_LMR)
    prate= mask( prate,lsm.eq.0,False)
    prate = mask( prate,lsm.eq.2,False)
    prate= mask( prate,lsm.eq.4,False); 
    delete(lsm)

    ;=========different period========================
    iyyyy_MCA  =  ind(Year.ge.MCA_str.and.Year.le.MCA_end)
    iyyyy_LIA  =  ind(Year.ge.LIA_str.and.Year.le.LIA_end)
    iyyyy_PD   =  ind(Year.ge.PD_str.and.Year.le.PD_end)

    pr_LMR_std_MCA  = dim_avg_n_Wrap(dim_avg_n_Wrap(prate(iyyyy_MCA,:,{latS(0):latN(0)},{lonL(0):lonR(0)}),0),0)
    pr_LMR_std_LIA  = dim_avg_n_Wrap(dim_avg_n_Wrap(prate(iyyyy_LIA,:,{latS(0):latN(0)},{lonL(0):lonR(0)}),0),0)
    pr_LMR_std_PD   = dim_avg_n_Wrap(dim_avg_n_Wrap(prate(iyyyy_PD,:,{latS(0):latN(0)},{lonL(0):lonR(0)}),0),0)

    if(ns.eq.0) then
        pr_LMR_std_MCA  = pr_LMR_std_MCA/pr_warm_clm(0,{latS(0):latN(0)},{lonL(0):lonR(0)})
        pr_LMR_std_LIA  = pr_LMR_std_LIA/pr_warm_clm(0,{latS(0):latN(0)},{lonL(0):lonR(0)})
        pr_LMR_std_PD  = pr_LMR_std_PD/pr_warm_clm(0,{latS(0):latN(0)},{lonL(0):lonR(0)})
    else
        pr_LMR_std_MCA  = pr_LMR_std_MCA/pr_cool_clm(0,{latS(0):latN(0)},{lonL(0):lonR(0)})
        pr_LMR_std_LIA  = pr_LMR_std_LIA/pr_cool_clm(0,{latS(0):latN(0)},{lonL(0):lonR(0)})
        pr_LMR_std_PD  = pr_LMR_std_PD/pr_cool_clm(0,{latS(0):latN(0)},{lonL(0):lonR(0)})
    end if
    delete(lat_LMR)
    delete(lon_LMR)
    ;=========different regions===============
  	do nr=0,dimsizes(region)-1
	    pr_LMR_std = dim_avg_n_Wrap(prate(:,:,{latS(nr):latN(nr)},{lonL(nr):lonR(nr)}),1)
	    printVarSummary(pr_warm_clm(:,{latS(nr):latN(nr)},{lonL(nr):lonR(nr)}))
	    printVarSummary(pr_LMR_std)
	    if(ns.eq.0) then
	       pr_LMR_std = pr_LMR_std/pr_warm_clm(:,{latS(nr):latN(nr)},{lonL(nr):lonR(nr)})
	    else
	       pr_LMR_std = pr_LMR_std/pr_cool_clm(:,{latS(nr):latN(nr)},{lonL(nr):lonR(nr)})
	    end if
	    lat_LMR= pr_LMR_std&lat
	    lon_LMR= pr_LMR_std&lon
	    rad    = 4.0*atan(1.0)/180.0
	    re     = 6371220.0
	    rr     = re*rad
	    clat   := cos(lat_LMR*rad)
	    pr_std_LMR = wgt_areaave(pr_LMR_std, clat, 1.0, 0)  
	    pr_std(:,nr) = pr_std_LMR
	    delete(lat_LMR)
	    delete(lon_LMR)
	    delete(pr_LMR_std)
  	end do
    ;=======plot============
    res                      = True     ; plot mods desired
    res@gsnDraw              = False
    res@gsnFrame             = False
    res@cnFillOn             = True     ; turn on color fill
    res@cnLinesOn            = False    ; turn of contour lines
    res@cnLevelSelectionMode = "ManualLevels"  ; manual contour levels
    res@cnMinLevelValF       = 0
    res@cnMaxLevelValF       =  0.7
    res@cnLevelSpacingF      =  0.05
    res@cnFillPalette        = "amwg_blueyellowred"
    res@gsnAddCyclic         = False    ; data already has cyclic point
    res@gsnLeftStringFontHeightF =0.022
    res@gsnRightStringFontHeightF =0.022
    res@mpMinLatF            = latS(0)    ; range to zoom in on
    res@mpMaxLatF            = latN(0)
    res@mpMinLonF            = lonL(0)
    res@mpMaxLonF            = lonR(0)
    res@lbLabelBarOn        = False  
    res@gsnLeftStringFontHeightF =0.022
    res@gsnRightString = Season2(ns)+" season"
    if(ns.eq.0) then
    res@gsnLeftString  ="(a) MCA"
    else
    res@gsnLeftString  ="(b) MCA"
    end if
    plot(ns,0) = gsn_csm_contour_map(wks,pr_LMR_std_MCA , res)
     if(ns.eq.0) then
    res@gsnLeftString  ="(c) LIA"
    else
    res@gsnLeftString  ="(d) LIA"
    end if

    plot(ns,1) = gsn_csm_contour_map(wks,pr_LMR_std_LIA , res)
     if(ns.eq.0) then
    res@gsnLeftString  ="(e) PD"
    else
    res@gsnLeftString  ="(f) PD"
    end if
    plot(ns,2) = gsn_csm_contour_map(wks,pr_LMR_std_PD , res)
    delete(pr_LMR_std_MCA)
    delete(pr_LMR_std_LIA)
    delete(pr_LMR_std_PD)
    delete(res)


    ;========Ploting===============
    res                    = True                    ; plot mods desired
    res@gsnFrame=False
    res@gsnDraw=False
    res@trXMaxF   =  2000
    res@trXMinF   =  800
    res@tmXBMode          = "Manual"
    res@tmXBMinorPerMajor = 1
    res@tmXBTickEndF      = 2000
    res@tmXBTickSpacingF  = 200
    res@tmXBTickStartF    = 800
    if(ns.eq.0) then
        res@trYMaxF   =  0.4
        res@trYMinF  =0.1
    else
        res@trYMaxF   =  0.4
        res@trYMinF  =0.1
    end if
   
    res@tmXTOn= False
    ;res@tmYROn= False
    res@xyLineThicknessF   = 2.                      ; line thickness
    res@tiXAxisString  ="Year"
    res@tiYAxisString  ="Normalized standard deviation" ;
    res@vpWidthF=0.65
    res@vpHeightF=0.4
    res@xyDashPattern  = 0
    res@xyLineColors   =  (/"black","red","blue","purple","black"/)
    res@xyDashPatterns = (/0,0,0,0,2/) 
    res@tiYAxisFontHeightF = 0.015
    res@pmLegendDisplayMode    = "Always"            ; turn on legend
    res@pmLegendSide           = "Top"               ; Change location of 
    res@pmLegendParallelPosF   = 0.55                 ; move units right
    res@pmLegendOrthogonalPosF = -0.37                ; move units down
    res@pmLegendWidthF         = 0.12                ; Change width and
    res@pmLegendHeightF        = 0.12               ; height of legend.
    res@lgPerimOn              = False               ; turn off box around
    res@lgLabelFontHeightF     = .015                ; label font height

    res@xyExplicitLegendLabels:= (/"NA","Western US","Southeastern US","Northeast NA","The number of proxies"/)
    
    resL = res
    res@xyLineColors   :=  (/"black"/)
    res@xyDashPatterns := (/2/) 
    res@trYMaxF   =  700
    res@trYMinF  =0
    res@tiYAxisString  ="The number of proxies" 
    res@gsnLeftString = "  "
    res@tiYAxisFontHeightF = 0.017
    res@pmLegendDisplayMode    = "NoCreate" 

    resR = res 
    resR@gsnRightString = Season2(ns)+" season"
    resR@gsnRightStringFontHeightF = 0.02
    resR@gsnLeftStringFontHeightF = 0.02
    if(ns.eq.0) then
    resR@gsnLeftString  ="(g) Regional average"
    else
    resR@gsnLeftString  ="(h) Regional average"
  end if  
    c=pr_std(:,3)
    c=-9999
    plot(ns,3)  = gsn_csm_xy2 (wks,Year,(/pr_std(:,0),pr_std(:,1),pr_std(:,2),pr_std(:,3),c/),(/data/),resL,resR) 
    delete(res)
end do

resP                     = True         ; modify the panel plot
resP@gsnPanelYWhiteSpacePercent = 5
resP@gsnPanelXWhiteSpacePercent = 5
resP@gsnPanelLabelBar    = True  
resP@gsnPanelBottom      = 0.28  
resP@gsnFrame=False  
gsn_panel(wks,(/plot(0,0),plot(1,0),plot(0,1),plot(1,1),plot(0,2),plot(1,2)/),(/3,2/),resP) 
resP@gsnPanelTop      = 0.28 
resP@gsnPanelBottom      = 0.05  
gsn_panel(wks,(/plot(0,3),plot(1,3)/),(/1,2/),resP)
exit










