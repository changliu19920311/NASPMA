; ======calculate and plot RE between true vector and estimation using forced prior/unforced prior =================
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
 load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
 load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
 load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"
;================set the parameters=====================
noise="white_noise"
season=(/"warm","cool"/)
seasons=(/"Warm season","Cool season"/)
periods = (/"1-2000"/)
str_period = 1
end_period = 2000
;the region of North America
LatS=10
LatN=55
LonW=-60
LonE=-135
colors = (/"black","purple","blue","red"/)
plot=new((/dimsizes(season)/),"graphic")

wks = gsn_open_wks("pdf" ,"prior_number")      ; send graphics to PNG file
gsn_define_colormap(wks,"amwg_blueyellowred") 
;===========ending of parameters setting==========================
do ns=0,dimsizes(season)-1
      rc_ave = new((/8/),"float")
      ce_ave = new((/8/),"float")
     
            ;==========read the data==========
            f = addfile("/Users/jjyan/Desktop/NA_reconstrution/data/"+season(ns)+"_season/prior/pr_sfc_Amon_config_season_prior_number.nc","r")
            time=f->time
            year=ispan(1,dimsizes(time),1)
            iyyyy=ind(year.ge.str_period.and.year.le.end_period)
            pr_a=f->pr_sfc_Anom_analysis  ; the estimation analysis state with different number of priors
            pr=pr_a(ite|:,time|iyyyy,lat|:,lon|:)
            delete(pr_a)
            lat=f->lat
            lon=f->lon
            if(max(lon).ge.200) then
                  pr&lon=pr&lon-360
                  lon=pr&lon
            end if


            pr_tru=f->pr_sfc_Amon_truth
            pr_truth=pr_tru(time|iyyyy,lat|:,lon|:)       ; the true vector
            latitude=f->lat
            longitude=f->lon
            if(max(longitude).ge.200) then
                  pr_truth&lon=pr_truth&lon-360
                  longitude=pr_truth&lon
            end if
            delete(f)
            delete(pr_tru)

            ;======mask ============================
            a    = addfile("$NCARG_ROOT/lib/ncarg/data/cdf/landsea.nc","r")
            lsdata = a->LSMASK
            lsm  = landsea_mask(lsdata,lat,lon)
            pr = mask(pr,lsm.eq.0,False)
            pr = mask(pr,lsm.eq.2,False)
            pr = mask(pr,lsm.eq.4,False)
            delete(a)
            delete(lsm)


            ;========calculate the correlation==============
            rad    = 4.0*atan(1.0)/180.0
            rr     = 6371220.0*rad
            ilat=ind(lat.ge.LatS.and.lat.le.LatN)
            clat   = cos(lat(ilat)*rad)
            rc = new((/dimsizes(pr(:,0,0,0)),dimsizes(pr(0,0,{LatS:LatN},0)),dimsizes(pr(0,0,0,{LonW:LonE}))/),"float")
            do nt=0,dimsizes(rc(:,0,0))-1
                  rc(nt,:,:)=tofloat(escorc_n(pr(nt,:,{LatS:LatN},{LonW:LonE}),pr_truth(:,{LatS:LatN},{LonW:LonE}),0,0))
            end do
            rc_ave(:) = wgt_areaave(rc , clat, 1.0, 0)
            printVarSummary(rc_ave)
   
            rc_ave(7)= rc_ave(7)-rc_ave(6)
            rc_ave(6)= rc_ave(6)-rc_ave(5)
            rc_ave(5)= rc_ave(5)-rc_ave(4)
            rc_ave(4)= rc_ave(4)-rc_ave(3)                        
            rc_ave(3)= rc_ave(3)-rc_ave(2)            
            rc_ave(2)= rc_ave(2)-rc_ave(1)
            rc_ave(1)= rc_ave(1)-rc_ave(0)
            rc_ave(0) = rc_ave@_FillValue
            ;========calculate the CE==============
            ce = new((/dimsizes(pr(:,0,0,0)),dimsizes(pr(0,0,{LatS:LatN},0)),dimsizes(pr(0,0,0,{LonW:LonE}))/),"float")
            do nt=0,dimsizes(ce(:,0,0))-1
                  truth_value_ave=dim_avg_n_Wrap(pr_truth(:,{LatS:LatN},{LonW:LonE}),0)
                  truth_value_ave_copy = conform(pr_truth(:,{LatS:LatN},{LonW:LonE}),truth_value_ave, (/1,2/))
                  aa=(pr_truth(:,{LatS:LatN},{LonW:LonE})-truth_value_ave_copy)^2
                  tt_1=dim_sum_n_Wrap(aa,0)
                  delete(aa)
                  aa=(pr(nt,:,{LatS:LatN},{LonW:LonE})-pr_truth(:,{LatS:LatN},{LonW:LonE}))^2
                  tt_2=dim_sum_n_Wrap(aa,0)
                  delete(aa)
                  ce(nt,:,:)=tofloat(1-tt_2/tt_1)
            end do
            ce = where(ce.le.-1,ce@_FillValue,ce)
            copy_VarCoords(pr(:,0,{LatS:LatN},{LonW:LonE}),ce) 
            ce_ave(:) = wgt_areaave(ce , clat, 1.0, 0)
            ce_ave(7)= ce_ave(7)-ce_ave(6)
            ce_ave(6)= ce_ave(6)-ce_ave(5)
            ce_ave(5)= ce_ave(5)-ce_ave(4)
            ce_ave(4)= ce_ave(4)-ce_ave(3)                        
            ce_ave(3)= ce_ave(3)-ce_ave(2)            
            ce_ave(2)= ce_ave(2)-ce_ave(1)
            ce_ave(1)= ce_ave(1)-ce_ave(0)
            ce_ave(0) = ce_ave@_FillValue
            delete(truth_value_ave)
            delete(truth_value_ave_copy)
            delete(tt_1)
            delete(tt_2)
            delete(pr)
            delete(pr_truth)
            delete(iyyyy)
            delete(time)
            delete(year)
            delete(lat)
            delete(lon)
            delete(latitude)
            delete(longitude)
            delete(clat)
            delete(rc)
            delete(ce)
            delete(ilat)
   
       ;============plot============



     res                    = True                    ; plot mods desired
      res@gsnFrame=False
      res@gsnDraw=False
      res@tmXBMode          = "Explicit"              ; explicit labels
      res@tmXBValues        = ispan(1,7,1)                       ; location of labels 
      res@trXMinF  =1.
      res@tmXBLabels        = (/"10 - 5","20 - 10","50 - 20","100 - 50","200 - 100","300 - 200","500 - 300"/)  
      res@xyDashPattern  = 0  
      res@gsnLeftStringFontHeightF =0.018
      res@xyLineThicknessF   = 2.                      ; line thickness
      res@vpWidthF=0.7
      res@vpHeightF=0.3
        res@tmXTOn= False
      res@tmYROn= False
      res@tiXAxisString  ="Number of priors"
      res@xyLineColors       = (/"red","blue"/)
      if(ns.eq.0) then
        res@trYMaxF   =  0.1
        res@trYMinF  =-0.01
        res@gsnLeftString="(a) "+seasons(ns)
      else
        res@trYMaxF   =  0.04
        res@trYMinF  =-0.01        
       res@gsnLeftString="(b) "+seasons(ns)
      end if
      res@gsnYRefLine           = 0.0   
      res@pmLegendDisplayMode    = "Always"            ; turn on legend
          res@pmLegendSide           = "Top"               ; Change location of 
          res@pmLegendParallelPosF   = .12                 ; move units right
          res@pmLegendOrthogonalPosF = -0.35               ; move units down
          res@pmLegendWidthF         = 0.05              ; Change width and
          res@pmLegendHeightF        = 0.05              ; height of legend.
          res@lgPerimOn              = False               ; turn off box around
          res@lgLabelFontHeightF     = .015                 ; label font height
          res@xyExplicitLegendLabels = (/"r","CE"/)
      plot(ns)  = gsn_csm_xy (wks,ispan(1,7,1),(/rc_ave(1:),ce_ave(1:)/),res)   ;  create plot
      delete(res)
end do
resP                     = True         
resP@gsnPanelYWhiteSpacePercent=5 
resP@gsnFrame=False
resP@gsnPanelXWhiteSpacePercent=5 
gsn_panel(wks,(/plot(0),plot(1)/),(/2,1/),resP)
exit