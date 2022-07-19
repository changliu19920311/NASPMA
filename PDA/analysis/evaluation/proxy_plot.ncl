

; ======evaluating the reconstucted monthly data=================
 load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
 load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
 load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
 load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"
;================set the parameters=====================
LatS=10
LatN=90
LonW=-60
LonE=-145
diri1="/Users/chang/Desktop/LMR_NA/output/"
expr=(/"output_NASPA/","output_NASPA2/","output_global/"/)
Season1=(/"WARM","COOL"/)
dum_fill_0=new((/dimsizes(Season1),dimsizes(expr)/),"graphic")
dum_fill_1=new((/dimsizes(Season1),dimsizes(expr)/),"graphic")
dum_fill_2=new((/dimsizes(Season1),dimsizes(expr)/),"graphic")
dum_fill_3=new((/dimsizes(Season1),dimsizes(expr)/),"graphic")
dum_hollow=new((/dimsizes(Season1),dimsizes(expr)/),"graphic")
plot=new((/dimsizes(Season1),dimsizes(expr)/),"graphic")
wks = gsn_open_wks("pdf" ,"proxy_distribution")      ; send graphics to PNG file
gsn_define_colormap(wks,"amwg_blueyellowred") 
type_unique =(/"Corals and Sclerosponges_Rates","Corals and Sclerosponges_SrCa","Corals and Sclerosponges_d18O","Ice Cores_Accumulation","Ice Cores_MeltFeature","Ice Cores_d18O","Ice Cores_dD","Lake Cores_Misc","Lake Cores_Varve","Tree Rings_WidthBreit","Tree Rings_WidthPages2","Tree Rings_WoodDensity","Tree Rings_Isotopes","Warm Season Tree Rings","Cool Season Tree Rings","Seasonal Tree Rings","Marine Cores_d18O","Speleothems_d18O"/)          

do ns=0,dimsizes(Season1)-1
      do ne=0,dimsizes(expr)-1
            lines = asciiread(diri1+expr(ne)+"/test_recon_season_GPCC_"+Season1(ns)+"/VerifFigs/lat_lon.txt",-1,"string")
            delim = ","
            lat_sta =  tofloat(str_get_field(lines,1,delim))
            lon_sta =  tofloat(str_get_field(lines,2,delim))
            type  =  str_get_field(lines,3,delim)
            res                       = True     ; plot mods desired
            res@gsnDraw=False
            res@gsnFrame=False
            res@mpFillOn                    = False        ; turn off gray fill
            res@mpOutlineBoundarySets       = "National"   ; turn on country boundaries
            res@mpGeophysicalLineColor      = "gray"       ; color of cont. outlines
            res@mpGeophysicalLineThicknessF = 1.5          ; thickness of outlines
            res@gsnLeftStringFontHeightF =0.022
            res@gsnRightStringFontHeightF =0.022
            res@mpMinLatF            =  LatS    ; range to zoom in on
            res@mpMaxLatF            =  LatN
            res@mpMinLonF            =  LonE
            res@mpMaxLonF            =  LonW
            plot(ns,ne) = gsn_csm_map(wks,res)

            mkres =True
            mkres@gsMarkerIndex  = 16          ; Filled dots
            mkres@gsMarkerSizeF  = 0.007
            mkres@gsMarkerIndex  = 16
            iyyyy=ind(type.eq.type_unique(9).or.type.eq.type_unique(10).or.type.eq.type_unique(11).or.type.eq.type_unique(12).or.type.eq.type_unique(13).or.type.eq.type_unique(14).or.type.eq.type_unique(15))
            mkres@gsMarkerColor  = "forestgreen"
            if(.not.all(ismissing(iyyyy))) then
                  dum_fill_0(ns,ne) = gsn_add_polymarker(wks,plot(ns,ne),lon_sta(iyyyy),lat_sta(iyyyy),mkres)
            end if
            delete(iyyyy)
            iyyyy=ind(type.eq.type_unique(7).or.type.eq.type_unique(8))
            mkres@gsMarkerColor  = "red"
            if(.not.all(ismissing(iyyyy))) then
                  dum_fill_1(ns,ne) = gsn_add_polymarker(wks,plot(ns,ne),lon_sta(iyyyy),lat_sta(iyyyy),mkres)
            end if
            delete(iyyyy)
            iyyyy=ind(type.eq.type_unique(3).or.type.eq.type_unique(4).or.type.eq.type_unique(5).or.type.eq.type_unique(6))
            mkres@gsMarkerColor  = "blue"
            if(.not.all(ismissing(iyyyy))) then
                  dum_fill_2(ns,ne) = gsn_add_polymarker(wks,plot(ns,ne),lon_sta(iyyyy),lat_sta(iyyyy),mkres)
            end if
            delete(iyyyy)
            iyyyy=ind(type.eq.type_unique(0).or.type.eq.type_unique(1).or.type.eq.type_unique(2))
            mkres@gsMarkerColor  = "orange"
            if(.not.all(ismissing(iyyyy))) then
                  dum_fill_3(ns,ne) = gsn_add_polymarker(wks,plot(ns,ne),lon_sta(iyyyy),lat_sta(iyyyy),mkres)
            end if
            delete(iyyyy) 
            mkres@gsMarkerIndex = 4           ; Hollow dots
            mkres@gsMarkerColor = "black"
            dum_hollow(ns,ne) = gsn_add_polymarker(wks,plot(ns,ne),lon_sta,lat_sta,mkres)

            delete(lines)
            delete(lat_sta)
            delete(lon_sta)
            delete(type)
      end do
end do
  resP                     = True         ; modify the panel plot
  resP@gsnPanelYWhiteSpacePercent = 5
  gsn_panel(wks,(/plot(0,0),plot(0,1),plot(0,2)/),(/3,1/),resP) 
  gsn_panel(wks,(/plot(1,0),plot(1,1),plot(1,2)/),(/3,1/),resP) 








