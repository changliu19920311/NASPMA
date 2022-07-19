; ======calculate and plot RE between true vector and estimation using forced prior/unforced prior =================
 load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
 load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
 load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
 load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"

diri="/Users/chang/Desktop/LMR_test/data/data/model/ccsm4_last_millenium/"
variable=(/"pr","zg","zg","ua"/)
names=(/"pr_sfc_Amon_CCSM4_past1000_085001-185012.nc","zg_700hpa_Amon_CCSM4_past1000_085001-185012.nc","zg_500hpa_Amon_CCSM4_past1000_085001-185012.nc","ua_200hpa_Amon_CCSM4_past1000_085001-185012.nc"/)
;======CCSM4 prate data==========
do nm=0,dimsizes(names)-1
	f  = addfile(diri+names(nm),"r")
	time=f->time
	pr=f->$variable(nm)$
	experiments=(/"test_recon_season_GPCC_WARM","test_recon_season_GPCC_COOL"/)
	do ne=0,dimsizes(experiments)-1
		data=asciiread("./"+experiments(ne)+"/r0/prior_sampling_info.txt",-1,"integer")
	    if(ne.eq.0) then
			pr_MJJ=month_to_season(pr,"MJJ")
			pr_MJJ_clm=dim_avg_n_Wrap(pr_MJJ(data,:,:),0)
		end if
		if(ne.eq.1) then
			utc_date = cd_calendar(time, 0)
			year   = tointeger(utc_date(:,0)) 
			month  = tointeger(utc_date(:,1)) 
			Year=ispan(min(year),max(year),1)
			pr_DJFMA=pr_MJJ
			do ny=0,dimsizes(Year)-1
			    if(ny.eq.0) then
			      iyyyy=ind(year.eq.Year(ny).and.month.le.4)
			    else
			      iyyyy=ind((year.eq.Year(ny).and.month.le.4).or.(year.eq.Year(ny-1).and.month.eq.12))
			    end if
			   	pr_DJFMA(ny,:,:)=dim_avg_n_Wrap(pr(iyyyy,:,:),0)
			    delete(iyyyy)
			end do
			pr_DJFMA_clm=dim_avg_n_Wrap(pr_DJFMA(data,:,:),0)
	    end if
		setfileoption("nc","Format","LargeFile") 
		system("rm ./"+experiments(ne)+"/clm_"+names(nm))
		fout=addfile("./"+experiments(ne)+"/clm_"+names(nm),"c")
		setfileoption("nc","DefineMode",True)
		if(ne.eq.0) then
			fout->var_clm=pr_MJJ_clm
		end if
		if(ne.eq.1) then
			fout->var_clm=pr_DJFMA_clm
		end if
	end do
end do

