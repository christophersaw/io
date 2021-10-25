clear all
cd "/Users/christophersaw/Documents/UCLA YEAR 2/ECON271A INDUSTRIAL ORGANIZATION/PS1/PS1_Data"
import delimited "/Users/christophersaw/Documents/UCLA YEAR 2/ECON271A INDUSTRIAL ORGANIZATION/PS1/PS1_Data/OTC_Data.csv"

* Clean variable names (note that 'brand' is product = brand x packagesize)
rename sales_ sales
rename price_ price
rename cost_ cost
rename prom_ prom
rename brand product

* Brand dummies
gen tylenol = 0
replace tylenol = 1 if product>=1 & product<=3
gen advil = 0
replace advil = 1 if product>=4 & product<=6
gen bayer = 0
replace bayer = 1 if product>=7 & product<=9

* Normalize sales to /50 tablets
gen packagesales = sales
replace sales = 0.5*packagesales if product==1 | product==4 | product==7
replace sales = 2*packagesales if product==3 | product==6 | product==9 | product==11

* Define (unit) price = revenue / normalized sales
gen packageprice = price
replace price = (packageprice*packagesales)/sales

* Define (unit) cost = gross cost / normalized sales
gen packagecost = cost
replace cost = (packagecost*packagesales)/sales

* Market = store x week
gen s = "s"
gen w = "w"
egen mkt_id = concat(s store w week) 
egen mkt = group(mkt_id)
drop s w mkt_id

* Market size = count
* calculate market share based on normalized sales
gen mktshare = sales/count

bysort mkt: egen inside = total(mktshare)
bysort mkt: gen outside = 1 - inside

gen y = log(mktshare) - log(outside)
local model_1 prom
local model_2 prom tylenol advil bayer
local model_3 prom tylenol#i.store advil#i.store bayer#i.store i.store
*local model_4 prom tylenol advil bayer tylenol#i.store advil#i.store bayer#i.store i.store (results identical to 3)

* Questions 1 to 3 (OLS/Logit model)
forvalues m = 1/3{
	quietly reg y price `model_`m''
	outreg2 using PS1Q1, append
}

* Questions 4 and 5 (IV/Logit model)
* Hausman instruments
* sum mkt (there are 3504 unique store-week markets in the data)
quietly {
	forvalues t = 1/3504 {
		gen dummy = 1
		replace dummy = 0 if mkt == `t'
		gen price2 = dummy*price
		replace price2 = . if price2==0
		bysort product week: egen price3_`t' = mean(price2)
		replace price3_`t' = 0 if mkt != `t'
		drop dummy price2
	}
	egen hausmanprice = rowtotal(price3_*)
	drop price3_* 
}

forvalues m = 1/3{
	quietly ivregress 2sls y `model_`m'' (price = cost)
	outreg2 using PS1Q1, append
}

forvalues m = 1/3{
	quietly ivregress 2sls y `model_`m'' (price = hausmanprice)
	outreg2 using PS1Q1, append
}

* Question 6 (own-price elasticities from analytic formula)
gen a_1 = -0.139
gen a_2 = -0.387
gen a_3 = -0.387
gen a_4 = -0.0813
gen a_5 = -0.359
gen a_6 = -0.360
gen a_7 = -0.141
gen a_8 = -0.405
gen a_9 = -0.405

forvalues m = 1/9 {
	gen eta_`m' = a_`m'*price*(1-mktshare)
}
drop a_*
gen brand = 0
replace brand = 1 if tylenol==1
replace brand = 2 if advil==1
replace brand = 3 if bayer==1
label define brand1 0 "store brand" 1 "tylenol" 2 "advil" 3 "bayer"
label values brand brand1

tabstat eta_*, stat(median) by(brand) nototal
/*
Summary statistics: p50
Group variable: brand 

      brand |     eta_1     eta_2     eta_3     eta_4     eta_5     eta_6     eta_7     eta_8     eta_9
------------+------------------------------------------------------------------------------------------
store brand | -.2765646 -.7700037 -.7700037 -.1617605 -.7142928 -.7162825  -.280544 -.8058178 -.8058178
    tylenol | -.6833545 -1.902577 -1.902577 -.3996886 -1.764923 -1.769839 -.6931869 -1.991069 -1.991069
      advil | -.7349775 -2.046304 -2.046304 -.4298825 -1.898251 -1.903539 -.7455528 -2.141481 -2.141481
      bayer | -.4850911 -1.350578 -1.350578  -.283726 -1.252861 -1.256351 -.4920709 -1.413395 -1.413395
-------------------------------------------------------------------------------------------------------
*/

tabstat eta_*, stat(mean) by(brand) nototal
/*
Summary statistics: Mean
Group variable: brand 

      brand |     eta_1     eta_2     eta_3     eta_4     eta_5     eta_6     eta_7     eta_8     eta_9
------------+------------------------------------------------------------------------------------------
store brand | -.2883863 -.8029173 -.8029173 -.1686749 -.7448251 -.7468998 -.2925358 -.8402623 -.8402623
    tylenol | -.7079446  -1.97104  -1.97104 -.4140712 -1.828432 -1.833526 -.7181308 -2.062716 -2.062716
      advil | -.7017934 -1.953914 -1.953914 -.4104734 -1.812546 -1.817595 -.7118912 -2.044794 -2.044794
      bayer | -.5065942 -1.410446 -1.410446  -.296303 -1.308398 -1.312043 -.5138834 -1.476048 -1.476048
-------------------------------------------------------------------------------------------------------
*/
