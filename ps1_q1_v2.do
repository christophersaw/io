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

* Market size = all sales of headache medicine
* Define store brands as the outside good
bysort mkt: egen totalsales = total(sales)
gen mktshare = sales/totalsales
replace mktshare = . if product==10 | product==11

bysort mkt: egen inside = total(mktshare)
bysort mkt: gen outside = 1 - inside

gen y = log(mktshare) - log(outside)
replace y = . if product==10 | product==11
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
tabstat eta_*, stat(mean median min max) by(tylenol advil bayer)

