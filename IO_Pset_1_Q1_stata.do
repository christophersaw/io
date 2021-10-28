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

* set store 9 in week 10 as the base market: "store 0"
replace store=0 if store==9 & week==10

* market size
bysort mkt: egen total_sales = total(sales)
bysort store: egen best_week = max(total_sales)
gen mktsize = 2*best_week
gen mktshare = sales/mktsize
sum mktshare
bysort mkt: egen inside = total(mktshare)
bysort mkt: gen outside = 1 - inside


/*
* Market size = count
* calculate market share based on normalized sales
gen mktshare = sales/count


bysort mkt: egen inside = total(mktshare)
sum inside
/* check that share of inside good is always between 0 and 1
    Variable |        Obs        Mean    Std. dev.       Min        Max
-------------+---------------------------------------------------------
      inside |     38,544    .0061503    .0017557   .0005391   .0147917
*/
bysort mkt: gen outside = 1 - inside
*/
gen y = log(mktshare) - log(outside)
local model_1 prom
local model_2 prom tylenol advil bayer
local model_3 prom tylenol#i.store advil#i.store bayer#i.store
*local model_4 prom tylenol#i.store advil#i.store bayer#i.store i.store (results identical to 3)

* Questions 1 to 3 (OLS/Logit model)
forvalues m = 1/3{
	quietly reg y price `model_`m'', noconstant
	outreg2 using PS1Q1_27oct_v3, append
}

* Questions 4 and 5 (IV/Logit model)
* Hausman instruments
bysort product: egen total_price = sum(price)
gen price_excl_own = total_price-price
gen hausmanprice = price_excl_own/3503

forvalues m = 1/3{
	quietly ivregress 2sls y `model_`m'' (price = cost), noconstant
	outreg2 using PS1Q1_27oct_v3, append
}

forvalues m = 1/3{
	quietly ivregress 2sls y `model_`m'' (price = hausmanprice), noconstant
	outreg2 using PS1Q1_27oct_v3, append
}
/*
* Question 6 (own-price elasticities from analytic formula)
gen a_1 = -1.623
gen a_2 = -1.645
gen a_3 = -0.419
forvalues m = 1/3 {
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
      brand |     eta_1     eta_2     eta_3
------------+------------------------------
store brand |  -3.22924 -3.273013 -.8336733
    tylenol | -7.979024 -8.087181 -2.059896
      advil | -8.581788 -8.698115 -2.215508
      bayer |  -5.66405 -5.740827 -1.462253
-------------------------------------------
*/
tabstat eta_*, stat(mean) by(brand) nototal
/*
Summary statistics: Mean
Group variable: brand 

      brand |     eta_1     eta_2     eta_3
------------+------------------------------
store brand | -3.367273 -3.412917 -.8693084
    tylenol | -8.266144 -8.378193  -2.13402
      advil | -8.194322 -8.305397 -2.115478
      bayer | -5.915126 -5.995306 -1.527072
-------------------------------------------
*/
