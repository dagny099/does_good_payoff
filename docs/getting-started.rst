Getting started
===============

*This is where you describe how to get set up on a clean install, including the
commands necessary to get the raw data (using the `sync_data_from_s3` command,
for example), and then how to make the cleaned, final data sets.*


# URBAN INSTITUTE - Education Data Explorer

From: https://educationdata.urban.org/data-explorer/colleges/

Scroll down until you see these two tabs:

## "CHOOSE INSTITUTIONS" tab


### GEOGRAPHY 

Rd_1: TX, FL   ... Approximate results: 188k records from 838 institutions
Rd_2: OR, AZ, MI, OH ... Approximate results: 169k records from 752 institutions
Rd_3: NY, GA, MN  ... Approximate results: 170k records from 755 institutions
Rd_4: CA  ... Approximate results: 164k records from 729 institutions
Rd_5: VA, PA, IN, WI  ... Approximate results: 178k records from 791 institutions 

Note: There appears to be a query limit of around 200k records, hence multiple rounds of downloads. 

**State**
- Select 1 or more states. You can't choose data until you've selected at least one. 

**Institution (within selected states)**
- I tried different things here. Ultimately leave blank to get most data. 

**Institution level**
- "Four or more years"

**Institution control**
- "Public"


### Time Frame
- Select a Start year and End year (defaults to 2017 End year)

**Start year:**  1990  
**End year:** 2017


## "CHOOSE YOUR DATA" tab

Each of the bolded categories listed has one or more metrics, visible from the drop-down. 

By default, some things are pre-checked. Here's what I ended up choosing from each category:

Admissions information
- Number of applicants  (2001-2017)
- Number of admissions  (2001-2017)
- Number of students enrolled full-time   (2001-2017)
- Number of students enrolled part-time   (2001-2017)
- Number of students enrolled

College characteristics
- Institutional category  (2004-2017)
- Institutional size category  (2004-2017)
- Student-faculty ratio  (2009-2017)

Geographic and identification information
– Unit ID number
– Institution (entity) name
- Zip code
- Bureau of Economic Analysis (BEA) regions
- Longitude of institution
- Latitude of institution

Finance information
- Total revenue, investment return, and other additions  (1996-2017)
- Total expenses and deductions  (1996-2017)

- Revenue: Tuition and fees
- Revenue: Tuition and fees, net of discounts and allowances  (1996-2017)
- Revenue: Federal appropriations, grants, and contracts
- Revenue: State and local appropriations, grants, and contracts
- Revenue: Local appropriations
- Revenue: Local grants and contracts
- Revenue: Private gifts, grants, and contracts
- Revenue: Other sources
- Revenue: Total operating revenue  (2001-2017)
- Revenue: Total nonoperating revenue  (2001-2017)
- Revenue: Total

- Expenditures: Instruction—total
- Expenditures: Research—total
- Expenditures: Public service—total
- Expenditures: Research and public service—total
- Expenditures: Academic support—total
- Expenditures: Student services—total
- Expenditures: Total
- Expenditures: Salaries and wages
- Expenditures: Benefits


Earnings information
- Students not working and not enrolled  (2003-2014)
- Students working and not enrolled  (2003-2014)  
- Mean earnings of students working and not enrolled  (2003-2014)

Enrollment and student characteristics
- 12-month instructional activity contact hours for undergraduates  (1996-2016)
- Reported full-time equivalent enrollment  (1996-2016)
- Unduplicated head count of students enrolled over a 12-month period  (1996-2016)

Default and repayment information
- Cohort default rate  (1996-2017)
- Number of students in the cohort for the cohort default rate  (1996-2017)

Graduation information
- Completers within 150% of normal time  (1996-2016)
- Graduation rate within 150% of normal time  (1996-2016)

Room and board charges information
- Combined charge for room and board  (1992-2017)

Student aid applicant characteristics
- Share of financially independent students  (1997-2016)
- Share of female students  (1997-2016)
- Share of married students  (1997-2016)
- Median family income (in real 2015 dollars)  (1997-2016)


