import datetime as dt
import pandas as pd
import numpy as np

# Prepare list of dicts for STATE drop-down menu
def make_options(df):
    n=df.unitid.nunique()
    state_options=[{'label': 'All '+str(n)+" schools", 'value':''}]
    for state in df.state_name.unique():
        n=df.groupby(['state_name'])['unitid'].nunique().loc[state]
        state_options.append({'label': state+" ("+str(n)+" schools)", 'value': state })
    return state_options

# Standard error of mean distribution
def sem_btwn(x):
    return round(np.std(x)/np.sqrt(x.count()),3)

# Create a summary of early yrs - late yrs:
def tbl_early_late(df):
    tbl = pd.concat([pd.DataFrame(df[0:2].apply(np.mean),columns=['Avg 2001-02']),
            pd.DataFrame(df[-2:].apply(np.mean),columns=['Avg 2016-17'])],axis=1)
    return round(pd.concat([tbl, 
            pd.DataFrame(df[0:2].apply(np.mean) - df[-2:].apply(np.mean), columns=['DIFF'])],axis=1),3)


# Filter data for a balanced dataset w.r.t. measure-of-interest
def get_school_data(df, which_columns=['number_enrolled_total'], earliestYr=0, nYrs=0):
    
    # Keep these columns (subset from CSV) as potential features for model:
    keepcols = ['admission_rate','enrollement_rate','number_applied','number_admitted','number_enrolled_total',
        'rev_tuition_fees_gross', 'rev_tuition_fees_net','rev_total_current','rev_fed_approps_grants','rev_state_local_approps_grants','rev_other',
       'exp_total_current','exp_instruc_total','exp_acad_supp_total','exp_student_serv_total','exp_res_pub_serv_total',
        'completers_150pct','completion_rate_150pct','female_pct','married_pct',
      'year','unitid','inst_name','state_name']
    dropcols = [c for c in df.columns if c not in keepcols]
    df.drop(dropcols,axis=1,inplace=True)

    # Set 'year' as datetime 
    df['year'] = df['year'].apply(pd.to_datetime, format='%Y')

    # Use this to find the most common number of years for which data exists:
    if nYrs==0:
        nYrs = df.groupby(['unitid'])[which_columns[0]].count().value_counts().index[0]

    # Filter years, if desired
    df = df[df.year.dt.year>=earliestYr]

    # Make a temp df with number of years w/ data available for key measure
    tmpDf = df.groupby('unitid')[which_columns].count()

    # Make a list of schools w/ data in all years, only include those:
    unitids = tmpDf[tmpDf[which_columns[0]]==nYrs].index.to_list()
    filt = df.apply(lambda row: row['unitid'] in unitids, axis=1)

    # print(f"These were the input parameters: {which_columns[0]}, {earliestYr}")
    # print(f"Returning a data frame w: {df[filt].unitid.nunique()} schools data in, from {df[filt].year.min().year} thru {df[filt].year.max().year} (that makes {nYrs} yrs of data for {which_columns[0]})")
    
    # Return a dataframe with balanced data for measure of interest
    return df[filt]