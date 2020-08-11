import time
import pandas as pd
import numpy as np

pd.options.display.float_format = '{:.3f}'.format

# Import saved csv to dataframe:
path2file="./data/interim/analyzeMe_n175.csv"
df = pd.read_csv(path2file, na_values=np.nan, parse_dates=['year'], 
                   dtype={'unitid':'category', 'inst_name': 'category', 'state_name': 'category',
                         'enrollement_rate': 'float64', 'female_pct': 'float64', 'married_pct': 'float64'})

    # --------------------------------
    # HELPER FUNCTIONS
    # --------------------------------

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

def augmented_dicky_fuller_statistics(time_series):
    from statsmodels.tsa.stattools import adfuller
    """
    Run the augmented Dicky-Fuller test on a time series to determine if it's stationary
    Arguments:
        time_series:  series. Time series to be tested. 
    Output: 
        Test statistics for the augmented Dicky Fuller test in the console
    """
    result = adfuller(time_series.values)
#     print('Critical Values:')
#     for key, value in result[4].items():
#         print('\t%s: %.3f' % (key, value))
    return result

def make_item(i):
    # we use this function to make the example items to avoid code duplication
    return dbc.Card(
        [
            dbc.CardHeader(
                html.H2(
                    dbc.Button(
                        f"Collapsible group #{i}",
                        color="link",
                        id=f"group-{i}-toggle",
                    )
                )
            ),
            dbc.Collapse(
                dbc.CardBody(f"This is the content of group {i}..."),
                id=f"collapse-{i}",
            ),
        ]
    )


# --------------------------------
# Prune Features, axis=1: (Drop some columns)
# --------------------------------

# Keep these columns (subset from CSV) as potential features for model:
keepcols = ['admission_rate','enrollement_rate','number_applied','number_admitted','number_enrolled_total',
        'rev_tuition_fees_gross', 'rev_tuition_fees_net','rev_total_current','rev_fed_approps_grants','rev_state_local_approps_grants','rev_other',
       'exp_total_current','exp_instruc_total','exp_acad_supp_total','exp_student_serv_total','exp_res_pub_serv_total',
        'completers_150pct','completion_rate_150pct','female_pct','married_pct',
      'year','unitid','inst_name','state_name']

dropcols = [c for c in df.columns if c not in keepcols]
df.drop(dropcols,axis=1,inplace=True)

# --------------------------------
# Prune Data, axis=0: (Check for availability of key measure in enough years to make a time series)
# --------------------------------

# DATA FOR ENROLLMENT SECTION
df_enroll  = get_school_data(df, ['enrollement_rate'], 2001)
state_options_enrollment = make_options(df_enroll)

# DATA FOR FINANCE SECTION
# which_columns = ['rev_total_current','exp_total_current','rev_tuition_fees_gross','rev_tuition_fees_net','exp_instruc_total']
dfFin1 = get_school_data(df, ['rev_total_current'])
state_options_finance = make_options(dfFin1)

# --------------------------------
# Prep data for showing Expenses & Revenue breakdowns
# --------------------------------
dfFin1['unitid'] = dfFin1['unitid'].astype('object')  #for plotting later
dfFin1['inst_name'] = dfFin1['inst_name'].astype('object')   #for plotting later

# Make a Df to show breakdown of expenses:
dfFin1['exp_OTHER_STUFF'] = dfFin1['exp_total_current'] - \
    (dfFin1['exp_instruc_total']+dfFin1['exp_res_pub_serv_total']+dfFin1['exp_acad_supp_total']+dfFin1['exp_student_serv_total'])

dfFin1['rev_OTHER_STUFF'] = dfFin1['rev_total_current'] - \
    (dfFin1['rev_tuition_fees_net']+dfFin1['rev_fed_approps_grants']+dfFin1['rev_state_local_approps_grants']+dfFin1['rev_other'])

which_cols = ['exp_instruc_total', 'exp_res_pub_serv_total', 'exp_acad_supp_total', 'exp_student_serv_total', 'exp_OTHER_STUFF']
dfExp = pd.melt(dfFin1, id_vars =['year','unitid','inst_name','state_name'], value_vars= which_cols)

which_cols = ['rev_tuition_fees_net', 'rev_fed_approps_grants', 'rev_state_local_approps_grants', 'rev_other', 'rev_OTHER_STUFF']
dfRev = pd.melt(dfFin1, id_vars =['year','unitid','inst_name','state_name'], value_vars= which_cols)

# -----------------------------
# MAKE SUMMARY STATS DATA TABLE (summary_transformed)
# -----------------------------
# STEP 1 DEFINE A MULTILEVEL INDEX
keepcols = keepcols[:-4]
index=[]
for k in keepcols:
    index.append((k, 'Raw'))
    index.append((k, 'Log'))
    index.append((k, 'Log-Diff'))
    index.append((k, 'Diff'))

# STEP 2: CREATE ALL THE SERIES I NEED TO SUMMARIZE:
mean, cnt, varz, adf_stat, adf_pval = [], [], [], [], []
for t in index:
    if t[1]=='Raw':
        ser=df[t[0]]
        ts=df[[t[0],'year']].groupby('year').agg(np.mean)
    elif t[1]=='Log':
        ser=np.log(df[t[0]])
        df['tmp']=np.log(df[t[0]])
        ts=df[['tmp','year']].groupby('year').agg(np.mean)
    elif t[1]=='Log-Diff':
        ser=np.log(df[t[0]]).diff(1)
        df['tmp']=np.log(df[t[0]]).diff(1)
        ts=df[['tmp','year']].groupby('year').agg(np.mean)
    elif t[1]=='Diff':
        ser=df[t[0]].diff(1)
        df['tmp']=df[t[0]].diff(1)
        ts=df[['tmp','year']].groupby('year').agg(np.mean)
    # mean, cnt, and varz are collapsed across all years, unitids for that feature:
    mean.append(np.mean(ser))
    cnt.append(len(ser.dropna()))
    varz.append(np.var(ser))   
    # adf test looks at stationarity of feature over time, store results:
    res=augmented_dicky_fuller_statistics(ts.dropna())
    adf_stat.append(res[0])
    adf_pval.append(res[1])
    df.drop('tmp',axis=1,inplace=True,errors='ignore')

# STEP 3: PUT IT ALL TOGETHER, 
index = pd.MultiIndex.from_tuples(index)

tmp1 = pd.DataFrame(mean, index=index, columns=['mean']).reindex(index)
tmp2 = pd.DataFrame(cnt, index=index, columns=['count']).reindex(index)
tmp3 = pd.DataFrame(varz, index=index, columns=['variance']).reindex(index)
tmp4 = pd.DataFrame(adf_stat, index=index, columns=['ADF_stat']).reindex(index)
tmp5 = pd.DataFrame(adf_pval, index=index, columns=['ADF_pval']).reindex(index)

summary_transformed = pd.concat([tmp2, tmp1, tmp3, tmp4, tmp5], axis=1)
# --------------------------------

# --------------------------------
# Set figure aesthetics and labels and more:
# --------------------------------
seriez = {'number_applied': {'color': '#F44DDB', 'label': "Number of student applications", 'timeDep': 'Yes', 'marker_color': 'rgba(152, 0, 0, .8)'},
            'number_admitted': {'color': '#CF1214', 'label': "Number of students admitted", 'timeDep': 'Yes', 'marker_color': 'rgba(152, 0, 0, .8)'},
            'number_enrolled_total': {'color': '#0E3DEC', 'label': "Number enrolled", 'timeDep': 'Yes', 'marker_color': 'rgba(152, 0, 0, .8)'},
            'admission_rate': {'color': '#CF1214', 'label': "Admission rate (# applications/# admissions)", 'timeDep': 'Yes', 'marker_color': 'rgba(152, 0, 0, .8)'},
            'enrollement_rate': {'color': '#0E3DEC', 'label': "Enrollment rate (# admissions/# enrolled", 'timeDep': 'Yes', 'marker_color': 'rgba(152, 0, 0, .8)'},
            'rev_tuition_fees_gross': {'color': '#8E44AD', 'label': "Revenue: Tuition&Fees (gross)", 'timeDep': 'Yes', 'marker_color': 'rgba(250, 250, 250, .8)'},
            'rev_tuition_fees_net': {'color': '#2E86C1 ', 'label': "Revenue: Tuition&Fees (net)", 'timeDep': 'Yes', 'marker_color': 'rgba(250, 250, 250, .8)'},
            'rev_total_current': {'color': '#1E8449', 'label': "Revenue: Total", 'timeDep': 'Yes', 'marker_color': 'rgba(250, 250, 250, .8)'},
            'rev_fed_approps_grants': {'color': '#F1948A', 'label': "Revenue: Fed grants & approp", 'timeDep': 'Yes', 'marker_color': 'rgba(250, 250, 250, .8)'},
            'rev_state_local_approps_grants': {'color': '#C0392B', 'label': "Revenue: State grants & approp", 'timeDep': 'Yes', 'marker_color': 'rgba(250, 250, 250, .8)'},
            'rev_other': {'color': '#00ACC1', 'label': "Revenue: Other", 'timeDep': 'Yes', 'marker_color': 'rgba(250, 250, 250, .8)'},
            'exp_total_current': {'color': '#F4D03F', 'label': "Expenses: Total", 'timeDep': 'Yes', 'marker_color': 'rgba(250, 250, 250, .8)'},
            'exp_instruc_total': {'color': '#FFA726', 'label': "Expenses: Instruction", 'timeDep': 'Yes', 'marker_color': 'rgba(250, 250, 250, .8)'},
            'exp_acad_supp_total': {'color': '#B2EBF2', 'label': "Expenses: Acad Supp", 'timeDep': 'Yes', 'marker_color': 'rgba(250, 250, 250, .8)'},
            'exp_student_serv_total': {'color': '##D7CCC8', 'label': "Expenses: Stud Serv", 'timeDep': 'Yes', 'marker_color': 'rgba(250, 250, 250, .8)'},
            'exp_res_pub_serv_total': {'color': '#0E3DEC', 'label': "Expenses: Resch & Pub Serv", 'timeDep': 'Yes', 'marker_color': 'rgba(250, 250, 250, .8)'},
            'completers_150pct': {'color': '#0E3DEC', 'label': "Completers 150%", 'timeDep': 'Yes', 'marker_color': 'rgba(250, 250, 250, .8)'},
            'completion_rate_150pct': {'color': '#0E3DEC', 'label': "Completion Rate 150%", 'timeDep': 'Yes', 'marker_color': 'rgba(250, 250, 250, .8)'},
            'female_pct': {'color': '#0E3DEC', 'label': "% Female", 'timeDep': 'Yes', 'marker_color': 'rgba(250, 250, 250, .8)'},
            'married_pct': {'color': '#0E3DEC', 'label': "% Marries", 'timeDep': 'Yes', 'marker_color': 'rgba(250, 250, 250, .8)'},
            'state_name': {'color': '#0E3DEC', 'label': "State", 'timeDep': 'No', 'marker_color': 'rgba(250, 250, 250, .8)'},
        }

# --------------------------------
# Move this to a text file and load variables
# --------------------------------
top_markdown_text = '''
**v1.2**
This project models enrollment trends at public universities (2001-17) with classical time series forecasting as well as machine learning methods.

SET THE SCENE:  
Here we are in the summer of 2020, with a country full of would-be college freshmen heading into a fall semester with a LOT of uncertainty.
Imagine being a student (or their parents) deciding whether Tuition (and Fees!) will be "worth it" at this point in time. 

LET'S BACK UP A MOMENT:  
Remember way back at the beginning of the year, when college-related journalism was about the outrage over admissions cheating scandals or grumpiness about universities becoming more selective over time? 

SET THE STORY STRAIGHT ABOUT SELECTIVITY OVER TIME:  
Pew research (2019) reports that ["Majority of US colleges admit most of their applicants."](https://www.pewresearch.org/fact-tank/2019/04/09/a-majority-of-u-s-colleges-admit-most-students-who-apply/#:~:text=But%20for%20all%20the%20attention,Center%20analysis%20of%20U.S.%20Education)

TO HIGHLIGHT THE IMPORTANCE OF THIS QUESTION:  
**Public colleges and universities educate nearly 75% of all college students.**

PROPOSED METHOD:  
-Collect panel data on university admissions, enrollment, finances, and graduation stats over time.  
-Goal is to model the number of students enrolling at a specific public university (like a "within subjects" model).  
-Train models on a school's data from 2001-16, evaluate performance on predicting 2017 enrollment.  

COMPARE PERFORMANCE:  
-> **Classical Time Series forecasting methods**, like ARIMA and Holt Winter's Exponential Smoothing  
-> **Machine Learning methods** for Time Series forecasting, perhaps something Bayesian and XGBoost  

BUILD A VISUAL TIME SERIES TEACHING TOOL!  
-Use this dashboard to visualize the accuracy and uncertainty of different Models after they're trained  
-The actual 2017 numbers will stay visualized  
-User will see a list of checkboxes with choices of how to forecast (e.g. OLS, Fixed effects, Random effects, ML-based model)  
-Graph will dynamically show the forecasted predictions and uncertainty  

'''

markdown_text_background_1="""

## 1st:  Let's Explore the Data: 
**The goal is to model the blue line (number of students enrolled) as a time series composed of time-varying and non-time varying predictor variables.**
"""
markdown_text_background_2= """


## 2nd:  Visualize Finance time series: 
**Features of the model will include institution features, such as proportion of expenses spent on instruction. Below is a view of the raw data. **
"""

markdown_approach="""
# Why have enrollment rates at public universities been decreasing over the last 2 decades?

## PLAN OF ATTACK: 
Collect panel data on university admissions, enrollment, finances, and graduation stats over time.

### 1) Use panel data on a subset of the universities to develop a model
- **Time Series of Enrollment = (Base Level) + (Trend) + (Seasonality) + (Error)**
- Use a Fixed effects model to look at time characteristics of predictor variables on time series of dependent var
- Cluster by state (OLS w NW SE's adjustment for heteroscedasticity and autocorrelation)
- Random effects model(?) to look at cross sectional variation, independent of time
- HOW TO TEST: For each model, predict enrollment in 2017 based on training on data from 2001-2016

If model-ability of this variable still seems viable then...

### 2) Collect a larger dataset
Using same criteria as before:
- Public university
- Has enrollment data from 2001 or 2001, AND 2017, AND at least 5 other time points
- Incoming enrollment class >999

### 3) With more data, expand the models to compare:
--> Classical Time Series forecasting methods, like ARIMA and Holt Winter's Exponential Smoothing
--> Machine Learning methods for Time Series forecasting, perhaps something Bayesian and XGBoost
    
### 4) Publish app with results 
"""

# --------------------------------
# Import stats modules:
from sklearn.metrics import r2_score

# --------------------------------
# Import dash and visualization modules:
# --------------------------------

import dash
import dash_table
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

# Prune as necessary:
import chart_studio.plotly as py
import plotly.graph_objects as go
import plotly.express as px
import cufflinks as cf
import plotly
import seaborn as sns
import matplotlib.pyplot as plt

# --------------------------------
# DASH APP!
# --------------------------------
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', dbc.themes.BOOTSTRAP]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server = app.server

tabs_styles = {
    'height': '30px',
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}
# 
# tab_selected_style = {
#     'box-shadow': '1px 1px 0px white',
#     'borderTop': '3px solid #e36209',
#     'borderBottom': '1px solid #d6d6d6',
#     # 'backgroundColor': '#119DFF',
#     # 'color': 'white',
#     'padding': '6px'
# }

app.layout = dbc.Container(
    [
        #HEADER
        html.Div([html.H1("Data Incubator Project Proposal")],
            style={'textAlign': "center", "padding-bottom": "5"}),
        html.Hr(),
        
        #PROJECT DESCRIPTION AT THE TOP (COLLAPSABLE)
        html.Div(
            [
                dbc.Button(
                    "Show Project Overview",
                    id="collapse-button",
                    className="mb-3",
                    color="info",
                ),
                dbc.Collapse(
                    dbc.Card(dbc.CardBody(dcc.Markdown(children=top_markdown_text))),
                    id="collapse",
                ),
            ]),
        html.Hr(),
        
        # DROPDOWN w STATES
        html.Div([
            html.Span("Select a state to filter results: ", className="six columns",
                style={"text-align": "right", "width": "40%", "padding-top": 5}),
            dcc.Dropdown(id="value-selected", value='',
                options=state_options_enrollment,
                style={"display": "block", "margin-left": "auto", "margin-right": "auto","width": "50%"},
                )],style={"display": "block","width":"100%"}),
                
        # BACKGROUND 1:
        dcc.Markdown(children=markdown_text_background_1),
        dbc.Card(
            [
                dbc.CardHeader(
                    dbc.Tabs(
                        [
                            dbc.Tab(label="Cumulative Admissions", tab_id="CumulativeAdmissions",label_style={"font-size": "12px"}),
                            dbc.Tab(label="Enrollment over Time", tab_id="RateAdmissions",label_style={"font-size": "12px"}),
                            dbc.Tab(label="Visualize Relationships", tab_id="VizRelationships",label_style={"font-size": "12px"}),
                            dbc.Tab(label="Cumulative Revenue and Expenses", tab_id="CumulativeFin",label_style={"font-size": "12px"}),
                            dbc.Tab(label="Tuition & Fees Trends", tab_id="TuitionTrends",label_style={"font-size": "12px"}),
                        ],
                        id="tabs", card=True,# style=tabs_styles,
                        active_tab="CumulativeAdmissions",
                        )
                    ),
                dbc.CardBody(html.Div(id='tab-content')),
            ]
        ),
        html.Hr(),

        # SUMMARIZE FEATURES USED FOR PREDICTION:
        html.Div(style={"padding-top": 10}),
        dcc.Markdown(children="""## Summarize Features: Raw and Transformed"""),
        dbc.Card(
            [
                dbc.CardHeader(
                    dbc.Tabs(
                        [
                            dbc.Tab(label="List of features/predictors", tab_id="FeatList"),
                            dbc.Tab(label="Summary statistics", tab_id="SumStats"),
                        ],
                        id="tabs-data", card=True,
                        active_tab="FeatList",
                        )
                    ),
                dbc.CardBody(html.Div(id='tab-content-data')),
            ]
        ),
        html.Hr(),

        # ANALYSIS SECION 1:
        # html.Div(style={"padding-top": 10}),
        # dcc.Markdown(children="""## Analysis Part 1: Compare Linear Modelsn_text_background_2"""),
        # dbc.Card(
        #     [
        #         dbc.CardHeader(
        #             dbc.Tabs(
        #                 [
        #                     dbc.Tab(label="VisibleLabelHere", tab_id="tab_1",label_style={"font-size": "12px"}),
        #                     dbc.Tab(label="VisibleLabelHere", tab_id="tab_1",label_style={"font-size": "12px"}),
        #                 ],
        #                 id="tabs-analysis", card=True,
        #                 active_tab="tab_1",
        #                 )
        #             ),
        #         dbc.CardBody(html.Div(id='tab-content-models-1')),
        #     ]
        # ),
        # html.Hr(),
        # 
        # # ANALYSIS SECION 2:
        # html.Div(style={"padding-top": 10}),
        # dcc.Markdown(children="""
        #     ## Analysis Part 2: Other Classical Time Series Forecasting Methods
        #     - ARIMA: Autoregressive Integrated Moving Average
        #     ---> Use 60% of data to find parameters of the model AR(p), I(d), and MA(q)
        #     - SARIMAX: Seasona
        #     """),
        # # dbc.Tabs(
        # #     [
        # #         dbc.Tab(label="Graph2", tab_id="model_res_2"),
        # #         dbc.Tab(label="MoreTable Results", tab_id="model_tab_2"),
        # #     ],
        # #     id="tabs-4",
        # #     active_tab="model_res_2",
        # # ),
        # # html.Div(id="tab-content-4"),
        # html.Hr(),
        # 
        # # Analysis Part 3:
        # dcc.Markdown(children="""#### Analysis Part 3: Other Classical Time Series Forecasting Methods"""),

    ]
)

# ----------------------
# SECTION 1 Callback
# ----------------------
@app.callback(
    Output("tab-content", "children"),
    [Input("tabs", "active_tab"), Input("value-selected", "value")],
)
def render_tab_content(active_tab, selected):
    df = df_enroll
    # Set xlabel based on selection (useful for debuggins)
    xlab=[opt['label'] for opt in state_options_enrollment if opt['value']==selected]
    
    # Render the tab content based on value of 'active_tab' (other input, 'selected' used to crossfilter)
    if active_tab == "CumulativeAdmissions":
        which_columns = ['number_applied','number_admitted','number_enrolled_total']
        graph_title = 'Enrollment in College Fails to Keep Pace with Admissions'+": "+xlab[0]
        ylabel = 'Total Number of Students'        
        markdown_comments = """METHOD: Use each school's own data to model enrollment using data from 2001-16 and evaluate error in predicting 2017 enrollment."""
        # Data for figure:
        if len(selected)==0:
            df_fig = df.groupby(['year'])[which_columns].aggregate([('Sum','sum'), ('Nschools','count')])
        else:
            df_fig = df[df.state_name==selected].groupby(['year'])[which_columns].aggregate([('Sum','sum'), ('Nschools','count')])
        fig = go.Figure()
        for col in which_columns:
            fig.add_trace(
                go.Scatter(x=df_fig.index, y=df_fig[col]['Sum'], 
                    name = seriez[col]['label'], marker_color='rgba(152, 0, 0, .8)',
                    line = dict(color = seriez[col]['color']), opacity = 0.8))
        # Raw Data for table:
        df_tab = df_fig.xs(key='Sum', axis=1, level=1)  
        df_tab = tbl_early_late(df_tab).rename_axis('').reset_index()
        # Set options common to all traces with fig.update_traces
        fig.update_traces(mode='lines+markers', marker_line_width=2, marker_size=10)
        fig.update_layout(xaxis={'title': xlab[0]}, yaxis={'title': ylabel},
            title={'text': graph_title, 'font':dict(size=16), 'yref': 'paper', 'y': 1, 'yanchor': 'bottom', 'pad':dict(l=20, r=0, t=0, b=5)}, 
            width=800, height=450,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(yanchor="top",y=0.95,xanchor="left",x=0.05), legend_title_text='',
            yaxis_zeroline=True, xaxis_zeroline=True)
        
    elif active_tab == "RateAdmissions":
        which_columns = ['admission_rate','enrollement_rate']
        graph_title = 'Admission and Enrollment Rates over time'+": "+xlab[0]
        ylabel = 'Rate'
        markdown_comments = """This tab will likely be eliminated in favor of the version with trendlines."""
        # Data for figure:
        if len(selected)==0:
            df_fig = df.groupby(['year'])[which_columns].aggregate([('Avg',np.mean), ('stdev',np.std), ('Nschools','count'), ('SEM', sem_btwn)])
        else:
            df_fig = df[df.state_name==selected].groupby(['year'])[which_columns].aggregate([('Avg',np.mean), ('stdev',np.std), ('Nschools','count'), ('SEM', sem_btwn)])
        fig = go.Figure()
        for col in which_columns:
            ys=df_fig[col]['Avg']
            xs=np.array(range(0,len(ys)))
            mod = np.polyfit(xs[:-1],ys[:-1],1)
            predict=np.poly1d(mod)
            r2 = round(r2_score(ys[:-1], predict(xs[:-1])),2)

            # Add line graph w/ data except last one:
            fig.add_trace(go.Scatter(x=df_fig.index[:-1], y=ys[:-1], name = seriez[col]['label'], 
                    line_color = seriez[col]['color'], marker_color=seriez[col]['color'],
                    error_y = dict(type='data', array=df_fig[col]['SEM'], visible=True),
                    mode='lines+markers', opacity = 0.9))
            # Add best fit line
            fig.add_trace(go.Scatter(x=df_fig.index, y=predict(xs), mode='lines', name='Slope = '+str(round(mod[0],3))+' (R^2 '+str(r2)+')',
                              line_color = seriez[col]['color'], line_dash='dash', opacity = 0.8))
            # Add PREDICTED last data point w/ different marker
            fig.add_trace(go.Scatter(x=pd.Series(df_fig.index[-1]), y=pd.Series(predict(xs[-1])), showlegend=False, marker_symbol="x", marker_line_color='white', mode='markers', marker_color=seriez[col]['color']))
            # Add TRUE last data point w/ different marker
            fig.add_trace(go.Scatter(x=pd.Series(df_fig.index[-1]), y=pd.Series(ys[-1]), showlegend=False, mode='markers', marker_color=seriez[col]['color']))
                              
        # Raw Data for table:
        df_tab = df_fig.xs(key='Avg', axis=1, level=1)        
        df_tab = tbl_early_late(df_tab).rename_axis('').reset_index()
        # Set options common to all traces with fig.update_traces
        fig.update_traces(marker_line_width=2, marker_size=10)
        fig.update_layout(xaxis={'title': xlab[0]}, yaxis={'title': ylabel, 'range': [0.08,0.95]},
            title={'text': graph_title, 'font':dict(size=16), 'yref': 'paper', 'y': 1, 'yanchor': 'bottom', 'pad':dict(l=20, r=0, t=0, b=5)}, 
            margin=dict(l=20, r=20, t=20, b=20),
                    legend=dict(font = dict(size=11, color='black')),
                    legend_title_text='',
                    width=1000, height=450,
            yaxis_zeroline=True, xaxis_zeroline=True)
        # fig.update_layout(title=graph_title, xaxis={'title': xlab[0]}, 
        #             yaxis={'title': ylabel, 'range': [0.08,0.95]},
        #             yaxis_zeroline=True, xaxis_zeroline=True)  
        # fig.show()
    elif active_tab == "VizRelationships":
        which_columns = ['admission_rate','enrollement_rate']
        xlab[0] = 'Enrollement Rate'
        ylabel = 'Admission Rate'
        graph_title = 'Admission and Enrollment for Each School, All Years'
        df_enroll['unitid'] = df_enroll['unitid'].astype('object')
        fig = px.scatter(df_enroll, x='enrollement_rate', y='admission_rate', color='state_name', hover_data=['unitid', 'year'], labels={'state_name': ' '})
        fig.update_traces(marker_line_width=1, marker_size=8, marker_line_color='white')
        fig.update_layout(title=graph_title, xaxis={'title': xlab[0]}, yaxis={'title': ylabel},
                    legend=dict(font = dict(size=8, color='black')),
                    width=800, height=400)  
        # Raw Data for table:
        df_fig = df_enroll.groupby(['year'])['number_admitted','number_enrolled_total'].aggregate([('Avg',np.mean), ('stdev',np.std), ('Nschools','count'), ('SEM', sem_btwn)])
        df_tab = df_fig.xs(key='Avg', axis=1, level=1)        
        df_tab = tbl_early_late(df_tab).rename_axis('').reset_index()
        markdown_comments="""Comments here"""
    # ---- $$$ CUMULATIVE REVENUE & EXPENSES -----
    elif active_tab == "CumulativeFin":
        which_columns = ['rev_total_current','exp_total_current','rev_tuition_fees_gross','rev_tuition_fees_net','exp_instruc_total']
        xlab=[opt['label'] for opt in state_options_finance if opt['value']==selected]
        graph_title = 'Net Revenue and Expenses over Time'+": "+xlab[0]
        ylabel = '$$$'        
        markdown_comments = """Trend towards more profitable institutions?"""
        if len(selected)==0:
            df_fig = dfFin1.groupby(['year'])[which_columns].aggregate([('Sum','sum'), ('Nschools','count')])
        else:
            df_fig = dfFin1[dfFin1.state_name==selected].groupby(['year'])[which_columns].aggregate([('Sum','sum'), ('Nschools','count')])
        fig = go.Figure()
        for col in which_columns:
            fig.add_trace(
                go.Scatter(x=df_fig.index, y=df_fig[col]['Sum'], name = seriez[col]['label'], 
                           marker_color=seriez[col]['marker_color'], line = dict(color = seriez[col]['color']), opacity = 0.8))
        # FIGURE OUT WHAT IS APT FOR THIS TABLE
        df_tab = df_fig.xs(key='Sum', axis=1, level=1)        
        df_tab = tbl_early_late(df_tab).rename_axis('').reset_index()
        # Set options common to all traces with fig.update_traces
        fig.update_traces(mode='lines+markers', marker_line_width=2, marker_size=10)
        fig.update_layout(xaxis={'title': xlab[0]}, yaxis={'title': ylabel},
            title={'text': graph_title, 'font':dict(size=16), 'yref': 'paper', 'y': 1, 'yanchor': 'bottom', 'pad':dict(l=20, r=0, t=0, b=5)}, 
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.01), legend_title_text='',
            yaxis_zeroline=True, xaxis_zeroline=True)
                            
    # ---- $$$ TUITION TRENDS -----
    elif active_tab == "TuitionTrends":
        which_columns = ['rev_tuition_fees_gross','rev_tuition_fees_net']
        xlab=[opt['label'] for opt in state_options_finance if opt['value']==selected]
        graph_title = 'Deductions in Revenue from Tuition & Fees'+": "+xlab[0]
        ylabel = '$$$'
        markdown_comments = """I suspect this trend of growing deductions in tuition & fees stems from **INCREASE in financial aid packages**. Research TODO."""
        # fig, df_tab = make_figure_section_1(dfFin1, which_columns, selected, 'Sum')        
        if len(selected)==0:
            df_fig = dfFin1.groupby(['year'])[which_columns].aggregate([('Avg','mean'), ('Nschools','count')])
        else:
            df_fig = dfFin1[dfFin1.state_name==selected].groupby(['year'])[which_columns].aggregate([('Avg','mean'), ('Nschools','count')])
        fig = go.Figure()
        for col in which_columns:
            fig.add_trace(
                go.Scatter(x=df_fig.index, y=df_fig[col]['Avg'], name = seriez[col]['label'], 
                           marker_color=seriez[col]['marker_color'], line = dict(color = seriez[col]['color']), opacity = 0.8))
        # FIGURE OUT WHAT IS APT FOR THIS TABLE
        df_tab = df_fig.xs(key='Avg', axis=1, level=1)        
        df_tab = tbl_early_late(df_tab).rename_axis('').reset_index()
        # Set options common to all traces with fig.update_traces
        fig.update_traces(mode='lines+markers', marker_line_width=2, marker_size=10)
        fig.update_layout(xaxis={'title': xlab[0]}, yaxis={'title': ylabel},
                    title={'text': graph_title, 'font':dict(size=16), 'yref': 'paper', 'y': 1, 'yanchor': 'bottom', 'pad':dict(l=20, r=0, t=0, b=5)}, 
                    margin=dict(l=20, r=20, t=20, b=20),
                    legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.01),
                    # width=800, height=400,
                    yaxis_zeroline=True, xaxis_zeroline=True)

    return html.Div([
        dcc.Graph(figure = fig),
        dash_table.DataTable(
            id='table',
            style_data={
                'whiteSpace': 'normal',
                'height': 'auto'},
            style_table={
                'maxWidth': '400px'},
            style_header={ 'border': '2px solid grey', 'fontWeight': 'bold', 'backgroundColor':'rgb(230, 230, 230)','textAlign': 'center' },
            style_cell={ 'border': '1px solid grey' , 'textAlign': 'left', 'font_family': 'Sans-serif', 'font_size': '14px'},
            style_as_list_view=True,
            # style_data_conditional=[{'if': {'column_id': 'DIFF','filter_query': '{DIFF} lt 0'},
            #     'backgroundColor': '#99423d','color': 'white',}],
            columns=[{"name": i, "id": i} for i in df_tab.columns],
            data=df_tab.to_dict("rows")),
        dcc.Markdown(children=markdown_comments),
        ])
        

# ----------------------
# Section 2 Callback
# ----------------------
# @app.callback(
#     Output("tab-content-2", "children"),
#     [Input("tabs-2", "active_tab"), Input("value-selected", "value")],
# )
# def render_tab2_content(active_tab, selected):

# ----------------------
# Summarize Features callback
# ----------------------
@app.callback(
    Output("tab-content-data", "children"),
    [Input("tabs-data", "active_tab")],
)
def update_table(active_tab):
    # get df_tab
    if active_tab=="FeatList":
        df_tab = pd.DataFrame(seriez.keys(),columns=['Features'])
        # print('')
    elif active_tab=="SumStats":
        df_tab=summary_transformed
    
    return html.Div([
        # dbc.Table.from_dataframe(df_tab, striped=True, bordered=True, hover=True),
        dash_table.DataTable(
            id='table',
            style_data={
                'whiteSpace': 'normal',
                'height': 'auto'},
            style_table={
                'maxWidth': '400px'},
            style_header={ 'border': '2px solid grey', 'fontWeight': 'bold', 'backgroundColor':'rgb(230, 230, 230)','textAlign': 'center' },
            style_cell={ 'border': '1px solid grey' , 'textAlign': 'left', 'font_family': 'Sans-serif', 'font_size': '11px'},
            style_as_list_view=True,
            # style_data_conditional=[{'if': {'column_id': 'DIFF','filter_query': '{DIFF} lt 0'},
            #     'backgroundColor': '#99423d','color': 'white',}],
            columns=[{"name": i, "id": i} for i in df_tab.columns],
            data=df_tab.to_dict("rows")),        
        dcc.Markdown(children="""Also including the breakdowns above as a proportion of the year's total revenue and expenses."""),
        ])

@app.callback(
    Output("collapse", "is_open"),
    [Input("collapse-button", "n_clicks")],
    [State("collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


if __name__ == '__main__':
	app.run_server(debug=True)
