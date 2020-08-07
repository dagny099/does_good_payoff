import time
import pandas as pd
import numpy as np

# Import saved csv to dataframe:
path2file="./data/interim/analyzeMe_n175.csv"
df = pd.read_csv(path2file, na_values=np.nan,#parse_dates=['year'], 
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

# --------------------------------
# Data Prune:
# --------------------------------
# Set 'year' as datetime 
df['year'] = df['year'].apply(pd.to_datetime, format='%Y')

# Keep these columns (subset from CSV) as potential features for model:
keepcols = ['admission_rate','enrollement_rate','number_applied','number_admitted','number_enrolled_total',
        'rev_tuition_fees_gross', 'rev_tuition_fees_net','rev_total_current','rev_fed_approps_grants','rev_state_local_approps_grants','rev_other',
       'exp_total_current','exp_instruc_total','exp_acad_supp_total','exp_student_serv_total','exp_res_pub_serv_total',
        'completers_150pct','completion_rate_150pct','female_pct','married_pct',
      'year','unitid','inst_name','state_name']

dropcols = [c for c in df.columns if c not in keepcols]

df.drop(dropcols,axis=1,inplace=True)

# DATA FOR ENROLLMENT SECTION
df_enroll  = get_school_data(df, ['enrollement_rate'], 2001)
state_options_enrollment = make_options(df_enroll)

# DATA FOR FINANCE SECTION
# which_columns = ['rev_total_current','exp_total_current','rev_tuition_fees_gross','rev_tuition_fees_net','exp_instruc_total']
dfFin1 = get_school_data(df, ['rev_total_current'])
state_options_finance = make_options(dfFin1)

# --------------------------------
# Set figure aesthetics and labels:
# --------------------------------
seriez = {'number_applied': {'color': '#F44DDB', 'label': "Number of student applications"},
            'number_admitted': {'color': '#CF1214', 'label': "Number of students admitted"},
            'number_enrolled_total': {'color': '#0E3DEC', 'label': "Number enrolled"},
            'admission_rate': {'color': '#CF1214', 'label': "Admission rate (# applications/# admissions)"},
            'enrollement_rate': {'color': '#0E3DEC', 'label': "Enrollment rate (# admissions/# enrolled"},
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
'''

markdown_text_background_1="""

### 1st:  Visualize Admissions time series: 
**The goal is to model the blue line (number of students enrolled) as a time series.**
"""
markdown_text_background_2= """


### 2nd:  Visualize Finance time series: 
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
# Import dash and visualization modules:
# --------------------------------

import dash
import dash_table
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# Prune as necessary:
import chart_studio.plotly as py
import plotly.graph_objects as go
import plotly.express as px
import cufflinks as cf
import plotly

# --------------------------------
# DASH APP!
# --------------------------------
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', dbc.themes.BOOTSTRAP]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server = app.server

app.layout = dbc.Container(
    [
        # dcc.Store(id="store"),
        html.Div([html.H1("Data Incubator Project Proposal")],
            style={'textAlign': "center", "padding-bottom": "5"}),
        html.Hr(),
        
        dcc.Markdown(children=top_markdown_text),
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
        dbc.Tabs(
            [
                dbc.Tab(label="Cumulative Admissions", tab_id="CumulativeAdmissions"),
                dbc.Tab(label="Enrollment over Time", tab_id="RateAdmissions"),
                dbc.Tab(label="Model Enrollment", tab_id="PlotlyExpress"),
            ],
            id="tabs",
            active_tab="CumulativeAdmissions",
        ),
        html.Div(id="tab-content"),
        html.Hr(),

        # BACKGROUND 2:
        dcc.Markdown(children=markdown_text_background_2),
        dbc.Tabs(
            [
                dbc.Tab(label="Cumulative Revenue and Expenses", tab_id="CumulativeFin"),
                dbc.Tab(label="Tuition & Fees Trends", tab_id="TuitionTrends"),
            ],
            id="tabs-2",
            active_tab="CumulativeFin",
        ),
        html.Div(id="tab-content-2"),
        html.Hr(),

        # PROJECT PROPOSAL & TOC for the present analysis:  
        dcc.Markdown(children=markdown_approach),
        html.Hr(),

        # Analysis Part 1:
        dcc.Markdown(children="""
        #### Summarize Features: Raw and Transformed
        - 
        """),
        html.Hr(),

        # Analysis Part 1:
        dcc.Markdown(children="""
        #### Analysis Part 1: Compare Linear Models
        """),
        # dbc.Tabs(
        #     [
        #         dbc.Tab(label="Graph1", tab_id="model_res_1"),
        #         dbc.Tab(label="Table Results", tab_id="model_tab_1"),
        #     ],
        #     id="tabs-3",
        #     active_tab="model_res_1",
        # ),
        # html.Div(id="tab-content-3"),
        html.Hr(),

        # Analysis Part 2:
        dcc.Markdown(children="""
            #### Analysis Part 2: Other Classical Time Series Forecasting Methods
            - ARIMA: Autoregressive Integrated Moving Average
            ---> Use 60% of data to find parameters of the model AR(p), I(d), and MA(q)
            - SARIMAX: Seasona
            """),
        # dbc.Tabs(
        #     [
        #         dbc.Tab(label="Graph2", tab_id="model_res_2"),
        #         dbc.Tab(label="MoreTable Results", tab_id="model_tab_2"),
        #     ],
        #     id="tabs-4",
        #     active_tab="model_res_2",
        # ),
        # html.Div(id="tab-content-4"),
        html.Hr(),

        # Analysis Part 3:
        dcc.Markdown(children="""#### Analysis Part 3: Other Classical Time Series Forecasting Methods"""),

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
        fig.update_layout(title=graph_title, xaxis={'title': xlab[0]}, yaxis={'title': ylabel},
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
            fig.add_trace(
                go.Scatter(x=df_fig.index, y=df_fig[col]['Avg'], #trendline='ols'
                name = seriez[col]['label'], marker_color='rgba(152, 0, 0, .8)',
                line = dict(color = seriez[col]['color']), 
                error_y = dict(type='data', array=df_fig[col]['SEM'], visible=True),
                opacity = 0.8))
        # Raw Data for table:
        df_tab = df_fig.xs(key='Avg', axis=1, level=1)        
        df_tab = tbl_early_late(df_tab).rename_axis('').reset_index()
        # Set options common to all traces with fig.update_traces
        fig.update_traces(mode='lines+markers', marker_line_width=2, marker_size=10)
        fig.update_layout(title=graph_title, xaxis={'title': xlab[0]}, yaxis={'title': ylabel},
                  yaxis_zeroline=True, xaxis_zeroline=True)
    elif active_tab == "PlotlyExpress":
        which_columns = ['admission_rate','enrollement_rate']
        graph_title = 'Predict 2017 Enrollment using 2001-16'
        ylabel = 'Rate'
        markdown_comments = """NEXT STEPS FOR THIS GRAPH:  
        (1) Disconnect the line between '16-17. 
        (2) Add a checkbox to enable predictions using different features. 
        (3) Dynamically plot the model predictions for 2017 based on which feature(s) or model chosen."""
        # Data for figure:
        if len(selected)==0:
            df_fig = df.groupby(['year'])[which_columns].aggregate([('Avg',np.mean), ('stdev',np.std), ('Nschools','count'), ('SEM', sem_btwn)])
        else:
            df_fig = df[df.state_name==selected].groupby(['year'])[which_columns].aggregate([('Avg',np.mean), ('stdev',np.std), ('Nschools','count'), ('SEM', sem_btwn)])
        fig = px.scatter(df_fig, x=df_fig.index, y=df_fig['admission_rate',]['Avg'], trendline="ols")
        fig.update_layout(title= dict(text=graph_title)) # , font=dict(size=16)
        fig.update_traces(marker=dict(color=seriez['admission_rate']['color']), line=dict(color=seriez['admission_rate']['color'], width=4, dash='dot'))
        res_tmp = px.get_trendline_results(fig)
        res_tmp = res_tmp.px_fit_results.iloc[0].summary().as_html()
        trend_AR = pd.read_html(res_tmp, header=0, index_col=0)[0]
        trend_AR.drop(['Date:','Time:'], axis=0, inplace=True)
        trend_AR.rename(columns={'y': 'admission_rate'},inplace=True)
        fig2 = px.scatter(df_fig, x=df_fig.index, y=df_fig['enrollement_rate']['Avg'], trendline="ols")
        fig2.update_traces(marker=dict(color=seriez['enrollement_rate']['color']), line=dict(color=seriez['enrollement_rate']['color'], width=4, dash='dot'))
        res_tmp = px.get_trendline_results(fig2)
        res_tmp = res_tmp.px_fit_results.iloc[0].summary().as_html()
        trend_ER = pd.read_html(res_tmp, header=0, index_col=0)[0]
        trend_ER.drop(['Date:','Time:'], axis=0, inplace=True)
        trend_ER.rename(columns={'y': 'enrollment_rate', 'R-squared:':'R-squared'},inplace=True)
        fig.add_trace(fig2.data[0])
        fig.add_trace(fig2.data[1])
        fig.update_traces(marker_line_width=2, marker_size=10)
        fig.update_layout(title=graph_title,yaxis={'title': ylabel},xaxis={'title': xlab[0]}, 
                  yaxis_zeroline=True, xaxis_zeroline=True)
        df_tab = pd.concat([trend_AR, trend_ER], axis=1)

    # fig.show()
    
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
@app.callback(
    Output("tab-content-2", "children"),
    [Input("tabs-2", "active_tab"), Input("value-selected", "value")],
)
def render_tab2_content(active_tab, selected):
    # Set xlabel based on selection (useful for debuggins)
    xlab=[opt['label'] for opt in state_options_finance if opt['value']==selected]

    # Render the tab content based on value of 'active_tab' (other input, 'selected' used to crossfilter)
    if active_tab == "CumulativeFin":
        which_columns = ['rev_total_current','exp_total_current','rev_tuition_fees_gross','rev_tuition_fees_net','exp_instruc_total']
        graph_title = 'Net Revenue and Expenses over Time'+": "+xlab[0]
        ylabel = '$$$'        
        markdown_fin = """Trend towards more profitable institutions?"""
        if len(selected)==0:
            tmp = dfFin1.groupby(['year'])[which_columns].sum()
        else:
            tmp = dfFin1[dfFin1.state_name==selected].groupby(['year'])[which_columns].sum()
    elif active_tab == "TuitionTrends":
        which_columns = ['rev_tuition_fees_gross','rev_tuition_fees_net']
        graph_title = 'Deductions in Revenue from Tuition & Fees'+": "+xlab[0]
        ylabel = '$$$'
        markdown_fin = """I suspect this trend of growing deductions in tuition & fees stems from **INCREASE in financial aid packages**. Research TODO."""
        if len(selected)==0:
            tmp = dfFin1.groupby(['year'])[which_columns].apply(np.mean)
        else:
            tmp = dfFin1[df.state_name==selected].groupby(['year'])[which_columns].apply(np.mean)
    return html.Div([
        dcc.Graph(figure = 
            tmp.iplot(kind='scatter',width=4,title=graph_title,xTitle=xlab[0],yTitle=ylabel,theme='white',asFigure=True)
            ),
        dcc.Markdown(children=markdown_fin),
        ])


if __name__ == '__main__':
	app.run_server(debug=True)
