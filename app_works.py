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
        n=df.groupby(['state_name'])['number_applied'].count().loc[state]
        state_options.append({'label': state+" ("+str(n)+" schools)", 'value': state })
    return state_options

# --------------------------------
# Data Prune:
# --------------------------------
# Drop years earlier than 2002 (availability of key measures)
df = df[df.year>=2001];

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


# GOAL: Want a Balanced dataset w XYZ schools over 17 yrs 

# ENROLLMENT SECTION
nYrs = 17
which_columns = ['enrollement_rate']
tmpDf = df.groupby('unitid')[which_columns].count()
# Identify schools with data in all years, only include those:
unitids = tmpDf[tmpDf[which_columns[0]]==nYrs].index.to_list()
filt = df.apply(lambda row: row['unitid'] in unitids, axis=1)
# Make a DF & list of options for the dropdown menu - ENROLLMENT SECTION
df = df[filt]  
state_options_enrollment = make_options(df)

# FINANCE SECTION
nYrs= 17
which_columns = ['rev_total_current','exp_total_current','rev_tuition_fees_gross','rev_tuition_fees_net','exp_instruc_total']
tmpDf = df.groupby(['unitid'])[which_columns].count()
# Identify schools with data in all years, only include those:
unitids = tmpDf[tmpDf[which_columns[0]]==nYrs].index.to_list()
filt = df.apply(lambda row: row['unitid'] in unitids, axis=1)
dfFin1 = df[filt]
state_options_finance = make_options(dfFin1)

# DEFINE COLORS & LABELS FOR CONSISTENTLY GRAPHING SERIES
seriez = {'number_applied': {'color': '#F44DDB', 'label': "Number of student applications"},
            'number_admitted': {'color': '#CF1214', 'label': "Number of students admitted"},
            'number_enrolled_total': {'color': '#0E3DEC', 'label': "Number enrolled"},
            'admission_rate': {'color': '#CF1214', 'label': "Admission rate (# applications/# admissions)"},
            'enrollement_rate': {'color': '#0E3DEC', 'label': "Enrollment rate (# admissions/# enrolled"},
        }
                            

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from dash.dependencies import Input, Output

# Prune as necessary:
import chart_studio.plotly as py
import plotly.graph_objects as go
import plotly
import cufflinks as cf

top_markdown_text = '''
**v1.2**
This project models enrollment trends at public universities (2001-17).
*TODO: INSERT PROJECT DESCRIPTION HERE.*
'''

markdown_text_background_1="""

### Background 1: 
**What temporal patterns do you notice in the data below?**
"""
markdown_text_background_2= """
### Background 2: 
**And, again, what temporal patterns do you notice in the data below?**
"""

last_updated_text = "Insert project approach here"

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
        dcc.Markdown(children="""### Project Proposal Goes here!"""),
        html.Hr(),

        # Analysis Part 1:
        dcc.Markdown(children="""#### Analysis Part 1: Compare Linear Models"""),
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
        dcc.Markdown(children="""#### Analysis Part 2: Predictive Forecasting"""),
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

    ]
)

# ----------------------
# Background 1 Callback
# ----------------------
@app.callback(
    Output("tab-content", "children"),
    [Input("tabs", "active_tab"), Input("value-selected", "value")],
)
def render_tab_content(active_tab, selected):
    # Set xlabel based on selection (useful for debuggins)
    xlab=[opt['label'] for opt in state_options_enrollment if opt['value']==selected]
    
    # Render the tab content based on value of 'active_tab' (other input, 'selected' used to crossfilter)
    if active_tab == "CumulativeAdmissions":
        which_columns = ['number_applied','number_admitted','number_enrolled_total']
        graph_title = 'Enrollment in College Fails to Keep Pace with Admissions'+": "+xlab[0]
        ylabel = 'Total Number of Students'        
        markdown_comments = """Insert comments A, B, C"""
        if len(selected)==0:
            df_fig = df.groupby(['year'])[which_columns].sum()
        else:
            df_fig = df[df.state_name==selected].groupby(['year'])[which_columns].sum()
    elif active_tab == "RateAdmissions":
        which_columns = ['admission_rate','enrollement_rate']
        graph_title = 'Admission and Enrollment Rates over time'+": "+xlab[0]
        ylabel = 'Rate'
        markdown_comments = """Insert comments D, E"""
        if len(selected)==0:
            df_fig = df.groupby(['year'])[which_columns].apply(np.mean)
        else:
            df_fig = df[df.state_name==selected].groupby(['year'])[which_columns].apply(np.mean)
            # error_barz = df[df.state_name==selected].groupby(['year'])[which_columns].apply(np.std)
        # error_y=dict(
        #     type='data',
        #     symmetric=False,
        #     array=[0.1, 0.2, 0.1, 0.1],
        #     arrayminus=[0.2, 0.4, 1, 0.2])
            
    fig = go.Figure()
    for s_id in which_columns:
        fig.add_trace(
            go.Scatter(x=df_fig.index, y=df_fig[s_id], 
                name = seriez[s_id]['label'], marker_color='rgba(152, 0, 0, .8)',
                line = dict(color = seriez[s_id]['color']), opacity = 0.8))
    # Set options common to all traces with fig.update_traces
    fig.update_traces(mode='lines+markers', marker_line_width=2, marker_size=10)
    fig.update_layout(title=graph_title,
                  yaxis_zeroline=True, xaxis_zeroline=True)
        # fig.show()

    return html.Div([
        dcc.Graph(figure = fig
            # df_fig.iplot(kind='scatter',width=4,title=graph_title,xTitle=xlab[0],yTitle=ylabel,theme='white',asFigure=True)
            ),
        dcc.Markdown(children=markdown_comments),
        ])

# ----------------------
# Background 2 Callback
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
        markdown_fin = """Insert comments 1, 2, 3"""
        if len(selected)==0:
            tmp = dfFin1.groupby(['year'])[which_columns].sum()
        else:
            tmp = dfFin1[dfFin1.state_name==selected].groupby(['year'])[which_columns].sum()
    elif active_tab == "TuitionTrends":
        which_columns = ['rev_tuition_fees_gross','rev_tuition_fees_net']
        graph_title = 'Deductions in Revenue from Tuition & Fees'+": "+xlab[0]
        ylabel = '$$$'
        markdown_fin = """Insert comments 4, 5"""
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