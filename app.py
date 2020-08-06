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

# --------------------------------
# GOAL: Want a Balanced dataset w XYZ schools over 17 yrs 
# --------------------------------

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
import dash_table
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# Prune as necessary:
import chart_studio.plotly as py
import plotly.graph_objects as go
import plotly.express as px
import plotly

top_markdown_text = '''
**v1.2**
This project models enrollment trends at public universities (2001-17).

- SET THE SCENE: 
Here we are in the summer of 2020, with a country full of would-be college freshmen heading into a fall semester with a LOT of uncertainty.
Imagine being a student (or their parents) deciding whether Tuition (and Fees!) will be "worth it" at this point in time. 

- LET'S BACK UP A MOMENT:
Remember way back at the beginning of the year, when journalism about  
The context: Outrage over admissions cheating scandals. Grumpiness about universities becoming more selective over time. 

- SET THE STORY STRAIGHT:  
Pew research report from 2019 headline "Majority of US colleges admit most of their applicants."

- TO HIGHLIGHT THE IMPORTANCE OF THIS QUESTION:
**Public colleges and universities educate nearly 75% of all college students.**
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
        markdown_comments = """Insert comments D, E"""
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
        markdown_comments = """Insert comments F, G, H"""
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
        trend_ER.rename(columns={'y': 'enrollment_rate'},inplace=True)
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
