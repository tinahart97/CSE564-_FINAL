import dash_bootstrap_components as dbc
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output
import numpy as np
import plotly.graph_objects as go
import copy

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

from urllib.request import urlopen
import json

with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

import plotly.express as px


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# DATASETS
police_shootings = pd.read_excel(r"C:\Users\tinah\Desktop\FINAL PROJ\mergedPoliceShooting.xlsx",
                               dtype={"fips_txt": str})

county_stuff = pd.read_excel(r"C:\Users\tinah\Desktop\FINAL PROJ\County.xlsx",
                           dtype={"fips_txt": str}, )
population = pd.read_csv(r"C:\Users\tinah\Desktop\FINAL PROJ\Population.csv",
                         dtype={"fips_txt": str})

county_df = pd.read_excel(r"C:\Users\tinah\Desktop\FINAL PROJ\Region info by county.xlsx",
                        dtype={"FIPS": str})

police_shooting = pd.read_excel(r"C:\Users\tinah\Desktop\FINAL PROJ\LargeDataset.xlsx",
                              dtype={"fips_txt": str})

df1 = pd.read_excel(r"C:\Users\tinah\Desktop\FINAL PROJ\df.xlsx",
                  dtype={"fips_txt": str})


county_df["Percent Poverty"]= county_df["POVALL_2019"]/ county_df["pop"]



# Group these into the other/unknown category
county_stuff['two_pop'] = county_stuff['two_pop'] + county_stuff['pacific_pop_x']

# Reformat polulation numbers into percentages
county_stuff.rename(columns={'white_pop_x': 'White Population',
                             'black_pop_x': 'Black Population',
                             'asian_pop_x': 'Asian Population',
                             'indian_pop_x': 'Native American Population',
                             'hisp_pop_x': 'Hispanic Population',
                             'two_pop': 'Other/Unknown'}, inplace=True)
county_stuff['Population Adjusted'] = county_stuff['White Population'] + county_stuff['Black Population'] + \
                                      county_stuff['Asian Population'] + county_stuff['Native American Population'] + \
                                      county_stuff['Hispanic Population'] + county_stuff['Other/Unknown']
# divide each column by adjusted population total
for c in ['White Population', 'Black Population', 'Asian Population', 'Native American Population',
          'Hispanic Population', 'Other/Unknown']:
    county_stuff[c] = np.round(county_stuff[c] / county_stuff['Population Adjusted'] * 100, 1)
police_shootings = pd.merge(police_shootings, county_stuff.loc[:, ['Geographic Area', 'state', 'county_name']],
                            on=['Geographic Area', 'county_name'], how='inner')










## PIE CHART PREP
# aggregate the county info to get state info
county_df['two_pop'] = county_df['two_pop'] + county_df['pacific_pop']
# Rename columns
county_df.rename(columns={'white_pop': 'White Population',
                          'black_pop': 'Black Population',
                          'asian_pop': 'Asian Population',
                          'indian_pop': 'Native American Population',
                          'hisp_pop': 'Hispanic Population',
                          'two_pop': 'Other/Unknown',
                          'state_abbrev': 'Geographic Area'}, inplace=True)
county_df['Population Adjusted'] = county_df['White Population'] + county_df['Black Population'] + \
                                   county_df['Asian Population'] + county_df['Native American Population'] + \
                                   county_df['Hispanic Population'] + county_df['Other/Unknown']
sumdf = pd.DataFrame(county_df.groupby('Geographic Area').sum())
sumdf.reset_index(inplace=True)


sumdfC = pd.DataFrame(county_df.groupby('FIPS').sum())
sumdfC.reset_index(inplace=True)



county_stuff['Perentagedeathsperthousand'] = round(county_stuff['Deaths'] / (county_stuff['pop_x']) * 100000, 4)
new = county_stuff[['fips_txt', 'Deaths', 'pop_x', 'Perentagedeathsperthousand']]

# PC Plot data


cleanup = {"year_x": {"2015": 0, "2016": 1, "2017": 2, "2018": 3, "2019": 4, "2020": 5, "2021": 6},
           "manner_of_death": {"shot": 0, "shot and Tasered": 1},
           "gender": {"M": 0, "F": 1},
           "race": {"W": 0, "B": 1, "A": 2, "O": 3, "H": 4, "N": 5},
           "signs_of_mental_illness": {"FALSE": 0, "TRUE": 1},
           "threat_level": {"other": 0, "attack": 1, "undetermined": 2},
           "flee": {"Not fleeing": 0, "Foot": 1, "Car": 2, "Other": 3},
           "States": {"AL": 0, "AK": 1, "AR": 2, "AZ": 3, "CA": 4, "CO": 5,
                      "CT": 6, "DE": 7, "FL": 8, "GA": 9, "HI": 10,
                      "IA": 11, "ID": 12, "IL": 13, "IN": 14, "KS": 15,
                      "KY": 16, "LA": 17, "MA": 18, "MD": 19, "ME": 20,
                      "MI": 21, "MN": 22, "MO": 23, "MS": 24, "MT": 25,
                      "NC": 26, "ND": 27, "NE": 28, "NH": 29, "NJ": 30, "NM": 31,
                      "NV": 32, "NY": 33, "OH": 34, "OK": 35, "OR": 36,
                      "PA": 37, "RI": 38, "SC": 39, "SD": 40, "TN": 41,
                      "TX": 42, "UT": 43, "VA": 44, "VT": 45, "WA": 46, "WI": 47,
                      "WV": 48, "WY": 49}}
# df = police_shooting.replace(cleanup)
df = copy.deepcopy(county_stuff)
df.rename(columns={'pop_x': 'Population', 'Perentagedeathsperthousand': 'Deaths per 100,000 Residents',
                   'Unemployment_rate_2019': 'Unemployment Rate (%)',
                   'Median_Household_Income_2019': 'Median Household Income',
                   'Percent of adults with less than a high school diploma, 2015-19': '% of Adults Without HS Diploma'},
          inplace=True)

# Line plot state populations
state_list = list(pd.Series(county_stuff['state'].unique()).sort_values())

# Scatter stuff


df1.groupby('region')['Deaths'].sum()
df1.groupby('region')['Population'].sum()

region = df1['region'].unique()
unemp = round(df1.groupby('region')['Unemployment Rate (%)'].mean(), 2)
medinc = round(df1.groupby('region')['Median Household Income'].mean(), 2)
HSdip = round(df1.groupby('region')['% of Adults Without HS Diploma'].mean(), 2)
popwhite = df1.groupby('region')['white_pop_y'].sum() / df1.groupby('region')['Population'].sum() * 100
popbl = df1.groupby('region')['black_pop_y'].sum() / df1.groupby('region')['Population'].sum() * 100
popas = df1.groupby('region')['asian_pop_y'].sum() / df1.groupby('region')['Population'].sum() * 100
popind = df1.groupby('region')['indian_pop_y'].sum() / df1.groupby('region')['Population'].sum() * 100
pophis = df1.groupby('region')['hisp_pop_y'].sum() / df1.groupby('region')['Population'].sum() * 100
populationpercent = (df1.groupby('region')['Deaths'].sum() / df1.groupby('region')['Population'].sum()) * 1000000

data = {"unemprate": unemp, "HSdip": HSdip, "popwhite": popwhite, "popbl": popbl, "popas": popas, "popind": popind,
        "pophis": pophis, "populationpercent": populationpercent, "medinc": medinc}
df2 = pd.DataFrame(data=data)

Midwestdata = list(df2.iloc[0])
Northeastdata = list(df2.iloc[1])
Southdata = list(df2.iloc[2])
Westdata = list(df2.iloc[3])

data = {"unemprate": unemp, "HSdip": HSdip, "popwhite": popwhite, "popbl": popbl, "popas": popas, "popind": popind,
        "pophis": pophis, "populationpercent": populationpercent, "medinc": medinc}
df2 = pd.DataFrame(data=data)
place = '36103'
selected_region = df1.loc[df1['fips_txt'] == place, 'region'].values[0]
state_df = df1.loc[df1['region'] == selected_region, :]

Midwestdata = list(df2.iloc[0])
Northeastdata = list(df2.iloc[1])
Southdata = list(df2.iloc[2])
Westdata = list(df2.iloc[3])

df2["color"] = [0.4, ] * df2.shape[0]
df2.loc[df2.index == selected_region, 'color'] = 1

### Race percentage bar stuff


police_shootings.replace(
    {'race': {'W': 'White Population', 'B': 'Black Population', 'A': 'Asian Population', 'H': 'Hispanic Population',
              'N': 'Native American Population', 'O': 'Other/Unknown'},
     'gender': {'M': 'Male', 'F': 'Female'}}, inplace=True)

county_df.rename(columns={'FIPS': 'fips_txt'}, inplace=True)
county_df['Deaths per 100,000 Residents'] = round(county_df['Deaths'] / (county_df['pop']) * 100000, 4)

print(max(county_df['Deaths per 100,000 Residents']))

county_df.rename(columns={'pop': 'Population'}, inplace=True)

place = '36103'

selected_state = county_stuff.loc[county_stuff['fips_txt'] == place, 'Geographic Area'].values[0]

county_df['pc_color'] = [0, ] * county_df.shape[0]
county_stuff.loc[county_stuff['Geographic Area'] == selected_state, 'pc_color'] = 1
















# Default Choropleth plot
fig = px.choropleth(county_stuff, geojson=counties, locations='fips_txt', color='Deaths',
                    color_continuous_scale='Plasma',
                    range_color=(0, 50),
                    scope="usa",
                    labels={'Deaths': 'Deaths', 'county_name': 'County'},
                    hover_data=['county_name']
                    )
fig.update_layout(height= 400,margin={"r": 0, "t": 0, "l": 0, "b": 0})
# fig.update_layout(#coloraxis_showscale=False,
# height=300, paper_bgcolor='lightsteelblue', plot_bgcolor='lightsteelblue')





### MAIN LAYOUT
# Bootstrap themes by Ann: https://hellodash.pythonanywhere.com/theme_explorer
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])

app.layout = dbc.Container([

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Select one to change the Bar Charts",
                               style={'font-weight': 'bold', 'text-align': 'center', 'font-size': '16px'}, ),
                dbc.CardBody([
                    dcc.RadioItems(id='radio', options=[
                        {'label': 'County', 'value': 'County'},
                        {'label': 'State', 'value': 'State'},
                        {'label': 'USA', 'value': 'USA'},

                    ], value='County', style={'font-weight': 'bold', 'text-align': 'center', 'font-size': '18px',
                                             'color': "#df6262"} ),
                ])
            ], ),
        ], width=3),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H2(['Police Shootings in the United States 2015 - Present'],
                            style={'font-weight': 'bold', 'text-align': 'center', 'font-size': '22px'}, ),
                ])
            ], ),
        ], width=6),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Victims", style={'font-weight': 'bold', 'text-align': 'center', 'font-size': '18px'}),
                dbc.CardBody([
                    html.H2(id='Chosen_info_death', children="000",
                            style={'color': "#df6262", 'font-weight': 'bold', 'textAlign': 'center',
                                   'font-size': '22px'})
                ], )
            ]),
        ], width=1),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Armed", style={'font-weight': 'bold', 'text-align': 'center', 'font-size': '18px'}),
                dbc.CardBody([
                    html.H2(id='Chosen_info_armed', children="000",
                            style={'color': "#df6262", 'font-weight': 'bold', 'textAlign': 'center',
                                   'font-size': '22px'})
                ])
            ]),
        ], width=1),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Threatening",
                               style={'font-weight': 'bold', 'text-align': 'center', 'font-size': '18px'}),
                dbc.CardBody([
                    html.H2(id='Chosen_info_threat', children="000",
                            style={'color': "#df6262", 'font-weight': 'bold', 'textAlign': 'center',
                                   'font-size': '22px'})
                ], )
            ]),
        ], width=1),

    ], className="mt-auto", no_gutters=True, justify="start"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H2(['Click a County for More Detailed State and County Data:'],
                            style={'font-weight': 'bold', 'text-align': 'center', 'font-size': '18px'}, ),
                    html.H2([], style={'font-weight': 'bold', 'text-align': 'center', 'font-size': '18px'},
                            id='chosen_county'),

                    dcc.Graph(id='county_map', clickData=None, figure=fig),
                ])
            ]),
        ], width={'size': 7}),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H2(['USA Regional Information'],
                            style={'font-weight': 'bold', 'text-align': 'center', 'font-size': '20px'}, ),

                    dcc.Dropdown(id='my_dropdown',
                                 options=[
                                     {'label': 'Unemployment', 'value': 'unemprate'},
                                     {'label': 'HS dropout rate', 'value': 'HSdip'},
                                     {'label': 'Killed in relation to population size', 'value': 'populationpercent'},
                                     {'label': 'Median Income', 'value': 'medinc'},
                                     {'label': 'Percent White', 'value': 'popwhite'},
                                     {'label': 'Percent Black', 'value': 'popbl'},
                                     {'label': 'Percent Asian', 'value': 'popas'},
                                     {'label': 'Percent Indian', 'value': 'popind'},
                                     {'label': 'Percent Hispanic', 'value': 'pophis'},
                                 ],
                                 optionHeight=40,
                                 value='populationpercent',
                                 disabled=False,
                                 multi=False,
                                 searchable=True,
                                 search_value='',
                                 placeholder='Select a Value',
                                 clearable=True,
                                 style={'width': "100%"},
                                 ),

                    dcc.Graph(id='scatter', figure={}, config={'displayModeBar': False}),
                ])
            ]),
        ], width={'size': 5}),
    ], className="mt-auto", no_gutters=False, justify="start"),

    dbc.CardGroup([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H2(['% of Population and % of Victims by Race in the USA'],
                            style={'font-weight': 'bold', 'text-align': 'center', 'font-size': '20px'},
                            id='race_bar_title'),
                    dcc.Graph(id='race_bar', figure={}),
                ])
            ]),
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H2(['Deaths by Age Group, Race, and Gender in the USA'],
                            style={'font-weight': 'bold', 'text-align': 'center', 'font-size': '20px'},
                            id='age_and_gender_title'),
                    dcc.Graph(id='age_and_gender', figure={}, config={'displayModeBar': False}),
                ])
            ]),
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                        html.H2(['Parallel Coordinates: US Counties'],
                            style={'font-weight': 'bold', 'text-align': 'center', 'font-size': '18px'}, ),
                        dcc.Dropdown(id='my_dropdown1',
                                 options=[
                                    {'label': 'All Regions', 'value': 'All Regions'},
                                     {'label': 'South', 'value': 'South'},
                                     {'label': 'Northeast', 'value': 'Northeast'},
                                     {'label': 'West', 'value': 'West'},
                                     {'label': 'Midwest', 'value': 'Midwest'},
                                        ],
                                 optionHeight=40,
                                 value='All Regions',
                                 disabled=False,
                                 multi=False,
                                 searchable=True,
                                 search_value='',
                                 placeholder='Select a Value',
                                 clearable=True,
                                 style={'width': "100%"},
                                 ),
                    dcc.Graph(id='PC', figure={}, config={'displayModeBar': False}),
                ])
            ]),
        ], width=4),
    ], className="mt-auto"),
], fluid=True)





@app.callback(
    Output(component_id='PC', component_property='figure'),
    [Input(component_id='county_map', component_property='clickData'),
    Input(component_id='my_dropdown1', component_property='value')]
)
def change_plots(cd, drop):
    county_df["Percent Poverty"]= round((county_df["POVALL_2019"]/ county_df["Population"])*100,2)

    dims = ['Population', 'Deaths per 100,000 Residents', 'Unemployment Rate (%)',
            '% of Adults Without HS Diploma', 'Median Household Income', 'Percent Poverty']

    if drop == "All Regions":
        if cd is not None:
            place = cd['points'][0]['location']  # This will return the value in fips_txt
        else:
            place = '36103'  # Start at Suffolk County, NY by default

        county_df['pc_color'] = [0, ] * county_df.shape[0]
        county_df.loc[county_df['fips_txt'] == place, 'pc_color'] = 1

            # dimensions argument to parallel coordinates plot
        fig_pc = px.parallel_coordinates(county_df, dimensions=dims, color=county_df['pc_color'],
                                             color_continuous_scale="Plasma", color_continuous_midpoint=0.5)
        fig_pc.update_layout(coloraxis_showscale=False)
        fig_pc.update_traces(labelangle=-10, labelfont_size=10, labelside='bottom')
        fig_pc.update_layout(height= 440, margin={"r": 50, "t": 10, "l": 30, "b": 70})


    elif drop == "South":
        if cd is not None:
            place = cd['points'][0]['location']  # This will return the value in fips_txt
        else:
            place = '36103'  # Start at Suffolk County, NY by default

        county_df['pc_color'] = [0, ] * county_df.shape[0]
        county_df.loc[county_df['region'] == "South", 'pc_color'] = .2
        county_df.loc[county_df['fips_txt'] == place, 'pc_color'] = 1

            # dimensions argument to parallel coordinates plot
        fig_pc = px.parallel_coordinates(county_df, dimensions=dims, color=county_df['pc_color'],
                                             color_continuous_scale="Plasma", color_continuous_midpoint=0.5)
        fig_pc.update_layout(coloraxis_showscale=False)
        fig_pc.update_traces(labelangle=-10, labelfont_size=10, labelside='bottom')
        fig_pc.update_layout(height= 440,margin={"r": 50, "t": 10, "l": 30, "b": 70})

    elif drop == "Northeast":
                if cd is not None:
                    place = cd['points'][0]['location']  # This will return the value in fips_txt
                else:
                    place = '36103'  # Start at Suffolk County, NY by default

                county_df['pc_color'] = [0, ] * county_df.shape[0]

                county_df.loc[county_df['region'] == "Northeast", 'pc_color'] = .5
                county_df.loc[county_df['fips_txt'] == place, 'pc_color'] = 1

                # dimensions argument to parallel coordinates plot
                fig_pc = px.parallel_coordinates(county_df, dimensions=dims, color=county_df['pc_color'],
                                                     color_continuous_scale="Plasma", color_continuous_midpoint=0.5)
                fig_pc.update_layout(coloraxis_showscale=False)
                fig_pc.update_traces(labelangle=-10, labelfont_size=10, labelside='bottom')
                fig_pc.update_layout(height= 440,margin={"r": 50, "t": 10, "l": 30, "b": 70})


    elif drop == "West":
        if cd is not None:
            place = cd['points'][0]['location']  # This will return the value in fips_txt
        else:
            place = '36103'  # Start at Suffolk County, NY by default

        county_df['pc_color'] = [0, ] * county_df.shape[0]

        county_df.loc[county_df['region'] == "West", 'pc_color'] = .75
        county_df.loc[county_df['fips_txt'] == place, 'pc_color'] = 1

            # dimensions argument to parallel coordinates plot
        fig_pc = px.parallel_coordinates(county_df, dimensions=dims, color=county_df['pc_color'],
                                             color_continuous_scale="Plasma", color_continuous_midpoint=0.5)
        fig_pc.update_layout(coloraxis_showscale=False)
        fig_pc.update_traces(labelangle=-10, labelfont_size=10, labelside='bottom')
        fig_pc.update_layout(height= 440,margin={"r": 50, "t": 10, "l": 30, "b": 70})

    elif drop == "Midwest":
        if cd is not None:
            place = cd['points'][0]['location']  # This will return the value in fips_txt
        else:
            place = '36103'  # Start at Suffolk County, NY by default


        county_df['pc_color'] = [0, ] * county_df.shape[0]

        county_df.loc[county_df['region'] == "Midwest", 'pc_color'] = .37
        county_df.loc[county_df['fips_txt'] == place, 'pc_color'] = 1

            # dimensions argument to parallel coordinates plot
        fig_pc = px.parallel_coordinates(county_df, dimensions=dims, color=county_df['pc_color'],
                                             color_continuous_scale="Plasma", color_continuous_midpoint=0.5)
        fig_pc.update_layout(coloraxis_showscale=False)
        fig_pc.update_traces(labelangle=-10, labelfont_size=10, labelside='bottom')
        fig_pc.update_layout(height= 440,margin={"r": 50, "t": 10, "l": 30, "b": 70})



    return fig_pc


# multi-bar bar chart
@app.callback(
    Output(component_id='age_and_gender', component_property='figure'),
    [Input(component_id='county_map', component_property='clickData'),
     Input(component_id='radio', component_property='value')]
)
def change_agfig(cd, radio_val):
    # create age ranges
    cut_bins = [x * 10 for x in range(11)]
    police_shooting['Age Range'] = pd.cut(police_shooting['age'], bins=cut_bins)
    police_shooting['Age Range'] = police_shooting['Age Range'].astype(str)
    # clean up
    police_shooting.replace({'race': {'W': 'White', 'B': 'Black', 'A': 'Asian', 'H': 'Hispanic', 'N': 'Native American',
                                      'O': 'Other/Unknown'},
                             'gender': {'M': 'Male', 'F': 'Female'}}, inplace=True)

    bar_color_df = pd.DataFrame(
        {'race': ['White', 'Black', 'Asian', 'Hispanic', 'Native American', 'Other/Unknown', ] * 2,
         'gender': (['Male', 'Male', 'Male', 'Male', 'Male', 'Male',
                     'Female', 'Female', 'Female', 'Female', 'Female', 'Female']),
         'color': ['#0d0887', '#7904a6', '#bd3786', '#df6262', '#f8983f', '#f4ec22',
                   '#5a02a3', '#9413a0', '#cc4877', '#e66f5a', '#fcb62f', '#e8efc1']})

    if radio_val == 'USA':
        ag_df = pd.DataFrame(police_shooting.groupby(by=['Age Range', 'race', 'gender'], as_index=False).size())
        ag_df.rename(columns={'size': 'deaths'}, inplace=True)

        ag_fig = go.Figure()

        # Making the bars for each gender and race group
        for i in ['Male', 'Female']:
            base_vals = pd.DataFrame({'Age Range': np.sort(police_shooting['Age Range'].unique()),
                                      'cum_deaths': [0, ] * 10})  # sets the "base" for each bar for stacking
            for j in ag_df['race'].unique():
                if i == 'Male':
                    bar_group = 0
                else:
                    bar_group = 1

                bar_df = ag_df[(ag_df['race'] == j) & (ag_df['gender'] == i)]

                # update base values to stack the next bar on top
                if 'deaths' in base_vals.columns:
                    base_vals.drop(['deaths'], axis=1, inplace=True)
                base_vals = pd.merge(base_vals, bar_df.loc[:, ['deaths', 'Age Range']], how='outer', on='Age Range')
                base_vals.fillna(0, inplace=True)

                ag_fig.add_trace(go.Bar(name=j + ' ' + i,
                                        x=base_vals['Age Range'],
                                        y=base_vals['deaths'],
                                        offsetgroup=bar_group,
                                        base=base_vals['cum_deaths'],
                                        marker_color=bar_color_df.loc[(bar_color_df['race'] == j) & (
                                                    bar_color_df['gender'] == i), 'color'].values[0],
                                        hovertext=base_vals['deaths']))
                base_vals['cum_deaths'] = base_vals['cum_deaths'] + base_vals['deaths']
                ag_fig.update_layout(height= 450 ,xaxis_tickangle=70, xaxis_tickfont_size=12,
                                     yaxis_title='Deaths',
                                     xaxis_title={'text': 'Age Group (Bar Order: Male, Female)', 'font_size': 14},
                                     margin={"r": 10, "t": 30, "l": 10, "b": 70})

        # ag_fig.update_layout(title={'text': 'Deaths by Age Group, Race, and Gender in the USA', 'x': 0.5,
        #                             'font_size':14, 'yanchor': 'top', 'y': 0.95})

    elif radio_val== "State":
        if cd is not None:
            place = cd['points'][0]['location']  # This will return the value in fips_txt
        else:
            place = '36103'  # Start at Suffolk County, NY by default
        selected_state = county_stuff.loc[county_stuff['fips_txt'] == place, 'Geographic Area'].values[0]

        state_df = police_shooting.loc[police_shooting['Geographic Area_x'] == selected_state, :]
        ag_df = pd.DataFrame(state_df.groupby(by=['Age Range', 'race', 'gender'], as_index=False).size())
        ag_df.rename(columns={'size': 'deaths'}, inplace=True)
        ag_fig = go.Figure()
        ag_fig.update_layout(height= 450, xaxis_tickangle=70, xaxis_tickfont_size=12,
                             yaxis_title='Deaths',
                             xaxis_title={'text': 'Age Group (Bar Order: Male, Female)', 'font_size': 14},
                             margin={"r": 10, "t": 30, "l": 10, "b": 70})

        for i in ag_df['gender'].unique():
            base_vals = pd.DataFrame({'Age Range': np.sort(police_shooting['Age Range'].unique()),
                                      'cum_deaths': [0, ] * 10})  # sets the "base" for each bar for stacking
            # print(base_vals)
            for j in np.sort(pd.Series(ag_df.loc[ag_df['gender'] == i, 'race']).unique()):
                if i == 'Male':
                    bar_group = 0
                else:
                    bar_group = 1

                bar_df = ag_df[(ag_df['race'] == j) & (ag_df['gender'] == i)]

                # update base values to stack the next bar on top
                if 'deaths' in base_vals.columns:
                    base_vals.drop(['deaths'], axis=1, inplace=True)
                base_vals = pd.merge(base_vals, bar_df.loc[:, ['deaths', 'Age Range']], how='outer', on='Age Range')
                base_vals.fillna(0, inplace=True)

                ag_fig.add_trace(go.Bar(name=j + ' ' + i,
                                        x=base_vals['Age Range'],
                                        y=base_vals['deaths'],
                                        offsetgroup=bar_group,
                                        base=base_vals['cum_deaths'],
                                        marker_color=bar_color_df.loc[(bar_color_df['race'] == j) & (
                                                    bar_color_df['gender'] == i), 'color'].values[0],
                                        hovertext=base_vals['deaths'], ))
                base_vals['cum_deaths'] = base_vals['cum_deaths'] + base_vals['deaths']
                ag_fig.update_layout(height= 450, xaxis_tickangle=70, xaxis_tickfont_size=12,
                                     yaxis_title='Deaths',
                                     xaxis_title={'text': 'Age Group (Bar Order: Male, Female)', 'font_size': 14},
                                     margin={"r": 10, "t": 30, "l": 10, "b": 70})

    elif radio_val == "County":
        if cd is not None:
            place = cd['points'][0]['location']  # This will return the value in fips_txt
        else:
            place = '36103'  # Start at Suffolk County, NY by default
        selected_state = county_stuff.loc[county_stuff['fips_txt'] == place, 'Geographic Area'].values[0]

        state_df = police_shooting.loc[police_shooting['fips_txt'] == place, :]


        ag_df = pd.DataFrame(state_df.groupby(by=['Age Range', 'race', 'gender'], as_index=False).size())
        ag_df.rename(columns={'size': 'deaths'}, inplace=True)
        ag_fig = go.Figure()
        ag_fig.update_layout(height= 450, xaxis_tickangle=70, xaxis_tickfont_size=12,
                             yaxis_title='Deaths',
                             xaxis_title={'text': 'Age Group (Bar Order: Male, Female)', 'font_size': 14},
                             margin={"r": 10, "t": 30, "l": 10, "b": 70})

        for i in ag_df['gender'].unique():
            base_vals = pd.DataFrame({'Age Range': np.sort(police_shooting['Age Range'].unique()),
                                      'cum_deaths': [0, ] * 10})  # sets the "base" for each bar for stacking
            # print(base_vals)
            for j in np.sort(pd.Series(ag_df.loc[ag_df['gender'] == i, 'race']).unique()):
                if i == 'Male':
                    bar_group = 0
                else:
                    bar_group = 1

                bar_df = ag_df[(ag_df['race'] == j) & (ag_df['gender'] == i)]

                # update base values to stack the next bar on top
                if 'deaths' in base_vals.columns:
                    base_vals.drop(['deaths'], axis=1, inplace=True)
                base_vals = pd.merge(base_vals, bar_df.loc[:, ['deaths', 'Age Range']], how='outer', on='Age Range')
                base_vals.fillna(0, inplace=True)

                ag_fig.add_trace(go.Bar(name=j + ' ' + i,
                                        x=base_vals['Age Range'],
                                        y=base_vals['deaths'],
                                        offsetgroup=bar_group,
                                        base=base_vals['cum_deaths'],
                                        marker_color=bar_color_df.loc[(bar_color_df['race'] == j) & (
                                                bar_color_df['gender'] == i), 'color'].values[0],
                                        hovertext=base_vals['deaths'], ))
                base_vals['cum_deaths'] = base_vals['cum_deaths'] + base_vals['deaths']
                ag_fig.update_layout(height= 450, xaxis_tickangle=70, xaxis_tickfont_size=12,
                                     yaxis_title='Deaths',
                                     xaxis_title={'text': 'Age Group (Bar Order: Male, Female)', 'font_size': 14},
                                     margin={"r": 10, "t": 30, "l": 10, "b": 80})

    return ag_fig


@app.callback(
    Output(component_id='scatter', component_property='figure'),
    [Input(component_id='my_dropdown', component_property='value'),
     Input(component_id='county_map', component_property='clickData')]
)
def line_plot(dropdownvalue, cd):
    if cd is not None:
        place = cd['points'][0]['location']  # This will return the value in fips_txt
        print(place)
    else:
        place = '36103'  # Start at Suffolk County, NY by default

    df1.groupby('region')['Deaths'].sum()
    df1.groupby('region')['Population'].sum()

    region = df1['region'].unique()
    unemp = round(df1.groupby('region')['Unemployment Rate (%)'].mean(), 2)
    medinc = round(df1.groupby('region')['Median Household Income'].mean(), 2)
    HSdip = round(df1.groupby('region')['% of Adults Without HS Diploma'].mean(), 2)
    popwhite = df1.groupby('region')['white_pop_y'].sum() / df1.groupby('region')['Population'].sum() * 100
    popbl = df1.groupby('region')['black_pop_y'].sum() / df1.groupby('region')['Population'].sum() * 100
    popas = df1.groupby('region')['asian_pop_y'].sum() / df1.groupby('region')['Population'].sum() * 100
    popind = df1.groupby('region')['indian_pop_y'].sum() / df1.groupby('region')['Population'].sum() * 100
    pophis = df1.groupby('region')['hisp_pop_y'].sum() / df1.groupby('region')['Population'].sum() * 100
    populationpercent = (df1.groupby('region')['Deaths'].sum() / df1.groupby('region')['Population'].sum()) * 1000000

    data = {"unemprate": unemp, "HSdip": HSdip, "popwhite": popwhite, "popbl": popbl, "popas": popas, "popind": popind,
            "pophis": pophis, "populationpercent": populationpercent, "medinc": medinc, "region": region}
    df2 = pd.DataFrame(data=data)

    selected_region = df1.loc[df1['fips_txt'] == place, 'region'].values[0]
    state_df = df1.loc[df1['region'] == selected_region, :]

    Midwestdata = list(df2.iloc[0])
    Northeastdata = list(df2.iloc[1])
    Southdata = list(df2.iloc[2])
    Westdata = list(df2.iloc[3])

    df2["color"] = [0.30, ] * df2.shape[0]
    df2.loc[df2.index == selected_region, 'color'] = 1

    bar = px.scatter(df, x=df2.index, y=df2[dropdownvalue], color=df2["color"], color_continuous_scale="Plasma")

    label_dict = {'unemprate': 'Unemployment Rate', 'HSdip': 'HS Dropout Rate', 'popwhite': 'White Population %',
                  'popbl': 'Black Population %', 'popas': 'Asian Population %',
                  'popind': 'Native American Population %',
                  'pophis': 'Hispanic Population %', 'populationpercent': 'Deaths per 1 Million People',
                  'medinc': 'Median Income'}
    bar.update_layout(xaxis_title='Region', yaxis_title=label_dict[dropdownvalue])
    bar.update_traces(marker=dict(size=20,
                                  line=dict(width=0.5, color='DarkSlateGrey')))
    bar.update_layout(height= 390, margin={"r": 0, "t": 10, "l": 0, "b": 0})
    bar.update_layout(coloraxis_showscale=False)

    return bar


@app.callback(
    Output(component_id='race_bar', component_property='figure'),
    [Input(component_id='radio', component_property='value'),
     Input(component_id='county_map', component_property='clickData')]
)
def race_bar(radio_val, cd):
    if cd is not None:
        place = cd['points'][0]['location']  # This will return the value in fips_txt
    else:
        place = '36103'  # Start at Suffolk County, NY by default


    if radio_val == 'State':
        # get the sum of all deaths in the state

        ddf = pd.DataFrame(police_shootings.groupby(by=['race', 'Geographic Area'], as_index=False).size())

        # get the sum of all deaths in the state
        selected_state = county_stuff.loc[county_stuff['fips_txt'] == place, 'Geographic Area'].values[0]
        sumdf_state1 = pd.DataFrame(ddf.loc[ddf['Geographic Area'] == selected_state])

        selected_state = county_stuff.loc[county_stuff['fips_txt'] == place, 'Geographic Area'].values[0]
        sumdf_state = pd.DataFrame(sumdf.loc[sumdf['Geographic Area'] == selected_state,
                                             ['White Population', 'Black Population', 'Asian Population',
                                              'Native American Population',
                                              'Hispanic Population', 'Other/Unknown', 'Population Adjusted']])

        pie_df1 = pd.DataFrame({'race': sumdf_state.columns[0:6],
                                'population': sumdf_state.iloc[0, 0:6]})

        pie_df = pd.merge(sumdf_state1, pie_df1, on=["race"])

        # express in terms of percentages
        pie_df['population percent'] = round((pie_df['population'] / pie_df1['population'].sum()) * 100, 2)
        pie_df['Death percent'] = round((pie_df['size'] / pie_df['size'].sum()) * 100, 2)

        # pie = px.sunburst(pie_df, path=['Deathper', 'population'], values='size')

        bar = go.Figure()
        bar.add_trace(go.Bar(
            name='Population Percent',
            x=pie_df["race"],
            y=pie_df['population percent'],
            marker_color='#0d0887',
        )),
        bar.add_trace(
            go.Bar(name='Death Percent', x=pie_df["race"], y=pie_df['Death percent'], marker_color='#df6262'))

        # Change the bar mode
        bar.update_layout(height= 450, barmode='group', yaxis_title='%',
                          xaxis_title='Race', xaxis_tickangle=20, xaxis_tickfont_size=10)
        bar.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 60})


        return bar

    elif radio_val == 'USA':
        ddf = pd.DataFrame(police_shootings.groupby(by=['race'], as_index=False).size())
        sumdf_usa = sumdf.loc[:,
                    ['White Population', 'Black Population', 'Asian Population', 'Native American Population',
                     'Hispanic Population', 'Other/Unknown']].sum()
        pie_df1 = pd.DataFrame(sumdf_usa, columns=['population'])
        pie_df1.reset_index(inplace=True)
        pie_df1.rename(columns={'index': 'race'}, inplace=True)

        pie_df = pd.merge(ddf, pie_df1, on=["race"])

        pie_df['population percent'] = round((pie_df['population'] / pie_df['population'].sum()) * 100, 2)
        pie_df['Death percent'] = round((pie_df['size'] / pie_df['size'].sum()) * 100, 2)

        bar = go.Figure()
        bar.add_trace(go.Bar(
            name='Population Percent',
            x=pie_df["race"],
            y=pie_df['population percent'],
            marker_color='#0d0887',
        )),
        bar.add_trace(
            go.Bar(name='Death Percent', x=pie_df["race"], y=pie_df['Death percent'], marker_color='#bd3786'))

        # bar = go.Figure(data=[
        #     go.Bar(name='population percent', x=pie_df["race"], y=pie_df['population percent']),
        #     go.Bar(name='Death percent', x=pie_df["race"], y=pie_df['Death percent']),
        # ])
        # Change the bar mode
        bar.update_layout(height= 450, barmode='group', yaxis_title='%',
                          xaxis_title='Race', xaxis_tickangle=20, xaxis_tickfont_size=10, )
        # title={'text': '% of Population and % of Victims by Race in the USA',
        # 'x': 0.5, 'font_size': 14, 'yanchor': 'top', 'y': 0.95}
        bar.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 60})

        return bar


    elif radio_val == 'County':
            # get the sum of all deaths in the state

        ddf = pd.DataFrame(police_shootings.groupby(by=['race', 'fips_txt'], as_index=False).size())

            # get the sum of all deaths in the state
        selected_state = county_stuff.loc[county_stuff['fips_txt'] == place, 'fips_txt'].values[0]
        sumdf_state1 = pd.DataFrame(ddf.loc[ddf['fips_txt'] == selected_state])

        selected_state = county_stuff.loc[county_stuff['fips_txt'] == place, 'fips_txt'].values[0]
        sumdf_state = pd.DataFrame(sumdfC.loc[sumdfC['FIPS'] == selected_state,
                                                 ['White Population', 'Black Population', 'Asian Population',
                                                  'Native American Population',
                                                  'Hispanic Population', 'Other/Unknown', 'Population Adjusted']])

        pie_df1 = pd.DataFrame({'race': sumdf_state.columns[0:6],
                                    'population': sumdf_state.iloc[0, 0:6]})

        pie_df = pd.merge(sumdf_state1, pie_df1, on=["race"])

            # express in terms of percentages
        pie_df['population percent'] = round((pie_df['population'] / pie_df1['population'].sum()) * 100, 2)
        pie_df['Death percent'] = round((pie_df['size'] / pie_df['size'].sum()) * 100, 2)

            # pie = px.sunburst(pie_df, path=['Deathper', 'population'], values='size')

        bar = go.Figure()
        bar.add_trace(go.Bar(
                name='Population Percent',
                x=pie_df["race"],
                y=pie_df['population percent'],
                marker_color='#0d0887',
            )),
        bar.add_trace(
                go.Bar(name='Death Percent', x=pie_df["race"], y=pie_df['Death percent'], marker_color='#df6262'))

            # Change the bar mode
        bar.update_layout(height= 450,barmode='group', yaxis_title='%',
                              xaxis_title='Race', xaxis_tickangle=20, xaxis_tickfont_size=10)
        bar.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 60})

        return bar


@app.callback(
    Output(component_id='age_and_gender_title', component_property='children'),
    [Input(component_id='county_map', component_property='clickData'),
     Input(component_id='radio', component_property='value')]
)
def change_race_and_gender_title(cd, radio_val):
    if radio_val == 'USA':
        return 'Deaths by Age Group, Race, and Gender in the USA'
    elif radio_val == "County":
        if cd is not None:
            place = cd['points'][0]['location']  # This will return the value in fips_txt
        else:
            place = '36103'  # Start at Suffolk County, NY by default
        selected_county = county_stuff.loc[county_stuff['fips_txt'] == place, 'county_name'].values[0]
        return 'Deaths by Age Group, Race, and Gender in ' + selected_county
    else:
        if cd is not None:
            place = cd['points'][0]['location']  # This will return the value in fips_txt
        else:
            place = '36103'  # Start at Suffolk County, NY by default
        selected_state = county_stuff.loc[county_stuff['fips_txt'] == place, 'state'].values[0]
        return 'Deaths by Age Group, Race, and Gender in ' + selected_state


@app.callback(
    Output(component_id='race_bar_title', component_property='children'),
    [Input(component_id='county_map', component_property='clickData'),
     Input(component_id='radio', component_property='value')]
)
def change_race_bar_title(cd, radio_val):
    if radio_val == 'USA':
        return '% of Population and % of Victims by Race in the USA'
    elif radio_val == "County":
        if cd is not None:
            place = cd['points'][0]['location']  # This will return the value in fips_txt
        else:
            place = '36103'  # Start at Suffolk County, NY by default
        selected_county = county_stuff.loc[county_stuff['fips_txt'] == place, 'county_name'].values[0]
        return '% of Population and % of Victims by Race in ' + selected_county
    else:
        if cd is not None:
            place = cd['points'][0]['location']  # This will return the value in fips_txt
        else:
            place = '36103'  # Start at Suffolk County, NY by default

        selected_state = county_stuff.loc[county_stuff['fips_txt'] == place, 'state'].values[0]
        return '% of Population and % of Victims by Race in ' + selected_state


@app.callback(
    Output(component_id='chosen_county', component_property='children'),
    [Input(component_id='county_map', component_property='clickData')]
)
def change_subtitle(cd):
    if cd is not None:
        place = cd['points'][0]['location']  # This will return the value in fips_txt
    else:
        place = '36103'  # Start at Suffolk County, NY by default
    selected_county = county_stuff.loc[county_stuff['fips_txt'] == place, 'county_name'].values[0]
    selected_state = county_stuff.loc[county_stuff['fips_txt'] == place, 'Geographic Area'].values[0]
    return "You selected: " + selected_county + ', ' + selected_state


@app.callback(
    Output(component_id='Chosen_info_death', component_property='children'),
    Output(component_id='Chosen_info_armed', component_property='children'),
    Output(component_id='Chosen_info_threat', component_property='children'),
    [Input(component_id='county_map', component_property='clickData')]
)
def change_subtitle(cd):
    Armed = 5346 - 350
    Deaths = police_shootings["armed"].count()
    HighThreat = police_shootings.groupby("threat_level")["threat_level"].count()["attack"]

    return Deaths, Armed, HighThreat


if __name__ == '__main__':
    app.run_server(debug=False, port=8004)