# region imports

# from zmq import VERSION_PATCH
import dash
from dash.dependencies import Input, Output
from dash import dcc, html, callback

import plotly.express as px
import plotly.graph_objects as go
import plotly as py
import pandas as pd
import sys
import numpy as np
import os
import geopandas as gpd
import pandas as pd

# from dask import dataframe as dd
import plotly.express as px
import plotly as py
from plotly.subplots import make_subplots
import os
from datetime import datetime, date, timedelta
from flask_caching import Cache

# from matplotlib import rcParams, cycler
from itertools import cycle

import json

import seaborn as sns
import matplotlib.pyplot as plt
import contextily as ctx
import plotly.figure_factory as ff

# endregion imports

# region value initialization
mapbox_token = "pk.eyJ1Ijoia2ltc29vaWwiLCJhIjoiY2wxa3Byd3A0MDI3dTNibzg0czF3dHd3aCJ9.hrsnzqpk-4MtEvfh_DZRdg"

START_DATE = datetime.now()
END_DATE = datetime.now() + timedelta(days=31)

SF_cases = 0.02
SF_admissions = 0.05
SF_deaths = 1

ZIPS = [
    "33510",
    "33511",
    "33527",
    "33534",
    "33547",
    "33548",
    "33549",
    "33556",
    "33558",
    "33559",
    "33563",
    "33565",
    "33566",
    "33567",
    "33569",
    "33570",
    "33572",
    "33573",
    "33578",
    "33579",
    "33584",
    "33592",
    "33594",
    "33596",
    "33598",
    "33602",
    "33603",
    "33604",
    "33605",
    "33606",
    "33607",
    "33609",
    "33610",
    "33611",
    "33612",
    "33613",
    "33614",
    "33615",
    "33616",
    "33617",
    "33618",
    "33619",
    "33624",
    "33625",
    "33626",
    "33629",
    "33634",
    "33635",
    "33637",
    # "33647",
    # '33620', '33503'
]

ZIPS_centers = {
    "33510": [27.96, -82.30],
    "33511": [27.90, -82.30],
    "33527": [27.97, -82.22],
    "33534": [27.83, -82.38],
    "33547": [27.8, -82.1],
    "33548": [28.15, -82.48],
    "33549": [28.14, -82.45],
    "33556": [28.2, -82.6],
    "33558": [28.16, -82.51],
    "33559": [28.16, -82.40],
    "33563": [28.02, -82.13],
    "33565": [28.10, -82.15],
    "33566": [27.99, -82.13],
    "33567": [27.91, -82.12],
    "33569": [27.85, -82.29],
    "33570": [27.70, -82.47],
    "33572": [27.77, -82.40],
    "33573": [27.73, -82.36],
    "33578": [27.84, -82.35],
    "33579": [27.80, -82.28],
    "33584": [28.00, -82.29],
    "33592": [28.10, -82.28],
    "33594": [27.94, -82.24],
    "33596": [27.89, -82.23],
    "33598": [27.7, -82.3],
    "33602": [27.95, -82.46],
    "33603": [27.99, -82.46],
    "33604": [28.01, -82.45],
    "33605": [27.94, -82.43],
    "33606": [27.93, -82.46],
    "33607": [27.96, -82.54],
    "33609": [27.94, -82.52],
    "33610": [28.00, -82.38],
    "33611": [27.89, -82.51],
    "33612": [28.05, -82.45],
    "33613": [28.09, -82.45],
    "33614": [28.01, -82.50],
    "33615": [28.00, -82.58],
    "33616": [27.86, -82.53],
    "33617": [28.04, -82.39],
    "33618": [28.08, -82.50],
    "33619": [27.90, -82.38],
    "33624": [28.08, -82.52],
    "33625": [28.07, -82.56],
    "33626": [28.06, -82.61],
    "33629": [27.92, -82.51],
    "33634": [28.00, -82.54],
    "33635": [28.02, -82.61],
    "33637": [28.05, -82.36],
    "33647": [28.12, -82.35],
    "33620": [28.062, -82.410],
    "33503": [27.753, -82.2883],
}

if len(sys.argv) == 2:
    print(
        "Try to launch plotly-dash webapp using results in ./sim_output/"
        + str(sys.argv[1])
    )
    folder_name = sys.argv[1]
    path = os.path.join(".", "sim_output", folder_name)
    if os.path.exists(path):
        pass
    else:
        print(path + " does not exist")
        exit()
else:
    print("Please type result-folder name.")
    print("Usage: python " + str(sys.argv[0]) + " result_folder_name")
    exit()

asset_path = os.path.join(".", "assets")

period_explain = "(Simulation Run dates: {} ~ ".format(
    START_DATE.strftime("%B %d, %Y")
) + "{})".format(END_DATE.strftime("%B %d, %Y"))
period_explain2 = "(Vaccination period: {} ~ ".format(
    START_DATE.strftime("%B %d, %Y")
) + "{})".format(END_DATE.strftime("%B %d, %Y"))

default_zipcode = "33647"
default_year = "2022"
default_sampling = 1.0
default_zoom = 9

year_for_all = "2022"
zipcode_for_all = "33647"
sampling_for_all = 1.0  # 0.25 # 0.5=50%
show_whole_county = False

heatmap_size = 0
scatter_size = 0

graph_width = 900
graph_height = 750


legend_map = {
    "susceptible": "blue",
    "asymptomatic": "purple",
    "vaccinated": "olive",
    "boosted": "olive",
    "recovered": "green",
    "critical": "#F1948A",
    "dead": "black",
    "exposed": "orange",
    # "mild": "#F5B7B1",
    "mild": "magenta",
    "presymptomatic": "purple",
    # "presymptomatic": "#F2D7D5",
    "severe": "#EC7063",
}

fillcolor1 = "rgb(184, 247, 212)"
fillcolor2 = "rgb(111, 231, 219)"
fillcolor3 = "rgb(127, 166, 238)"
fillcolor4 = "rgb(131, 90, 241)"
fillcolor5 = "rgb(141, 80, 251)"
# endregion value initialization

# region dash initialization
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = dash.Dash(
    __name__,
    title="COVID-19 Dashboard powered by CitySEIRCast (USF-COPH-Dr.Edwin Michael Lab)",
)

app.prevent_initial_callbacks = True

cache = Cache(
    app.server,
    config={
        # need local redis server installation using docker(windows) or apt install(linux-ubuntu)
        "CACHE_TYPE": "redis",
        "CACHE_REDIS_URL": "redis://localhost:6379",
    },
)

CACHE_TIMEOUT = 24 * 60 * 60
# endregion dash initialization

# region SVI definitions
df_svi = pd.read_excel(
    os.path.join(asset_path, "allDataPython.xlsx"), sheet_name="Sheet1"
)
# Remove zip_code column
df_PR = df_svi.copy().drop("zip code", axis=1)

# Get percentile rank for every column and calculate sum
for column in list(df_PR):
    df_PR[column] = df_svi[column].rank(pct=True)
df_PR["PR_sum"] = df_PR.sum(axis=1)

# Add zip_code column back
df_PR.insert(loc=0, column="zip_code", value=df_svi["zip code"].to_list())

# Create new column
df_PR["community"] = 0
df_PR["SVI_Index"] = 0

# the range reprsenting the 4 groups involved by the quantile function
qr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# calculate the quantile function of the SVI = PR_sum
PR_Quartiles = np.quantile(df_PR["PR_sum"], qr)
PR_Index = {}

# classify the SVI : 0: low vulnerability 1: Medium vulnerability 2: High medium vulnerability 3: High vulnerability
for i in range(len(PR_Quartiles)):
    PR_Index[i] = PR_Quartiles[i]
# Set SVI_Index using PR_Index
for i in range(len(df_PR)):
    for j in PR_Index:
        if j == 0 and df_PR["PR_sum"][i] < PR_Index[j]:
            df_PR.loc[i, "SVI_Index"] = j / 10
        if (
            j > 0
            and df_PR["PR_sum"][i] > PR_Index[j - 1]
            and df_PR["PR_sum"][i] <= PR_Index[j]
        ):
            df_PR.loc[i, "SVI_Index"] = j / 10
ward_index = {}
ward_file = pd.read_csv(os.path.join(asset_path, "wardNo_to_zipcode.csv"))
for i in range(len(ward_file)):
    ward_index[ward_file.loc[i][1]] = ward_file.loc[i][0]
for i in range(len(df_PR)):
    if ward_index.get(df_PR["zip_code"][i]) != None:
        df_PR.loc[i, "community"] = ward_index[df_PR["zip_code"][i]]

df_PR["zipcode"] = df_PR["zip_code"].astype(str)
# endregion SVI definitions

# region Graph functions


# region Time Plots
# @cache.memoize(timeout=CACHE_TIMEOUT)
def load_SEIR(mode):
    # region dataframes

    # region actualDF
    actualDF = pd.concat([pd.read_csv(os.path.join(asset_path, "actual.csv"))], axis=1)
    actualDF.drop("date", axis=1, inplace=True)

    date_range = pd.date_range(start=START_DATE, periods=1061, freq="D")
    actualDF["date"] = pd.Series(date_range, index=(actualDF["time_step"] - 1))
    actualDF.drop("time_step", axis=1, inplace=True)

    # forecast - cut off days before start date from actualDF
    actualDF = actualDF[actualDF["date"] >= START_DATE]
    # endregion actualDF

    # region define dates
    dates = np.arange("2020-03-01", END_DATE, timedelta(days=1)).astype(datetime)
    # endregion define dates

    # region read parquet
    infectedDF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "infectedDF.parquet")
    )
    hospitalizedDF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "hospitalizedDF.parquet")
    )
    deadDF = pd.read_parquet(os.path.join(asset_path, "time_plots", "deadDF.parquet"))
    infectedAgeGroup1DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "infected_age_group_1.parquet")
    )
    infectedAgeGroup2DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "infected_age_group_2.parquet")
    )
    infectedAgeGroup3DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "infected_age_group_3.parquet")
    )
    infectedAgeGroup4DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "infected_age_group_4.parquet")
    )
    infectedAgeGroup5DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "infected_age_group_5.parquet")
    )
    infectedAgeGroup6DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "infected_age_group_6.parquet")
    )
    infectedAgeGroup7DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "infected_age_group_7.parquet")
    )
    infectedAgeGroup8DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "infected_age_group_8.parquet")
    )
    infectedAgeGroup9DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "infected_age_group_9.parquet")
    )
    infectedIncomeGroup1DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "infected_income_group_1.parquet")
    )
    infectedIncomeGroup2DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "infected_income_group_2.parquet")
    )
    infectedIncomeGroup3DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "infected_income_group_3.parquet")
    )
    infectedIncomeGroup4DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "infected_income_group_4.parquet")
    )
    infectedIncomeGroup5DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "infected_income_group_5.parquet")
    )
    infectedIncomeGroup6DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "infected_income_group_6.parquet")
    )
    infectedIncomeGroup7DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "infected_income_group_7.parquet")
    )
    infectedIncomeGroup8DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "infected_income_group_8.parquet")
    )
    infectedRaceGroup1DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "infected_race_group_1.parquet")
    )
    infectedRaceGroup2DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "infected_race_group_2.parquet")
    )
    infectedRaceGroup3DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "infected_race_group_3.parquet")
    )
    infectedRaceGroup4DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "infected_race_group_4.parquet")
    )
    infectedRaceGroup5DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "infected_race_group_5.parquet")
    )
    infectedRaceGroup6DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "infected_race_group_6.parquet")
    )
    infectedRaceGroup7DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "infected_race_group_7.parquet")
    )
    infectedEthnicityGroup1DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "infected_ethnicity_group_1.parquet")
    )
    infectedEthnicityGroup2DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "infected_ethnicity_group_2.parquet")
    )
    infectedGenderGroup1DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "infected_gender_group_1.parquet")
    )
    infectedGenderGroup2DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "infected_gender_group_2.parquet")
    )
    hospitalisedAgeGroup1DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "hospitalised_age_group_1.parquet")
    )
    hospitalisedAgeGroup2DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "hospitalised_age_group_2.parquet")
    )
    hospitalisedAgeGroup3DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "hospitalised_age_group_3.parquet")
    )
    hospitalisedAgeGroup4DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "hospitalised_age_group_4.parquet")
    )
    hospitalisedAgeGroup5DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "hospitalised_age_group_5.parquet")
    )
    hospitalisedAgeGroup6DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "hospitalised_age_group_6.parquet")
    )
    hospitalisedAgeGroup7DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "hospitalised_age_group_7.parquet")
    )
    hospitalisedAgeGroup8DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "hospitalised_age_group_8.parquet")
    )
    hospitalisedAgeGroup9DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "hospitalised_age_group_9.parquet")
    )
    hospitalisedIncomeGroup1DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "hospitalised_income_group_1.parquet")
    )
    hospitalisedIncomeGroup2DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "hospitalised_income_group_2.parquet")
    )
    hospitalisedIncomeGroup3DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "hospitalised_income_group_3.parquet")
    )
    hospitalisedIncomeGroup4DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "hospitalised_income_group_4.parquet")
    )
    hospitalisedIncomeGroup5DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "hospitalised_income_group_5.parquet")
    )
    hospitalisedIncomeGroup6DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "hospitalised_income_group_6.parquet")
    )
    hospitalisedIncomeGroup7DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "hospitalised_income_group_7.parquet")
    )
    hospitalisedIncomeGroup8DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "hospitalised_income_group_8.parquet")
    )
    hospitalisedRaceGroup1DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "hospitalised_race_group_1.parquet")
    )
    hospitalisedRaceGroup2DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "hospitalised_race_group_2.parquet")
    )
    hospitalisedRaceGroup3DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "hospitalised_race_group_3.parquet")
    )
    hospitalisedRaceGroup4DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "hospitalised_race_group_4.parquet")
    )
    hospitalisedRaceGroup5DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "hospitalised_race_group_5.parquet")
    )
    hospitalisedRaceGroup6DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "hospitalised_race_group_6.parquet")
    )
    hospitalisedRaceGroup7DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "hospitalised_race_group_7.parquet")
    )
    hospitalisedEthnicityGroup1DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "hospitalised_ethnicity_group_1.parquet")
    )
    hospitalisedEthnicityGroup2DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "hospitalised_ethnicity_group_2.parquet")
    )
    hospitalisedGenderGroup1DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "hospitalised_gender_group_1.parquet")
    )
    hospitalisedGenderGroup2DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "hospitalised_gender_group_2.parquet")
    )
    deadAgeGroup1DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "dead_age_group_1.parquet")
    )
    deadAgeGroup2DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "dead_age_group_2.parquet")
    )
    deadAgeGroup3DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "dead_age_group_3.parquet")
    )
    deadAgeGroup4DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "dead_age_group_4.parquet")
    )
    deadAgeGroup5DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "dead_age_group_5.parquet")
    )
    deadAgeGroup6DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "dead_age_group_6.parquet")
    )
    deadAgeGroup7DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "dead_age_group_7.parquet")
    )
    deadAgeGroup8DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "dead_age_group_8.parquet")
    )
    deadAgeGroup9DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "dead_age_group_9.parquet")
    )
    deadIncomeGroup1DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "dead_income_group_1.parquet")
    )
    deadIncomeGroup2DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "dead_income_group_2.parquet")
    )
    deadIncomeGroup3DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "dead_income_group_3.parquet")
    )
    deadIncomeGroup4DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "dead_income_group_4.parquet")
    )
    deadIncomeGroup5DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "dead_income_group_5.parquet")
    )
    deadIncomeGroup6DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "dead_income_group_6.parquet")
    )
    deadIncomeGroup7DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "dead_income_group_7.parquet")
    )
    deadIncomeGroup8DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "dead_income_group_8.parquet")
    )

    deadRaceGroup1DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "dead_race_group_1.parquet")
    )
    deadRaceGroup2DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "dead_race_group_2.parquet")
    )
    deadRaceGroup3DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "dead_race_group_3.parquet")
    )
    deadRaceGroup4DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "dead_race_group_4.parquet")
    )
    deadRaceGroup5DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "dead_race_group_5.parquet")
    )
    deadRaceGroup6DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "dead_race_group_6.parquet")
    )
    deadRaceGroup7DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "dead_race_group_7.parquet")
    )

    deadEthnicityGroup1DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "dead_ethnicity_group_1.parquet")
    )
    deadEthnicityGroup2DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "dead_ethnicity_group_2.parquet")
    )

    deadGenderGroup1DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "dead_gender_group_1.parquet")
    )
    deadGenderGroup2DF = pd.read_parquet(
        os.path.join(asset_path, "time_plots", "dead_gender_group_2.parquet")
    )
    # endregion read parquet

    # region Create dictionaries for arguments
    gender_dict = {
        "infectedGenderGroup1DF": infectedGenderGroup1DF,
        "infectedGenderGroup2DF": infectedGenderGroup2DF,
        "hospitalisedGenderGroup1DF": hospitalisedGenderGroup1DF,
        "hospitalisedGenderGroup2DF": hospitalisedGenderGroup2DF,
        "deadGenderGroup1DF": deadGenderGroup1DF,
        "deadGenderGroup2DF": deadGenderGroup2DF,
    }

    age_dict = {
        "infectedAgeGroup1DF": infectedAgeGroup1DF,
        "infectedAgeGroup2DF": infectedAgeGroup2DF,
        "infectedAgeGroup3DF": infectedAgeGroup3DF,
        "infectedAgeGroup4DF": infectedAgeGroup4DF,
        "infectedAgeGroup5DF": infectedAgeGroup5DF,
        "infectedAgeGroup6DF": infectedAgeGroup6DF,
        "infectedAgeGroup7DF": infectedAgeGroup7DF,
        "infectedAgeGroup8DF": infectedAgeGroup8DF,
        "infectedAgeGroup9DF": infectedAgeGroup9DF,
        "hospitalisedAgeGroup1DF": hospitalisedAgeGroup1DF,
        "hospitalisedAgeGroup2DF": hospitalisedAgeGroup2DF,
        "hospitalisedAgeGroup3DF": hospitalisedAgeGroup3DF,
        "hospitalisedAgeGroup4DF": hospitalisedAgeGroup4DF,
        "hospitalisedAgeGroup5DF": hospitalisedAgeGroup5DF,
        "hospitalisedAgeGroup6DF": hospitalisedAgeGroup6DF,
        "hospitalisedAgeGroup7DF": hospitalisedAgeGroup7DF,
        "hospitalisedAgeGroup8DF": hospitalisedAgeGroup8DF,
        "hospitalisedAgeGroup9DF": hospitalisedAgeGroup9DF,
        "deadAgeGroup1DF": deadAgeGroup1DF,
        "deadAgeGroup2DF": deadAgeGroup2DF,
        "deadAgeGroup3DF": deadAgeGroup3DF,
        "deadAgeGroup4DF": deadAgeGroup4DF,
        "deadAgeGroup5DF": deadAgeGroup5DF,
        "deadAgeGroup6DF": deadAgeGroup6DF,
        "deadAgeGroup7DF": deadAgeGroup7DF,
        "deadAgeGroup8DF": deadAgeGroup8DF,
        "deadAgeGroup9DF": deadAgeGroup9DF,
    }

    race_dict = {
        "infectedRaceGroup1DF": infectedRaceGroup1DF,
        "infectedRaceGroup2DF": infectedRaceGroup2DF,
        "infectedRaceGroup3DF": infectedRaceGroup3DF,
        "infectedRaceGroup4DF": infectedRaceGroup4DF,
        "infectedRaceGroup5DF": infectedRaceGroup5DF,
        "infectedRaceGroup6DF": infectedRaceGroup6DF,
        "infectedRaceGroup7DF": infectedRaceGroup7DF,
        "hospitalisedRaceGroup1DF": hospitalisedRaceGroup1DF,
        "hospitalisedRaceGroup2DF": hospitalisedRaceGroup2DF,
        "hospitalisedRaceGroup3DF": hospitalisedRaceGroup3DF,
        "hospitalisedRaceGroup4DF": hospitalisedRaceGroup4DF,
        "hospitalisedRaceGroup5DF": hospitalisedRaceGroup5DF,
        "hospitalisedRaceGroup6DF": hospitalisedRaceGroup6DF,
        "hospitalisedRaceGroup7DF": hospitalisedRaceGroup7DF,
        "deadRaceGroup1DF": deadRaceGroup1DF,
        "deadRaceGroup2DF": deadRaceGroup2DF,
        "deadRaceGroup3DF": deadRaceGroup3DF,
        "deadRaceGroup4DF": deadRaceGroup4DF,
        "deadRaceGroup5DF": deadRaceGroup5DF,
        "deadRaceGroup6DF": deadRaceGroup6DF,
        "deadRaceGroup7DF": deadRaceGroup7DF,
    }

    ethnic_dict = {
        "infectedEthnicityGroup1DF": infectedEthnicityGroup1DF,
        "infectedEthnicityGroup2DF": infectedEthnicityGroup2DF,
        "hospitalisedEthnicityGroup1DF": hospitalisedEthnicityGroup1DF,
        "hospitalisedEthnicityGroup2DF": hospitalisedEthnicityGroup2DF,
        "deadEthnicityGroup1DF": deadEthnicityGroup1DF,
        "deadEthnicityGroup2DF": deadEthnicityGroup2DF,
    }

    income_dict = {
        "infectedIncomeGroup1DF": infectedIncomeGroup1DF,
        "infectedIncomeGroup2DF": infectedIncomeGroup2DF,
        "infectedIncomeGroup3DF": infectedIncomeGroup3DF,
        "infectedIncomeGroup4DF": infectedIncomeGroup4DF,
        "infectedIncomeGroup5DF": infectedIncomeGroup5DF,
        "infectedIncomeGroup6DF": infectedIncomeGroup6DF,
        "infectedIncomeGroup7DF": infectedIncomeGroup7DF,
        "infectedIncomeGroup8DF": infectedIncomeGroup8DF,
        "hospitalisedIncomeGroup1DF": hospitalisedIncomeGroup1DF,
        "hospitalisedIncomeGroup2DF": hospitalisedIncomeGroup2DF,
        "hospitalisedIncomeGroup3DF": hospitalisedIncomeGroup3DF,
        "hospitalisedIncomeGroup4DF": hospitalisedIncomeGroup4DF,
        "hospitalisedIncomeGroup5DF": hospitalisedIncomeGroup5DF,
        "hospitalisedIncomeGroup6DF": hospitalisedIncomeGroup6DF,
        "hospitalisedIncomeGroup7DF": hospitalisedIncomeGroup7DF,
        "hospitalisedIncomeGroup8DF": hospitalisedIncomeGroup8DF,
        "deadIncomeGroup1DF": deadIncomeGroup1DF,
        "deadIncomeGroup2DF": deadIncomeGroup2DF,
        "deadIncomeGroup3DF": deadIncomeGroup3DF,
        "deadIncomeGroup4DF": deadIncomeGroup4DF,
        "deadIncomeGroup5DF": deadIncomeGroup5DF,
        "deadIncomeGroup6DF": deadIncomeGroup6DF,
        "deadIncomeGroup7DF": deadIncomeGroup7DF,
        "deadIncomeGroup8DF": deadIncomeGroup8DF,
    }
    # endregion Create dictionaries for dataframes

    # endregion dataframes

    if mode == "All cases":
        return time_all(dates, actualDF, infectedDF, hospitalizedDF, deadDF)
    elif mode == "By Gender":
        return time_gender(dates, gender_dict)
    elif mode == "By Age":
        return time_age(dates, age_dict)
    elif mode == "By Race":
        return time_race(dates, race_dict)
    elif mode == "By Ethnicity":
        return time_ethinic(dates, ethnic_dict)
    elif mode == "By Income":
        return time_income(dates, income_dict)


# @cache.memoize(timeout=CACHE_TIMEOUT)
def time_all(dates, actualDF, infectedDF, hospitalizedDF, deaddDF):
    fig = go.Figure()

    sub_groups = [
        "Cases",  # 'Actual cases',
        "Hospitalizations",  # 'Actual admissions',
        "Deaths",
        # '(Accumulated) Deaths', #'Actual deaths'
    ]
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=sub_groups,
        shared_xaxes=True,
        # specs=[[{"secondary_y": True}],[{"secondary_y": True}],[{"secondary_y": True}]],
        vertical_spacing=0.1,
        row_width=[0.2, 0.2, 0.2],
    )

    # region subplot 1,1 - cases
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=infectedDF["maximum"],
            name="Simul. Cases Maximum",
            line=dict(width=1, color="orange"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=infectedDF["minimum"],
            fill="tonexty",
            fillcolor="orange",
            name="Simul. Cases Minimum",
            line=dict(width=1, color="orange"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=infectedDF["meandata"],
            name="Simul. Cases Mean",
            line=dict(width=1, color="white"),
        ),
        row=1,
        col=1,
    )
    # endregion subplot 1,1 - cases

    # region subplot 2,1 - hospitalizations
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=hospitalizedDF["maximum"],
            name="Simul. Hospitalizations Maximum",
            line=dict(width=1, color="green"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=hospitalizedDF["minimum"],
            fill="tonexty",
            fillcolor="green",
            name="Simul. Hospitalizations Minimum",
            line=dict(width=1, color="green"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=hospitalizedDF["meandata"],
            name="Simul. Cases Mean",
            line=dict(width=1, color="white"),
        ),
        row=2,
        col=1,
    )
    # endregion subplot 2,1 - hospitalizations

    # region subplot 3,1 - deaths
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=deaddDF["maximum"],
            name="Simul. Deaths Maximum",
            line=dict(width=1, color="grey"),
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=deaddDF["minimum"],
            fill="tonexty",
            fillcolor="grey",
            name="Simul. Deaths Minimum",
            line=dict(width=1, color="grey"),
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=deaddDF["meandata"],
            name="Simul. Deaths Mean",
            line=dict(width=1, color="white"),
        ),
        row=3,
        col=1,
    )
    # endregion subplot 3,1 - deaths

    # region helper lines
    fig.add_vline(
        x=datetime.now().timestamp() * 1000,
        line_width=1.5,
        line_dash="dot",
        line_color="black",
        annotation_text="Today ",
        annotation=dict(font_size=12, font_family="Times New Roman"),
        annotation_font_color="purple",
        annotation_position="top left",
    )
    # endregion helper lines

    # region axis labels
    # show x ticks as Month and Year
    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        ticklabelmode="period",
        dtick="M1",
        tickangle=45,
    )

    fig.update_yaxes(showline=True, linewidth=1, linecolor="black", mirror=True)
    fig.update_layout(
        # showlegend=True,
        showlegend=False,
        autosize=False,
        # legend={'traceorder':'normal', 'x':1.02, 'y':0.7},
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
        font=dict(family="Arial", size=9),
    )

    fig.update_xaxes(range=[START_DATE, END_DATE], row=1, col=1)

    # forecast - set y axis range to match dashboard figures
    fig.update_yaxes(range=[0, 2000], row=1, col=1)
    fig.update_yaxes(range=[0, 500], row=2, col=1)
    fig.update_yaxes(range=[0, 75], row=3, col=1)
    # endregion axis labels
    return fig


# @cache.memoize(timeout=CACHE_TIMEOUT)
def time_gender(dates, gender_dict):
    sub_groups = [
        "Cases (male)",
        "Hospitalizations(male)",
        "Deaths(male)",
        "Cases (female)",
        "Hospitalizations(female)",
        "Deaths(female)",
    ]

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=sub_groups,
        shared_xaxes=True,
        # shared_yaxes=True,  # use same y axis range
        # vertical_spacing=0.1,
        # horizontal_spacing=0.1,
        row_width=[0.5, 0.5],
        column_width=[0.5, 0.5, 0.5],
    )

    # region gender group 1
    # region subplot 1,1 - infected for gender group 1
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=gender_dict["infectedGenderGroup1DF"]["meandata"],
            name="Simul. Cases Mean",
            line=dict(width=2, color="orange"),
        ),
        row=1,
        col=1,
    )
    # endregion subplot 1,1 - infected for gender group 1

    # region subplot 1,2 - hospitalized for gender group 1
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=gender_dict["hospitalisedGenderGroup1DF"]["meandata"],
            name="Simul. Hospitalized Mean",
            line=dict(width=2, color="blue"),
        ),
        row=1,
        col=2,
    )
    # endregion subplot 1,2 - hospitalized for gender group 1

    # region subplot 1,3 - dead for gender group 1
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=gender_dict["deadGenderGroup1DF"]["meandata"],
            name="Simul. Dead Mean",
            line=dict(width=2, color="red"),
        ),
        row=1,
        col=3,
    )
    # endregion subplot 3,1 - dead for gender group 1
    # endregion gender group 1

    # region gender group 2
    # region subplot 2,1 - infected for gender group 2
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=gender_dict["infectedGenderGroup2DF"]["meandata"],
            name="Simul. Cases Mean",
            line=dict(width=2, color="orange"),
        ),
        row=2,
        col=1,
    )
    # endregion subplot 2,1 - infected for gender group 2

    # region subplot 2,2 - hospitalized for gender group 2
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=gender_dict["hospitalisedGenderGroup2DF"]["meandata"],
            name="Simul. Hospitalized Mean",
            line=dict(width=2, color="blue"),
        ),
        row=2,
        col=2,
    )
    # endregion subplot 2,2 - hospitalized for gender group 2

    # region subplot 2,3 - dead for gender group 2
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=gender_dict["deadGenderGroup2DF"]["meandata"],
            name="Simul. Dead Mean",
            line=dict(width=2, color="red"),
        ),
        row=2,
        col=3,
    )
    # endregion subplot 2,3 - dead for gender group 2
    # endregion gender group 1

    # region axis labels
    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        ticklabelmode="period",
        dtick="M1",
    )
    fig.update_yaxes(showline=True, linewidth=1, linecolor="black", mirror=True)
    fig.update_layout(
        # showlegend=True,
        showlegend=False,
        legend=dict(traceorder="normal"),
        autosize=False,
        font=dict(family="Arial", size=12),
    )
    fig.update_xaxes(dtick="M2", tickformat="%b %Y")

    fig.update_xaxes(range=[START_DATE, END_DATE])

    # forecast - set y axis range to match dashboard figures
    fig.update_yaxes(range=[0, 500], col=1)
    fig.update_yaxes(range=[0, 100], col=2)
    fig.update_yaxes(range=[0, 20], col=3)

    # endregion axis labels

    return fig


# @cache.memoize(timeout=CACHE_TIMEOUT)
def time_age(dates, age_dict):
    sub_groups = [
        "Cases by ages",
        "Hospitalizations by ages",
        "Deaths by ages",
    ]

    fig = make_subplots(
        rows=9,
        cols=3,
        subplot_titles=sub_groups,
        shared_xaxes=True,
        shared_yaxes="columns",
        # shared_yaxes=True,  # use same y axis range
        # vertical_spacing=0.1,
        # horizontal_spacing=0.1,
        # row_width=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        # column_width=[0.5, 0.5, 0.5],
    )

    # region age group 1
    # region subplot 1,1 - infected for age group 1
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=age_dict["infectedAgeGroup1DF"]["meandata"],
            name="Simul. Cases Mean",
            line=dict(width=2, color="orange"),
        ),
        row=1,
        col=1,
    )
    # endregion subplot 1,1 - infected for age group 1

    # region subplot 1,2 - hospitalized for age group 1
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=age_dict["hospitalisedAgeGroup1DF"]["meandata"],
            name="Simul. Hospitalized Mean",
            line=dict(width=2, color="blue"),
        ),
        row=1,
        col=2,
    )
    # endregion subplot 1,2 - hospitalized for age group 1

    # region subplot 1,3 - dead for age group 1
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=age_dict["deadAgeGroup1DF"]["meandata"],
            name="Simul. Dead Mean",
            line=dict(width=2, color="red"),
        ),
        row=1,
        col=3,
    )
    # endregion subplot 1,3 - dead for age group 1
    # endregion age group 1

    # region age group 2
    # region subplot 2,1 - infected for age group 2
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=age_dict["infectedAgeGroup2DF"]["meandata"],
            name="Simul. Cases Mean",
            line=dict(width=2, color="orange"),
        ),
        row=2,
        col=1,
    )
    # endregion subplot 2,1 - infected for age group 2

    # region subplot 2,2 - hospitalized for age group 2
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=age_dict["hospitalisedAgeGroup2DF"]["meandata"],
            name="Simul. Hospitalized Mean",
            line=dict(width=2, color="blue"),
        ),
        row=2,
        col=2,
    )
    # endregion subplot 2,2 - hospitalized for age group 2

    # region subplot 2,3 - dead for age group 2
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=age_dict["deadAgeGroup2DF"]["meandata"],
            name="Simul. Dead Mean",
            line=dict(width=2, color="red"),
        ),
        row=2,
        col=3,
    )
    # endregion subplot 2,3 - dead for age group 2
    # endregion age group 2

    # region age group 3
    # region subplot 3,1 - infected for age group 3
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=age_dict["infectedAgeGroup3DF"]["meandata"],
            name="Simul. Cases Mean",
            line=dict(width=2, color="orange"),
        ),
        row=3,
        col=1,
    )
    # endregion subplot 3,1 - infected for age group 3

    # region subplot 3,2 - hospitalized for age group 3
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=age_dict["hospitalisedAgeGroup3DF"]["meandata"],
            name="Simul. Hospitalized Mean",
            line=dict(width=2, color="blue"),
        ),
        row=3,
        col=2,
    )
    # endregion subplot 3,2 - hospitalized for age group 3

    # region subplot 3,3 - dead for age group 3
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=age_dict["deadAgeGroup3DF"]["meandata"],
            name="Simul. Dead Mean",
            line=dict(width=2, color="red"),
        ),
        row=3,
        col=3,
    )
    # endregion subplot 3,3 - dead for age group 3
    # endregion age group 3

    # region age group 4
    # region subplot 4,1 - infected for age group 4
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=age_dict["infectedAgeGroup4DF"]["meandata"],
            name="Simul. Cases Mean",
            line=dict(width=2, color="orange"),
        ),
        row=4,
        col=1,
    )
    # endregion subplot 4,1 - infected for age group 4

    # region subplot 4,2 - hospitalized for age group 4
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=age_dict["hospitalisedAgeGroup4DF"]["meandata"],
            name="Simul. Hospitalized Mean",
            line=dict(width=2, color="blue"),
        ),
        row=4,
        col=2,
    )
    # endregion subplot 4,2 - hospitalized for age group 4

    # region subplot 4,3 - dead for age group 4
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=age_dict["deadAgeGroup4DF"]["meandata"],
            name="Simul. Dead Mean",
            line=dict(width=2, color="red"),
        ),
        row=4,
        col=3,
    )
    # endregion subplot 4,3 - dead for age group 4
    # endregion age group 4

    # region age group 5
    # region subplot 5,1 - infected for age group 5
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=age_dict["infectedAgeGroup5DF"]["meandata"],
            name="Simul. Cases Mean",
            line=dict(width=2, color="orange"),
        ),
        row=5,
        col=1,
    )
    # endregion subplot 5,1 - infected for age group 5

    # region subplot 5,2 - hospitalized for age group 5
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=age_dict["hospitalisedAgeGroup5DF"]["meandata"],
            name="Simul. Hospitalized Mean",
            line=dict(width=2, color="blue"),
        ),
        row=5,
        col=2,
    )
    # endregion subplot 5,2 - hospitalized for age group 5

    # region subplot 5,3 - dead for age group 5
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=age_dict["deadAgeGroup5DF"]["meandata"],
            name="Simul. Dead Mean",
            line=dict(width=2, color="red"),
        ),
        row=5,
        col=3,
    )
    # endregion subplot 5,3 - dead for age group 5
    # endregion age group 5

    # region age group 6
    # region subplot 6,1 - infected for age group 6
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=age_dict["infectedAgeGroup6DF"]["meandata"],
            name="Simul. Cases Mean",
            line=dict(width=2, color="orange"),
        ),
        row=6,
        col=1,
    )
    # endregion subplot 6,1 - infected for age group 6

    # region subplot 6,2 - hospitalized for age group 6
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=age_dict["hospitalisedAgeGroup6DF"]["meandata"],
            name="Simul. Hospitalized Mean",
            line=dict(width=2, color="blue"),
        ),
        row=6,
        col=2,
    )
    # endregion subplot 6,2 - hospitalized for age group 6

    # region subplot 6,3 - dead for age group 6
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=age_dict["deadAgeGroup6DF"]["meandata"],
            name="Simul. Dead Mean",
            line=dict(width=2, color="red"),
        ),
        row=6,
        col=3,
    )
    # endregion subplot 6,3 - dead for age group 6
    # endregion age group 6

    # region age group 7
    # region subplot 7,1 - infected for age group 7
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=age_dict["infectedAgeGroup7DF"]["meandata"],
            name="Simul. Cases Mean",
            line=dict(width=2, color="orange"),
        ),
        row=7,
        col=1,
    )
    # endregion subplot 7,1 - infected for age group 7

    # region subplot 7,2 - hospitalized for age group 7
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=age_dict["hospitalisedAgeGroup7DF"]["meandata"],
            name="Simul. Hospitalized Mean",
            line=dict(width=2, color="blue"),
        ),
        row=7,
        col=2,
    )
    # endregion subplot 7,2 - hospitalized for age group 7

    # region subplot 7,3 - dead for age group 7
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=age_dict["deadAgeGroup7DF"]["meandata"],
            name="Simul. Dead Mean",
            line=dict(width=2, color="red"),
        ),
        row=7,
        col=3,
    )
    # endregion subplot 7,3 - dead for age group 7
    # endregion age group 7

    # region age group 8
    # region subplot 8,1 - infected for age group 8,1
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=age_dict["infectedAgeGroup8DF"]["meandata"],
            name="Simul. Cases Mean",
            line=dict(width=2, color="orange"),
        ),
        row=8,
        col=1,
    )
    # endregion subplot 8,1 - infected for age group 8,1

    # region subplot 8,2 - hospitalized for age group 8,1
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=age_dict["hospitalisedAgeGroup8DF"]["meandata"],
            name="Simul. Hospitalized Mean",
            line=dict(width=2, color="blue"),
        ),
        row=8,
        col=2,
    )
    # endregion subplot 8,2 - hospitalized for age group 8,1

    # region subplot 8,3 - dead for age group 8,1
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=age_dict["deadAgeGroup8DF"]["meandata"],
            name="Simul. Dead Mean",
            line=dict(width=2, color="red"),
        ),
        row=8,
        col=3,
    )
    # endregion subplot 8,3 - dead for age group 8,1
    # endregion age group 8

    # region age group 9
    # region subplot 9,1 - infected for age group 9
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=age_dict["infectedAgeGroup9DF"]["meandata"],
            name="Simul. Cases Mean",
            line=dict(width=2, color="orange"),
        ),
        row=9,
        col=1,
    )
    # endregion subplot 9,1 - infected for age group 9

    # region subplot 9,2 - hospitalized for age group 9
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=age_dict["hospitalisedAgeGroup9DF"]["meandata"],
            name="Simul. Hospitalized Mean",
            line=dict(width=2, color="blue"),
        ),
        row=9,
        col=2,
    )
    # endregion subplot 9,2 - hospitalized for age group 9

    # region subplot 9,3 - dead for age group 9
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=age_dict["deadAgeGroup9DF"]["meandata"],
            name="Simul. Dead Mean",
            line=dict(width=2, color="red"),
        ),
        row=9,
        col=3,
    )
    # endregion subplot 9,3 - dead for age group 9
    # endregion age group 9

    # region axis labels
    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        ticklabelmode="period",
        dtick="M1",
    )
    fig.update_yaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
    )
    fig.update_layout(
        showlegend=False,
        autosize=False,
        legend=dict(traceorder="normal"),
        # width=1000, height=800,
        # legend=dict(orientation="h",x=0, y=-0.1, traceorder="normal"),
        # legend=dict(orientation="h"),
        font=dict(family="Arial", size=11),
    )
    fig.update_xaxes(dtick="M3", tickformat="%b %Y")

    fig.update_xaxes(range=[START_DATE, END_DATE])

    # forecast - set y axis range to match dashboard figures
    fig.update_yaxes(range=[0, 150], col=1)
    fig.update_yaxes(range=[0, 75], col=2)
    fig.update_yaxes(range=[0, 15], col=3)

    # endregion axis labels

    return fig


# @cache.memoize(timeout=CACHE_TIMEOUT)
def time_race(dates, race_dict):
    sub_groups = [
        "Cases (White)",
        "Hospitalizations (White)",
        "Deaths (White)",
        "Cases (African American)",
        "Hospitalizations (African American)",
        "Deaths (African American)",
        "Cases (Asian)",
        "Hospitalizations (Asian)",
        "Deaths (Asian)",
        "Cases (Hawaiian/Pacific Islander)",
        "Hospitalizations (Hawaiian/Pacific Islander)",
        "Deaths (Hawaiian/Pacific Islander)",
        "Cases (Native American)",
        "Hospitalizations (Native American)",
        "Deaths (Native American)",
        "Cases (Some other race)",
        "Hospitalizations (Some other race)",
        "Deaths (Some other race)",
        "Cases (Two or more races)",
        "Hospitalizations (Two or more races)",
        "Deaths (Two or more races)",
    ]
    fig = make_subplots(
        rows=7,
        cols=3,
        subplot_titles=sub_groups,
        shared_xaxes=True,
        shared_yaxes="columns",  # use same y axis range
        vertical_spacing=0.05,
        horizontal_spacing=0.15,
        row_width=[0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
    )

    # region race group 1
    # region subplot 1,1 - infected for race group 1
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=race_dict["infectedRaceGroup1DF"]["meandata"],
            name="Simul. Cases Mean",
            line=dict(width=2, color="orange"),
        ),
        row=1,
        col=1,
    )
    # endregion subplot 1,1 - infected for race group 1

    # region subplot 1,2 - hospitalized for race group 1
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=race_dict["hospitalisedRaceGroup1DF"]["meandata"],
            name="Simul. Hospitalized Mean",
            line=dict(width=2, color="blue"),
        ),
        row=1,
        col=2,
    )
    # endregion subplot 1,2 - hospitalized for race group 1

    # region subplot 1,3 - dead for race group 1
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=race_dict["deadRaceGroup1DF"]["meandata"],
            name="Simul. Dead Mean",
            line=dict(width=2, color="red"),
        ),
        row=1,
        col=3,
    )
    # endregion subplot 1,3 - dead for race group 1

    # endregion race group 1

    # region race group 2
    # region subplot 2,1 - infected for race group 2
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=race_dict["infectedRaceGroup2DF"]["meandata"],
            name="Simul. Cases Mean",
            line=dict(width=2, color="orange"),
        ),
        row=2,
        col=1,
    )
    # endregion subplot 2,1 - infected for race group 2

    # region subplot 2,2 - hospitalized for race group 2
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=race_dict["hospitalisedRaceGroup2DF"]["meandata"],
            name="Simul. Hospitalized Mean",
            line=dict(width=1, color="blue"),
        ),
        row=2,
        col=2,
    )
    # endregion subplot 2,2 - hospitalized for race group 2

    # region subplot 2,3 - dead for race group 2
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=race_dict["deadRaceGroup2DF"]["meandata"],
            name="Simul. Dead Mean",
            line=dict(width=2, color="red"),
        ),
        row=2,
        col=3,
    )
    # endregion subplot 2,3 - dead for race group 2

    # endregion race group 2

    # region race group 3
    # region subplot 3,1 - infected for race group 3
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=race_dict["infectedRaceGroup3DF"]["meandata"],
            name="Simul. Cases Mean",
            line=dict(width=2, color="orange"),
        ),
        row=3,
        col=1,
    )
    # endregion subplot 3,1 - infected for race group 3

    # region subplot 3,2 - hospitalized for race group 3
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=race_dict["hospitalisedRaceGroup3DF"]["meandata"],
            name="Simul. Hospitalized Mean",
            line=dict(width=2, color="blue"),
        ),
        row=3,
        col=2,
    )
    # endregion subplot 3,2 - hospitalized for race group 3

    # region subplot 3,3 - dead for race group 3
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=race_dict["deadRaceGroup3DF"]["meandata"],
            name="Simul. Dead Mean",
            line=dict(width=2, color="red"),
        ),
        row=3,
        col=3,
    )
    # endregion subplot 3,3 - dead for race group 3

    # endregion race group 3

    # region race group 4
    # region subplot 4,1 - infected for race group 4
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=race_dict["infectedRaceGroup4DF"]["meandata"],
            name="Simul. Cases Mean",
            line=dict(width=2, color="orange"),
        ),
        row=4,
        col=1,
    )
    # endregion subplot 4,1 - infected for race group 4

    # region subplot 4,2 - hospitalized for race group 4
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=race_dict["hospitalisedRaceGroup4DF"]["meandata"],
            name="Simul. Hospitalized Mean",
            line=dict(width=2, color="blue"),
        ),
        row=4,
        col=2,
    )
    # endregion subplot 4,2 - hospitalized for race group 4

    # region subplot 4,3 - dead for race group 4
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=race_dict["deadRaceGroup4DF"]["meandata"],
            name="Simul. Dead Mean",
            line=dict(width=2, color="red"),
        ),
        row=4,
        col=3,
    )
    # endregion subplot 4,3 - dead for race group 4

    # endregion race group 4

    # region race group 5
    # region subplot 5,1 - infected for race group 5
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=race_dict["infectedRaceGroup5DF"]["meandata"],
            name="Simul. Cases Mean",
            line=dict(width=2, color="orange"),
        ),
        row=5,
        col=1,
    )
    # endregion subplot 5,1 - infected for race group 5

    # region subplot 5,2 - hospitalized for race group 5
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=race_dict["hospitalisedRaceGroup5DF"]["meandata"],
            name="Simul. Hospitalized Mean",
            line=dict(width=2, color="blue"),
        ),
        row=5,
        col=2,
    )
    # endregion subplot 5,2 - hospitalized for race group 5

    # region subplot 5,3 - dead for race group 5
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=race_dict["deadRaceGroup5DF"]["meandata"],
            name="Simul. Dead Mean",
            line=dict(width=2, color="red"),
        ),
        row=5,
        col=3,
    )
    # endregion subplot 5,3 - dead for race group 5

    # endregion race group 5

    # region race group 6
    # region subplot 6,1 - infected for race group 6
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=race_dict["infectedRaceGroup6DF"]["meandata"],
            name="Simul. Cases Mean",
            line=dict(width=2, color="orange"),
        ),
        row=6,
        col=1,
    )
    # endregion subplot 6,1 - infected for race group 6

    # region subplot 6,2 - hospitalized for race group 6
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=race_dict["hospitalisedRaceGroup6DF"]["meandata"],
            name="Simul. Hospitalized Mean",
            line=dict(width=2, color="blue"),
        ),
        row=6,
        col=2,
    )
    # endregion subplot 6,2 - hospitalized for race group 6

    # region subplot 6,3 - dead for race group 6
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=race_dict["deadRaceGroup6DF"]["meandata"],
            name="Simul. Dead Mean",
            line=dict(width=2, color="red"),
        ),
        row=6,
        col=3,
    )
    # endregion subplot 6,3 - dead for race group 6

    # endregion race group 6

    # region race group 7
    # region subplot 7,1 - infected for race group 7
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=race_dict["infectedRaceGroup7DF"]["meandata"],
            name="Simul. Cases Mean",
            line=dict(width=2, color="orange"),
        ),
        row=7,
        col=1,
    )
    # endregion subplot 7,1 - infected for race group 7

    # region subplot 7,2 - hospitalized for race group 7
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=race_dict["hospitalisedRaceGroup7DF"]["meandata"],
            name="Simul. Hospitalized Mean",
            line=dict(width=2, color="blue"),
        ),
        row=7,
        col=2,
    )
    # endregion subplot 7,2 - hospitalized for race group 7

    # region subplot 7,3 - dead for race group 7
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=race_dict["deadRaceGroup7DF"]["meandata"],
            name="Simul. Dead Mean",
            line=dict(width=2, color="red"),
        ),
        row=7,
        col=3,
    )
    # endregion subplot 7,3 - dead for race group 7

    # endregion race group 7

    # region axis labels
    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        ticklabelmode="period",
        dtick="M1",
    )
    fig.update_yaxes(showline=True, linewidth=1, linecolor="black", mirror=True)
    fig.update_layout(
        showlegend=False,
        autosize=False,
        legend=dict(traceorder="normal"),
        # width=1000, height=1200,
        # legend=dict(orientation="h",x=0, y=-0.16, traceorder="normal"),
        # legend=dict(orientation="h"),
        font=dict(family="Arial", size=11),
    )
    fig.update_xaxes(dtick="M3", tickformat="%b %Y")

    fig.update_xaxes(range=[START_DATE, END_DATE])

    # forecast - set y axis range to match dashboard figures
    fig.update_yaxes(range=[0, 500], row=1, col=1)
    fig.update_yaxes(range=[0, 100], row=1, col=2)
    fig.update_yaxes(range=[0, 25], row=1, col=3)

    # endregion axis labels

    return fig


# @cache.memoize(timeout=CACHE_TIMEOUT)
def time_ethinic(dates, ethnic_dict):
    sub_groups = [
        "Cases (Hispanic)",
        "Hospitalizations (Hispanic)",
        "Deaths (Hispanic)",
        "Cases (Non-hispanic)",
        "Hospitalizations (Non-hispanic)",
        "Deaths (Non-hispanic)",
    ]
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=sub_groups,
        shared_xaxes=True,
        shared_yaxes="columns",
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
        row_width=[0.5, 0.5],
        column_width=[0.5, 0.5, 0.5],
    )

    # region ethnic group 1
    # region subplot 1,1 - infected for ethnic group 1
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=ethnic_dict["infectedEthnicityGroup1DF"]["meandata"],
            name="Simul. Cases Mean",
            line=dict(width=2, color="orange"),
        ),
        row=1,
        col=1,
    )
    # endregion subplot 1,1 - infected for ethnic group 1

    # region subplot 1,2 - hospitalized for ethnic group 1
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=ethnic_dict["hospitalisedEthnicityGroup1DF"]["meandata"],
            name="Simul. Hospitalized Mean",
            line=dict(width=2, color="blue"),
        ),
        row=1,
        col=2,
    )
    # endregion subplot 1,2 - hospitalized for ethnic group 1

    # region subplot 1,3 - dead for ethnic group 1
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=ethnic_dict["deadEthnicityGroup1DF"]["meandata"],
            name="Simul. Dead Mean",
            line=dict(width=2, color="red"),
        ),
        row=1,
        col=3,
    )
    # endregion subplot 3,1 - dead for ethnic group 1
    # endregion ethnic group 1

    # region ethnic group 2
    # region subplot 2,1 - infected for ethnic group 2
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=ethnic_dict["infectedEthnicityGroup2DF"]["meandata"],
            name="Simul. Cases Mean",
            line=dict(width=2, color="orange"),
        ),
        row=2,
        col=1,
    )
    # endregion subplot 2,1 - infected for ethnic group 2

    # region subplot 2,2 - hospitalized for ethnic group 22,
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=ethnic_dict["hospitalisedEthnicityGroup2DF"]["meandata"],
            name="Simul. Hospitalized Mean",
            line=dict(width=2, color="blue"),
        ),
        row=2,
        col=2,
    )
    # endregion subplot 2,2 - hospitalized for ethnic group 2

    # region subplot 2,3 - dead for ethnic group 2
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=ethnic_dict["deadEthnicityGroup2DF"]["meandata"],
            name="Simul. Dead Mean",
            line=dict(width=2, color="red"),
        ),
        row=2,
        col=3,
    )
    # endregion subplot 2,3 - dead for ethnic group 2
    # endregion ethnic group 2

    # region axis labels
    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        ticklabelmode="period",
        dtick="M1",
    )
    fig.update_yaxes(showline=True, linewidth=1, linecolor="black", mirror=True)
    fig.update_layout(
        showlegend=False,
        autosize=False,
        legend=dict(traceorder="normal"),
        # width=1000, height=1200,
        # legend=dict(orientation="h",x=0, y=-0.16, traceorder="normal"),
        # legend=dict(orientation="h"),
        font=dict(family="Arial", size=11),
    )
    fig.update_xaxes(dtick="M3", tickformat="%b %Y")

    fig.update_xaxes(range=[START_DATE, END_DATE])

    # forecast - set y axis range to match dashboard figures
    fig.update_yaxes(range=[0, 500], row=1, col=1)
    fig.update_yaxes(range=[0, 100], row=1, col=2)
    fig.update_yaxes(range=[0, 25], row=1, col=3)

    # endregion axis labels

    return fig


# @cache.memoize(timeout=CACHE_TIMEOUT)
def time_income(dates, income_dict):
    sub_groups = [
        "Cases(<$25000)",
        "Hospitalizations(<$25000)",
        "Deaths(<$25000)",
        "Cases($25000 ~ $34999)",
        "Hospitalizations($25000 ~ $34999)",
        "Deaths($25000 ~ $34999)",
        "Cases($35000 ~ $49999)",
        "Hospitalizations($35000 ~ $49999)",
        "Deaths($35000 ~ $49999)",
        "Cases($50000 ~ $74999)",
        "Hospitalizations($50000 ~ $74999)",
        "Deaths($50000 ~ $74999)",
        "Cases($75000 ~ $99999)",
        "Hospitalizations($75000 ~ $99999)",
        "Deaths($75000 ~ $99999)",
        "Cases($100000 ~ $124999)",
        "Hospitalizations($100000 ~ $124999)",
        "Deaths($100000 ~ $124999)",
        "Cases($125000 ~ $149999)",
        "Hospitalizations($125000 ~ $149999)",
        "Deaths($125000 ~ $149999)",
        "Cases(>$150000)",
        "Hospitalizations(>$150000)",
        "Deaths(>$150000)",
    ]
    fig = make_subplots(
        rows=8,
        cols=3,
        subplot_titles=sub_groups,
        shared_xaxes=True,
        shared_yaxes="columns",  # use same y axis range
        vertical_spacing=0.05,
        horizontal_spacing=0.05,
        # column_width=[0.25, 0.25,0.25,0.25,0.25],
        row_width=[0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
    )

    # region income group 1
    # region subplot 1,1 - infected for income group 1
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=income_dict["infectedIncomeGroup1DF"]["meandata"],
            name="Simul. Cases Mean",
            line=dict(width=2, color="orange"),
        ),
        row=1,
        col=1,
    )
    # endregion subplot 1,1 - infected for income group 1

    # region subplot 1,2 - hospitalized for income group 1
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=income_dict["hospitalisedIncomeGroup1DF"]["meandata"],
            name="Simul. Hospitalized Mean",
            line=dict(width=2, color="blue"),
        ),
        row=1,
        col=2,
    )
    # endregion subplot 1,2 - hospitalized for income group 1

    # region subplot 1,3 - dead for income group 1
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=income_dict["deadIncomeGroup1DF"]["meandata"],
            name="Simul. Dead Mean",
            line=dict(width=2, color="red"),
        ),
        row=1,
        col=3,
    )
    # endregion subplot 1,3 - dead for income group 1
    # endregion income group 1

    # region income group 2
    # region subplot 2,1 - infected for income group 2
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=income_dict["infectedIncomeGroup2DF"]["meandata"],
            name="Simul. Cases Mean",
            line=dict(width=2, color="orange"),
        ),
        row=2,
        col=1,
    )
    # endregion subplot 2,1 - infected for income group 2

    # region subplot 2,2 - hospitalized for income group 2
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=income_dict["hospitalisedIncomeGroup2DF"]["meandata"],
            name="Simul. Hospitalized Mean",
            line=dict(width=2, color="blue"),
        ),
        row=2,
        col=2,
    )
    # endregion subplot 2,2 - hospitalized for income group 2

    # region subplot 2,3 - dead for income group 2
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=income_dict["deadIncomeGroup2DF"]["meandata"],
            name="Simul. Dead Mean",
            line=dict(width=2, color="red"),
        ),
        row=2,
        col=3,
    )
    # endregion subplot 2,3 - dead for income group 2

    # endregion income group 2

    # region income group 3
    # region subplot 3,1 - infected for income group 3
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=income_dict["infectedIncomeGroup3DF"]["meandata"],
            name="Simul. Cases Mean",
            line=dict(width=2, color="orange"),
        ),
        row=3,
        col=1,
    )
    # endregion subplot 3,1 - infected for income group 3

    # region subplot 3,2 - hospitalized for income group 3
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=income_dict["hospitalisedIncomeGroup3DF"]["meandata"],
            name="Simul. Hospitalized Mean",
            line=dict(width=2, color="blue"),
        ),
        row=3,
        col=2,
    )
    # endregion subplot 3,2 - hospitalized for income group 3

    # region subplot 3,3 - dead for income group 3
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=income_dict["deadIncomeGroup3DF"]["meandata"],
            name="Simul. Dead Mean",
            line=dict(width=2, color="red"),
        ),
        row=3,
        col=3,
    )
    # endregion subplot 3,3 - dead for income group 3
    # endregion income group 3

    # region income group 4
    # region subplot 4,1 - infected for income group 4
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=income_dict["infectedIncomeGroup4DF"]["meandata"],
            name="Simul. Cases Mean",
            line=dict(width=2, color="orange"),
        ),
        row=4,
        col=1,
    )
    # endregion subplot 4,1 - infected for income group 4

    # region subplot 4,2 - hospitalized for income group 4
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=income_dict["hospitalisedIncomeGroup4DF"]["meandata"],
            name="Simul. Hospitalized Mean",
            line=dict(width=2, color="blue"),
        ),
        row=4,
        col=2,
    )
    # endregion subplot 4,2 - hospitalized for income group 4

    # region subplot 4,3 - dead for income group 4
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=income_dict["deadIncomeGroup4DF"]["meandata"],
            name="Simul. Dead Mean",
            line=dict(width=2, color="red"),
        ),
        row=4,
        col=3,
    )
    # endregion subplot 4,3 - dead for income group 4
    # endregion income group 4

    # region income group 5
    # region subplot 5,1 - infected for income group 5
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=income_dict["infectedIncomeGroup5DF"]["meandata"],
            name="Simul. Cases Mean",
            line=dict(width=2, color="orange"),
        ),
        row=5,
        col=1,
    )
    # endregion subplot 5,1 - infected for income group 5

    # region subplot 5,2 - hospitalized for income group 5
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=income_dict["hospitalisedIncomeGroup5DF"]["meandata"],
            name="Simul. Hospitalized Mean",
            line=dict(width=2, color="blue"),
        ),
        row=5,
        col=2,
    )
    # endregion subplot 5,2 - hospitalized for income group 5

    # region subplot 5,3 - dead for income group 5
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=income_dict["deadIncomeGroup5DF"]["meandata"],
            name="Simul. Dead Mean",
            line=dict(width=2, color="red"),
        ),
        row=5,
        col=3,
    )
    # endregion subplot 5,3 - dead for income group 5

    # endregion income group 5

    # region income group 6
    # region subplot 6,1 - infected for income group 6
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=income_dict["infectedIncomeGroup6DF"]["meandata"],
            name="Simul. Cases Mean",
            line=dict(width=2, color="orange"),
        ),
        row=6,
        col=1,
    )
    # endregion subplot 6,1 - infected for income group 6

    # region subplot 6,2 - hospitalized for income group 6
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=income_dict["hospitalisedIncomeGroup6DF"]["meandata"],
            name="Simul. Hospitalized Mean",
            line=dict(width=2, color="blue"),
        ),
        row=6,
        col=2,
    )
    # endregion subplot 6,2 - hospitalized for income group 6

    # region subplot 6,3 - dead for income group 6
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=income_dict["deadIncomeGroup6DF"]["meandata"],
            name="Simul. Dead Mean",
            line=dict(width=2, color="red"),
        ),
        row=6,
        col=3,
    )
    # endregion subplot 6,3 - dead for income group 6

    # endregion income group 6

    # region income group 7
    # region subplot 7,1 - infected for income group 7
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=income_dict["infectedIncomeGroup7DF"]["meandata"],
            name="Simul. Cases Mean",
            line=dict(width=2, color="orange"),
        ),
        row=7,
        col=1,
    )
    # endregion subplot 7,1 - infected for income group 7

    # region subplot 7,2 - hospitalized for income group 7
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=income_dict["hospitalisedIncomeGroup7DF"]["meandata"],
            name="Simul. Hospitalized Mean",
            line=dict(width=2, color="blue"),
        ),
        row=7,
        col=2,
    )
    # endregion subplot 7,2 - hospitalized for income group 7

    # region subplot 7,3 - dead for income group 7
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=income_dict["deadIncomeGroup7DF"]["meandata"],
            name="Simul. Dead Mean",
            line=dict(width=2, color="red"),
        ),
        row=7,
        col=3,
    )
    # endregion subplot 7,3 - dead for income group 7

    # endregion income group 7

    # region income group 8
    # region subplot 8,1 - infected for income group 8
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=income_dict["infectedIncomeGroup8DF"]["meandata"],
            name="Simul. Cases Mean",
            line=dict(width=2, color="orange"),
        ),
        row=8,
        col=1,
    )
    # endregion subplot 8,1 - infected for income group 8

    # region subplot 8,2 - hospitalized for income group 8
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=income_dict["hospitalisedIncomeGroup8DF"]["meandata"],
            name="Simul. Hospitalized Mean",
            line=dict(width=2, color="blue"),
        ),
        row=8,
        col=2,
    )
    # endregion subplot 8,2 - hospitalized for income group 8

    # region subplot 8,3 - dead for income group 8
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=income_dict["deadIncomeGroup8DF"]["meandata"],
            name="Simul. Dead Mean",
            line=dict(width=2, color="red"),
        ),
        row=8,
        col=3,
    )
    # endregion subplot 8,3 - dead for income group 8

    # endregion income group 8

    # region axis labels
    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        ticklabelmode="period",
        dtick="M1",
    )
    fig.update_yaxes(showline=True, linewidth=1, linecolor="black", mirror=True)
    fig.update_layout(
        showlegend=False,
        autosize=False,
        # width=1000, height=800,
        legend=dict(traceorder="normal"),
        # legend=dict(orientation="h"),
        font=dict(family="Arial", size=11),
    )
    fig.update_xaxes(dtick="M2", tickformat="%b %Y")

    fig.update_xaxes(range=[START_DATE, END_DATE])

    # forecast - set y axis range to match dashboard figures
    fig.update_yaxes(range=[0, 500], row=1, col=1)
    fig.update_yaxes(range=[0, 100], row=1, col=2)
    fig.update_yaxes(range=[0, 25], row=1, col=3)

    # endregion axis labels

    return fig


# endregion Time Plots


# region Zip Code Time Plots
# @cache.memoize(timeout=CACHE_TIMEOUT)
def load_allzipcodes():
    # region dataframes
    csvContent = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file == "csvContent.csv":
                csvContent.append(pd.read_csv(os.path.join(path, "csvContent.csv")))

    # region csvContentDF
    csvContentDF = pd.concat(csvContent, axis=1)
    # endregion csvContentDF

    # Define dates
    dates = np.arange(START_DATE, END_DATE, timedelta(days=1)).astype(datetime)

    # region infectedZipDF
    # Create a new dataframe infectedZipDF
    infectedZipDF = csvContentDF[
        [
            "Time",
            "community",
            "infected_ethnicity_group_1",
            "infected_ethnicity_group_2",
        ]
    ]

    # Create a new column infectedZip.
    infectedZipDF["infectedZip"] = (
        infectedZipDF["infected_ethnicity_group_1"]
        + infectedZipDF["infected_ethnicity_group_2"]
    )
    infectedZipDF = infectedZipDF.drop(
        columns=["infected_ethnicity_group_1", "infected_ethnicity_group_2"]
    )

    # read the "wardNo_to_zipcode.csv" file into a dictionary
    wardNo_to_zipcode = (
        pd.read_csv(os.path.join(asset_path, "wardNo_to_zipcode.csv"))
        .set_index("wardNo")["zipcode"]
        .to_dict()
    )

    # Map the community column to zipcode using the dictionary
    infectedZipDF["zipcode"] = infectedZipDF["community"].map(wardNo_to_zipcode)
    infectedZipDF = infectedZipDF.drop(columns=["community"])
    infectedZipDF = infectedZipDF[pd.notnull(infectedZipDF["zipcode"])]
    infectedZipDF["zipcode"] = infectedZipDF["zipcode"].astype(int)
    infectedZipDF["zipcode"] = infectedZipDF["zipcode"].astype(str)

    # create a pivot table with zipcode as index, Time as columns and infectedZip as values
    infectedZipDF = (
        infectedZipDF.pivot(index="zipcode", columns="Time", values="infectedZip")
        .groupby(lambda x: int(x), axis=1)
        .mean()
    )

    # Create a date range starting from 2020-03-01 with a frequency of 1 day
    date_range = pd.date_range(
        start="2020-03-01", freq="D", periods=infectedZipDF.shape[1]
    )

    # Convert the Time column to datetime format and set it as the new column names
    infectedZipDF.columns = pd.to_datetime(date_range)
    # endregion infectedZipDF

    # region hospitalisedZipDF
    # Create a new dataframe hospitalisedZipDF
    hospitalisedZipDF = csvContentDF[
        [
            "Time",
            "community",
            "hospitalised_ethnicity_group_1",
            "hospitalised_ethnicity_group_2",
        ]
    ]

    # Create a new column hospitalisedZip.
    hospitalisedZipDF["hospitalisedZip"] = (
        hospitalisedZipDF["hospitalised_ethnicity_group_1"]
        + hospitalisedZipDF["hospitalised_ethnicity_group_2"]
    )
    hospitalisedZipDF = hospitalisedZipDF.drop(
        columns=["hospitalised_ethnicity_group_1", "hospitalised_ethnicity_group_2"]
    )

    # read the "wardNo_to_zipcode.csv" file into a dictionary
    wardNo_to_zipcode = (
        pd.read_csv(os.path.join(asset_path, "wardNo_to_zipcode.csv"))
        .set_index("wardNo")["zipcode"]
        .to_dict()
    )

    # Map the community column to zipcode using the dictionary
    hospitalisedZipDF["zipcode"] = hospitalisedZipDF["community"].map(wardNo_to_zipcode)
    hospitalisedZipDF = hospitalisedZipDF.drop(columns=["community"])
    hospitalisedZipDF = hospitalisedZipDF[pd.notnull(hospitalisedZipDF["zipcode"])]
    hospitalisedZipDF["zipcode"] = hospitalisedZipDF["zipcode"].astype(int)
    hospitalisedZipDF["zipcode"] = hospitalisedZipDF["zipcode"].astype(str)

    # create a pivot table with zipcode as index, Time as columns and hospitalisedZip as values
    hospitalisedZipDF = (
        hospitalisedZipDF.pivot(
            index="zipcode", columns="Time", values="hospitalisedZip"
        )
        .groupby(lambda x: int(x), axis=1)
        .mean()
    )

    # Create a date range starting from 2020-03-01 with a frequency of 1 day
    date_range = pd.date_range(
        start="2020-03-01", freq="D", periods=hospitalisedZipDF.shape[1]
    )

    # Convert the Time column to datetime format and set it as the new column names
    hospitalisedZipDF.columns = pd.to_datetime(date_range)
    # endregion hospitalisedZipDF

    # endregion dataframes

    sub_groups = [
        "Infected",
        "Hospitalised",
        # 'Deaths_'+str(days_interval)+'-day average',
    ]
    fig_subplots = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=sub_groups,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.08,
        row_width=[0.1, 0.1],
    )

    palette = cycle(px.colors.sequential.Plotly3)
    for zip in ZIPS:
        try:
            c = next(palette)
            fig_subplots.add_trace(
                go.Scatter(
                    x=dates,
                    y=infectedZipDF.loc[zip],
                    mode="lines",
                    name=zip,
                    text=zip,
                    legendgroup=zip,
                    line_shape="spline",
                    line=dict(color=c, width=0.75),
                ),
                row=1,
                col=1,
            )
            fig_subplots.add_trace(
                go.Scatter(
                    x=dates,
                    y=hospitalisedZipDF.loc[zip],
                    mode="lines",
                    name=zip,
                    text=zip,
                    line_shape="spline",
                    legendgroup=zip,
                    showlegend=False,
                    line=dict(color=c, width=0.75),
                ),
                row=2,
                col=1,
            )
        except KeyError:
            pass

    fig_subplots.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        ticklabelmode="period",
        dtick="M1",
    )
    fig_subplots.update_yaxes(
        showline=True, linewidth=1, linecolor="black", mirror=True
    )
    fig_subplots.update_layout(
        showlegend=True,
        autosize=True,
        # width=900, height=800,
        legend=dict(
            # orientation="h",
            x=1.05,
            y=1.0,
            # traceorder="normal"
        ),
        font=dict(family="Arial", size=10),
    )
    return fig_subplots


# endregion Zip Code Time Plots


# region Risk by zip codes with SVI
# @cache.memoize(timeout=CACHE_TIMEOUT)
def draw_risky_zipcodes():
    dfm = gpd.read_file(
        os.path.join(asset_path, "hillsborough-zipcodes-boundarymap.geojson")
    )
    riskzips_df = pd.read_parquet(
        os.path.join(asset_path, "risky_zipcodes_forecast.parquet"),
    )
    riskzips_df["date_str"] = riskzips_df["Date"].dt.strftime("%Y-%m-%d")

    riskzips_df = riskzips_df[riskzips_df["date_str"] <= END_DATE.strftime("%Y-%m-%d")]

    # forecast - cut off days before start date from riskzips_df
    riskzips_df = riskzips_df[
        riskzips_df["date_str"] >= START_DATE.strftime("%Y-%m-%d")
    ]

    fig = px.choropleth_mapbox(
        riskzips_df,
        geojson=dfm,
        locations="zipcode",
        color="symptomatic",
        featureidkey="properties.zipcode",
        color_continuous_scale="Viridis_r",
        mapbox_style="open-street-map",
        zoom=9,
        center={"lat": 27.91, "lon": -82.4},
        opacity=0.5,
        animation_frame="date_str",
        width=900,
        height=800,
    )

    return fig


# @cache.memoize(timeout=CACHE_TIMEOUT)
def basemap(label_selected):
    geojson = gpd.read_file(
        os.path.join(asset_path, "hillsborough-zipcodes-boundarymap.geojson")
    )
    fig = px.choropleth_mapbox(
        df_PR,
        title="Social Vulnerability Index (Hillsborough, FL)",
        geojson=geojson,
        locations="zipcode",
        labels="zipcode",
        # locations="zip_code",
        featureidkey="properties.zipcode",
        color=label_selected,
        color_continuous_scale="sunsetdark",
        range_color=[0, 1],
        color_continuous_midpoint=0.5,
        opacity=0.6,  # 0~1
        width=800,
        height=600,
    ).update_layout(
        mapbox={
            "accesstoken": mapbox_token,
            # "style": "carto-positron",
            # "style":"light",
            # "style":"streets",
            "style": "outdoors",
            # "zoom": 10,
            "zoom": 9,
            "center": {
                "lon": -82.4,
                "lat": 27.91,
            },
        },
        margin={"l": 10, "r": 0, "t": 40, "b": 10},
    )
    return fig


# @cache.memoize(timeout=CACHE_TIMEOUT)
def bar_zipcodes():
    plot_df = pd.read_parquet(
        os.path.join(asset_path, "risky_zipcodes_bar_forecast.parquet")
    )
    plot_df["zipcode groups"] = plot_df["color"].astype(str)
    plot_df.sort_values(by=["color", "zipcode groups"], inplace=True)

    # Filter the DataFrame based on the START_DATE and END_DATE
    plot_df = plot_df[(plot_df["Date"] >= START_DATE) & (plot_df["Date"] <= END_DATE)]

    # Convert 'Date' column to string
    plot_df["Date"] = plot_df["Date"].astype(str)

    zipcode_legend = {
        0: "33547, 33556, 33596, 33620, 33629",
        1: "33558, 33572, 33579, 33606, 33626",
        2: "33503, 33548, 33559, 33609, 33647",
        3: "33549, 33573, 33611, 33616, 33625",
        4: "33569, 33578, 33594, 33618, 33624, 33635",
        5: "33510, 33511, 33602, 33634, 33637",
        6: "33527, 33534, 33566, 33567, 33584",
        7: "33565, 33592, 33598, 33603, 33615",
        8: "33563, 33570, 33604, 33614, 33617",
        9: "33605, 33607, 33610, 33612, 33613, 33619",
    }

    # Add a new column 'zipcodes' to the plot_df DataFrame
    plot_df["zipcodes"] = plot_df["color"].map(zipcode_legend)

    fig = px.bar(
        plot_df,
        x="symptomatic",
        y="zipcode groups",
        animation_frame="Date",  # Make sure it's using the string column
        color="color",
        color_continuous_scale="balance",
        hover_data={
            "Date": False,
            "color": False,
            "zipcode groups": False,
            "zipcodes": True,
            "symptomatic": True,
        },  # Set the hover data, hide the 'color' column
    )
    fig.update_layout(
        yaxis_title="zipcode groups",
        coloraxis_showscale=True,
        coloraxis_colorbar=dict(
            tickvals=[1, 4.5, 8],
            ticktext=["low", "medium", "high"],
            title="Vulnerability",
        ),
    )
    fig.update_xaxes(range=[0, plot_df["symptomatic"].max()])

    return fig


# endregion Risk by zip codes with SVI


# region Spatial Temporal Patterns
# @cache.memoize(timeout=CACHE_TIMEOUT)
def draw_infected_map():
    mapDF = pd.read_parquet(os.path.join(asset_path, "spatial_infected.parquet"))
    mapDF["date_str"] = mapDF["date"].apply(lambda x: str(x))

    mapDF = mapDF[mapDF["date_str"] >= START_DATE.strftime("%Y-%m-%d")]
    mapDF = mapDF[mapDF["date_str"] <= END_DATE.strftime("%Y-%m-%d")]

    px.set_mapbox_access_token(mapbox_token)
    fig = px.scatter_mapbox(
        mapDF,
        lat=mapDF.lat,
        lon=mapDF.lon,
        # hover_name="date",
        # size="size",
        size_max=5,
        # color="Time_step",
        color_continuous_scale="haline",
        zoom=8,
        center={"lat": 27.91, "lon": -82.4},
        opacity=1,
        animation_frame="date_str",
        width=600,
        height=600,
    )
    geojson = gpd.read_file(
        os.path.join(asset_path, "hillsborough-zipcodes-boundarymap.geojson")
    )
    fig.update_layout(
        mapbox={
            "layers": [
                {
                    "source": geojson["geometry"].__geo_interface__,
                    "type": "fill",
                    "below": "traces",
                    "color": "#dedede",
                    "opacity": 0.2,
                }
            ]
        },
    )

    fig.update_coloraxes(showscale=False)

    return fig


# @cache.memoize(timeout=CACHE_TIMEOUT)
def draw_hospitalised_map():
    mapDF = pd.read_parquet(os.path.join(asset_path, "spatial_hospitalized.parquet"))
    mapDF["date_str"] = mapDF["date"].apply(lambda x: str(x))

    mapDF = mapDF[mapDF["date_str"] >= START_DATE.strftime("%Y-%m-%d")]
    mapDF = mapDF[mapDF["date_str"] <= END_DATE.strftime("%Y-%m-%d")]

    px.set_mapbox_access_token(mapbox_token)
    fig = px.scatter_mapbox(
        mapDF,
        lat=mapDF.lat,
        lon=mapDF.lon,
        # hover_name="date",
        # size="size",
        size_max=5,
        # color="Time_step",
        color_continuous_scale="OrRd",
        zoom=8,
        center={"lat": 27.91, "lon": -82.4},
        opacity=1,
        animation_frame="date_str",
        width=600,
        height=600,
        # mapbox_style='open-street-map'
    )
    geojson = gpd.read_file(
        os.path.join(asset_path, "hillsborough-zipcodes-boundarymap.geojson")
    )
    fig.update_layout(
        mapbox={
            "layers": [
                {
                    "source": geojson["geometry"].__geo_interface__,
                    "type": "fill",
                    "below": "traces",
                    "color": "#dedede",
                    #'color': "blue",
                    "opacity": 0.2,
                }
            ]
        }
    )
    fig.update_coloraxes(showscale=False)

    return fig


# @cache.memoize(timeout=CACHE_TIMEOUT)
def draw_dead_map():
    mapDF = pd.read_parquet(os.path.join(asset_path, "spatial_dead.parquet"))
    mapDF["date_str"] = mapDF["date"].apply(lambda x: str(x))

    mapDF = mapDF[mapDF["date_str"] >= START_DATE.strftime("%Y-%m-%d")]
    mapDF = mapDF[mapDF["date_str"] <= END_DATE.strftime("%Y-%m-%d")]

    px.set_mapbox_access_token(mapbox_token)
    fig = px.scatter_mapbox(
        mapDF,
        lat=mapDF.lat,
        lon=mapDF.lon,
        size_max=5,
        # color="Time_step",
        color_continuous_scale="gray",
        zoom=8,
        center={"lat": 27.91, "lon": -82.4},
        opacity=1,
        animation_frame="date_str",
        width=600,
        height=600,
        # mapbox_style='open-street-map'
    )
    geojson = gpd.read_file(
        os.path.join(asset_path, "hillsborough-zipcodes-boundarymap.geojson")
    )
    fig.update_layout(
        mapbox={
            "layers": [
                {
                    "source": geojson["geometry"].__geo_interface__,
                    "type": "fill",
                    "below": "traces",
                    "color": "#dedede",
                    # 'color': "red",
                    "opacity": 0.2,
                }
            ]
        }
    )
    fig.update_coloraxes(showscale=False)

    return fig


# @cache.memoize(timeout=CACHE_TIMEOUT)
def draw_recovered_map():
    mapDF = pd.read_parquet(os.path.join(asset_path, "spatial_recovered.parquet"))
    mapDF["date_str"] = mapDF["date"].apply(lambda x: str(x))

    mapDF = mapDF[mapDF["date_str"] >= START_DATE.strftime("%Y-%m-%d")]
    mapDF = mapDF[mapDF["date_str"] <= END_DATE.strftime("%Y-%m-%d")]

    px.set_mapbox_access_token(mapbox_token)
    fig = px.scatter_mapbox(
        mapDF,
        lat=mapDF.lat,
        lon=mapDF.lon,
        # hover_name="date",
        # size="size",
        size_max=5,
        # color="Time_step",
        color_continuous_scale="blugrn",
        zoom=8,
        center={"lat": 27.91, "lon": -82.4},
        opacity=1,
        animation_frame="date_str",
        width=600,
        height=600,
    )
    geojson = gpd.read_file(
        os.path.join(asset_path, "hillsborough-zipcodes-boundarymap.geojson")
    )
    fig.update_layout(
        mapbox={
            "layers": [
                {
                    "source": geojson["geometry"].__geo_interface__,
                    "type": "fill",
                    "below": "traces",
                    "color": "#dedede",
                    "opacity": 0.2,
                }
            ]
        }
    )
    fig.update_coloraxes(showscale=False)

    return fig


# @cache.memoize(timeout=CACHE_TIMEOUT)
def draw_vaccinated_map():
    mapDF = pd.read_parquet(os.path.join(asset_path, "spatial_vaccinated.parquet"))
    mapDF["date_str"] = mapDF["date"].apply(lambda x: str(x))

    mapDF = mapDF[mapDF["date_str"] >= START_DATE.strftime("%Y-%m-%d")]
    mapDF = mapDF[mapDF["date_str"] <= END_DATE.strftime("%Y-%m-%d")]

    px.set_mapbox_access_token(mapbox_token)
    mapDF["date_str"] = mapDF["date"].apply(lambda x: str(x))
    fig = px.scatter_mapbox(
        mapDF,
        lat=mapDF.lat,
        lon=mapDF.lon,
        # hover_name="date",
        # size="size",
        size_max=5,
        # color="Time_step",
        color_continuous_scale="plasma",
        zoom=8,
        center={"lat": 27.91, "lon": -82.4},
        opacity=1,
        animation_frame="date_str",
        width=600,
        height=600,
    )
    geojson = gpd.read_file(
        os.path.join(asset_path, "hillsborough-zipcodes-boundarymap.geojson")
    )
    fig.update_layout(
        mapbox={
            "layers": [
                {
                    "source": geojson["geometry"].__geo_interface__,
                    "type": "fill",
                    "below": "traces",
                    "color": "#dedede",
                    "opacity": 0.2,
                }
            ]
        }
    )
    fig.update_coloraxes(showscale=False)

    return fig


# endregion Spatial Temporal Patterns


# region Kernel


# @cache.memoize(timeout=CACHE_TIMEOUT)
def draw_infected_kernel():
    kernelDF = pd.read_parquet(os.path.join(asset_path, "spatial_infected.parquet"))
    kernelDF["date_str"] = kernelDF["date"].apply(lambda x: str(x))
    kernelDF = gpd.GeoDataFrame(
        kernelDF, geometry=gpd.points_from_xy(kernelDF.lon, kernelDF.lat)
    )
    kernelDF.set_crs(epsg=4326, inplace=True)

    # Create a Kernel Density Estimation plot
    fig = px.density_mapbox(
        kernelDF,
        lat=kernelDF.geometry.y,
        lon=kernelDF.geometry.x,
        z=kernelDF.index,  # dummy z values to ensure heatmap is plotted
        radius=10,
        center={"lat": 27.91, "lon": -82.4},
        zoom=8,
        mapbox_style="carto-positron",
        opacity=0.9,
        animation_frame="date_str",
        color_continuous_scale="YlOrRd",
        range_color=(0, 17),
        labels={"index": ""},
        width=600,
        height=600,
    )
    geojson = gpd.read_file(
        os.path.join(asset_path, "hillsborough-zipcodes-boundarymap.geojson")
    )
    fig.update_layout(
        mapbox={
            "layers": [
                {
                    "source": geojson["geometry"].__geo_interface__,
                    "type": "fill",
                    "below": "traces",
                    "color": "#dedede",
                    "opacity": 0.2,
                }
            ]
        }
    )

    return fig


# @cache.memoize(timeout=CACHE_TIMEOUT)
def draw_hospitalised_kernel():
    kernelDF = pd.read_parquet(os.path.join(asset_path, "spatial_hospitalized.parquet"))
    kernelDF["date_str"] = kernelDF["date"].apply(lambda x: str(x))
    kernelDF = gpd.GeoDataFrame(
        kernelDF, geometry=gpd.points_from_xy(kernelDF.lon, kernelDF.lat)
    )
    kernelDF.set_crs(epsg=4326, inplace=True)

    # Create a Kernel Density Estimation plot
    fig = px.density_mapbox(
        kernelDF,
        lat=kernelDF.geometry.y,
        lon=kernelDF.geometry.x,
        z=kernelDF.index,  # dummy z values to ensure heatmap is plotted
        radius=10,
        center={"lat": 27.91, "lon": -82.4},
        zoom=8,
        mapbox_style="carto-positron",
        opacity=0.9,
        animation_frame="date_str",
        color_continuous_scale="YlOrRd",
        range_color=(0, 17),
        labels={"index": ""},
        width=600,
        height=600,
    )
    geojson = gpd.read_file(
        os.path.join(asset_path, "hillsborough-zipcodes-boundarymap.geojson")
    )
    fig.update_layout(
        mapbox={
            "layers": [
                {
                    "source": geojson["geometry"].__geo_interface__,
                    "type": "fill",
                    "below": "traces",
                    "color": "#dedede",
                    "opacity": 0.2,
                }
            ]
        }
    )

    return fig


# @cache.memoize(timeout=CACHE_TIMEOUT)
def draw_dead_kernel():
    kernelDF = pd.read_parquet(os.path.join(asset_path, "spatial_dead.parquet"))
    kernelDF["date_str"] = kernelDF["date"].apply(lambda x: str(x))
    kernelDF = gpd.GeoDataFrame(
        kernelDF, geometry=gpd.points_from_xy(kernelDF.lon, kernelDF.lat)
    )
    kernelDF.set_crs(epsg=4326, inplace=True)

    # Create a Kernel Density Estimation plot
    fig = px.density_mapbox(
        kernelDF,
        lat=kernelDF.geometry.y,
        lon=kernelDF.geometry.x,
        z=kernelDF.index,  # dummy z values to ensure heatmap is plotted
        radius=10,
        center={"lat": 27.91, "lon": -82.4},
        zoom=8,
        mapbox_style="carto-positron",
        opacity=0.9,
        animation_frame="date_str",
        color_continuous_scale="YlOrRd",
        range_color=(0, 17),
        labels={"index": ""},
        width=600,
        height=600,
    )
    geojson = gpd.read_file(
        os.path.join(asset_path, "hillsborough-zipcodes-boundarymap.geojson")
    )
    fig.update_layout(
        mapbox={
            "layers": [
                {
                    "source": geojson["geometry"].__geo_interface__,
                    "type": "fill",
                    "below": "traces",
                    "color": "#dedede",
                    "opacity": 0.2,
                }
            ]
        }
    )

    return fig


# @cache.memoize(timeout=CACHE_TIMEOUT)
def draw_recovered_kernel():
    kernelDF = pd.read_parquet(os.path.join(asset_path, "spatial_recovered.parquet"))
    kernelDF["date_str"] = kernelDF["date"].apply(lambda x: str(x))
    kernelDF = gpd.GeoDataFrame(
        kernelDF, geometry=gpd.points_from_xy(kernelDF.lon, kernelDF.lat)
    )
    kernelDF.set_crs(epsg=4326, inplace=True)

    # Create a Kernel Density Estimation plot
    fig = px.density_mapbox(
        kernelDF,
        lat=kernelDF.geometry.y,
        lon=kernelDF.geometry.x,
        z=kernelDF.index,  # dummy z values to ensure heatmap is plotted
        radius=10,
        center={"lat": 27.91, "lon": -82.4},
        zoom=8,
        mapbox_style="carto-positron",
        opacity=0.9,
        animation_frame="date_str",
        color_continuous_scale="YlOrRd",
        range_color=(0, 17),
        labels={"index": ""},
        width=600,
        height=600,
    )
    geojson = gpd.read_file(
        os.path.join(asset_path, "hillsborough-zipcodes-boundarymap.geojson")
    )
    fig.update_layout(
        mapbox={
            "layers": [
                {
                    "source": geojson["geometry"].__geo_interface__,
                    "type": "fill",
                    "below": "traces",
                    "color": "#dedede",
                    "opacity": 0.2,
                }
            ]
        }
    )

    return fig


# @cache.memoize(timeout=CACHE_TIMEOUT)
def draw_vaccinated_kernel():
    kernelDF = pd.read_parquet(os.path.join(asset_path, "spatial_vaccinated.parquet"))
    kernelDF["date_str"] = kernelDF["date"].apply(lambda x: str(x))
    kernelDF = gpd.GeoDataFrame(
        kernelDF, geometry=gpd.points_from_xy(kernelDF.lon, kernelDF.lat)
    )
    kernelDF.set_crs(epsg=4326, inplace=True)

    # Create a Kernel Density Estimation plot
    fig = px.density_mapbox(
        kernelDF,
        lat=kernelDF.geometry.y,
        lon=kernelDF.geometry.x,
        z=kernelDF.index,  # dummy z values to ensure heatmap is plotted
        radius=10,
        center={"lat": 27.91, "lon": -82.4},
        zoom=8,
        mapbox_style="carto-positron",
        opacity=0.9,
        animation_frame="date_str",
        color_continuous_scale="YlOrRd",
        range_color=(0, 17),
        labels={"index": ""},
        width=600,
        height=600,
    )
    geojson = gpd.read_file(
        os.path.join(asset_path, "hillsborough-zipcodes-boundarymap.geojson")
    )
    fig.update_layout(
        mapbox={
            "layers": [
                {
                    "source": geojson["geometry"].__geo_interface__,
                    "type": "fill",
                    "below": "traces",
                    "color": "#dedede",
                    "opacity": 0.2,
                }
            ]
        }
    )

    return fig


# endregion Kernel

# endregion Graph functions

# region Styles
colors = {
    # 'background': '#111111',
    "text": "#318ce7"
}
tab_container_style = {"height": "44px", "align-items": "center"}
tab_style = {
    "borderBottom": "1px solid #d6d6d6",
    "padding": "6px",
    "fontWeight": "bold",
    "border-radius": "15px",
    "background-color": "#F2F2F2",
    "box-shadow": "4px 4px 4px 4px lightgrey",
}
tab_selected_style = {
    "borderTop": "1px solid #d6d6d6",
    "borderBottom": "1px solid #d6d6d6",
    "backgroundColor": "#119DFF",
    "color": "white",
    "padding": "6px",
    "border-radius": "15px",
}
# endregion Styles

# region Container
# HTML for container
app.layout = html.Div(
    children=[
        html.Div(
            className="container",
            children=[
                html.Div(
                    className="header",
                    children=[
                        html.H2(
                            "CitySEIRCast: Geospatial Agent-Based Simulator for Epidemic Analytics",
                            style={"textAlign": "center", "color": colors["text"]},
                        ),
                    ],
                ),
                html.Div(
                    style={
                        "margin-left": "10px",
                        "width": "10%",
                        "text-align": "center",
                        "display": "inline-block",
                    },
                    className="nav",
                ),
                html.Div(
                    style={
                        "margin-left": "10px",
                        "width": "80%",
                        "text-align": "center",
                        "display": "inline-block",
                    },
                    className="section",
                    children=[
                        dcc.Tabs(
                            id="tabsgraph",
                            value="tab1",
                            children=[
                                # dcc.Tab(
                                #     label="About CitySEIRCast",
                                #     value="moretab",
                                #     style=tab_style,
                                #     selected_style=tab_selected_style,
                                # ),
                                dcc.Tab(
                                    label="Time Plots",
                                    value="tab1",
                                    style=tab_style,
                                    selected_style=tab_selected_style,
                                ),
                                dcc.Tab(
                                    label="Zip Codes Time Plots",
                                    value="tab1_1",
                                    style=tab_style,
                                    selected_style=tab_selected_style,
                                ),
                                # dcc.Tab(
                                #     label="Diversity(County)",
                                #     value="tab6",
                                #     style=tab_style,
                                #     selected_style=tab_selected_style,
                                # ),
                                dcc.Tab(
                                    label="Risk by zip codes with SVI",
                                    value="tab5",
                                    style=tab_style,
                                    selected_style=tab_selected_style,
                                ),
                                dcc.Tab(
                                    label="Spatial Temporal Patterns",
                                    value="tab7",
                                    style=tab_style,
                                    selected_style=tab_selected_style,
                                ),
                                # dcc.Tab(
                                #     label="Kernel Maps",
                                #     value="tab8",
                                #     style=tab_style,
                                #     selected_style=tab_selected_style,
                                # ),
                                dcc.Tab(
                                    label="Dashboard",
                                    value="tab_dashboard",
                                    style=tab_style,
                                    selected_style=tab_selected_style,
                                ),
                            ],
                            style=tab_container_style,
                        ),
                        html.Div(
                            id="tabs-contentgraph",
                            className="tabs-content",
                            style={
                                "margin-left": 0,
                                "margin-right": 0,
                                "width": "100%",
                            },
                        ),
                        html.Div(
                            className="footer",
                            children=[
                                html.H3(
                                    children="* Simulation results are provided by Dr. Edwin Michael Lab, USF College of Public Health *"
                                ),
                                html.H4(
                                    "Team members: Edwin Michael (PI), Kenneth Newcomb, Shakir Bilal, Wajdi Zaatour, Soo I. Kim, Yilian Alonso Otano, Jun Kim"
                                ),
                                html.H3(
                                    children="- This research was supported by Hillsborough County(FL) Health Care Services. -",
                                    style={"textAlign": "center", "color": "blue"},
                                ),
                                html.H3(
                                    children="- Azure credits for implementing CitySEIRCast was provided by Microsoft® -",
                                    style={"textAlign": "center", "color": "blue"},
                                ),
                                html.Img(
                                    src=app.get_asset_url("usf-logo-white-bg.jfif"),
                                    style={"margin-left": 10, "width": "200px"},
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ]
)


# Callback for container
@app.callback(
    Output("tabs-contentgraph", "children"), Input("tabsgraph", "value")
)  # first page... if uncommented, it will not be displayed
# @cache.memoize(timeout=CACHE_TIMEOUT)  # in seconds
# endregion Container


# region Tabs
# HTML for each tabs
def render_content(tab):
    global year_for_all
    global zipcode_for_all
    global sampling_for_all
    global scatter_size
    global heatmap_size

    scatter_size = scatter_size
    heatmap_size = heatmap_size

    # About CitySEIRCast
    if tab == "moretab":
        return html.Div(
            [
                html.P(
                    "We (CitySEIRCast team at USF's Edwin Michael Lab in Tampa, Florida, USA) are developing an Geospatial Agent-Based Simulator for Epidemic Analytics. In CitySEIRCast, coupling digital replicas of populated places \
                    with libraries of selectable disease transmission models can offer \
                    a new components-based framework to overcome the data and modeling issues. \
                    In this scheme, a digital twin (DT) of a place can firstly be used to assemble data \
                    from various publicly available sources (such as locational maps, \
                    remote sensed environmental data, and increasingly Internet of Things (IoT)-based traffic \
                    and other real-time sensor data) to allow simulation of key disease risk factors, \
                    such as climate conditions and people’s demographics, occupation, dwelling and work places, \
                    and behavior and movements",
                    style={"font-size": "20px", "justify-content": "center"},
                ),
                html.P(
                    "For more info, contact Dr. Edwin Michael (emichael443@usf.edu).",
                    style={"font-size": "20px", "justify-content": "center"},
                ),
                html.Br(),
                html.Br(),
                html.Img(
                    src=app.get_asset_url("USF-EMichael-ABM-EDEN.png"),
                    style={"width": "497px", "justify-content": "center"},
                ),
                html.Br(),
                html.Br(),
            ],
        )

    # Time Plots
    elif tab == "tab1":
        return html.Div(
            [
                html.Br(),
                html.H4(
                    "Time Plots Filters:",
                    className="control_label",
                    style={"padding": 10, "flex": 1},
                ),
                dcc.RadioItems(
                    id="filter_type",
                    options=[
                        {"label": i, "value": i}
                        for i in [
                            "All cases",
                            "By Gender",
                            "By Age",
                            "By Race",
                            "By Ethnicity",
                            "By Income",
                        ]
                    ],
                    value="All cases",
                    labelStyle={"display": "inline-block"},
                ),
                html.P(period_explain),
                html.Div(
                    children=[
                        dcc.Graph(
                            id="graph1",
                            figure=load_SEIR("All cases"),
                            config={"displayModeBar": False},
                            responsive=True,
                            style={
                                "width": "100%",
                                "height": "100%",
                            },
                        )
                    ],
                    style={
                        "width": "900px",
                        "height": "750px",
                        "display": "inline-block",
                        "overflow": "hidden",
                    },
                ),
            ]
        )

    # Zip Codes Time Plots
    elif tab == "tab1_1":
        return html.Div(
            [
                html.Br(),
                html.H2("Zip Codes Time Plots"),
                html.P("To select one zip, double-click a zip code in the legend"),
                html.Div(
                    children=[
                        dcc.Graph(
                            id="graph11",
                            figure=load_allzipcodes(),
                            config={"displayModeBar": False},
                            responsive=True,
                            style={
                                "width": "100%",
                                "height": "100%",
                            },
                        )
                    ],
                    style={
                        "width": "900px",
                        "height": "900px",
                        "display": "inline-block",
                        "overflow": "hidden",
                    },
                ),
                html.Br(),
            ]
        )

    # Risk by zip codes with SVI
    elif tab == "tab5":
        div1 = html.Div(
            [
                html.H2("Risk by zip codes (Weekly)"),
                html.P(period_explain),
                html.Div(
                    dcc.Graph(
                        id="graph5_risky_zipcode",
                        figure=draw_risky_zipcodes(),
                        config={"displayModeBar": False},
                    ),
                    style={"display": "inline-block"},
                ),
            ]
        )
        div2 = html.Div(
            id="click-data-div",
            children=[
                html.Img(id="click-data"),
            ],
        )
        div3 = html.Div(
            [
                html.Br(),
                html.H2("Social Vulnerability Index"),
                html.Br(),
                html.Div(dcc.Graph(id="my-graph"), style={"display": "inline-block"}),
                html.P("Choose one SVI:"),
                html.Br(),
                dcc.RadioItems(
                    options=[
                        {"label": r, "value": r}
                        for r in df_PR.columns
                        if r not in ["zipcode", "zip_code", "PR_sum", "community"]
                    ],
                    value=df_PR.columns[1],  # initial value
                    id="my-dropdown",
                    inline=True,
                    labelStyle={
                        "display": "inline-block",
                        "margin-right": "7px",
                        "font-weight": 300,
                    },
                    style={
                        "display": "inline-block",
                        "margin-left": "20px",
                        "width": "100%",
                    },
                ),
            ]
        )
        div4 = html.Div(
            [
                html.Br(),
                html.H2("SVI Bar Chart"),
                html.P(period_explain),
                html.Div(
                    dcc.Graph(
                        id="graph5_bar_zipcode",
                        figure=bar_zipcodes(),
                        config={"displayModeBar": False},
                    ),
                    style={"display": "inline-block"},
                ),
            ]
        )
        row1 = html.Tr([html.Td([div1]), html.Td([div3]), html.Td([div4])])
        row2 = html.Tr([html.Td([div2])])
        table = html.Table(
            className="table_risk_svi",
            children=[row1],
            style={
                # "border-style": "ridge",
                "text-align": "center",
                "marginLeft": "auto",
                "marginRight": "auto",
                "width": "90%",
            },
        )
        return html.Div(
            [table, div2],
            style={"width": "95%", "display": "inline-block"},
        )

    # Diversity(County)
    elif tab == "tab6":
        return html.Div(
            children=[
                html.Br(),
                html.H2("Diversity analysis"),
                html.Br(),
                html.H3("Infections(cases) by age"),
                html.Img(
                    src=app.get_asset_url("diversity_county_new/diversity-age.png"),
                    style={"margin-left": 10, "width": "850px"},
                ),
                html.Br(),
                html.H3("Hospitalizations by age"),
                html.Img(
                    src=app.get_asset_url(
                        "diversity_county_new/diversity-age-hospitalization.png"
                    ),
                    style={"margin-left": 10, "width": "850px"},
                ),
                html.Br(),
                html.H3("Deaths by age"),
                html.Img(
                    src=app.get_asset_url(
                        "diversity_county_new/diversity-age-death.png"
                    ),
                    style={"margin-left": 10, "width": "850px"},
                ),
                html.Br(),
                html.H3("Infections(cases) by gender"),
                html.Img(
                    src=app.get_asset_url("diversity_county_new/diversity-gender.png"),
                    style={"margin-left": 10, "width": "600px"},
                ),
                html.Br(),
                html.H3("Hospitalizations by gender"),
                html.Img(
                    src=app.get_asset_url(
                        "diversity_county_new/diversity-gender-hospitalization.png"
                    ),
                    style={"margin-left": 10, "width": "600px"},
                ),
                html.Br(),
                html.H3("Deaths by gender"),
                html.Img(
                    src=app.get_asset_url(
                        "diversity_county_new/diversity-gender-death.png"
                    ),
                    style={"margin-left": 10, "width": "600px"},
                ),
                html.Br(),
                html.H3("Infections(cases) by race"),
                html.Img(
                    src=app.get_asset_url("diversity_county_new/diversity-race.png"),
                    style={"margin-left": 10, "width": "800px"},
                ),
                html.Br(),
                html.H3("Hospitalizations by race"),
                html.Img(
                    src=app.get_asset_url(
                        "diversity_county_new/diversity-race-hospitalization.png"
                    ),
                    style={"margin-left": 10, "width": "800px"},
                ),
                html.Br(),
                html.H3("Deaths by race"),
                html.Img(
                    src=app.get_asset_url(
                        "diversity_county_new/diversity-race-death.png"
                    ),
                    style={"margin-left": 10, "width": "800px"},
                ),
                html.Br(),
                html.H3("Infections(cases) by ethnicity"),
                html.Img(
                    src=app.get_asset_url(
                        "diversity_county_new/diversity-ethnicity.png"
                    ),
                    style={"margin-left": 10, "width": "600px"},
                ),
                html.Br(),
                html.H3("Hospitalizations by ethnicity"),
                html.Img(
                    src=app.get_asset_url(
                        "diversity_county_new/diversity-ethnicity-hospitalization.png"
                    ),
                    style={"margin-left": 10, "width": "600px"},
                ),
                html.Br(),
                html.H3("Deaths by ethnicity"),
                html.Img(
                    src=app.get_asset_url(
                        "diversity_county_new/diversity-ethnicity-death.png"
                    ),
                    style={"margin-left": 10, "width": "600px"},
                ),
                html.Br(),
                html.H3("Infections(cases) by income"),
                html.Img(
                    src=app.get_asset_url("diversity_county_new/diversity-income.png"),
                    style={"margin-left": 10, "width": "800px"},
                ),
                html.Br(),
                html.H3("Hospitalizations by income"),
                html.Img(
                    src=app.get_asset_url(
                        "diversity_county_new/diversity-income-hospitalization.png"
                    ),
                    style={"margin-left": 10, "width": "800px"},
                ),
                html.Br(),
                html.H3("Deaths by income"),
                html.Img(
                    src=app.get_asset_url(
                        "diversity_county_new/diversity-income-death.png"
                    ),
                    style={"margin-left": 10, "width": "800px"},
                ),
            ],
            style={"width": "100%", "display": "inline-block"},
        )

    # Spatial Temporal Patterns
    elif tab == "tab7":
        div1 = html.Div(
            [
                html.H2("Infection cases Spread Map"),
                html.P(period_explain),
                html.Div(
                    dcc.Graph(
                        id="infected_map",
                        figure=draw_infected_map(),
                        config={"displayModeBar": False},
                    ),
                    style={"display": "inline-block"},
                ),
            ]
        )
        div2 = html.Div(
            [
                html.H2("Hospitalization Spread Map"),
                html.P(period_explain),
                html.Div(
                    dcc.Graph(
                        id="hospitalised_map",
                        figure=draw_hospitalised_map(),
                        config={"displayModeBar": False},
                    ),
                    style={"display": "inline-block"},
                ),
            ]
        )
        div3 = html.Div(
            [
                html.H2("Death Spread Map"),
                html.P(period_explain),
                html.Div(
                    dcc.Graph(
                        id="dead_map",
                        figure=draw_dead_map(),
                        config={"displayModeBar": False},
                    ),
                    style={"display": "inline-block"},
                ),
            ]
        )
        div4 = html.Div(
            [
                html.H2("Sampled Naturally Immuned agents Map"),
                html.P(period_explain),
                html.Div(
                    dcc.Graph(
                        id="infected_map",
                        figure=draw_recovered_map(),
                        config={"displayModeBar": False},
                    ),
                    style={"display": "inline-block"},
                ),
            ]
        )
        div5 = html.Div(
            [
                html.H2("Sampled Vaccinated agents Map"),
                html.P(period_explain2),
                html.Div(
                    dcc.Graph(
                        id="infected_map",
                        figure=draw_vaccinated_map(),
                        config={"displayModeBar": False},
                    ),
                    style={"display": "inline-block"},
                ),
            ]
        )
        row1 = html.Tr(
            [html.Td([div1]), html.Td([div2]), html.Td([div3])],
            style={"display": "inline-block", "width": "100%"},
        )
        row2 = html.Tr([html.Td([div2])])
        table = html.Table(
            className="table_risk_svi",
            children=[row1],
            style={
                # "border-style": "ridge",
                "text-align": "center",
                "marginLeft": "auto",
                "marginRight": "auto",
                "width": "100%",
            },
        )
        row2 = html.Tr(
            [html.Td([div4]), html.Td([div5])],
            style={"display": "inline-block", "width": "100%"},
        )
        table2 = html.Table(
            className="table_recovered_vaccinated",
            children=[row2],
            style={
                # "border-style": "ridge",
                "text-align": "center",
                "marginLeft": "auto",
                "marginRight": "auto",
                "width": "100%",
            },
        )

        return html.Div(
            [
                table,
                # Hide Immune and Vaccinated
                # table2
            ],
            style={"width": "100%", "display": "inline-block"},
        )

    # Kernel Maps
    elif tab == "tab8":
        div1 = html.Div(
            [
                html.H2("Infection cases Spread Kernel Map"),
                html.P(period_explain),
                html.Div(
                    dcc.Graph(
                        id="infected_map",
                        figure=draw_infected_kernel(),
                        config={"displayModeBar": False},
                    ),
                    style={"display": "inline-block"},
                ),
            ]
        )
        div2 = html.Div(
            [
                html.H2("Hospitalization Spread Kernel Map"),
                html.P(period_explain),
                html.Div(
                    dcc.Graph(
                        id="hospitalised_map",
                        figure=draw_hospitalised_kernel(),
                        config={"displayModeBar": False},
                    ),
                    style={"display": "inline-block"},
                ),
            ]
        )
        div3 = html.Div(
            [
                html.H2("Death Spread Kernel Map"),
                html.P(period_explain),
                html.Div(
                    dcc.Graph(
                        id="dead_map",
                        figure=draw_dead_kernel(),
                        config={"displayModeBar": False},
                    ),
                    style={"display": "inline-block"},
                ),
            ]
        )
        div4 = html.Div(
            [
                html.H2("Sampled Naturally Immuned agents Kernel Map"),
                html.P(period_explain),
                html.Div(
                    dcc.Graph(
                        id="infected_map",
                        figure=draw_recovered_kernel(),
                        config={"displayModeBar": False},
                    ),
                    style={"display": "inline-block"},
                ),
            ]
        )
        div5 = html.Div(
            [
                html.H2("Sampled Vaccinated agents Kernel Map"),
                html.P(period_explain2),
                html.Div(
                    dcc.Graph(
                        id="infected_map",
                        figure=draw_vaccinated_kernel(),
                        config={"displayModeBar": False},
                    ),
                    style={"display": "inline-block"},
                ),
            ]
        )
        row1 = html.Tr(
            [html.Td([div1]), html.Td([div2]), html.Td([div3])],
            style={"display": "inline-block", "width": "100%"},
        )
        # row2 = html.Tr([html.Td([div2])])
        table = html.Table(
            className="table_risk_svi",
            children=[row1],
            style={
                # "border-style": "ridge",
                "text-align": "center",
                "marginLeft": "auto",
                "marginRight": "auto",
                "width": "100%",
            },
        )
        row2 = html.Tr(
            [html.Td([div4]), html.Td([div5])],
            style={"display": "inline-block", "width": "100%"},
        )
        table2 = html.Table(
            className="table_recovered_vaccinated",
            children=[row2],
            style={
                # "border-style": "ridge",
                "text-align": "center",
                "marginLeft": "auto",
                "marginRight": "auto",
                "width": "100%",
            },
        )

        return html.Div(
            [table, table2],
            style={"width": "100%", "display": "inline-block"},
        )

    # Dashboard
    elif tab == "tab_dashboard":
        # redirect to dashboard url
        return html.Div(
            [dcc.Location(id="url", href="http://abm.seircast.org:8050", refresh=True)]
        )


# region Callback for each tabs
# Time Plots
# Graph 1
@app.callback(
    Output("graph1", "figure"), Input("filter_type", "value"), prevent_initial_call=True
)
# @cache.memoize(timeout=CACHE_TIMEOUT)  # in seconds
def update_SEIR(filter_type1):
    global filter_type
    filter_type = filter_type1
    return load_SEIR(filter_type1)


# Risk by zip codes with SVI
# Risk by zip codes (Weekly)
# Graph 5
@app.callback(
    Output("click-data-div", "children"), Input("graph5_risky_zipcode", "clickData")
)
def display_click_data(clickData):
    cdata = json.loads(json.dumps(clickData))
    zipcode_clicked = cdata["points"][0]["location"]
    print("zipcode_clicked: ", zipcode_clicked)

    img1 = html.Img(
        src=app.get_asset_url(
            os.path.join(
                "diversity_by_zip",
                ("diversity_gender_i_" + str(zipcode_clicked) + ".png"),
            )
        ),
        style={
            "margin-left": 1,
            "width": "650px",
        },
    )
    img2 = html.Img(
        src=app.get_asset_url(
            os.path.join(
                "diversity_by_zip",
                ("diversity_age_i_" + str(zipcode_clicked) + ".png"),
            )
        ),
        style={
            "margin-left": 1,
            "width": "650px",
            # 'position': 'absolute', 'z-index': 1, 'top': 550, 'left': 0, 'onmouseover': 'this.style.cursor = "pointer"'
        },
    )
    img3 = html.Img(
        src=app.get_asset_url(
            os.path.join(
                "diversity_by_zip",
                ("diversity_race_i_" + str(zipcode_clicked) + ".png"),
            )
        ),
        style={
            "margin-left": 1,
            "width": "650px",
            # 'position': 'absolute', 'z-index': 1, 'top': 950, 'left': 0, 'onmouseover': 'this.style.cursor = "pointer"'
        },
    )
    img4 = html.Img(
        src=app.get_asset_url(
            os.path.join(
                "diversity_by_zip",
                ("diversity_ethnicity_i_" + str(zipcode_clicked) + ".png"),
            )
        ),
        style={
            "margin-left": 1,
            "width": "650px",
            # 'position': 'absolute', 'z-index': 1, 'top': 1350, 'left': 0, 'onmouseover': 'this.style.cursor = "pointer"'
        },
    )
    img5 = html.Img(
        src=app.get_asset_url(
            os.path.join(
                "diversity_by_zip",
                ("diversity_income_i_" + str(zipcode_clicked) + ".png"),
            )
        ),
        style={
            "margin-left": 1,
            "width": "650px",
            # 'position': 'absolute', 'z-index': 1, 'top': 1750, 'left': 0, 'onmouseover': 'this.style.cursor = "pointer"'
        },
    )
    img11 = html.Img(
        src=app.get_asset_url(
            os.path.join(
                "diversity_by_zip",
                ("diversity_gender_h_" + str(zipcode_clicked) + ".png"),
            )
        ),
        style={
            "margin-left": 1,
            "width": "650px",
        },
    )
    img12 = html.Img(
        src=app.get_asset_url(
            os.path.join(
                "diversity_by_zip",
                ("diversity_age_h_" + str(zipcode_clicked) + ".png"),
            )
        ),
        style={
            "margin-left": 1,
            "width": "650px",
            # 'position': 'absolute', 'z-index': 1, 'top': 550, 'left': 0, 'onmouseover': 'this.style.cursor = "pointer"'
        },
    )
    img13 = html.Img(
        src=app.get_asset_url(
            os.path.join(
                "diversity_by_zip",
                ("diversity_race_h_" + str(zipcode_clicked) + ".png"),
            )
        ),
        style={
            "margin-left": 1,
            "width": "650px",
            # 'position': 'absolute', 'z-index': 1, 'top': 950, 'left': 0, 'onmouseover': 'this.style.cursor = "pointer"'
        },
    )
    img14 = html.Img(
        src=app.get_asset_url(
            os.path.join(
                "diversity_by_zip",
                ("diversity_ethnicity_h_" + str(zipcode_clicked) + ".png"),
            )
        ),
        style={
            "margin-left": 1,
            "width": "650px",
            # 'position': 'absolute', 'z-index': 1, 'top': 1350, 'left': 0, 'onmouseover': 'this.style.cursor = "pointer"'
        },
    )
    img15 = html.Img(
        src=app.get_asset_url(
            os.path.join(
                "diversity_by_zip",
                ("diversity_income_h_" + str(zipcode_clicked) + ".png"),
            )
        ),
        style={
            "margin-left": 1,
            "width": "650px",
            # 'position': 'absolute', 'z-index': 1, 'top': 1750, 'left': 0, 'onmouseover': 'this.style.cursor = "pointer"'
        },
    )
    img21 = html.Img(
        src=app.get_asset_url(
            os.path.join(
                "diversity_by_zip",
                ("diversity_gender_d_" + str(zipcode_clicked) + ".png"),
            )
        ),
        style={
            "margin-left": 1,
            "width": "650px",
        },
    )
    img22 = html.Img(
        src=app.get_asset_url(
            os.path.join(
                "diversity_by_zip",
                ("diversity_age_d_" + str(zipcode_clicked) + ".png"),
            )
        ),
        style={
            "margin-left": 1,
            "width": "650px",
            # 'position': 'absolute', 'z-index': 1, 'top': 550, 'left': 0, 'onmouseover': 'this.style.cursor = "pointer"'
        },
    )
    img23 = html.Img(
        src=app.get_asset_url(
            os.path.join(
                "diversity_by_zip",
                ("diversity_race_d_" + str(zipcode_clicked) + ".png"),
            )
        ),
        style={
            "margin-left": 1,
            "width": "650px",
            # 'position': 'absolute', 'z-index': 1, 'top': 950, 'left': 0, 'onmouseover': 'this.style.cursor = "pointer"'
        },
    )
    img24 = html.Img(
        src=app.get_asset_url(
            os.path.join(
                "diversity_by_zip",
                ("diversity_ethnicity_d_" + str(zipcode_clicked) + ".png"),
            )
        ),
        style={
            "margin-left": 1,
            "width": "650px",
            # 'position': 'absolute', 'z-index': 1, 'top': 1350, 'left': 0, 'onmouseover': 'this.style.cursor = "pointer"'
        },
    )
    img25 = html.Img(
        src=app.get_asset_url(
            os.path.join(
                "diversity_by_zip",
                ("diversity_income_d_" + str(zipcode_clicked) + ".png"),
            )
        ),
        style={
            "margin-left": 1,
            "width": "650px",
            # 'position': 'absolute', 'z-index': 1, 'top': 1750, 'left': 0, 'onmouseover': 'this.style.cursor = "pointer"'
        },
    )
    # row1 = html.Tr([html.Td([img1]), html.Td([img2]),html.Td([img3]), html.Td([img4]), html.Td([img5])])
    # row2 = html.Tr([html.Td([img11]), html.Td([img12]),html.Td([img13]), html.Td([img14]), html.Td([img15])])
    # row3 = html.Tr([html.Td([img21]), html.Td([img22]),html.Td([img23]), html.Td([img24]), html.Td([img25])])
    row1 = html.Tr([html.Td([img1]), html.Td([img11]), html.Td([img21])])
    row2 = html.Tr([html.Td([img2]), html.Td([img12]), html.Td([img22])])
    row3 = html.Tr([html.Td([img3]), html.Td([img13]), html.Td([img23])])
    row4 = html.Tr([html.Td([img4]), html.Td([img14]), html.Td([img24])])
    row5 = html.Tr([html.Td([img5]), html.Td([img15]), html.Td([img25])])
    return html.Table(
        className="table_diversity_zip",
        # children=[row1, row2, row3],
        children=[row1, row2, row3, row4, row5],
        style={
            "border-style": "ridge",
            "text-align": "left",
            "marginLeft": "auto",
            "marginRight": "auto",
            # "width": "1000px"
        },
    )


# Risk by zip codes with SVI
# Social Vulnerability Index
# my-dropdown
@callback(Output("my-graph", "figure"), Input("my-dropdown", "value"))
def update_graph(label_selected):
    geojson = gpd.read_file(
        os.path.join(asset_path, "hillsborough-zipcodes-boundarymap.geojson")
    )
    gdf = (
        gpd.GeoDataFrame.from_features(geojson)
        .merge(df_PR, on="zipcode")
        .assign(
            lat=lambda d: d.geometry.centroid.y, lon=lambda d: d.geometry.centroid.x
        )
        .set_index("zipcode", drop=False)
    )
    texttrace = go.Scattermapbox(
        lat=gdf.geometry.centroid.y,
        lon=gdf.geometry.centroid.x,
        text=gdf["zipcode"].astype(str),
        textfont={"color": "white", "size": 20, "family": "Courier New"},
        mode="text",
        name="zipcode",
    )
    fig = basemap(label_selected).add_trace(texttrace)
    return fig


# endregion Callback for each tabs
# endregion Tabs

if __name__ == "__main__":
    app.run_server(
        debug=False,
        # use_reloader=False,
        threaded=True,
        host="0.0.0.0",
        port=8051,
    )
