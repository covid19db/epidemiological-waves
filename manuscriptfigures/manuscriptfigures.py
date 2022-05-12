import numpy as np
import geopandas as gpd
import datetime
import psycopg2
import matplotlib.pyplot as plt
import seaborn as sns
from shapely import wkt
from tqdm import tqdm
from config import Config
from data_provider import DataProvider
from epidemicwaveclassifier import EpidemicWaveClassifier
import os
import pandas as pd
from plotnine import *
import pickle

import src.wavefinder as wf


class ManuscriptFigures:
    def __init__(self, config: Config, data: dict):
        self.data_path = config.manuscript_figures_data_path
        self.output_path = os.path.join(self.data_path, "..", "output")
        self.data = data
        return

    def _figure1b(self):
        # read in the GNI data
        file = os.path.join(self.data_path, "2020-09-15/gni_data.csv")
        x = pd.read_csv(file,
                        header=0,
                        usecols=['countrycode', 'gni_per_capita'])

        # read in the t_0 data
        file = os.path.join(self.data_path, "2021-09-15/figure_1b.csv")
        t0_df = pd.read_csv(file,
                            delimiter=";",
                            header=0,
                            usecols=["countrycode", "days_to_t0_10_dead"],
                            na_values=["N/A", "NA", "#N/A", " ", "", "None"]).rename(
            columns={'days_to_t0_10_dead': 'days_to_t0'})

        # merge dataframes to plot
        z = pd.merge(x, t0_df, how="outer", on="countrycode").dropna()

        def number_formatter(array):
            return ["{:,}".format(int(n)) for n in array]

        g = (ggplot(z, aes(x='gni_per_capita', y='days_to_t0'))
             + geom_point(shape=1)
             + geom_smooth(method="lm", colour="#7a0177", fill="#c51b8a")
             + scale_x_log10(labels=number_formatter)
             + scale_y_log10()
             + labs(x="GNI per capita", y="Days until epidemic established")
             + theme_bw()
             + theme(axis_title=element_text(face="bold"))
             )

        file = os.path.join(self.output_path, "1B.png")
        g.save(filename=file, height=7.4, width=14.7, units="cm", dpi=500)

    def _figure1c(self):
        data_path = os.path.join(self.data_path, "2021-09-15")
        file = os.path.join(data_path, "figure_1b.csv")
        extra_df = pd.read_csv(file,
                               delimiter=";",
                               header=0,
                               na_values=["N/A", "NA", "#N/A", " ", "", "None"])

        extra_df = extra_df[extra_df["class"] == 4]

        y_chr = extra_df[["countrycode"]]
        y_pop = extra_df[["countrycode", "population"]]

        ## check that we are not missing the population for any of the countries we are
        ## considering...
        ##
        ## We can get population estimates from OWID, but very similar values are
        ## available from WorldOMeter (https://www.worldometers.info/).
        ##
        ## - GLP is Guadeloupe  400,013
        ## - MTQ is Martinique  374,743
        ## - MYT is Mayotte     279,507
        ##

        y_pop['population'] = np.where(y_pop['countrycode'] == 'GLP', 400013, y_pop['population'])
        y_pop['population'] = np.where(y_pop['countrycode'] == 'MTQ', 374743, y_pop['population'])
        y_pop['population'] = np.where(y_pop['countrycode'] == 'MYT', 279507, y_pop['population'])

        ## read in the time series of cases and deaths so we have the actual data to
        ## plot.
        file = os.path.join(data_path, "figure_1a.csv")
        x = pd.read_csv(file,
                        header=0,
                        usecols=["countrycode", "date", "new_per_day", "dead_per_day"],
                        parse_dates=['date'],
                        na_values=["N/A", "NA", "#N/A", " ", ""])
        x = x[x["countrycode"].isin(y_chr['countrycode'])]
        x = pd.melt(x, id_vars=["countrycode", "date"])

        ## read in the t_0 data to align the time series correctly.
        file = os.path.join(data_path, "figure_1b.csv")
        t0_df = pd.read_csv(file,
                            delimiter=";",
                            header=0,
                            usecols=["countrycode", "days_to_t0_10_dead"],
                            na_values=["N/A", "NA", "#N/A", " ", "", "None"]).rename(
            columns={'days_to_t0_10_dead': 'days_to_t0'})

        ## convert the date to an integer and then subtract the number of days until the
        ## threshold value is reached. There is an extra offset there so that the
        ## resulting numbers start at zero. This is also where we calculate the values
        ## per 10k people.
        min_num_date = x["date"].apply(lambda x: x.toordinal()).min()

        aligned_df = pd.merge(left=x, right=t0_df, how="left", on="countrycode")
        aligned_df["num_aligned_date"] = aligned_df["date"].apply(lambda x: x.toordinal()) - aligned_df[
            "days_to_t0"] - min_num_date
        aligned_df = aligned_df[aligned_df["num_aligned_date"] >= 0]
        aligned_df = aligned_df.merge(y_pop, how="left", on="countrycode")
        aligned_df["value_per_10k"] = aligned_df["value"] * 10000 / aligned_df["population"]
        aligned_df = aligned_df[["variable", "countrycode", "value_per_10k", "num_aligned_date"]]

        facet_labels = {"dead_per_day": "Deaths", "new_per_day": "Confirmed Cases"}

        aligned_df = aligned_df.drop(
            aligned_df[(aligned_df.countrycode == "USA") | (aligned_df.countrycode == "IND")].index)

        smooth_df_npd = aligned_df[aligned_df.variable == "new_per_day"]
        smooth_df_npd = smooth_df_npd.groupby(["variable", "num_aligned_date"]).mean().reset_index()
        smooth_df_npd["value_per_10k"] = smooth_df_npd["value_per_10k"].rolling(7).median()

        smooth_df_dpd = aligned_df[aligned_df.variable == "dead_per_day"]
        smooth_df_dpd = smooth_df_dpd.groupby(["variable", "num_aligned_date"]).mean().reset_index()
        smooth_df_dpd["value_per_10k"] = smooth_df_dpd["value_per_10k"].rolling(7).median()

        smooth_df = pd.concat([smooth_df_npd, smooth_df_dpd])

        g = (ggplot(aligned_df, aes(x="num_aligned_date", y="value_per_10k", group="countrycode"))
             + geom_point(shape=1, alpha=0.1)
             + geom_line(data=smooth_df, group="NA", colour="#7a0177", size=2)
             + facet_wrap(facets="~ variable", scales="free_y", labeller=facet_labels)
             + scale_y_sqrt(n_breaks=4)
             + labs(y="Per 10,000 people (square root scale)",
                    x="Days since epidemic threshold reached")
             + theme_bw()
             + theme(strip_background=element_blank(), strip_text=element_text(face="bold")))

        file = os.path.join(self.output_path, "1C.png")
        g.save(filename=file, height=7.4, width=14.7, units="cm", dpi=500)

    def _figure3(self):

        (cases, deaths) = self.data['GHA']
        wf.plot_peaks([cases, deaths], 'Figure 3', True, self.output_path)

    def main(self):
        # Figure 1a is a map - TODO
        self._figure1b()
        self._figure1c()
        # Figure 2 is a hand-drawn figure
        self._figure3()
        return
