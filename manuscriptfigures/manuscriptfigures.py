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
from mizani.palettes import brewer_pal
import cv2

import src.wavefinder as wf


class ManuscriptFigures:
    def __init__(self, config: Config, data_provider: DataProvider, data: dict):
        self.data_path = config.manuscript_figures_data_path
        self.figure_5_data_path = config.data_path
        self.output_path = os.path.join(self.data_path, "..", "output")
        self.data = data
        self.data_provider = data_provider
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

    def _figure4(self):
        epi_table = self.data_provider.get_epi_table()

        countries = ['ZMB', 'GBR', 'GHA', 'CRI']

        for country in countries:

            raw_cases = epi_table[epi_table['countrycode'] == country]['new_per_day']
            raw_deaths = epi_table[epi_table['countrycode'] == country]['dead_per_day']
            raw_data = [raw_cases, raw_deaths]

            fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(14, 7))

            plt.suptitle(country)

            for i, wavelist in enumerate(self.data[country]):
                raw = raw_data[i].values

                smoothed = wavelist.raw_data.values

                peaks_and_troughs = wavelist.peaks_sub_c
                peaks = peaks_and_troughs[peaks_and_troughs['peak_ind'] == 1]['location'].values

                axs[i].set_xlabel(wavelist.series_name)
                axs[i].plot(raw, color='lightgrey', zorder=0)
                axs[i].plot(smoothed, color='black', zorder=1)
                axs[i].scatter(peaks, smoothed[peaks.astype(int)], color='red', marker='o', zorder=2)

            fig.tight_layout()

            plt.savefig(os.path.join(self.output_path, f'4_{country}.png'))
            plt.close('all')

    def _figure5(self):
        # Read the data
        file = os.path.join(self.figure_5_data_path, 'figure_2.csv')
        figure_5_data = pd.read_csv(file, header=0, na_values=["N/A", "NA", "#N/A", " ", "", "None"])

        # Cut the positive rate before April due to low denominator
        figure_5_data.loc[figure_5_data['date'] < '2020-04-01', 'positive_rate'] = np.nan
        figure_5_data.loc[figure_5_data['date'] < '2020-05-01', 'cfr_smooth'] = np.nan
        # For italy, need to exclude some dates from CFR calculation when cases were very low
        figure_5_data.loc[(figure_5_data['country'] == 'Italy') & (figure_5_data['date'] >= '2020-08-16') &
                          (figure_5_data['date'] <= '2020-08-22'), 'cfr_smooth'] = np.nan

        # Normalize the ratios so that they fit on the plot
        figure_5_data[['cfr_smooth_normalized', 'positive_rate_smooth_normalized']] = np.nan
        for country in figure_5_data['country'].unique():
            max_dead_a = figure_5_data.loc[figure_5_data['country'] == country, 'dead_per_day'].max()
            max_cfr_a = figure_5_data.loc[figure_5_data['country'] == country, 'cfr_smooth'].max()
            figure_5_data.loc[figure_5_data['country'] == country,
                              'cfr_smooth_normalized'] = figure_5_data.loc[figure_5_data['country'] == country,
                                                                           'cfr_smooth'] * (max_dead_a / max_cfr_a)

            max_tests_a = figure_5_data.loc[figure_5_data['country'] == country, 'new_tests'].max()
            max_positive_rate_a = figure_5_data.loc[figure_5_data['country'] == country, 'positive_rate_smooth'].max()
            figure_5_data.loc[figure_5_data['country'] == country,
                              'positive_rate_smooth_normalized'] = figure_5_data.loc[figure_5_data['country'] == country, 'positive_rate_smooth'] * (max_tests_a / max_positive_rate_a)

        # Define which countries to plot
        country_list = ["Italy", "United States"]

        # Set up colour palette
        my_palette_1 = brewer_pal(palette="YlGnBu")(4)[1]
        my_palette_2 = brewer_pal(palette="YlGnBu")(4)[3]
        my_palette_3 = brewer_pal(palette="Oranges")(4)[3]

        # Store plots in a 3 x len(country_list) array
        g = [[],[],[]]

        for j, country_a in enumerate(country_list):
            country_data = figure_5_data[figure_5_data['country'] == country_a]

            # Row 1: cases per day
            g0 = (ggplot(country_data)
                            + geom_line(aes(x='date', y='new_per_day'), group=1, size=0.3, color=my_palette_1, na_rm=True)
                            + geom_line(aes(x='date', y='new_per_day_smooth'), group=1, color=my_palette_2, na_rm=True)
                            + labs(title=country_a, y="New Cases per Day", x=element_blank())
                            + theme_classic(base_size=8, base_family='serif')
                            + scale_y_continuous(expand=[0, 0], limits=[0, np.nan])
                            + theme(plot_title=element_text(size=8, hjust=0.5)))
            # removed 'plot_margin=unit([0, 0, 0, 0], "pt")' from theme, not sure what it should be in plotnine
            #figure_5_a._draw_using_figure(fig, axs[0,j])
            g[0].append(g0)

            # Row 2: Deaths per day with CFR
            # seems like we will need to do some trick with the twinx feature of matplotlib
            g1 = (ggplot(country_data)
                            + geom_line(aes(x='date', y='dead_per_day'), group=1, size=0.3, color=my_palette_1, na_rm=True)
                            + geom_line(aes(x='date', y='dead_per_day_smooth'), group=1, color=my_palette_2, na_rm=True)
                            + geom_line(aes(x='date', y='cfr_smooth_normalized'), group=1, color=my_palette_3,
                                        na_rm=True)
                            + scale_y_continuous(name="Deaths per Day", expand=[0, 0], limits=[0, np.nan])
                            + theme(plot_title=element_text(hjust=0.5))
                            + theme_classic(base_size=8, base_family='serif')
                            + theme(plot_title=element_text(hjust=0.5), axis_title_y=element_text(color=my_palette_2)))
            # removed 'plot_margin=unit([0, 0, 0, 0], "pt")' from both themes, not sure what it should be in plotnine
            # should be an argument in scale_y_continuous:
            # sec.axis = sec_axis(~. / (max_dead_a / max_cfr_a), name = "Case Fatality Rate")
            # similarly had to drop axis_title_y_right and rename axis_title_y_left as axis_title_y
            #figure_5_b._draw_using_figure(fig, axs[1, j])
            g[1].append(g1)

            # Row 3: Tests per day with positivity ratio
            g2 = (ggplot(country_data)
                            + geom_line(aes(x='date', y='new_tests'), group=1, size=0.3, color=my_palette_1, na_rm=True)
                            + geom_line(aes(x='date', y='new_tests_smooth'), group=1, color=my_palette_2, na_rm=True)
                            + geom_line(aes(x='date', y='positive_rate_smooth_normalized'), group=1,
                                        color=my_palette_3, na_rm=True)
                            + scale_y_continuous(name="Tests per Day", expand=[0, 0], limits=[0, np.nan])
                            + labs(x="Date")
                            + theme_classic(base_size=8, base_family='serif')
                            + theme(plot_title=element_text(hjust=0.5), axis_title_y=element_text(color=my_palette_2)))
            # removed 'plot_margin=unit([0, 0, 0, 0], "pt")' from both themes, not sure what it should be in plotnine
            # should be an argument in scale_y_continuous:
            # sec.axis = sec_axis(~. / (max_tests_a / max_positive_rate_a), name = "Positive Rate")
            g[2].append(g2)

        # convert all ggplots to cv2 objects (via matplotlib fig objects
        # then concatenate them, first by row, then by column
        cv2_images = [[],[],[]]
        for i in range(3):
            for j in range(len(country_list)):
                fig = g[i][j].draw()

                # convert canvas to image
                fig.canvas.draw()
                img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                                    sep='')
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                # img is rgb, convert to opencv's default bgr
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2_images[i].append(img)

            cv2_images[i] = cv2.hconcat(cv2_images[i])
        cv2_images = cv2.vconcat(cv2_images)

        # save final output
        cv2.imwrite(os.path.join(self.output_path, '5.png'), cv2_images)

    def main(self):
        # Figure 1a is a map - TODO
        self._figure1b()
        self._figure1c()
        # Figure 2 is a hand-drawn figure
        self._figure3()
        # Figure 4 is looking a little out of alignment - TODO
        self._figure4()
        # Figure 5 requires work to establish the secondary axes - TODO
        self._figure5()
        return
