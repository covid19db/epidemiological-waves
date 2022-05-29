import os
import numpy as np
import pandas as pd
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


class Figures:
    def __init__(self, config: Config, epi_panel: pd.core.frame.DataFrame,
                 data_provider: DataProvider, epi_classifier: EpidemicWaveClassifier ):
        self.data_dir = config.manuscript_figures_data_path
        self.config = config
        self.data_provider = data_provider
        self.epi_panel = epi_panel
        self.epi_classifier = epi_classifier
        return

    def _figure_0(self):
        # not being used - safe to remove?
        countries = ['ZMB','GBR','GHA','CRI']
        figure_0_series = self.data_provider.epidemiology_series.loc[
            self.data_provider.epidemiology_series['countrycode'].isin(countries),
            ['country', 'countrycode', 'date', 'new_per_day',
             'new_per_day_smooth', 'dead_per_day', 'dead_per_day_smooth']]
        figure_0_panel = self.epi_panel.loc[self.epi_panel['countrycode'].isin(countries),
                                            ['class', 'country', 'countrycode', 'population', 't0_10_dead',
                                             'date_peak_1', 'peak_1', 'wave_start_1', 'wave_end_1',
                                             'date_peak_2', 'peak_2', 'wave_start_2', 'wave_end_2',
                                             'date_peak_3', 'peak_3', 'wave_start_3', 'wave_end_3']]
        figure_0_series.to_csv(os.path.join(self.data_dir, 'figure_0_series.csv'))
        figure_0_panel.to_csv(os.path.join(self.data_dir, 'figure_0_panel.csv'))
        return

    def _initialise_postgres(self):
        conn = psycopg2.connect(
            host='covid19db.org',
            port=5432,
            dbname='covid19',
            user='covid19',
            password='covid19')
        conn.cursor()
        return conn

    def _figure_6(self):
        # this looks to generate the figure 6 files
        CUTOFF_DATE = datetime.date(2021, 7, 1)
        conn = self._initialise_postgres()
        sql_command = """SELECT * FROM administrative_division WHERE countrycode='USA'"""
        usa_map = gpd.GeoDataFrame.from_postgis(sql_command, conn, geom_col='geometry')
        usa_populations = pd.read_csv(
            'https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/UID_ISO_FIPS_LookUp_Table.csv')
        usa_cases = pd.read_csv('https://github.com/nytimes/covid-19-data/raw/master/us-counties.csv')
        #Calculate number of new cases per day
        usa_cases = usa_cases.sort_values(by=['fips','date']).reset_index(drop=True)
        for fips in usa_cases['fips'].unique():
            usa_cases.loc[usa_cases['fips']==fips,'new_cases'] = usa_cases.loc[usa_cases['fips']==fips,'cases'].diff()
            usa_cases.loc[usa_cases['new_cases']<0,'new_cases'] = 0
            
        # Get only the peak dates
        dates = ['2020-04-08', '2020-07-21', '2021-01-04']
        usa_cases = usa_cases.loc[usa_cases['date'].isin(dates),]

        # Using OxCOVID translation csv to map US county FIPS code to GIDs for map_data matching
        translation_csv = pd.read_csv('https://github.com/covid19db/fetchers-python/raw/master/' +
                                      'src/plugins/USA_NYT/translation.csv')

        figure_6 = usa_cases.merge(translation_csv[['input_adm_area_1', 'input_adm_area_2', 'gid']],
                                   left_on=['state', 'county'], right_on=['input_adm_area_1', 'input_adm_area_2'],
                                   how='left').merge(
            usa_populations[['FIPS', 'Population']], left_on=['fips'], right_on=['FIPS'], how='left')

        figure_6 = figure_6[['date', 'gid', 'fips', 'cases', 'new_cases', 'Population']].sort_values(by=['gid', 'date']).dropna(
            subset=['gid'])
        figure_6 = usa_map[['gid', 'geometry']].merge(figure_6, on=['gid'], how='right')
        figure_6.astype({'geometry': str}).to_csv(os.path.join(self.data_dir, 'figure_6.csv'), sep=';')

        cols = 'countrycode, adm_area_1, date, confirmed'
        sql_command = """SELECT """ + cols + \
        """ FROM epidemiology WHERE countrycode = 'USA' AND source = 'USA_NYT' AND adm_area_1 IS NOT NULL AND adm_area_2 IS NULL"""
        raw_usa = pd.read_sql(sql_command, conn)
        raw_usa = raw_usa.sort_values(by=['adm_area_1', 'date']).reset_index(drop=True)
        raw_usa = raw_usa[raw_usa['date'] <= CUTOFF_DATE].reset_index(drop=True)

        states = raw_usa['adm_area_1'].unique()
        figure_6a = pd.DataFrame(
            columns=['countrycode', 'adm_area_1', 'date', 'confirmed', 'new_per_day', 'new_per_day_smooth'])

        for state in tqdm(states, desc='Processing USA Epidemiological Data'):
            data = raw_usa[raw_usa['adm_area_1'] == state].set_index('date')
            data = data.reindex([x.date() for x in pd.date_range(data.index.values[0], data.index.values[-1])])
            data['confirmed'] = data['confirmed'].interpolate(method='linear')
            data['new_per_day'] = data['confirmed'].diff()
            data.reset_index(inplace=True)
            data['new_per_day'].iloc[np.array(data[data['new_per_day'] < 0].index)] = \
                data['new_per_day'].iloc[np.array(data[data['new_per_day'] < 0].index) - 1]
            data['new_per_day'] = data['new_per_day'].fillna(method='bfill')
            # x = np.arange(len(data['date']))
            # y = data['new_per_day'].values
            # ys = csaps(x, y, x, smooth=SMOOTH)
            ys = data[['new_per_day', 'date']].rolling(window=7, on='date').mean()['new_per_day']
            data['new_per_day_smooth'] = ys
            figure_6a = pd.concat((figure_6a, data)).reset_index(drop=True)
            continue

        sql_command = """SELECT adm_area_1, latitude, longitude FROM administrative_division WHERE adm_level=1 AND countrycode='USA'"""
        states_lat_long = pd.read_sql(sql_command, conn)
        figure_6a = figure_6a.merge(states_lat_long, on='adm_area_1')
        figure_6a.to_csv(os.path.join(self.data_dir, 'figure_6a.csv'), sep=',')
        return

    def _figure_1(self, start_date=datetime.date(2019,12,31)):
        # query map data (serialising issues in format so not caching)
        # hence the use of the semi-colon delimiter later
        conn = self._initialise_postgres()
        sql_command = """SELECT * FROM administrative_division WHERE adm_level=0"""
        map_data = gpd.GeoDataFrame.from_postgis(sql_command, conn, geom_col='geometry')
        # get figure 1a
        figure_1a = self.data_provider.epidemiology_series[
            ['countrycode', 'date', 'new_per_day', 'new_cases_per_rel_constant', 'dead_per_day', 'new_deaths_per_rel_constant']]
        # get figure 1b
        figure_1b = self.epi_panel[['countrycode', 't0_10_dead', 'class', 'class_coarse', 'population']].merge(
            self.data_provider.wbi_table[['countrycode', 'gni_per_capita']], on=['countrycode'], how='left')
        figure_1b['days_to_t0_10_dead'] = (figure_1b['t0_10_dead'] - start_date).apply(lambda x: x.days)
        figure_1b = figure_1b.merge(map_data[['countrycode', 'geometry']], on=['countrycode'], how='left')
        # cache CSV files for later use
        figure_1a.to_csv(os.path.join(self.data_dir, 'figure_1a.csv'))
        figure_1b.astype({'geometry': str}).to_csv(os.path.join(self.data_dir, 'figure_1b.csv'), sep=';')

        # panel A
        panel_a = self.data_provider.epidemiology_series[
            ['countrycode', 'date', 'days_since_t0_10_dead', 'new_cases_per_rel_constant', 'new_deaths_per_rel_constant']].merge(
            self.epi_panel[['countrycode', 't0_10_dead', 'class', 'population']], on=['countrycode'], how='left').merge(
            self.data_provider.wbi_table[['countrycode', 'gni_per_capita']], on=['countrycode'], how='left')
        panel_a = panel_a[panel_a['class'] >= 3]
        # cases
        cases = panel_a[['date', 'countrycode', 'new_cases_per_rel_constant']].groupby('date').agg(
            min=pd.NamedAgg(column='new_cases_per_rel_constant', aggfunc='min'),
            max=pd.NamedAgg(column='new_cases_per_rel_constant', aggfunc='max'),
            median=pd.NamedAgg(column='new_cases_per_rel_constant', aggfunc=np.median),
        )
        fig, ax = plt.subplots(nrows=1, ncols=1)
        sns.lineplot(data=cases.reset_index().dropna(), x='date', y='median', ax=ax, color='darkred', linewidth=3)
        ax.set(xlabel='Date', ylabel='Confirmed Cases')
        ax.fill_between(cases.reset_index().dropna()['date'].values, cases.reset_index().dropna()['max'].values,
                        cases.reset_index().dropna()['min'].values, alpha=0.3)

        # deaths
        deaths = panel_a[['date', 'countrycode', 'new_deaths_per_rel_constant']].groupby('date').agg(
            min=pd.NamedAgg(column='new_deaths_per_rel_constant', aggfunc='min'),
            max=pd.NamedAgg(column='new_deaths_per_rel_constant', aggfunc='max'),
            median=pd.NamedAgg(column='new_deaths_per_rel_constant', aggfunc=np.median),
        )




        # panel B
        # panel C
        panel_c = map_data[['countrycode', 'geometry']].merge(
            figure_1b[['countrycode', 'days_to_t0_10_dead']], on=['countrycode'], how='left')

        cmap = plt.get_cmap('viridis', int(figure_1b['class'].max() - figure_1b['class'].min() + 1))
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.set_axis_off()
        panel_c.plot(column='days_to_t0_10_dead', linewidth=0.5, edgecolor='0.5',
                     legend_kwds={'orientation': 'horizontal', 'label': 'Days to T0'},
                     figsize=(20, 7), legend=True,
                     missing_kwds={"color": "lightgrey", "edgecolor": "black"})
        plt.axis('equal')

        return figure_1a, figure_1b

    def _figure_5(self):
        # this looks to generate the figure_5.csv file
        countries = ['ITA', 'FRA', 'USA', 'ZMB','GBR','GHA','CRI']
        figure_5 = self.data_provider.epidemiology_series.loc[
            self.data_provider.epidemiology_series['countrycode'].isin(countries),
            ['country', 'countrycode', 'date', 'new_per_day', 'new_per_day_smooth',
             'dead_per_day', 'dead_per_day_smooth', 'new_tests', 'new_tests_smooth',
             'positive_rate', 'positive_rate_smooth', 'cfr_smooth']]
        figure_5.to_csv(os.path.join(self.data_dir, 'figure_5.csv'))
        return


    def main(self):
        self._figure_1()
        self._figure_5()
        self._figure_6()
        return
