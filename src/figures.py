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
        self.data_dir = config.data_path
        self.config = config
        self.data_provider = data_provider
        self.epi_panel = epi_panel
        self.epi_classifier = epi_classifier
        return

    def _figure_0(self):
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

    def _figure_4(self):
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

        figure_4 = usa_cases.merge(translation_csv[['input_adm_area_1', 'input_adm_area_2', 'gid']],
                                   left_on=['state', 'county'], right_on=['input_adm_area_1', 'input_adm_area_2'],
                                   how='left').merge(
            usa_populations[['FIPS', 'Population']], left_on=['fips'], right_on=['FIPS'], how='left')

        figure_4 = figure_4[['date', 'gid', 'fips', 'cases', 'new_cases', 'Population']].sort_values(by=['gid', 'date']).dropna(
            subset=['gid'])
        figure_4 = usa_map[['gid', 'geometry']].merge(figure_4, on=['gid'], how='right')
        figure_4.astype({'geometry': str}).to_csv(os.path.join(self.data_dir, 'figure_4.csv'), sep=';')


        cols = 'countrycode, adm_area_1, date, confirmed'
        sql_command = """SELECT """ + cols + \
        """ FROM epidemiology WHERE countrycode = 'USA' AND source = 'USA_NYT' AND adm_area_1 IS NOT NULL AND adm_area_2 IS NULL"""
        raw_usa = pd.read_sql(sql_command, conn)
        raw_usa = raw_usa.sort_values(by=['adm_area_1', 'date']).reset_index(drop=True)
        raw_usa = raw_usa[raw_usa['date'] <= CUTOFF_DATE].reset_index(drop=True)

        states = raw_usa['adm_area_1'].unique()
        figure_4a = pd.DataFrame(
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
            figure_4a = pd.concat((figure_4a, data)).reset_index(drop=True)
            continue

        sql_command = """SELECT adm_area_1, latitude, longitude FROM administrative_division WHERE adm_level=1 AND countrycode='USA'"""
        states_lat_long = pd.read_sql(sql_command, conn)
        figure_4a = figure_4a.merge(states_lat_long, on='adm_area_1')
        figure_4a.to_csv(os.path.join(self.data_dir, 'figure_4a.csv'), sep=',')
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

    def _figure_2(self):
        countries = ['ITA', 'FRA', 'USA', 'ZMB','GBR','GHA','CRI']
        figure_2 = self.data_provider.epidemiology_series.loc[
            self.data_provider.epidemiology_series['countrycode'].isin(countries),
            ['country', 'countrycode', 'date', 'new_per_day', 'new_per_day_smooth',
             'dead_per_day', 'dead_per_day_smooth', 'new_tests', 'new_tests_smooth',
             'positive_rate', 'positive_rate_smooth', 'cfr_smooth']]
        figure_2.to_csv(os.path.join(self.data_dir, 'figure_2.csv'))
        return

    def _get_government_panel(self):
        SI_THRESHOLD = 60
        flags = ['c1_school_closing', 'c2_workplace_closing', 'c3_cancel_public_events',
                 'c4_restrictions_on_gatherings', 'c5_close_public_transport',
                 'c6_stay_at_home_requirements', 'c7_restrictions_on_internal_movement',
                 'c8_international_travel_controls',
                 'h2_testing_policy', 'h3_contact_tracing']
        flag_thresholds = {'c1_school_closing': 3,
                           'c2_workplace_closing': 3,
                           'c3_cancel_public_events': 2,
                           'c4_restrictions_on_gatherings': 4,
                           'c5_close_public_transport': 2,
                           'c6_stay_at_home_requirements': 2,
                           'c7_restrictions_on_internal_movement': 2,
                           'c8_international_travel_controls': 4,
                           'h2_testing_policy': 3,
                           'h3_contact_tracing': 2}

        government_response_series = {
            'countrycode': np.empty(0),
            'country': np.empty(0),
            'date': np.empty(0),
            'stringency_index': np.empty(0)
        }

        for flag in flags:
            government_response_series[flag] = np.empty(0)
            government_response_series[flag + '_days_above_threshold'] = np.empty(0)

        countries = np.sort(self.data_provider.gsi_table['countrycode'].unique())
        for country in tqdm(countries, desc='Processing Government Response Time Series Data'):
            data = self.data_provider.gsi_table[self.data_provider.gsi_table['countrycode'] == country]

            government_response_series['countrycode'] = np.concatenate((
                government_response_series['countrycode'], data['countrycode'].values))
            government_response_series['country'] = np.concatenate(
                (government_response_series['country'], data['country'].values))
            government_response_series['date'] = np.concatenate(
                (government_response_series['date'], data['date'].values))
            government_response_series['stringency_index'] = np.concatenate(
                (government_response_series['stringency_index'], data['stringency_index'].values))

            for flag in flags:
                days_above = (data[flag] >= flag_thresholds[flag]).astype(int).values

                government_response_series[flag] = np.concatenate(
                    (government_response_series[flag], data[flag].values))
                government_response_series[flag + '_days_above_threshold'] = np.concatenate(
                    (government_response_series[flag + '_days_above_threshold'], days_above))
        government_response_series = pd.DataFrame.from_dict(government_response_series)

        government_response_panel = pd.DataFrame(columns=['countrycode', 'country', 'max_si', 'date_max_si',
                                                          'si_days_to_max_si', 'si_at_t0', 'si_at_peak_1',
                                                          'si_days_to_threshold',
                                                          'si_days_above_threshold',
                                                          'si_days_above_threshold_first_wave',
                                                          'si_integral'] +
                                                         [flag + '_at_t0' for flag in flags] +
                                                         [flag + '_at_peak_1' for flag in flags] +
                                                         [flag + '_days_to_threshold' for flag in flags] +
                                                         [flag + '_days_above_threshold' for flag in flags] +
                                                         [flag + '_days_above_threshold_first_wave' for flag in flags] +
                                                         [flag + '_raised' for flag in flags] +
                                                         [flag + '_lowered' for flag in flags] +
                                                         [flag + '_raised_again' for flag in flags])

        countries = self.data_provider.gsi_table['countrycode'].unique()
        for country in tqdm(countries, desc='Processing Gov Response Panel Data'):
            data = dict()
            country_series = government_response_series[government_response_series['countrycode'] == country]
            data['countrycode'] = country
            data['country'] = country_series['country'].iloc[0]
            if all(pd.isnull(country_series['stringency_index'])):  # if no values for SI, skip to next country
                continue
            data['max_si'] = country_series['stringency_index'].max()
            data['date_max_si'] = country_series[country_series['stringency_index'] == data['max_si']]['date'].iloc[0]
            population = np.nan if \
                len(self.data_provider.wbi_table[
                        self.data_provider.wbi_table['countrycode'] == country]['value']) == 0 else \
                self.data_provider.wbi_table[self.data_provider.wbi_table['countrycode'] == country]['value'].iloc[0]
            t0 = np.nan if len(self.epi_panel[self.epi_panel['countrycode'] == country]['t0_10_dead']) == 0 \
                else self.epi_panel[self.epi_panel['countrycode'] == country]['t0_10_dead'].iloc[0]
            data['si_days_to_max_si'] = np.nan if pd.isnull(t0) else (data['date_max_si'] - t0).days
            data['si_days_above_threshold'] = sum(country_series['stringency_index'] >= SI_THRESHOLD)
            data['si_integral'] = np.trapz(y=country_series['stringency_index'].dropna(),
                                           x=[(a - country_series['date'].values[0]).days for a in
                                              country_series['date'][~np.isnan(country_series['stringency_index'])]])
            # Initialize columns as nan first for potential missing values
            data['si_days_above_threshold_first_wave'] = np.nan
            data['si_at_t0'] = np.nan
            data['si_at_peak_1'] = np.nan
            for flag in flags:
                data[flag + '_raised'] = np.nan
                data[flag + '_lowered'] = np.nan
                data[flag + '_raised_again'] = np.nan
                data[flag + '_at_t0'] = np.nan
                data[flag + '_at_peak_1'] = np.nan
                data[flag + '_days_above_threshold_first_wave'] = np.nan
            if country in self.epi_panel['countrycode'].values:
                date_peak_1 = \
                self.epi_panel.loc[self.epi_panel['countrycode'] == country, 'date_peak_1'].values[0]
                first_wave_start = \
                self.epi_panel.loc[self.epi_panel['countrycode'] == country, 'wave_start_1'].values[0]
                first_wave_end = \
                self.epi_panel.loc[self.epi_panel['countrycode'] == country, 'wave_end_1'].values[0]
                if not pd.isnull(t0) and t0 in country_series['date']:
                    # SI value at T0
                    data['si_at_t0'] = country_series.loc[country_series['date'] == t0, 'stringency_index'].values[0]
                    # days taken to reach threshold
                    data['si_days_to_threshold'] = (
                                min(country_series.loc[country_series['stringency_index'] >= SI_THRESHOLD, 'date']) - t0).days \
                        if sum(country_series['stringency_index'] >= SI_THRESHOLD) > 0 else np.nan
                    for flag in flags:
                        data[flag + '_at_t0'] = country_series.loc[country_series['date'] == t0, flag].values[0]
                        data[flag + '_days_to_threshold'] = (min(
                            country_series.loc[country_series[flag] >= flag_thresholds[flag], 'date']) - t0).days \
                            if sum(country_series[flag] >= flag_thresholds[flag]) > 0 else np.nan
                if not (pd.isnull(date_peak_1) or pd.isnull(first_wave_start) or pd.isnull(first_wave_end)) \
                        and date_peak_1 in country_series['date']:
                    # SI value at peak date
                    data['si_at_peak_1'] = country_series.loc[
                        country_series['date'] == date_peak_1, 'stringency_index'].values[0]
                    # number of days SI above the threshold during the first wave
                    data['si_days_above_threshold_first_wave'] = sum((country_series['stringency_index'] >= SI_THRESHOLD) &
                                                                     (country_series['date'] >= first_wave_start) &
                                                                     (country_series['date'] <= first_wave_end))
                    for flag in flags:
                        # flag value at peak date
                        data[flag + '_at_peak_1'] = \
                        country_series.loc[country_series['date'] == date_peak_1, flag].values[0]
                        # number of days each flag above threshold during first wave
                        data[flag + '_days_above_threshold_first_wave'] = country_series[
                            (country_series['date'] >= first_wave_start) &
                            (country_series['date'] <= first_wave_end)][flag + '_days_above_threshold'].sum()
            for flag in flags:
                days_above = pd.Series(country_series[flag + '_days_above_threshold'])
                waves = [[cat[1], grp.shape[0]] for cat, grp in
                         days_above.groupby([days_above.ne(days_above.shift()).cumsum(), days_above])]
                if len(waves) >= 2:
                    data[flag + '_raised'] = country_series['date'].iloc[waves[0][1]]
                if len(waves) >= 3:
                    data[flag + '_lowered'] = country_series['date'].iloc[
                        waves[0][1] + waves[1][1]]
                if len(waves) >= 4:
                    data[flag + '_raised_again'] = country_series['date'].iloc[
                        waves[0][1] + waves[1][1] + waves[2][1]]
                data[flag + '_days_above_threshold'] = country_series[flag + '_days_above_threshold'].sum()

            government_response_panel = government_response_panel.append(data, ignore_index=True)
        return government_response_panel, government_response_series

    # epidemic wave classifier only fills start of wave three once the wave is definitively formed
    def _handle_wave_start(self, country, wave_number):
        if wave_number <= 1:
            raise ValueError
        wave = self.epi_panel.loc[
            self.epi_panel['countrycode'] == country, 'wave_start_{}'.format(wave_number)].values[0]
        if pd.isnull(wave):
            wave = self.epi_panel.loc[
            self.epi_panel['countrycode'] == country, 'wave_end_{}'.format(wave_number - 1)].values[0]
        return wave

    # if the end is not defined unless the wave is a full wave
    def _handle_wave_end(self, country, wave_number):
        if wave_number < 1:
            raise ValueError
        wave = self.epi_panel.loc[
            self.epi_panel['countrycode'] == country, 'wave_end_{}'.format(wave_number)].values[0]
        if pd.isnull(wave):
            wave = datetime.datetime.today().date()
        return wave

    def _figure_3(self):
        TESTS_THRESHOLD = [1, 10, 100, 1000]
        SI_THRESHOLD = 60
        flags = ['c1_school_closing', 'c2_workplace_closing', 'c3_cancel_public_events',
                 'c4_restrictions_on_gatherings', 'c5_close_public_transport',
                 'c6_stay_at_home_requirements', 'c7_restrictions_on_internal_movement',
                 'c8_international_travel_controls',
                 'h2_testing_policy', 'h3_contact_tracing']
        flag_thresholds = {'c1_school_closing': 3,
                           'c2_workplace_closing': 3,
                           'c3_cancel_public_events': 2,
                           'c4_restrictions_on_gatherings': 4,
                           'c5_close_public_transport': 2,
                           'c6_stay_at_home_requirements': 2,
                           'c7_restrictions_on_internal_movement': 2,
                           'c8_international_travel_controls': 4,
                           'h2_testing_policy': 3,
                           'h3_contact_tracing': 2}
        government_response_panel, government_response_series = self._get_government_panel()
        figure_3_wave_level = pd.DataFrame()
        countries = self.epi_panel['countrycode'].unique()
        for country in tqdm(countries, desc='Processing figure 3 wave level data'):
            dead = self.data_provider.get_series(country, 'dead_per_day')
            tests = self.data_provider.get_series(country, 'new_tests')
            data = dict()
            data['dead_during_wave'] = np.nan
            data['tests_during_wave'] = np.nan
            data['si_integral_during_wave'] = np.nan
            data['countrycode'] = country
            data['country'] = self.epi_panel.loc[self.epi_panel['countrycode'] == country, 'country'].values[0]
            data['class'] = self.epi_panel.loc[self.epi_panel['countrycode'] == country, 'class'].values[0]
            data['population'] = self.epi_panel.loc[self.epi_panel['countrycode'] == country, 'population'].values[0]
            if data['class'] >= 1:
                # First wave
                data['wave'] = 1
                data['wave_start'] = \
                self.epi_panel.loc[self.epi_panel['countrycode'] == country, 'wave_start_1'].values[0]
                data['wave_end'] = self._handle_wave_end(country, 1)
                data['t0_10_dead'] = \
                self.epi_panel.loc[self.epi_panel['countrycode'] == country, 't0_10_dead'].values[0]
                data['dead_during_wave'] = \
                dead[(dead['date'] >= self.epi_panel.loc[
                    self.epi_panel['countrycode'] == country, 'wave_start_1'].values[0]) &
                     (dead['date'] <= self.epi_panel.loc[
                         self.epi_panel['countrycode'] == country, 'wave_end_1'].values[0])]['dead_per_day'].sum()
                data['tests_during_wave'] = \
                    tests[(tests['date'] >= self.epi_panel.loc[
                        self.epi_panel['countrycode'] == country, 'wave_start_1'].values[0]) &
                         (tests['date'] <= self.epi_panel.loc[
                             self.epi_panel['countrycode'] == country, 'wave_end_1'].values[0])]['new_tests'].sum()

                # if tests during first wave is na due to missing data, linear interpolate low test numbers
                if pd.isnull(data['tests_during_wave']):
                    country_series = self.epi_panel[self.epi_panel['countrycode'] == country]
                    if not pd.isnull(data['wave_start']) and not np.all(pd.isnull(country_series['tests'])):
                        min_date = min(country_series['date'])
                        min_tests = np.nanmin(country_series['tests'])
                        if pd.isnull(country_series.loc[country_series['date'] == min_date, 'tests'].values[0]) \
                                and min_tests <= 1000:
                            country_series.loc[country_series['date'] == min_date, 'tests'] = 0
                            country_series['tests'] = country_series['tests'].interpolate(method='linear')
                        if not pd.isnull(
                                country_series.loc[country_series['date'] == data['wave_start'], 'tests'].values[
                                    0]) and not pd.isnull(
                                country_series.loc[country_series['date'] == data['wave_end'], 'tests'].values[0]):
                            data['tests_during_wave'] = \
                            country_series.loc[country_series['date'] == data['wave_end'], 'tests'].values[0] - \
                            country_series.loc[country_series['date'] == data['wave_start'], 'tests'].values[0]

                si_series = self.data_provider.gsi_table.loc[
                    (self.data_provider.gsi_table['countrycode'] == country) &
                    (self.data_provider.gsi_table['date'] >= data['wave_start']) &
                    (self.data_provider.gsi_table['date'] <= data['wave_end']), ['stringency_index', 'date']]

                if len(si_series) == 0:
                    data['si_integral_during_wave'] = np.nan
                else:
                    data['si_integral_during_wave'] = np.trapz(y=si_series['stringency_index'].dropna(),
                                                               x=[(a - si_series['date'].values[0]).days for a in
                                                                  si_series['date'][~np.isnan(
                                                                      si_series['stringency_index'])]])
                figure_3_wave_level = figure_3_wave_level.append(data, ignore_index=True)

                if data['class'] >= 3:
                    # Second wave
                    country_series = self.data_provider.epidemiology_series[
                        self.data_provider.epidemiology_series['countrycode'] == country]
                    data['wave'] = 2
                    data['t0_10_dead'] = np.nan
                    data['wave_start'] = self._handle_wave_start(country, 2)
                    data['wave_end'] = self._handle_wave_end(country, 2)
                    data['dead_during_wave'] = \
                        dead[(dead['date'] >= self.epi_panel.loc[
                            self.epi_panel['countrycode'] == country, 'wave_start_2'].values[0]) &
                             (dead['date'] <= self.epi_panel.loc[
                                 self.epi_panel['countrycode'] == country, 'wave_end_2'].values[0])]['dead_per_day'].sum()
                    data['tests_during_wave'] = \
                    tests[(tests['date'] >= self.epi_panel.loc[
                        self.epi_panel['countrycode'] == country, 'wave_start_2'].values[0]) &
                         (tests['date'] <= self.epi_panel.loc[
                             self.epi_panel['countrycode'] == country, 'wave_end_2'].values[0])]['new_tests'].sum()

                    dead_at_start = country_series.loc[country_series['date'] == data['wave_start'], 'dead'].values[0]
                    data['t0_10_dead'] = country_series.loc[(country_series['date'] > data['wave_start']) & \
                                                            (country_series['date'] <= data['wave_end']) & \
                                                            (country_series['dead'] >= dead_at_start + 10), \
                                                            'date']
                    if len(data['t0_10_dead']) > 0:
                        data['t0_10_dead'] = data['t0_10_dead'].values[0]
                    else:
                        data['t0_10_dead'] = np.nan
                    si_series = self.data_provider.gsi_table.loc[
                        (self.data_provider.gsi_table['countrycode'] == country) &
                        (self.data_provider.gsi_table['date'] >= data['wave_start']) &
                        (self.data_provider.gsi_table['date'] <= data['wave_end']), ['stringency_index', 'date']]
                    if len(si_series) == 0:
                        data['si_integral_during_wave'] = np.nan
                    else:
                        data['si_integral_during_wave'] = np.trapz(y=si_series['stringency_index'].dropna(),
                                                                   x=[(a - si_series['date'].values[0]).days for a in
                                                                      si_series['date'][~np.isnan(si_series['stringency_index'])]])
                    figure_3_wave_level = figure_3_wave_level.append(data, ignore_index=True)

                    if data['class'] >= 5:
                        # third wave
                        data['wave'] = 3
                        data['t0_10_dead'] = np.nan
                        data['wave_start'] = self._handle_wave_start(country, 3)
                        data['wave_end'] = self._handle_wave_end(country, 3)
                        data['dead_during_wave'] = \
                            dead[(dead['date'] >= self.epi_panel.loc[
                                self.epi_panel['countrycode'] == country, 'wave_start_3'].values[0]) &
                                 (dead['date'] <= self.epi_panel.loc[
                                     self.epi_panel['countrycode'] == country, 'wave_end_3'].values[0])]['dead_per_day'].sum()

                        data['tests_during_wave'] = \
                            tests[(tests['date'] >= self.epi_panel.loc[
                                self.epi_panel['countrycode'] == country, 'wave_start_3'].values[0]) &
                                  (tests['date'] <= self.epi_panel.loc[
                                      self.epi_panel['countrycode'] == country, 'wave_end_3'].values[0])]['new_tests'].sum()

                        dead_at_start = country_series.loc[
                            country_series['date'] == data['wave_start'], 'dead'].values[0]
                        data['t0_10_dead'] = country_series.loc[(country_series['date'] > data['wave_start']) & \
                                                                (country_series['date'] <= data['wave_end']) & \
                                                                (country_series['dead'] >= dead_at_start + 10), \
                                                                'date']
                        if len(data['t0_10_dead']) > 0:
                            data['t0_10_dead'] = data['t0_10_dead'].values[0]
                        else:
                            data['t0_10_dead'] = np.nan
                        si_series = self.data_provider.gsi_table.loc[
                            (self.data_provider.gsi_table['countrycode'] == country) &
                            (self.data_provider.gsi_table['date'] >= data['wave_start']) &
                            (self.data_provider.gsi_table['date'] <= data['wave_end']), ['stringency_index', 'date']]
                        if len(si_series) == 0:
                            data['si_integral_during_wave'] = np.nan
                        else:
                            data['si_integral_during_wave'] = np.trapz(y=si_series['stringency_index'].dropna(),
                                                                       x=[(a - si_series['date'].values[0]).days for a
                                                                          in si_series['date'][
                                                                              ~np.isnan(si_series['stringency_index'])]])
                        figure_3_wave_level = figure_3_wave_level.append(data, ignore_index=True)

        class_coarse = {
            0: 'EPI_OTHER',
            1: 'EPI_FIRST_WAVE',
            2: 'EPI_FIRST_WAVE',
            3: 'EPI_SECOND_WAVE',
            4: 'EPI_SECOND_WAVE',
            5: 'EPI_THIRD_WAVE',
            6: 'EPI_THIRD_WAVE',
            7: 'EPI_FOURTH_WAVE',
            8: 'EPI_FOURTH_WAVE',
        }

        # figure_3_total: all waves
        data = self.epi_panel[['countrycode', 'country', 'class', 't0_10_dead', 'population']].merge(
            government_response_panel[['countrycode', 'si_integral']], on='countrycode', how='left')
        data['last_confirmed'] = data['countrycode'].apply(
            lambda x: self.data_provider.get_series(x, 'confirmed')['confirmed'].iloc[-1])
        data['last_dead'] = data['countrycode'].apply(
            lambda x: self.data_provider.get_series(x, 'dead')['dead'].iloc[-1])
        data['last_tests'] = data['countrycode'].apply(
            lambda x: self.data_provider.get_series(x, 'new_tests')['new_tests'].sum())

        data['class_coarse'] = [class_coarse[x] if x in class_coarse.keys() else 'EPI_OTHER' for x in
                                data['class'].values]
        data['last_confirmed_per_10k'] = 10000 * data['last_confirmed'] / data['population']
        data['last_dead_per_10k'] = 10000 * data['last_dead'] / data['population']
        data['last_tests_per_10k'] = 10000 * data['last_tests'] / data['population']
        data['first_date_si_above_threshold'] = np.nan
        for flag in flags:
            data['first_date_' + flag[0:2] + '_above_threshold'] = np.nan
        for country in tqdm(self.epi_panel.countrycode):
            gov_country_series = government_response_series[government_response_series['countrycode'] == country]
            country_series = self.data_provider.epidemiology_series[
                self.data_provider.epidemiology_series['countrycode'] == country]
            if sum(gov_country_series['stringency_index'] >= SI_THRESHOLD) > 0:
                data.loc[data['countrycode'] == country, 'first_date_si_above_threshold'] = min(
                    gov_country_series.loc[gov_country_series['stringency_index'] >= SI_THRESHOLD, 'date'])
                if not pd.isnull(data.loc[data['countrycode'] == country, 't0_10_dead']).values[0]:
                    data.loc[data['countrycode'] == country, 'si_response_time'] = (
                                data.loc[data['countrycode'] == country, 'first_date_si_above_threshold'].values[0] -
                                data.loc[data['countrycode'] == country, 't0_10_dead'].values[0]).days
            for flag in flags:
                if sum(gov_country_series[flag] >= flag_thresholds[flag]) > 0:
                    data.loc[data['countrycode'] == country, 'first_date_' + flag[0:2] + '_above_threshold'] = min(
                        gov_country_series.loc[gov_country_series[flag] >= flag_thresholds[flag], 'date'])
                    if not pd.isnull(data.loc[data['countrycode'] == country, 't0_10_dead']).values[0]:
                        data.loc[data['countrycode'] == country, flag[0:2] + '_response_time'] = (data.loc[data[
                                                                                                               'countrycode'] == country, 'first_date_' + flag[
                                                                                                                                                          0:2] + '_above_threshold'].values[
                                                                                                      0] - data.loc[
                                                                                                      data[
                                                                                                          'countrycode'] == country, 't0_10_dead'].values[
                                                                                                      0]).days
            for t in TESTS_THRESHOLD:
                tests_threshold_pop = t * data.loc[data['countrycode'] == country, 'population'].values[0] / 10000
                if sum(country_series['tests'] >= tests_threshold_pop) > 0:
                    data.loc[data['countrycode'] == country, 'first_date_tests_above_threshold_' + str(t)] = min(
                        country_series.loc[country_series['tests'] >= tests_threshold_pop, 'date'])
                    if not pd.isnull(data.loc[data['countrycode'] == country, 't0_10_dead']).values[0]:
                        data.loc[data['countrycode'] == country, 'testing_response_time_' + str(t)] = \
                            (data.loc[data[
                                                                                                                    'countrycode'] == country, 'first_date_tests_above_threshold_' + str(
                            t)].values[0] - data.loc[data['countrycode'] == country, 't0_10_dead'].values[0]).days

        figure_3_all = figure_3_wave_level.merge(
            data[['countrycode', 'class_coarse', 'si_integral', 'last_dead_per_10k', 'last_tests_per_10k',
                  'si_response_time',
                  'c1_response_time', 'c2_response_time', 'c3_response_time', 'c4_response_time', 'c5_response_time',
                  'c6_response_time',
                  'c7_response_time', 'c8_response_time', 'h2_response_time', 'h3_response_time'] + [
                     'testing_response_time_' + str(t) for t in TESTS_THRESHOLD]]
            , on='countrycode', how='left')

        figure_3_all.to_csv(os.path.join(self.data_dir, 'figure_3_all.csv'))
        return

    def _figure_deaths(self):
        out = pd.DataFrame(columns=['countrycode', 'date', 'location', 'dead_per_day_smooth'])
        for k, v in self.epi_classifier.deaths_summary_output.items():
            if len(v) == 0:
                continue
            for i, wave in enumerate(v):
                if wave['peak_ind'] == 0:
                    continue
                upsert = {}
                upsert['countrycode'] = k
                upsert['date'] = wave['date']
                upsert['location'] = wave['location']
                upsert['dead_per_day_smooth'] = wave['y_position']
                out = out.append(upsert, ignore_index=True)
            continue
        out.to_csv(os.path.join(self.data_dir, 'figure_dead_all.csv'))
        return

    def main(self):
        self._figure_1()
        self._figure_2()
        self._figure_3()
        self._figure_4()
        self._figure_deaths()
        return
