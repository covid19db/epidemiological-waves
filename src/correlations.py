import os
import subprocess

import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm

from config import Config
from data_provider import DataProvider
from epidemicwaveclassifier import EpidemicWaveClassifier


class Correlations:
    def __init__(self, config: Config, epi_panel: pd.core.frame.DataFrame,
                 data_provider: DataProvider, epi_classifier: EpidemicWaveClassifier):
        self.data_dir = config.manuscript_figures_data_path
        self.config = config
        self.data_provider = data_provider
        self.epi_panel = epi_panel
        self.epi_classifier = epi_classifier
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

        government_response_panel = pd.DataFrame(columns=(
            ['countrycode', 'country', 'max_si', 'date_max_si', 'si_days_to_max_si', 'si_at_t0', 'si_at_peak_1',
             'si_days_to_threshold', 'si_days_above_threshold', 'si_days_above_threshold_first_wave', 'si_integral']
            + [flag + '_at_t0' for flag in flags]
            + [flag + '_at_peak_1' for flag in flags]
            + [flag + '_days_to_threshold' for flag in flags]
            + [flag + '_days_above_threshold' for flag in flags]
            + [flag + '_days_above_threshold_first_wave' for flag in flags]
            + [flag + '_raised' for flag in flags]
            + [flag + '_lowered' for flag in flags]
            + [flag + '_raised_again' for flag in flags]
        ))

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
            t0 = (np.nan if len(self.epi_panel[self.epi_panel['countrycode'] == country]['t0_10_dead']) == 0
                  else self.epi_panel[self.epi_panel['countrycode'] == country]['t0_10_dead'].iloc[0])
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
                        (min(country_series.loc[country_series['stringency_index'] >= SI_THRESHOLD, 'date']) - t0).days
                        if sum(country_series['stringency_index'] >= SI_THRESHOLD) > 0
                        else np.nan)
                    for flag in flags:
                        data[flag + '_at_t0'] = country_series.loc[country_series['date'] == t0, flag].values[0]
                        data[flag + '_days_to_threshold'] = (
                            (min(country_series.loc[country_series[flag] >= flag_thresholds[flag], 'date']) - t0).days
                            if sum(country_series[flag] >= flag_thresholds[flag]) > 0
                            else np.nan)
                if (not (pd.isnull(date_peak_1) or pd.isnull(first_wave_start) or pd.isnull(first_wave_end))
                        and date_peak_1 in country_series['date']):
                    # SI value at peak date
                    data['si_at_peak_1'] = country_series.loc[
                        country_series['date'] == date_peak_1, 'stringency_index'].values[0]
                    # number of days SI above the threshold during the first wave
                    data['si_days_above_threshold_first_wave'] = sum(
                        (country_series['stringency_index'] >= SI_THRESHOLD) &
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
        wave_level = pd.DataFrame()
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
                data['wave_start'] = self.epi_panel.loc[
                    self.epi_panel['countrycode'] == country, 'wave_start_1'].values[0]
                data['wave_end'] = self._handle_wave_end(country, 1)
                data['t0_10_dead'] = self.epi_panel.loc[
                    self.epi_panel['countrycode'] == country, 't0_10_dead'].values[0]
                data['dead_during_wave'] = (
                    dead[
                        (dead['date'] >=
                         self.epi_panel.loc[self.epi_panel['countrycode'] == country, 'wave_start_1'].values[0]) &
                        (dead['date'] <=
                         self.epi_panel.loc[self.epi_panel['countrycode'] == country, 'wave_end_1'].values[0])
                    ]['dead_per_day'].sum())
                data['tests_during_wave'] = (
                    tests[
                        (tests['date'] >=
                         self.epi_panel.loc[self.epi_panel['countrycode'] == country, 'wave_start_1'].values[0]) &
                        (tests['date'] <=
                         self.epi_panel.loc[self.epi_panel['countrycode'] == country, 'wave_end_1'].values[0])
                    ]['new_tests'].sum())

                # if tests during first wave is na due to missing data, linear interpolate low test numbers
                if pd.isnull(data['tests_during_wave']):
                    country_series = self.epi_panel[self.epi_panel['countrycode'] == country]
                    if not pd.isnull(data['wave_start']) and not np.all(pd.isnull(country_series['tests'])):
                        min_date = min(country_series['date'])
                        min_tests = np.nanmin(country_series['tests'])
                        if (pd.isnull(country_series.loc[country_series['date'] == min_date, 'tests'].values[0]) and
                                min_tests <= 1000):
                            country_series.loc[country_series['date'] == min_date, 'tests'] = 0
                            country_series['tests'] = country_series['tests'].interpolate(method='linear')
                        if (not pd.isnull(country_series.loc[country_series['date'] == data['wave_start'], 'tests']
                                                  .values[0])
                            and not pd.isnull(country_series.loc[country_series['date'] == data['wave_end'], 'tests']
                                                      .values[0])):
                            data['tests_during_wave'] = (
                                    country_series.loc[country_series['date'] == data['wave_end'], 'tests'].values[0] -
                                    country_series.loc[country_series['date'] == data['wave_start'], 'tests'].values[0])

                si_series = self.data_provider.gsi_table.loc[
                    (self.data_provider.gsi_table['countrycode'] == country) &
                    (self.data_provider.gsi_table['date'] >= data['wave_start']) &
                    (self.data_provider.gsi_table['date'] <= data['wave_end']), ['stringency_index', 'date']]

                if len(si_series) == 0:
                    data['si_integral_during_wave'] = np.nan
                else:
                    data['si_integral_during_wave'] = np.trapz(
                        y=si_series['stringency_index'].dropna(),
                        x=[
                            (a - si_series['date'].values[0]).days
                            for a in
                            si_series['date'][~np.isnan(si_series['stringency_index'])]
                        ]
                    )
                wave_level = wave_level.append(data, ignore_index=True)

                if data['class'] >= 3:
                    # Second wave
                    country_series = self.data_provider.epidemiology_series[
                        self.data_provider.epidemiology_series['countrycode'] == country]
                    data['wave'] = 2
                    data['t0_10_dead'] = np.nan
                    data['wave_start'] = self._handle_wave_start(country, 2)
                    data['wave_end'] = self._handle_wave_end(country, 2)
                    data['dead_during_wave'] = (
                        dead[(dead['date'] >=
                              self.epi_panel.loc[self.epi_panel['countrycode'] == country, 'wave_start_2'].values[0]) &
                             (dead['date'] <=
                              self.epi_panel.loc[self.epi_panel['countrycode'] == country, 'wave_end_2'].values[0])
                        ]['dead_per_day'].sum())
                    data['tests_during_wave'] = (
                        tests[(tests['date'] >=
                               self.epi_panel.loc[self.epi_panel['countrycode'] == country, 'wave_start_2'].values[0]) &
                              (tests['date'] <=
                               self.epi_panel.loc[self.epi_panel['countrycode'] == country, 'wave_end_2'].values[0])
                        ]['new_tests'].sum())

                    dead_at_start = country_series.loc[country_series['date'] == data['wave_start'], 'dead'].values[0]
                    data['t0_10_dead'] = (country_series.loc[(country_series['date'] > data['wave_start'])
                                                             & (country_series['date'] <= data['wave_end'])
                                                             & (country_series['dead'] >= dead_at_start + 10), 'date'])
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
                        data['si_integral_during_wave'] = np.trapz(
                            y=si_series['stringency_index'].dropna(),
                            x=[(a - si_series['date'].values[0]).days
                               for a in
                               si_series['date'][~np.isnan(si_series['stringency_index'])]])
                    wave_level = wave_level.append(data, ignore_index=True)

                    if data['class'] >= 5:
                        # third wave
                        data['wave'] = 3
                        data['t0_10_dead'] = np.nan
                        data['wave_start'] = self._handle_wave_start(country, 3)
                        data['wave_end'] = self._handle_wave_end(country, 3)
                        data['dead_during_wave'] = (
                            dead[
                                (dead['date'] >=
                                 self.epi_panel.loc[self.epi_panel['countrycode'] == country, 'wave_start_3'].values[0])
                                & (dead['date'] <=
                                   self.epi_panel.loc[self.epi_panel['countrycode'] == country, 'wave_end_3'].values[0])
                            ]['dead_per_day'].sum())

                        data['tests_during_wave'] = (
                            tests[
                                (tests['date'] >=
                                 self.epi_panel.loc[self.epi_panel['countrycode'] == country, 'wave_start_3'].values[0])
                                & (tests['date'] <=
                                   self.epi_panel.loc[self.epi_panel['countrycode'] == country, 'wave_end_3'].values[0])
                            ]['new_tests'].sum())

                        dead_at_start = country_series.loc[
                            country_series['date'] == data['wave_start'], 'dead'].values[0]
                        data['t0_10_dead'] = (country_series.loc[
                            (country_series['date'] > data['wave_start'])
                            & (country_series['date'] <= data['wave_end'])
                            & (country_series['dead'] >= dead_at_start + 10), 'date'])
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
                            data['si_integral_during_wave'] = np.trapz(
                                y=si_series['stringency_index'].dropna(),
                                x=[(a - si_series['date'].values[0]).days
                                   for a in
                                   si_series['date'][~np.isnan(si_series['stringency_index'])]])
                        wave_level = wave_level.append(data, ignore_index=True)

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
                        data.loc[data['countrycode'] == country, flag[0:2] + '_response_time'] = (
                            (data.loc[data['countrycode'] == country,
                                      'first_date_' + flag[0:2] + '_above_threshold'].values[0]
                             - data.loc[data['countrycode'] == country, 't0_10_dead'].values[0]).days)
            for t in TESTS_THRESHOLD:
                tests_threshold_pop = t * data.loc[data['countrycode'] == country, 'population'].values[0] / 10000
                if sum(country_series['tests'] >= tests_threshold_pop) > 0:
                    data.loc[data['countrycode'] == country, 'first_date_tests_above_threshold_' + str(t)] = min(
                        country_series.loc[country_series['tests'] >= tests_threshold_pop, 'date'])
                    if not pd.isnull(data.loc[data['countrycode'] == country, 't0_10_dead']).values[0]:
                        data.loc[data['countrycode'] == country, 'testing_response_time_' + str(t)] = (
                            (data.loc[data['countrycode'] == country,
                                      'first_date_tests_above_threshold_' + str(t)].values[0]
                             - data.loc[data['countrycode'] == country, 't0_10_dead'].values[0]).days)

        all_data = wave_level.merge(
            data[['countrycode', 'class_coarse', 'si_integral', 'last_dead_per_10k', 'last_tests_per_10k',
                  'si_response_time', 'c1_response_time', 'c2_response_time', 'c3_response_time', 'c4_response_time',
                  'c5_response_time', 'c6_response_time', 'c7_response_time', 'c8_response_time', 'h2_response_time',
                  'h3_response_time']
                 + ['testing_response_time_' + str(t) for t in TESTS_THRESHOLD]],
            on='countrycode',
            how='left')

        all_data.to_csv(os.path.join(self.data_dir, 'correlations.csv'))
        return

    def main(self):
        print(r"Preparing correlation data - run manuscriptfigures/run.sh to generate table of correlations")
        self._figure_3()
        return
