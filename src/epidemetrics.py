import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import datetime
from pandas import DataFrame
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import json
from data_provider import DataProvider
from implementation.pre_algo import PreAlgo
from implementation.algorithm_a import AlgorithmA
from implementation.algorithm_b import AlgorithmB
from implementation.algorithm_c import AlgorithmC
from implementation.algorithm_e import AlgorithmE
from implementation.config import Config
from plot_helper import plot_peaks


class Epidemetrics:
    def __init__(self, config: Config, data_provider: DataProvider):
        self.config = config
        self.data_provider = data_provider
        self.prepare_output_dirs(self.config.plot_path)
        self.summary_output = dict()

        self.pre_algo = PreAlgo(self.config, self.data_provider)

    def prepare_output_dirs(self, path: str):
        print(f'Preparing output directory: {path}')
        try:
            shutil.rmtree(path)
        except OSError as e:
            print(f"Error: {path}: {e.strerror}")
        Path(path).mkdir(parents=True, exist_ok=True)

    def find_peaks(self, country: str, field: str) -> DataFrame:

        data, peaks_initial, peaks_cleaned, prominence_updater = self.pre_algo.run(country=country, field=field)

        peaks_sub_a = AlgorithmA(self.config).run(
            input_data_df=peaks_cleaned,
            prominence_updater=prominence_updater)

        peaks_sub_b = AlgorithmB(self.config).run(
            raw_data=data[field],
            input_data_df=peaks_sub_a,
            prominence_updater=prominence_updater)

        peaks_sub_c = AlgorithmC(self.config, self.data_provider, country, field).run(
            raw_data=data[field],
            input_data_df=peaks_sub_b,
        )

        return peaks_initial, peaks_cleaned, peaks_sub_a, peaks_sub_b, peaks_sub_c

    def epi_find_peaks(self, country: str, plot: bool = False, save: bool = False) -> DataFrame:
        # match parameter tries to use death waves to detect case waves under sub_algorithm_e
        cases = self.data_provider.get_series(country=country, field='new_per_day_smooth')
        if len(cases) == 0:
            raise ValueError

        cases_initial, cases_cleaned, cases_sub_a, cases_sub_b, cases_sub_c = self.find_peaks(
            country, field='new_per_day_smooth')

        # compute equivalent series for deaths
        deaths = self.data_provider.get_series(country=country, field='dead_per_day_smooth')
        if len(deaths) == 0:
            raise ValueError

        deaths_initial, deaths_cleaned, deaths_sub_a, deaths_sub_b, deaths_sub_c = self.find_peaks(
            country, field='dead_per_day_smooth')

        # run sub algorithm e
        cases_sub_e = AlgorithmE(self.config, self.data_provider, country).run(
            cases_sub_b, cases_sub_c, deaths_sub_c, plot=plot)

        # compute plots
        if plot:
            plot_peaks(cases, deaths, country, cases_initial, cases_cleaned, cases_sub_a, cases_sub_b, cases_sub_c,
                       deaths_initial, deaths_cleaned, deaths_sub_a, deaths_sub_b, deaths_sub_c,
                       self.config.plot_path, save)

        summary = []
        for row, peak in cases_sub_e.iterrows():
            peak_data = dict({"index": row, "location": peak.location, "date": cases.iloc[int(peak.location)].date,
                              "peak_ind": peak.peak_ind, "y_position": peak.y_position})
            summary.append(peak_data)
        self.summary_output[country] = summary

        return cases_sub_e

    def save_summary(self):
        json_data = dict({'data': []})
        for country, summary in self.summary_output.items():
            country_summary = dict({'country': country, 'waves': []})
            for wave in summary:
                copied_wave = wave.copy()
                copied_wave['date'] = copied_wave['date'].strftime('%Y-%m-%d')
                country_summary['waves'].append(copied_wave)
            json_data['data'].append(country_summary)
        with open(os.path.join(self.config.plot_path, 'summary_output.json'), 'w') as f:
            json.dump(json_data, f)
