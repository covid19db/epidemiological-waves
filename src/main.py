import os
from tqdm import tqdm
from epidemicwaveclassifier import EpidemicWaveClassifier
from data_provider import DataProvider
from config import Config
from waveanalysispanel import WaveAnalysisPanel
from table_1 import Table1
from figures import Figures
from correlations import Correlations
from manuscriptfigures.manuscriptfigures import ManuscriptFigures


if __name__ == '__main__':
    config = Config(os.path.dirname(os.path.realpath(__file__)))

    data_provider = DataProvider(config)
    data_provider.fetch_data(use_cache=True)
    countries = data_provider.get_countries()

    # collect some data for manuscript plots
    countries_for_manuscript = ['ZMB', 'GHA', 'GBR', 'CRI']
    manuscript_data = dict()

    epidemic_wave_classifier = EpidemicWaveClassifier(config, data_provider)

    t = tqdm(countries, desc='Finding peaks for all countries')
    for country in t:
        t.set_description(f"Finding peaks for: {country}")
        t.refresh()
        try:
            cases, deaths, cross = epidemic_wave_classifier.epi_find_peaks(country, plot=True, save=True)
        except ValueError:
            print(f'Unable to find peaks for: {country}')
        except KeyboardInterrupt:
            exit()

        if country in countries_for_manuscript:
            manuscript_data[country] = (cases, deaths)

    wave_analysis_panel = WaveAnalysisPanel(
        config, data_provider, epidemic_wave_classifier.summary_output).get_epi_panel()

    Table1(config, wave_analysis_panel).table_1()
    Figures(config, wave_analysis_panel, data_provider, epidemic_wave_classifier).main()
    Correlations(config, wave_analysis_panel, data_provider, epidemic_wave_classifier).main()
    ManuscriptFigures(config, data_provider, manuscript_data).main()
