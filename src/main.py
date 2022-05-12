import os
import pickle
from tqdm import tqdm
from epidemicwaveclassifier import EpidemicWaveClassifier
from data_provider import DataProvider
from config import Config
from waveanalysispanel import WaveAnalysisPanel
from table_1 import Table1
from figures import Figures
from manuscriptfigures import ManuscriptFigures


if __name__ == '__main__':
    config = Config(os.path.dirname(os.path.realpath(__file__)))

    data_provider = DataProvider(config)
    data_provider.fetch_data(use_cache=True)
    countries = data_provider.get_countries()

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

        if country == 'GHA':
            figure_3 = (cases, deaths)
            file = os.path.join(config.manuscript_figures_data_path, 'figure3.pkl')
            with open('file', 'wb') as handle:
                pickle.dump(figure_3, handle)
                print(f'Data for Figure 3 written to {file}')

    wave_analysis_panel = WaveAnalysisPanel(
        config, data_provider, epidemic_wave_classifier.summary_output).get_epi_panel()

    table_1 = Table1(config, wave_analysis_panel)
    table_1.table_1()
    #figures = Figures(config, wave_analysis_panel, data_provider, epidemic_wave_classifier)
    #figures.main()
