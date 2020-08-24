import numpy as np
import matplotlib.pyplot as plt
import psycopg2
import pandas as pd
import geopandas as gpd
import seaborn as sns
import os
import warnings
from tqdm import tqdm
import datetime
import re

from scipy.signal import find_peaks

warnings.filterwarnings('ignore')

'''Initialise script parameters'''

DISTANCE = 21
PROMINENCE_THRESHOLD = 5
MIN_PERIOD = 1
PATH = './charts/table_figures/'
#PATH = './'

conn = psycopg2.connect(
    host='covid19db.org',
    port=5432,
    dbname='covid19',
    user='covid19',
    password='covid19')
cur = conn.cursor()

# GET EPIDEMIOLOGY TABLE
source = "WRD_ECDC"
exclude = ['Other continent', 'Asia', 'Europe', 'America', 'Africa', 'Oceania']

sql_command = """SELECT * FROM epidemiology WHERE source = %(source)s"""
raw_epidemiology = pd.read_sql(sql_command, conn, params={'source': source})
raw_epidemiology = raw_epidemiology[raw_epidemiology['adm_area_1'].isnull()].sort_values(by=['countrycode', 'date'])
raw_epidemiology = raw_epidemiology[~raw_epidemiology['country'].isin(exclude)].reset_index(drop=True)
raw_epidemiology = raw_epidemiology[['countrycode','country','date','confirmed']]
### Check no conflicting values for each country and date
assert not raw_epidemiology[['countrycode','date']].duplicated().any()

# GET MOBILITY TABLE
source='GOOGLE_MOBILITY'
mobilities=['residential']

sql_command = """SELECT * FROM mobility WHERE source = %(source)s AND adm_area_1 is NULL"""
raw_mobility = pd.read_sql(sql_command, conn, params={'source': source})
raw_mobility = raw_mobility[['countrycode','country','date']+mobilities]
raw_mobility = raw_mobility.sort_values(by=['countrycode','date']).reset_index(drop=True)
### Check no conflicting values for each country and date
assert not raw_mobility[['countrycode','date']].duplicated().any()

# GET GOVERNMENT RESPONSE TABLE
flags=['stringency_index','c1_school_closing','c2_workplace_closing','c6_stay_at_home_requirements']
flag_conversion = {'c1_school_closing': 2, 'c2_workplace_closing': 2, 'c6_stay_at_home_requirements': 2}

sql_command = """SELECT * FROM government_response"""
raw_government_response = pd.read_sql(sql_command, conn)
raw_government_response = raw_government_response.sort_values(by=['countrycode','date']).reset_index(drop=True)
raw_government_response = raw_government_response[['countrycode','country','date']+flags]
raw_government_response = raw_government_response.sort_values(by=['country','date'])\
    .drop_duplicates(subset=['countrycode','date'])#.dropna(subset=['stringency_index'])
raw_government_response = raw_government_response.sort_values(by=['countrycode','date'])
### Check no conflicting values for each country and date
assert not raw_government_response[['countrycode','date']].duplicated().any()

# GET COUNTRY STATS TABLE
required_stats = ['Population, total','Surface area (sq. km)']
sql_command = """SELECT * FROM world_bank"""
raw_country_stats = pd.read_sql(sql_command, conn)
raw_country_stats = raw_country_stats[raw_country_stats['indicator_name'].isin(required_stats)]

# GET ADMINISTRATIVE DIVISIONS
sql_command = """SELECT * FROM administrative_division WHERE adm_level=0"""
map_data = gpd.GeoDataFrame.from_postgis(sql_command, conn, geom_col='geometry')[['countrycode','geometry']]

##EPIDEMIOLOGY PRE-PROCESSING LOOP
countries = raw_epidemiology['countrycode'].unique()
epidemiology = pd.DataFrame(columns=['countrycode','country','date','confirmed','new_per_day','days_since_first'])
for country in countries:
    data = raw_epidemiology[raw_epidemiology['countrycode']==country].set_index('date')
    data = data.reindex([x.date() for x in pd.date_range(data.index.values[0],data.index.values[-1])])
    data[['countrycode','country']] = data[['countrycode','country']].fillna(method='backfill')
    data['confirmed'] = data['confirmed'].interpolate(method='linear')
    data['new_per_day'] = data['confirmed'].diff()
    data.reset_index(inplace=True)
    data['new_per_day'].iloc[np.array(data[data['new_per_day']<0].index)] = \
        data['new_per_day'].iloc[np.array(epidemiology[epidemiology['new_per_day']<0].index)-1]
    data['new_per_day'] = data['new_per_day'].fillna(method='bfill')
    days_since_first = np.zeros(len(data))
    placeholder_days_since_first = np.arange(1,len(data[data['confirmed']>0])+1)
    days_since_first[-len(placeholder_days_since_first)::] = placeholder_days_since_first
    data['days_since_first'] = days_since_first
    epidemiology = pd.concat((epidemiology,data)).reset_index(drop=True)
    continue

LABELLED_COLUMNS = pd.read_csv('./peak_labels.csv')

CLASS_DICTIONARY = {
    'EPI_ENTERING_FIRST' : 1,
    'EPI_PAST_FIRST' : 2,
    'EPI_ENTERING_SECOND' : 3,
    'EPI_PAST_SECOND' : 4
}

classes = np.zeros(len(LABELLED_COLUMNS))
for k, v in CLASS_DICTIONARY.items():
    classes[np.where(LABELLED_COLUMNS[k])] += v
LABELLED_COLUMNS['CLASS'] = classes

epidemiology = epidemiology.merge(LABELLED_COLUMNS[['COUNTRYCODE','CLASS']], left_on = ['countrycode'],
                   right_on = ['COUNTRYCODE'], how = 'left')

##MOBILITY PRE-PROCESSING LOOP
countries = raw_mobility['countrycode'].unique()
mobility = pd.DataFrame(columns=['countrycode','country','date']+mobilities)

for country in countries:
    data = raw_mobility[raw_mobility['countrycode']==country].set_index('date')
    data = data.reindex([x.date() for x in pd.date_range(data.index.values[0],data.index.values[-1])])
    data[['countrycode','country']] = data[['countrycode','country']].fillna(method='backfill')
    data[mobilities] = data[mobilities].interpolate(method='linear')
    data.reset_index(inplace=True)
    mobility = pd.concat((mobility,data)).reset_index(drop=True)
    continue

##GOVERNMENT_RESPONSE PRE-PROCESSING LOOP
countries = raw_government_response['countrycode'].unique()
government_response = pd.DataFrame(columns=['countrycode','country','date']+flags)

for country in countries:
    data = raw_government_response[raw_government_response['countrycode']==country].set_index('date')
    data = data.reindex([x.date() for x in pd.date_range(data.index.values[0],data.index.values[-1])])
    data[['countrycode','country']] = data[['countrycode','country']].fillna(method='backfill')
    data[flags] = data[flags].fillna(method='ffill')
    data.reset_index(inplace=True)
    government_response = pd.concat((government_response,data)).reset_index(drop=True)
    continue

##COUNTRY STATS PRE-PROCESSING LOOP
countries = raw_country_stats['countrycode'].unique()
country_stats = pd.DataFrame(columns = ['countrycode','country']+required_stats)
for country in countries:
    data = raw_country_stats[raw_country_stats['countrycode']==country]
    data = data[['countrycode','country','indicator_name','value']]
    stats = {k:0 for k in required_stats}
    for stat in required_stats:
        stats[stat] = [data[data['indicator_name']==stat]['value'].values[0]]
    stats['countrycode'] = [data['countrycode'].iloc[0]]
    stats['country'] = [data['country'].iloc[0]]
    stats = pd.DataFrame.from_dict(stats)
    country_stats = pd.concat((country_stats,stats)).reset_index(drop=True)


countries = np.union1d(epidemiology['countrycode'].unique(),
                       np.union1d(government_response['countrycode'].unique(),country_stats['countrycode'].unique()))

exclude = []
for country in countries:
    if len(epidemiology[epidemiology['countrycode'] == country]) == 0 \
            or len(government_response[government_response['countrycode'] == country]) == 0\
            or len(country_stats[country_stats['countrycode'] == country]) == 0:
        exclude.append(country)


epidemiology = epidemiology[~epidemiology['countrycode'].isin(exclude)]
epidemiology = epidemiology[epidemiology['countrycode'].isin(countries)]
government_response = government_response[~government_response['countrycode'].isin(exclude)]
government_response = government_response[government_response['countrycode'].isin(countries)]
country_stats = country_stats[~country_stats['countrycode'].isin(exclude)]
country_stats = country_stats[country_stats['countrycode'].isin(countries)]

## Processing epidemiology

EPI = {
    'countrycode':np.empty(0),
    'country':np.empty(0),
    'date':np.empty(0),
    'confirmed':np.empty(0),
    'class':np.empty(0),
    'new_per_day':np.empty(0),
    'new_per_day_per_10k':np.empty(0),
    'days_since_first':np.empty(0),
    'days_since_50_cases':np.empty(0),
    'days_since_T0':np.empty(0)
}

day_first_case = epidemiology[epidemiology['countrycode']=='CHN']['date'].iloc[0]
for country in epidemiology['countrycode'].unique():
    data = epidemiology[epidemiology['countrycode']==country]
    if len(country_stats[country_stats['countrycode']==country]) == 0 or len(data[data['confirmed']>=50]['date']) == 0:
        continue
    EPI['countrycode'] = np.concatenate((EPI['countrycode'], data['countrycode'].values))
    EPI['country'] = np.concatenate((EPI['country'], data['country'].values))
    EPI['date'] = np.concatenate((EPI['date'], data['date'].values))
    EPI['confirmed'] = np.concatenate((EPI['confirmed'], data['confirmed'].values))
    EPI['class'] = np.concatenate((EPI['class'], data['CLASS'].values))
    EPI['new_per_day'] = np.concatenate((EPI['new_per_day'], data['new_per_day'].values))
    EPI['days_since_first'] = np.concatenate((EPI['days_since_first'], data['days_since_first'].values))

    EPI['new_per_day_per_10k'] = np.concatenate((
        EPI['new_per_day_per_10k'],
        (data['new_per_day'].values /
         country_stats[country_stats['countrycode']==country]['Population, total'].values) * 10000))

    #np.max(data['new_per_day'].values)

    date_of_50 = data[data['confirmed']>=50]['date'].iloc[0]
    EPI['days_since_50_cases'] = np.concatenate((EPI['days_since_50_cases'],
                                                   np.concatenate((np.arange(-len(data[data['date']<date_of_50]),0),
                                                                   np.arange(len(data[data['date']>=date_of_50]))))))
    EPI['days_since_T0'] = np.concatenate((EPI['days_since_T0'],
                                           np.repeat((date_of_50 - day_first_case).days, len(data))))


EPI = pd.DataFrame.from_dict(EPI)

## Processing Gov Response

GOV = {
    'countrycode':np.empty(0),
    'country':np.empty(0),
    'last_confirmed':np.empty(0),
}

for flag in flag_conversion.keys():
    GOV[flag + '_date_raised'] = np.empty(0)
    GOV[flag + '_date_lowered'] = np.empty(0)
    GOV[flag + '_date_raised_again'] = np.empty(0)
    GOV[flag + '_more?'] = np.empty(0)
    GOV[flag + '_days_since_50_raised'] = np.empty(0)
    GOV[flag + '_days_since_50_lowered'] = np.empty(0)
    GOV[flag + '_days_since_50_raised_again'] = np.empty(0)
    GOV[flag + '_date_raised_cases_day'] = np.empty(0)
    GOV[flag + '_date_lowered_cases_day'] = np.empty(0)
    GOV[flag + '_date_raised_again_cases_day'] = np.empty(0)

countries = EPI['countrycode'].unique()
for country in countries:
    data = government_response[government_response['countrycode']==country]
    if (len(data) == 0) or (len(epidemiology[epidemiology['countrycode']==country]) == 0):
        continue
    GOV['countrycode'] = np.concatenate((GOV['countrycode'], [data['countrycode'].values[0]]))
    GOV['country'] = np.concatenate((GOV['country'], [data['country'].values[0]]))
    GOV['last_confirmed'] = np.concatenate((GOV['last_confirmed'],
                                            [epidemiology[epidemiology['countrycode']==country]['confirmed'].iloc[-1]]))
    for flag in flag_conversion.keys():
        days_above = (data[flag] >= flag_conversion[flag]).astype(int)
        waves = [[cat[1], grp.shape[0]] for cat, grp in
                 days_above.groupby([days_above.ne(days_above.shift()).cumsum(), days_above])]

        for i, wave in enumerate(waves):
            if wave[0] == 0:
                continue
            if wave[1] < MIN_PERIOD:
                waves[i-1][1] += waves[i][1]
                del waves[i]

        if len(waves) <= 1:
            GOV[flag + '_date_raised'] = np.concatenate((GOV[flag + '_date_raised'], [np.nan]))
            GOV[flag + '_date_lowered'] = np.concatenate((GOV[flag + '_date_lowered'], [np.nan]))
            GOV[flag + '_date_raised_again'] = np.concatenate((GOV[flag + '_date_raised_again'], [np.nan]))
            GOV[flag + '_more?'] = np.concatenate((GOV[flag + '_more?'], [np.nan]))
            GOV[flag + '_days_since_50_raised'] = np.concatenate((GOV[flag + '_days_since_50_raised'], [np.nan]))
            GOV[flag + '_days_since_50_lowered'] = np.concatenate((GOV[flag + '_days_since_50_lowered'], [np.nan]))
            GOV[flag + '_days_since_50_raised_again'] = np.concatenate((GOV[flag + '_days_since_50_raised_again'], [np.nan]))
            GOV[flag + '_date_raised_cases_day'] = np.concatenate((GOV[flag + '_date_raised_cases_day'], [np.nan]))
            GOV[flag + '_date_lowered_cases_day'] = np.concatenate((GOV[flag + '_date_lowered_cases_day'], [np.nan]))
            GOV[flag + '_date_raised_again_cases_day'] = np.concatenate((GOV[flag + '_date_raised_again_cases_day'], [np.nan]))
            continue

        date_raised = data['date'].iloc[waves[0][1] - 1] \
            if data['date'].iloc[waves[0][1] - 1] != None else np.nan
        date_lowered = data['date'].iloc[waves[1][1]+waves[0][1] - 1] \
            if data['date'].iloc[waves[1][1]+waves[0][1] - 1] != None else np.nan
        date_raised_again = data['date'].iloc[sum([waves[x][1] for x in range(3)]) - 1] \
            if len(waves) >= 4 else np.nan
        more = True if len(waves) >= 5 else False

        days_since_50_raised = \
            EPI[(EPI['countrycode'] == country) & (EPI['date'] == date_raised)]['days_since_50_cases'].values
        days_since_50_raised = [np.nan] if len(days_since_50_raised) == 0 else days_since_50_raised

        days_since_50_lowered = \
            EPI[(EPI['countrycode'] == country) & (EPI['date'] == date_lowered)]['days_since_50_cases'].values
        days_since_50_lowered = [np.nan] if len(days_since_50_lowered) == 0 else days_since_50_lowered

        days_since_50_raised_again = \
            EPI[(EPI['countrycode'] == country) & (EPI['date'] == date_raised_again)]['days_since_50_cases'].values \
                if len(waves) >= 4 else [np.nan]
        days_since_50_raised_again = [np.nan] if len(days_since_50_raised_again) == 0 else days_since_50_raised_again

        date_raised_cases_day = \
            EPI[(EPI['countrycode'] == country) & (EPI['date'] == date_raised)]['new_per_day'].values
        date_raised_cases_day = [np.nan] if len(date_raised_cases_day) == 0 else date_raised_cases_day

        date_lowered_cases_day = \
            EPI[(EPI['countrycode'] == country) & (EPI['date'] == date_lowered)]['new_per_day'].values
        date_lowered_cases_day = [np.nan] if len(date_lowered_cases_day) == 0 else date_lowered_cases_day

        date_raised_again_cases_day = \
            EPI[(EPI['countrycode'] == country) & (EPI['date'] == date_raised_again)]['new_per_day'].values \
                if len(waves) >= 4 else [np.nan]
        date_raised_again_cases_day = np.nan if len(date_raised_again_cases_day) == 0 else date_raised_again_cases_day

        GOV[flag + '_date_raised'] = np.concatenate((GOV[flag + '_date_raised'], [date_raised]))
        GOV[flag + '_date_lowered'] = np.concatenate((GOV[flag + '_date_lowered'], [date_lowered]))
        GOV[flag + '_date_raised_again'] = np.concatenate((GOV[flag + '_date_raised_again'], [date_raised_again]))
        GOV[flag + '_more?'] = np.concatenate((GOV[flag + '_more?'], [more]))
        GOV[flag + '_days_since_50_raised'] = np.concatenate((GOV[flag + '_days_since_50_raised'], days_since_50_raised))
        GOV[flag + '_days_since_50_lowered'] = np.concatenate((GOV[flag + '_days_since_50_lowered'], days_since_50_lowered))
        GOV[flag + '_days_since_50_raised_again'] = np.concatenate(
            (GOV[flag + '_days_since_50_raised_again'], days_since_50_raised_again))
        GOV[flag + '_date_raised_cases_day'] = np.concatenate((GOV[flag + '_date_raised_cases_day'], date_raised_cases_day))
        GOV[flag + '_date_lowered_cases_day'] = np.concatenate((GOV[flag + '_date_lowered_cases_day'], date_lowered_cases_day))
        GOV[flag + '_date_raised_again_cases_day'] = np.concatenate(
            (GOV[flag + '_date_raised_again_cases_day'], date_raised_again_cases_day))

GOV = pd.DataFrame.from_dict(GOV)
GOV = GOV.merge(EPI.drop_duplicates(subset=['countrycode','class'])
                [['countrycode','class']], on = ['countrycode'], how = 'left')
GOV['class'] = GOV['class'].astype(int)


"""
chosen_flag = 'c6_stay_at_home_requirements'
f, ax = plt.subplots(figsize=(20, 7))
plt.ylim(0,0.002)
plt.xlim(0,10000)
sns.distplot(GOV[chosen_flag + '_date_raised_cases_day'],
             bins = 1000, hist=False, label = 'Flag raised')
sns.distplot(GOV[chosen_flag + '_date_lowered_cases_day'],
             bins = 1000, hist=False, label = 'Flag lowered')
sns.distplot(GOV[chosen_flag + '_date_raised_again_cases_day'],
             bins = 1000, hist=False, label = 'Flag raised again')
plt.savefig(PATH + 'flag_c6_new_cases_per_day.png')
"""

### FIGURE 1

figure_1 = map_data.merge(EPI.drop_duplicates(
    subset = ['countrycode'])[['countrycode', 'days_since_T0']], how = 'left').dropna()

plt.figure(figsize = (20,10))
figure_1.plot(column = 'days_since_T0', cmap='inferno', edgecolor = 'black',
              linewidth = 0.2, legend_kwds={'label': 'Days to T0','orientation':'horizontal'}, legend = True)
plt.savefig(PATH + 'days_to_T0.png')

### FIGURE 4
chosen_flag = 'c6_stay_at_home_requirements'

for cls in GOV['class'].unique():
    if cls == 0 or cls == 4:
        continue

    plt.figure(figsize=(20,7))
    countries = EPI[EPI['class'] == cls]['countrycode'].unique()

    avg_raised = np.mean(GOV[GOV['countrycode'].isin(countries)][chosen_flag+'_days_since_50_raised'])
    std_raised = np.std(GOV[GOV['countrycode'].isin(countries)][chosen_flag+'_days_since_50_raised'])
    n1 = len(GOV[GOV['countrycode'].isin(countries)][chosen_flag+'_days_since_50_raised'].dropna())

    avg_lowered = np.mean(GOV[GOV['countrycode'].isin(countries)][chosen_flag+'_days_since_50_lowered'])
    std_lowered = np.std(GOV[GOV['countrycode'].isin(countries)][chosen_flag+'_days_since_50_lowered'])
    n2 = len(GOV[GOV['countrycode'].isin(countries)][chosen_flag+'_days_since_50_lowered'].dropna())

    avg_raised_again = np.mean(GOV[GOV['countrycode'].isin(countries)][chosen_flag+'_days_since_50_raised_again'])
    std_raised_again = np.std(GOV[GOV['countrycode'].isin(countries)][chosen_flag+'_days_since_50_raised_again'])
    n3 = len(GOV[GOV['countrycode'].isin(countries)][chosen_flag+'_days_since_50_raised_again'].dropna())


    countries = pd.Series({country:EPI[EPI['countrycode']==country]['confirmed'].iloc[-1]
                           for country in countries}).nlargest(n = 10).index.to_numpy()
    aggregate = []
    for country in countries:
        data = EPI[(EPI['countrycode'] == country) & (EPI['days_since_50_cases'] >= 0)]
        data['new_per_day_7d_ma'] = data['new_per_day_per_10k'].rolling(7).mean() / data['new_per_day_per_10k'].max()
        aggregate.append(data['new_per_day_7d_ma'].values)
        sns.lineplot(x='days_since_50_cases',y='new_per_day_7d_ma',data=data, label=country)
        continue

    aggregate = np.array([y[0:np.min([len(x) for x in aggregate])] for y in aggregate])
    aggregate = np.nanmean(aggregate, axis=0)
    sns.lineplot(x = np.arange(len(aggregate)),y = aggregate, color = 'black', linewidth = 5, label = 'aggregate')

    if cls == 1:
        plt.vlines([avg_raised - std_raised, avg_raised, avg_raised + std_raised], 0, 1,
                   color = 'black', linestyles=['dashed', 'solid', 'dashed'])
        plt.fill_betweenx([0, 1], [avg_raised - std_raised, avg_raised - std_raised],
                      [avg_raised + std_raised, avg_raised + std_raised],
                      facecolor='salmon', alpha=0.3)
        plt.text(avg_raised, 0.5,'Flag Raised', rotation = 90)
        plt.text(avg_raised, 1,'n = ' + str(n1))

    if cls == 2:
        plt.vlines([avg_raised - std_raised, avg_raised, avg_raised + std_raised], 0, 1,
                   color = 'black', linestyles=['dashed', 'solid', 'dashed'])
        plt.fill_betweenx([0, 1], [avg_raised - std_raised, avg_raised - std_raised],
                      [avg_raised + std_raised, avg_raised + std_raised],
                      facecolor='salmon', alpha=0.3)
        plt.text(avg_raised, 0.5, 'Flag Raised', rotation = 90)
        plt.text(avg_raised, 1, 'n = ' + str(n1))

        plt.vlines([avg_lowered - std_lowered, avg_lowered, avg_lowered + std_lowered], 0, 1,
                   color = 'black', linestyles=['dashed', 'solid', 'dashed'])
        plt.fill_betweenx([0, 1], [avg_lowered - std_lowered, avg_lowered - std_lowered],
                      [avg_lowered + std_lowered, avg_lowered + std_lowered],
                      facecolor='salmon', alpha=0.3)
        plt.text(avg_lowered, 0.5, 'Flag Lowered', rotation = 90)
        plt.text(avg_lowered, 1, 'n = ' + str(n2))

    if cls == 3:
        plt.vlines([avg_raised - std_raised, avg_raised, avg_raised + std_raised], 0, 1,
                   color = 'black', linestyles=['dashed', 'solid', 'dashed'])
        plt.fill_betweenx([0, 1], [avg_raised - std_raised, avg_raised - std_raised],
                      [avg_raised + std_raised, avg_raised + std_raised],
                      facecolor='salmon', alpha=0.3)
        plt.text(avg_raised, 0.5, 'Flag Raised', rotation = 90)
        plt.text(avg_raised, 1, 'n = ' + str(n1))

        plt.vlines([avg_lowered - std_lowered, avg_lowered, avg_lowered + std_lowered], 0, 1,
                   color = 'black', linestyles=['dashed', 'solid', 'dashed'])
        plt.fill_betweenx([0, 1], [avg_lowered - std_lowered, avg_lowered - std_lowered],
                      [avg_lowered + std_lowered, avg_lowered + std_lowered],
                      facecolor='salmon', alpha=0.3)
        plt.text(avg_lowered, 0.5, 'Flag Lowered', rotation = 90)
        plt.text(avg_lowered, 1, 'n = ' + str(n2))

        plt.vlines([avg_raised_again - std_raised_again, avg_raised_again, avg_raised_again + std_raised_again], 0, 1,
                   color = 'black', linestyles=['dashed', 'solid', 'dashed'])
        plt.fill_betweenx([0, 1], [avg_raised_again - std_raised_again, avg_raised_again - std_raised_again],
                      [avg_raised_again + std_raised_again, avg_raised_again + std_raised_again],
                      facecolor='salmon', alpha=0.3)
        plt.text(avg_raised_again, 0.5, 'Flag Raised Again', rotation = 90)
        plt.text(avg_raised_again, 1, 'n = ' + str(n3))

    plt.legend()
    plt.savefig(PATH + 'stage_' + str(cls) + '_timeline.png')

### APPENDIX

LABELLED_COLUMNS = pd.read_csv('./' + 'peak_labels.csv')
SPLINE_FITS = pd.read_csv('./' + 'SPLINE_FITS.csv')

countries_in_second = LABELLED_COLUMNS[LABELLED_COLUMNS['EPI_ENTERING_SECOND']]['COUNTRYCODE'].values

os.makedirs('./charts/table_figures/countries_in_second_wave/', exist_ok=True)
for country in countries_in_second:
    data = SPLINE_FITS[SPLINE_FITS['countrycode'] == country]
    if len(data) == 0 or len(find_peaks(-data['new_per_day_smooth'].values, prominence = 5, distance = 35)[0]) == 0:
        continue
    day_of_second_wave = find_peaks(-data['new_per_day_smooth'].values, prominence = 5, distance = 35)[0][-1]
    data.plot(x='date',y='new_per_day_smooth')
    plt.vlines(day_of_second_wave, 0, data['new_per_day_smooth'].max(), linestyles='dashed', colors='black')
    plt.title(country)
    plt.savefig(PATH + 'countries_in_second_wave/' + country + '.png')

ep = dict()
si = dict()
mob = dict()
exclude = list()

for country in countries_in_second:
    new_per_day_series = SPLINE_FITS[SPLINE_FITS['countrycode'] == country]['new_per_day_smooth'].values
    ep_date_series = SPLINE_FITS[SPLINE_FITS['countrycode'] == country]['date'].values
    ep_date_series = np.array([datetime.datetime.strptime(date,'%Y-%m-%d').date() for date in ep_date_series])

    si_series = government_response[government_response['countrycode']==country]['stringency_index'].values
    si_date_series = government_response[government_response['countrycode']==country]['date'].values

    mobility_series = mobility[mobility['countrycode']==country]['residential'].values
    mob_date_series = mobility[mobility['countrycode']==country]['date'].values

    if len(new_per_day_series) == 0:
        exclude.append(country)
        continue

    ##Getting bases of genuine peak to determine 'wave'
    peak_characteristics = find_peaks(new_per_day_series, prominence=PROMINENCE_THRESHOLD, distance=DISTANCE)
    genuine_peaks = LABELLED_COLUMNS[LABELLED_COLUMNS['COUNTRYCODE'] ==
                                     country].values[0][1:4].astype(int)[0:len(peak_characteristics[0])]

    ##Get dates of first and second wave
    first_wave = (peak_characteristics[1]['left_bases'][np.where(genuine_peaks!=0)],
                  peak_characteristics[1]['right_bases'][np.where(genuine_peaks!=0)])
    second_wave = peak_characteristics[0][np.where(genuine_peaks!=0)][0] + \
                  np.argmin(new_per_day_series[peak_characteristics[0][np.where(genuine_peaks!=0)][0]::])
    first_wave = (ep_date_series[first_wave[0]][0],
                  ep_date_series[first_wave[1]][0])
    second_wave = ep_date_series[second_wave]

    ##Slice first wave, second wave data
    ep_first_wave = new_per_day_series[np.where((ep_date_series >= first_wave[0]) & (ep_date_series <= first_wave[1]))]
    ep_second_wave = new_per_day_series[np.where(ep_date_series >= second_wave)]
    si_first_wave = si_series[np.where((si_date_series >= first_wave[0]) & (si_date_series <= first_wave[1]))]
    si_second_wave = si_series[np.where(si_date_series >= second_wave)]
    mob_first_wave = mobility_series[np.where((mob_date_series >= first_wave[0]) & (mob_date_series <= first_wave[1]))]
    mob_second_wave = mobility_series[np.where(mob_date_series >= second_wave)]

    ep[country + '_first_wave'] = pd.Series(ep_first_wave)
    ep[country + '_second_wave'] = pd.Series(ep_second_wave)
    mob[country + '_first_wave'] = pd.Series(mob_first_wave)
    mob[country + '_second_wave'] = pd.Series(mob_second_wave)
    si[country + '_first_wave'] = pd.Series(si_first_wave)
    si[country + '_second_wave'] = pd.Series(si_second_wave)

disp_limit = 75
plot_countries = ['AUS','BEL','ESP','JPN','USA']
countries = [country for country in countries_in_second if not(country in exclude)]

ep = pd.DataFrame.from_dict(ep)
mob = pd.DataFrame.from_dict(mob)
si = pd.DataFrame.from_dict(si)

norm_ep = ep/ep.max()
ep_aggr_first_wave = norm_ep[[
    country + '_first_wave' for country in countries]].mean(axis= 1, skipna= True).iloc[0:disp_limit]
ep_aggr_second_wave = norm_ep[[
    country + '_second_wave' for country in countries]].mean(axis= 1, skipna= True).iloc[0:disp_limit]

mob_aggr_first_wave = mob[[
    country + '_first_wave' for country in countries]].mean(axis= 1, skipna= True).iloc[0:disp_limit]
mob_aggr_second_wave = mob[[
    country + '_second_wave' for country in countries]].mean(axis= 1, skipna= True).iloc[0:disp_limit]

si_aggr_first_wave = si[[
    country + '_first_wave' for country in countries]].mean(axis= 1, skipna= True).iloc[0:disp_limit]
si_aggr_second_wave = si[[
    country + '_second_wave' for country in countries]].mean(axis= 1, skipna= True).iloc[0:disp_limit]

color_dict = {
    'AUS':'orange',
    'BEL':'red',
    'ESP':'purple',
    'JPN':'blue',
    'USA':'salmon'
}

plt.figure(figsize=(20,7))
for country in plot_countries:
    plt.plot(norm_ep[country + '_first_wave'].iloc[0:disp_limit],
             linestyle = 'solid',color = color_dict[country],linewidth = 0.75, label = country)
    plt.plot(norm_ep[country + '_second_wave'].iloc[0:disp_limit],
             linestyle = 'dashed',color = color_dict[country],linewidth = 0.75)
plt.plot(ep_aggr_first_wave, linestyle = 'solid', color = 'black', linewidth = 2)
plt.plot(ep_aggr_second_wave, linestyle = 'dashed', color = 'black', linewidth = 2)
plt.title('Comparison of new cases per day the first ' + str(disp_limit)
          + ' days for the first and second wave (second wave dashed)')
plt.legend()
plt.savefig(PATH + 'new_cases_per_day_first_vs_second.png')

plt.figure(figsize=(20,7))
for country in plot_countries:
    plt.plot(mob[country + '_first_wave'].iloc[0:disp_limit],
             linestyle = 'solid',color = color_dict[country],linewidth = 0.75, label = country)
    plt.plot(mob[country + '_second_wave'].iloc[0:disp_limit],
             linestyle = 'dashed',color = color_dict[country],linewidth = 0.75)
plt.plot(mob_aggr_first_wave, linestyle = 'solid', color = 'black', linewidth = 2)
plt.plot(mob_aggr_second_wave, linestyle = 'dashed', color = 'black', linewidth = 2)
plt.title('Comparison of mobility for the first ' + str(disp_limit)
          + ' days for the first and second wave (second wave dashed)')
plt.legend()
plt.savefig(PATH + 'mobility_first_vs_second.png')

plt.figure(figsize=(20,7))
for country in plot_countries:
    plt.plot(si[country + '_first_wave'].iloc[0:disp_limit],
             linestyle = 'solid',color = color_dict[country],linewidth = 0.75, label = country)
    plt.plot(si[country + '_second_wave'].iloc[0:disp_limit],
             linestyle = 'dashed',color = color_dict[country],linewidth = 0.75)
plt.plot(si_aggr_first_wave, linestyle = 'solid', color = 'black', linewidth = 2)
plt.plot(si_aggr_second_wave, linestyle = 'dashed', color = 'black', linewidth = 2)
plt.title('Comparison of SI for the first ' + str(disp_limit)
          + ' days for the first and second wave (second wave dashed)')
plt.legend()
plt.savefig(PATH + 'si_first_vs_second.png')


'''
FIGURE 1a - gpd.GeoDataFrame(figure_1a,geometry='geometry').plot(column='days_to_t0')
'''