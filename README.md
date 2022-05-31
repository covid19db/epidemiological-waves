# Description

This repository contains all the code necessary to replicate the work in our paper "Epidemiological waves - types, drivers and modulators in the COVID-19 pandemic".

It provides a Python package `wavefinder` in the directory `src/wavefinder` to identify "waves" in time series.

This is used by the Python modules in the `src` directory to classify epidemic waves in time series of
cases of and deaths due to COVID-19 and analyse the results.

# How to run

The Python code in the `src` directory can be run with the following.

```
docker-compose build
docker-compose run epidemetrics
```
or
```
pip3 install -r ./requirements.txt
cd ./src
python3 ./main.py
```

Once the main sub-routine has finished running successfully, the `data` directory will have been populated with the results of the analysis.
The `plots` directory will contain the plots showing the observed waves.
The `manuscriptfigures` directory will contain the necessary data as well as some of the figures, which are generated with Python.

The remaining figures can be generated using R by running the script `manuscriptfigures/R.sh` will generate these.
Alternatively, they can be generated using the functions in `manuscriptfigures/manuscriptfigures.py`.

# How to run tests
```
docker-compose -f ./docker-compose.test.yml up --build
```
or
```
python -m pytest tests 
```

# Wavefinder

The `wavefinder` package, found in `src\wavefinder` provides the `WaveList` class and two associated plotting functions. 

## WaveList

The `WaveList` class implements an algorithm to
identify the waves in time series. Calling
`wavelist = WaveList(raw_data, series_name, t_sep_a, prominence_threshold, prominence_height_threshold)`
will initialise a WaveList object from the time series `raw_data`.
The time series is processed by the Sub-Algorithms contained in `src\wavefinder\subalgorithms` and described in our paper.

The identified waves can then be accessed through `wavelist.waves`, with interim steps in the algorithm also accessible.
These waves will all have duration at least `t_sep_a`, prominence at least `prominence_threshold`, and at each peak the prominence will be at least `prominence_height_threshold` multiplied by the value at the peak.

Calling  `wavelist.cross_validate(reference_wavelist)` 
implements an algorithm to impute the presence of additional waves
in `wavelist` from those in `reference_wavelist`.
It will inspect `wavelist.waves` to determine whether a peak exists 
for every wave in `reference_wavelist.waves`. If a peak is not found, 
it will attempt to recover one from an interim step in the algorithm, 
`wavelist.peaks_sub_b`. 
It returns a DataFrame containing the revised list of peaks and troughs 
for `wavelist` and updates `wavelist.waves`. 

The DataFrame `wavelist.waves` contains a row for each peak or trough.
The columns are `location` and `y_position`, which give the index and value of
the peak or trough within `wavelist.raw_data`, `prominence`, giving
the prominence of the peak or trough (as calculated with respect to 
other peaks and troughs, not with resepct to all of `wavelist.raw_data`)
and `peak_ind`, which is 0 for a trough and 1 for a peak.

## Plotting functions

The package also provides two plotting functions, `plot_peaks` and `plot_cross_validator`.

Calling `plot_peaks(wavelists, title, save, plot_path)` with a list of `WaveList` objects
will plot the peaks and troughs identified in each `WaveList` under the title `title` and optionally
save it to `plot_path` with the filename `title.png`.

Calling `plot_cross_validator(input_wavelist, reference_wavelist, results, filename, plot_path)`
with `results = wavelist.cross_validate(reference_wavelist, plot=True, plot_path, title)`
will produce a plot showing how `WaveCrossValidator` added peaks to the `input_wavelist` in order to better align it with the `reference_wavelist`.
