#!/usr/bin/env bash

Rscript R/figure-1-cases-and-deaths.R
Rscript R/figure-1-t0-and-gni.R
Rscript R/figure-1-t0-map.R
Rscript R/generate_figure_5.R
Rscript R/generate_figure_6.R

Rscript ../src/correlations.R