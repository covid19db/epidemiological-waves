# Load Packages, Clear, Sink -------------------------------------------------------

# load packages
package_list <- c("readr","ggplot2","gridExtra","plyr","dplyr","ggsci","RColorBrewer",
                  "viridis","sf","reshape2","ggpubr","egg","scales","plotrix","ggallin", "stats")
for (package in package_list){
  if (!package %in% installed.packages()){
    install.packages(package)
  }
}
lapply(package_list, require, character.only = TRUE)

# clear workspace
rm(list=ls())

# Import Data ------------------------------------------------------------

# Import csv file for Figure 3 
figure_3_all_data <- read_csv("../manuscriptfigures/data/correlations.csv",
                                na = c("N/A","NA","#N/A"," ",""))
figure_3_all_data$countrycode = as.factor(figure_3_all_data$countrycode)
figure_3_all_data$country = as.factor(figure_3_all_data$country)
figure_3_all_data$class = as.factor(figure_3_all_data$class)

# Process Data ----------------------------------------------

# Normalise by population size
figure_3_all_data$dead_during_wave_per_10k <- figure_3_all_data$dead_during_wave * 10000 / figure_3_all_data$population
figure_3_all_data$tests_during_wave_per_10k <- figure_3_all_data$tests_during_wave * 10000 / figure_3_all_data$population

# Highlight some countries
highlight_countries = c('Australia','Belgium','United States')
figure_3_all_data$country_highlight = ''
for (c in highlight_countries){
  figure_3_all_data[figure_3_all_data$country==c,'country_highlight'] <- c
}
figure_3_all_data$country_highlight <- factor(figure_3_all_data$country_highlight, levels=c('Australia','Belgium','United States',''))

# Remove very small countries as their T0 are skewed
figure_3_data <- subset(figure_3_all_data,population>=2500000)

# Compute correlations for all flag response times  ------------------------------------------------------------

flags = c('si_integral','last_tests_per_10k',
          'si_response_time','c1_response_time','c2_response_time','c3_response_time','c4_response_time','c5_response_time',
          'c6_response_time','c7_response_time','c8_response_time',
          'h2_response_time','h3_response_time',
          'testing_response_time_1','testing_response_time_10','testing_response_time_100','testing_response_time_1000')
df <- data.frame(flag=character(),
                wave=factor(),
                estimate=character(),
                p_value=double(),
                stringsAsFactors=FALSE)
figure_3_data <- subset(figure_3_all_data,population>=2500000)

for (f in flags){
  for (w in c('1','2')){
    x <- f
    y <- 'dead_during_wave_per_10k'
    data = figure_3_data[figure_3_data[['wave']] == w, ]
    corr <- cor.test(data[[x]], data[[y]], method = "kendall")
    p_value_str <- if (corr$p.value<0.0001) {"<0.0001"} else {toString(signif(corr$p.value,2))}
    estimate_str <- toString(signif(corr$estimate,2))
    df <- rbind(df, data.frame(flag=f, wave=w, estimate=estimate_str, p_value=p_value_str))
  }
  y <- 'last_dead_per_10k'
  data = figure_3_data[figure_3_data[['wave']] == 1, ]
  corr <- cor.test(data[[x]], data[[y]], method = "kendall")
  p_value_str <- if (corr$p.value<0.0001) {"<0.0001"} else {toString(signif(corr$p.value,2))}
  estimate_str <- toString(signif(corr$estimate,2))
  df <- rbind(df, data.frame(flag=f, wave='all', estimate=estimate_str, p_value=p_value_str))
  
  y <- 'class'
  data$class = as.numeric(data$class)
  corr <- cor.test(data[[x]], data[[y]], method = "kendall")
  p_value_str <- if (corr$p.value<0.0001) {"<0.0001"} else {toString(signif(corr$p.value,2))}
  estimate_str <- toString(signif(corr$estimate,2))
  df <- rbind(df, data.frame(flag=f, wave='class', estimate=estimate_str, p_value=p_value_str))
  
}

write.csv(df,"../output/analysis/corr.csv")
