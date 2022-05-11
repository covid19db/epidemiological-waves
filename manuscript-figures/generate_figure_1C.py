import os
import pandas as pd
import numpy as np
from plotnine import *

data_path = os.path.join(os.getcwd(), "manuscript-figures/data/2021-09-15")
output_path = os.path.join(os.getcwd(), "manuscript-figures/output")

# Import Data for figure 1 -------------------------------------------------------------------

file = os.path.join(data_path, "figure_1b.csv")
extra_df = pd.read_csv(file,
                       delimiter=";",
                       header=0,
                       na_values=["N/A", "NA", "#N/A", " ", "", "None"])

# in alex's R code the filter was class == 4
extra_df = extra_df[extra_df["class"] >= 4]

y_chr = extra_df[["countrycode"]]
y_pop = extra_df[["countrycode", "population"]]

## check that we are not missing the population for any of the countries we are
## considering...
##
## We can get population estimates from OWID, but very similar values are
## available from WorldOMeter (https://www.worldometers.info/).
##
## - GLP is Guadeloupe  400,013
## - MTQ is Martinique  374,743
## - MYT is Mayotte     279,507
##

y_pop['population'] = np.where(y_pop['countrycode'] == 'GLP', 400013, y_pop['population'])
y_pop['population'] = np.where(y_pop['countrycode'] == 'MTQ', 374743, y_pop['population'])
y_pop['population'] = np.where(y_pop['countrycode'] == 'MYT', 279507, y_pop['population'])

## read in the time series of cases and deaths so we have the actual data to
## plot.
file = os.path.join(data_path, "figure_1a.csv")
x = pd.read_csv(file,
                header=0,
                usecols=["countrycode", "date", "new_per_day", "dead_per_day"],
                parse_dates=['date'],
                na_values=["N/A", "NA", "#N/A", " ", ""])
x = x[x["countrycode"].isin(y_chr['countrycode'])]
x = pd.melt(x, id_vars=["countrycode", "date"])

## read in the t_0 data to align the time series correctly.
file = os.path.join(data_path, "figure_1b.csv")
t0_df = pd.read_csv(file,
                    delimiter=";",
                    header=0,
                    usecols=["countrycode", "days_to_t0_10_dead"],
                    na_values=["N/A", "NA", "#N/A", " ", "", "None"]).rename(
    columns={'days_to_t0_10_dead': 'days_to_t0'})

## convert the date to an integer and then subtract the number of days until the
## threshold value is reached. There is an extra offset there so that the
## resulting numbers start at zero. This is also where we calculate the values
## per 10k people.
min_num_date = x["date"].apply(lambda x: x.toordinal()).min()

aligned_df = pd.merge(left=x, right=t0_df, how="left", on="countrycode")
aligned_df["num_aligned_date"] = aligned_df["date"].apply(lambda x: x.toordinal()) - aligned_df[
    "days_to_t0"] - min_num_date
aligned_df = aligned_df[aligned_df["num_aligned_date"] >= 0]
aligned_df = aligned_df.merge(y_pop, how="left", on="countrycode")
aligned_df["value_per_10k"] = aligned_df["value"] * 10000 / aligned_df["population"]
aligned_df = aligned_df[["variable", "countrycode", "value_per_10k", "num_aligned_date"]]

facet_labels = {"dead_per_day": "Deaths", "new_per_day": "Confirmed Cases"}

aligned_df = aligned_df.drop(aligned_df[(aligned_df.countrycode == "USA") | (aligned_df.countrycode == "IND")].index)

smooth_df_npd = aligned_df[aligned_df.variable == "new_per_day"]
smooth_df_npd = smooth_df_npd.groupby(["variable", "num_aligned_date"]).mean().reset_index()
smooth_df_npd["value_per_10k"] = smooth_df_npd["value_per_10k"].rolling(7).median()

smooth_df_dpd = aligned_df[aligned_df.variable == "dead_per_day"]
smooth_df_dpd = smooth_df_dpd.groupby(["variable", "num_aligned_date"]).mean().reset_index()
smooth_df_dpd["value_per_10k"] = smooth_df_dpd["value_per_10k"].rolling(7).median()

smooth_df = pd.concat([smooth_df_npd, smooth_df_dpd])

g = (ggplot(aligned_df, aes(x="num_aligned_date", y="value_per_10k", group="countrycode"))
     + geom_point(shape=1, alpha=0.1)
     + geom_line(data=smooth_df, group="NA", colour="#7a0177", size=2)
     + facet_wrap(facets="~ variable", scales="free_y", labeller=facet_labels)
     + scale_y_sqrt(n_breaks=4)
     + labs(y="Per 10,000 people (square root scale)",
            x="Days since epidemic threshold reached")
     + theme_bw()
     + theme(strip_background=element_blank(), strip_text=element_text(face="bold")))

file = os.path.join(output_path, "1C.png")
g.save(filename=file, height=7.4, width=14.7, units="cm", dpi=500)

"""

figure_1a_data = pd.read_csv("./data/figure_1a.csv",
                             header=0,
                             parse_dates=['date'],
                             na_values=["N/A", "NA", "#N/A", " ", ""])
figure_1a_data["countrycode"] = figure_1a_data["countrycode"].astype(CategoricalDtype(ordered=False))

figure_1b_data = pd.read_csv("./data/figure_1b.csv",
                             delimiter=";",
                             header=0,
                             na_values=["N/A", "NA", "#N/A", " ", "", "None"])
columns = ["countrycode", "class", "class_coarse"]
for col in columns:
    figure_1b_data[col] = figure_1b_data[col].astype(CategoricalDtype(ordered=False))

# Process Data for figure 1 -------------------------------------------------------------------

# Remove rows with NA in geometry. Required to convert column to shape object
# figure_1b_data <- subset(figure_1b_data,!is.na(geometry))
# Convert "geometry" column to a sfc shape column 
# figure_1b_data$geometry <- st_as_sfc(figure_1b_data$geometry)
# Convert dataframe to a sf shape object with "geometry" containing the shape information
# figure_1b_data <- st_sf(figure_1b_data)

# Figure 1a   ----------------------------------------------
# Set up colour palette
my_palette_1 = brewer_pal(type='seq', palette="YlGnBu", direction=1)(2)[1]
my_palette_2 = brewer_pal(palette="YlGnBu")(4)[3]
# my_palette_3 <- "GnBu"
my_palette_4 = "#cb181d"

figure_1a1 = (ggplot(figure_1a_data, aes(x="date", y="new_cases_per_rel_constant"))
              + geom_line(size=0.1, alpha=0.4, na_rm=True, color=my_palette_2, show_legend=False)
              # mapping=aes(color="countrycode") removed from geom_line, see what happens
              + geom_smooth(method="loess", se=False, span=0.2, na_rm=True, color=my_palette_4)
              + labs(title="Confirmed Cases per Day", x=element_blank(), y=element_blank())
              + theme_classic(base_size=8, base_family='serif')
              + scale_y_continuous(trans='sqrt', breaks=[10, 20, 30, 40])
              + scale_x_date(date_breaks='3 months', date_labels='%b %Y')
              + theme(plot_title=element_text(size=8, hjust=0.5),
                      panel_grid_major_y=element_line(size=.1, color="grey")))
# figure_1a1

figure_1a1.save(filename='./plots/figure_1a1.png', width=10, height=7, units='cm', dpi=300)


figure_1a2 <- (ggplot(figure_1a_data, aes(x=date, y=dead_per_day))
               + geom_line(aes(color=countrycode)
                           , size=0.1, alpha=0.4, na.rm=TRUE, color=my_palette_2, show.legend=FALSE)
               + geom_smooth(method = "loess", se = FALSE, span=0.2, na.rm=TRUE, color=my_palette_4)
               + labs(title="Deaths per Day", x=element_blank(), y=element_blank())
               + theme_classic(base_size=8,base_family='serif')
               + scale_y_continuous(trans='log', breaks=c(1,2,5,10,20,50,100,200,500,1000,2000,5000))
               + scale_x_date(date_breaks='3 months', date_labels='%b %Y')
               + theme(plot.title=element_text(size=8, hjust = 0.5), panel.grid.major.y = element_line(size=.1, 
               color="grey")))
#figure_1a2
ggsave('./plots/figure_1a2.png', plot=figure_1a2, width=10, height=7, units='cm', dpi=300)



# Figure 1b1: Chloropleth   ----------------------------------------------
figure_1b1 <- (ggplot(data = figure_1b_data) 
               + geom_sf(aes(fill=days_to_t0_10_dead), lwd=0, color=NA, na.rm=TRUE)
               + labs(title=element_blank(), fill="Days until Epidemic Established")
               + scale_fill_distiller(palette=my_palette_3, trans="sqrt", breaks=c(1,100,200,300,400,500))
               #+ scale_x_continuous(expand=c(0,0), limits=c(-125, -65))
               #+ scale_y_continuous(expand=c(0,0), limits=c(24, 50))
               + theme_void()
               + guides(fill = guide_colourbar(barwidth = 20, barheight = 0.5))
               + theme(legend.text=element_text(size=8,family='serif'),legend.title=element_text(vjust=1,size=8,
               family='serif')
                       , panel.grid.major=element_line(colour = "transparent"),legend.position="bottom"))
#figure_1b1
ggsave('./plots/figure_1b1.png', plot=figure_1b1, width=20, height=12, units='cm', dpi=300)


# Figure 1b2: Boxplot   ----------------------------------------------
figure_1b2 <- (ggplot(data = figure_1b_data) 
               + geom_boxplot(aes(x=class_coarse, group=class_coarse, y=days_to_t0_10_dead)
                              , na.rm=TRUE, outlier.colour=my_palette_2, outlier.shape=1, fill=my_palette_2)
               + labs(title=element_blank(), x=element_blank(), y="Days until Epidemic Established")
               + scale_x_discrete(limits=rev(levels(figure_1b_data$class_coarse)),labels=c('Third Wave or Above',
               'Second Wave','First Wave'))
               + scale_y_continuous(expand=c(0,0), limits=c(0, NA))
               + coord_flip()
               + theme_classic(base_size=8,base_family='serif')
               + theme(panel.grid.major.x = element_line(size=.1, color="grey")))
#figure_1b2
ggsave('./plots/figure_1b2.png', plot=figure_1b2, width=10, height=7, units='cm', dpi=300)


# Figure 1b3: Scatterplot of GNI   ----------------------------------------------
figure_1b3 <- (ggplot(data = figure_1b_data, aes(x=gni_per_capita, y=days_to_t0_10_dead)) 
               + geom_point(na.rm=TRUE, color=my_palette_2, shape=1)
               + geom_smooth(method='lm', color=my_palette_4, se=FALSE)
               + labs(title=element_blank(), x='GNI per Capita', y="Days until Epidemic Established")
               + scale_x_continuous(trans='log',breaks=c(1000,2000,5000,10000,20000,50000))
               + scale_y_continuous(trans='log', breaks=c(100,200,300,400,500))
               + theme_classic(base_size=8,base_family='serif')
               + theme(panel.grid.major = element_line(size=.1, color="grey")))
#figure_1b3
ggsave('./plots/figure_1b3.png', plot=figure_1b3, width=10, height=7, units='cm', dpi=300)
"""
