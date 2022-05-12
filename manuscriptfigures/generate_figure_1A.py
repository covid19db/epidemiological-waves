import os
import pandas as pd
from plotnine import *

data_path = os.path.join(os.getcwd(), "manuscriptfigures/data")
output_path = os.path.join(os.getcwd(), "manuscriptfigures/output")

# read in the t_0 data
file = os.path.join(data_path, "2021-09-15/figure_1b.csv")
t0_df = pd.read_csv(file,
                    delimiter=";",
                    header=0,
                    usecols=["countrycode", "days_to_t0_10_dead"],
                    na_values=["N/A", "NA", "#N/A", " ", "", "None"]).rename(
    columns={'days_to_t0_10_dead': 'days_to_t0', 'countrycode': 'GID_0'})

## read in the geometry of each region and link it to the data via the GID_0.
world_sf <- topojson_read("data/2020-09-13/gadm36_0.json")
plot_sf <- left_join(world_sf, t0_df, by = "GID_0")

# read in the GNI data
file = os.path.join(data_path, "2020-09-15/gni_data.csv")
x = pd.read_csv(file,
                       header=0,
                       usecols=['countrycode', 'gni_per_capita'])



# merge dataframes to plot
z = pd.merge(x, t0_df, how="outer", on="countrycode").dropna()

def number_formatter(array):
    return ["{:,}".format(int(n)) for n in array]

g = (ggplot(z, aes(x='gni_per_capita', y='days_to_t0'))
     + geom_point(shape = 1)
     + geom_smooth(method = "lm", colour = "#7a0177", fill = "#c51b8a")
     + scale_x_log10(labels=number_formatter)
     + scale_y_log10()
     + labs(x="GNI per capita", y="Days until epidemic established")
     + theme_bw()
     + theme(axis_title = element_text(face = "bold"))
     )

file = os.path.join(output_path, "1B.png")
g.save(filename=file, height=7.4, width=14.7, units="cm", dpi=500)
