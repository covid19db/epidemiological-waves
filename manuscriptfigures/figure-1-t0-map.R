package_list <- c("magrittr","dplyr", "ggplot2","scales","purrr", "GADMTools", "geojsonio")
for (package in package_list){
  if (!package %in% installed.packages()){
    install.packages(package)
  }
}
lapply(package_list, require, character.only = TRUE)


# clear workspace
rm(list=ls())

# disable scientific notation
options(scipen=999)



## read in the t_0 data to colour the regions with.
t0_df <- read.table(
  file = "data/2021-09-15/figure_1b.csv",
  header = TRUE,
  sep = ";",
  stringsAsFactors = FALSE
) %>%
  select(countrycode, days_to_t0_10_dead) %>%
  rename(GID_0 = countrycode, days_to_t0 = days_to_t0_10_dead)

## read in the geometry of each region and link it to the data via the GID_0.
world_sf <- topojson_read("data/2020-09-13/gadm36_0.json")
plot_sf <- left_join(world_sf, t0_df, by = "GID_0")

t0_breaks <- round(boxplot.stats(plot_sf$days_to_t0)$stats / 10) * 10

## make the actual plot and do some preliminary styling.
g <- ggplot() +
  geom_sf(
    data = plot_sf,
    mapping = aes(fill = days_to_t0),
    colour = "white",
    size = 0.1
  ) +
  scale_fill_fermenter(
    breaks = t0_breaks,
    type = "seq",
    direction = -1,
    palette = "RdPu"
  ) +
  labs(fill = "Days until epidemic\nthreshold reached") +
  theme_void() +
  theme(
    legend.box.background = element_rect(fill = "white", colour = "grey"),
    legend.box.margin = margin(t = 0.1, r = 0.2, b = 0.1, l = 0.2, unit = "cm"),
    legend.position = c(0.25, 0.15),
    legend.direction = "horizontal",
    legend.title = element_text(vjust = 1.0, size = 7),
    legend.key.height = unit(0.5, "line"),
    legend.text = element_text(size = 5)
  )

if (interactive()) {
  print(g)
} else {
  ## Save this to 70% height of a landscape A5 page.
  ggsave(
    filename = "./output/png/figure-1-t0-map.png",
    plot = g,
    height = 14.8 * 0.7,
    width = 21.0,
    units = "cm"
  )
}
