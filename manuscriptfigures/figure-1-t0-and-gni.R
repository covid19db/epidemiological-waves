package_list <- c("magrittr","dplyr", "ggplot2","scales","purrr")
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


x <- read.csv("data/2020-09-15/gni_data.csv",
  stringsAsFactors = FALSE
) %>%
  select(countrycode, gni_per_capita)

## read in the t_0 data to colour the regions with.
t0_df <- read.table(
  file = "data/2021-09-15/figure_1b.csv",
  header = TRUE,
  sep = ";",
  stringsAsFactors = FALSE
) %>%
  select(countrycode, days_to_t0_10_dead) %>%
  rename(days_to_t0 = days_to_t0_10_dead)

z <- full_join(x = x, y = t0_df, by = "countrycode") %>%
  filter(
    not(is.na(gni_per_capita)),
    not(is.na(days_to_t0))
  )

g <- ggplot(
  data = z,
  mapping = aes(
    x = gni_per_capita,
    y = days_to_t0
  )
) +
  geom_point(shape = 1) +
  geom_smooth(method = "lm", colour = "#7a0177", fill = "#c51b8a") +
  scale_x_log10(
    labels = scales::comma_format(big.mark = ",")
  ) +
  scale_y_log10() +
  labs(
    x = "GNI per capita",
    y = "Days until epidemic established"
  ) +
  theme_bw() +
  theme(axis.title = element_text(face = "bold"))

if (interactive()) {
  print(g)
} else {
  ## Save this to 50% height and 30% width of a landscape A5 page.
  ggsave(
    filename = "./output/png/figure-1-t0-and-gni.png",
    plot = g,
    height = 0.5 * 14.8,
    width = 0.3 * 21.0,
    units = "cm"
  )

  sink("./output/txt/figure-1-t0-and-gni.txt")
  summary(lm(log(days_to_t0) ~ log(gni_per_capita), z))
  cat("And again removing the outlier\n")
  summary(lm(log(days_to_t0) ~ log(gni_per_capita), filter(z, days_to_t0 > 30)))
  sink()
}
