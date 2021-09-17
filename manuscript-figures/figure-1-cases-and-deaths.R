library(magrittr)
library(dplyr)
library(reshape2)
library(ggplot2)
library(scales)
library(zoo)

## we make a collection of countries that we care about so we can subset by them
## later.
y_chr <- read.table(
  file = "data/2021-09-15/figure_1b.csv",
  header = TRUE,
  sep = ";",
  stringsAsFactors = FALSE
) %>%
  filter(class == 4) %>%
  use_series(countrycode)

## read in the time series of cases and deaths so we have the actual data to
## plot.
x <- read.csv("data/2021-09-15/figure_1a.csv") %>%
  select(countrycode, date, new_per_day, dead_per_day) %>%
  filter(is.element(el = countrycode, set = y_chr)) %>%
  mutate(date = as.Date(date)) %>%
  melt(id.vars = c("countrycode", "date"))

## read in the t_0 data to align the time series correctly.
t0_df <- read.table(
  file = "data/2021-09-15/figure_1b.csv",
  header = TRUE,
  sep = ";",
  stringsAsFactors = FALSE
) %>%
  select(countrycode, days_to_t0_10_dead) %>%
  rename(days_to_t0 = days_to_t0_10_dead)

## convert the date to an integer and then subtract the number of days until the
## threshold value is reached. There is an extra offset there so that the
## resulting numbers start at zero.
min_num_date <- x$date %>% as.numeric() %>% min()
aligned_df <- left_join(x = x, y = t0_df, by = "countrycode") %>%
  mutate(num_aligned_date = as.numeric(date) - days_to_t0 - min_num_date) %>%
  filter(num_aligned_date >= 0)


facet_labels <- c(
  dead_per_day = "Deaths",
  new_per_day = "Confirmed cases"
)

smooth_df_npd <- aligned_df %>%
  filter(variable == "new_per_day", countrycode != "USA", countrycode != "IND") %>%
  group_by(date) %>%
  summarise(mean_val = mean(value)) %>%
  mutate(value = rollmedian(mean_val, 7, na.pad = TRUE), variable = "new_per_day")

smooth_df_dpd <- aligned_df %>%
  filter(variable == "dead_per_day", countrycode != "USA", countrycode != "IND") %>%
  group_by(date) %>%
  summarise(mean_val = mean(value)) %>%
  mutate(value = rollmedian(mean_val, 7, na.pad = TRUE), variable = "dead_per_day")

smooth_df <- rbind(smooth_df_npd, smooth_df_dpd)

g <- ggplot(
  data = filter(aligned_df, countrycode != "USA", countrycode != "IND"),
  mapping = aes(x = date, y = value, group = countrycode)
) +
  geom_point(shape = 1) +
  geom_line(data = smooth_df, group = NA, colour = "#7a0177", size = 2) +
  facet_wrap(~variable, scales = "free_y", labeller = labeller(variable = facet_labels)) +
  scale_y_sqrt() +
  scale_x_date(labels = label_date_short()) +
  labs(y = NULL, x = NULL) +
  theme_bw() +
  theme(
    strip.background = element_blank(),
    strip.text = element_text(face = "bold")
  )

if (interactive()) {
  plot(g)
} else {
  ## Save this to 50% height and 70% width of a landscape A5 page.
  ggsave(
    filename = "./output/png/figure-1-cases-and-deaths.png",
    plot = g,
    height = 0.5 * 14.8,
    width = 0.7 * 21.0,
    units = "cm"
  )
}
