library(magrittr)
library(dplyr)
library(reshape2)
library(ggplot2)
library(scales)
library(zoo)

## we make a collection of countries that we care about so we can subset by them
## later. We also need to extract the population of each of the countries so
## that the time series values can be expressed as per capita.
extra_df <- read.table(
  file = "data/2021-09-15/figure_1b.csv",
  header = TRUE,
  sep = ";",
  stringsAsFactors = FALSE
) %>%
  filter(class == 4)
y_chr <- extra_df$countrycode
y_pop <- extra_df %>% select(countrycode, population)


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
y_pop[y_pop$countrycode == "GLP", "population"] <- 400013
y_pop[y_pop$countrycode == "MTQ", "population"] <- 374743
y_pop[y_pop$countrycode == "MYT", "population"] <- 279507

if (any(is.na(y_pop$population))) {
  cat("There appear to be some NA values in the population data frame...\n")
  for (ix in seq.int(nrow(y_pop))) {
    if (is.na(y_pop[ix,"population"])) {
      cat("\tmissing population for ", y_pop[ix,"countrycode"], "\n")
    }
  }
  stop("Stopping because there is missing population data")
}


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
  filter(num_aligned_date >= 0) %>%
  left_join(y = y_pop, by = "countrycode") %>%
  mutate(value_per_10k = value / population * 1e4)


facet_labels <- c(
  dead_per_day = "Deaths",
  new_per_day = "Confirmed cases"
)

smooth_df_npd <- aligned_df %>%
  filter(variable == "new_per_day", countrycode != "USA", countrycode != "IND") %>%
  group_by(num_aligned_date) %>%
  summarise(mean_val = mean(value_per_10k)) %>%
  mutate(value_per_10k = rollmedian(mean_val, 7, na.pad = TRUE), variable = "new_per_day")

smooth_df_dpd <- aligned_df %>%
  filter(variable == "dead_per_day", countrycode != "USA", countrycode != "IND") %>%
  group_by(num_aligned_date) %>%
  summarise(mean_val = mean(value_per_10k)) %>%
  mutate(value_per_10k = rollmedian(mean_val, 7, na.pad = TRUE), variable = "dead_per_day")

smooth_df <- rbind(smooth_df_npd, smooth_df_dpd)

g <- ggplot(
  data = filter(aligned_df, countrycode != "USA", countrycode != "IND"),
  mapping = aes(x = num_aligned_date, y = value_per_10k, group = countrycode)
) +
  geom_point(shape = 1) +
  geom_line(data = smooth_df, group = NA, colour = "#7a0177", size = 2) +
  facet_wrap(~variable, scales = "free_y", labeller = labeller(variable = facet_labels)) +
  scale_y_sqrt() +
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
