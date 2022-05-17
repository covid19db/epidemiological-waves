
# Plot figure 6: USA Choropleth ---------------------------------------------------------------

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


# Import Data for figure 6 -------------------------------------------------------------------

figure_6a_data <- read_csv(file="./data/figure_6a.csv",
                           na = c("N/A","NA","#N/A"," ",""))
figure_6a_data$countrycode <- as.factor(figure_6a_data$countrycode)
figure_6a_data$adm_area_1 <- as.factor(figure_6a_data$adm_area_1)

figure_6b_data <- read_delim(file="./data/figure_6.csv",
                             delim=";",
                             na = c("N/A","NA","#N/A"," ","","None"))
figure_6b_data$gid <- as.factor(figure_6b_data$gid)
figure_6b_data$fips <- as.factor(figure_6b_data$fips)

# Process Data for figure 6 -------------------------------------------------------------------
# figure 6a processing
# Get top n states by total confirmed cases, group others into Others
figure_6a_max <- aggregate(figure_6a_data[c("confirmed")],
                           by = list(figure_6a_data$adm_area_1),
                           FUN = max,
                           na.rm=TRUE)
figure_6a_max <- plyr::rename(figure_6a_max, c("Group.1"="adm_area_1"))
figure_6a_max <- figure_6a_max[order(-figure_6a_max$confirmed),]
n=15
top_n <- head(figure_6a_max$adm_area_1,n)
figure_6a_longitudes <- unique(figure_6a_data[figure_6a_data$adm_area_1%in%top_n,c("adm_area_1","longitude")])
figure_6a_longitudes <- figure_6a_longitudes[order(figure_6a_longitudes$longitude),]
figure_6a_longitudes <- figure_6a_longitudes$adm_area_1

figure_6a_data$State <- figure_6a_data$adm_area_1
levels(figure_6a_data$State) <- c(levels(figure_6a_data$State), "Others")
figure_6a_data[!figure_6a_data$adm_area_1%in%top_n,"State"] <- "Others"
figure_6a_data$State <- factor(figure_6a_data$State, levels=c(lapply(figure_6a_longitudes, as.character), "Others"))

figure_6a_agg <- aggregate(figure_6a_data[c("new_per_day_smooth")],
                           by = list(figure_6a_data$State,figure_6a_data$date),
                           FUN = sum,
                           na.rm=TRUE)
figure_6a_agg <- plyr::rename(figure_6a_agg, c("Group.1"="State","Group.2"="date"))

# figure 6b processing
# Compute new cases per 10000 popuation
figure_6b_data$new_cases_per_10k <- 10000*figure_6b_data$new_cases/figure_6b_data$Population

# Define which dates to plot in choropleth
date_1 <- as.Date("2020-04-08")
date_2 <- as.Date("2020-07-21")
date_3 <- as.Date("2021-01-04")

# Subset for the two dates select
figure_6b1_data <- subset(figure_6b_data,date==date_1)
figure_6b2_data <- subset(figure_6b_data,date==date_2)
figure_6b3_data <- subset(figure_6b_data,date==date_3)

# Set max value to show. Censor any values above this 
color_max <- 250
figure_6b1_data$new_cases_censored <- figure_6b1_data$new_cases
figure_6b1_data$new_cases_censored[figure_6b1_data$new_cases_censored > color_max] <- color_max
figure_6b2_data$new_cases_censored <- figure_6b2_data$new_cases
figure_6b2_data$new_cases_censored[figure_6b2_data$new_cases_censored > color_max] <- color_max
figure_6b3_data$new_cases_censored <- figure_6b3_data$new_cases
figure_6b3_data$new_cases_censored[figure_6b3_data$new_cases_censored > color_max] <- color_max

# Convert the dataframe for figure_6b1 and 4b2 data into spatial dataframe
# Remove rows with NA in geometry. Required to convert column to shape object
figure_6b1_data <- subset(figure_6b1_data,!is.na(geometry))
figure_6b2_data <- subset(figure_6b2_data,!is.na(geometry))
figure_6b3_data <- subset(figure_6b3_data,!is.na(geometry))
# Convert "geometry" column to a sfc shape column 
figure_6b1_data$geometry <- st_as_sfc(figure_6b1_data$geometry)
figure_6b2_data$geometry <- st_as_sfc(figure_6b2_data$geometry)
figure_6b3_data$geometry <- st_as_sfc(figure_6b3_data$geometry)
# Convert dataframe to a sf shape object with "geometry" containing the shape information
figure_6b1_data <- st_sf(figure_6b1_data)
figure_6b2_data <- st_sf(figure_6b2_data)
figure_6b3_data <- st_sf(figure_6b3_data)


# figure 6: USA time series and choropleth ----------------------------------------------
# Set up colour palette
my_palette_1 <- brewer.pal(name="YlGnBu",n=4)[2]
my_palette_2 <- brewer.pal(name="YlGnBu",n=4)[4]
my_palette_3 <- "GnBu"
my_palette_4 <- brewer.pal(name="Oranges",n=4)[4]
# Viridis color palette with last item gray
v_palette <-  viridis(n+1, option="D")
v_palette[n+1] <- "#C0C0C0"

# figure 6a: Stacked Area Time series of US counties
figure_6a <-  (ggplot(data=figure_6a_agg, aes(x=date,y=new_per_day_smooth,fill=State))
               + geom_area(alpha=0.8, colour="white", na.rm=TRUE, size=0.3)
               + scale_fill_manual(values = v_palette)
               + labs(title="figure 6: New Cases Over Time for US States", y="New Cases per Day", x="Date")
               + scale_x_date(date_breaks="months", date_labels="%b")
               + scale_y_continuous(expand=c(0,0), limits=c(0, NA))
               + theme_classic(base_size=8,base_family='serif')
               + theme(plot.title = element_text(hjust = 0.5, size=9),
                       plot.margin=unit(c(0,0,0,0),"pt"), legend.position = c(0.09, 0.55),legend.key.size = unit(0.25, "cm"),
                       legend.title = element_text(size = 8),legend.text = element_text(size = 8)))
ggsave("./output/png/figure_6a.png", plot = figure_6a, width = 16,  height = 7, units='cm', dpi=300)

# figure 6b: Choropleth of US counties at USA peak dates
figure_6b1 <- (ggplot(data = figure_6b1_data) 
               + geom_sf(aes(fill=new_cases_censored), lwd=0, color=NA, na.rm=TRUE, show.legend=FALSE)
               + labs(title=date_1, fill="New Cases per Day")
               + scale_fill_distiller(palette=my_palette_3, trans="reverse", limits=c(color_max,0))
               + scale_x_continuous(expand=c(0,0), limits=c(-125, -65)) # coordinates are cropped to exclude Alaska
               + scale_y_continuous(expand=c(0,0), limits=c(24, 50))
               + theme_void()
               #+ guides(fill = guide_colourbar(barwidth = 30, barheight = 0.6, reverse=T))
               + theme(plot.title = element_text(hjust = 0.5,size=8,family='serif'), panel.grid.major=element_line(colour = "transparent"),legend.position="bottom"))
ggsave("./output/png/figure_6b1.png", plot = figure_6b1, width = 5.3,  height = 2.9, units='cm', dpi=300)

figure_6b2 <- (ggplot(data = figure_6b2_data) 
               + geom_sf(aes(fill=new_cases_censored), lwd=0, color=NA, na.rm=TRUE, show.legend=FALSE)
               + labs(title=date_2, fill="New Cases per Day")
               + scale_fill_distiller(palette=my_palette_3, trans="reverse", limits=c(color_max,0))
               + scale_x_continuous(expand=c(0,0), limits=c(-125, -65)) # coordinates are cropped to exclude Alaska
               + scale_y_continuous(expand=c(0,0), limits=c(24, 50))
               + theme_void()
               #+ guides(fill = guide_colourbar(barwidth = 30, barheight = 0.6, reverse=T))
               + theme(plot.title = element_text(hjust = 0.5,size=8,family='serif'), panel.grid.major=element_line(colour = "transparent"),legend.position="bottom"))
ggsave("./output/png/figure_6b2.png", plot = figure_6b2, width = 5.3,  height = 2.9, units='cm', dpi=300)

figure_6b3 <- (ggplot(data = figure_6b3_data) 
               + geom_sf(aes(fill=new_cases_censored), lwd=0, color=NA, na.rm=TRUE)#, show.legend=FALSE)
               + labs(title=date_3, fill="New Cases per Day")
               + scale_fill_distiller(palette=my_palette_3, trans="reverse", limits=c(color_max,0))
               + scale_x_continuous(expand=c(0,0), limits=c(-125, -65)) # coordinates are cropped to exclude Alaska
               + scale_y_continuous(expand=c(0,0), limits=c(24, 50))
               + theme_void()
               + guides(fill = guide_colourbar(barwidth = 20, barheight = 0.3, reverse=T))
               + theme(plot.title = element_text(hjust = 0.5,size=8,family='serif'), panel.grid.major=element_line(colour = "transparent"),legend.position="bottom",
                       legend.title = element_text(vjust=1,size=9,family='serif'), legend.text = element_text(size=8,family='serif')))
ggsave("./output/png/figure_6b3.png", plot = figure_6b3, width = 5.3,  height = 2.9, units='cm', dpi=300)
ggsave("./output/png/figure_6b_legend.png", plot = figure_6b3, width = 15.9,  height = 2.9, units='cm', dpi=300)
