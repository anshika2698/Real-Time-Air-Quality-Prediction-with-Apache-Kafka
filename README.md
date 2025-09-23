# Real-Time-Air-Quality-Prediction-with-Apache-Kafka
This project uses Apache Kafka for real-time data streaming and environmental time series analysis. Develop an end-to-end data pipeline using the UCI Air Quality dataset to demonstrate merit in streaming architecture, exploratory data analysis, and predictive modeling deployment. 

## Table of Contents
- [Description](#description)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Dataset
- The dataset contains hourly air quality measurements from a gas multisensor device deployed on the field in an Italian city from March 2004 to February 2005.
- Features include :
0 Date	(DD/MM/YYYY)
1 Time	(HH.MM.SS)
2 True hourly averaged concentration CO in mg/m^3  (reference analyzer)
3 PT08.S1 (tin oxide)  hourly averaged sensor response (nominally  CO targeted)	
4 True hourly averaged overall Non Metanic HydroCarbons concentration in microg/m^3 (reference analyzer)
5 True hourly averaged Benzene concentration  in microg/m^3 (reference analyzer)
6 PT08.S2 (titania) hourly averaged sensor response (nominally NMHC targeted)	
7 True hourly averaged NOx concentration  in ppb (reference analyzer)
8 PT08.S3 (tungsten oxide) hourly averaged sensor response (nominally NOx targeted) 
9 True hourly averaged NO2 concentration in microg/m^3 (reference analyzer)	
10 PT08.S4 (tungsten oxide) hourly averaged sensor response (nominally NO2 targeted)	
11 PT08.S5 (indium oxide) hourly averaged sensor response (nominally O3 targeted)
12 Temperature in Â°C	
13 Relative Humidity (%) 	
14 AH Absolute Humidity
- Source: [UCI Machine Learning Repository - Air Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Air+Quality)


