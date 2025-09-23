# Real-Time-Air-Quality-Prediction-with-Apache-Kafka
This project uses Apache Kafka for real-time data streaming and environmental time series analysis. Develop an end-to-end data pipeline using the UCI Air Quality dataset to demonstrate merit in streaming architecture, exploratory data analysis, and predictive modeling deployment. 

## Table of Contents
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Dataset
- The dataset contains hourly air quality measurements from a gas multisensor device deployed on the field in an Italian city from March 2004 to February 2005.
- Features include: Date	(DD/MM/YYYY), Time	(HH.MM.SS), True hourly averaged concentration CO in mg/m^3  (reference analyzer), PT08.S1 (tin oxide)  hourly averaged sensor response (nominally  CO targeted), True hourly averaged overall Non Metanic HydroCarbons concentration in microg/m^3 (reference analyzer), True hourly averaged Benzene concentration  in microg/m^3 (reference analyzer), PT08.S2 (titania) hourly averaged sensor response (nominally NMHC targeted), True hourly averaged NOx concentration  in ppb (reference analyzer), PT08.S3 (tungsten oxide) hourly averaged sensor response (nominally NOx targeted), True hourly averaged NO2 concentration in microg/m^3 (reference analyzer), PT08.S4 (tungsten oxide) hourly averaged sensor response (nominally NO2 targeted), PT08.S5 (indium oxide) hourly averaged sensor response (nominally O3 targeted), Temperature in Â°C, Relative Humidity (%), AH Absolute Humidity
- Source: [UCI Machine Learning Repository - Air Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Air+Quality)

## Installation

Docker Installation

-Visit [Docker Desktop](https://www.docker.com/products/docker-desktop/) and download the version for your OS (Windows/macOS/Linux).
- Follow the installation instructions on the website.
  
Verify Installation
 ```bash 
 docker --version
 docker-compose --version
  ```
- Start Docker by launching Docker Desktop

Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/repo.git](https://github.com/anshika2698/Real-Time-Air-Quality-Prediction-with-Apache-Kafka.git)
   cd repo
   ```



