# Analysis of Airbnb listings in Amsterdam

In this project, we analyze the airbnb listings in Amsterdam for the year 2022-2023. 

We have written a medium article that highlights the most importan results from this project. The article can be found [here](https://medium.com/@subash.gireesan/to-airbnb-or-not-the-amsterdam-story-c1d92b06f2c5)

## Motivation

This project is a part of the Data Scientist Nanodegree program by Udacity. Here we use the CRoss Industry Standard Process for Data Mining (CRISP-DM) process to analyze the trends in the prices, reviews and availability of the various Airbnb listings in Amsterdam. 

The dataset that we use is obtained from the opensource website - http://insideairbnb.com/get-the-data/

We are interested in answering the following questions:

  1) Which neighbourhood is the most popular among tourists in Amsterdam? What are the average prices in the neighbourhoods? Is there a trend in the prices and reviews for the neighbourhoods? 
  2) What are the type of listings in Amsterdam and what is their average price? What are the most sought-after amenities provided by hosts in Amsterdam?
  3) What time of the year is the demand high for listings in Amsterdam? How does the price of a listing change through the year? 
  4) What factors affect the price of a listing?
  5) How effective are the regulations imposed by the municipality? Are there any listings that are breaking the rules?
  
## Getting Started

### File Description

Data files: 

- Data/calendar.csv - Detailed calendar data inlcuding the price, availability etc
- Data/listings.csv - Detailed description of the listings including price, review scores, location etc
- Data/listings_summary.csv - Summary of all the listings in Amsterdam
- Data/neighborhoods.csv - List of the neighborhoods in Amsterdam
- Data/neighbourhoods.geojson - GeoJSON file of neighbourhoods of Amsterdam
- Data/reviews.csv - Detailed reviews for the listings
- Data/reviews_summary.csv - Details about listing and the date of review

calendar.csv and reviews.csv are in their corresponding zip format files. Please unzip before using the Jupyter Notebook. 

Analysis and modeling file:

- Notebooks/Amsterdam_Airbnb.ipynb - Jupyter notebook detailing the CRISP-DM process followed in the project along with the analysis and results

### Dependencies

* Python 3.9.7 with 
* Packages - numpy 1.21.5, pandas 1.4.1, seaborn 0.11.2, sklearn 1.0.2
* Juptyer notebook

## Authors

Contributors names and contact info

Subash Gireesan, Data Scientist, Shell Projects & Technologies

## Acknowledgments

* Credits must be given to insideairbnb for the data. Details related to licensing for the data and other descriptive information can be obtained [here](http://insideairbnb.com/get-the-data/).
* Authors of the following Notebooks for the inspiration
  - https://www.kaggle.com/code/erikbruin/airbnb-the-amsterdam-story-with-interactive-maps/data
  - https://github.com/debjani-bhowmick/airbnb-amsterdam
  - https://github.com/labrijisaad/exploratory-data-analysis-in-Python/blob/main/Cleaning_Data_in_Python.ipynb
* Users of Stackoverflow for their helpful posts and responses
