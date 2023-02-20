# Disaster Response Pipeline Project

In this project, we have built a NLP tool to classify real messages that are sent during disaster such that the messages can be send to an appropriate disaster relief agency in real time. We also have included a web app where an emergency worker can input a new message and get classification results in several categories.

## Motivation

This project is a part of the Data Scientist Nanodegree program by Udacity. The initial dataset contains pre-labelled tweet and messages from real-life disasters. The aim of this project is to build a Natural Language Processing tool that categorize messages. 

The dataset has been provided by [Apppen] [https://www.figure-eight.com/] (formally Figure 8). 

The project follows these sequential steps: 

1) ETL Pipeline to extract data from source, clean data and save it in a proper database structure
2) ML Pipeline to train a model that can classify text messages into different categories
3) Web App to visualize model results in real time

## Getting Started

### Folder Structure 
```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # Categories of different messages
|- disaster_messages.csv  # Messages data
|- process_data.py # ETL pipeline script
|- DisasterResponse.db   # Database with clean data

- models
|- train_classifier.py # ML pipeline script
|- classifier.pkl  # saved model 

- README.md
```

### Dependencies

* Python 3.9.7 
* Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
* Natural Language Process Libraries: NLTK
* SQLlite Database Libraqries: SQLalchemy
* Web App and Data Visualization: Flask, Plotly

### Installation:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

## Authors

Subash Gireesan

## Acknoweledgment

* Lecture notes for the Udacity Data Science Nanodegree
* Credit must be given to Figure-8 for the dataset. 
* Authors of the following projects for inspiration: 
    - https://github.com/matteobonanomi/dsnd-disaster-response
    - https://github.com/kaish114/Disaster-Response-Pipelines
    - https://github.com/Swatichanchal/Disaster-Response-Pipeline
* Users of Stackoverflow for their helpful posts and responses
