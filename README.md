# Disaster Response Pipeline Project

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [Project Components](#projectComponents)
4. [Folder Structure](#folderStructure)
5. [Modelling](#model)
6. [Instructions](#instructions)
7. [Licensing, Authors, and Acknowledgements](#licensing)

### Installation <a name="installation"></a>

Apart from Anaconda distribution of Python, this code should requires stopwords downloaded from nltk package.

### Project Motivation <a name="motivation"></a>
In this project, I have applied Data Science skills to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.


### Project Components <a name="projectComponents"></a>
There are three components in this project.

1. ETL Pipeline

	- `process_data.py`

		- Loads the messages and categories datasets
		- Merges the two datasets
		- Cleans the data
		- Stores it in a SQLite database

2. ML Pipeline
	
	- `train_classifier.py`

		- Loads data from the SQLite database
		- Splits the dataset into training and test sets
		- Builds a text processing and machine learning pipeline
		- Trains and tunes a model using GridSearchCV
		- Outputs results on the test set
		- Exports the final model as a pickle file

3. Flask Web App


### Folder Structure <a name="folderStructure"></a>

```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- utilts.py # has the custom Estimators required for the pipeline

- README.md
```


### Modelling <a name="model"></a>

Pipeline makes use of the the Multioutput Classifier on top of RandomForrest Classifier in order to categories the given message into different categories. Grid Search is used in order to find the best parameters for the models. Model is trained based on the following features

* Messages
* Genre

### Instructions <a name="instructions"></a>

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python app/run.py`

3. Go to http://0.0.0.0:3001/

### Licensing, Authors, and Acknowledgements <a name="licensing"></a>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

* [Udacity](https://www.udacity.com/)
* [Figure Eight](https://www.figure-eight.com/)
