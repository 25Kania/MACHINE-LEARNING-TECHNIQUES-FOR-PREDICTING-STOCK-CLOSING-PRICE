# MACHINE-LEARNING-TECHNIQUES-FOR-PREDICTING-STOCK-CLOSING-PRICE
Repo for data and models used within my Master Thesis.
## Data
Data folder contains two subfolders. The first raw_data subfolder contains raw data obtained after scraping for each stock.
The second subfolder expanded_data contains data for each stock containing expanded features and sentiment information.
## Code
Python file models.py contain implementation of mentioned models in my Master Thesis.
The data_handler.py module transforms data into suitable form for the training and testing of the models and works together with models.py module.
The data_handler module contains function used for visualisation of obtained predictions and obtained results.