# Jeopardy Question-and-Answer Database

![alt text](https://github.com/sgiles13/jeopardy/blob/main/jeopardy.png?raw=true)

Scraping data from j!-archive.com to build a comprehensive database of Jeopardy questions and answers. NLP and ML techniques are used on the data for classification.

## Loading the environment
The environment is included here as "environment.yml", and can be cloned via "conda env create -f environment.yml", and activated with "conda activate jeopardy".

## Scraping the data
The data used to populate the database is obtained from j!-archive.com. Running python main.py --write_csv will extract all episode data available from the j!-archive.com website, and write a CSV file of all the data. 

## Performing NLP on the data
One of my currently ongoing projects is using this data that has been scraped from j!-archive.com to perform NLP along with deep learning. I will push code here that does these taks in the near future.
