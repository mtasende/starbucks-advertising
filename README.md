# Starbucks Advertising
Starbucks challenge to select the best offers to show to each user in the cellphone app. This is the Capstone Project for Udacity's Data Scientist Nanodegree.

## Installations
The project uses Python 3.
To install it run:
```
$ conda env create -f sbucks.yml
$ source activate sbucks
$ pip install -e .
$ python src/data/make_dataset.py
```
The "make_dataset.py" script takes a long time to run (between 40 minutes and 1 hour). It is not necessary to run it, but the datasets have to be generated somewhere. If the script is not run, the main notebook will generate the datasets (taking longer to run).

## Project Motivation
The project is based on simulated data from Starbucks. Starbucks has different kinds of offers to their customers that use the mobile app. The offers are of three kinds:
 - BOGO (Buy One Get One): The customer gets a free product with the purchase of one. Valid for a determined duration.
 - Discount: For a period of time the product can be bought at a discount.
 - Informational: Just show ads to the customer.

The aim of the project is to find the best offers for each customer to maximize the probabilities of "offer completion", or maximize the profits. Only one product is considered.

I chose this project, as the Capstone project for Udacity's Data Scientist Nanodegree, because I like the company (as a customer), and I had previously completed a [challenge](https://github.com/mtasende/data-scientist-nanodegree/blob/master/projects/p04_starbucks/Starbucks.ipynb) from Starbucks with, what I think are, very good results, and I thought I could have good intuitions with this kind of data.

## Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- Documentation files for the project
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt` (it's better to use the conda
    |                         environment with this project)
    │
    ├── sbucks.yml         <- The file with the conda environment (preferred way to build the environment)
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── evaluation       <- Functions to evaluate models
    │   │   └── offer_success.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── clustering.py
    │   │   └── lagged.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── utils.py           <- Utility functions


## How to interact with your project
The main notebook is in `notebooks/Starbucks_Capstone_notebook.ipynb`. To reproduce the main results run that notebook. Most users will only want to run this.

In the folder `notebooks/offer_success_experiments` there is one notebook for each experiment that was run to try to predict the probability of "success" (that is, that an offer is viewed and completed) for an offer.

In the folder `notebooks/profit_10_days_experiments` there is one notebook for each experiment that was run to try to predict expected profit in the 10 days following the reception of an offer.

The `notebooks/scratchpad` folder contains notebooks used for the development process. They are not supposed to be run, unless you want to know more about the development process. They may not be well organized or easy to read.

## Licensing, Authors, Acknowledgements, etc.
Code released under the [MIT](https://github.com/mtasende/starbucks-advertising/blob/master/LICENSE) license.

This project was authored by Miguel Tasende.

Thanks to Starbucks for the dataset, and to Udacity for bringing the opportunity to work with it.

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
