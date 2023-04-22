# Biomarker Case Study

[![License](https://img.shields.io/github/license/levon003/biomarker-case-study)](https://github.com/levon003/biomarker-case-study/blob/main/LICENSE)


Modeling with patient biomarker data. This repository is a self-contained demonstration of my approach to exploring a dataset and building a machine learning model for a binary classification task with missing data.

Author: [Zachary Levonian](https://github.com/levon003)

## Summary

A good entrypoint to this analysis is [the Jupyter notebook that trains and evaluates models to predict the binary outcome](/notebook/DataModeling.ipynb). Initial exploration and description of the data is in [this Jupyter notebook](/notebook/DataExploration.ipynb).


## Data

Synthetic patient data provided by [Tempus](https://www.tempus.com). I don't have permission to share the data, although you can see excerpts in the analysis notebooks.
Data is assumed to be present in the `data` folder.

## Build and dependencies

Just `make install`. Requires Python 3.10 or greater.
Poetry is used for managing Python dependencies, and will be installed if it isn't already available.

## Repository structure

The directory layout is:

- `notebook` contains the analysis notebooks.
- `src` contains the `bcs` Python package with helper functions and classes to support the analysis.
- `data` is presumed to be the location of the input data... see the Data section for more details.
- `figures` contains any images produced within the analysis notebooks.
