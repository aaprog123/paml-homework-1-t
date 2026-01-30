[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/S8ViMLHq)
# Practical Applications in Machine Learning 

# Homework 1: Predict Housing Prices Using Regression

The goal of this assignment is to build a regression machine learning pipeline in a Streamlit web application to use as a tool to analyze the models to gain useful insights about model performance. Note, that the ‘Explore Dataset’ and ‘Preprocess Data’ steps from homework 0 are used in this assignment on Page A.

The <b>learning outcomes</b> for this assignment are:
* Develop multiple regression, polynomial, ridge, and lasso regression models for scratch.
* Evaluate regression methods using standard metrics including root mean squared error (RMSE), mean absolute error (MAE), and coefficient of determination (R2).
* Plot learning curves to inspect the cost function, detect underfitting, overfitting, to identify an ideal model.
* Apply cross validation to assess model performance across data splits (Extra Credit).

## Assignment Outline
* Setup
* End-to-End Regression Models & Evaluation
* Testing Code with Github Autograder

# Reading Prerequisite 

* Review the jupyter notebook in Chapter 4 Training Models of “Machine Géron, Aurélien. Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow.” O’Reilly Media, Inc., 2022. (Available on Canvas->Library Reserves). We will build on that example.
* Review Regression Notebooks in the notebooks folder

# Join Github Classrooom

* Create a team name for Github Classroom
* Clone your groups repository by join the Github Classroom for Homework 1

# End-to-End Regression Models for Housing Prices 

# California Housing Data

This assignment involves testing the end-to-end pipeline in a web application using a California Housing dataset from the textbook: Géron, Aurélien. Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow. O’Reilly Media, Inc., 2022. The dataset was captured from California census data in 1990. We have added additional features to the dataset. The features include:
* longitude - longitudinal coordinate
* latitude - latitudinal coordinate
* housing_median_age - median age of district
* total_rooms - total number of rooms per district
* total_bedrooms - total number of bedrooms per district
* population - total population of district
* households - total number of households per district'
* median_income - median income
* ocean_proximity - distance from the ocean
* median_house_value - median house value
* city - city location of house
* county - county of house
* road - road of the house
* postcode - zip code 

The Github repository contains two datasets in the dataset directory:
* housing_dataset - modified dataset for HW1.

# Training Models 

Complete the checkpoint functions for the following regression classes:
* LinearRegression class
* PolynomialRegression class
* RidgeRegression class
* LassoRegression class
* Cross Validation function

# Testing Code with Github Autograder

Test your homework solution as needed using Github Autograder. Clone your personal private copy of the homework assignment. Once the code is downloaded and stored on your computer in your directory of choice, you can start testing the assignment. To test your code, open a terminal and navigate to the directory of the homework1.py file. Then, type ‘pytest’ ad press enter. The autograder with print feedback to inform you what checkpoint functions are failing. Test homework1.py using this command in your terminal.

To run all test:

```
pytest
```

To run in verbose mode:

```
pytest -v
```

To run an specific checkpoint (i.e., checkpoint1):

```
pytest -m checkpoint1
```

To start and visualize your assignment's progress a in web interface:
```
streamlit run homework1.py
```

# Reflection Assessment

Complete on Canvas.

# Further Issues and questions ❓

If you have issues or questions, don't hesitate to contact the teaching team:
* Angelique Taylor, Instructor, amt298@cornell.edu
* Yuanchen Bai, TA, yb299@cornell.edu 
* Xinzi He, TA, xh278@cornell.edu
* Giuliano Pioldi, Grader, gp433@cornell.edu
* Yilan Fan, Grader, yf429@cornell.edu
* Onyinye Okoli, Grader, ojo5@cornell.edu
* Wenxuan Yu, Grader, wy279@cornell.edu