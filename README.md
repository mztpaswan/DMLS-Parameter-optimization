# DMLS Parameter Optimization
This project is about optimizing Direct Metal Laser Sintering (DMLS) process parameters to reduce porosity (%) in SS316L material using machine learning.

The main idea is to compare different regression models, select the one that performs best, and then use it to predict a set of process parameters that results in minimum porosity.

## Why this project?
In DMLS, small changes in process parameters like laser power or scan speed can significantly affect the final part quality.
Finding the right combination of parameters manually requires a lot of trial and error.

I built this project to understand:

How machine learning models can be used for regression problems
        
How process parameters influence porosity
        
How optimization can be done using model predictions instead of manual tuning

## Dataset Information
Material used:
        SS316L
        
Target variable: 
        Porosity_percent
        
Input parameters:
        Layer thickness (microns),
        Laser power (W),
        Scan speed (mm/s),
        Hatch spacing (mm),
        Powder bed temperature (°C),
        Material density (g/cm³),
        Quality rating
        
The dataset is stored in:
        ss316l.csv        
        
## Models Used
 * Linear & Polynomial Regression,
 * Ridge & Lasso,
 * Decision Tree,
 * Random Forest,
 * Gradient Boosting,
 * SVR,
 * kNN,
 * Neural Network (MLP)

## How to Run
Download or clone this repository  
Install the required libraries 
```bash
$ pip install numpy pandas scikit-learn
```
Run the Code  
```bash
$ python par.py
```
When prompted, choose a layer thickness from:
```bash
$ 20, 40, 70, or 90 microns
```
## What I learned
Comparing ML regression models

Importance of feature scaling

Using ML for manufacturing optimization

## Note
This project was built for learning purposes. Results should be experimentally validated.

## Author: Manjeet Kumar Paswan
