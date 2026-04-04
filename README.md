#                                                                          scratch-linear-regression
Linear regression implemented from scratch in C++ using Eigen, with gradient descent, Python scripts for data preprocessing, experiment automation and results on the Boston Housing dataset.

This project implements a linear regression model from scratch in C++, leveraging the Eigen library for efficient matrix operations. The model is trained using gradient descent, with support for configurable learning rate, maximum iterations, and tolerance for early stopping.

The repository includes scripts to preprocess data using Python, run multiple randomized experiments, and analyze results with metrics such as R² and mean squared error (MSE) on the boston housing dataset. It serves both as a learning project for understanding linear regression and a demonstration of custom C++ machine learning implementation.

#                                                              Linear Regression in C++ Using Eigen (From Scratch)

The aim of this project was to implement a linear regression algorithm from scratch in C++ and conduct experiments to test its performance.
The algorithm itself is written in C++ and uses the Eigen library for matrix calculations.
The method used is gradient descent, where the weights are iteratively updated by calculating the error between predicted and actual values, nudging the weights in the direction that reduces this error.

# Linear Regression Concept
The concept of linear regression in itself is similar to a straight best fit line drawn to predict future values on trends, except its across multiple dimensions instead of being limited to a 2 dimensional graph. Of course its hard to visualize a line beyond 4 dimensions but we don’t need to as the math can handle that part for us.
      
      y=mx+c

The straight line equation which is used to draw a straight line on a 2d space.

    y=m1x1 + m1x2 +mnxn+c

This is used to represent a straight line through multiple dimension space.

In the context on linear regression y is referred to as predictions, m is referred to as weight represented with w and c is referred to as the bias.

    Predictions =w1x1 + w1x2+……+wnxn+bias

or
    
$$
y_{\text{pred}} = b + \sum_{j=1}^{n} w_j x_j
$$

# Gradient Descent

Gradient descent works by:

Calculating predictions using the current weights.
Computing the error between predictions and actual values.
Updating weights slightly to reduce the error.

This process repeats for multiple iterations until the error stops improving. The final weights can then be used to make predictions on new data.

# Training Parameters
learningRate: Determines how much the weights are updated each iteration.
maxIter: Maximum number of iterations for training.
tolerance: Early stopping criterion if error improvement is smaller than this threshold.

# Data Preprocessing

Python and scikit-learn were used to clean and normalize the data before feeding it to the C++ program.

# Experimental Setup
The custom linear regression model was run 100 independent times on the Boston Housing dataset, using randomized train-test splits for each run.

Learning rate: 0.077
Max iterations: 50,000
Tolerance: 1e-6

Experiment Summary
|Metric|	Average|	Min|	Max|
|-------|--------|--------|-------|
|R²	|0.716	|0.558|	0.835|
|MSE	|23.18|	14.27|	39.78|

These results demonstrate that the model consistently captures the majority of the trend in the data, with some variability due to different train-test splits.

# Possible Improvements

The model could be further enhanced through:

Stochastic Gradient Descent (SGD)
Mini-Batch Gradient Descent
Momentum
Adaptive Learning Rates
Feature Engineering
Regularization techniques

# Follow the README to run the experiment yourself.

# Requirements

Before running the project, make sure you have the following installed:

# C++
A C++ compiler that supports C++11 or later.
The project uses the Eigen library for matrix operations. Make sure it’s installed or included in the project folder.
# Python
Python 3.

# Recommended: Use a virtual environment

It’s highly recommended to create a virtual environment to keep project dependencies isolated:

# Create a virtual environment
    python -m venv venv

# Activate it (Windows)
    venv\Scripts\activate

# Activate it (Linux/macOS)
    source venv/bin/activate

#   Install required Python libraries

You can install all necessary packages using pip:

    pip install numpy pandas matplotlib scikit-learn

Or use a requirements.txt:

    pip install -r requirements.txt

# Tutorial: How to Run the Program

This project includes a custom C++ linear regression model, Python scripts for data preprocessing, and experiment automation. Below is a step-by-step guide to running it.

# 1: Compile the C++ Program

The repository includes a Makefile that compiles the two C++ source files into a single executable.

    make

This will produce an executable, e.g., main.exe (Windows) or main (Linux/macOS).

# A: Single Instance Run

To run the program once:

# A-1: Clean the dataset
Run the Python script cleaning.py. This produces two CSV files: train.csv and test.csv.

    python cleaning.py

# A-2: Run the C++ executable
Execute the compiled program. It will automatically read the train.csv and test.csv files.

    ./main   # or main.exe on Windows

Provide training parameters
The program will prompt you to enter:

Learning rate
Maximum iterations
File name prefix

The results will automatically be saved in the results/ folder (make sure it exists).

# A-3: Analyze the results
After the program finishes, run analytics.py. It will prompt you to enter the file prefix and automatically generate statistics and plots.

    python analytics.py

# B: Multiple Experiment Runs

To run multiple experiments automatically:

# B-1: Run RunExperiment.py. It will prompt you for:

Learning rate
Maximum iterations
File prefix
Number of runs

Example:

    python RunExperiment.py

This script will run the C++ executable multiple times, saving each output file with the specified prefix.

# B-2: Analyze after all runs
Once all experiments finish, run analytics.py manually to process the generated files and summarize results.

#  Notes
The results/ folder must exist before running the program.
Python scripts interactively prompt the user for necessary inputs.
The C++ program performs the core linear regression training, while Python handles data cleaning, experiment automation, and analytics.
Train/test splits are randomized each run for reproducibility.

