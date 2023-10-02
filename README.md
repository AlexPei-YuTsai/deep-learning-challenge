# Neural Networks Challenge
> How can we make predictions in highly complex systems where the relationships between variables aren't necessarily linearly classifiable?

## Folder Contents
- A `.gitignore` file that ignores common things like PyCache, Jupyter Notebook checkpoints, and other common gitignorable Python entities. The optimization log is also omitted for the purposes of this project.
- A main `AlphabetSoupCharity` Jupyter Notebook file that imports data from a static link somewhere and throws it into a neural network we'll train
  - A resulting `AlphabetSoupCharity` HDF5 file containing the model fitted to the data used.
- An `AlphabetSoupCharity_Optimization` Jupyter Notebook file that attempts to improve the validation accuracy and performance of the model from the previous file.
  - A resulting `AlphabetSoupCharity_Optimization` HDF5 file containing the final model after Hyperband Tuning.
- This `README` serves as both the installation instructions and the analysis for this project.

> Apparently HDF5 files are considered *legacy*, so it's better to follow the [official instructions](https://www.tensorflow.org/tutorials/keras/save_and_load) in the future.

### Installation/Prerequisites
- Make sure you can run Python. The development environment I used was set-up with:
```
conda create -n dev python=3.10 anaconda -y
```

#### Imported Modules
- Installing via the conda command given should give you access to most, if not all, of the script's modules locally. However, if you don't have them, be sure to grab yourself the following libraries:
  - [Pandas](https://pandas.pydata.org/docs/getting_started/install.html) for basic data management
  - [Scikit-Learn](https://scikit-learn.org/stable/install.html) for standard machine learning preprocessing functions
  - [Tensorflow](https://www.tensorflow.org/install) to program and run our neural networks
  - [Keras Tuner](https://keras.io/guides/keras_tuner/getting_started/) to optimize the neural network's hyperparameters

---

# Alphabet Soup Charity Funding Analysis Report

## Overview of the Analysis

The Alphabet Soup organization, like other venture capital groups, wants to ensure that their investments provide returns. Given their illustrious portfolio and history with various organizations, we could potentially seek out a pattern through our neural network and identify what future applicants would be likely to succeed. Ideally, by referring to their application types, use cases, and other classification data, we can predict, to a good amount of accuracy, whether this project will succeed in the future.

By predicting whether a group will succeed or not, the Alphabet Soup group can be more cautious with their funds and ensure that more of their resources are properly used.

### Data Definitions

Each row of the `lending_data.csv` file used represents an organization the Alphabet Soup group funded. Each feature column describes something about the funded organization:

**Omitted Columns:** These are mainly descriptors for the dataset and are otherwise irrelevant to the machine learning model. These are removed during data preprocessing.
- `EIN` and `NAME`: Identification columns

**Feature Columns:** These are the clues we seek to use to predict what we want to predict. These are the independent variables for our model and we hope to use it to find some kind of pattern in the data with our neural network. Categorical variables are dummified into binary columns and numerical variables are normalized with a scaler.
- `APPLICATION_TYPE`: Alphabet Soup application type
- `AFFILIATION`: Affiliated sector of industry
- `CLASSIFICATION`: Government organization classification
- `USE_CASE`: Use case for funding. Use cases include "Product Development" and "Healthcare".
- `ORGANIZATION`: Organization type
- `STATUS`: Active status
- `INCOME_AMT`: Income classification
- `SPECIAL_CONSIDERATIONS`: Special considerations for application
- `ASK_AMT`: Funding amount requested

**Target Column:** This is what we want to predict. This is the answer to the question, "Given the information above, will this next project succeed or tank?", we'd like to see.
- `IS_SUCCESSFUL`: Was the money used effectively. `1` means yes, `0` means no.

## Neural Network Process

### Initial Test
For a preliminary test, we used a basic 3-layer neural network just to see how far we can get. Ideally, we'd shoot for an 85% validation accuracy for a first pass.
1. Input Layer - ReLU Activation and as many neurons as there are input features (42 after dummification).
2. A single Hidden Layer - ReLU Activation and as many neurons as there are input features (42 after dummification). It appears that *generally* most problems can be handled with just a single Hidden Layer, so this is what we'll start with.
3. Output Layer - Sigmoid Activation and a single output neuron as our target variable is a binary value.

#### Initial Results

With those initial parameters, we managed to hit a test accuracy of **73.3%**, which is a C+ and not exactly the kind of performance we'd entrust our expensive decisions to. The model is trained for 100 epochs and only managed a 1.7% accuracy improvement during this time.

### Optimization
To improve performance, we tried to change up the number of neurons, the number of hidden layers, the activation functions used in the input and hidden layers, and extended our training time to 200 epochs. We used Keras Tuner's Hyperband to see if there's good combination of hyperparameters we could use. The following is a list of tested combinations, denoted in `[Lists like these]`.
1. Input Layer - `[ReLU, Sigmoid, or TanH]` Activation and `[1-2]` times as many neurons as there are input features (42-84, steps of 10 used because my computer is not that strong).
2. `[1-3]` Hidden Layers - `[ReLU, Sigmoid, or TanH]` Activation and `[1-2]` times as many neurons as there are input features (42-84, steps of 10 used because my computer is not that strong).
3. Output Layer - Sigmoid Activation and a single output neuron as our target variable is a binary value. This layer remains the same.

After about 2 hours of hyperparameter tuning, we've settled on the following structure:
1. Input Layer - 82 Neurons. TanH Activation.
2. 3 Hidden Layers - 42, 72, 82 Neurons in that order. TanH Activation.
3. Output Layer - 1 Neuron. Sigmoid Activation.

#### Optimized Results

We've improved to a remarkable **73.37%** validation accuracy, which means that just fiddling with a neural network's parameters is not sufficient for our purposes. Perhaps we need more data or a completely different approach entirely. Perhaps we're overfitting by virtue of there having 42 input columns and that we needed to shrink it down to a more manageable size. Regardless, more thinking is required.

## Results

It's probably a better idea to run this through a Decision Tree, Random Forest, or Adaboost given how this is a classification problem and these methods are best suited for something like this. It would also be a good idea to do some exploratory data analysis and aggregate data to see if any particular labels were more successful than the others or see if any numerical variables correlate with the logistic representation of our success column. 

Speaking of which, doing Logistical Regression with the same data yields a balanced accuracy score of **72.2%**, so a thorough reexamining of the data is recommended instead.

## Summary

Regardless of the method used and the optimization or lack thereof, accuracy remained low at around 73%. The problem probably comes from the data and would require more samples, more numerical features, or a review of what is considered relevant information.

---
## Resources that helped a lot
We aren't coding any of the machine learning algorithms from scratch. There's no need to reinvent the wheel or rediscover calculus for the purposes of this exercise. However, it's still important to learn about how the algorithms work and when these can be applied. I found these theory videos to be very useful:
- Cassie Kozyrkov's [Making Friends with Machine Learning](https://www.youtube.com/watch?v=1vkb7BCMQd0) 6-hour course is also great for giving people a look into the black boxes that now govern our data-centric world.
- Josh Starmer's [Neural Networks Playlist](https://www.youtube.com/playlist?list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1) is a little lengthy, but gets into the nitty gritty of the mathematics behind these giant black boxes. It's a long series, but his explanations are very intuitive.

Frankly, the best way to learn Tensorflow is just to do Tensorflow. This is something [Google's Development Team](https://www.tensorflow.org/learn) made, so it's going to be more intuitive by default given their enormous market share and research funds. 

## FINAL NOTES
> Project completed on September 14, 2023
