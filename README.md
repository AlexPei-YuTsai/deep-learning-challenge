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

### Machine Learning Process

**Initial Test**
For a preliminary test, we used a basic 3-layer neural network just to see how far we can get. Ideally, we'd shoot for an 85% validation accuracy for a first pass.
1. Input Layer - ReLU Activation and as many neurons as there are input features (42 after dummification).
2. A single Hidden Layer - ReLU Activation and as many neurons as there are input features (42 after dummification). It appears that *generally* most problems can be handled with just a single Hidden Layer, so this is what we'll start with.
3. Output Layer - Sigmoid Activation and a single output neuron as our target variable is a binary value.

With those initial parameters, we managed to hit a test accuracy of **73.3%**, which is a C+ and not exactly the kind of performance we'd entrust our expensive decisions to. The model is trained for 100 epochs and only managed a 1.7% accuracy improvement during this time.

**Optimization**
To improve performance, we tried to change up the number of neurons, the number of hidden layers, the activation functions used in the input and hidden layers, and extended our training time to 200 epochs. We used Keras Tuner's Hyperband to see if there's good combination of hyperparameters we could use. The following is a list of tested combinations, denoted in `[Lists like these]`.
1. Input Layer - `[ReLU, Sigmoid, or TanH]` Activation and `[1-2]` times as many neurons as there are input features (42-84, steps of 10 used because my computer is not that strong).
2. `[1-3]` Hidden Layers - `[ReLU, Sigmoid, or TanH]` Activation and `[1-2]` times as many neurons as there are input features (42-84, steps of 10 used because my computer is not that strong).
3. Output Layer - Sigmoid Activation and a single output neuron as our target variable is a binary value. This layer remains the same.

TODO

## Results

The following performance metrics are used:
- [Balanced Accuracy Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html): The average recall of each class.
- Precision: The number of correctly made calls out of all calls that *predicted* a certain Class.
  - ${Precision} = \frac{True(Class)}{True(Class)+False(Class)}$
  - Class `0` Precision describes how many approved loans are offered to the right people. However, because of class imbalance, this is not the most informative statistic to use. 99% Class `0` Precision on 20,000 predictions means that 200 loans will probably default.
  - Class `1` Precision describes how many denied loans are denied from those who actually would default and don't deserve the chance.
- Recall: The number of correctly made calls out of all calls that *actually* belong to a certain Class.
  - ${Recall} = \frac{True(Class)}{True(Class)+False(Other Class)}$
  - Class `0` Recall describes how many applicants who would repay their loans just fine would have their loans approved.
  - Class `1` Recall describes how many applicants who would default on their loans would have their loans denied.
 
This bank's job is to reduce giving loans to those who don't deserve it, so *Class `1` Recall should be prioritized*. It's more harmful to let a defaulter have their loan than it is to accidentally deny a perfectly healthy application and have the aspiring borrower reapply at a later date.
 
**Standard Logistic Regression**
  - Balanced Accuracy Score: 95.2%
  - Precision:
    - Class `0`: ~100%
    - Class `1`: 85%
  - Recall:
    - Class `0`: 99%
    - Class `1`: 91%

**Randomly Over-Sampled Logistic Regression**
  - Balanced Accuracy Score: 99.4%
  - Precision:
    - Class `0`: ~100%
    - Class `1`: 84%
  - Recall:
    - Class `0`: 99%
    - Class `1`: **99%**

## Summary

As we defined success to mean "offering future defaulters less loans", the model using Randomly Over-Sampled Logistic Regression is the better one here with a Class `1` Recall of almost 100% - Only 4 defaulters were mistakenly allowed loans with this regression model. 
> The other model's 91% Class `1` Recall meant that almost 10% of all future defaults would have their applications approved. This is not conducive to the bank's financial wellbeing.

Though it's unfortunate that, regardless of the model, 15% of those who are denied are incorrectly denied, this is an acceptable tradeoff to make as those applicants can simply reapply at a later date and try their luck then.

---

## Code Breakdown
The premise of this challenge is simple: Given TODO? Overall, the machine learning code used will look like this:
```python
# Initialize a machine learning model
model = someAlgorithm(parameter1 = x1, parameter2 = x2, ...)

# Fit the model to your data
model.fit(yourDataHere)

# Predict unknown results based on your trained model
predictions = model.predict(yourDataHere)
```

## Resources that helped a lot
We aren't coding any of the machine learning algorithms from scratch. There's no need to reinvent the wheel or rediscover calculus for the purposes of this exercise. However, it's still important to learn about how the algorithms work and when these can be applied. I found these theory videos to be very useful:
- Cassie Kozyrkov's [Making Friends with Machine Learning](https://www.youtube.com/watch?v=1vkb7BCMQd0) 6-hour course is also great for giving people a look into the black boxes that now govern our data-centric world.

Frankly, the best way to learn Tensorflow is just to do Tensorflow. This is something [Google's Development Team](https://www.tensorflow.org/learn) made, so it's going to be more intuitive by default given their enormous market share and research funds. 

## FINAL NOTES
> Project completed on September 14, 2023
