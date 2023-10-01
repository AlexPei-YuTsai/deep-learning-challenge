# Neural Networks Challenge
> How can we make predictions in highly complex systems where the relationships between variables aren't necessarily linearly classifiable?

## Folder Contents
- A `.gitignore` file that ignores common things like PyCache, Jupyter Notebook checkpoints, and other common gitignorable Python entities. 
- A main `Alphabet Soup Charity` Jupyter Notebook file that imports data from a static link somewhere and throws it into a neural network we'll train.
- An `Model` folder containing the exported HDF5 neural network we trained in the main notebook.

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
- Josh Starmer's [Neural Networks Playlist](https://www.youtube.com/playlist?list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1) is a little lengthy, but gets into the nitty gritty of the mathematics behind these giant black boxes. It's a long series, but his explanations are generally intuitive and easily demystify foreign concepts.
- Cassie Kozyrkov's [Making Friends with Machine Learning](https://www.youtube.com/watch?v=1vkb7BCMQd0) 6-hour course is also great for giving people a look into the black boxes that now govern our data-centric world.

## FINAL NOTES
> Project completed on September 14, 2023
