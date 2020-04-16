# Artificial-Neural-Network---to-predict-churn-rates-of-a-bank
The code is written in python and implements Artificial Neural Networks along with data preprocessing (Dealing with categorical variables, Standard Scaling), Keras Classifier from keras.wrappers.scikit_learn, Dropout from keras.layers, cross validation and GridSearchCV to predict the probability of a customer leaving the bank or not with an accuracy of about 86%.

The data set has been taken from https://www.superdatascience.com/pages/deep-learning.

If your OS kills the process, try setting the n_jobs parameter of GridSearchCV to 1.

i used the random_state = 2 for the train_test_split and got an accuracy of 86%.

The best params from GridSearchCV were - estimator = 'rmsprop',
                                         epochs = 500,
                                         batch_size = 25
