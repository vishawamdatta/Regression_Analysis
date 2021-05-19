# Parameter Importance In Regression 
Determining the relative importance of feature variables of the boston housing dataset by varying lambda values in ridge and lasso regression.

Used the ridge and lasso regression models and plotted a graph
between different values of regularization coefficient (lambda)
and the coefficients of the feature variables.On increasing lambda 
the features whose coefficients decrease at a rapid rate are found 
to be less important and may cause overfitting because these may adjust 
to the errors between the actual and the predicted values instead of predicting 
the required values.

Since the OLS (ordinary least square) model is prone to overfitting , the ridge 
and the lasso regression model try to remove this shortcoming by introducing some bias
to reduce the variance , hence along with determining feature importance, an analysis of the 
relative performance of the three models is also done and analysed.

Every important inference and insights are mentioned in the detailed results and inferences document 
provided along with the source code(which fits the data into the required model and obtains the required plots for the parameter importance estimation and analysis).
