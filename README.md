Logistic Regression from Scratch in Python

5 minuteread

In this post, I’m going to implement standard logistic regression from scratch. Logistic regression is a generalized linear model that we can use to model or predict categorical outcome variables. For example, we might use logistic regression to predict whether someone will be denied or approved for a loan, but probably not to predict the value of someone’s house.

So, how does it work? In logistic regression, we’re essentially trying to find the weights that maximize the likelihood of producing our given data and use them to categorize the response variable. Maximum Likelihood Estimation is a well covered topic in statistics courses (my Intro to Statistics professor has a straightforward, high-level description here), and it is extremely useful.

Since the likelihood maximization in logistic regression doesn’t have a closed form solution, I’ll solve the optimization problem with gradient ascent. Gradient ascent is the same as gradient descent, except I’m maximizing instead of minimizing a function. Before I do any of that, though, I need some data.

Generating Data
I’m going to use simulated data. I can easily simulate separable data by sampling from a multivariate normal distribution.

Picking a Link Function

Generalized linear models usually tranform a linear model of the predictors by using a link function. In logistic regression, the link function is the sigmoid.

Maximizing the Likelihood

To maximize the likelihood, I need equations for the likelihood and the gradient of the likelihood. Fortunately, the likelihood (for binary classification) can be reduced to a fairly intuitive form by switching to the log-likelihood. We’re able to do this without affecting the weights parameter estimation because log transformations are monotonic.



Calculating the Log-Likelihood

The log-likelihood can be viewed as a sum over all the training data. Mathematically,
ll=N∑i=1yiβTxi−log(1+eβTxi)

where y
is the target class (0 or 1), xi is an individual data point, and β

is the weights vector.

I can easily turn that into a function and take advantage of matrix algebra.

Calculating the Gradient

Now I need an equation for the gradient of the log-likelihood. By taking the derivative of the equation above and reformulating in matrix form, the gradient becomes:
▽ll=XT(Y−Predictions)

Like the other equation, this is really easy to implement. It’s so simple I don’t even need to wrap it into a function.
Building the Logistic Regression Function

Finally, I’m ready to build the model function. I’ll add in the option to calculate the model with an intercept, since it’s a good option to have.

