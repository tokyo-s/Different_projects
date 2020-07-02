import pandas
import numpy
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pandas.read_csv("student-mat.csv", sep=";")

print(data.head())

data = data[["G1", "G2", "G3", "studytime", "failures"]]

print(data.head())

predict = "G3"

x = numpy.array(data.drop([predict], 1))
y = numpy.array(data[predict])


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

best = 0
for _ in range(1000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)
    if acc > best:
        best = acc
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f) 

print("best: ", best)

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

#print(linear.score(x_test, y_test))

print(linear.coef_)
print(linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

p = "G2"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"] )
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()


