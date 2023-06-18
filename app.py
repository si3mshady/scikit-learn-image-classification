import os
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

PATH = 'clf-data/'

categories = ['empty','not_empty']

data = []
labels = []

for index,category in enumerate(categories):
    for file in os.listdir(os.path.join(PATH,category)):
        img_path = os.path.join(PATH, category,file)
        img = imread(img_path)
        img_resize =resize(img,(15,15))
        data.append(img_resize.flatten())#make data a array
        labels.append(index)
        #resize


data = np.asarray(data)
labels = np.asarray(labels)


x_train,x_test, y_train, y_test = train_test_split(data,labels,test_size=.2,shuffle=True,stratify=labels)

# This line of code is splitting some data into two parts, called x_train and x_test,
#  and splitting some labels into two parts, called y_train and y_test.
#  It's like taking a bag of objects and dividing them into two groups. 
# The test_size=.2 means that 20% of the data will be put in the x_test and y_test groups,
#  while the remaining 80% will be in the x_train and y_train groups. 
# The shuffle=True part means that the data will be mixed up randomly before splitting, 
# like shuffling a deck of cards. Finally, the stratify=labels part ensures that both 
# the x_train and x_test groups have a similar proportion of different labels, so it's fair and representative.


#classifier - model
model = SVC()
params =  [{  "gamma":[0.01,0.001,0.0001], "C": [1,10,100,1000] }]
grid_search =  GridSearchCV(model,params)

# This code is about training a machine learning model to classify things.
#  The model is like a special kind of robot that learns from examples. In this case, the model is called "SVC".
# The next part, params, is like a list of options for the model to try out. 
# It includes different values for "gamma" and "C". 
# Think of these as settings or choices that the model can adjust to make better predictions.
# Then comes grid_search. It's like a helper that helps the model find the best combination of settings from the options in params. It tries different combinations of settings and compares how well the model performs with each combination.
# So, the code is saying to the model, "Hey, try out different settings from params and use grid_search to find the best ones for making accurate predictions."


#models are the most important component to ML applications it's critical to understand how they work and how to configure 
#them for use in applications

grid_search.fit(x_train,y_train)

#test 

#best_estimator will become the model
best_estimator = grid_search.best_estimator_

y_prediction = best_estimator.predict(x_test)

score = accuracy_score(y_prediction,y_test)

print(f"{score * 100}% accuracy")


data = {"classifier": best_estimator}
with open('image_classifier.pkl', 'wb') as file:
    pickle.dump(data,file)



#make numpy


#I always wondered how image classifcation  works
#create numpy arrays

#train test split