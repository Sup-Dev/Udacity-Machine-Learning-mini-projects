# Enron Data POI classifier

### Summary

The main objective of this exercise was to find the persons of interest in Enron corporate fraud. This makes it a very good candiate for machine learning as we can create classifiers based on machine learning to look for persons of interest in the dataset. The dataset contains a list Enron employees and their financial data, like salary, bonus, stock options etc. The most obvious outlier in the data was the poi "TOTAL", which in fact is the total of the fields and not a person of interest.

### Feature Select

For this step I decided to create two aditional fields, namely **to_poi_messages_percent** and **from_poi_messages_percent**. These are the percentage of messages received and sent to pois.

To select the number of features, I decide to use SelectKBest with k=11 based on the output graph.

### Algorithm 

The algorithm that I picked was **NearestCentroid** as it had both precision and recall over 0.3, with an accuracy over 0.85. The other algorithm that I tried was AdaBoost, but the NearestCentroid had better overall results.

### Tuning

The process that I used to tune the algorithm was using the **GridSearchCV** as it tests out different parameters and returns the best value as per the scoring formulation specified.

In case of **NearestCentroid** I picked the *metric* from 'cosine', 'euclidean', 'l1', 'l2'. And similarly, I picked the *shrink_threshold* as 0.9, 0.7, 0.5.

### Validation

The main reason behind valid is to check if the classifier has a tendency to overfit and also it give an estimate of the performace on an independent dataset. For cross validation **StratifiedShuffleSplit** was used. This algo created a random splits k-number of times to find the best split based on the normal distribution.

### Evaluation 

For the evaluation metric, it was focused on the accuracy and precision. The accuracy tell us the percentage of times our algorithm guessed correctly. Precision tell us for a given entry how many times it guesses it correctly.

In terms of POI, precision will tell us how good our algorithm is at catching POIs. As for recall, it tells us how many of the predicted POIs are actual POIs.