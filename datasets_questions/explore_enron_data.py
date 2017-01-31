#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print("Number of data points: " + str(len(enron_data)))
print("Number of features: " + str(len(enron_data[enron_data.keys()[0]])))
# print("List of features: " + str(enron_data[enron_data.keys()[0]]))

num_poi = 0
num_nan_total_payments = 0
num_nan_poi_total_payments = 0

for k,v in enron_data.items():
    if v["poi"] == 1:
        num_poi += 1

        if v["total_payments"] == "NaN":
            num_nan_poi_total_payments += 1

    if v["total_payments"] == "NaN":
        num_nan_total_payments += 1

print("Number of NaN for total payments: " + str(num_nan_total_payments))

print("Number of POI's: " + str(num_poi))
print("Number of stocks belonging to James Prentice: " + str(enron_data["Prentice James".upper()]["total_stock_value"]))
print("Number of stocks exercised by Jeffrey K Skilling: " + str(enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]))
print("Total payments to Jeffrey K Skilling: " + str(enron_data["SKILLING JEFFREY K"]["total_payments"]))
print("Total payments to Kenneth L Lay: " + str(enron_data["LAY KENNETH L"]["total_payments"]))
print("Total payments to Andrew S Fastow: " + str(enron_data["FASTOW ANDREW S"]["total_payments"]))

print("Percent of NaN for total payments: " + str((float(num_nan_total_payments)/len(enron_data))*100))
print("Percent of NaN POIs for total payments: " + str((float(num_nan_poi_total_payments)/num_poi)*100))
