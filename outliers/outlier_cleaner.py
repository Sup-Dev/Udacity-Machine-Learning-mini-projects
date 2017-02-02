#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    number_of_points = len(predictions)/10

    for i in range(len(predictions)):
        cleaned_data.append((ages[i], net_worths[i], abs(predictions[i] - net_worths[i])))

    cleaned_data.sort(key=lambda t: t[2], reverse=True)
    cleaned_data = cleaned_data[number_of_points:]

    return cleaned_data

