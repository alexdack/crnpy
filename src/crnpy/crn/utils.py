import numpy as np
import csv

def open_csv(file):
    # Function to open .csv files and extract them for plotting
    out_data = []
    with open(file, newline='') as csvfile:
        data = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in data:
            out_data.append([float(datapoint) for datapoint in row[0].split(',')])
    out_data = np.asarray(out_data) 
    return out_data