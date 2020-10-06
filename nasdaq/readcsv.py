import csv
import numpy as np
import matplotlib.pyplot as plt

with open('predict_data_train.csv','r') as f:
    reader = csv.reader(f)
    data = list(reader)
    data = data[1:][0]
    temp = []
    for num in data:
         temp.append(float(num))
    print(temp)
    plt.plot(temp,label = 'predict-train')
    plt.show()
