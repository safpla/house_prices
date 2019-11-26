import pandas as pd
import numpy as np

def location(data):
    print(len(data))
    for i in range(333):
        for j in range(len(data) - 1):
            if i != j and \
                    data['Bedrooms'][i] == data['Bedrooms'][j] and \
                    data['Bathrooms'][i] == data['Bathrooms'][j] and \
                    data['Heating'][i] == data['Heating'][j] and \
                    abs(int(data['Year built'][i]) - int(data['Year built'][j])) < 5:
                print(data.iloc[i,:])
                print(data.iloc[j, :])
                print(int(data['Year built'][i]))

if __name__ == "__main__":
    #data_file = 'Data/Zillow_dataset_v1.0_HOA.csv'
    #data = pd.read_csv(data_file)
    #data.drop(['Zestimate'], inplace=True, axis=1)

    #location(data)
    a = np.array([
        [27327.880376344085, 1408899230.117965, 0.03175360248241531, 0.6666666666666666],
        [24690.073588709667, 1269112194.3558993, 0.02684142101566086, 0.7311827956989247],
        [22535.37130376344, 1097748475.9367757, 0.028583634027721832, 0.7419354838709677],
        [25451.096438172044, 1632820944.9479067, 0.02475269042733604, 0.7526881720430108],
        [24835.122983870966, 1280679428.6747308, 0.02698794265316089, 0.6881720430107527],
        [21491.04233870969, 1178777063.5070987, 0.01721840276288129, 0.7741935483870968],
        [22737.229838709678, 1232944168.2852824, 0.022045192386916043, 0.7956989247311828],
        [23726.779905913983, 1126619710.3452518, 0.029908822011005917, 0.6881720430107527]])
    print(np.mean(a, axis=0))
