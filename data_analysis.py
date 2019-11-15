import pandas as pd

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
    data_file = 'Data/Zillow_dataset_v1.0_HOA.csv'
    data = pd.read_csv(data_file)
    data.drop(['Zestimate'], inplace=True, axis=1)

    location(data)
