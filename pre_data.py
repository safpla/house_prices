import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 500)

"""
input: datafile
output: dataset used for training and testing

"""


#normalization
def range(data):
    min = data.min()
    max = data.max()
    for i, price in data.iteritems():
        data[i] = (data[i] - min) / (max-min)
    return data

def label(df_train):

    #price
    df_train['price'] = df_train['price'].replace('\D+', '', regex=True)
    df_train['price'] = pd.to_numeric(df_train['price']) # transfer the data type
    df_train = df_train.dropna(subset=["price"])
    df_train['price'] = range(df_train['price'])
    # sns.distplot(df_train['price'])
    # plt.show()

    #Area
    #df_train['Area'] = df_train['Area'].replace(',\s', '', regex=True)
    df_train['Area'] = df_train['Area'].replace('\D+', '', regex=True)
    df_train['Area'] = df_train['Area'].replace('^No Data', '', regex=True)
    df_train['Area'] = pd.to_numeric(df_train['Area'])
    mean_bd = df_train['Area'].mean()
    df_train['Area'] = df_train['Area'].fillna(mean_bd)
    df_train['Area'] = range(df_train['Area'])




    #bedrooms
    # nan, exceptional data(like xxxsqft, xxx acres, Studio)
    df_train['Bedrooms'] = df_train['Bedrooms'].replace('[bds|Studio]+', '', regex=True)
    df_train['Bedrooms'] = df_train['Bedrooms'].replace('[\d+\w\s\D]+[\w]+', '', regex=True)
    df_train['Bedrooms'] = df_train['Bedrooms'].replace('[\W]+', '', regex=True)
    df_train['Bedrooms'] = pd.to_numeric(df_train['Bedrooms'])
    mean_bd = df_train['Bedrooms'].mean()
    df_train['Bedrooms'] = df_train['Bedrooms'].fillna(mean_bd)
    df_train['Bedrooms'] = range(df_train['Bedrooms'])

    #bathrooms
    df_train['Bathrooms'] = df_train['Bathrooms'].replace('\D+', '', regex=True)
    df_train['Bathrooms'] = pd.to_numeric(df_train['Bathrooms'])
    mean_bd = df_train['Bathrooms'].mean()
    df_train['Bathrooms'] = df_train['Bathrooms'].fillna(mean_bd)
    df_train['Bathrooms'] = range(df_train['Bathrooms'])

    #zestimate
    df_train['Zestimate'] = df_train['Zestimate'].replace('[\D]+', '', regex=True)
    df_train['Zestimate'] = pd.to_numeric(df_train['Zestimate'])  # transfer the data type
    mean_bd = df_train['Zestimate'].mean()
    df_train['Zestimate'] = df_train['Zestimate'].fillna(mean_bd)
    df_train['Zestimate'] = range(df_train['Zestimate'])


    #Year built
    df_train['Year Built'] = df_train['Year Built'].replace('\D+', '', regex=True)
    df_train['Year Built'] = pd.to_numeric(df_train['Year Built'])  # transfer the data type
    mean_bd = df_train['Year Built'].mean()
    df_train['Year Built'] = df_train['Year Built'].fillna(mean_bd)
    df_train['Year Built'] = range(df_train['Year Built'])

    # parking - the number of parking spaces
    df_train['Parking'] = df_train['Parking'].replace('[\D]+', '', regex=True)
    #df_train['Parking'] = df_train['Parking'].replace('[\s]+', '1', regex=True)
    df_train['Parking'] = pd.to_numeric(df_train['Parking'])
    mean_bd = df_train['Parking'].mean()
    df_train['Parking'] = df_train['Parking'].fillna(mean_bd)
    df_train['Parking'] = range(df_train['Parking'])

    #parkinglot rent fee per month, especially for the apartment
    df_train['parkingrent'] = df_train['Lot']
    df_train['parkingrent'] = df_train['parkingrent'].replace(r"^\S+ sqft", '', regex=True)
    df_train['parkingrent'] = df_train['parkingrent'].replace(r"^\S+ acres", '', regex=True)
    df_train['parkingrent'] = df_train['parkingrent'].replace(r"^\S+ acre", '', regex=True)
    df_train['parkingrent'] = df_train['parkingrent'].replace(r"^No Data", '', regex=True)
    df_train['parkingrent'] = df_train['parkingrent'].replace("[\D]+", '', regex=True)
    df_train['parkingrent'] = pd.to_numeric(df_train['parkingrent'])
    mean_bd = df_train['parkingrent'].mean()
    df_train['parkingrent'] = df_train['parkingrent'].fillna(mean_bd)
    df_train['parkingrent'] = range(df_train['parkingrent'])

    #lot-the area of the parking lot, there are two kinds of unit
    df_train['Lot'] = df_train['Lot'].replace(',', '', regex=True)
    df_train['Lot'] = df_train['Lot'].replace('^\$\d+\Wmonth','',regex=True)
    df_train['Lot'] = df_train['Lot'].replace('^\$\d+', '', regex=True)
    df_train['Lot'] = df_train['Lot'].replace('^No Data', '', regex=True)
    #获取lot数据列中单位为acres的数据
    rows_with_sqft = df_train['Lot'].str.contains('sqft').fillna(False)
    df_train[rows_with_sqft]
    #将acres的数据转换为sqft数据
    for i, sqft_row in df_train[rows_with_sqft].iterrows():
        area = str(float(sqft_row['Lot'][:-5])/43560)
        df_train['Lot'][i] = '{} sqft'.format(area)
    #去除所有的非数字字符
    df_train['Lot'] = df_train['Lot'].replace("[\ssqft]+", '', regex=True)
    df_train['Lot'] = df_train['Lot'].replace("[\sacres]+", '', regex=True)
    df_train['Lot'] = pd.to_numeric(df_train['Lot'])
    mean_bd = df_train['Lot'].mean()
    df_train['Lot'] = df_train['Lot'].fillna(mean_bd)
    df_train['Lot'] = range(df_train['Lot'])

    #Address
    df_train['Address'] = df_train['Address'].str.extract('GA (30\d+)')  # extract zip code from address
    df_train['Address'].replace(np.nan, 'No Data', inplace=True)
    df_train['Address'].value_counts()

    onehotdata = df_train[['Address','list-card-type','Type','Heating','Cooling']]
    df_train = df_train.join(pd.get_dummies(onehotdata))
    to_drop = ['Address','list-card-type','Type','Heating','Cooling']
    df_train.drop(to_drop, inplace=True, axis=1)
    df_train.to_csv('.\output.csv')
    print(df_train.info())

    data = {}
    data['target'] = df_train['price'].values
    df_train.drop(['price'],inplace=True,axis=1)
    data['data'] = df_train.values

    return data

def pre_data(datafile=None):
    if datafile is None:
        datafile = "train100.csv"
    df_train = pd.read_csv(datafile)
    df_train = df_train.drop(columns=['Title_link', 'list-card-variable-text', 'Thumbnail'])
    df_train = df_train.fillna('No Data').replace('#NAME?', 'No Data').rename(
        columns={'Heaitng': 'Heating', 'Title': 'Address'})  # drop NaN and other no data
    data = label(df_train)
    return data





if __name__ == "__main__":
    df_train = pd.read_csv("train100.csv")
    df_train = df_train.drop(columns=[ 'Title_link', 'list-card-variable-text','Thumbnail'])
    df_train = df_train.fillna('No Data').replace('#NAME?', 'No Data').rename(
        columns={'Heaitng': 'Heating', 'Title': 'Address'})  # drop NaN and other no data
    data = label(df_train)
