import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 500)

"""
input: datafile
output: dataset used for training and testing

"""

class housedata():
    def __init__(self,data=None,target=None):
        self.data = data
        self.target=target



#normalization
def range(data):
    min = data.min()
    max = data.max()
    for i, price in data.iteritems():
        data[i] = (data[i] - min) / (max-min)
    return data

def label(df_train,type):

    #price
    df_train['price'] = df_train['price'].replace('\D+', '', regex=True)
    df_train['price'] = pd.to_numeric(df_train['price']) # transfer the data type
    df_train = df_train.dropna(subset=["price"])
    df_train['price'].hist()
    df_train['price'] = range(df_train['price'])



    #bedrooms
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

    #Area
    df_train['Area'] = df_train['Area'].replace('\D+', '', regex=True)
    df_train['Area'] = df_train['Area'].replace('^No Data', '', regex=True)
    df_train['Area'] = pd.to_numeric(df_train['Area'])
    mean_bd = df_train['Area'].mean()
    df_train['Area'] = df_train['Area'].fillna(mean_bd)
    df_train['Area'] = range(df_train['Area'])


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

    if type== 1:
        #money per month
        df_train['HL'] = df_train['HL'].replace(r"^No Data", '', regex=True)
        df_train['HL'] = df_train['HL'].replace("[\D]+", '', regex=True)

        df_train['HL'] = pd.to_numeric(df_train['HL'])
        df_train['HL'][df_train['HL']>1000000]=None
        mean_bd = df_train['HL'].mean()
        df_train['HL'] = df_train['HL'].fillna(mean_bd)
        df_train['HL'] = range(df_train['HL'])
    if type == 0:
        #area
        df_train['HL'] = df_train['HL'].replace(',', '', regex=True)
        df_train['HL'] = df_train['HL'].replace('^No Data', '', regex=True)
        #获取lot数据列中单位为sqft的数据
        rows_with_sqft = df_train['HL'].str.contains('sqft').fillna(False)
        df_train[rows_with_sqft]
        #将sqft的数据转换为acres数据
        for i, sqft_row in df_train[rows_with_sqft].iterrows():
            area = str(float(sqft_row['HL'][:-5])/43560)
            df_train['HL'][i] = '{} acres'.format(area)
        #去除所有的非数字字符
        df_train['HL'] = df_train['HL'].replace("[\ssqft]+", '', regex=True)
        df_train['HL'] = df_train['HL'].replace("[\sacres]+", '', regex=True)
        df_train['HL'] = pd.to_numeric(df_train['HL'])
        mean_bd = df_train['HL'].mean()
        df_train['HL'] = df_train['HL'].fillna(mean_bd)
        df_train['HL'] = range(df_train['HL'])


    #monthly cost Monthcost
    df_train['Monthcost'] = df_train['Monthcost'].replace('[\D]+', '', regex=True)
    df_train['Monthcost'] = pd.to_numeric(df_train['Monthcost'])  # transfer the data type
    df_train['Monthcost'][df_train['Monthcost'] > 1000000] = None
    mean_bd = df_train['Monthcost'].mean()
    df_train['Monthcost'] = df_train['Monthcost'].fillna(mean_bd)
    df_train['Monthcost'] = range(df_train['Monthcost'])
    #
    # #Principal & interest
    df_train['PI'] = df_train['PI'].replace('[\D]+', '', regex=True)
    df_train['PI'] = pd.to_numeric(df_train['PI'])  # transfer the data type
    df_train['PI'][df_train['PI'] > 1000000] = None
    mean_bd = df_train['PI'].mean()
    df_train['PI'] = df_train['PI'].fillna(mean_bd)
    df_train['PI'] = range(df_train['PI'])
    #
    # #Property taxes
    df_train['PT'] = df_train['PT'].replace('[\D]+', '', regex=True)
    df_train['PT'] = pd.to_numeric(df_train['PT'])  # transfer the data type
    df_train['PT'][df_train['PT'] > 1000000] = None
    mean_bd = df_train['PT'].mean()
    df_train['PT'] = df_train['PT'].fillna(mean_bd)
    df_train['PT'] = range(df_train['PT'])
    #
    # #Home insurance
    df_train['HI'] = df_train['HI'].replace('[\D]+', '', regex=True)
    df_train['HI'] = pd.to_numeric(df_train['HI'])  # transfer the data type
    df_train['HI'][df_train['HI'] > 1000000] = None
    mean_bd = df_train['HI'].mean()
    df_train['HI'] = df_train['HI'].fillna(mean_bd)
    df_train['HI'] = range(df_train['HI'])
    #
    # #Home fee
    df_train['HF'] = df_train['HF'].replace('[\D]+', '', regex=True)
    df_train['HF'] = pd.to_numeric(df_train['HF'])  # transfer the data type
    df_train['HF'][df_train['HF'] > 1000000] = None
    mean_bd = df_train['HF'].mean()
    df_train['HF'] = df_train['HF'].fillna(mean_bd)
    df_train['HF'] = range(df_train['HF'])
    #
    # #Address
    df_train['Address'] = df_train['Address'].str.extract('GA (30\d+)')  # extract zip code from address
    df_train['Address'].replace(np.nan, 'No Data', inplace=True)
    df_train['Address'].value_counts()

    onehotdata = df_train[['Address','Type','Heating','Cooling']]
    df_train = df_train.join(pd.get_dummies(onehotdata))
    to_drop = ['Address','Type','Heating','Cooling']
    df_train.drop(to_drop, inplace=True, axis=1)

    if type == 1:
        df_train.to_csv('.\output_HOA.csv')
    if type == 0:
        df_train.to_csv('.\output_LOT.csv')



    target = df_train['Zestimate'].values
    df_train.drop(['Zestimate'],inplace=True,axis=1)
    data = df_train.values
    dataset = housedata(data,target)

    return dataset

def pre_data():
    if os.path.exists('.\output_HOA.csv'):
        data1 = pd.read_csv(".\output_HOA.csv")
        target = data1['Zestimate'].values
        data1.drop(['Zestimate'], inplace=True, axis=1)
        data = data1.values
        data_HOA = housedata(data,target)

    else:
        data1 = pd.read_csv("Zillow_dataset_v1.0_HOA.csv")
        data1 = data1.drop(columns=['Sunscore'])
        train1 = data1.fillna('No Data').replace('#NAME?', 'No Data').rename(
            columns={'Sale price': 'price', 'Title': 'Address', 'Year built': 'Year Built', 'HOA / Lot': 'HL',
                     'Monthly cost': 'Monthcost',
                     'Principal & interest': 'PI', 'Property taxes': 'PT', 'Home insurance': 'HI',
                     'HOA fee': 'HF'})  # drop NaN and other no data
        data_HOA = label(train1, 1)

    if os.path.exists('.\output_LOT.csv'):
        data2 = pd.read_csv(".\output_LOT.csv")
        target = data2['Zestimate'].values
        data2.drop(['Zestimate'], inplace=True, axis=1)
        data = data2.values
        data_LOT = housedata(data, target)

    else:
        data2 = pd.read_csv("Zillow_dataset_v1.0_Lot.csv")
        data2 = data2.drop(columns=['Sunscore'])
        train2 = data2.fillna('No Data').replace('#NAME?', 'No Data').rename(
            columns={'Sale price': 'price', 'Title': 'Address', 'Year built': 'Year Built', 'HOA / Lot': 'HL',
                     'Monthly cost': 'Monthcost',
                     'Principal & interest': 'PI', 'Property taxes': 'PT', 'Home insurance': 'HI',
                     'HOA fee': 'HF'})  # drop NaN and other no data
        data_LOT = label(train2, 0)

    return data_HOA,data_LOT





if __name__ == "__main__":


    if os.path.exists('.\output_HOA.csv'):
        data1 = pd.read_csv(".\output_HOA.csv")
        target = data1['Zestimate'].values
        data1.drop(['Zestimate'], inplace=True, axis=1)
        data = data1.values
        data_HOA = housedata(data,target)

    else:
        data1 = pd.read_csv("Zillow_dataset_v1.0_HOA.csv")
        data1 = data1.drop(columns=['Sunscore'])
        train1 = data1.fillna('No Data').replace('#NAME?', 'No Data').rename(
            columns={'Sale price': 'price', 'Title': 'Address', 'Year built': 'Year Built', 'HOA / Lot': 'HL',
                     'Monthly cost': 'Monthcost',
                     'Principal & interest': 'PI', 'Property taxes': 'PT', 'Home insurance': 'HI',
                     'HOA fee': 'HF'})  # drop NaN and other no data
        data_HOA = label(train1, 1)

    if os.path.exists('.\output_LOT.csv'):
        data2 = pd.read_csv(".\output_LOT.csv")
        target = data2['Zestimate'].values
        data2.drop(['Zestimate'], inplace=True, axis=1)
        data = data2.values
        data_LOT = housedata(data, target)

    else:
        data2 = pd.read_csv("Zillow_dataset_v1.0_Lot.csv")
        data2 = data2.drop(columns=['Sunscore'])
        train2 = data2.fillna('No Data').replace('#NAME?', 'No Data').rename(
            columns={'Sale price': 'price', 'Title': 'Address', 'Year built': 'Year Built', 'HOA / Lot': 'HL',
                     'Monthly cost': 'Monthcost',
                     'Principal & interest': 'PI', 'Property taxes': 'PT', 'Home insurance': 'HI',
                     'HOA fee': 'HF'})  # drop NaN and other no data
        data_LOT = label(train2, 0)
    print("data_HOA",data_HOA.data, data_HOA.target)
    print("data_LOT", data_LOT.data, data_LOT.target)
   