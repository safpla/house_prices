import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os, sys
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import warnings
from pandas import DataFrame
from sklearn.utils import shuffle
from sklearn.preprocessing import  OneHotEncoder
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 500)
root_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, root_path)

"""
input: datafile
output: dataset used for training and testing

"""

class Housedata():
    def __init__(self,data=None,target=None,otherinfo=None):
        k = 9
        self.data = data
        self.target = target
        self.otherinfo = otherinfo
        X_train, X_test, y_train, y_test = train_test_split(self.data.values,
                                                            self.target,
                                                            test_size=0.1,
                                                            random_state=0)
        self.test_features = X_test
        self.test_targets = y_test
        self.train_features = []
        self.train_targets = []
        for i in np.arange(k):
            self.train_features.append([])
            self.train_targets.append([])
        self.valid_features = []
        self.valid_targets = []
        cv_features = []
        cv_targets = []
        split_size = round(np.shape(y_train)[0] / k)
        for i in np.arange(k-1):
            cv_features.append(X_train[split_size*i : split_size*(i+1), :])
            cv_targets.append(y_train[split_size*i : split_size*(i+1)])
        cv_features.append(X_train[split_size*(k-1) : , :])
        cv_targets.append(y_train[split_size*(k-1) :])
        for i in np.arange(k):
            self.train_features
        for i in np.arange(k):
            for j in np.arange(k):
                if i == j:
                    self.valid_features.append(cv_features[i])
                    self.valid_targets.append(cv_targets[i])
                else:
                    self.train_features[j].append(cv_features[i])
                    self.train_targets[j].append(cv_targets[i])
        for i in np.arange(k):
            self.train_features[i] = np.concatenate(self.train_features[i])
            self.train_targets[i] = np.concatenate(self.train_targets[i])



def Geocode(data):
    zip_geo = pd.read_csv(
            "D:\\SUNZHIMIN\Grad life\\Grad School\\CS 6220 Big data sys and analysis\\Project\\zipcode-geolocation\\US Zip Codes from 2013 Government Data.csv")

    zip_geo = DataFrame(zip_geo)
    data['lat'] = data['Address']
    data['lng'] = data['Address']
    data['geo'] = data['Address']
    for i, item in data.iterrows():
        if type(item['Address']) == type(1.0) and item['Address'] is not np.nan:
            zipcode = item['Address']
            index = (zip_geo['ZIP'][zip_geo['ZIP'] == int(zipcode)].index)
            lat = zip_geo.at[int(index[0]),'LAT']
            lng = zip_geo.at[int(index[0]),'LNG']

            data['lat'][i] = lat
            data['lng'][i] = lng
            data['geo'][i] = lat,lng

    data['lat'] = data['lat'].replace( 'No Data', '0', regex=True)
    data['lng'] = data['lng'].replace('No Data', '0', regex=True)

    return data

def cluster(data):
    new_data = Geocode(data)
    geolocation = np.c_[new_data['lat'], new_data['lng']]
    print(geolocation)
    kmeans = KMeans(n_clusters=5, random_state=0).fit(geolocation)
    print(type(kmeans.labels_))
    print((new_data['geo']))
    new_data['label'] = kmeans.labels_
    return new_data

#normalization
def range(data):
    min = data.min()
    max = data.max()
    data = (data - min) / (max - min)
    return data

def reverse(data,min,max):
    data = data * (max - min) + min
    return data



def label(df_train,type,outfile):
    # zestimate
    df_train['Zestimate'] = df_train['Zestimate'].replace('[\D]+', '', regex=True)
    df_train['Zestimate'] = pd.to_numeric(df_train['Zestimate'])  # transfer the data type
    df_train = df_train.dropna(subset=["Zestimate"])
    zestimate_max = df_train['Zestimate'].max()
    zestimate_min = df_train['Zestimate'].min()
    df_train['Zestimate'] = range(df_train['Zestimate'])
    # drop outliers
    df_train.sort_values(by="Zestimate", ascending=False)
    df_train = df_train[df_train['Zestimate'] <= 0.2]
    df_train['Zestimate'] = reverse(df_train['Zestimate'],zestimate_min,zestimate_max)
    #renormalization
    df_train['Zestimate'] = range(df_train['Zestimate'])
    print(np.shape(df_train))

    #price
    df_train['price'] = df_train['price'].replace('\D+', '', regex=True)
    df_train['price'] = pd.to_numeric(df_train['price']) # transfer the data type

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

    # #Address
    df_train['Address'] = df_train['Address'].str.extract(r'.*(\d{5}(\-\d{4})?)$')  # extract zip code from address
    df_train['Address'] = pd.to_numeric(df_train['Address'], downcast='integer')
    df_train['Address'] = df_train['Address'].replace(np.nan, 'No Data', regex=True)
    print(np.shape(df_train))
    df_train = cluster(df_train)

    #df_train['label'] = pd.to_numeric(df_train['label'], downcast='integer')
    print(pd.get_dummies(df_train['label']))
    df_train = df_train.join(pd.get_dummies(df_train['label'],prefix="label"))

    print(df_train.info())
    onehotdata = df_train[['Type','Heating','Cooling']]
    df_train = df_train.join(pd.get_dummies(onehotdata))

    to_drop = ['Type','Heating','Cooling','lat', 'lng','label']
    df_train.drop(to_drop, inplace=True, axis=1)

    #shuffle the data before saving to the file
    df_train = shuffle(df_train)

    df_train.to_csv(outfile,index = 0)

    target = df_train['Zestimate'].values
    otherinfo = df_train[['Address','geo']].values
    df_train.drop(['Zestimate','Address', 'geo'],inplace=True,axis=1)
    data = df_train

    dataset = Housedata(data,target,otherinfo)

    return dataset

def pre_data(data_file=None, data_type=None):
    if not data_file is None:
        output_file = '/'.join(data_file.split('/')[0:-1]) + '/output_{}.csv'.format(data_type)
        if os.path.exists(output_file):
            data1 = pd.read_csv(output_file)
            target = data1['Zestimate'].values
            otherinfo = data1['Address','geo'].values
            data1.drop(['Zestimate','Address', 'geo'], inplace=True, axis=1)
            data = data1
            data_ = Housedata(data, target,otherinfo)
            return data_
        data1 = pd.read_csv(data_file)
        data1 = data1.drop(columns=['Sunscore'])
        train1 = data1.fillna('No Data').replace('#NAME?', 'No Data').rename(
            columns={'Sale price': 'price', 'Title': 'Address', 'Year built': 'Year Built', 'HOA / Lot': 'HL',
                     'Monthly cost': 'Monthcost',
                     'Principal & interest': 'PI', 'Property taxes': 'PT', 'Home insurance': 'HI',
                     'HOA fee': 'HF'})  # drop NaN and other no data
        if data_type == 'HOA':
            data_HOA = label(train1, 1,  output_file)
            return data_HOA
        else:
            data_LOT = label(train1, 0,  output_file)
            return data_LOT

    if os.path.exists('..\Data\output_HOA.csv'):
        data1 = pd.read_csv("..\Data\output_HOA.csv")
        target = data1['Zestimate'].values
        otherinfo = data1[['Address', 'geo']].values
        data1.drop(['Zestimate','Address', 'geo'], inplace=True, axis=1)
        data = data1
        data_HOA = Housedata(data,target,otherinfo)

    else:
        data1 = pd.read_csv("..\Data\Zillow_dataset_v1.0_HOA.csv")
        data1 = data1.drop(columns=['Sunscore'])
        train1 = data1.fillna('No Data').replace('#NAME?', 'No Data').rename(
            columns={'Sale price': 'price', 'Title': 'Address', 'Year built': 'Year Built', 'HOA / Lot': 'HL',
                     'Monthly cost': 'Monthcost',
                     'Principal & interest': 'PI', 'Property taxes': 'PT', 'Home insurance': 'HI',
                     'HOA fee': 'HF'})  # drop NaN and other no data
        data_HOA = label(train1, 1,  '..\Data\output_HOA.csv')

    if os.path.exists('..\Data\output_LOT.csv'):
        data2 = pd.read_csv("..\Data\output_LOT.csv")
        target = data2['Zestimate'].values
        otherinfo = data2[['Address', 'geo']].values
        data2.drop(['Zestimate','Address', 'geo'], inplace=True, axis=1)
        data = data2
        data_LOT = Housedata(data, target,otherinfo)

    else:
        data2 = pd.read_csv("..\Data\Zillow_dataset_v1.0_Lot.csv")
        data2 = data2.drop(columns=['Sunscore'])
        train2 = data2.fillna('No Data').replace('#NAME?', 'No Data').rename(
            columns={'Sale price': 'price', 'Title': 'Address', 'Year built': 'Year Built', 'HOA / Lot': 'HL',
                     'Monthly cost': 'Monthcost',
                     'Principal & interest': 'PI', 'Property taxes': 'PT', 'Home insurance': 'HI',
                     'HOA fee': 'HF'})  # drop NaN and other no data
        data_LOT = label(train2,0, '..\Data\output_LOT.csv')

    return data_HOA,data_LOT


if __name__ == "__main__":
    if os.path.exists('.\Data\output_HOA.csv'):
        data1 = pd.read_csv(".\Data\output_HOA.csv")
        target = data1['Zestimate'].values
        otherinfo = data1[['Address', 'geo']].values
        data1.drop(['Zestimate','Address', 'geo'], inplace=True, axis=1)
        data = data1
        data_HOA = Housedata(data,target,otherinfo)
    else:
        data1 = pd.read_csv(".\Data\Zillow_dataset_v1.0_HOA.csv")
        print("data1")
        data1 = data1.drop(columns=['Sunscore'])
        train1 = data1.fillna('No Data').replace('#NAME?', 'No Data').rename(
            columns={'Sale price': 'price', 'Title': 'Address', 'Year built': 'Year Built', 'HOA / Lot': 'HL',
                     'Monthly cost': 'Monthcost',
                     'Principal & interest': 'PI', 'Property taxes': 'PT', 'Home insurance': 'HI',
                     'HOA fee': 'HF'})  # drop NaN and other no data
        data_HOA = label(train1,1, '.\Data\output_HOA.csv')

    if os.path.exists('.\Data\output_LOT.csv'):
        data2 = pd.read_csv(".\Data\output_LOT.csv")
        target = data2['Zestimate'].values
        otherinfo = data1[['Address', 'geo']].values
        data2.drop(['Zestimate','Address', 'geo'], inplace=True, axis=1)
        data = data2
        data_LOT = Housedata(data, target,otherinfo)


    else:
        data2 = pd.read_csv(".\Data\Zillow_dataset_v1.0_Lot.csv")
        data2 = data2.drop(columns=['Sunscore'])
        train2 = data2.fillna('No Data').replace('#NAME?', 'No Data').rename(
            columns={'Sale price': 'price', 'Title': 'Address', 'Year built': 'Year Built', 'HOA / Lot': 'HL',
                     'Monthly cost': 'Monthcost',
                     'Principal & interest': 'PI', 'Property taxes': 'PT', 'Home insurance': 'HI',
                     'HOA fee': 'HF'})  # drop NaN and other no data
        data_LOT = label(train2, 0, '.\Data\output_LOT.csv')

    print("data_HOA",data_HOA.data, data_HOA.target)
    print("data_LOT", data_LOT.data, data_LOT.target)
