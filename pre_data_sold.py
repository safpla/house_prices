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
import json
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
        k = 8
        self.data = data
        self.target = target
        self.otherinfo = otherinfo
        X_train, X_test, y_train, y_test = train_test_split(self.data.values,
                                                            self.target,
                                                            test_size=0.2,
                                                            random_state=0)
        X_test1, X_test2, y_test1, y_test2 = train_test_split(X_test, y_test,
                                                              test_size=0.5,
                                                              random_state=0)
        self.test_features = X_test1
        self.test_targets = y_test1
        self.stacking_train_features = X_test2
        self.stacking_train_targets = y_test2
        self.train_features_all = X_train
        self.train_targets_all = y_train
        self.train_features = []
        self.train_targets = []
        for i in np.arange(k):
            self.train_features.append([])
            self.train_targets.append([])
        self.valid_features = []
        self.valid_targets = []
        self.cv_features = []
        self.cv_targets = []
        split_size = round(np.shape(y_train)[0] / k)
        for i in np.arange(k-1):
            self.cv_features.append(X_train[split_size*i : split_size*(i+1), :])
            self.cv_targets.append(y_train[split_size*i : split_size*(i+1)])
        self.cv_features.append(X_train[split_size*(k-1) : , :])
        self.cv_targets.append(y_train[split_size*(k-1) :])
        for i in np.arange(k):
            self.train_features
        for i in np.arange(k):
            for j in np.arange(k):
                if i == j:
                    self.valid_features.append(self.cv_features[i])
                    self.valid_targets.append(self.cv_targets[i])
                else:
                    self.train_features[j].append(self.cv_features[i])
                    self.train_targets[j].append(self.cv_targets[i])
        for i in np.arange(k):
            self.train_features[i] = np.concatenate(self.train_features[i])
            self.train_targets[i] = np.concatenate(self.train_targets[i])

def Geocode(data):
    #zip_geo = pd.read_csv(
    #        "D:\\SUNZHIMIN\Grad life\\Grad School\\CS 6220 Big data sys and analysis\\Project\\zipcode-geolocation\\US Zip Codes from 2013 Government Data.csv")
    zip_geo = pd.read_csv(os.path.join(root_path, 'Data/US Zip Codes from 2013 Government Data.csv'))

    zip_geo = DataFrame(zip_geo)
    data['lat'] = data['Address']
    data['lng'] = data['Address']
    data['lat'] = pd.to_numeric(data['lat'],downcast="float")  # transfer the data type\
    data['lng'] = pd.to_numeric(data['lng'],downcast="float")  # transfer the data type\
    for i, item in data.iterrows():
        if type(item['Address']) == type(1) and item['Address'] is not np.nan:
            zipcode = item['Address']
            index = (zip_geo['ZIP'][zip_geo['ZIP'] == int(zipcode)].index)
            if len(index) == 0:
                lat = 0
                lng = 0
            else:
                lat = zip_geo.at[int(index[0]),'LAT']
                lng = zip_geo.at[int(index[0]),'LNG']

            data['lat'][i] = float(lat)
            data['lng'][i] = float(lng)

    data['lat'] = data['lat'].replace( 'No Data', '0', regex=True)
    data['lng'] = data['lng'].replace('No Data', '0', regex=True)
    print(data['lat'])
    print(data['lat'])
    return data

def cluster(data,cluter_num):
    new_data = Geocode(data)
    geolocation = np.c_[new_data['lat'], new_data['lng']]
    kmeans = KMeans(n_clusters=cluter_num, random_state=0).fit(geolocation)
    new_data['label'] = kmeans.labels_
    return new_data

#normalization
# TODO(Haowen): try another normalization
#def range(data):
#    min = data.min()
#    max = data.max()
#    data = (data - min) / (max - min)
#    return data

def normal(data):
    data_mean = data.mean()
    data_std = data.std()
    if data_std != 0:
        data = (data - data_mean) / data_std
    else:
        data = (data - data_mean)
    return data


def reverse(data,mean,std):
    data = data * std + mean
    return data

def formhousedata(df_train):
    target = df_train['Soldprice'].values
    otherinfo = df_train[['Address', 'lat','lng', 'Zestimate']].values
    df_train.drop(['Zestimate', 'Address', 'lat','lng','Soldprice'], inplace=True,
                  axis=1)
    data = df_train
    dataset = Housedata(data, target, otherinfo)
    return dataset

def savejson(mean,std,min,max,outfile):
    file = open(outfile, 'w', encoding='utf-8')
    data = {'mean': mean, "std": std, "min":min, "max":max}
    print(data)
    json.dump(data, file, ensure_ascii=False)

def readjson(infile):
    file = open(infile,'r',encoding='utf-8')
    s = json.load(file)
    print (s['mean'],s['std'],s['min'],s['max'])

def label(df_train,type,outfile):
    cluter_num = 15

    #df_train = df_train.fillna('No Data').replace('#NAME?', 'No Data')
    df_train = df_train.rename(columns={'Sold price': 'Soldprice'})
    df_train.replace('No Data','')
    print(df_train.info())
    #Sold price
    df_train['Soldprice'] = df_train['Soldprice'].replace('\$', '', regex=True)
    df_train['Soldprice'] = df_train['Soldprice'].replace(',', '', regex=True)

    rows_with_M = df_train['Soldprice'].str.contains('M').fillna(False)
    df_train[rows_with_M]
    # 将sqft的数据转换为acres数据
    for i, M_row in df_train[rows_with_M].iterrows():
        price = str(float(M_row['Soldprice'][:-1]) * 1000000)
        df_train['Soldprice'][i] = price
    df_train['Soldprice'] = pd.to_numeric(df_train['Soldprice'])  # transfer the data type

    soldprice_mean = df_train['Soldprice'].mean()
    soldprice_std = df_train['Soldprice'].std()
    soldprice_min = df_train['Soldprice'].min()
    soldprice_max = df_train['Soldprice'].max()
    df_train['Soldprice'] = normal(df_train['Soldprice'])

    # drop outliers
    df_train.sort_values(by="Soldprice", ascending=False)
    df_train = df_train[df_train['Soldprice'] <= 0.2]

    df_train['Soldprice'] = reverse(df_train['Soldprice'], soldprice_mean, soldprice_std)

    soldprice_mean = df_train['Soldprice'].mean()
    soldprice_std = df_train['Soldprice'].std()
    soldprice_min = df_train['Soldprice'].min()
    soldprice_max = df_train['Soldprice'].max()
    if type == 1:
        if os.name == 'posix':
            jsonoutfile = os.path.join(root_path, 'Data/para_HOA.json')
        else:
            jsonoutfile = '.\Data\para_HOA.json'
        savejson(soldprice_mean, soldprice_std, soldprice_min, soldprice_max, jsonoutfile)
    else:
        if os.name == 'posix':
            jsonoutfile = os.path.join(root_path, 'Data/para_LOT.json')
        else:
            jsonoutfile = '.\Data\para_LOT.json'
        savejson(soldprice_mean, soldprice_std, soldprice_min, soldprice_max, jsonoutfile)

    # renormalization
    df_train['Soldprice'] = normal(df_train['Soldprice'])

    #bedrooms clear
    df_train['Bedrooms'] = df_train['Bedrooms'].replace('\D+', '', regex=True)
    df_train['Bedrooms'] = pd.to_numeric(df_train['Bedrooms'])
    mean_bd = df_train['Bedrooms'].mean()
    df_train['Bedrooms'] = df_train['Bedrooms'].fillna(mean_bd)
    df_train['Bedrooms'] = normal(df_train['Bedrooms'])

    #bathrooms clear
    df_train['Bathrooms'] = df_train['Bathrooms'].replace('\D+', '', regex=True)
    df_train['Bathrooms'] = pd.to_numeric(df_train['Bathrooms'])
    mean_bd = df_train['Bathrooms'].mean()
    df_train['Bathrooms'] = df_train['Bathrooms'].fillna(mean_bd)
    df_train['Bathrooms'] = normal(df_train['Bathrooms'])

    #Area  clear
    df_train['Area'] = df_train['Area'].replace('\D+', '', regex=True)
    df_train['Area'] = df_train['Area'].replace('^No Data', '', regex=True)
    df_train['Area'] = pd.to_numeric(df_train['Area'])
    mean_bd = df_train['Area'].mean()
    df_train['Area'] = df_train['Area'].fillna(mean_bd)
    df_train['Area'] = normal(df_train['Area'])


    #Year built clear
    df_train['Year built'] = df_train['Year built'].replace('\D+', '', regex=True)
    df_train['Year built'] = pd.to_numeric(df_train['Year built'])  # transfer the data type
    mean_bd = df_train['Year built'].mean()
    df_train['Year built'] = df_train['Year built'].fillna(mean_bd)
    df_train['Year built'] = normal(df_train['Year built'])

    #Last remodel year
    df_train = df_train.rename(columns={'Last remodel year':'LRY'})
    df_train['LRY'] = df_train['LRY'].replace('\D+', '', regex=True)
    df_train['LRY'] = pd.to_numeric(df_train['LRY'])  # transfer the data type
    mean_bd = df_train['LRY'].mean()
    df_train['LRY'] = df_train['LRY'].fillna(mean_bd)
    df_train['LRY'] = normal(df_train['LRY'])

    # parking - the number of parking spaces clear
    df_train['Parking'] = df_train['Parking'].replace('^Attached Garage', '3', regex=True)
    df_train['Parking'] = df_train['Parking'].replace('[\D]+', '', regex=True)
    #df_train['Parking'] = df_train['Parking'].replace('[\s]+', '1', regex=True)
    df_train['Parking'] = pd.to_numeric(df_train['Parking'])
    mean_bd = df_train['Parking'].mean()
    df_train['Parking'] = df_train['Parking'].fillna(mean_bd)
    df_train['Parking'] = normal(df_train['Parking'])


    if type== 1:
        # Lot money per month-HOA
        df_train['Lot'] = df_train['Lot'].replace(r"^No Data", '', regex=True)
        df_train['Lot'] = df_train['Lot'].replace('[\D]+', '', regex=True)
        df_train['Lot'] = pd.to_numeric(df_train['Lot'])  # transfer the data type
        df_train['Lot'][df_train['Lot'] > 1000000] = None
        mean_bd = df_train['Lot'].mean()
        df_train['Lot'] = df_train['Lot'].fillna(mean_bd)
        df_train['Lot'] = normal(df_train['Lot'])

    if type == 0:
        #area-Lot
        df_train['Lot'] = df_train['Lot'].replace(',', '', regex=True)
        df_train['Lot'] = df_train['Lot'].replace('^No Data', '', regex=True)
        #获取lot数据列中单位为sqft的数据
        rows_with_sqft = df_train['Lot'].str.contains('sqft').fillna(False)
        df_train[rows_with_sqft]
        #将sqft的数据转换为acres数据
        for i, sqft_row in df_train[rows_with_sqft].iterrows():
            area = '%.7f'%(float(sqft_row['Lot'][:-5])/43560)
            df_train['Lot'][i] = '{} acres'.format(area)
        #去除所有的非数字字符
        df_train['Lot'] = df_train['Lot'].replace("[\ssqft]+", '', regex=True)
        df_train['Lot'] = df_train['Lot'].replace("[\sacres]+", '', regex=True)
        df_train['Lot'] = pd.to_numeric(df_train['Lot'])
        mean_bd = df_train['Lot'].mean()
        df_train['Lot'] = df_train['Lot'].fillna(mean_bd)
        df_train['Lot'] = normal(df_train['Lot'])

    # zestimate
    df_train['Zestimate'] = df_train['Zestimate'].replace('[\D]+', '', regex=True)
    df_train['Zestimate'] = pd.to_numeric(df_train['Zestimate'])  # transfer the data type
    df_train = df_train.dropna(subset=["Zestimate"])
    df_train['Zestimate'] = normal(df_train['Zestimate'])

    # #Address
    df_train['Address'] = df_train['Address'].str.extract(r'.*(\d{5}(\-\d{4})?)$')  # extract zip code from address
    df_train['Address'] = pd.to_numeric(df_train['Address'])
    df_train['Address'] = df_train['Address'].replace(np.nan, 'No Data', regex=True)
    df_train = cluster(df_train,cluter_num)

    #average house price
    df_train['Avgprice'] = df_train['label']
    df_train['label'].astype(str)
    df_train['Avgprice'] = pd.to_numeric(df_train['Avgprice'],downcast="float")
    for i in range(cluter_num):
        labeldata = df_train[df_train['label']==i]
        Avgprice = labeldata['Soldprice'].mean()
        rows = df_train[df_train['label'] == i].index
        for j in rows:
            df_train['Avgprice'][j] = float(Avgprice)



    #df_train = df_train.join(pd.get_dummies(df_train['label'],prefix="label")) #one-hot label
    onehotdata = df_train[['Type','Heating','Cooling']]
    df_train = df_train.join(pd.get_dummies(onehotdata))

    # TODO(Haowen) drop price
    if type==0:
        to_drop = ['Unnamed: 18', 'Type','Heating','Cooling','label','Sunscore', 'Title_link', 'Sold date','Zestimate range','Last 30 day change']
    else:
        to_drop = ['Type','Heating','Cooling','label','Sunscore', 'Title_link', 'Sold date','Zestimate range','Last 30 day change']

    df_train.drop(to_drop, inplace=True, axis=1)

    #shuffle the data before saving to the file
    df_train = shuffle(df_train)

    df_train.to_csv(outfile,index = 0)

    dataset = formhousedata(df_train)

    return dataset

def pre_data(data_file=None, data_type=None, rebuild=False):
    if not data_file is None:
        output_file = '/'.join(data_file.split('/')[0:-1]) + '/output_{}_Sold.csv'.format(data_type)
        if os.path.exists(output_file) and not rebuild:
            data1 = pd.read_csv(output_file)
            data_ = formhousedata(data1)
            return data_
        data1 = pd.read_csv(data_file)
        train1 = data1.fillna('No Data').replace('#NAME?', 'No Data')  # drop NaN and other no data
        if data_type == 'HOA':
            data_HOA = label(train1, 1,  output_file)
            return data_HOA
        else:
            data_LOT = label(train1, 0,  output_file)
            return data_LOT

    if os.path.exists('..\Data\output_HOA_Sold.csv'):
        data1 = pd.read_csv("..\Data\output_HOA_Sold.csv")
        data_HOA = formhousedata(data1)

    else:
        data1 = pd.read_csv("..\Data\SoldData-HOA-V1.0.csv")
        train1 = data1.fillna('No Data').replace('#NAME?', 'No Data')
        data_HOA = label(train1, 1,  '..\Data\output_HOA_Sold.csv')

    if os.path.exists('..\Data\output_LOT_Sold.csv'):
        data2 = pd.read_csv("..\Data\output_LOT_Sold.csv")
        data_LOT = formhousedata(data2)

    else:
        data2 = pd.read_csv("..\Data\SoldData-Lot-V1.0.csv")
        train2 = data2.fillna('No Data').replace('#NAME?', 'No Data')
        data_LOT = label(train2,0, '..\Data\output_LOT_Sold.csv')

    return data_HOA,data_LOT


if __name__ == "__main__":
    if os.path.exists('.\Data\output_HOA_Sold.csv'):
        data1 = pd.read_csv(".\Data\output_HOA_Sold.csv")
        data_HOA = formhousedata(data1)
        print(data_HOA.data.info())
    else:
        print("else HOA")
        data1 = pd.read_csv(".\Data\SoldData-HOA-V1.0.csv")
        train1 = data1.fillna('No Data').replace('#NAME?', 'No Data')
        data_HOA = label(train1,1, '.\Data\output_HOA_Sold.csv')


    if os.path.exists('.\Data\output_LOT_Sold.csv'):
        data2 = pd.read_csv(".\Data\output_LOT_Sold.csv")
        data_LOT = formhousedata(data2)
        print(data_LOT.data.info())

    else:
        print("else LOT")
        data2 = pd.read_csv(".\Data\SoldData-Lot-V1.0.csv")
        train2 = data2.fillna('No Data').replace('#NAME?', 'No Data')
        data_LOT = label(train2, 0, '.\Data\output_LOT_Sold.csv')


    print("data_HOA",data_HOA.data.info(), data_HOA.target)
    print("data_LOT", data_LOT.data.info(), data_LOT.target)
    readjson(".\Data\para_HOA.json")
    readjson(".\Data\para_LOT.json")
