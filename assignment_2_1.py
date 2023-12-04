import pandas as pd
from pandas import Series, DataFrame

#데이터 불러오기
data = pd.read_csv('2019_kbo_for_kaggle_v2.csv')

#1번
data_2015 = data[data['year'] == 2015]
data_2016 = data[data['year'] == 2016]
data_2017 = data[data['year'] == 2017]
data_2018 = data[data['year'] == 2018]

array = [data_2015,data_2016,data_2017,data_2018]

for i in range(4):
    print(array[i].sort_values(by='H', ascending=False)[['batter_name', 'H', 'year']].head(10))
    print(array[i].sort_values(by='avg', ascending=False)[['batter_name', 'avg', 'year']].head(10))
    print(array[i].sort_values(by='HR', ascending=False)[['batter_name', 'HR', 'year']].head(10))
    print(array[i].sort_values(by='OBP', ascending=False)[['batter_name', 'OBP', 'year']].head(10))

print('-'*100)

#2번
cp_list = ["포수","1루수","2루수","3루수","유격수","좌익수","중견수","우익수"]

for i in range(8):
    data_by_cp = data[data['cp'] == cp_list[i]]
    print(data_by_cp.sort_values(by='war', ascending=False)[['batter_name','war','cp','year']].head(1))

print('-'*100)

#3번
col_list = ["R","H","HR","RBI","SB","war","avg","OBP","SLG"]
corr_list = [0,0,0,0,0,0,0,0,0]

for i in range(9):
    temp_corr = data['salary'].corr(data[col_list[i]])
    print("Correlation", col_list[i] ,"with Salary is", temp_corr)
    corr_list[i] = temp_corr

print()
max_corr = 0
max_idx = 0
for i in range(9):
    if max_corr < corr_list[i]:
        max_corr = corr_list[i]
        max_idx = i

print("The Highest Correlation with salary is",col_list[max_idx],":",max_corr)