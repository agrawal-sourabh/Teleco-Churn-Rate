

```python
import numpy as np 
import pandas as pd

#visualization
import seaborn as sns
import matplotlib.pyplot as plt
```


```python
df = pd.read_csv('data.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customerID</th>
      <th>gender</th>
      <th>SeniorCitizen</th>
      <th>Partner</th>
      <th>Dependents</th>
      <th>tenure</th>
      <th>PhoneService</th>
      <th>MultipleLines</th>
      <th>InternetService</th>
      <th>OnlineSecurity</th>
      <th>...</th>
      <th>DeviceProtection</th>
      <th>TechSupport</th>
      <th>StreamingTV</th>
      <th>StreamingMovies</th>
      <th>Contract</th>
      <th>PaperlessBilling</th>
      <th>PaymentMethod</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7590-VHVEG</td>
      <td>Female</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
      <td>1</td>
      <td>No</td>
      <td>No phone service</td>
      <td>DSL</td>
      <td>No</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>29.85</td>
      <td>29.85</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5575-GNVDE</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>34</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>...</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>One year</td>
      <td>No</td>
      <td>Mailed check</td>
      <td>56.95</td>
      <td>1889.5</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3668-QPYBK</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>2</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Mailed check</td>
      <td>53.85</td>
      <td>108.15</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7795-CFOCW</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>45</td>
      <td>No</td>
      <td>No phone service</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>...</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>One year</td>
      <td>No</td>
      <td>Bank transfer (automatic)</td>
      <td>42.30</td>
      <td>1840.75</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9237-HQITU</td>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>2</td>
      <td>Yes</td>
      <td>No</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>70.70</td>
      <td>151.65</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
df.dtypes
```




    customerID           object
    gender               object
    SeniorCitizen         int64
    Partner              object
    Dependents           object
    tenure                int64
    PhoneService         object
    MultipleLines        object
    InternetService      object
    OnlineSecurity       object
    OnlineBackup         object
    DeviceProtection     object
    TechSupport          object
    StreamingTV          object
    StreamingMovies      object
    Contract             object
    PaperlessBilling     object
    PaymentMethod        object
    MonthlyCharges      float64
    TotalCharges         object
    Churn                object
    dtype: object




```python
for item in df.columns:
    print(item)
    print (df[item].unique())
```

    customerID
    ['7590-VHVEG' '5575-GNVDE' '3668-QPYBK' ... '4801-JZAZL' '8361-LTMKD'
     '3186-AJIEK']
    gender
    ['Female' 'Male']
    SeniorCitizen
    [0 1]
    Partner
    ['Yes' 'No']
    Dependents
    ['No' 'Yes']
    tenure
    [ 1 34  2 45  8 22 10 28 62 13 16 58 49 25 69 52 71 21 12 30 47 72 17 27
      5 46 11 70 63 43 15 60 18 66  9  3 31 50 64 56  7 42 35 48 29 65 38 68
     32 55 37 36 41  6  4 33 67 23 57 61 14 20 53 40 59 24 44 19 54 51 26  0
     39]
    PhoneService
    ['No' 'Yes']
    MultipleLines
    ['No phone service' 'No' 'Yes']
    InternetService
    ['DSL' 'Fiber optic' 'No']
    OnlineSecurity
    ['No' 'Yes' 'No internet service']
    OnlineBackup
    ['Yes' 'No' 'No internet service']
    DeviceProtection
    ['No' 'Yes' 'No internet service']
    TechSupport
    ['No' 'Yes' 'No internet service']
    StreamingTV
    ['No' 'Yes' 'No internet service']
    StreamingMovies
    ['No' 'Yes' 'No internet service']
    Contract
    ['Month-to-month' 'One year' 'Two year']
    PaperlessBilling
    ['Yes' 'No']
    PaymentMethod
    ['Electronic check' 'Mailed check' 'Bank transfer (automatic)'
     'Credit card (automatic)']
    MonthlyCharges
    [29.85 56.95 53.85 ... 63.1  44.2  78.7 ]
    TotalCharges
    ['29.85' '1889.5' '108.15' ... '346.45' '306.6' '6844.5']
    Churn
    ['No' 'Yes']



```python
for item in df.columns:
    try:
        df[item] = df[item].str.lower()
    except:
        print(item, "couldn't convert")
df.head()
```

    SeniorCitizen couldn't convert
    tenure couldn't convert
    MonthlyCharges couldn't convert





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customerID</th>
      <th>gender</th>
      <th>SeniorCitizen</th>
      <th>Partner</th>
      <th>Dependents</th>
      <th>tenure</th>
      <th>PhoneService</th>
      <th>MultipleLines</th>
      <th>InternetService</th>
      <th>OnlineSecurity</th>
      <th>...</th>
      <th>DeviceProtection</th>
      <th>TechSupport</th>
      <th>StreamingTV</th>
      <th>StreamingMovies</th>
      <th>Contract</th>
      <th>PaperlessBilling</th>
      <th>PaymentMethod</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7590-vhveg</td>
      <td>female</td>
      <td>0</td>
      <td>yes</td>
      <td>no</td>
      <td>1</td>
      <td>no</td>
      <td>no phone service</td>
      <td>dsl</td>
      <td>no</td>
      <td>...</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>month-to-month</td>
      <td>yes</td>
      <td>electronic check</td>
      <td>29.85</td>
      <td>29.85</td>
      <td>no</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5575-gnvde</td>
      <td>male</td>
      <td>0</td>
      <td>no</td>
      <td>no</td>
      <td>34</td>
      <td>yes</td>
      <td>no</td>
      <td>dsl</td>
      <td>yes</td>
      <td>...</td>
      <td>yes</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>one year</td>
      <td>no</td>
      <td>mailed check</td>
      <td>56.95</td>
      <td>1889.5</td>
      <td>no</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3668-qpybk</td>
      <td>male</td>
      <td>0</td>
      <td>no</td>
      <td>no</td>
      <td>2</td>
      <td>yes</td>
      <td>no</td>
      <td>dsl</td>
      <td>yes</td>
      <td>...</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>month-to-month</td>
      <td>yes</td>
      <td>mailed check</td>
      <td>53.85</td>
      <td>108.15</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7795-cfocw</td>
      <td>male</td>
      <td>0</td>
      <td>no</td>
      <td>no</td>
      <td>45</td>
      <td>no</td>
      <td>no phone service</td>
      <td>dsl</td>
      <td>yes</td>
      <td>...</td>
      <td>yes</td>
      <td>yes</td>
      <td>no</td>
      <td>no</td>
      <td>one year</td>
      <td>no</td>
      <td>bank transfer (automatic)</td>
      <td>42.30</td>
      <td>1840.75</td>
      <td>no</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9237-hqitu</td>
      <td>female</td>
      <td>0</td>
      <td>no</td>
      <td>no</td>
      <td>2</td>
      <td>yes</td>
      <td>no</td>
      <td>fiber optic</td>
      <td>no</td>
      <td>...</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>month-to-month</td>
      <td>yes</td>
      <td>electronic check</td>
      <td>70.70</td>
      <td>151.65</td>
      <td>yes</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
columns_to_convert = ['Partner', 
                      'Dependents', 
                      'PhoneService', 
                      'PaperlessBilling', 
                      'Churn']

for item in columns_to_convert:
    df[item].replace(to_replace='yes', value=1, inplace=True)
    df[item].replace(to_replace='no',  value=0, inplace=True)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customerID</th>
      <th>gender</th>
      <th>SeniorCitizen</th>
      <th>Partner</th>
      <th>Dependents</th>
      <th>tenure</th>
      <th>PhoneService</th>
      <th>MultipleLines</th>
      <th>InternetService</th>
      <th>OnlineSecurity</th>
      <th>...</th>
      <th>DeviceProtection</th>
      <th>TechSupport</th>
      <th>StreamingTV</th>
      <th>StreamingMovies</th>
      <th>Contract</th>
      <th>PaperlessBilling</th>
      <th>PaymentMethod</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7590-vhveg</td>
      <td>female</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>no phone service</td>
      <td>dsl</td>
      <td>no</td>
      <td>...</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>month-to-month</td>
      <td>1</td>
      <td>electronic check</td>
      <td>29.85</td>
      <td>29.85</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5575-gnvde</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>34</td>
      <td>1</td>
      <td>no</td>
      <td>dsl</td>
      <td>yes</td>
      <td>...</td>
      <td>yes</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>one year</td>
      <td>0</td>
      <td>mailed check</td>
      <td>56.95</td>
      <td>1889.5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3668-qpybk</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>no</td>
      <td>dsl</td>
      <td>yes</td>
      <td>...</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>month-to-month</td>
      <td>1</td>
      <td>mailed check</td>
      <td>53.85</td>
      <td>108.15</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7795-cfocw</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>45</td>
      <td>0</td>
      <td>no phone service</td>
      <td>dsl</td>
      <td>yes</td>
      <td>...</td>
      <td>yes</td>
      <td>yes</td>
      <td>no</td>
      <td>no</td>
      <td>one year</td>
      <td>0</td>
      <td>bank transfer (automatic)</td>
      <td>42.30</td>
      <td>1840.75</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9237-hqitu</td>
      <td>female</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>no</td>
      <td>fiber optic</td>
      <td>no</td>
      <td>...</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>month-to-month</td>
      <td>1</td>
      <td>electronic check</td>
      <td>70.70</td>
      <td>151.65</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
df.dtypes
```




    customerID           object
    gender               object
    SeniorCitizen         int64
    Partner               int64
    Dependents            int64
    tenure                int64
    PhoneService          int64
    MultipleLines        object
    InternetService      object
    OnlineSecurity       object
    OnlineBackup         object
    DeviceProtection     object
    TechSupport          object
    StreamingTV          object
    StreamingMovies      object
    Contract             object
    PaperlessBilling      int64
    PaymentMethod        object
    MonthlyCharges      float64
    TotalCharges         object
    Churn                 int64
    dtype: object




```python
# ax = sns.countplot(x="Churn", data=df, orient='v')
print("Customer Churn:",(df['Churn'] == 1).sum())
sns.barplot(x="Churn", data=df)
```

    Customer Churn: 1869





    <matplotlib.axes._subplots.AxesSubplot at 0x10c4a0630>




![png](output_7_2.png)



```python
#replace if total charges has any spaces and convert it to numeric
df['TotalCharges'] = df['TotalCharges'].replace(r'\s+', np.nan, regex=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
```


```python
#get total number of null in each columns 
df.isnull().sum(axis = 0)
```




    customerID           0
    gender               0
    SeniorCitizen        0
    Partner              0
    Dependents           0
    tenure               0
    PhoneService         0
    MultipleLines        0
    InternetService      0
    OnlineSecurity       0
    OnlineBackup         0
    DeviceProtection     0
    TechSupport          0
    StreamingTV          0
    StreamingMovies      0
    Contract             0
    PaperlessBilling     0
    PaymentMethod        0
    MonthlyCharges       0
    TotalCharges        11
    Churn                0
    dtype: int64




```python
#replacing null with zeros
df = df.fillna(value=0)
```


```python
churners_count = len(df[df['Churn'] == 1])
churners = (df[df['Churn'] == 1])
non_churners = df[df['Churn'] == 0].sample(n=churners_count)
df2 = churners.append(non_churners)
```


```python
def show_correlations(dataframe, show_chart = True):
    fig = plt.figure(figsize = (20,10))
    corr = dataframe.corr()
    if show_chart == True:
        sns.heatmap(corr, 
                    xticklabels=corr.columns.values,
                    yticklabels=corr.columns.values,
                    annot=True)
    return corr

show_correlations(df2,show_chart=True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SeniorCitizen</th>
      <th>Partner</th>
      <th>Dependents</th>
      <th>tenure</th>
      <th>PhoneService</th>
      <th>PaperlessBilling</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>SeniorCitizen</th>
      <td>1.000000</td>
      <td>0.031414</td>
      <td>-0.206534</td>
      <td>0.009267</td>
      <td>0.004877</td>
      <td>0.152010</td>
      <td>0.202857</td>
      <td>0.083953</td>
      <td>0.158910</td>
    </tr>
    <tr>
      <th>Partner</th>
      <td>0.031414</td>
      <td>1.000000</td>
      <td>0.432514</td>
      <td>0.388178</td>
      <td>0.002734</td>
      <td>-0.007327</td>
      <td>0.104198</td>
      <td>0.339102</td>
      <td>-0.160169</td>
    </tr>
    <tr>
      <th>Dependents</th>
      <td>-0.206534</td>
      <td>0.432514</td>
      <td>1.000000</td>
      <td>0.170856</td>
      <td>-0.008911</td>
      <td>-0.100993</td>
      <td>-0.110354</td>
      <td>0.079321</td>
      <td>-0.167564</td>
    </tr>
    <tr>
      <th>tenure</th>
      <td>0.009267</td>
      <td>0.388178</td>
      <td>0.170856</td>
      <td>1.000000</td>
      <td>0.026022</td>
      <td>-0.030844</td>
      <td>0.226365</td>
      <td>0.854732</td>
      <td>-0.414469</td>
    </tr>
    <tr>
      <th>PhoneService</th>
      <td>0.004877</td>
      <td>0.002734</td>
      <td>-0.008911</td>
      <td>0.026022</td>
      <td>1.000000</td>
      <td>0.015908</td>
      <td>0.300104</td>
      <td>0.122363</td>
      <td>0.003702</td>
    </tr>
    <tr>
      <th>PaperlessBilling</th>
      <td>0.152010</td>
      <td>-0.007327</td>
      <td>-0.100993</td>
      <td>-0.030844</td>
      <td>0.015908</td>
      <td>1.000000</td>
      <td>0.335932</td>
      <td>0.108964</td>
      <td>0.221131</td>
    </tr>
    <tr>
      <th>MonthlyCharges</th>
      <td>0.202857</td>
      <td>0.104198</td>
      <td>-0.110354</td>
      <td>0.226365</td>
      <td>0.300104</td>
      <td>0.335932</td>
      <td>1.000000</td>
      <td>0.589363</td>
      <td>0.225301</td>
    </tr>
    <tr>
      <th>TotalCharges</th>
      <td>0.083953</td>
      <td>0.339102</td>
      <td>0.079321</td>
      <td>0.854732</td>
      <td>0.122363</td>
      <td>0.108964</td>
      <td>0.589363</td>
      <td>1.000000</td>
      <td>-0.238282</td>
    </tr>
    <tr>
      <th>Churn</th>
      <td>0.158910</td>
      <td>-0.160169</td>
      <td>-0.167564</td>
      <td>-0.414469</td>
      <td>0.003702</td>
      <td>0.221131</td>
      <td>0.225301</td>
      <td>-0.238282</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




![png](output_12_1.png)


Here we see if the Tenure is high then churn is low.


```python
try:
    customer_id = df2['customerID'] # Store this as customer_id variable
    del df2['customerID'] # Don't need in ML DF
except:
    print("already removed customerID")
```


```python
df2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>SeniorCitizen</th>
      <th>Partner</th>
      <th>Dependents</th>
      <th>tenure</th>
      <th>PhoneService</th>
      <th>MultipleLines</th>
      <th>InternetService</th>
      <th>OnlineSecurity</th>
      <th>OnlineBackup</th>
      <th>DeviceProtection</th>
      <th>TechSupport</th>
      <th>StreamingTV</th>
      <th>StreamingMovies</th>
      <th>Contract</th>
      <th>PaperlessBilling</th>
      <th>PaymentMethod</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>no</td>
      <td>dsl</td>
      <td>yes</td>
      <td>yes</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>month-to-month</td>
      <td>1</td>
      <td>mailed check</td>
      <td>53.85</td>
      <td>108.15</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>female</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>no</td>
      <td>fiber optic</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>month-to-month</td>
      <td>1</td>
      <td>electronic check</td>
      <td>70.70</td>
      <td>151.65</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>female</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>yes</td>
      <td>fiber optic</td>
      <td>no</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>yes</td>
      <td>yes</td>
      <td>month-to-month</td>
      <td>1</td>
      <td>electronic check</td>
      <td>99.65</td>
      <td>820.50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>female</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>28</td>
      <td>1</td>
      <td>yes</td>
      <td>fiber optic</td>
      <td>no</td>
      <td>no</td>
      <td>yes</td>
      <td>yes</td>
      <td>yes</td>
      <td>yes</td>
      <td>month-to-month</td>
      <td>1</td>
      <td>electronic check</td>
      <td>104.80</td>
      <td>3046.05</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>49</td>
      <td>1</td>
      <td>yes</td>
      <td>fiber optic</td>
      <td>no</td>
      <td>yes</td>
      <td>yes</td>
      <td>no</td>
      <td>yes</td>
      <td>yes</td>
      <td>month-to-month</td>
      <td>1</td>
      <td>bank transfer (automatic)</td>
      <td>103.70</td>
      <td>5036.30</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Using one-hot encoding to convert categorical data to binary (0 or 1)
ml_dummies = pd.get_dummies(df2)
ml_dummies.fillna(value=0, inplace=True)
ml_dummies.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SeniorCitizen</th>
      <th>Partner</th>
      <th>Dependents</th>
      <th>tenure</th>
      <th>PhoneService</th>
      <th>PaperlessBilling</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>Churn</th>
      <th>gender_female</th>
      <th>...</th>
      <th>StreamingMovies_no</th>
      <th>StreamingMovies_no internet service</th>
      <th>StreamingMovies_yes</th>
      <th>Contract_month-to-month</th>
      <th>Contract_one year</th>
      <th>Contract_two year</th>
      <th>PaymentMethod_bank transfer (automatic)</th>
      <th>PaymentMethod_credit card (automatic)</th>
      <th>PaymentMethod_electronic check</th>
      <th>PaymentMethod_mailed check</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>53.85</td>
      <td>108.15</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>70.70</td>
      <td>151.65</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>1</td>
      <td>99.65</td>
      <td>820.50</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>28</td>
      <td>1</td>
      <td>1</td>
      <td>104.80</td>
      <td>3046.05</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>49</td>
      <td>1</td>
      <td>1</td>
      <td>103.70</td>
      <td>5036.30</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 42 columns</p>
</div>




```python
try:
    target = ml_dummies['Churn'] # Remove the label before training the model
    del ml_dummies['Churn']
except:
    print("target already removed.")

from sklearn.model_selection import train_test_split
feature_train, feature_test, label_train, label_test = train_test_split(ml_dummies, target, test_size=0.3)
```


```python
# Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

classifiers = [
    KNeighborsClassifier(5),    
    DecisionTreeClassifier(max_depth=5)]
    

# iterate over classifiers
for item in classifiers:
    classifier_name = ((str(item)[:(str(item).find("("))]))
    print (classifier_name)
    
    # Create classifier, train it and test it.
    clf = item
    clf.fit(feature_train, label_train)
    pred = clf.predict(feature_test)
    score = clf.score(feature_test, label_test)
    print (round(score,3),"\n", "- - - - - ", "\n")
    
    # Cross Validation
    cv_results = cross_val_score(clf, ml_dummies, target, cv=10)
    print (round(np.mean(cv_results),3),"\n", "- - - - - ", "\n")
    
    
feature_df = pd.DataFrame()
feature_df['features'] = ml_dummies.columns
feature_df['importance'] = clf.feature_importances_
feature_df.set_index('features').sort_values(by='importance', ascending=True).plot(kind='barh', figsize=(20, 15))
print(feature_df.sort_values(by='importance', ascending=True))
```

    KNeighborsClassifier
    0.672 
     - - - - -  
    
    0.687 
     - - - - -  
    
    DecisionTreeClassifier
    0.745 
     - - - - -  
    
    0.743 
     - - - - -  
    
                                       features  importance
    0                             SeniorCitizen    0.000000
    38    PaymentMethod_credit card (automatic)    0.000000
    35                        Contract_one year    0.000000
    33                      StreamingMovies_yes    0.000000
    32      StreamingMovies_no internet service    0.000000
    30                          StreamingTV_yes    0.000000
    29          StreamingTV_no internet service    0.000000
    28                           StreamingTV_no    0.000000
    27                          TechSupport_yes    0.000000
    26          TechSupport_no internet service    0.000000
    25                           TechSupport_no    0.000000
    24                     DeviceProtection_yes    0.000000
    23     DeviceProtection_no internet service    0.000000
    22                      DeviceProtection_no    0.000000
    18                       OnlineSecurity_yes    0.000000
    17       OnlineSecurity_no internet service    0.000000
    20         OnlineBackup_no internet service    0.000000
    15                       InternetService_no    0.000000
    1                                   Partner    0.000000
    2                                Dependents    0.000000
    4                              PhoneService    0.000000
    8                             gender_female    0.000000
    9                               gender_male    0.000000
    10                         MultipleLines_no    0.000000
    40               PaymentMethod_mailed check    0.000000
    12                        MultipleLines_yes    0.000000
    13                      InternetService_dsl    0.000000
    21                         OnlineBackup_yes    0.000620
    31                       StreamingMovies_no    0.000640
    37  PaymentMethod_bank transfer (automatic)    0.003461
    39           PaymentMethod_electronic check    0.005260
    5                          PaperlessBilling    0.007269
    36                        Contract_two year    0.009599
    19                          OnlineBackup_no    0.012117
    11           MultipleLines_no phone service    0.018830
    7                              TotalCharges    0.036166
    14              InternetService_fiber optic    0.043532
    16                        OnlineSecurity_no    0.093619
    6                            MonthlyCharges    0.096882
    3                                    tenure    0.121391
    34                  Contract_month-to-month    0.550615



![png](output_18_1.png)



```python
# Confusion matrix
from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(label_test, pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
class_names = ['Not churned','churned']

plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()


from sklearn.metrics import classification_report
eval_metrics = classification_report(label_test, pred, target_names=class_names)
print(eval_metrics)
```

    Confusion matrix, without normalization
    [[354 191]
     [ 95 482]]
    Normalized confusion matrix
    [[0.65 0.35]
     [0.16 0.84]]



![png](output_19_1.png)



![png](output_19_2.png)


                 precision    recall  f1-score   support
    
    Not churned       0.79      0.65      0.71       545
        churned       0.72      0.84      0.77       577
    
    avg / total       0.75      0.75      0.74      1122
    



```python
feature_df = pd.DataFrame()
feature_df['prediction'] = clf.predict_proba(ml_dummies)[:,1]
feature_df['prediction'].sort_values(ascending=False)
```




    3579    1.000000
    1800    1.000000
    3517    1.000000
    1091    1.000000
    1206    1.000000
    1696    1.000000
    1225    0.979021
    1220    0.979021
    233     0.979021
    3423    0.979021
    1212    0.979021
    1142    0.979021
    1203    0.979021
    1195    0.979021
    245     0.979021
    1236    0.979021
    246     0.979021
    509     0.979021
    251     0.979021
    1166    0.979021
    1146    0.979021
    1133    0.979021
    1135    0.979021
    1246    0.979021
    318     0.979021
    1028    0.979021
    1034    0.979021
    312     0.979021
    1037    0.979021
    1038    0.979021
              ...   
    3094    0.007937
    3088    0.007937
    2099    0.007937
    3081    0.007937
    1231    0.007937
    3080    0.007937
    3078    0.007937
    3073    0.007937
    3069    0.007937
    2103    0.007937
    3065    0.007937
    3061    0.007937
    2106    0.007937
    2111    0.007937
    3058    0.007937
    3045    0.007937
    2353    0.007937
    3042    0.007937
    1214    0.007937
    2129    0.007937
    3028    0.007937
    2131    0.007937
    3024    0.007937
    3015    0.007937
    2137    0.007937
    3001    0.007937
    2142    0.007937
    2983    0.007937
    2979    0.007937
    2113    0.007937
    Name: prediction, Length: 3738, dtype: float64




```python
# Preprocessing original dataframe
def preprocess_df(dataframe):
    x = dataframe.copy()
    try:
        customer_id = x['customerID']
        del x['customerID'] # Don't need in ML DF
    except:
        print("already removed customerID")
    ml_dummies = pd.get_dummies(x)
    ml_dummies.fillna(value=0, inplace=True)

    try:
        label = ml_dummies['Churn']
        del ml_dummies['Churn']
    except:
        print("label already removed.")
    return ml_dummies, customer_id, label

original_df = preprocess_df(df)

output_df = original_df[0].copy()
output_df['prediction'] = clf.predict_proba(output_df)[:,1]
output_df['churn'] = original_df[2]
output_df['customerID'] = original_df[1]

print('Mean predict proba of churn:',round(output_df[output_df['churn'] == 1]['prediction'].mean(),2))
print('Mean predict proba of NON-churn:',round(output_df[output_df['churn'] == 0]['prediction'].mean(),2))

activate = output_df[output_df['churn'] == 0]
activate.sort_values(by='prediction', ascending=False)
```

    Mean predict proba of churn: 0.69
    Mean predict proba of NON-churn: 0.32





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SeniorCitizen</th>
      <th>Partner</th>
      <th>Dependents</th>
      <th>tenure</th>
      <th>PhoneService</th>
      <th>PaperlessBilling</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>gender_female</th>
      <th>gender_male</th>
      <th>...</th>
      <th>Contract_month-to-month</th>
      <th>Contract_one year</th>
      <th>Contract_two year</th>
      <th>PaymentMethod_bank transfer (automatic)</th>
      <th>PaymentMethod_credit card (automatic)</th>
      <th>PaymentMethod_electronic check</th>
      <th>PaymentMethod_mailed check</th>
      <th>prediction</th>
      <th>churn</th>
      <th>customerID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>321</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>72</td>
      <td>0</td>
      <td>1</td>
      <td>60.00</td>
      <td>4264.00</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1.000000</td>
      <td>0</td>
      <td>9880-tdqac</td>
    </tr>
    <tr>
      <th>1969</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>60</td>
      <td>0</td>
      <td>0</td>
      <td>53.60</td>
      <td>3237.05</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1.000000</td>
      <td>0</td>
      <td>5110-chopy</td>
    </tr>
    <tr>
      <th>114</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>64</td>
      <td>0</td>
      <td>1</td>
      <td>54.60</td>
      <td>3423.50</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1.000000</td>
      <td>0</td>
      <td>5256-skjgo</td>
    </tr>
    <tr>
      <th>4968</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>68</td>
      <td>0</td>
      <td>1</td>
      <td>53.00</td>
      <td>3656.25</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1.000000</td>
      <td>0</td>
      <td>2197-omwgi</td>
    </tr>
    <tr>
      <th>5338</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>37</td>
      <td>0</td>
      <td>0</td>
      <td>45.40</td>
      <td>1593.10</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1.000000</td>
      <td>0</td>
      <td>2580-asvvy</td>
    </tr>
    <tr>
      <th>2538</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>34</td>
      <td>0</td>
      <td>1</td>
      <td>40.55</td>
      <td>1325.85</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1.000000</td>
      <td>0</td>
      <td>7758-ujwys</td>
    </tr>
    <tr>
      <th>6093</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>71</td>
      <td>0</td>
      <td>1</td>
      <td>47.05</td>
      <td>3263.60</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1.000000</td>
      <td>0</td>
      <td>7110-bdtwg</td>
    </tr>
    <tr>
      <th>1634</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>63</td>
      <td>0</td>
      <td>0</td>
      <td>59.00</td>
      <td>3707.60</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1.000000</td>
      <td>0</td>
      <td>9995-hotoh</td>
    </tr>
    <tr>
      <th>6547</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>68</td>
      <td>0</td>
      <td>1</td>
      <td>41.95</td>
      <td>2965.75</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1.000000</td>
      <td>0</td>
      <td>3908-mkimj</td>
    </tr>
    <tr>
      <th>4726</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>72</td>
      <td>0</td>
      <td>1</td>
      <td>49.20</td>
      <td>3580.95</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1.000000</td>
      <td>0</td>
      <td>2192-ckrlv</td>
    </tr>
    <tr>
      <th>6514</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>72</td>
      <td>0</td>
      <td>1</td>
      <td>64.70</td>
      <td>4746.05</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1.000000</td>
      <td>0</td>
      <td>6166-yipfo</td>
    </tr>
    <tr>
      <th>6416</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>67</td>
      <td>0</td>
      <td>0</td>
      <td>43.90</td>
      <td>3097.20</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1.000000</td>
      <td>0</td>
      <td>9610-wcesf</td>
    </tr>
    <tr>
      <th>255</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>67</td>
      <td>0</td>
      <td>1</td>
      <td>59.55</td>
      <td>4103.90</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1.000000</td>
      <td>0</td>
      <td>4111-bnxif</td>
    </tr>
    <tr>
      <th>5674</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>64</td>
      <td>0</td>
      <td>0</td>
      <td>49.85</td>
      <td>3210.35</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1.000000</td>
      <td>0</td>
      <td>3566-caayu</td>
    </tr>
    <tr>
      <th>1813</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>75.50</td>
      <td>75.50</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.979021</td>
      <td>0</td>
      <td>0817-hsuse</td>
    </tr>
    <tr>
      <th>3740</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>71.65</td>
      <td>135.75</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.979021</td>
      <td>0</td>
      <td>0508-oolto</td>
    </tr>
    <tr>
      <th>3633</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>71.25</td>
      <td>71.25</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.979021</td>
      <td>0</td>
      <td>3878-avsoq</td>
    </tr>
    <tr>
      <th>3536</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>79.15</td>
      <td>79.15</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.979021</td>
      <td>0</td>
      <td>2254-dlxri</td>
    </tr>
    <tr>
      <th>3529</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>74.20</td>
      <td>140.10</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.979021</td>
      <td>0</td>
      <td>2880-fpnae</td>
    </tr>
    <tr>
      <th>5071</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>75.35</td>
      <td>75.35</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.979021</td>
      <td>0</td>
      <td>1746-tgtwv</td>
    </tr>
    <tr>
      <th>1929</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>81.05</td>
      <td>81.05</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.979021</td>
      <td>0</td>
      <td>2018-qkygt</td>
    </tr>
    <tr>
      <th>684</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>71.10</td>
      <td>71.10</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.979021</td>
      <td>0</td>
      <td>8040-mnrtf</td>
    </tr>
    <tr>
      <th>2441</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>81.50</td>
      <td>162.55</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.979021</td>
      <td>0</td>
      <td>9050-ikdza</td>
    </tr>
    <tr>
      <th>4094</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>74.00</td>
      <td>74.00</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.979021</td>
      <td>0</td>
      <td>8687-bafgu</td>
    </tr>
    <tr>
      <th>345</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>72.10</td>
      <td>72.10</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.979021</td>
      <td>0</td>
      <td>0021-ikxgc</td>
    </tr>
    <tr>
      <th>4916</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>79.55</td>
      <td>151.75</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.979021</td>
      <td>0</td>
      <td>3420-yjlqt</td>
    </tr>
    <tr>
      <th>532</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>79.95</td>
      <td>174.45</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.979021</td>
      <td>0</td>
      <td>4234-xtnea</td>
    </tr>
    <tr>
      <th>3435</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>71.30</td>
      <td>157.75</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.979021</td>
      <td>0</td>
      <td>4450-dllmh</td>
    </tr>
    <tr>
      <th>5307</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>90.10</td>
      <td>90.10</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.979021</td>
      <td>0</td>
      <td>1941-hosam</td>
    </tr>
    <tr>
      <th>2519</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>91.45</td>
      <td>171.45</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.979021</td>
      <td>0</td>
      <td>4927-wwooz</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5511</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>70</td>
      <td>1</td>
      <td>1</td>
      <td>76.95</td>
      <td>5289.80</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.007937</td>
      <td>0</td>
      <td>6586-mygkd</td>
    </tr>
    <tr>
      <th>5508</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>70</td>
      <td>1</td>
      <td>0</td>
      <td>25.15</td>
      <td>1940.85</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.007937</td>
      <td>0</td>
      <td>6010-ddppw</td>
    </tr>
    <tr>
      <th>5582</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>48</td>
      <td>1</td>
      <td>0</td>
      <td>24.00</td>
      <td>1183.05</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.007937</td>
      <td>0</td>
      <td>7601-dhfwz</td>
    </tr>
    <tr>
      <th>3179</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>27</td>
      <td>1</td>
      <td>1</td>
      <td>19.60</td>
      <td>561.15</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.007937</td>
      <td>0</td>
      <td>3721-wkiil</td>
    </tr>
    <tr>
      <th>3181</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>64</td>
      <td>1</td>
      <td>1</td>
      <td>81.30</td>
      <td>5129.30</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.007937</td>
      <td>0</td>
      <td>1265-zfosd</td>
    </tr>
    <tr>
      <th>5496</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>43</td>
      <td>1</td>
      <td>0</td>
      <td>24.25</td>
      <td>1077.95</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.007937</td>
      <td>0</td>
      <td>2208-nkvvh</td>
    </tr>
    <tr>
      <th>5492</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>65</td>
      <td>1</td>
      <td>1</td>
      <td>25.30</td>
      <td>1748.55</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.007937</td>
      <td>0</td>
      <td>2799-tslag</td>
    </tr>
    <tr>
      <th>3184</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>71</td>
      <td>1</td>
      <td>1</td>
      <td>83.30</td>
      <td>5894.50</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.007937</td>
      <td>0</td>
      <td>5197-lqxxh</td>
    </tr>
    <tr>
      <th>3186</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>58</td>
      <td>1</td>
      <td>0</td>
      <td>20.30</td>
      <td>1160.75</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.007937</td>
      <td>0</td>
      <td>3457-pqbyh</td>
    </tr>
    <tr>
      <th>3190</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>66</td>
      <td>0</td>
      <td>0</td>
      <td>54.65</td>
      <td>3632.00</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.007937</td>
      <td>0</td>
      <td>3745-hrphi</td>
    </tr>
    <tr>
      <th>539</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>60</td>
      <td>1</td>
      <td>1</td>
      <td>80.60</td>
      <td>4946.70</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.007937</td>
      <td>0</td>
      <td>5394-meitz</td>
    </tr>
    <tr>
      <th>5535</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>55</td>
      <td>1</td>
      <td>1</td>
      <td>85.10</td>
      <td>4657.95</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.007937</td>
      <td>0</td>
      <td>2404-jibfc</td>
    </tr>
    <tr>
      <th>5542</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>67</td>
      <td>1</td>
      <td>0</td>
      <td>86.15</td>
      <td>5883.85</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.007937</td>
      <td>0</td>
      <td>2990-ogytd</td>
    </tr>
    <tr>
      <th>1390</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>72</td>
      <td>1</td>
      <td>0</td>
      <td>68.75</td>
      <td>4888.20</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.007937</td>
      <td>0</td>
      <td>8039-aclpl</td>
    </tr>
    <tr>
      <th>5578</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>42</td>
      <td>1</td>
      <td>0</td>
      <td>20.00</td>
      <td>833.55</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.007937</td>
      <td>0</td>
      <td>8215-ngspe</td>
    </tr>
    <tr>
      <th>1407</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>44.55</td>
      <td>343.45</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.007937</td>
      <td>0</td>
      <td>0895-uadgo</td>
    </tr>
    <tr>
      <th>5572</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>70</td>
      <td>1</td>
      <td>1</td>
      <td>74.80</td>
      <td>5315.80</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.007937</td>
      <td>0</td>
      <td>4654-ulttn</td>
    </tr>
    <tr>
      <th>3120</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>69</td>
      <td>1</td>
      <td>1</td>
      <td>24.95</td>
      <td>1718.35</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.007937</td>
      <td>0</td>
      <td>3148-aoiqt</td>
    </tr>
    <tr>
      <th>528</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>35</td>
      <td>1</td>
      <td>0</td>
      <td>19.60</td>
      <td>686.95</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.007937</td>
      <td>0</td>
      <td>7601-gndyk</td>
    </tr>
    <tr>
      <th>1406</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>66</td>
      <td>1</td>
      <td>1</td>
      <td>25.30</td>
      <td>1673.80</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.007937</td>
      <td>0</td>
      <td>9337-srrni</td>
    </tr>
    <tr>
      <th>5566</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>72</td>
      <td>1</td>
      <td>0</td>
      <td>82.15</td>
      <td>5784.30</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.007937</td>
      <td>0</td>
      <td>2302-ouzxb</td>
    </tr>
    <tr>
      <th>3124</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>68</td>
      <td>1</td>
      <td>0</td>
      <td>82.85</td>
      <td>5776.45</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.007937</td>
      <td>0</td>
      <td>3889-vwbid</td>
    </tr>
    <tr>
      <th>530</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>24</td>
      <td>1</td>
      <td>1</td>
      <td>21.10</td>
      <td>490.65</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.007937</td>
      <td>0</td>
      <td>8067-nioym</td>
    </tr>
    <tr>
      <th>531</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>70</td>
      <td>1</td>
      <td>0</td>
      <td>20.05</td>
      <td>1360.25</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.007937</td>
      <td>0</td>
      <td>1403-gyafu</td>
    </tr>
    <tr>
      <th>3131</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>72</td>
      <td>1</td>
      <td>0</td>
      <td>78.45</td>
      <td>5682.25</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.007937</td>
      <td>0</td>
      <td>8336-tavkx</td>
    </tr>
    <tr>
      <th>1404</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>35</td>
      <td>1</td>
      <td>0</td>
      <td>23.30</td>
      <td>797.10</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.007937</td>
      <td>0</td>
      <td>1970-kkfwl</td>
    </tr>
    <tr>
      <th>5555</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>70</td>
      <td>1</td>
      <td>0</td>
      <td>19.80</td>
      <td>1378.75</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.007937</td>
      <td>0</td>
      <td>8750-qwzaj</td>
    </tr>
    <tr>
      <th>5554</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>63</td>
      <td>1</td>
      <td>1</td>
      <td>20.60</td>
      <td>1298.70</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.007937</td>
      <td>0</td>
      <td>8838-gphzp</td>
    </tr>
    <tr>
      <th>5548</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>70</td>
      <td>1</td>
      <td>0</td>
      <td>24.25</td>
      <td>1724.15</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.007937</td>
      <td>0</td>
      <td>4589-iuajb</td>
    </tr>
    <tr>
      <th>1843</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>71</td>
      <td>1</td>
      <td>0</td>
      <td>19.70</td>
      <td>1415.85</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.007937</td>
      <td>0</td>
      <td>3173-wssue</td>
    </tr>
  </tbody>
</table>
<p>5174 rows × 44 columns</p>
</div>


