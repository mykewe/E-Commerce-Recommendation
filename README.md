# E-Commerce-Recommendation

The goal of this project is to create an e-commerce recommender system using data from the e-commerce website http://ecom.uelstore.com/. Using collaborative filtering, the recommender is created and implemented. This dataset for a product recommendation system contains user ratings for a specific product. Based on how closely related the rated products are, the system will try to recommend products to current users.


```python
#Import libraries
import pandas as pd
from surprise import Reader
```

##### Customer data


```python
customers=pd.read_json("data/customers.json")
```


```python
customers.size
```




    1356




```python
customers.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>NickName</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>103603</td>
      <td>1000kgthanh</td>
    </tr>
    <tr>
      <th>1</th>
      <td>103760</td>
      <td>999999999ok</td>
    </tr>
    <tr>
      <th>2</th>
      <td>103829</td>
      <td>ac7ive</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>admin</td>
    </tr>
    <tr>
      <th>4</th>
      <td>103839</td>
      <td>ahkk.nguyen</td>
    </tr>
  </tbody>
</table>
</div>



##### Product data


```python
products=pd.read_json("data/products.json")
```


```python
products.size
```




    2073




```python
products.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>Name</th>
      <th>UnitPrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Build your own computer</td>
      <td>1200.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Digital Storm VANQUISH 3 Custom Performance PC</td>
      <td>1259.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Lenovo IdeaCentre 600 All-in-One PC</td>
      <td>500.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Apple MacBook Pro 13-inch</td>
      <td>1800.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Asus N551JK-XO076H Laptop</td>
      <td>1500.0</td>
    </tr>
  </tbody>
</table>
</div>



##### Rating data


```python
ratings=pd.read_json("data/ratings.json")
```


```python
ratings.size
```




    523016




```python
ratings.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CustomerID</th>
      <th>ProductID</th>
      <th>Rate</th>
      <th>CreateDate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>103416</td>
      <td>619</td>
      <td>1</td>
      <td>2018/01/01 01:36:30</td>
    </tr>
    <tr>
      <th>1</th>
      <td>103654</td>
      <td>411</td>
      <td>1</td>
      <td>2018/01/01 01:36:35</td>
    </tr>
    <tr>
      <th>2</th>
      <td>103954</td>
      <td>298</td>
      <td>3</td>
      <td>2018/01/01 01:36:38</td>
    </tr>
    <tr>
      <th>3</th>
      <td>103672</td>
      <td>361</td>
      <td>5</td>
      <td>2018/01/01 01:37:15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>103960</td>
      <td>536</td>
      <td>5</td>
      <td>2018/01/01 02:36:25</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Distribution of ratings
print(ratings['Rate'].value_counts())
```

    5    35512
    4    30458
    1    27876
    2    21070
    3    15838
    Name: Rate, dtype: int64



```python
import seaborn as sns
sns.countplot(x=ratings['Rate'])
```




    <AxesSubplot:xlabel='Rate', ylabel='count'>




    
![png](output_15_1.png)
    



```python

```


```python
#Drop the CreateDate column as it is not needed
ratings.drop('CreateDate', inplace=True, axis=1)
ratings.head()
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
      <th>CustomerID</th>
      <th>ProductID</th>
      <th>Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>103416</td>
      <td>619</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>103654</td>
      <td>411</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>103954</td>
      <td>298</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>103672</td>
      <td>361</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>103960</td>
      <td>536</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



#### Build recommender system


```python
# Prepare data for surprise: build a Suprise reader object
from surprise import Reader
reader = Reader(rating_scale=(1, 5))
```


```python
# Load `ratings` into a Surprise Dataset
from surprise import Dataset
rec_data = Dataset.load_from_df(ratings,
                                reader)
```


```python
# Create a 80:20 train-test split and set the random state to 7
from surprise.model_selection import train_test_split
trainset, testset = train_test_split(rec_data, test_size=.2, random_state=7)

```


```python
# Use KNNBasic from Surprise to train a collaborative filter
from surprise import KNNBasic
recommender = KNNBasic()
recommender.fit(trainset)
```

    Computing the msd similarity matrix...
    Done computing similarity matrix.





    <surprise.prediction_algorithms.knns.KNNBasic at 0x7ff602ce5b80>




```python
# Evaluate the recommender system
from surprise import accuracy
predictions = recommender.test(testset)
accuracy.rmse(predictions)
```

    RMSE: 1.1021





    1.1021485303764857




```python
testset[0]
```




    (103441, 450, 5.0)




```python
# Prediction on a user 103441 who gave the product 450 a rating of 5
print(recommender.predict('103441', '450').est)
```

    3.1877575212948



```python
#Try other algorithms
from surprise import NormalPredictor
from surprise import KNNWithMeans
from surprise import KNNWithZScore
from surprise import KNNBaseline
from surprise import SVD
from surprise import BaselineOnly
from surprise import SVDpp
from surprise import NMF
from surprise import SlopeOne
from surprise import CoClustering

algorithms = [SVD(), SlopeOne(), NMF(), NormalPredictor(),
              KNNBaseline(),KNNWithMeans(), KNNWithZScore(),
              BaselineOnly(), CoClustering()]
```


```python
print ("Attempting: ", str(algorithms), '\n\n\n')
benchmark=[]    
for algorithm in algorithms:
    print("Starting: " ,str(algorithm))

    algorithm.fit(trainset)

    predictions=algorithm.test(testset)

    score_rmse=accuracy.rmse(predictions)

    metrics=[str(algorithm).split(' ')[0].split('.')[-1],score_rmse]

    benchmark.append(metrics)   

print ('\n\tDONE\n')
report=pd.DataFrame(benchmark,columns=["Algorithm","RMSE"])
report.set_index("Algorithm")
print("-------------------------------------------------")
print(report)
```

    Attempting:  [<surprise.prediction_algorithms.matrix_factorization.SVD object at 0x7ff5ec0aba30>, <surprise.prediction_algorithms.slope_one.SlopeOne object at 0x7ff5ec0ab430>, <surprise.prediction_algorithms.matrix_factorization.NMF object at 0x7ff5ec0abb50>, <surprise.prediction_algorithms.random_pred.NormalPredictor object at 0x7ff5ec0ab820>, <surprise.prediction_algorithms.knns.KNNBaseline object at 0x7ff5ec0ab4f0>, <surprise.prediction_algorithms.knns.KNNWithMeans object at 0x7ff5ec0ab400>, <surprise.prediction_algorithms.knns.KNNWithZScore object at 0x7ff5ec0abdc0>, <surprise.prediction_algorithms.baseline_only.BaselineOnly object at 0x7ff5ec0aba90>, <surprise.prediction_algorithms.co_clustering.CoClustering object at 0x7ff5ec0b3250>] 
    
    
    
    Starting:  <surprise.prediction_algorithms.matrix_factorization.SVD object at 0x7ff5ec0aba30>
    RMSE: 1.2014
    Starting:  <surprise.prediction_algorithms.slope_one.SlopeOne object at 0x7ff5ec0ab430>
    RMSE: 1.0598
    Starting:  <surprise.prediction_algorithms.matrix_factorization.NMF object at 0x7ff5ec0abb50>
    RMSE: 1.0702
    Starting:  <surprise.prediction_algorithms.random_pred.NormalPredictor object at 0x7ff5ec0ab820>
    RMSE: 1.9648
    Starting:  <surprise.prediction_algorithms.knns.KNNBaseline object at 0x7ff5ec0ab4f0>
    Estimating biases using als...
    Computing the msd similarity matrix...
    Done computing similarity matrix.
    RMSE: 1.0786
    Starting:  <surprise.prediction_algorithms.knns.KNNWithMeans object at 0x7ff5ec0ab400>
    Computing the msd similarity matrix...
    Done computing similarity matrix.
    RMSE: 1.0779
    Starting:  <surprise.prediction_algorithms.knns.KNNWithZScore object at 0x7ff5ec0abdc0>
    Computing the msd similarity matrix...
    Done computing similarity matrix.
    RMSE: 1.0827
    Starting:  <surprise.prediction_algorithms.baseline_only.BaselineOnly object at 0x7ff5ec0aba90>
    Estimating biases using als...
    RMSE: 1.0569
    Starting:  <surprise.prediction_algorithms.co_clustering.CoClustering object at 0x7ff5ec0b3250>
    RMSE: 1.0675
    
    	DONE
    
    -------------------------------------------------
             Algorithm      RMSE
    0              SVD  1.201366
    1         SlopeOne  1.059767
    2              NMF  1.070244
    3  NormalPredictor  1.964785
    4      KNNBaseline  1.078623
    5     KNNWithMeans  1.077890
    6    KNNWithZScore  1.082702
    7     BaselineOnly  1.056871
    8     CoClustering  1.067525


The algorithm `BaselineOnly` has the lowest RMSE score


```python
best_recommender = BaselineOnly()
best_recommender.fit(trainset)
```

    Estimating biases using als...





    <surprise.prediction_algorithms.baseline_only.BaselineOnly at 0x7ff60318a190>




```python

```
