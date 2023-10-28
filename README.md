# Project 2 - Bangkok Housing price

#### introduction
This is Project 2 of the General Assembly Data Science Immersive course. The objective of this project is to create regression model to predict price of property given the known data. Such attribute as Province, area, number of BTS nearby, etc. The to provide the explanation and justification of model selection, with brief walk-through of how to achieve the selected model.

#### Business problem
Home mortgage is the essential tool in making the dream of home ownership to reality for millions of individuals and families. As for the financial institution or bank, home mortgage is also one of the lowest risk financial products across the board, but there is still risk. To mitigate the risk, the approval credit should reflect the market value of the real estate. So in case of payment default and the real estate need to be sell by auction, the value should be equivalent  or greater than the approved credit, to minimize loss.

As a analyst, the task is to develop the tool to predict the market price from variety of input, such as location, size, number of train station nearby, etc. So the banker will have estimated value of property, based on the property itself.


#### Dataset
12,470 record of housing price in Bangkok, Nonthaburi, and Samutprakarn is provide. The data also contain feature such province name, property type, number of bedrooms, etc. All 22 (+ a property id) features explanation are provided in Data dictionary below.


#### Data Dictionary
Column|Data type|Description|
|---|---|---|
id|int|ID of selling item
province|string|province name: this dataset only includes Bangkok,Samut Prakan and Nonthaburi
district|string|district name
subdistrict|string|subdtistrict name
address|string|address e.g. street name, area name, soi number
property_type|string|type of the house: Condo, Townhouse or Detached House
total_units|float|the number of rooms/houses that the condo/village has
bedrooms|int|the number of bedrooms
baths|int|the number of baths
floor_area|float|total area of inside floor [㎡]
floor_level|int|floor level of the room
land_area|float|total area of the land [㎡]
latitude|float|latitude of the house
longitude|float|longitude of the house
nearby_stations|int|the number of nearby stations (within 1km)
nearby_station_distance|list|list of (station name, distance[m]). Each station name consists of station ID, station name, and Line such as "E4 Asok BTS"
nearby_bus_stops|int|the number of nearby bus stops
nearby_supermarkets|int|the number of nearby supermarkets
nearby_shops|int|the number of nearby shops
year_built|int|year built
month_built|string|month built: January-December
price|float|[TARGET VALUE] selling price


#### Data Exploration and Analysis (and cleasing)
##### Missing data treatment
10 out of 22 feature has missing data. Many of them missing significantly. 
> nearby_station_distance

49% of data is missing. But they're not missing by random, the missing data is where nearby_stations is 0. So, we can replace them with 0

> nearby_bus_stops

58% of data is missing. They are missing at random. And we can't remove them all, because if bus stop information is missing, but not the rest of features. removing more than half of all data could make lots of different to outcome. So I applied multiple imputation method, this is aiming to minimize impact on distribution and mean of this set of data.

> land_area

65% of data missing in this feature. Almost all condo type missing this information. This feature may only related to non-condo type. Consider to drop this feature from modeling

> total_unit

27% of data missing in this feature. They are missing at random. And we can't remove them all, because if bus stop information is missing, but not the rest of features. removing more than half of all data could make lots of different to outcome. So I applied multiple imputation method, this is aiming to minimize impact on distribution and mean of this set of data.

> month_built

41% of data is missing in this feature. They are missing at random. And we can't remove them all, because if bus stop information is missing, but not the rest of features. removing more than half of all data could make lots of different to outcome. Consider to drop this feature from modeling, since the correlation doesn't show any significant relation to price. But before drop them, fill the null value with multiple imputation method and convert month name to number (1 to 12), just incase for model exploration.

> floor_level

43% of data is missing in this feature. They are missing at random. And we can't remove them all, because if bus stop information is missing, but not the rest of features. removing more than half of all data could make lots of different to outcome. So I applied multiple imputation method, this is aiming to minimize impact on distribution and mean of this set of data.

> The rest

the rest of features has less than 3% missing data. This is not significant. So consider replace with mean where data is continuous and most frequent where data is categorical



#### Pre-processing
The property type, and district, and province data type is string. For model training purpose, I'd convert the these categorical data to numeric by apply one-hot code (dummy fied) and drop the first column to eliminate dependency of features.

The facility is also came in categorical data type. Since there are so many category within the facility, convert them to numeric data by using hot-one code may have impact on the complexity of model. So consider convert them to number by counting number of facility.

Prepare data for training.

- Create X by using train dataset
- Dropped id feature to prevent the data leakage
- Dropped price because this is for y
- Dropped 'subdistrict' because this is categorical data, and if we convert them, it will make the model to complex
- Dropped 'address', 'latitude', 'longitude', 'nearby_station_distance' because they are irrelevant.
- Split these train dataset into train-test data
- Standize date since they have different range

#### Modelling evaluation

Created 4 model: Linear regression and 3 regularization Lasso, Ridge, and Elastic net

###### Performance by metric - 1st run

| Model |R2 train | R2 test | RMSE train| RMSE test| CV score|
|---|---|---|---|---|---|
Linear regression|0.5449|0.579|1,463,788|1,434,566|0.5788

*CV = 5 fold

Model doesn't show sign of overfitting, but the performance is not good. The model can only explain 57% of data, and the have error margin approximately 1.4 million Baht.

##### Model tuning

There are still room to improve model, by increase features aiming to reduce bias. So I do the hot-code for `district` feature. And let build the model again.



###### Performance by metric - 2nd run

| Model |R2 train | R2 test | RMSE train| RMSE test| CV score|
|---|---|---|---|---|---|
Linear regression|0.6496|0.6737|1,284,991|1,260,572|0.6428
Ridge|0.6497|0.6733|1,284,787|1,260,122|0.6428
Lasso|0.6497|0.6734|1,284,785|1,261,223|0.6428
Elastic Net|0.6441|0.66424|1,294,521|1,278,692|0.6382

*CV = 5 fold

R2 score improved, RMSE also improve. And there are also no sign of overfitting. So perhaps I can try to increase the complexity of model by adding polynomial feature. and run the model again.

###### Performance by metric - 3rd run

| Model |R2 train | R2 test | RMSE train| RMSE test| CV score|
|---|---|---|---|---|---|
Polynomial|0.821| invalid value |916,920| invalid value | invalid value
Ridge|0.660|0.564|1,286,718|1,45,117|0.608


Now the model is too complex. The model is overfitting. The model can explain 82% of data, but the error margin is too high. Attemp to tune the hyperparameter is failed due to my PC limitation (the model is too complex). So I decided to drop the polynomial feature and fall back to the previous model.



##### Selected model
From the metric evaluation
Model selection is "Ridge" Model
- Not overfit
- Lowest RMSE
- Highest R2 score

#### Business recommendation
The model can be used to predict the price of property in Bangkok, Nonthaburi, and Samutprakarn. By input the data of property as follow: total_units', 'bedrooms', 'baths', 'floor_area', 'floor_level', 'land_area', 'nearby_stations', 'nearby_bus_stops', 'nearby_supermarkets', 'nearby_shops', 'year_built', 'month_built',
'facilities', 'property_type' and 'province'.

The model can explain up to 65% of data, and the have error margin approximately 1.2 million Baht. 

Interesting Finding from this model is
- Number of Baht has the most impact on price of property

