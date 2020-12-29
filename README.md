## <MBA_Datamining>
### Final Project : PEP Prediction Project

### 1. Introduction

**▷ Objective**  
```
To predict whether a pension insurance product is subscribed or not by utilizing customer data.
```
---
**▷ Data**  
```
- 12 variables, 600 records
```
<STYLE>
tr{
font-size: 10px
}
td{
font-size: 10px}
</STYLE>
|No|Column|Count|Type|
|:-:|:-:|:-:|:-:|
|0|ID|600|Object|
|1|Age|600|Int64|
|2|Sex|600|Int64|
|3|Region|600|Int64|
|4|Income|600|Float64|
|5|Married|600|Int64|
|6|Children|600|Int64|
|7|Car|600|Int64|
|8|Save_Act|600|Int64|
|9|Current_Act|600|Int64|
|10|Mortgage|600|Int64|
|11|PEP|600|Int64|

---
**▷ Requirements**
```
- X variables: 10 variables excluding ID and PEP + additional generating variables
- Y variable: PEP
- Train / Test ratio: 70% / 30%
- 3 or more predictive models to implement
- Perform preprocessing on X variables
- Create at least 5 additional variables
```
---
**▷ Modeling**
```  
• Optimize parameters used in the modeling process by Grid Search  
• Selecting various modeling techniques and algorithms that can accurately predict PEP values  
• Chosen 10 trial Model (*Top 3 model)
 - Decision Tree
 - Random Forest* (90.56%)
 - Catboost* (91.11%)
 - Voting* (90.56%)
 - Gradient Boosting
 - Multilayer Perceptron
 - Support Vector Machine
 - Adaboost
 - K-Nearest Neighbor
 - Logistic Regression
 ```
