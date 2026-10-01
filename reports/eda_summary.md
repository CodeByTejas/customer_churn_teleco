# EDA summary
Shape: (7043, 21)

## Raw dtypes
```
customerID              str
gender                  str
SeniorCitizen         int64
Partner                 str
Dependents              str
tenure                int64
PhoneService            str
MultipleLines           str
InternetService         str
OnlineSecurity          str
OnlineBackup            str
DeviceProtection        str
TechSupport             str
StreamingTV             str
StreamingMovies         str
Contract                str
PaperlessBilling        str
PaymentMethod           str
MonthlyCharges      float64
TotalCharges            str
Churn                   str
```
TotalCharges is parsed as **str**; 11 rows contain blank strings.

## Missing after type coercion
```
TotalCharges    11
```
All blank TotalCharges rows have tenure==0: True

## Target balance
```
Churn
No     0.735
Yes    0.265
```

## Numeric summary
```
        tenure  MonthlyCharges  TotalCharges
count  7043.00         7043.00       7032.00
mean     32.37           64.76       2283.30
std      24.56           30.09       2266.77
min       0.00           18.25         18.80
25%       9.00           35.50        401.45
50%      29.00           70.35       1397.48
75%      55.00           89.85       3794.74
max      72.00          118.75       8684.80
```

## Churn rate by key categoricals
**Contract**
```
Contract
Month-to-month    0.427
One year          0.113
Two year          0.028
```
**InternetService**
```
InternetService
Fiber optic    0.419
DSL            0.190
No             0.074
```
**PaymentMethod**
```
PaymentMethod
Electronic check             0.453
Mailed check                 0.191
Bank transfer (automatic)    0.167
Credit card (automatic)      0.152
```
**TechSupport**
```
TechSupport
No                     0.416
Yes                    0.152
No internet service    0.074
```