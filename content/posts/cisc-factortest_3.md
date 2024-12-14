+++
title = 'Test the Validity of a Factor'
date = 2024-02-02T02:29:56+08:00
draft = true
+++

This project reproduced the factor validity test part in a report that focuses on the information of heavy stocks under the fund selection, and how to use the data of heavy stocks of the preferred fund to build a stable investment portfolio.

### Rank IC test

If a factor has a predictive effect on the expected return of the stock, there will be a certain correlation between the factor value of the stock in the current period and the return of the stock in the next period. If we use the Pearson linear correlation coefficient between the two, then the presence of certain outliers may greatly affect the results. Therefore, we will use the more robust Spearman rank correlation coefficient to measure the validity of factors. 

As shown in the figure below, the ranking sequence A is obtained by sorting all stocks according to the factor value of the current period, and the ranking sequence B is obtained by sorting all stocks according to the return of the next period. The larger the absolute value of Rank IC is, the stronger the predictive power of this factor for stock returns is. Generally, we will count the mean, standard deviation and t-statistics of Rank IC in the sample interval to analyze the performance of factors from multiple perspectives such as significance and stability of predictive ability.

![fig1](/img/cisc_3_1.png)

For stock \(i\) (where \(i=1, \cdots, n\)), denote the **ranks** of the current factor values of stocks as \(f_1, \cdots, f_n\), the **ranks** of the returns of the next period as \(r_1, \cdots, r_n\). 

$$
\text{RankIC}(f)=corr(A,B)=\frac{cov(\mathbf{A},\mathbf{B})}{\sqrt{var(\mathbf{A})\cdot var(\mathbf{B})}}
$$

where

$$
\mathbf{A}=\begin{bmatrix}
f_1\\
f_2\\
\vdots\\
f_n
\end{bmatrix},\quad
\mathbf{B}=\begin{bmatrix}
r_1\\
r_2\\
\vdots\\
r_n
\end{bmatrix}
$$

This process can be programmed following the steps.

**Step 1: Create a folder with a structure as below**

```
folder
├── data
│   ├── factors
│   └── return
└── main
```

**Step 2: Add raw data into the "data" folder**

- factors: 
  - [Institute holding proportion](/raw_data/cisc_3(institute_prop).csv)
  - [Security selecting ability.](/raw_data/cisc_3(select_ability).csv)
  - [Fund share](/raw_data/cisc_3(fund_share).csv)
  - [Recent annual return](/raw_data/cisc_3(annual_return).csv)
  - [Max drawdown](/raw_data/cisc_3(max_drawdown).csv)
- return: [Fund's net value](/raw_data/cisc_3(net_value).csv)

**Step 3: Create a `.py` file under the "main" folder, with the code below**

```
import pandas as pd
import numpy as np
import os
import math
import scipy.stats as stats
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
from datetime import datetime

factor_path = r'../data/factors/'
return_path = r'../data/return/'
result_dir = r'../result/'

if not os.path.exists(result_dir):
    os.mkdir(result_dir)

def get_fund_return(n):
    # Return rate of the next n terms
    df = pd.read_csv(os.path.join(return_path, 'cisc_3(net_value).csv'))
    df = df.rename(columns={'Unnamed: 0': 'Date'}).set_index('Date').fillna(0).astype(float)
    fund_value = df.copy().shift(n)
    fund_value_future = df.copy()
    date = fund_value.index[:-n].tolist()
    fund_return = fund_value_future / fund_value - 1
    # Replace infinity by NaN
    fund_return.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Delete all NaN
    fund_return = fund_return.dropna(axis=1, how='all')
    fund_return = fund_return.fillna(0)
    fund_return = fund_return.iloc[n:,:].reset_index(drop=True)
    fund_return.index = date
    return fund_return

# Select the funds whose corresponding annual returns are not all zero as the sample space
fund_return = get_fund_return(4)
fund_pool = list(fund_return.columns[fund_return.any()])
fund_return = fund_return.filter(items=fund_pool)
quarter_return = get_fund_return(1).filter(items=fund_pool).iloc[:-3, :]

# Read factor values
factor_csv_list = os.listdir(factor_path)
factor_csv_list = [x for x in factor_csv_list if x[-3:] == "csv"]
factor_name_list = [x[:-4] for x in factor_csv_list]
factor_df_list = []
for i in range(len(factor_csv_list)):
    file_path = os.path.join(factor_path, factor_csv_list[i])
    tmp_factor = pd.read_csv(file_path)
    tmp_factor.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
    tmp_factor = tmp_factor.set_index('Date')
    tmp_factor = tmp_factor.filter(items=fund_pool)
    tmp_factor.fillna(0, inplace=True)
    tmp_factor = tmp_factor.astype(str).apply(lambda x: x.str.replace(",", "")).astype(float)
    print(f"Reading：{factor_name_list[i]}")
    factor_df_list.append({factor_name_list[i]: tmp_factor})

# RankIC test
def get_RankIC(factor_name_list, factor_df_list):
    # Return of next year
    ICmean_list = np.zeros(len(factor_name_list))
    ICstd_list = np.zeros(len(factor_name_list))
    ICIR_list = np.zeros(len(factor_name_list))
    average_sample = np.zeros(len(factor_name_list))
    for i in range(len(factor_name_list)):
        factor_name = factor_name_list[i]
        factor_df = factor_df_list[i][factor_name]
        tmp_corr = np.zeros(factor_df.shape[0])
        sum_sample = 0
        for day in range(factor_df.shape[0]):
            return_series = fund_return.iloc[day, :]
            factor_series = factor_df.iloc[day, :]
            # Only keep non-zero columns
            return_series_nonzero = return_series[return_series != 0]
            factor_series_nonzero = factor_series[factor_series != 0]
            # Only keep the shared parts
            combine = pd.merge(return_series_nonzero, factor_series_nonzero, how='inner', left_index=True, right_index=True)
            return_series_common = combine.iloc[:, 0]
            factor_series_common = combine.iloc[:, 1]
            np_return = np.array(return_series_common)
            np_factor = np.array(factor_series_common)
            num_sample = len(np_return)
            sum_sample += num_sample
            rankic, _ = stats.spearmanr(np_return, np_factor)
            tmp_corr[day] = 0 if math.isnan(rankic) else rankic

        average_sample[i] = sum_sample / factor_df.shape[0]
        ICmean = np.mean(tmp_corr)
        ICstd = np.std(tmp_corr)
        ICIR = ICmean / ICstd
        ICmean_list[i] = ICmean
        ICstd_list[i] = ICstd
        ICIR_list[i] = ICIR

    IC_summary = pd.DataFrame(
        [ICmean_list, ICstd_list, ICIR_list, average_sample]
    )
    return IC_summary

def summarized_IC(factor_name_list, factor_df_list):
    col_names = ['ICmean', 'ICstd', 'ICIR', 'AvgSample']
    IC_df = get_RankIC(factor_name_list, factor_df_list).T
    IC_df.columns = col_names
    IC_df.index = factor_name_list
    return IC_df

IC_df = summarized_IC(factor_name_list, factor_df_list)
os.chdir(result_dir)
IC_df.to_excel("RankIC.xlsx")
```

### Portfolio test



### Reference

- [Research on heavy stock information under fund selection](/pdf/cisc_3_1.pdf)
- [Deep Analysis of Major Factors: Value Reproduction](/pdf/cisc_3_2.pdf)