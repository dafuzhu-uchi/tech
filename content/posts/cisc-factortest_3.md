+++
title = 'Test the Validity of a Factor'
date = 2024-02-02T02:29:56+08:00
draft = false
tags = ["intern", "cisc", "2024", "factor"]
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

```plain
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

**Step 3: Create a `main.py` file under the "main" folder, with the code below**

Remind that the working directory should be the root folder when running this program.

```python 
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

factor_path = 'data/factors/'
return_path = 'data/return/'
result_dir = 'result/'

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
            combine = pd.merge(return_series_nonzero, factor_series_nonzero, 
                               how='inner', left_index=True, right_index=True)
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
print(IC_df)
os.chdir(result_dir)
IC_df.to_excel("RankIC.xlsx")
```

|              | ICmean   |  ICstd    |  ICIR | AvgSample |
|:--:|:--:|:--:|:--:|:--:|
|cisc_3(select_ability) |  0.089101 | 0.169185 | 0.526649  |  1038.46|
|cisc_3(fund_share)  |  -0.096019 | 0.091791 | -1.046066 |   1279.50|
|cisc_3(max_drawdown) | -0.062207 | 0.200550| -0.310182  |  1270.88|
|cisc_3(institute_prop) | 0.077228 | 0.117013 | 0.659989 |   1133.38|
|cisc_3(annual_return) |  0.092428 | 0.146760 | 0.629790  |  1045.56 |

### Portfolio test

First of all, according to the factor value of the current period, the stocks are divided into five equal weight quantile portfolios, and the returns of the portfolios in the next period are denoted as \(R_1, R_2, \cdots, R_5\) We judge the effectiveness of the factor according to the return \(R_1\sim R_5\) of the long-short combination. If the return of the long-short combination is significantly different from zero, it indicates that the factor is effective. The higher the combination sharpe ratio, the more effective the factor is. However, it is worth noting that since the quantile array method only considers the returns of the two extreme combinations of long and short, and ignores the relevant information of the middle combinations of quantiles, there may be some limitations in the characterization of factor effectiveness.

![fig7](/img/cisc_3_7.png)

Add the following code to `main.py`

```python
# Quantile portfolio test
def quantile_test(IC_df):
    target_dict = {}
    for i in range(len(factor_name_list)):
        # Build a T*5 matrix
        target = np.zeros([quarter_return.shape[0], 5])
        factor_name = factor_name_list[i]
        factor_df = factor_df_list[i][factor_name]
        IC_mean = IC_df.loc[factor_name, 'ICmean']
        for day in range(factor_df.shape[0]):
            factor_value = pd.DataFrame({'factor_value': factor_df.iloc[day, :].tolist()})
            factor_value = factor_value[factor_value != 0]
            return_value = pd.DataFrame({'return_value': quarter_return.iloc[day, :].tolist()})
            return_value = return_value[return_value != 0]
            if IC_mean > 0:
                quantile = pd.qcut(factor_value['factor_value'], q=5, labels=False)
            else:
                quantile = pd.qcut(-factor_value['factor_value'], q=5, labels=False)
            quantile_df = pd.concat([factor_value['factor_value'], return_value['return_value'], quantile], axis=1)
            quantile_df.index = factor_df.columns
            quantile_df.columns = [factor_name, 'return_value', 'quantile']
            quantile_df.dropna(inplace=True)
            quantile_df = quantile_df.sort_values(by='quantile', ascending=True)
            target[day] = quantile_df.groupby('quantile')['return_value'].mean().tolist()

        tmp = pd.DataFrame(target, columns=['Group 1', 'Group 2',
                                                       'Group 3', 'Group 4', 'Group 5'], index=quarter_return.index)
        target_dict[factor_name] = (1 + tmp).cumprod()
    return target_dict

# Plot portfolio test result
target = quantile_test(IC_df)
for factor in factor_name_list:
    df = target[factor]
    dates = [datetime.strptime(date_str, '%Y/%m/%d').date() for date_str in df.index.tolist()]
    for group in df.columns:
        plt.subplot(1, 2, 1)
        plt.plot(dates, df[group], label=group)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.subplot(1, 2, 2)
        plt.bar(group, np.mean(df[group]), label=group)
    plt.legend()
    plt.title(factor)
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.show()
```

![fig2](/img/cisc_3_2.png)

![fig3](/img/cisc_3_3.png)

![fig4](/img/cisc_3_4.png)

![fig5](/img/cisc_3_5.png)

![fig6](/img/cisc_3_6.png)

### Factor composition

Add the following code to `main.py`

```python
def composite(IC_df):
    weight = IC_df.iloc[:, 0] / sum(IC_df.iloc[:, 0])
    norm_factor_list = []
    for i in range(len(factor_name_list)):
        factor_name = factor_name_list[i]
        factor_df = factor_df_list[i][factor_name]
        norm_factor = factor_df.copy()
        nonzero = norm_factor[norm_factor != 0]
        norm_factor[norm_factor != 0] = nonzero.apply(lambda x: x.rank(pct=True), axis=1)
        norm_factor_list.append({factor_name: pd.DataFrame(norm_factor, dtype=float)})

    norm_df = sum([weight[i] * norm_factor_list[i][factor_name_list[i]] for i in range(len(factor_name_list))])

    return norm_df

norm_df = composite(IC_df)
composed = summarized_IC(['Composite'], [{'Composite': norm_df}])
print(composed)
```

|              | ICmean   |  ICstd    |  ICIR | AvgSample |
|:--:|:--:|:--:|:--:|:--:|
| Composite |  0.130709 | 0.102025 | 1.281154 | 1280.56 |

### Reference

- [Research on heavy stock information under fund selection](/pdf/cisc_3_1.pdf)
- [Deep Analysis of Major Factors: Value Reproduction](/pdf/cisc_3_2.pdf)