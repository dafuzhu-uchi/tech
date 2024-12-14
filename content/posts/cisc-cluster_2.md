+++
title = '2010-2023 Market Resumption'
date = 2023-11-23T01:11:46+08:00
draft = false
tags = ["intern", "cisc"]
+++

### Introduction

*Figure 1, 2010-2023 five indexes relative to the Wind All China A Index*

![fig1](/img/cisc_2_1.png)

*Source: Wind*

We use K-means algorithm to cluster the daily return data of 28 industry indexes, reduce dimension into five indexes, and do equal weight average of the relative net worth of each industry included. The industry distribution covered by each index is as follows:

- Index 1: Non-ferrous metals, Basic chemical industry, Building materials, Light industrial manufacturing, Machinery, Agriculture, Forestry, Animal husbandry and fishery, Banking, Communications, Media
- Index 2: Steel, Retail, Transportation
- Index 3: Home appliances, Food and beverage, Electronic components, Computers
- Index 4: Petroleum and petrochemicals, Coal, Power and utilities, Construction, Textiles and apparel, Non-bank finance, Real estate
- Index 5: Power equipment, National defense, Automobiles, Catering and tourism, Medicine

From the overall trend point of view, Index 3 and 5 outperform the market for a long time, Index 1 is close to the market, and Index 2 and 4 are weak. Index 3 and 5 have obvious positive correlation, and Index 2 has negative correlation.

### Clustering

Raw data: [Index performance](/raw_data/cisc_2(index_performance).xlsx)

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_excel("cisc_2(index_performance).xlsx")
indexes = data.iloc[:,2:]
back = indexes.iloc[1:,:].reset_index(drop=True)
front = indexes.iloc[:-1,:].reset_index(drop=True)
rate = (back - front) / front

rate.loc[-1] = 1
rate.index = rate.index + 1  # Shift the index by 1 to make room for the new row
rate.sort_index(inplace=True)
```

Use K-means algorithm after averaging the daily returns, and set \(k=5\).

```
code = {
    'CI005001.WI': '石油石化',
    'CI005002.WI': '煤炭',
    'CI005003.WI': '有色金属',
    'CI005004.WI': '电力及公用事业',
    'CI005005.WI': '钢铁',
    'CI005006.WI': '基础化工',
    'CI005007.WI': '建筑',
    'CI005008.WI': '建材',
    'CI005009.WI': '轻工制造',
    'CI005010.WI': '机械',
    'CI005011.WI': '电力设备',
    'CI005012.WI': '国防军工',
    'CI005013.WI': '汽车',
    'CI005014.WI': '商贸零售',
    'CI005015.WI': '餐饮旅游',
    'CI005016.WI': '家电',
    'CI005017.WI': '纺织服装',
    'CI005018.WI': '医药',
    'CI005019.WI': '食品饮料',
    'CI005020.WI': '农林牧渔',
    'CI005021.WI': '银行',
    'CI005022.WI': '非银行金融',
    'CI005023.WI': '房地产',
    'CI005024.WI': '交通运输',
    'CI005025.WI': '电子元器件',
    'CI005026.WI': '通信',
    'CI005027.WI': '计算机',
    'CI005028.WI': '传媒'
}

df_code = pd.DataFrame(list(code.items()),columns=['Code','Industry'])

# Model
rate_mean = pd.DataFrame({'mean': np.mean(rate,axis=0)})
# rate_var = pd.DataFrame({'var': np.var(rate,axis=0)})
# features = pd.merge(rate_mean,rate_var,left_index=True,right_index=True)

np.random.seed(42)
k = 5
kmeans = KMeans(n_clusters=k)
kmeans.fit(rate_mean)
# kmeans.fit(features)

labels = kmeans.labels_
centers = kmeans.cluster_centers_

df_code['Label'] = labels + 1
```

Visualize the result of clustering.

```
# Plot the data points with color-coded clusters
plt.figure(figsize=(10,6), dpi=500)
plt.scatter(features['mean'], features['var'], c=labels, cmap='viridis')
plt.xlabel('Mean')
plt.ylabel('Variance')
plt.title('K-means Clustering')
plt.legend()
plt.show()
```

![fig6](/img/cisc_2_6.png)


Then, calculate the cummulative return of five indices.

```
cum = indexes / indexes.iloc[0,:]

idx_1_raw = cum.iloc[:,list(np.where(labels == 0)[0])]
idx_1 = np.mean(idx_1_raw,axis=1)

idx_2_raw = cum.iloc[:,list(np.where(labels == 1)[0])]
idx_2 = np.mean(idx_2_raw,axis=1)

idx_3_raw = cum.iloc[:,list(np.where(labels == 2)[0])]
idx_3 = np.mean(idx_3_raw,axis=1)

idx_4_raw = cum.iloc[:,list(np.where(labels == 3)[0])]
idx_4 = np.mean(idx_4_raw,axis=1)

idx_5_raw = cum.iloc[:,list(np.where(labels == 4)[0])]
idx_5 = np.mean(idx_5_raw,axis=1)

rate_df = pd.DataFrame(
    {
        'Date': np.array(data.iloc[:,0]),
        'idx_1': idx_1,
        'idx_2': idx_2,
        'idx_3': idx_3,
        'idx_4': idx_4,
        'idx_5': idx_5
    }
)
rate_df
```

Plot the trend of five indices from 2010 to 2023

```
mkt = data.iloc[:,1]
mkt_cum = mkt / mkt[0]

idx_1_ex = idx_1 / mkt_cum
idx_2_ex = idx_2 / mkt_cum
idx_3_ex = idx_3 / mkt_cum
idx_4_ex = idx_4 / mkt_cum
idx_5_ex = idx_5 / mkt_cum

ex_df = pd.DataFrame(
    {
        'Date': np.array(data.iloc[:,0]),
        'idx_1': idx_1_ex,
        'idx_2': idx_2_ex,
        'idx_3': idx_3_ex,
        'idx_4': idx_4_ex,
        'idx_5': idx_5_ex
    }
)

plt.figure(figsize=(10,6), dpi=500)
plt.plot(ex_df['Date'],ex_df['idx_1'],label="idx_1")
plt.plot(ex_df['Date'],ex_df['idx_2'],label="idx_2")
plt.plot(ex_df['Date'],ex_df['idx_3'],label="idx_3")
plt.plot(ex_df['Date'],ex_df['idx_4'],label="idx_4")
plt.plot(ex_df['Date'],ex_df['idx_5'],label="idx_5")
plt.legend()
plt.xlabel("year")
plt.ylabel("excess return")
plt.show()
```

![fig7](/img/cisc_2_7.png)

### Resumption from 2010 to 2015

*Figure 2: The five indexes from 2010 to 2015 relative to the Wind All China A Index*

![fig2](/img/cisc_2_2.png)

*Source: Wind*

The excess returns of Index 3 and Index 5 from 2010 to 2015 relative to the whole market reached 63.9% and 41.8% respectively, ranking the top two among the five indexes. The excess returns of computer and electronic components in Index 3 in this range are 152.3% and 82.4%, respectively, which are in the top five of the 28 CITIC industries. Since 2010, a large number of smart phones have been shipped, and high-tech fields such as electronic computers have ushered in a large increase; In 2013, with the advent of the Internet era, computer, media and other TMT sectors performed prominently; In 2015, the state introduced a number of reform policies and policy guidance, including "Internet +" and "mass entrepreneurship and innovation", which aroused market enthusiasm and rapid development of the computer and electronics industry.

In addition, the excess returns of catering and tourism and medicine in Index 5 are also obvious, at 111.1% and 57.9% respectively, ranking the top five among the 28 industries. After the financial crisis, medicine has become one of the few industries with sustained and stable high performance, and its ROE has basically remained above 10%, so it is favored by investors. Subsequently, various places in the pharmaceutical industry to control the cost of price reduction to a certain extent, resulting in frequent security incidents, to give the pharmaceutical industry a reasonable profit call to increase, since 2013, the performance of pharmaceutical companies gradually picked up, the industry increased by 35.2% throughout the year, ranking in the top of all industries.

### Resumption from 2016 to 2019

*Figure 3. The five indexes in 2016-2019 relative to the Wind All China A Index*

![fig3](/img/cisc_2_3.png)

*Source: Wind*

During the period from 2016 to 2019, the excess return of Index 3 relative to the whole market reached 57.0%. Among them, the excess income of food and beverage and home appliances is the largest, respectively 151.2% and 88.9%, ranking the top two in all industries. 

In 2016-2017, a resonance occurred both at home and abroad in fundamental sectors. Also, the supply-side reform, which caused the nominal growth rate of China's economy to rebound. Corporate profits have improved, and traditional industries such as food and beverage have led to catalyze a round of "slow bull" market. Historical data show that the growth rate of household income has a lagging effect on the growth rate of nominal GDP. The data of the first and second quarters of 2017 show that the growth rate of resident income has rebounded significantly, and the consumption sector led by food and home appliances has increased significantly, reflecting the important logic of the large consumption cycle. 

In 2018, the China-US trade friction began, the market fell across the board, from the style of the blue-chip market against the decline, the food and beverage industry cash flow is stable, the asset-liability ratio is low, and the ability to fight the economic downturn is strong. 

In 2019, the overfall of the previous year was repaired, and the industry showed a clear two-wheel drive market of consumption and technology, with food and beverage, home appliances and computers rising 25.8%, 21.5% and 11.1%, respectively, ranking among the few positive excess industries.

### Resumption from 2020 to 2021

*Figure 4, 2020-2021 five indexes relative to the Wind All China A Index*

![fig4](/img/cisc_2_4.png)

*Source: Wind*

Between 2020 and 2021, Indices 3 and 5 will have outperformed returns relative to the whole market of 3.5% and 35.1%, respectively. The other indexes underperformed the market, with Index 4 underperforming the market by 16%. Index 5 rose significantly, including the power equipment and automotive industries rose 91.8% and 31.5% respectively. A new round of energy reform is the theme of this stage, the use of greener and cleaner electric energy to gradually replace fossil energy has become the trend of The Times, coupled with China's "carbon neutral" goal and the energy policy proposed by the United States, power equipment and new energy vehicles have developed rapidly.

In addition, Index 4 non-bank finance, real estate fell by more than 37%, the largest decline in all industries. During this period, interest rates fell, the Federal Reserve kept interest rates unchanged from 0 to 0.25%, and the global pattern of ultra-low interest rates continued. With the emergence of ultra-low interest rates, the yield of fixed income products has declined, making it less attractive to invest, and the profits of non-bank financial institutions that originally took fixed income products as the main investment object have been negatively affected. At the same time, a number of real estate enterprises exploded. In the case of the policy of "Houses are for living, not for speculation; policies should be tailored to each city.", the real estate financing continues to be locked. The multiple pressures of land restriction, sales pressure, and financing obstruction have led to the severing of the capital chain of many housing enterprises, which has led to debt default and bankruptcy reorganization.

### From 2022 to present

*Figure 5. The five indexes have all A relative to the Wind All China A Index since 2022*

![fig5](/img/cisc_2_5.png)

*Source: Wind*

From the beginning of 2022 to now, Index 4 has reached a higher excess return of 19% relative to the whole market, and the trend of other indexes has stabilized. In the index 4, the excess of coal, petroleum and petrochemical is 55.6% and 26.2%, ranking the top two in the industry. In February 2022, the Russia-Ukraine conflict broke out, and the spot price of coal market showed high volatility due to the combination of factors such as rising international energy prices and landing of new energy facilities. In addition, the central government has introduced high-quality development measures for petroleum and petrochemical industries to help the reform of traditional fossil energy and promote the structural transformation of the petroleum and petrochemical industries.