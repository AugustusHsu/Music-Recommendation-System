# Music-Recommendation-System

https://hackmd.io/HXI6YoKcTfedJQ-y50Q9yQ?both

## Environment
python 3.6
tensorflow-gpu 1.12
numpy 1.15.4
pandas 0.23.4
wget 3.2
pylast 2.4.0

## Dataset
Last.fm 1k dataset
http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-1K.tar.gz
## Analysis
- Dataset : userid-timestamp-artid-artname-traid-traname.tsv
- Data feuture : 
- **Example**

userid  |      timestamp     |                artid               | artname |traid|                 traname
-----------|-------------------:|-----------------------------------:| -------:|----:|-----------------------------------------:
user_000001|2009-05-04T23:08:57Z|f1b1cf71-bd35-4e99-8624-24a6e15f133a|Deep Dish| NaN |Fuck Me Im Famous (Pacha Ibiza)-09-28-2007

- **Original Data Analysis**

describe|   userid  |      timestamp     |                artid               | artname |traid|                 traname
:-------|-----------|-------------------:|-----------------------------------:| -------:|----:|-----------------------------------------:
count   |19098853|19098853|18498005|19098853|16936134|19098841
unique  |992|17454730|107295|173921|960402|1083471
maxfreq|183103|248|115099|115099|3991|17561

- 去掉有NaN值的row

describe|   userid  |      timestamp     |                artid               | artname |traid|                 traname
:-------|-----------|-------------------:|-----------------------------------:| -------:|----:|-----------------------------------------:
count   |16936134|16936134|16936134|16936134|16936134|16936134
unique  |992|15631466|83905|81751|960402|693231
maxfreq |172042|193|111488|111488|3991|14908

- 取前六個月的資料

describe|   userid  |      timestamp     |                artid               | artname |traid|                 traname
:-------|-----------|-------------------:|-----------------------------------:| -------:|----:|-----------------------------------------:
count   |3052646|3052646|3052646|3052646|3052646|3052646
unique  |923|2757174|49603|48747|465798|353529
maxfreq |48822|193|31412|31412|2559|2635

- 取前三個月的資料

describe|   userid  |      timestamp     |                artid               | artname |traid|                 traname
:-------|-----------|-------------------:|-----------------------------------:| -------:|----:|-----------------------------------------:
count   |1645146|1645146|1645146|1645146|1645146|1645146
unique  |902|1484806|39084|38493|339118|264073
maxfreq |24413|193|13363|13363|1303|1346

- 取前兩個月的資料

describe|   userid  |      timestamp     |                artid               | artname |traid|                 traname
:-------|-----------|-------------------:|-----------------------------------:| -------:|----:|-----------------------------------------:
count   |1113398|1113398|1113398|1113398|1113398|1113398
unique  |886|1009962|33311|32849|270248|214317
maxfreq |20311|160|7828|7828|974|990

- 取前一個月的資料

describe|   userid  |      timestamp     |                artid               | artname |traid|                 traname
:-------|-----------|-------------------:|-----------------------------------:| -------:|----:|-----------------------------------------:
count   |588774|588774|588774|588774|588774|588774
unique  |858|541565|24978|24690|181590|148188
maxfreq |13530|160|5117|5117|654|660

 - 分析user聽歌次數
![](https://i.imgur.com/SixHrFS.png)
 - 建立不重複歌單(Session_time = 20 minutes)
 
 \ | 6-month | 3-month | 2-month | 1-month
 :---|---:|---:|---:|---:
 count|448074|325614|259027|173327
 
 - 提除沒有滿6個session的session list(針對一個月)
     - user數量:742
 
 - 提除沒有Tag的Song(針對一個月)
     - 數量:130009
     
 - 建立不重複的Tag
     - 數量:16141

 - 建立Song Data和Tag Data
     - Song Data:1119883
     - OK Song Data:457051
     - OK Tag Data:457051
 - 將song和tag轉換成one-hot
## 建立模型
- [ ] ~~One-Hot Encoding~~
    - ~~將Data裡面的資料轉成OneHot的格式~~
- [x] Embedding Layer
    - 使用embedding_lookup，可直接使用index來當input
- [x] Bi-GRU Layer
    - 使用bidirectional_dynamic_rnn來實作
- [x] Attention Layer
    - 建立weight和雙向的rnn進行矩陣相乘
    - concat起來
- [x] Fully Connection Layer
- [x] Output Layer
- [ ] grad_clip
    - 尚未實作
## 評估模型
- [ ] Top_N
    - 目前只有實作top_1
- [ ] 

## 執行test_model.py即可
