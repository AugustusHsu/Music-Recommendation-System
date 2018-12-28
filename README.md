# Music-Recommendation-System

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

 - 建立不重複track_id的track_name,track_name資料表
 - 建立tag的資料表
 - 
## Modeling
## Evaluation