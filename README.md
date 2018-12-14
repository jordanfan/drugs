# drugs
IEOR 142 final project 

# Description of Files 
* combine_dataset.ipynb  - combine druglib, drugcom, drug names, and drug prices together into one dataframe, creates drug_merged.csv
* drug_reduced.ipynb - combine review columns in druglib, drop entries with no conditions, price, and drugs that have less than 5 entries, creates drugs_reduced_reduced.csv
* missing_data.Rmd - combine effectiveness and side effects categories, calculate sentiment scores of reviews, predict missing effectiveness and side effects severity in drugCom data using LDA, random forest, and neural network, creates drugs_final.csv
* IEOR project Full .R - CART and random forest models using different features and with/without binned price data and bootstrap CI
* cluster.Rmd - Clustering by different effectiveness categories and predicting using random forest and bootstrap CI and some exploratory data analysis for binned price data 
* neural_net.Rmd - neural network models with different activation functions and number of layers on binned/unbinned data,
* plots.ipynb - futher exploratory data analysis 
