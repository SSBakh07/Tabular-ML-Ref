# Tabular Machine Learning Reference

**Note that this code is only for binary classification at the moment. In the future I'll be expanding on this.**

This is a repository for model references to be used in machine learning tabular challenges as a starting point. This repository also doubles as a submission of sorts for [Kaggle's Spaceship Titanic Challenge](https://www.kaggle.com/competitions/spaceship-titanic)

> The data used is from [here](https://www.kaggle.com/competitions/spaceship-titanic). To get started, simply download the data from the competition and place the csv files into the `data` directory.

Steps:

1. Install requirements using the command below:

```
pip install -r requirements.txt
```

2. Download the data from [Kaggle's Space Titanic Challenge](), and then place the csv files into the `data` directory.

3. Run the `data-preprocess.ipynb` notebook for EDA and to generate `preprocessed-data.csv`.
- When using your own data, make sure your data has been *completely* preprocessed before running `model_train.ipynb`.

4. Set the models you want to train in `model_train.ipynb`. Current available options include:
```
- XGBoost

```
- Additionally, you can specify your own cross-validation strategy. See [here](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection) for various strategies.


Feel free to fork this repository to make your own changes!

To-do:
- [ ] EDA for space titanic data
- [ ] Implement logger in `model_train`
- [ ] Utilize `Cabin` column in original data
- [ ] Add CatBoost, LightGBM, and various other GBM methods
- [ ] Add ensemble methods
- [ ] Create visual demo