# Since the Purchase column is not normally distributed
# Hence we will first create bins for the data and then
# divide the data into k-folds using StratifiedKFold

# importing libraries 
import numpy as np
import pandas as pd 
from sklearn import model_selection

# first we create last month purchase comlumn for each customer
def kfolds(df , fold):
    # This function will create bins for the
    # creating "fold" column and fill it with 0
    df['fold'] = 0 
    df['Last_Month_purchase'] = df.groupby('User_ID')['Purchase'].transform('sum')
    
    # calculating number of bins 
    num_bins = np.floor(1 + np.log2(len(df))).astype(int)

    # creating the bins columns
    df['Purchase_Bins'] = pd.cut(df['Last_Month_purchase'] , bins= num_bins , labels=False)

    # initializing the  StratifiedKFold
    kf = model_selection.StratifiedKFold(n_splits=fold)

    # fill the fold colmn
    for f ,(t_ , v_) in enumerate(kf.split(X = df , y = df.Purchase_Bins)):
        df.loc[v_ , 'fold'] = f

    return df

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv('dataset/train.csv')

    df = kfolds(df , 5)

    df.to_csv('dataset/train_fold.csv' , index = False)

