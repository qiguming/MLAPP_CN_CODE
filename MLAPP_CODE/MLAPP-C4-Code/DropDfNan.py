import numpy as np
def drop_df_nan(df_x, df_y):
    """ 
    删除样本中那些包含nan值的样本（列）
    Input:
        x: dataframe shape=(dim,n_samples)   
        y: dataframe shape=(1,n_sampels)   
    Output:
        df_x,df_y
    """
    nan_x_index = np.where(np.isnan(df_x))       # 返回np.nan值所在的行和列 （rows_index，columns_index）
    nan_y_index = np.where(np.isnan(df_y)) 
    nan_rows = np.union1d(nan_x_index[0], nan_y_index[0])   # 求取两个并集
    nan_cols = np.union1d(nan_x_index[1], nan_y_index[1])
    #print(list(df_x.columns[nan_cols]),type(list(df_x.columns[nan_cols])))
    if len(nan_cols) != 0:            # 如果返回的列索引值非空，即存在某些列存在np.nan值，我们进行删除,注意drop不改变元对象
        df_x = df_x.drop(columns = nan_cols)
        df_y = df_y.drop(columns = nan_cols)
    return df_x, df_y
    