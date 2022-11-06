import pandas as pd

import pickle
from datetime import datetime
import joblib

# 2022-10-27 17:02 Update the sampling function to avoid loading entire dataframe.
def load_csv(filename,filepath,column1_as_index=False,truncate=None):
    """
    Load a csv file as a dataframe using specified file path copied from windows file explorer.
    Back slashes in file path will be converted to forward slashes.
    Arguments:
    - filepath (raw string): Use the format r'<path>'.
    - filename (string).
    - colum1_as_index (bool): If true, take the first column as the index. 
        Useful when importing CSV files from previously exported dataframes.

    Returns: dataframe object.
    """
    filename = f'{filepath}/'.replace('\\','/')+filename
    df = pd.read_csv(filename)
    if column1_as_index==True:
        df.set_index(df.columns[0], inplace=True)
        df.index.name = None

    if truncate:
        return df.sample(n=truncate,random_state=0)
    else:
        return df

def save_csv(df,filename,path=None,append_version=False):
    """
    Export dataframe to CSV.
    Parameters:
    - df: Dataframe variable name.
    - filename: Root of the filename.
    - filepath (raw string): Use the format r'<path>'. If None, file is saved in same director.
    - append_version (bool): If true, append date and time to end of filename.
    """
    if path:
        path = f'{path}/'.replace('\\','/')
    if append_version == True:
        filename+=datetime.now().strftime('%Y-%m-%d_%H%M')
    df.to_csv(path+filename+'.csv')
    print('File saved: ',path+filename)


def savepickle(model,filename,path=None,append_version=False):
    """
    Export object as a pickle.
    Parameters:
    - model: Model variable name.
    - filename: Root of the filename.
    - filepath (raw string): Use the format r'<path>'. If None, file is saved in same director.
    - append_version (bool): If true, append date and time to end of filename.
    """
    if path:
        path = f'{path}/'.replace('\\','/')
    if append_version == True:
        filename+=datetime.now().strftime('%Y-%m-%d_%H%M')
    with open (path+filename, 'wb') as fh:
        pickle.dump(model, fh)
    print('File saved: ',path+filename)

def loadpickle(filename,filepath):
    """
    Load a pickled model using specified file path copied from windows file explorer.
    Back slashes in file path will be converted to forward slashes.
    Arguments:
    - filepath (raw string): Use the format r'<path>'.
    - filename (string).
    
    Returns saved object.
    """
    filename = f'{filepath}/'.replace('\\','/')+filename
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model

def joblib_save(model,filename,path=None,append_version=False):
    """
    Export object with joblib.
    Parameters:
    - model: Model variable name.
    - filename: Root of the filename.
    - filepath (raw string): Use the format r'<path>'. If None, file is saved in same director.
    - append_version (bool): If true, append date and time to end of filename.
    """
    if path:
        path = f'{path}/'.replace('\\','/')
    if append_version == True:
        filename+=datetime.now().strftime('%Y-%m-%d_%H%M')
    with open (path+filename, 'wb') as fh:
        joblib.dump(model, fh)
    print('File saved: ',path+filename)


def joblib_load(filename,filepath):
    """
    Load a pickled model using specified file path copied from windows file explorer.
    Back slashes in file path will be converted to forward slashes.
    Arguments:
    - filepath (raw string): Use the format r'<path>'.
    - filename (string).
    
    Returns saved object.
    """
    filename = f'{filepath}/'.replace('\\','/')+filename
    loaded_model = joblib.load(open(filename, 'rb'))
    return loaded_model

# convert dates from string to datetime objects
def date_columns(df,date_column='fl_date',format='%Y-%m-%d'):
    """ 
    Take the dates in a dateframes to create new columns:
        _date_standard: Datetime data 
        _year
        _month
    Parmaters:
    - df: Dataframe.
    - date_column: Name of the column containing the date strings.
    - Format: Original date format in the dateframe. Default: '%d.%m.%Y'
    
    Make sure to do the following import: 
    from datetime import datetime
    """

    date_column=str(date_column)
    
    # df[str(date_column+'_year')] = pd.to_datetime(df[date_column],
    #     format='%d.%m.%Y')
    date = pd.to_datetime(df[date_column],
        format=format)

    df[str(date_column+'_standard')] = date
    df[str(date_column+'_year')] = date.dt.year
    df[str(date_column+'_month')] = date.dt.month
    return df

def compare_id(df1, df1_column, df2, df2_column,print_common=False,print_difference=True):
    """
    Print the number of common values and unique values between two dataframe columns.
    
    """
    df1_values = df1[df1_column].unique()
    df2_values = df2[df2_column].unique()
    common_values = set(df1_values) & set(df2_values)
    if len(df1_values) > len(df2_values):
        different_values = set(df1_values) - set(df2_values)
        print(f'Proper subset = {set(df2_values) < set(df1_values)}')
    else:
        different_values = set(df2_values) - set(df1_values)
        print(f'Proper subset = {set(df1_values) < set(df2_values)}')
    print('Unique values in df1:',len(df1_values))
    print('Unique values in df2:',len(df2_values))
    print('Number of common values between df1 and df2:',len(common_values))
    print('Number of different values between df1 and df2:',len(different_values))
    if print_common == True:
        print('Values in common:',common_values)
    if print_difference == True:
        print('Different values:',different_values)
    
# function that prints null values

def explore(df,id=0,print_n_unique=False, printValues=False):
    """
    Explore dataframe data and print missing values.
    Parameters:
    - df: Dataframe.
    - id: Column number or name with the primary IDs. Default is zero.
    - print_n_unique (bool): If the number of unique values in the first column doesn't match 
        the number of rows in the df, print the number of unique values in each column to see if 
        there's another column that might serve as a unique id.
    """

    if (id==False) & (id !=0):
        pass
    elif isinstance(id,int):
    # if type(id)==int:
        print(f'Unique IDs: {len(set(df.iloc[:,0]))}. # of rows: {df.shape[0]}. Match: {len(set(df.iloc[:,0]))==df.shape[0]}')
    else:
        print(f'Unique IDs: {len(set(df[id]))}. # of rows: {df.shape[0]}. Match: {len(set(df[id]))==df.shape[0]}')
    
    # if the number of unique values in the first column doesn't match the number of rows in the df,
    # print the number of unique values in each column to see if there's another column that migh
    # serve as a unique id.
    if (print_n_unique==True):
        if len(set(df.iloc[:,0])) !=df.shape[0]: 
            for column in df.columns:
                print(len(df[column].value_counts()),'\t', column)
    
    # count amount of missing values in each column
    total = df.isnull().sum().sort_values(ascending=False) 
    # % of rows with missing data from each column
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False) 

    # create a table that lists total and % of missing values starting with the highest
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent']) 

    if (printValues == True):
        # extract the names of columns with missing values
        cols_with_missing = missing_data[missing_data.Percent > 0].index.tolist()
        print(df.dtypes[cols_with_missing])

    print(f'')
    return missing_data

def drop_features(df,threshold=100, show_update=True):
    """
    Drop columns in a dataframe with null values above the specified threshold.
    Parameters:
    - df: Dataframe.
    - threshold (float): Float between 0 and 100. 
        Threshold of % null values over which columns will be dropped.
    - show_update: If true, show missing values for the updated dataframe
        (calls the custom function explore)
    """ 
    
    # count amount of missing values in each column
    total = df.isnull().sum().sort_values(ascending=False) 
    # % of rows with missing data from each column
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False) 

    # create a table that lists total and % of missing values starting with the highest
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent']) 

    filter = missing_data['Percent'] > threshold
    to_drop = missing_data[filter].index.tolist()
    df.drop(to_drop, axis=1, inplace=True)
    print(f'Threshold of percentage values for dropping columns: {threshold}')
    print(f'Columns dropped: {to_drop}')
    if show_update == True:
        return explore(df,id=0,print_n_unique=False, printValues=False)
