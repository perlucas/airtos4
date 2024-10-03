import pandas as pd


# ====================================== Utils functions ======================================

def load_dataset(name, index_name='Date'):
    '''Load the contents of the given CSV stock prices file into a brand new pandas dataset
    :param name: str: The name of the CSV file to load
    :param index_name: str: The name of the index column
    :return: pd.DataFrame: The dataset created
    '''
    df = pd.read_csv(name, parse_dates=True, index_col=index_name)
    df = df.iloc[::-1]
    df = df.reset_index(drop=True)
    return df
