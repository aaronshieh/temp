import numpy as np

def format_data(df):
    df[['Open','High','Low','Close']] = df[['Open','High','Low','Close']].astype(dtype='float64', copy=True)
    df['Market Cap'] = df['Market Cap'].astype(dtype='int64', copy=True)
    df['Volume'] = df['Volume'].apply(lambda x: int(x) if x.isnumeric() else np.nan)