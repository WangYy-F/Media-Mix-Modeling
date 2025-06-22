import pandas as pd

def load_data(path):
    df = pd.read_csv(path, parse_dates=['date'])
    df = df.set_index('date')
    return df