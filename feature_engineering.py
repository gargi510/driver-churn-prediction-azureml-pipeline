import pandas as pd
from sklearn.impute import KNNImputer

def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def convert_dates(df: pd.DataFrame) -> pd.DataFrame:
    df['Reporting_Date'] = pd.to_datetime(
        df['MMM-YY'].apply(lambda x: '/'.join(x.split('/')[1:])),
        format='%m/%y',
        errors='coerce'
    )
    df['Dateofjoining'] = pd.to_datetime(df['Dateofjoining'], dayfirst=True, errors='coerce')
    df['LastWorkingDate'] = pd.to_datetime(df['LastWorkingDate'], dayfirst=True, errors='coerce')
    return df

def knn_impute(df: pd.DataFrame, columns: list, n_neighbors=5) -> pd.DataFrame:
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df[columns] = imputer.fit_transform(df[columns])
    return df

def aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    agg_dict = {
        'Age': 'mean',
        'Gender': lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0],
        'City': lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0],
        'Education_Level': lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0],
        'Income': 'mean',
        'Dateofjoining': 'min',
        'Joining Designation': lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0],
        'Grade': 'mean',
        'Total Business Value': 'mean',
        'Quarterly Rating': 'mean',
        'Reporting_Date': 'max',
        'LastWorkingDate': 'max'
    }
    return df.groupby('Driver_ID').agg(agg_dict).reset_index()

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    rating_inc = (
        df.sort_values('Reporting_Date')
          .groupby('Driver_ID')['Quarterly Rating']
          .apply(lambda x: int((x.diff() > 0).any()))
          .reset_index(name='QuarterlyRating_Increased')
    )
    income_inc = (
        df.sort_values('Reporting_Date')
          .groupby('Driver_ID')['Income']
          .apply(lambda x: int((x.diff() > 0).any()))
          .reset_index(name='Income_Increased')
    )
    target = (
        df.groupby('Driver_ID')['LastWorkingDate']
          .apply(lambda x: int(x.notnull().any()))
          .reset_index(name='Attrition')
    )
    return rating_inc, income_inc, target

def prepare_final_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = convert_dates(df)
    
    # Impute numeric columns
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    df = knn_impute(df, num_cols)
    
    # Impute 'Age' and 'Gender' specifically and round
    df = knn_impute(df, ['Age', 'Gender'])
    df['Gender'] = df['Gender'].round().astype(int)
    df['Age'] = df['Age'].round().astype(int)
    
    agg_df = aggregate_features(df)
    rating_inc, income_inc, target = create_features(df)
    
    final_df = agg_df.merge(rating_inc, on='Driver_ID') \
                     .merge(income_inc, on='Driver_ID') \
                     .merge(target, on='Driver_ID')
    
    final_df['Age'] = final_df['Age'].round(0).astype(int)
    final_df['Grade'] = final_df['Grade'].round(0).astype(int)
    final_df['Quarterly Rating'] = final_df['Quarterly Rating'].round(2)
    
    return final_df

if __name__ == "__main__":
    data_path = 'driver_data.csv'
    df = load_data(data_path)
    processed_df = prepare_final_dataset(df)
    processed_df.to_csv("driver_attrition_ml_table.csv", index=False)
    print(processed_df.dtypes)
    print(processed_df.head())
