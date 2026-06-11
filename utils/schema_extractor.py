import pandas as pd


def extract_schema(filepath: str) -> dict:
    """
    
    Reads a CSV file and returns only its structural metadata.
    We never pass raw rows to the LLm - only this blueprint.
    """
    df = pd.read_csv(filepath)
    schema = {
        "filepath": filepath,
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percent": (
            (df.isnull().sum() / len(df)) * 100
        ).round(2).to_dict(),
        "sample_values": df.head(3).to_dict(orient="list"),
        "numeric_columns": df.select_dtypes(
            include="number"
        ).columns.tolist(),
        "categorical_columns": df.select_dtypes(
            include="object"
        ).columns.tolist(),
    }

    return df, schema