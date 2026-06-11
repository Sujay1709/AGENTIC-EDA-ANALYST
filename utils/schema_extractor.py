"""
Dataset loading and schema extraction.

Loads any common tabular file (CSV/TSV/Excel/JSON), then infers column types so
date columns and "hidden numeric" columns (e.g. "1,234", "$5", "12%") are
recognised instead of being lumped in with free-text categoricals. Only the
structural blueprint (never raw rows) is passed downstream to the LLM.
"""
import os
import warnings
import pandas as pd

# A column is only re-typed if at least this fraction of its non-null values
# convert cleanly. Keeps free-text columns from being mistaken for dates/numbers.
_COERCE_THRESHOLD = 0.8
# How many non-null values to test before committing to a full datetime parse.
_DATE_SAMPLE = 50


def _read_csv_robust(filepath: str, sep=None) -> pd.DataFrame:
    """
    Read a delimited text file, tolerating unknown encoding and delimiter.

    sep=None with engine='python' lets pandas sniff the delimiter. We try utf-8
    first and fall back to latin-1, which decodes any byte sequence.
    """
    read_kwargs = {"sep": sep, "engine": "python"} if sep is None else {"sep": sep}
    for encoding in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(filepath, encoding=encoding, **read_kwargs)
        except UnicodeDecodeError:
            continue
    # Last resort: let pandas surface any encoding-related error clearly.
    return pd.read_csv(filepath, **read_kwargs)


def load_dataframe(filepath: str) -> pd.DataFrame:
    """
    Load a dataset from any supported file type into a DataFrame.

    Supported: .csv, .txt (delimiter sniffed), .tsv (tab), .xlsx/.xls (Excel),
    .json. Raises ValueError for anything else.
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext in (".csv", ".txt"):
        return _read_csv_robust(filepath, sep=None)
    if ext == ".tsv":
        return _read_csv_robust(filepath, sep="\t")
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(filepath)
    if ext == ".json":
        return pd.read_json(filepath)

    raise ValueError(
        f"Unsupported file type '{ext}'. Supported: .csv, .txt, .tsv, "
        f".xlsx, .xls, .json"
    )


def _to_datetime_quiet(values):
    """pd.to_datetime with the noisy 'could not infer format' warning silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(values, errors="coerce")


def _coerce_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert object columns that are mostly parseable dates into datetime dtype.

    A cheap sample of values is tested first so free-text columns are skipped
    before paying for a full element-by-element parse. Only converts when
    >= _COERCE_THRESHOLD of the non-null values parse.
    """
    for col in df.select_dtypes(include="object").columns:
        non_null = df[col].dropna()
        if non_null.empty:
            continue
        # Quick reject: if a small sample isn't date-like, skip the full parse.
        sample = _to_datetime_quiet(non_null.head(_DATE_SAMPLE))
        if sample.notna().mean() < _COERCE_THRESHOLD:
            continue
        parsed = _to_datetime_quiet(df[col])
        if parsed.notna().sum() / len(non_null) >= _COERCE_THRESHOLD:
            df[col] = parsed
    return df


def _coerce_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert object columns that are mostly numbers-as-text into numeric dtype.

    Strips thousands separators and currency/percent symbols first, e.g.
    "$1,200" -> 1200, "12%" -> 12. Only converts above the success threshold.
    """
    for col in df.select_dtypes(include="object").columns:
        non_null = df[col].notna().sum()
        if non_null == 0:
            continue
        cleaned = (
            df[col]
            .astype(str)
            .str.replace(r"[,$%\s]", "", regex=True)
            .replace({"": None, "nan": None})
        )
        parsed = pd.to_numeric(cleaned, errors="coerce")
        if parsed.notna().sum() / non_null >= _COERCE_THRESHOLD:
            df[col] = parsed
    return df


def extract_schema(filepath: str):
    """
    Load a dataset and return (df, schema).

    The schema is structural metadata only — never raw rows. Column types are
    inferred (dates and hidden numerics recovered) before the schema is built so
    downstream agents and plots categorise columns correctly.
    """
    df = load_dataframe(filepath)

    # Recover types hidden behind the 'object' dtype. Dates first so a column
    # that parses as a date isn't then stripped and read as a number.
    df = _coerce_datetimes(df)
    df = _coerce_numerics(df)

    datetime_columns = df.select_dtypes(include="datetime").columns.tolist()

    schema = {
        "filepath": filepath,
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percent": (
            ((df.isnull().sum() / len(df)) * 100).round(2).to_dict()
            if len(df) else {c: 0.0 for c in df.columns}
        ),
        "sample_values": df.head(3).astype(str).to_dict(orient="list"),
        "numeric_columns": df.select_dtypes(include="number").columns.tolist(),
        "categorical_columns": df.select_dtypes(include="object").columns.tolist(),
        "datetime_columns": datetime_columns,
    }

    return df, schema
