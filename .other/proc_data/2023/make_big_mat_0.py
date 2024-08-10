#
import pandas as pd


def is_missing(x):
    """Checks if a value is missing."""
    return pd.isna(x)


def match_sth(x, sth):
    """Compares a value to a target with optional missing value handling."""
    if is_missing(x) or is_missing(sth):
        return False
    return x == sth


def del_sth_rows(df, sth, skip_cols=[], max_perc_sth=1.0):
    """Deletes rows based on the presence of a value or missing values.

    Args:
        df: A Pandas DataFrame.
        sth: The target value, or None for missing values.
        skip_cols: A list of column indices to ignore.
        max_perc_sth: The maximum fraction of `sth` allowed per row.

    Returns:
        A new Pandas DataFrame with rows removed.
    """

    avail_cols = len(df.columns) - len(skip_cols)
    rows_to_del = []

    # Collect rows with all missing or matching values
    if is_missing(sth):
        for i in range(df.shape[0]):
            if all(is_missing(v) for v in df.iloc[i, ~skip_cols]):
                rows_to_del.append(i)
    else:
        for i in range(df.shape[0]):
            if all(match_sth(v, sth) for v in df.iloc[i, ~skip_cols]):
                rows_to_del.append(i)

    # Collect rows with more than `max_perc_sth` missing or matching values
    if max_perc_sth < 1.0:
        match_func = is_missing if is_missing(sth) else match_sth
        for i in range(df.shape[0]):
            count = sum(match_func(v, sth) for v in df.iloc[i, ~skip_cols])
            if count / avail_cols > max_perc_sth:
                rows_to_del.append(i)

    # Remove collected rows and return the filtered DataFrame
    return df.drop(rows_to_del)


# Example usage
df = pd.DataFrame({
    "A": [1, 2, None, 3, None, 4],
    "B": [None, None, 2, None, 2, 4],
    "C": [3, 4, None, 5, None, 6],
})

# Delete rows with all missing values in "A"
df_filtered = del_sth_rows(df, None)

# Delete rows with all values equal to 2 in "B"
df_filtered = del_sth_rows(df, 2, skip_cols=["A"])

# Delete rows with more than 50% missing values in any column
df_filtered = del_sth_rows(df, None, max_perc_sth=0.5)

print(df_filtered)
