import pandas as pd
import pyarrow as pa

df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})

print(df)

table = pa.Table.from_pandas(df)

path_parquet = "/home/wuch/Downloads/test_py.parquet"

# Write the table to a Parquet file
with pa.OSFile(path_parquet, 'wb') as sink:
    with pa.RecordBatchFileWriter(sink, table.schema) as writer:
        writer.write_table(table)
