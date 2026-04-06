import zipfile
from io import BytesIO, TextIOWrapper
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# Set this explicitly if you know it:
# "," for comma-separated
# ";" for semicolon-separated
# "\t" for tab-separated
CSV_SEPARATOR = ","


def read_csv_from_zip(zip_file: zipfile.ZipFile, member_name: str) -> pd.DataFrame:
    """
    Read one CSV/TXT file directly from the ZIP into a DataFrame.
    """
    with zip_file.open(member_name) as f:
        text_stream = TextIOWrapper(f, encoding="utf-8", errors="replace")
        df = pd.read_csv(
            text_stream,
            sep=CSV_SEPARATOR,
            dtype=str,
            low_memory=False,
        )
    return df


def read_json_from_zip(zip_file: zipfile.ZipFile, member_name: str) -> pd.DataFrame:
    """
    Read one JSON file directly from the ZIP into a DataFrame.
    """
    with zip_file.open(member_name) as f:
        data = f.read()
        df = pd.read_json(BytesIO(data))
    return df


def read_table_from_zip(zip_file: zipfile.ZipFile, member_name: str) -> pd.DataFrame:
    suffix = Path(member_name).suffix.lower()

    if suffix in [".csv", ".txt"]:
        return read_csv_from_zip(zip_file, member_name)
    elif suffix == ".json":
        return read_json_from_zip(zip_file, member_name)
    else:
        raise ValueError(f"Unsupported file type: {member_name}")


def peek_columns_from_zip(zip_file: zipfile.ZipFile, member_name: str) -> list[str]:
    suffix = Path(member_name).suffix.lower()
    if suffix in [".csv", ".txt"]:
        with zip_file.open(member_name) as f:
            text_stream = TextIOWrapper(f, encoding="utf-8", errors="replace")
            return pd.read_csv(
                text_stream,
                sep=CSV_SEPARATOR,
                dtype=str,
                nrows=0,
            ).columns.tolist()
    elif suffix == ".json":
        df = read_json_from_zip(zip_file, member_name)
        return df.columns.tolist()
    else:
        raise ValueError(f"Unsupported file type: {member_name}")


def canonical_column_order(all_cols: set) -> list[str]:
    base = sorted(c for c in all_cols if c != "source_file")
    return base + ["source_file"]


def collect_canonical_schemas(
    zip_path: Path,
) -> tuple[list[str] | None, list[str] | None]:
    info_cols: set[str] = set()
    status_cols: set[str] = set()

    with zipfile.ZipFile(zip_path, "r") as z:
        members = [m for m in z.namelist() if not m.endswith("/")]

        for member_name in members:
            file_name = Path(member_name).name.lower()
            try:
                if file_name.startswith("info_"):
                    info_cols.update(peek_columns_from_zip(z, member_name))
                elif file_name.startswith("status_"):
                    status_cols.update(peek_columns_from_zip(z, member_name))
            except Exception as e:
                print(f"⚠️ Skipped {member_name} during schema scan: {e}")

    info_canon = canonical_column_order(info_cols) if info_cols else None
    status_canon = canonical_column_order(status_cols) if status_cols else None
    return info_canon, status_canon


def align_to_canonical(
    df: pd.DataFrame,
    canonical_columns: list[str],
    source_file: str,
) -> pd.DataFrame:
    df = df.copy()
    df["source_file"] = source_file
    out = pd.DataFrame()
    for c in canonical_columns:
        if c in df.columns:
            out[c] = df[c]
        else:
            out[c] = pd.NA
    for c in out.columns:
        out[c] = out[c].astype("string")
    return out


def append_df_to_parquet(df: pd.DataFrame, parquet_path: Path, writer=None):
    """
    Append a DataFrame to a Parquet file using a ParquetWriter.
    """
    table = pa.Table.from_pandas(df, preserve_index=False)

    if writer is None:
        writer = pq.ParquetWriter(parquet_path, table.schema, compression="snappy")

    writer.write_table(table)
    return writer


def combine_zip_to_parquet():
    PROJECT_ROOT = Path(__file__).resolve().parent
    zip_path = PROJECT_ROOT / "scraped_data.zip"
    output_info = PROJECT_ROOT / "info_appended.parquet"
    output_status = PROJECT_ROOT / "status_appended.parquet"

    for out_path in (output_info, output_status):
        if out_path.exists():
            out_path.unlink()

    info_canon, status_canon = collect_canonical_schemas(zip_path)

    info_writer = None
    status_writer = None

    info_rows = 0
    status_rows = 0
    info_files = 0
    status_files = 0

    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            members = [m for m in z.namelist() if not m.endswith("/")]

            for member_name in members:
                file_name = Path(member_name).name.lower()

                try:
                    if file_name.startswith("info_") and info_canon is not None:
                        df = read_table_from_zip(z, member_name)
                        df = align_to_canonical(
                            df,
                            info_canon,
                            Path(member_name).name,
                        )
                        info_writer = append_df_to_parquet(df, output_info, info_writer)
                        info_rows += len(df)
                        info_files += 1
                        print(f"✅ Appended INFO: {member_name} ({len(df)} rows)")

                    elif file_name.startswith("status_") and status_canon is not None:
                        df = read_table_from_zip(z, member_name)
                        df = align_to_canonical(
                            df,
                            status_canon,
                            Path(member_name).name,
                        )
                        status_writer = append_df_to_parquet(
                            df, output_status, status_writer
                        )
                        status_rows += len(df)
                        status_files += 1
                        print(f"✅ Appended STATUS: {member_name} ({len(df)} rows)")

                except Exception as e:
                    print(f"⚠️ Skipped {member_name}: {e}")

    finally:
        if info_writer is not None:
            info_writer.close()
        if status_writer is not None:
            status_writer.close()

    if info_files:
        print(f"\n✅ Created {output_info}")
        print(f"   Files processed: {info_files}")
        print(f"   Total rows: {info_rows}")
    elif info_canon is None:
        print("\n⚠️ No info_ files found")
    else:
        print("\n⚠️ No info_ rows written (all reads failed?)")

    if status_files:
        print(f"\n✅ Created {output_status}")
        print(f"   Files processed: {status_files}")
        print(f"   Total rows: {status_rows}")
    elif status_canon is None:
        print("\n⚠️ No status_ files found")
    else:
        print("\n⚠️ No status_ rows written (all reads failed?)")


if __name__ == "__main__":
    combine_zip_to_parquet()
