# functions/core/processing.py
"""
functions.core.processing

Intent
- Deterministic, LLM-agnostic data processing utilities used across the Lightcast
  Data Quality exploration pipelines.

What’s inside
- clean_string_columns_robust():
    Normalize whitespace across string columns (newlines/tabs/unicode spaces),
    collapse repeated whitespace, and return per-column uniqueness reduction stats.
- row_to_json(), row_to_json_by_id():
    Extract a single record as JSON for debugging/traceability (row index or ID lookup),
    with optional null-handling and column exclusion.

Design principles
- Pure pandas + stdlib (no external services).
- Debug-first: human-readable outputs and safe defaults.
- Reproducible: deterministic transforms and explicit knobs.

Typical usage
- Pre-clean raw/structured Lightcast fields before comparisons.
- Quickly inspect a problematic JD record by ID when investigating DQ failures.
"""

import pandas as pd
import json
import re

def clean_string_columns_robust(df, columns=None, inplace=False):
    """
    Robust string cleaning that handles all types of whitespace.
    
    Args:
        df: pandas DataFrame
        columns: List of columns to clean
        inplace: If True, modifies original DataFrame
    
    Returns:
        Cleaned DataFrame and statistics
    """
    if not inplace:
        df = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=['object']).columns.tolist()
    
    print(f"{'='*100}")
    print(f"ROBUST STRING CLEANING")
    print(f"{'='*100}\n")
    
    cleaning_stats = []
    
    for col in columns:
        if col not in df.columns:
            continue
        
        unique_before = df[col].nunique()
        
        def clean_value(x):
            if not isinstance(x, str):
                return x
            
            # Remove all types of newlines
            x = x.replace('\r\n', ' ')
            x = x.replace('\n', ' ')
            x = x.replace('\r', ' ')
            
            # Replace tabs with space
            x = x.replace('\t', ' ')
            
            # Replace non-breaking spaces and other unicode spaces
            x = x.replace('\xa0', ' ')  # Non-breaking space
            x = x.replace('\u200b', '')  # Zero-width space
            x = x.replace('\u200c', '')  # Zero-width non-joiner
            x = x.replace('\u200d', '')  # Zero-width joiner
            x = x.replace('\ufeff', '')  # Zero-width no-break space (BOM)
            
            # Normalize multiple spaces to single space
            x = re.sub(r'\s+', ' ', x)
            
            # Strip leading and trailing whitespace
            x = x.strip()
            
            return x
        
        df[col] = df[col].apply(clean_value)
        
        unique_after = df[col].nunique()
        unique_reduced = unique_before - unique_after
        
        cleaning_stats.append({
            'Column': col,
            'Unique_Before': unique_before,
            'Unique_After': unique_after,
            'Unique_Reduced': unique_reduced,
            'Reduction_Pct': f"{(unique_reduced / unique_before * 100):.2f}%" if unique_before > 0 else "0.00%"
        })
        
        if unique_reduced > 0:
            print(f"✅ {col}: {unique_before:,} → {unique_after:,} (-{unique_reduced:,}, {(unique_reduced / unique_before * 100):.2f}%)")
        else:
            print(f"✓  {col}: No change ({unique_after:,} unique)")
    
    stats_df = pd.DataFrame(cleaning_stats)
    
    print(f"\n{'='*100}")
    print(f"Total unique values reduced: {stats_df['Unique_Reduced'].sum():,}")
    
    return df, stats_df

def row_to_json(df, row_idx, exclude_cols=None, pretty=True, null_handling='keep'):
    """
    Convert a specific row from DataFrame to JSON format using row index.
    
    Args:
        df: pandas DataFrame
        row_idx: Integer row index (0, 1, 2, ...)
        exclude_cols: List of columns to exclude from output (default: None)
        pretty: If True, returns formatted JSON string; if False, returns dict (default: True)
        null_handling: How to handle null values - 'keep', 'remove', or 'empty_string' (default: 'keep')
    
    Returns:
        JSON string or dictionary
    """
    if exclude_cols is None:
        exclude_cols = []
    
    # Check if index is valid
    if row_idx < 0 or row_idx >= len(df):
        raise ValueError(f"Row index {row_idx} out of bounds (valid range: 0-{len(df)-1})")
    
    # Get the row
    row = df.iloc[row_idx]
    
    # Convert to dictionary
    row_dict = {}
    for col in df.columns:
        if col in exclude_cols:
            continue
        
        value = row[col]
        
        # Handle null values
        if pd.isna(value):
            if null_handling == 'remove':
                continue
            elif null_handling == 'empty_string':
                value = ''
            else:  # 'keep'
                value = None
        
        row_dict[col] = value
    
    # Return based on pretty flag
    if pretty:
        return json.dumps(row_dict, indent=2, ensure_ascii=False)
    else:
        return row_dict


def row_to_json_by_id(
    df: pd.DataFrame,
    record_id: str,
    *,
    id_col: str = "ID",
    exclude_cols=None,
    pretty: bool = True,
    null_handling: str = "keep",
    strip_id: bool = True,
    allow_partial_match: bool = False,
):
    """
    Convert a specific row from DataFrame to JSON format using an ID value (not row index).

    Args:
        df: pandas DataFrame
        record_id: ID value to lookup (e.g., "025edde88fbcd1130b98e4d381f16b1237774b38")
        id_col: Column name containing the ID (default: "ID")
        exclude_cols: List of columns to exclude from output (default: None)
        pretty: If True, returns formatted JSON string; if False, returns dict (default: True)
        null_handling: How to handle null values - 'keep', 'remove', or 'empty_string' (default: 'keep')
        strip_id: If True, strip whitespace around record_id and df[id_col] values before matching
        allow_partial_match: If True, allow `record_id` to match as substring (useful for debugging)

    Returns:
        JSON string or dictionary

    Raises:
        ValueError if id_col not in df, record_id not found, or multiple matches found
    """
    if exclude_cols is None:
        exclude_cols = []

    if id_col not in df.columns:
        raise ValueError(f"id_col '{id_col}' not found in df.columns")

    if record_id is None or str(record_id).strip() == "":
        raise ValueError("record_id is empty")

    rid = str(record_id)
    if strip_id:
        rid = rid.strip()

    s = df[id_col].astype(str)
    if strip_id:
        s = s.str.strip()

    if allow_partial_match:
        mask = s.str.contains(rid, na=False)
    else:
        mask = s == rid

    hits = df[mask]
    if hits.empty:
        example_ids = df[id_col].astype(str).head(5).tolist()
        raise ValueError(
            f"record_id '{rid}' not found in column '{id_col}'. "
            f"Example IDs: {example_ids}"
        )

    if len(hits) > 1:
        raise ValueError(
            f"record_id '{rid}' matched {len(hits)} rows in '{id_col}'. "
            "Please deduplicate or provide a more specific key."
        )

    row = hits.iloc[0]

    row_dict = {}
    for col in df.columns:
        if col in exclude_cols:
            continue

        value = row[col]

        if pd.isna(value):
            if null_handling == "remove":
                continue
            elif null_handling == "empty_string":
                value = ""
            else:  # keep
                value = None

        row_dict[col] = value

    return json.dumps(row_dict, indent=2, ensure_ascii=False) if pretty else row_dict

