# functions/core/exploration.py
"""
functions.core.exploration

Intent
- Provide deterministic, notebook-friendly exploratory utilities for quick profiling
  of Lightcast job posting datasets before/alongside LLM-based validation.

What’s inside
- analyze_missing_data_detailed():
    Column-wise missingness breakdown that treats multiple “missing-like” tokens
    separately (NaN/None, empty string, 'Unclassified'), prints a readable summary,
    and returns a sortable DataFrame of counts and rates.
- analyze_top_values_table():
    Per-column frequency inspection (top-N + “Others”) with readability-focused
    console output, optional column exclusion, and value truncation for long text.

Design principles
- Pure pandas + stdlib; no LLM calls, no side effects beyond console prints.
- Debug-first output for fast iteration in notebooks and local runs.
- Return structured objects (DataFrame/dict) so results can be reused downstream.
"""

import pandas as pd

def analyze_missing_data_detailed(df):
    """
    Comprehensive missing data analysis with summary statistics.
    """
    print(f"{'='*80}")
    print(f"MISSING DATA ANALYSIS")
    print(f"{'='*80}")
    print(f"Total Rows: {len(df):,}")
    print(f"Total Columns: {len(df.columns)}\n")
    
    stats = []
    
    for col in df.columns:
        # Count different types of missing/invalid values
        null_count = df[col].isna().sum()
        empty_str_count = (df[col] == '').sum()
        unclassified_count = (df[col] == 'Unclassified').sum()
        
        # Total problematic values
        total_missing = null_count + empty_str_count + unclassified_count
        valid_count = len(df) - total_missing
        pct_missing = (total_missing / len(df)) * 100
        pct_with_data = (valid_count / len(df)) * 100
        
        stats.append({
            'Column': col,
            'Null/NA': null_count,
            'Empty': empty_str_count,
            'Unclassified': unclassified_count,
            'Total Missing': total_missing,
            'Valid': valid_count,
            'Percent_Missing': pct_missing,
            'Percent_w_Data': pct_with_data
        })
    
    result_df = pd.DataFrame(stats)
    result_df = result_df.sort_values('Total Missing', ascending=False)
    
    # Format for display
    display_df = result_df.copy()
    display_df['Percent_Missing'] = display_df['Percent_Missing'].apply(lambda x: f'{x:.2f}%')
    display_df['Percent_w_Data'] = display_df['Percent_w_Data'].apply(lambda x: f'{x:.2f}%')
    
    print(display_df.to_string(index=False))
    
    # Summary statistics
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    
    # Columns with most missing data
    top_missing = result_df.nlargest(5, 'Total Missing')
    print(f"\n🔴 Top 5 columns with most missing data:")
    for _, row in top_missing.iterrows():
        print(f"   {row['Column']}: {row['Total Missing']:,} ({row['Percent_Missing']:.2f}%)")
    
    # Columns with best data quality
    top_valid = result_df.nsmallest(5, 'Total Missing')
    print(f"\n✅ Top 5 columns with best data quality:")
    for _, row in top_valid.iterrows():
        print(f"   {row['Column']}: {row['Valid']:,} ({row['Percent_w_Data']:.2f}%)")
    
    # Columns with all valid data
    clean_cols = result_df[result_df['Total Missing'] == 0]
    print(f"\n✨ Columns with 100% valid data: {len(clean_cols)}")
    if len(clean_cols) > 0:
        print(f"   {', '.join(clean_cols['Column'].tolist())}")
    
    # Overall statistics
    total_cells = len(df) * len(df.columns)
    total_missing_cells = result_df['Total Missing'].sum()
    overall_pct_missing = (total_missing_cells / total_cells) * 100
    overall_pct_valid = 100 - overall_pct_missing
    
    print(f"\n📊 Overall Statistics:")
    print(f"   Total cells: {total_cells:,}")
    print(f"   Missing cells: {total_missing_cells:,} ({overall_pct_missing:.2f}%)")
    print(f"   Valid cells: {total_cells - total_missing_cells:,} ({overall_pct_valid:.2f}%)")
    
    return result_df


def analyze_top_values_table(df, top_n=10, exclude_cols=None, max_value_width=80):
    """
    Analyze top N values in clean table format with Others row.
    Value column is left-aligned for better readability.
    """
    if exclude_cols is None:
        exclude_cols = ['ID', 'URL']
    
    print(f"{'='*120}")
    print(f"TOP {top_n} VALUES ANALYSIS (+ Others)")
    print(f"{'='*120}")
    print(f"Total Rows: {len(df):,}\n")
    
    results = {}
    
    for col in df.columns:
        if col in exclude_cols:
            continue
        
        # Get value counts
        value_counts = df[col].value_counts(dropna=False)
        total_count = len(df)
        top_values = value_counts.head(top_n)
        unique_count = df[col].nunique(dropna=False)
        
        print(f"\n{'─'*120}")
        print(f"📊 Column: {col}")
        print(f"   Total Unique Values: {unique_count:,}")
        print(f"{'─'*120}")
        
        # Create DataFrame for display
        display_data = []
        for idx, (value, count) in enumerate(top_values.items(), 1):
            percentage = (count / total_count) * 100
            
            # Handle special values
            if pd.isna(value):
                display_value = "[NaN/Null]"
            elif value == '':
                display_value = "[Empty String]"
            else:
                display_value = str(value)[:max_value_width]  # Truncate at max_value_width
            
            display_data.append({
                'Rank': str(idx),
                'Count': f"{count:,}",
                'Percentage': f"{percentage:.2f}%",
                'Value': display_value
            })
        
        # Add Others row
        top_n_total = top_values.sum()
        others_count = total_count - top_n_total
        others_percentage = (others_count / total_count) * 100
        others_unique = unique_count - len(top_values)
        
        if others_count > 0:
            display_data.append({
                'Rank': '*',
                'Count': f"{others_count:,}",
                'Percentage': f"{others_percentage:.2f}%",
                'Value': f"[Others - {others_unique:,} unique values]"
            })
        
        display_df = pd.DataFrame(display_data)
        
        # Print with left-aligned Value column
        # Set column format specifications
        formatters = {
            'Rank': lambda x: f"{x:>4}",
            'Count': lambda x: f"{x:>7}",
            'Percentage': lambda x: f"{x:>10}",
            'Value': lambda x: f"{x}"  # Left-aligned (default)
        }
        
        # Print header
        print(f"{'Rank':>4}  {'Count':>7} {'Percentage':>10}  {'Value'}")
        
        # Print rows
        for _, row in display_df.iterrows():
            print(f"{row['Rank']:>4}  {row['Count']:>7} {row['Percentage']:>10}  {row['Value']}")
        
        # Coverage
        top_n_coverage = (top_n_total / total_count) * 100
        print(f"\n   📈 Top {top_n} coverage: {top_n_coverage:.2f}%")
        print(f"   📊 Others coverage: {others_percentage:.2f}%")
        
        results[col] = {
            'unique_count': unique_count,
            'top_values': display_data,
            'top_n_coverage': top_n_coverage,
            'others_coverage': others_percentage
        }
    
    print(f"\n{'='*120}\n")
    
    return results
