import pandas as pd
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).parent.parent.parent
RAW_DATA_DIR = BASE_DIR / "Data" / "Raw Data"
MACRO_DATA_DIR = BASE_DIR / "Data" / "Macro Data"
MACRO_DATA_DIR.mkdir(parents=True, exist_ok=True)

print("="*60)
print("MACRO DATA CLEANING")
print("="*60)

# Load raw FRED data
print("\n📂 Loading FRED data...")
fred_file = RAW_DATA_DIR / "macro_fred_data.csv"
if not fred_file.exists():
    print(f"❌ File not found: {fred_file}")
    exit(1)

macro_df = pd.read_csv(fred_file, index_col=0, parse_dates=True)
print(f"✓ Loaded: {len(macro_df)} rows, {len(macro_df.columns)} columns")
print(f"  Date range: {macro_df.index.min()} to {macro_df.index.max()}")
print(f"  NaN counts before filling:")
for col in macro_df.columns:
    nan_count = macro_df[col].isna().sum()
    if nan_count > 0:
        print(f"    {col}: {nan_count}")

# Forward-fill FRED data
print("\n🔧 Forward-filling FRED data...")
macro_df = macro_df.ffill(limit=12)  # Max 12 months forward-fill
macro_df = macro_df.bfill(limit=3)   # Backward-fill leading NaNs (max 3 months)

print(f"✓ After filling:")
remaining_nans = macro_df.isna().sum().sum()
print(f"  Remaining NaNs: {remaining_nans}")
if remaining_nans > 0:
    print(f"  By column:")
    for col in macro_df.columns:
        nan_count = macro_df[col].isna().sum()
        if nan_count > 0:
            print(f"    {col}: {nan_count}")

print(f"\nSample data:")
print(macro_df.head(10))

# Save FRED data
fred_output = MACRO_DATA_DIR / "macro_fred_data_filled.csv"
macro_df.to_csv(fred_output)
print(f"\n✅ Saved to: {fred_output}")


# Load raw Yahoo data
print("\n" + "="*60)
print("📂 Loading Yahoo Finance data...")
yahoo_file = RAW_DATA_DIR / "macro_yahoo_data.csv"
if not yahoo_file.exists():
    print(f"❌ File not found: {yahoo_file}")
    exit(1)

yahoo_df = pd.read_csv(yahoo_file, index_col=0, parse_dates=True)
print(f"✓ Loaded: {len(yahoo_df)} rows, {len(yahoo_df.columns)} columns")
print(f"  Date range: {yahoo_df.index.min()} to {yahoo_df.index.max()}")
print(f"  NaN counts before filling:")
for col in yahoo_df.columns:
    nan_count = yahoo_df[col].isna().sum()
    if nan_count > 0:
        print(f"    {col}: {nan_count}")

# Forward-fill Yahoo data
print("\n🔧 Forward-filling Yahoo data...")
yahoo_df = yahoo_df.ffill(limit=30)  # Daily data - max 30 days forward-fill
yahoo_df = yahoo_df.bfill(limit=10)  # Backward-fill leading NaNs

print(f"✓ After filling:")
remaining_nans = yahoo_df.isna().sum().sum()
print(f"  Remaining NaNs: {remaining_nans}")
if remaining_nans > 0:
    print(f"  By column:")
    for col in yahoo_df.columns:
        nan_count = yahoo_df[col].isna().sum()
        if nan_count > 0:
            print(f"    {col}: {nan_count}")

print(f"\nSample data:")
print(yahoo_df.head(10))

# Save Yahoo data
yahoo_output = MACRO_DATA_DIR / "macro_yahoo_data_filled.csv"
yahoo_df.to_csv(yahoo_output)
print(f"\n✅ Saved to: {yahoo_output}")

# Final summary
print("\n" + "="*60)
print("CLEANING COMPLETE")
print("="*60)
print(f"FRED data: {len(macro_df)} rows, {len(macro_df.columns)} columns")
print(f"  Remaining NaNs: {macro_df.isna().sum().sum()}")
print(f"Yahoo data: {len(yahoo_df)} rows, {len(yahoo_df.columns)} columns")
print(f"  Remaining NaNs: {yahoo_df.isna().sum().sum()}")
print(f"\nOutput files:")
print(f"  {fred_output}")
print(f"  {yahoo_output}")
print("="*60)