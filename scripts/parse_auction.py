#!/usr/bin/env python3
"""
parse_auction.py - Parse 行將拍賣 Markdown files to CSV

Extracts automobile and motorcycle data from OCR-processed Markdown files
and outputs to automobiles.csv and motorcycles.csv.

Usage:
    python scripts/parse_auction.py [--raw-dir data/raw] [--output-dir data/parsed]
"""

import os
import re
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd

# Brand classifications
AUTOMOBILE_BRANDS = {
    'BENZ', 'BMW', 'AUDI', 'LEXUS', 'PORSCHE', 'VOLVO', 'VW', 'VOLKSWAGEN',
    'MINI', 'TOYOTA', 'HONDA', 'NISSAN', 'MAZDA', 'SUBARU', 'FORD',
    'HYUNDAI', 'KIA', 'MITSUBISHI', 'SUZUKI', 'PEUGEOT', 'JAGUAR',
    'LAND ROVER', 'MG', 'OPEL', 'VW', 'BMW'
}

MOTORCYCLE_BRANDS = {'YAMAHA', 'SYM', 'KYMCO', 'GOGORO', 'PIAGGIO', 'SUZUKI', 'HONDA'}

# Grade ordinal mapping
GRADE_ORDINAL = {
    'A+': 7, 'A': 6, 'B+': 5, 'B': 4, 'C': 3, 'D': 2, 'E': 1, 'N': 0,
    'C.W': 0, 'W': 0
}


def extract_auction_date_from_filename(filename: str) -> Optional[str]:
    """Extract auction date from filename like '2026-06-19_行將競拍結果.md'"""
    match = re.match(r'(\d{4}-\d{2}-\d{2})_行將競拍結果\.md', filename)
    if match:
        return match.group(1)
    return None


def clean_mileage(mileage_str: str) -> tuple:
    """
    Clean mileage string and determine availability.
    Returns (mileage_km, mileage_available)
    
    Rules:
    - 9999999KM / 999999KM -> mileage_available=0, mileage=NULL
    - Otherwise -> clean commas and convert to int
    """
    if not mileage_str or mileage_str.strip() == '':
        return None, 1  # Available but unknown
    
    mileage_str = mileage_str.strip().upper().replace(' ', '')
    
    # Check for unreadable mileage markers
    if mileage_str in ['9999999KM', '999999KM', '9999999', '999999']:
        return None, 0  # Unavailable
    
    # Remove 'KM' suffix and commas
    mileage_str = mileage_str.replace('KM', '').replace(',', '')
    
    try:
        mileage = int(mileage_str)
        if mileage >= 999999:  # Sanity check
            return None, 0
        return mileage, 1
    except ValueError:
        return None, 1


def clean_price(price_str: str) -> int:
    """Clean price string, removing commas and converting to int. Empty=0."""
    if not price_str or price_str.strip() == '':
        return 0
    return int(price_str.strip().replace(',', ''))


def clean_year_month(ym_str: str) -> tuple:
    """
    Parse year-month string like '2024.08' or '2024.08'.
    Returns (year, month) or (None, None) if invalid.
    """
    if not ym_str or ym_str.strip() == '':
        return None, None
    
    parts = ym_str.strip().split('.')
    if len(parts) != 2:
        return None, None
    
    try:
        year = int(parts[0])
        month = int(parts[1])
        if 1990 <= year <= 2030 and 1 <= month <= 12:
            return year, month
    except ValueError:
        pass
    
    return None, None


def parse_markdown_file(filepath: str) -> List[Dict]:
    """
    Parse a single markdown file and extract vehicle records.
    
    Returns list of dicts with fields:
      - source_file, auction_date, number, brand, model, year, month,
        cc, color, transmission, tax, violation, strong_violation,
        price, grade, mileage_km, mileage_available
    """
    records = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    filename = os.path.basename(filepath)
    auction_date = extract_auction_date_from_filename(filename)
    
    # Find the table - look for markdown table pattern
    # Pattern: | header | ... | --- | ... | data | ... |
    lines = content.split('\n')
    
    table_started = False
    header_line = -1
    data_lines = []
    
    for i, line in enumerate(lines):
        # Look for table header with columns: 編號, 廠牌, 型式, etc.
        if '| 編號 |' in line and '| 廠牌 |' in line:
            table_started = True
            header_line = i
            continue
        
        # Skip separator line (|------|...)
        if table_started and re.match(r'\|[\s\-:]+\|', line):
            continue
        
        # Empty line after table
        if table_started and line.strip() == '':
            break
        
        # Data line starts with |
        if table_started and line.strip().startswith('|'):
            data_lines.append(line)
    
    for line in data_lines:
        # Parse: | 3830 | AUDI | A4 | 2014.02 | 1798 | 深灰 | 手自 |  | 0 |  | 63,000 | A | 148386KM |
        cells = [c.strip() for c in line.split('|')[1:-1]]  # Remove empty first/last
        
        if len(cells) < 13:
            continue
        
        try:
            number = cells[0].strip()
            if not number.isdigit():
                continue
            
            brand = cells[1].strip().upper()
            model = cells[2].strip()
            year_month_str = cells[3].strip()
            cc_str = cells[4].strip()
            color = cells[5].strip()
            transmission = cells[6].strip() if len(cells) > 6 else '自'
            tax_str = cells[7].strip() if len(cells) > 7 else ''
            violation_str = cells[8].strip() if len(cells) > 8 else '0'
            strong_violation_str = cells[9].strip() if len(cells) > 9 else '0'
            price_str = cells[10].strip() if len(cells) > 10 else '0'
            grade = cells[11].strip() if len(cells) > 11 else ''
            mileage_str = cells[12].strip() if len(cells) > 12 else ''
            
            # Parse year-month
            year, month = clean_year_month(year_month_str)
            
            # Parse cc
            try:
                cc = int(cc_str.replace(',', '')) if cc_str else 0
            except ValueError:
                cc = 0
            
            # Parse mileage
            mileage_km, mileage_available = clean_mileage(mileage_str)
            
            # Parse prices
            tax = clean_price(tax_str)
            violation = clean_price(violation_str)
            strong_violation = clean_price(strong_violation_str)
            price = clean_price(price_str)
            
            # Default transmission
            if not transmission:
                transmission = '自'
            
            record = {
                'source_file': filename,
                'auction_date': auction_date,
                'number': number,
                'brand': brand,
                'model': model,
                'year': year,
                'month': month,
                'cc': cc,
                'color': color,
                'transmission': transmission,
                'tax': tax,
                'violation': violation,
                'strong_violation': strong_violation,
                'price': price,
                'grade': grade,
                'mileage_km': mileage_km,
                'mileage_available': mileage_available
            }
            
            records.append(record)
            
        except Exception as e:
            print(f"Warning: Failed to parse line in {filename}: {line[:50]}... Error: {e}")
            continue
    
    return records


def classify_vehicle(brand: str, cc: int = 0) -> str:
    """
    Classify vehicle as 'automobile' or 'motorcycle' based on brand and cc.
    
    HONDA appears in both car and motorcycle lists.
    Rule: HONDA with cc < 500 is likely motorcycle, cc >= 500 is car.
    """
    brand_upper = brand.upper()
    
    # Check motorcycle brands (except HONDA which needs cc check)
    for mb in MOTORCYCLE_BRANDS:
        if mb in brand_upper:
            if mb == 'HONDA':
                # HONDA: use cc to distinguish
                if cc > 0 and cc < 500:
                    return 'motorcycle'
                else:
                    return 'automobile'
            return 'motorcycle'
    
    # Check automobile brands
    for ab in AUTOMOBILE_BRANDS:
        if ab in brand_upper:
            return 'automobile'
    
    # Default - check if brand contains motorcycle brand name
    return 'automobile'  # Default to automobile


def process_all_files(raw_dir: str) -> tuple:
    """
    Process all markdown files in directory.
    Returns (automobile_records, motorcycle_records)
    """
    automobile_records = []
    motorcycle_records = []
    
    raw_path = Path(raw_dir)
    md_files = sorted(raw_path.glob('*_行將競拍結果.md'))
    
    print(f"Found {len(md_files)} markdown files to process")
    
    for md_file in md_files:
        records = parse_markdown_file(str(md_file))
        
        for record in records:
            vehicle_type = classify_vehicle(record['brand'], record.get('cc', 0))
            record['vehicle_type'] = vehicle_type
            
            if vehicle_type == 'automobile':
                automobile_records.append(record)
            else:
                motorcycle_records.append(record)
        
        print(f"  {md_file.name}: {len(records)} records")
    
    return automobile_records, motorcycle_records


def write_csv(records: List[Dict], output_path: str):
    """Write records to CSV file."""
    if not records:
        print(f"Warning: No records to write to {output_path}")
        # Create empty file with headers
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'source_file', 'auction_date', 'number', 'brand', 'model',
                'year', 'month', 'cc', 'color', 'transmission', 'tax',
                'violation', 'strong_violation', 'price', 'grade',
                'mileage_km', 'mileage_available'
            ])
            writer.writeheader()
        return
    
    # Ensure consistent field order
    fieldnames = [
        'source_file', 'auction_date', 'number', 'brand', 'model',
        'year', 'month', 'cc', 'color', 'transmission', 'tax',
        'violation', 'strong_violation', 'price', 'grade',
        'mileage_km', 'mileage_available', 'vehicle_type'
    ]
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    
    print(f"Wrote {len(records)} records to {output_path}")


def run_eda(csv_path: str, vehicle_type: str):
    """Run simple EDA on parsed CSV."""
    if not os.path.exists(csv_path):
        print(f"EDA: {csv_path} not found, skipping")
        return
    
    df = pd.read_csv(csv_path)
    
    print(f"\n{'='*60}")
    print(f"EDA Summary: {vehicle_type}")
    print(f"{'='*60}")
    print(f"Total records: {len(df)}")
    print(f"Unique auction dates: {df['auction_date'].nunique()}")
    print(f"Auction date range: {df['auction_date'].min()} to {df['auction_date'].max()}")
    print(f"\nBrands: {df['brand'].nunique()}")
    print(f"Top 10 brands by count:")
    print(df['brand'].value_counts().head(10).to_string())
    
    print(f"\nPrice statistics:")
    print(f"  Min: ${df['price'].min():,}")
    print(f"  Max: ${df['price'].max():,}")
    print(f"  Mean: ${df['price'].mean():,.0f}")
    print(f"  Median: ${df['price'].median():,.0f}")
    
    print(f"\nMileage statistics (where available):")
    mileage_df = df[df['mileage_available'] == 1]['mileage_km']
    if len(mileage_df) > 0:
        print(f"  Min: {mileage_df.min():,} km")
        print(f"  Max: {mileage_df.max():,} km")
        print(f"  Mean: {mileage_df.mean():,.0f} km")
        print(f"  Median: {mileage_df.median():,.0f} km")
    print(f"  Unavailable (mileage_available=0): {(df['mileage_available']==0).sum()}")
    
    print(f"\nGrade distribution:")
    print(df['grade'].value_counts().to_string())
    
    print(f"\nYear distribution:")
    year_df = df[df['year'].notna()]
    if len(year_df) > 0:
        print(f"  Range: {int(year_df['year'].min())} to {int(year_df['year'].max())}")
        print(f"  Mean: {year_df['year'].mean():.1f}")
    
    print(f"\nCC distribution:")
    print(f"  Min: {df['cc'].min()}")
    print(f"  Max: {df['cc'].max()}")
    print(f"  Mean: {df['cc'].mean():.1f}")


def main():
    parser = argparse.ArgumentParser(description='Parse 行將拍賣 Markdown files to CSV')
    parser.add_argument('--raw-dir', default='data/raw',
                        help='Directory containing raw markdown files')
    parser.add_argument('--output-dir', default='data/parsed',
                        help='Directory for output CSV files')
    args = parser.parse_args()
    
    # Convert to absolute paths relative to repo root
    repo_root = Path(__file__).parent.parent
    raw_dir = repo_root / args.raw_dir
    output_dir = repo_root / args.output_dir
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing files from: {raw_dir}")
    print(f"Output directory: {output_dir}")
    
    # Process all files
    auto_records, moto_records = process_all_files(str(raw_dir))
    
    # Write CSV files
    auto_csv = output_dir / 'automobiles.csv'
    moto_csv = output_dir / 'motorcycles.csv'
    
    write_csv(auto_records, str(auto_csv))
    write_csv(moto_records, str(moto_csv))
    
    # Run EDA
    print("\n" + "="*60)
    print("EDA REPORTS")
    print("="*60)
    
    run_eda(str(auto_csv), "Automobiles")
    run_eda(str(moto_csv), "Motorcycles")
    
    print("\n" + "="*60)
    print("DONE - All files processed successfully")
    print("="*60)


if __name__ == '__main__':
    main()
