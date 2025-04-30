import os
import azure.functions as func
import logging
import pandas as pd
from io import BytesIO
import json
import re

app = func.FunctionApp()

@app.blob_trigger(arg_name="myblob", path="bronze/{name}",
                               connection="4cf1e2_STORAGE") 
@app.blob_output(arg_name="outputblob", path="silver/{name}", 
                 connection="AzureWebJobsStorage")
def process_files(myblob: func.InputStream,outputblob: func.Out[str]):
    logging.info(f"Processing blob: {myblob.name}")
    blob_name = myblob.name.split('/')[-1]
    file_ext = os.path.splitext(blob_name)[1].lower()
    
    try:
        if file_ext == '.json':
            cleaned_data = process_json(myblob)
            output = save_to_silver(cleaned_data, blob_name, 'json')
            outputblob.set(output)
        elif file_ext == '.csv':
            cleaned_data = process_csv(myblob)
            output = save_to_silver(cleaned_data, blob_name, 'csv')
            outputblob.set(output)
        else:
            logging.error(f"Unsupported file type: {file_ext}")
    except Exception as e:
        logging.error(f"Error processing {blob_name}: {str(e)}")

def process_csv(myblob):
    # Read CSV content
    content = myblob.read()
    df = pd.read_csv(BytesIO(content),low_memory=False)
    logging.info(f"CSV loaded. Shape: {df.shape}")
    logging.info(f"First few columns: {df.columns.tolist()[:10]}")
    # Column-based routing
    if len(df.columns) > 20:  
        return process_wide_format(df, myblob.name)
    else:
        return process_long_format(df, myblob.name)

def process_wide_format(df, filename):
    logging.info(f"Starting process_wide_format on {filename}")
    # Filter rows for 'CPT'
    df = df[df['code|1|type'] == 'CPT']

    # Sort by CPT code
    df = df.sort_values(by='code|1')

    # Removing duplicate rows
    df = df.drop_duplicates(subset='code|1', keep='first')

    columns_to_drop = ['code|2', 'code|2|type','modifiers']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    # Column standardization
    df.columns = df.columns.str.replace('standard_charge|', '', case=False, regex=False).str.strip('_')

    # Calculate unique CPT count for NaN filtering
    unique_cpt_count = df['code|1'].nunique()

    # Drop columns with NaN values
    df = df.loc[:, df.isna().sum() != unique_cpt_count]

    # Remove text-only columns
    text_only_columns = []
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, str) and not any(char.isdigit() for char in str(x))).all():
            text_only_columns.append(col)
    df = df.drop(columns=text_only_columns)

    # Drop 'additional_payer_notes' columns
    df = df.drop(columns=[col for col in df.columns if col.startswith('additional_payer_notes')])

    # Keeping required columns
    id_vars = df.columns[:4].tolist() + df.columns[-2:].tolist()

    # Reshaping the file
    df_melted = df.melt(id_vars=id_vars, var_name='combined', value_name='value')

    # New columns extraction
    value_types = ['negotiated_percentage', 'negotiated_dollar', 'estimated_amount']

    def extract_parts(col):
        for vt in value_types:
            if vt in col:
                parts = col.replace(vt, '').split('|')
                parts = [p for p in parts if p]  # Remove empty strings
                if vt == 'estimated_amount':
                    payer = parts[0] if len(parts) > 0 else None
                    plan = parts[1] if len(parts) > 1 else None
                else:
                    payer = parts[0] if len(parts) > 0 else None
                    plan = parts[1] if len(parts) > 1 else None
                return pd.Series([payer, plan, vt])
        return pd.Series([None, None, None])

    # Apply extraction function
    df_melted[['payer', 'plan', 'value_type']] = df_melted['combined'].apply(extract_parts)

    # Pivot the data
    df_cleaned = df_melted.pivot_table(
        index=id_vars + ['payer', 'plan'],
        columns='value_type',
        values='value',
        aggfunc='first'
    ).reset_index()

    # Remove column index name
    df_cleaned.columns.name = None

    # Calculate negotiated_percentage where missing
    mask = df_cleaned['negotiated_percentage'].isna() & df_cleaned['estimated_amount'].notna() & df_cleaned['max'].notna()
    df_cleaned.loc[mask, 'negotiated_percentage'] = (
        (df_cleaned.loc[mask, 'estimated_amount'] / df_cleaned.loc[mask, 'max']) * 100
    )

    # Drop negotiated_dollar columns
    df_cleaned = df_cleaned.drop(columns=[col for col in df_cleaned.columns if col.startswith('negotiated_dollar')])
    
    def clean_text(val):
        if pd.isna(val):
            return val
        val = re.sub(r'[^a-zA-Z\s]', '', str(val))  
        val = val.lower().strip()                   
        val = re.sub(r'\s+', ' ', val)               
        return val.title()                           

    # Apply to 'payer' and 'plan' columns
    df_cleaned['payer'] = df_cleaned['payer'].apply(clean_text)
    df_cleaned['plan'] = df_cleaned['plan'].apply(clean_text)
    return df_cleaned

def process_long_format(df, filename):
    logging.info(f"Starting process_long_format on {filename}")
    # Keep only required columns
    columns_to_keep = [
    'description', 
    'code|1', 
    'code|1|type', 
    'standard_charge|gross', 
    'standard_charge|discounted_cash', 
    'payer_name', 
    'plan_name', 
    'standard_charge|negotiated_dollar', 
    'standard_charge|negotiated_percentage', 
    'estimated_amount', 
    'standard_charge|min', 
    'standard_charge|max'
    ]
    df = df[columns_to_keep]

    df= df[df['code|1|type'].str.contains('CPT', na=False)]

    df = df.dropna(subset=['estimated_amount'])
    df.columns = df.columns.str.replace('standard_charge|', '', case=False, regex=False).str.strip('_')
    # Step 2: Convert columns to numeric
    df['estimated_amount'] = pd.to_numeric(df['estimated_amount'], errors='coerce')
    df['standard_charge|max'] = pd.to_numeric(df['standard_charge|max'], errors='coerce')

# Step 3: Calculate 'standard_charge|negotiated_percentage'
    df['standard_charge|negotiated_percentage'] = (
    (df['estimated_amount'] / df['standard_charge|max']) * 100)

    df = df.drop(columns=['negotiated_dollar'])

    df=df.rename(columns={
        'payer_name': 'payer',
        'plan_name': 'plan'})
    df.columns = df.columns.str.replace('standard_charge|', '', case=False, regex=False).str.strip('_')
    return df

def process_json(myblob):
    content = myblob.read()
    data = json.loads(content)
    # Add JSON transformations
    clean_data = {k: v for k, v in data.items() if v is not None}
    return clean_data

def save_to_silver(data, filename, data_type):
    output_name = f"{data_type}_{filename}"
    if isinstance(data, pd.DataFrame):
        return data.to_csv(index=False)
    else:
        return json.dumps(data)

