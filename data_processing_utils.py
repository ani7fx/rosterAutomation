import pandas as pd
import numpy as np
import re


def load_and_clean_data(file_path):
    availability_df = pd.read_excel(file_path)
    # df = availability_df.drop(index=[35,36]).reset_index(drop=True).drop(index=2).reset_index(drop=True)
    df = availability_df.drop(index=[35,36])
    df_index_reset=df.reset_index()
    df = df_index_reset.drop(index=2)
    df=df.reset_index()
    df = df.drop(columns = ["level_0","index"])
    time_slots = df.iloc[0, 1:].tolist()
    df_data = df.iloc[3:, :]
    names = df_data.iloc[:,0].tolist()
    availability_matrix = df_data.iloc[:, 1:].fillna('').applymap(lambda x: 1 if x == 'x' else 0)
    cleaned_availability_df = pd.DataFrame(availability_matrix.values, columns=time_slots)
    cleaned_availability_df.insert(0, 'Name', names)
    cleaned_availability_df['total_requested_hours'] = cleaned_availability_df.iloc[:,1:].sum(axis=1)
    return cleaned_availability_df, names

def convert_to_24_hour_format(time_str):
    # Convert 'AM/PM' to '24-hour' format
    match_am_pm = re.match(r'(\d+)(AM|PM)', time_str)
    if match_am_pm:
        hour = int(match_am_pm.group(1))
        period = match_am_pm.group(2)
        if period == 'PM' and hour != 12:
            hour += 12
        elif period == 'AM' and hour == 12:
            hour = 0
        return f'{hour:02}:00:00'
    # Return 'HH:MM:SS' as it is if already in 24-hour format
    return time_str

def rename_columns(df):
    new_columns = [convert_to_24_hour_format(str(col)) if col != 'Name' else col for col in df.columns]
    df.columns = new_columns
#     df['total_requested_hours'] = df.iloc[:, 1:].sum(axis=1)
    return df

def convert_to_multi_index(df, names):
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    hours = ['09:00:00', '10:00:00', '11:00:00', '12:00:00', '13:00:00', '14:00:00', '15:00:00', '16:00:00', '17:00:00', '18:00:00', '19:00:00', '20:00:00']
    multi_index = pd.MultiIndex.from_product([days, hours], names=['Day', 'Hour'])
    availability_df = df.iloc[:,:-1]
    availability_data = availability_df.drop(columns=['Name']).values.reshape(-1, len(multi_index))
    availability_df = pd.DataFrame(availability_data, columns=multi_index)
    availability_df.insert(0, 'Name', names)
    availability_columns = [col for col in availability_df.columns if col[0] != 'Name']
    # Create a mask to identify rows where all availability values are 0
    mask = (availability_df[availability_columns] == 0).all(axis=1)
    # Drop rows where the mask is True
    df_availability_filtered = availability_df[~mask]
    df_availability_filtered_indexed = df_availability_filtered.reset_index()
    df_availability_filtered_indexed_1 = df_availability_filtered_indexed.drop(columns=['index'])

    return df_availability_filtered_indexed_1

def create_availability_triples(df):
    triples = []
    for employee, row in df.iterrows():
        for (day, hour), value in row.items():
            if day != 'Name':  # Skip the Name column if it exists
                triples.append((employee, day, hour, int(value)))
    day_to_index = {
        'Monday': 0,
        'Tuesday': 1,
        'Wednesday': 2,
        'Thursday': 3,
        'Friday': 4
    }

    hour_to_index = {
        '09:00:00': 0,
        '10:00:00': 1,
        '11:00:00': 2,
        '12:00:00': 3,
        '13:00:00': 4,
        '14:00:00': 5,
        '15:00:00': 6,
        '16:00:00': 7,
        '17:00:00': 8,
        '18:00:00': 9,
        '19:00:00': 10,
        '20:00:00': 11
    }
    availability_triples_converted = [
        (employee, day_to_index[day], hour_to_index[hour], availability)
        for employee, day, hour, availability in triples
    ]
    return availability_triples_converted




