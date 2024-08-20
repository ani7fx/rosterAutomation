
import pandas as pd
import numpy as np



availability_df = pd.read_excel("C:\\Users\\aniru\\Downloads\\globalavailability.xlsx")
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



import re
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

# Rename columns
new_columns = []
for col in cleaned_availability_df.columns:
    if col != 'Name':
        new_columns.append(convert_to_24_hour_format(str(col)))
    else:
        new_columns.append(col)




cleaned_availability_df.columns = new_columns



cleaned_availability_df['total_requested_hours'] = cleaned_availability_df.iloc[:,1:].sum(axis=1)

# ------------------------------------------------------------

from ortools.init.python import init
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model


# In[24]:


num_ambassadors = len(names)
num_days = 5
num_hours = 12

all_ambassadors = range(num_ambassadors)
all_days = range(num_days)
all_hours = range(num_hours)



# In[53]:


# model = cp_model.CpModel()

# hours_worked={}
# for a in all_ambassadors:
#     hours_worked[a] = model.new_int_var(0, num_days * num_hours, f'hours_worked_a{a}')

# for d in all_days:
#     for h in all_hours:
#         model.Add(sum(shifts[a,d,h] for a in all_ambassadors) == 3)

employees = names
time_slots = cleaned_availability_df.columns[1:-1]



# shifts = {}
# for e in employees:
#     for t in time_slots:
#         shifts[(e, t)] = model.new_bool_var(f'shift_{e}_{t}')


# In[60]:


# for index, row in cleaned_availability_df.iterrows():
#     employee = row['Name']
#     print(employee)
#     for t in time_slots:
#         print(row[t])
#         model.Add(shifts[(employee, t)] <= row[t])


days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
hours = ['09:00:00', '10:00:00', '11:00:00', '12:00:00', '13:00:00', '14:00:00', '15:00:00', '16:00:00', '17:00:00', '18:00:00', '19:00:00', '20:00:00']
multi_index = pd.MultiIndex.from_product([days, hours], names=['Day', 'Hour'])

availability_df = cleaned_availability_df.iloc[:,:-1]
availability_data = availability_df.drop(columns=['Name']).values.reshape(-1, len(multi_index))
availability_df = pd.DataFrame(availability_data, columns=multi_index)
availability_df.insert(0, 'Name', employees)

time_slots = availability_df.columns[1:]


cleaned_availability_removed_zero_df = cleaned_availability_df.drop(cleaned_availability_df[cleaned_availability_df.total_requested_hours == 0].index)


availability_columns = [col for col in availability_df.columns if col[0] != 'Name']

# Create a mask to identify rows where all availability values are 0
mask = (availability_df[availability_columns] == 0).all(axis=1)

# Drop rows where the mask is True
df_availability_filtered = availability_df[~mask]


df_availability_filtered_indexed = df_availability_filtered.reset_index()


df_availability_filtered_indexed_1 = df_availability_filtered_indexed.drop(columns=['index'])

cleaned_availability_removed_zero_df['total_requested_hours']


# In[45]:


# model = cp_model.CpModel()
# work={}
num_employees = len(df_availability_filtered_indexed_1['Name'])
num_shifts = 12
num_days = 5

# for e in range(num_employees):
#     for s in range(num_shifts):
#         for d in range(num_days):
#             work[e,s,d] = model.NewBoolVar('work%i_%i_%i' % (e, s, d))


# for d in range(num_days):
#     for s in range(num_shifts):
#         model.Add(sum(work[e,s,d] for e in range(num_employees)) == 3)


requests_df = df_availability_filtered_indexed_1.iloc[:,1:]
def create_availability_triples(df):
    triples = []
    for employee, row in df.iterrows():
        for (day, hour), value in row.items():
            if day != 'Name':  # Skip the Name column if it exists
                triples.append((employee, day, hour, int(value)))
    return triples


availability_triples = create_availability_triples(df_availability_filtered_indexed_1)

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
    for employee, day, hour, availability in availability_triples
]


# for employee, day, hour, availability in availability_triples_converted:
#     model.Add(work[(employee, hour, day)] <= availability)

model2 = cp_model.CpModel()
roster = {}
for emp in range(num_employees):
    for day in range(num_days):
        for shift in range(num_shifts):
            roster[(emp, day, shift)] = model2.NewBoolVar(f'shift_{emp}_{day}_{shift}')


for day in range(num_days):
    for shift in range(num_shifts):
        model2.Add(sum(roster[emp, day, shift] for emp in range(num_employees)) == 3)

for(emp, day, shift, available) in availability_triples_converted:
    if available == 0:
        model2.Add(roster[emp, day, shift] == 0)

requested_hours = {}
for emp in range(num_employees):
    requested_hours[emp] = sum(available for (e, d, s, available) in availability_triples_converted if e == emp)

allocated_hours = {}
for emp in range(num_employees):
    allocated_hours[emp] = model2.NewIntVar(0, num_days * num_shifts, f'allocated_hours_{emp}')
    model2.Add(allocated_hours[emp] == sum(roster[emp, day, shift] for day in range(num_days) for shift in range(num_shifts)))

diff_hours = {}
for emp in range(num_employees):
    diff_hours[emp] = model2.NewIntVar(-num_days * num_shifts, num_days * num_shifts, f'diff_hours_{emp}')
    model2.Add(diff_hours[emp] == requested_hours[emp] - allocated_hours[emp])


abs_diff_hours = {}
for emp in range(num_employees):
    abs_diff_hours[emp] = model2.NewIntVar(0, num_days * num_shifts, f'abs_diff_hours_{emp}')
    model2.AddAbsEquality(abs_diff_hours[emp], diff_hours[emp])

total_abs_diff = model2.NewIntVar(0, num_days * num_shifts * num_employees, 'total_abs_diff')
model2.Add(total_abs_diff == sum(abs_diff_hours[emp] for emp in range(num_employees)))

max_diff = model2.NewIntVar(0, num_days * num_shifts, 'max_diff')
for emp in range(num_employees):
    model2.Add(max_diff >= diff_hours[emp])
    model2.Add(max_diff >= -diff_hours[emp])

avg_hours = (num_days*num_shifts*3)/num_employees 


# Ensure each employee with availability gets at least one hour
for emp in range(num_employees):
    # Check if the employee has any availability
    if any(available == 1 for (e, d, s, available) in availability_triples_converted if e == emp):
        # If they do, ensure they get at least one shift
        model2.Add(sum(roster[emp, day, shift] for day in range(num_days) for shift in range(num_shifts)) >= 1)
    model2.add(sum(roster[emp, day, shift] for day in range(num_days) for shift in range(num_shifts)) <= 20)


model2.Minimize(max_diff+total_abs_diff)

# Solve the model
solver = cp_model.CpSolver()
status = solver.Solve(model2)
names_available = df_availability_filtered_indexed_1['Name']

# Check the status and print results
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print('Solution found.')
    
    # Print the schedule for each employee
    for emp in range(num_employees):
        name = names_available[emp]
        print(f'\nSchedule for Employee {emp},{name}:')
        for day in range(num_days):
            for shift in range(num_shifts):
                if solver.Value(roster[emp, day, shift]) == 1:
                    print(f'  Working on Day {day}, Shift {shift}')
        
        # Print requested vs allocated hours
        print(f'  Requested hours: {requested_hours[emp]}')
        print(f'  Allocated hours: {solver.Value(allocated_hours[emp])}')
        print(f'  Difference: {solver.Value(diff_hours[emp])}')
    
    # Print the objective values
    print(f'\nMaximum difference between requested and allocated hours: {solver.Value(max_diff)}')
    print(f'Total absolute difference between requested and allocated hours: {solver.Value(total_abs_diff)}')
    
    # You can also print some statistics about the solution
    print('\nStatistics')
    print(f'  - Wall time : {solver.WallTime()} s')
    print(f'  - Branches  : {solver.NumBranches()}')
    print(f'  - Conflicts : {solver.NumConflicts()}')

else:
    print('No solution found.')


# In[86]:


for emp in range(num_employees):
    for day in range(num_days):
        for shift in range(num_shifts):
            print(solver.Value(roster[emp, day, shift]))


# In[104]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_roster(roster, num_employees, num_days, num_shifts):
    # Create a 3D numpy array to hold the roster data
    roster_array = np.zeros((num_employees, num_days, num_shifts))
    
    # Fill the array with the roster data
    for (emp, day, shift), value in roster.items():
        roster_array[emp, day, shift] = value
    
    # Create a DataFrame for easier plotting
    df = pd.DataFrame({
        'Employee': [f'Emp {e}' for e in range(num_employees) for _ in range(num_days * num_shifts)],
        'Day': [f'Day {d}' for _ in range(num_employees) for d in range(num_days) for _ in range(num_shifts)],
        'Shift': [f'Shift {s}' for _ in range(num_employees * num_days) for s in range(num_shifts)],
        'Working': roster_array.flatten()
    })
    
    # Pivot the DataFrame
    pivot = df.pivot_table(values='Working', index=['Shift'], columns='Day', aggfunc='first')
    
    # Create the heatmap
    plt.figure(figsize=(15, 10))
    sns.heatmap(pivot, cbar_kws={'label': 'Working'}, linewidths=0.5)
    
    plt.title('Employee Roster')
    plt.tight_layout()
    plt.show()

# Example usage:
# Assuming 'solver' is your CP-SAT solver object and 'roster' is your roster variable
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    solved_roster = {(emp, day, shift): solver.Value(roster[emp, day, shift]) 
                     for emp in range(num_employees) 
                     for day in range(num_days) 
                     for shift in range(num_shifts)}
    
    visualize_roster(solved_roster, num_employees, num_days, num_shifts)


# In[109]:


roster_array = np.zeros((num_employees, num_days, num_shifts))
for (emp, day, shift), value in solved_roster.items():
        roster_array[emp, day, shift] = value








