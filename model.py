from ortools.init.python import init
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model

def create_model(names, availability_triples):
    num_employees = len(names)
    num_days = 5
    num_shifts = 12
    model = cp_model.CpModel()
    roster = {}

    for emp in range(num_employees):
        for day in range(num_days):
            for shift in range(num_shifts):
                roster[(emp, day, shift)] = model.NewBoolVar(f'shift_{emp}_{day}_{shift}')

    for day in range(num_days):
        for shift in range(num_shifts):
            model.Add(sum(roster[emp, day, shift] for emp in range(num_employees)) == 3)

    for (emp, day, shift, available) in availability_triples:
        if available == 0:
            model.Add(roster[emp, day, shift] == 0)
    
    for emp in range(num_employees):
        if any(available == 1 for (e, d, s, available) in availability_triples if e == emp):
            model.Add(sum(roster[emp, day, shift] for day in range(num_days) for shift in range(num_shifts)) >= 1)
        model.Add(sum(roster[emp, day, shift] for day in range(num_days) for shift in range(num_shifts)) <= 20)

    requested_hours = {}
    for emp in range(num_employees):
        requested_hours[emp] = sum(available for (e, d, s, available) in availability_triples if e == emp)
    
    allocated_hours = {}
    for emp in range(num_employees):
        allocated_hours[emp] = model.NewIntVar(0, num_days*num_shifts, f'allocated_hours_{emp}')
        model.Add(allocated_hours[emp] == sum(roster[emp, day, shift] for day in range(num_days) for shift in range(num_shifts)))

    diff_hours = {}
    for emp in range(num_employees):
        diff_hours[emp] = model.NewIntVar(-num_days * num_shifts, num_days * num_shifts, f'diff_hours_{emp}')
        model.Add(diff_hours[emp] == requested_hours[emp] - allocated_hours[emp])

    abs_diff_hours = {}
    for emp in range(num_employees):
        abs_diff_hours[emp] = model.NewIntVar(0, num_days * num_shifts, f'abs_diff_hours_{emp}')
        model.AddAbsEquality(abs_diff_hours[emp], diff_hours[emp])

    total_abs_diff = model.NewIntVar(0, num_days * num_shifts * num_employees,'total_abs_diff')
    model.Add(total_abs_diff == sum(abs_diff_hours[emp] for emp in range(num_employees)))

    max_diff = model.NewIntVar(0, num_days * num_shifts, 'max_diff')
    for emp in range(num_employees):
        model.Add(max_diff >= diff_hours[emp])
        model.Add(max_diff >= -diff_hours[emp])

    model.Minimize(max_diff + total_abs_diff)

    return model, roster

def solve_model(model, names, roster):
    solver = cp_model.CpSolver
    status = solver.Solve(model=model)
    num_employees = len(names)
    num_days = 5
    num_shifts = 12

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
        #     print(f'  Requested hours: {requested_hours[emp]}')
        #     print(f'  Allocated hours: {solver.Value(allocated_hours[emp])}')
        #     print(f'  Difference: {solver.Value(diff_hours[emp])}')
        
        # # Print the objective values
        # print(f'\nMaximum difference between requested and allocated hours: {solver.Value(max_diff)}')
        # print(f'Total absolute difference between requested and allocated hours: {solver.Value(total_abs_diff)}')
        
        # You can also print some statistics about the solution
        print('\nStatistics')
        print(f'  - Wall time : {solver.WallTime()} s')
        print(f'  - Branches  : {solver.NumBranches()}')
        print(f'  - Conflicts : {solver.NumConflicts()}')

    else:
        print('No solution found.')