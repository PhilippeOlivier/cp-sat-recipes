"""A workaround for the not (yet) supported `model.AddAllowedAssignments(...).OnlyEnforceIf(b)`."""


from typing import Dict, List, Set, Tuple

from ortools.sat.python import cp_model


def table_optional(model: cp_model.CpModel,
                   variables: List[cp_model.IntVar],
                   tuples: List[Tuple[int]],
                   option: cp_model.IntVar) -> None:
    """Add an optional TABLE (AddAllowedAssignments()) constraint to a model.

    A TABLE constraint is added to `model`, where `variables` assume the values of one of `tuples`,
    but only when `option` is true.

    This only works for nonnegative integers.
    """
    assert isinstance(model, cp_model.CpModel)
    assert isinstance(variables, List | Tuple)
    assert len(variables) > 0
    for v in variables:
        assert isinstance(v, cp_model.IntVar)
    assert isinstance(tuples, List | Tuple)
    assert len(tuples) > 0
    for t in tuples:
        assert isinstance(t, List | Tuple)
        for i in t:
            assert isinstance(i, int)
            assert i >= 0
    assert len(variables) == len(tuples[0])
    assert isinstance(option, cp_model.IntVar)

    # One binary variable per tuple indicates if the values of that tuple are assigned to the
    # variables or not.
    b = [model.NewBoolVar(f"b[{i}]") for i in range(len(tuples))]

    # Set of values that can be assigned to the various variables.
    possible_values = {i: set() for i in range(len(variables))}
    max_value = max(max(t) for t in tuples)
    for t in tuples:
        for i, j in enumerate(t):
            possible_values[i].add(j)

    # is_assigned[i][j] indicates if variable `i` is assigned value `j`.
    is_assigned = [[model.NewBoolVar(f"is_assigned[{i}][{j}]")
                    for j in range(max_value+1)]
                   for i in range(len(variables))]

    # Some assignments are impossible since the value is found in no tuple.
    for i in range(len(variables)):
        for j in range(max_value+1):
            if j not in possible_values[i]:
                model.Add(is_assigned[i][j] == 0).OnlyEnforceIf(option)

    # One value must be assigned to each variable.
    for i in is_assigned:
        model.Add(cp_model.LinearExpr.Sum(i) == 1).OnlyEnforceIf(option)

    # Link `is_assigned` and `variables`.
    for i in range(len(variables)):
        for j in range(max_value+1):
            model.Add(variables[i] == j).OnlyEnforceIf(is_assigned[i][j])

    # TABLE constraint.
    for i, t in enumerate(tuples):
        model.AddBoolAnd([is_assigned[j][t[j]] for j in range(len(t))]).OnlyEnforceIf(b[i])

    # Only one tuple may be assigned to the variables.
    model.Add(cp_model.LinearExpr.Sum(b) == 1).OnlyEnforceIf(option)


if __name__ == "__main__":
    # Model
    m = cp_model.CpModel()

    # Variables
    v = [m.NewIntVar(0, 10, f"v[{i}]") for i in range(5)]

    # Tuples
    t = [(0, 1, 2, 3, 4),
         (5, 6, 7, 8, 9)]

    # Option
    b = m.NewBoolVar("b")

    # TABLE constraint
    table_optional(m, v, t, b)

    # Objective
    m.Minimize(cp_model.LinearExpr.Sum(v))

    # Activate or deactivate the TABLE constraint
    m.Add(b == 1)

    # Solve
    solver = cp_model.CpSolver()
    status = solver.Solve(m)
    print(f"Status: {solver.StatusName(status)}")
    print(f"Objective value: {solver.ObjectiveValue()}")

    print("Variable values: ", end="")
    for i in v:
        print(solver.Value(i), end=" ")
    print()
