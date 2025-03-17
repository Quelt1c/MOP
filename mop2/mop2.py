import numpy as np
from tabulate import tabulate

class SimplexSolver:
    def __init__(self, c, A, b, constraint_types=None, maximize=True):
        """
        Initialize the simplex method solver with support for different constraint types
        
        Parameters:
        c -- objective function coefficients
        A -- constraint coefficients matrix
        b -- constraints right-hand side values
        constraint_types -- list of constraint types ('<=', '=', '>='), default all '<='
        maximize -- True if maximization problem, False if minimization
        """
        self.c = np.array(c, dtype=float)
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        
        # Set constraint types, default is '<='
        if constraint_types is None:
            constraint_types = ['<='] * len(b)
        self.constraint_types = constraint_types
        
        # Make sure b is non-negative (multiply constraints by -1 if b < 0)
        for i in range(len(self.b)):
            if self.b[i] < 0:
                self.b[i] *= -1
                self.A[i, :] *= -1
                # Flip constraint type if we multiplied by -1
                if self.constraint_types[i] == '<=':
                    self.constraint_types[i] = '>='
                elif self.constraint_types[i] == '>=':
                    self.constraint_types[i] = '<='
        
        self.maximize = maximize
        if not maximize:
            # For minimization problems, negate the objective function
            self.c = -self.c
        
        self.m, self.n = self.A.shape  # m = constraints, n = original variables
        
        # Count additional variables (slack, surplus, artificial)
        self.num_slack = sum(1 for t in self.constraint_types if t == '<=')
        self.num_surplus = sum(1 for t in self.constraint_types if t == '>=')
        self.num_artificial = sum(1 for t in self.constraint_types if t != '<=')
        
        # Track if artificial variables are being used
        self.has_artificial = self.num_artificial > 0
        
        # Total variables including slack, surplus, and artificial
        self.total_vars = self.n + self.num_slack + self.num_surplus + self.num_artificial
        
        # Initialize for tracking basic variables
        self.basic_vars = [-1] * self.m
        self.artificial_vars = []
        
        self.iteration = 0
        
        # Create initial tableau with proper setup
        self.tableau = self._create_initial_tableau()
        
    def _create_initial_tableau(self):
        """Create the initial simplex tableau with correct variable setup"""
        # Create tableau with all required variables
        tableau = np.zeros((self.m + 1, self.total_vars + 1))
        
        # Set objective function coefficients for original variables
        tableau[0, :self.n] = self.c
        
        # Set constraint coefficients for original variables
        tableau[1:, :self.n] = self.A
        
        # Track current column for adding variables
        col = self.n
        
        # Add slack, surplus, and artificial variables to each constraint
        for i, ctype in enumerate(self.constraint_types):
            if ctype == '<=':
                # Add slack variable (+1)
                tableau[i + 1, col] = 1
                self.basic_vars[i] = col
                col += 1
            elif ctype == '>=':
                # Add surplus variable (-1) and artificial variable (+1)
                tableau[i + 1, col] = -1  # Surplus
                col += 1
                tableau[i + 1, col] = 1   # Artificial
                tableau[0, col] = -1000   # Big M penalty in objective
                self.basic_vars[i] = col
                self.artificial_vars.append(col)
                col += 1
            elif ctype == '=':
                # Add artificial variable (+1)
                tableau[i + 1, col] = 1   # Artificial
                tableau[0, col] = -1000   # Big M penalty in objective
                self.basic_vars[i] = col
                self.artificial_vars.append(col)
                col += 1
        
        # Set right-hand side values
        tableau[1:, -1] = self.b
        
        # Apply Big M method: subtract artificial variable rows from objective
        if self.has_artificial:
            for i in range(self.m):
                if self.basic_vars[i] in self.artificial_vars:
                    tableau[0, :] -= tableau[0, self.basic_vars[i]] * tableau[i + 1, :]
        
        return tableau
    
    def _get_pivot_column(self):
        """Find the pivot column (entering variable)"""
        # Find the most negative value in the objective row
        obj_row = self.tableau[0, :-1]
        min_val = np.min(obj_row)
        
        if min_val >= -1e-10:  # Small epsilon for numerical stability
            return -1  # Optimal solution found
        
        return np.argmin(obj_row)
    
    def _get_pivot_row(self, pivot_col):
        """Find the pivot row (leaving variable)"""
        ratios = []
        for i in range(1, self.m + 1):
            if self.tableau[i, pivot_col] <= 1e-10:  # Small epsilon for numerical stability
                ratios.append(float('inf'))
            else:
                ratios.append(self.tableau[i, -1] / self.tableau[i, pivot_col])
        
        if all(r == float('inf') for r in ratios):
            return -1  # Unbounded solution
        
        # Find the row with the smallest non-negative ratio
        min_ratio_idx = -1
        min_ratio = float('inf')
        
        for i, r in enumerate(ratios):
            if 0 <= r < min_ratio:
                min_ratio = r
                min_ratio_idx = i
        
        if min_ratio_idx == -1:
            return -1  # No valid pivot found
            
        return min_ratio_idx + 1
    
    def _pivot(self, pivot_row, pivot_col):
        """Perform pivot operation"""
        # Get the pivot element
        pivot_element = self.tableau[pivot_row, pivot_col]
        
        # Divide the pivot row by the pivot element
        self.tableau[pivot_row] = self.tableau[pivot_row] / pivot_element
        
        # Update other rows
        for i in range(self.m + 1):
            if i != pivot_row:
                factor = self.tableau[i, pivot_col]
                self.tableau[i] = self.tableau[i] - factor * self.tableau[pivot_row]
        
        # Update basic variables
        self.basic_vars[pivot_row - 1] = pivot_col
    
    def _check_artificial_variables(self):
        """Check if any artificial variables are in the basis with non-zero values"""
        for i, basic_var in enumerate(self.basic_vars):
            if basic_var in self.artificial_vars and abs(self.tableau[i + 1, -1]) > 1e-10:
                return False  # Artificial variable with non-zero value
        return True
    
    def solve(self):
        """Solve the linear programming problem using the simplex method"""
        max_iterations = 100  # Safety to prevent infinite loops
        
        while self.iteration < max_iterations:
            self.iteration += 1
            
            # Print current tableau
            self._print_tableau()
            
            # Find pivot column
            pivot_col = self._get_pivot_column()
            if pivot_col == -1:
                # Check if artificial variables are in basis
                if self.has_artificial and not self._check_artificial_variables():
                    print("Infeasible solution - artificial variables remain in basis.")
                    break
                print("Знайдено оптимальне рішення.")
                break
            
            # Find pivot row
            pivot_row = self._get_pivot_row(pivot_col)
            if pivot_row == -1:
                print("Необмежене рішення.")
                break
            
            # Perform pivot operation
            self._pivot(pivot_row, pivot_col)
        
        if self.iteration >= max_iterations:
            print("Reached maximum iterations without convergence.")
        
        # Print final solution
        self._print_solution()
    
    def _get_variable_name(self, idx):
        """Get the variable name for printing"""
        if idx < self.n:
            return f"x{idx+1}"
        slack_idx = idx - self.n
        if slack_idx < self.num_slack:
            return f"s{slack_idx+1}"
        surplus_idx = slack_idx - self.num_slack
        if surplus_idx < self.num_surplus:
            return f"e{surplus_idx+1}"
        artificial_idx = surplus_idx - self.num_surplus
        return f"a{artificial_idx+1}"
    
    def _get_variable_coef(self, idx):
        """Get the variable coefficient for objective function"""
        if idx < self.n:
            # Original variables
            return self.c[idx] * (-1 if not self.maximize else 1)
        elif idx in self.artificial_vars:
            # Artificial variables (Big M)
            return "M"
        else:
            # Slack/surplus variables
            return 0
    
    def _print_tableau(self):
        """Print the current tableau in a formatted way"""
        print(f"Ітерація {self.iteration}")
        
        # Create header row with objective coefficients
        header = ["", "ci"]
        for j in range(self.total_vars):
            var_coef = self._get_variable_coef(j)
            if isinstance(var_coef, str):
                header.append(var_coef)
            else:
                header.append(f"{var_coef:.1f}")
        header.append("")
        
        # Create variable names row
        var_names = ["b", "cb"]
        for j in range(self.total_vars):
            var_names.append(self._get_variable_name(j))
        var_names.append("v")
        
        # Create rows for basic variables
        rows = []
        
        # Create objective row (z-row)
        obj_row = ["", ""]
        for j in range(self.total_vars):
            obj_row.append(f"{self.tableau[0, j]:.1f}")
        obj_row.append(f"{self.tableau[0, -1]:.1f}")
        rows.append(obj_row)
        
        # Create constraint rows
        for i in range(1, self.m + 1):
            basic_idx = self.basic_vars[i - 1]
            basic_name = self._get_variable_name(basic_idx)
            
            var_coef = self._get_variable_coef(basic_idx)
            if isinstance(var_coef, str):
                basic_coef = var_coef
            else:
                basic_coef = f"{var_coef:.1f}"
            
            row = [basic_name, basic_coef]
            for j in range(self.total_vars):
                row.append(f"{self.tableau[i, j]:.1f}")
            row.append(f"{self.tableau[i, -1]:.1f}")
            rows.append(row)
        
        # Create calculations row (this simulates the M calculations shown in your example)
        calc_row = ["", ""]
        for j in range(self.total_vars):
            # Create a representation similar to the example in your image
            value = self.tableau[0, j]
            if abs(value) < 1e-10:
                calc_row.append("0.0")
            else:
                calc_row.append(f"{value:.1f}")
        calc_row.append("")
        rows.append(calc_row)
        
        # Print table
        table = [header, var_names] + rows
        print(tabulate(table, tablefmt="grid"))
        print()
    
    def _print_solution(self):
        """Print the final solution"""
        print("Розв'язок:")
        
        # Initialize solution vector for original variables
        solution = np.zeros(self.n)
        
        # Set values for basic variables
        for i, var in enumerate(self.basic_vars):
            if var < self.n:  # Only original variables, not slack/surplus/artificial
                solution[var] = self.tableau[i + 1, -1]
        
        # Print solution for all original variables
        for i in range(self.n):
            print(f"x{i+1} = {solution[i]:.1f}")
        
        # Print objective function value (reverse negation if it was a minimization)
        obj_value = self.tableau[0, -1]
        if not self.maximize:
            obj_value = -obj_value
        print(f"Значення функції мети: {obj_value:.1f}")


# Example usage
def main():
    # Example problem:
    # Maximize: z = x1 + 4x2 + 5x3 + 9x4 - 3x5
    # Subject to:
    # -x1 + x2 + 0x3 + x4 + 2x5 <= 1
    # x1 + x2 + 2x3 + 3x4 + 0x5 <= 5
    # x1, x2, x3, x4, x5 >= 0
    
    c = [1, 4, 5, 9, -3]  # Objective function coefficients
    A = [
        [-1, 1, 0, 1, 2],  # Constraint coefficients
        [1, 1, 2, 3, 0]
    ]
    b = [1, 5]  # Right-hand side values
    constraint_types = ['<=', '<=']  # Both constraints are <=
    
    # Create and solve
    solver = SimplexSolver(c, A, b, constraint_types=constraint_types, maximize=True)
    solver.solve()
    
    print("\n\nExample with mixed constraints:")
    # Example with mixed constraints:
    # Minimize: z = 2x1 + 3x2
    # Subject to:
    # x1 + x2 >= 10
    # x1 - x2 = 2
    # 2x1 + 5x2 <= 20
    
    c2 = [2, 3]  # Objective function coefficients
    A2 = [
        [1, 1],   # x1 + x2 >= 10
        [1, -1],  # x1 - x2 = 2
        [2, 5]    # 2x1 + 5x2 <= 20
    ]
    b2 = [10, 2, 20]  # Right-hand side values
    constraint_types2 = ['>=', '=', '<=']  # Mixed constraint types
    
    # Create and solve
    solver2 = SimplexSolver(c2, A2, b2, constraint_types=constraint_types2, maximize=False)
    solver2.solve()


if __name__ == "__main__":
    main()