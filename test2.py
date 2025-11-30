"""
Primal-Dual Linear Programming Problems
========================================

This module demonstrates the relationship between primal and dual linear programs,
including the use of Lagrange multipliers in optimization theory.

1. PRIMAL PROBLEM (Minimization):
   minimize:    c^T x
   subject to:  Ax >= b
                x >= 0
   
   where x is the decision variable vector, c is the cost vector,
   A is the constraint matrix, and b is the constraint bounds.

2. DUAL PROBLEM (Maximization):
   maximize:    b^T y
   subject to:  A^T y <= c
                y >= 0
   
   where y represents the Lagrange multipliers (shadow prices/dual variables).

3. STRONG DUALITY THEOREM:
   If either problem has an optimal solution, both do, and their optimal
   values are equal: c^T x* = b^T y*

4. WEAK DUALITY THEOREM:
   The optimal from the dual must be lower or equal than the optimal from the 
   primal:  c^T x* >= b^T y*

5. DUALITY GAP:
   With both previous theorems applying: c^T x - b^T y --> the difference between objectives.
   It is always non-negative due to weak duality, = 0 for the optimal solution due to strong
   duality and can be used to measure distance from the optimal (true optimal values lie in
   between the dual and primal optimals)

6. SLACKNESS:
   If a constraint applied to any of the problems is not equal to 0, then the variable 
   associated is equal to 0, because the constraint is not limiting the optimal values. The opposite 
   is also true.
   x_i > 0 ⟹ (A^T y)_i = c_i     ------    y_i > 0 ⟹ (A x)_i = b_i
   (A^T y)_i > c_i ⟹ x_i = 0     ------    (A x)_i > b_i ⟹ y_i = 0

6. LAGRANGE MULTIPLIERS:
   The dual variables (y) are the Lagrange multipliers for the primal constraints.
   They represent the rate of change of the objective function with respect to
   changes in the constraint bounds. They are also called "shadow prices" which
   show the marginal value of relaxing a constraint.
"""

import numpy as np
from scipy.optimize import linprog

class PrimalDualLP:
    """
    The primal problem is formulated as:
        minimize c^T x
        subject to Ax >= b, x >= 0
    
    The dual is automatically constructed as:
        maximize b^T y
        subject to A^T y <= c, y >= 0
    """
    
    def __init__(self, c: np.ndarray, A: np.ndarray, b: np.ndarray):
        """
        Initialize the primal problem.
        
        Parameters:
        -----------
        c : np.ndarray
            Cost vector for primal (n,)
        A : np.ndarray
            Constraint matrix for primal (m x n)
        b : np.ndarray
            Right-hand side vector for primal (m,)
        """
        self.c = np.array(c)
        self.A = np.array(A)
        self.b = np.array(b)
        
        self.m, self.n = A.shape  # m constraints, n variables
        
        print(f"Problem initialized:")
        print(f"  Variables: {self.n}")
        print(f"  Constraints: {self.m}")
    
    def solve_primal(self) -> dict:
        """
        Using linprog to solve the primal problem.
        
        linprog solves: minimize c^T x
                        subject to: A_ub x <= b_ub (upper bound)
                                    A_eq x == b_eq (equality)
                                    variable bounds
        
        Since our primal has Ax >= b, we convert to -Ax <= -b for linprog.
        
        Returns:
        dict : Solution dictionary containing x, objective value, and dual variables
        """
        print(f"\nPrimal problem")
        print(f"Minimize: {self.c}^T x")
        print(f"Subject to: Ax >= b")
        print(f"A = \n{self.A}")
        print(f"b = {self.b}")
        
        # Convert Ax >= b to -Ax <= -b
        A_ub = -self.A
        b_ub = -self.b
        
        # Solve
        result = linprog(
            c=self.c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=(0, None),  # x >= 0, which could be modified
        )
        
        if result.success:
            print(f"\n  Primal Solution:")
            print(f"  Optimal x = {result.x}")
            print(f"  Optimal objective = {result.fun:.6f}")
            
            # The dual variables from linprog correspond to the inequality constraints
            # These are the Lagrange multipliers (shadow prices)
            if hasattr(result, 'ineqlin') and result.ineqlin is not None:
                lagrange_multipliers = -result.ineqlin.marginals
                print(f"  Lagrange multipliers (y) = {lagrange_multipliers}")
                print(f"\n  Interpretation of Lagrange multipliers:")
                for i, mult in enumerate(lagrange_multipliers):
                    print(f"    y_{i+1} = {mult:.6f}: marginal value of constraint {i+1}")
            
            return {
                'x': result.x,
                'objective': result.fun,
                'dual_vars': -result.ineqlin.marginals if hasattr(result, 'ineqlin') else None,
                'success': True
            }
        else:
            print(f"\n  Primal solution failed: {result.message}")
            return {'success': False, 'message': result.message}
    
    def solve_dual(self) -> dict:
        """
        Solve the dual problem.
        
        The dual of: minimize c^T x, subject to Ax >= b, x >= 0
        is:          maximize b^T y, subject to A^T y <= c, y >= 0
        
        Since linprog minimizes, we convert to: minimize -b^T y
        
        Returns:
        dict : Solution dictionary containing y and objective value
        """
        print(f"\nDual problem")
        print(f"Maximize: {self.b}^T y")
        print(f"Subject to: A^T y <= c")
        print(f"A^T = \n{self.A.T}")
        print(f"c = {self.c}")
        
        # Dual define
        c_dual = -self.b  # Maximize b^T y <----> minimize -b^T y
        A_dual = self.A.T  # A^T y <= c
        b_dual = self.c
        
        # Solve
        result = linprog(
            c=c_dual,
            A_ub=A_dual,
            b_ub=b_dual,
            bounds=(0, None),  # y >= 0, same as before
        )
        
        if result.success:
            print(f"\n  Dual Solution:")
            print(f"  Optimal y = {result.x}")
            print(f"  Optimal objective = {-result.fun:.6f}")  # Negate back for max
            
            return {
                'y': result.x,
                'objective': -result.fun,  # Convert back to maximization
                'success': True
            }
        else:
            print(f"\n  Dual solution failed: {result.message}")
            return {'success': False, 'message': result.message}
    
    def verify_duality(self, primal_sol: dict, dual_sol: dict) -> None:
        """
        Verify the duality theorems: strong and weak based on the duality gap

        """
        print("\nDuality theorems:\n")        
        if primal_sol['success'] and dual_sol['success']:
            primal_obj = primal_sol['objective']
            dual_obj = dual_sol['objective']
            gap = abs(primal_obj - dual_obj)
            
            print(f"Primal optimal value: {primal_obj:.10f}")
            print(f"Dual optimal value:   {dual_obj:.10f}")
            print(f"Duality gap:          {gap:.2e}")
            
            if gap >= -1e-6:
                print(f"\nWeak duality verified: b^T y <= c^T x")
                print(f"    Optimal primal >= {dual_obj:.6f}")
                print(f"    Optimal dual <= {primal_obj:.6f}")
                
                if gap < 1e-6:
                    print("Strong duality verified (gap < 1e-6), both solutions are optimal")
                else:
                    print(f"    - Gap > 0 ---> At least one solution is suboptimal")
                    print(f"    - True optimal value is in the range [{dual_obj:.6f}, {primal_obj:.6f}]")
    
    def verify_complementary_slackness(self, primal_sol: dict, dual_sol: dict) -> None:
        """
        Verify complementary slackness conditions:
        1. If x_i > 0, then the i-th dual constraint is tight: (A^T y)_i = c_i
        2. If y_j > 0, then the j-th primal constraint is tight: (Ax)_j = b_j
        
        Its called complementary because the slackness from the primal carries consequences to the 
        variables of the dual and viceversa
        
        Active and inactive refers to when a slack is tight or not (value more or less 0)
        """        
        x = primal_sol['x']
        y = dual_sol['y']
        
        # Primal constraints
        primal_slack = self.A @ x - self.b
        print("\nPrimal constraint slacks (Ax - b):")
        for i, slack in enumerate(primal_slack):
            active = "active" if abs(slack) < 1e-6 else "inactive"
            print(f"  Constraint {i+1} --- Slack: {slack:.6f} ({active}), y_{i+1} = {y[i]:.6f}")
            if y[i] > 1e-6 and abs(slack) > 1e-6:
                print(f"y_{i+1} > 0 but the constraint is not tight (inactive)")
        
        # Dual constraints
        dual_slack = self.c - self.A.T @ y
        print("\nDual constraint slacks (c - A^T y):")
        for i, slack in enumerate(dual_slack):
            active = "active" if abs(slack) < 1e-6 else "inactive"
            print(f"  Constraint {i+1}: {slack:.6f} ({active}), x_{i+1} = {x[i]:.6f}")
            if x[i] > 1e-6 and abs(slack) > 1e-6:
                print(f"x_{i+1} > 0 but the constraint is not tight (inactive)")


def example_production_problem():
    """
    Example: Production Planning Problem
    
    A company produces two products using three resources.
    
    Decision variables:
      x1 = units of product 1
      x2 = units of product 2
    
    Objective: minimize cost = 2x1 + 3x2
    
    Resource constraints:
      x1 + 2x2 >= 4  (constraint 1: minimum production requirement)
      2x1 + x2 >= 5  (constraint 2: minimum demand)
      x1, x2 >= 0
    
    The dual variables will tell us the marginal cost of each constraint.
    """
    
    # Primal problem definition
    c1 = np.array([2.0, 3.0])        # Cost coefficients
    A1 = np.array([[1.0, 2.0],       # Constraint matrix
                  [2.0, 1.0]])
    b1 = np.array([4.0, 5.0])        # RHS vector
    
    c = np.array([2.1, 3.2, 5.6, 10.45, 1.45, 0.39])        # Cost coefficients
    A = np.array([[1.0, 2.0, 0, 1, 0.5, 4.5],       # Constraint matrix
                  [2.0, 1.0, 10.0, 2.0, 2.5, 1.6],
                  [2.0, 0, 5.3, 2, 0, 0],
                  [0, 0, 4.3, 6.4, 0, 2.0]])
    b = np.array([4.0, 5.0, 2.5, 2.6]) 

    # Create and solve
    problem = PrimalDualLP(c, A, b)
    primal_sol = problem.solve_primal()
    dual_sol = problem.solve_dual()
    
    # Verify results
    problem.verify_duality(primal_sol, dual_sol)
    problem.verify_complementary_slackness(primal_sol, dual_sol)
    
    print("\nThe Lagrange multipliers (dual variables) represent shadow prices:")
    print("- They show how much the objective would improve per unit increase")
    print("  in the right-hand side of each constraint.")
    print("- In this production problem, they indicate the marginal value of")
    print("  relaxing each production requirement.")


if __name__ == "__main__":
    example_production_problem()