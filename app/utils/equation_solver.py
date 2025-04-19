import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from sympy.printing.latex import latex
from typing import Tuple, List, Union

def parse_and_solve_equation(equation_str: str) -> Tuple[str, List[str]]:
    """
    Parse and solve the recognized equation using SymPy.
    
    Args:
        equation_str: The equation as a string (e.g., "3x+5=20")
        
    Returns:
        Tuple of (solution, steps) where solution is a string representation
        of the answer and steps is a list of LaTeX-formatted strings for the steps.
    """
    # Clean up the equation string
    equation_str = equation_str.replace(" ", "")
    
    # Check if it's an equation (contains '=')
    if '=' in equation_str:
        # Split into left and right sides
        left_side, right_side = equation_str.split('=')
        
        # Parse both sides using transformations for implicit multiplication
        transformations = (standard_transformations + (implicit_multiplication_application,))
        left_expr = parse_expr(left_side, transformations=transformations)
        right_expr = parse_expr(right_side, transformations=transformations)
        
        # Form the equation
        equation = sp.Eq(left_expr, right_expr)
        
        # Identify the variable(s)
        symbols = list(equation.free_symbols)
        
        if not symbols:
            # No variables, just evaluate the expression
            result = sp.simplify(left_expr - right_expr)
            if result == 0:
                return "True (equation is valid)", [f"{latex(left_expr)} = {latex(right_expr)}"]
            else:
                return "False (equation is invalid)", [f"{latex(left_expr)} \\neq {latex(right_expr)}"]
        
        # Solve for each variable
        steps = []
        solutions = {}
        
        for symbol in symbols:
            try:
                # Record initial equation
                steps.append(f"{latex(equation)}")
                
                # Solve for this variable
                solution = sp.solve(equation, symbol)
                
                if solution:
                    # Record steps
                    steps.append(f"{latex(symbol)} = {latex(solution[0])}")
                    solutions[symbol] = solution[0]
            except Exception as e:
                # Return any error as part of the solution
                return f"Error: {str(e)}", steps
        
        # Format the final solution
        if solutions:
            solution_str = ", ".join([f"{sym} = {sol}" for sym, sol in solutions.items()])
            return solution_str, steps
        else:
            return "No solution found", steps
    else:
        # It's an expression, not an equation
        try:
            # Parse with transformations for implicit multiplication
            transformations = (standard_transformations + (implicit_multiplication_application,))
            expr = parse_expr(equation_str, transformations=transformations)
            
            # Simplify the expression
            simplified = sp.simplify(expr)
            
            steps = [
                f"{latex(expr)}",
                f"{latex(simplified)}"
            ]
            
            return str(simplified), steps
        except Exception as e:
            # Return any error as part of the solution
            return f"Error: {str(e)}", [equation_str]

def solve_linear_equation(equation: sp.Eq, var: sp.Symbol) -> Tuple[sp.Expr, List[sp.Eq]]:
    """
    Solve a linear equation step by step.
    
    Args:
        equation: SymPy equation
        var: Variable to solve for
        
    Returns:
        Tuple of (solution, steps) where steps is a list of equations
    """
    steps = [equation]
    
    # Move all terms with var to the left side and all other terms to the right side
    left_side = equation.lhs
    right_side = equation.rhs
    
    # Collect terms with the variable on the left
    left_terms = sp.collect(left_side, var)
    var_coeff = left_terms.coeff(var, 1)
    
    # Get the constant term on the left side
    left_const = left_side - var_coeff * var
    
    # Move the constant term to the right side
    new_right = right_side - left_const
    new_left = var_coeff * var
    
    # Record this step
    steps.append(sp.Eq(new_left, new_right))
    
    # Divide both sides by the coefficient of the variable
    solution = new_right / var_coeff
    
    # Record final step
    steps.append(sp.Eq(var, solution))
    
    return solution, steps

def solve_quadratic_equation(equation: sp.Eq, var: sp.Symbol) -> Tuple[List[sp.Expr], List[sp.Eq]]:
    """
    Solve a quadratic equation step by step.
    
    Args:
        equation: SymPy equation
        var: Variable to solve for
        
    Returns:
        Tuple of (solutions, steps) where steps is a list of equations
    """
    steps = [equation]
    
    # Move all terms to the left side to get the form: ax^2 + bx + c = 0
    left_side = equation.lhs - equation.rhs
    right_side = 0
    
    standard_form = sp.Eq(left_side, right_side)
    steps.append(standard_form)
    
    # Get the coefficients
    a = left_side.coeff(var, 2)
    b = left_side.coeff(var, 1)
    c = left_side.subs(var, 0)
    
    # Use the quadratic formula
    discriminant = b**2 - 4*a*c
    
    # Record the discriminant
    steps.append(sp.Eq(sp.Symbol('Δ'), discriminant))
    
    # Calculate solutions
    if discriminant >= 0:
        solution1 = (-b + sp.sqrt(discriminant)) / (2*a)
        solution2 = (-b - sp.sqrt(discriminant)) / (2*a)
        
        steps.append(sp.Eq(var, solution1))
        if solution1 != solution2:  # Add the second solution only if it's different
            steps.append(sp.Eq(var, solution2))
        
        return [solution1, solution2], steps
    else:
        # Complex solutions
        solution1 = (-b + sp.sqrt(discriminant)) / (2*a)
        solution2 = (-b - sp.sqrt(discriminant)) / (2*a)
        
        steps.append(sp.Eq(var, solution1))
        steps.append(sp.Eq(var, solution2))
        
        return [solution1, solution2], steps 