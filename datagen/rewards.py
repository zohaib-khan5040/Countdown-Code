import re
import json
import ast
from typing import Dict, Any, List

def get_variable_from_code(code: str, varname: str):
    """
    Use ASTs to extract the value assigned to a particular variable in Python.
    Handles constants, lists, dicts, etc.
    Use this to extract the variable values inside `solution.py`
    """
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == varname:
                    try:
                        return ast.literal_eval(node.value)
                    except Exception:
                        # fallback for expressions
                        try:
                            return ast.unparse(node.value)
                        except Exception:
                            return None
    return None

def load_json_from_response(text: str):
    # Try to find JSON after </think>
    match = re.search(r"</think>\s*({.*})\s*$", text, re.DOTALL)
    if match:
        answer_content = match.group(1).strip()
        out_json = json.loads(answer_content)
        return out_json
    # Fallback: find any JSON object in the text
    match_any = re.search(r"({[^{}]*\{[^{}]*\}[^{}]*}|{.*})", text, re.DOTALL)
    if match_any:
        answer_content = match_any.group(1).strip()
        out_json = json.loads(answer_content)
        return out_json
    return None

def clean_code(code: str):
    return code.encode().decode('unicode_escape')

def get_function_from_code(code: str, funcname: str):
    """
    Extracts the definition of a function named `funcname` from the code string,
    defines it in a local namespace, and returns the function object.
    """
    import ast

    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == funcname:
            # Get the source code for the function definition
            func_code = ast.get_source_segment(code, node)
            if func_code is None:
                # Fallback: reconstruct source from AST
                func_code = "def " + node.name + "(" + ", ".join([arg.arg for arg in node.args.args]) + "):\n    pass"
            # Define the function in a local namespace
            local_ns = {}
            exec(func_code, {}, local_ns)
            return local_ns[funcname]
    return None

def test_pass_reward(response_obj: Dict[str, Any]) -> float:
    try:
        # Extract the completion
        if not isinstance(response_obj, str):
            response: str = response_obj['text']
        else:
            response = response_obj
        out_files: Dict = load_json_from_response(response)
        sol_file: str = clean_code(out_files['solution.py'])
        test_file: str = clean_code(out_files['test.py'])

        # Extract the pieces
        fn = get_function_from_code(test_file, "verify_solution")
        nums, target, expr = (
            get_variable_from_code(sol_file, "numbers"), 
            get_variable_from_code(sol_file, "target"), 
            get_variable_from_code(sol_file, "expr")
        )

        return int(fn(nums, target, expr))
    except Exception as e:
        print(f'[Error computing test reward] {e}')
        return 0

def equation_pass_reward(nums: List[int], target: int, response_obj: Dict[str, Any]) -> float:
    try:
        if not isinstance(response_obj, str):
            response: str = response_obj['text']
        else:
            response = response_obj
        response: str = response_obj['text']
        out_files: Dict = load_json_from_response(response)
        sol_file: str = clean_code(out_files['solution.py'])
        expr = get_variable_from_code(sol_file, "expr")

        # Extract all numbers from the equation
        used_numbers = [int(n) for n in re.findall(r"\d+", expr)]

        # Check if all numbers are used exactly once
        if sorted(used_numbers) != sorted(nums):
            return 0.0
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r"""^['"]?[\d\s+\-*/().]+['"]?$"""
        if not re.match(allowed_pattern, expr):
            return 0.0

        # Evaluate the expr with restricted globals and locals
        result = eval(expr)
        # Check if the expr is correct and matches the ground truth
        if abs(float(result) - float(target)) < 1e-5:
            return 1
        else:
            return 0.0
    except Exception as e:
        # If evaluation fails, reward is 0
        print(f'[Error computing equationreward] {e}')
        return 0.0