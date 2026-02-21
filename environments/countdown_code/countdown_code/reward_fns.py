import sys
import subprocess
import json
import re
from typing import Dict, Any, Optional, List

def load_json_from_response(text: str) -> Optional[Dict[str, str]]:
    """
    Extracts the JSON dictionary from the model's response.
    Prioritizes looking after </think>, then falls back to finding the last valid JSON object.
    """
    # 1. Look for json after the think block
    match = re.search(r"</think>\s*({.*})\s*$", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
            
    # 2. Fallback: look for the last valid json object in the entire string
    match_any = re.search(r"({[^{}]*\{[^{}]*\}[^{}]*}|{.*})", text, re.DOTALL)
    if match_any:
        try:
            return json.loads(match_any.group(1).strip())
        except json.JSONDecodeError:
            pass
    return None

def clean_code(code: str) -> str:
    """
    Cleans up potential unicode escape sequences in the generated code.
    """
    try:
        return code.encode().decode('unicode_escape')
    except Exception:
        return code

def extract_model_response(response: Any) -> str:
    """
    Normalizes the model response.
    """
    return response[-1]['content']

def run_execution_check(completion: str) -> float:
    """
    Executes the generated code in a secure subprocess with timeout.
    
    Returns:
      0.0 -> Parsing failed / Crash / Wrong Output
      0.2 -> Format Correct (JSON parseable)
      1.0 -> Format + Execution Correct (Prints "True")
    """
    completion = extract_model_response(completion)
    current_score = 0.0
    
    # Extract the JSON object from the response
    files = load_json_from_response(completion)
    if files is None:
        return 0.0
    current_score = 0.2 # partial credit for formatting

    # Extract the files and sanitize the text
    solution = clean_code(files.get("solution.py", "").strip())
    test = clean_code(files.get("test.py", "").strip())

    if not solution or not test:
        return current_score

    # Construct a sandbox script: necessary to avoid monkey-patching hacks
    full_script = (
        "import math\n"
        "import random\n"
        f"{solution}\n"
        f"{test}\n"
        # We assume the test code defines 'verify_solution' or runs main logic.
        # This hook ensures we print the boolean result if verify_solution exists.
        f"if 'verify_solution' in locals(): print(verify_solution(numbers, target, expr))"
    )

    try:
        # Run in isolated subprocess
        # sys.executable ensures we use the same python interpreter (access to installed libs)
        result = subprocess.run(
            [sys.executable, "-"],
            input=full_script,
            capture_output=True,
            text=True,
            timeout=2 # 1s timeout
        )
        
        # Check return code (0 = Success) and Stdout ("True")
        if result.returncode == 0 and result.stdout.strip() == "True":
            return 1.0  # Success!
            
    except subprocess.TimeoutExpired:
        pass 
    except Exception:
        pass
        
    return current_score

def run_equation_check(completion: str, info: Dict[str,str]) -> float:
    """
    Validates the mathematical correctness of the solution using Regex and Eval.
    Crucially, this ignores the 'test.py' provided by the model and checks the 'expr' variable directly.
    
    Returns:
        1.0 if valid and correct
        0.0 otherwise
    """
    try:
        completion = extract_model_response(completion)
        nums, target = info['numbers'], info['target']
        files = load_json_from_response(completion)
        if not files:
            return 0.0

        solution = clean_code(files.get("solution.py", "").strip())
        
        # Extract `expr=...` inside the solution.py file
        match = re.search(r"expr\s*=\s*(['\"])(.*?)\1", solution)
        if not match:
            return 0.0
        equation = match.group(2)

        # Check digits, ensure it matches exactly with the input from the Countdown task
        used_numbers = [int(n) for n in re.findall(r"\d+", equation)]
        if sorted(used_numbers) != sorted(nums):
            return 0.0

        # Ensure only allowed characters/ops are present
        allowed_pattern = r"""^['"]?[\d\s+\-*/().]+['"]?$"""
        if not re.match(allowed_pattern, equation):
            return 0.0

        # Math Evaluation
        # Evaluate with empty globals/locals for safety
        result = eval(equation, {"__builtins__": None}, {})
        
        # Check if result matches target (allowing for tiny float errors)
        if abs(float(result) - float(target)) < 1e-5:
            return 1.0

    except Exception:
        pass

    return 0.0