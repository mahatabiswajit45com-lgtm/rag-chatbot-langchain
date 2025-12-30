"""
Tools - Functions the AI agent can use
"""

import math
import json
from datetime import datetime
from typing import Optional
from langchain_core.tools import tool


# ===========================================
# CALCULATOR TOOL
# ===========================================

@tool
def calculator(expression: str) -> str:
    """
    Perform mathematical calculations.
    
    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 2", "sqrt(16)", "sin(45)")
    
    Returns:
        The result of the calculation
    """
    # Safe math functions
    safe_dict = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        # Math module functions
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "pi": math.pi,
        "e": math.e,
        "floor": math.floor,
        "ceil": math.ceil,
        "factorial": math.factorial
    }
    
    try:
        # Evaluate safely
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


# ===========================================
# DATE/TIME TOOL
# ===========================================

@tool
def get_current_datetime(format: Optional[str] = None) -> str:
    """
    Get the current date and time.
    
    Args:
        format: Optional datetime format string (e.g., "%Y-%m-%d", "%H:%M:%S")
    
    Returns:
        Current date and time as string
    """
    now = datetime.now()
    
    if format:
        try:
            return now.strftime(format)
        except:
            return now.isoformat()
    
    return json.dumps({
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "day": now.strftime("%A"),
        "timestamp": now.timestamp()
    })


@tool  
def calculate_date_difference(date1: str, date2: str) -> str:
    """
    Calculate the difference between two dates.
    
    Args:
        date1: First date in YYYY-MM-DD format
        date2: Second date in YYYY-MM-DD format
    
    Returns:
        The difference in days
    """
    try:
        d1 = datetime.strptime(date1, "%Y-%m-%d")
        d2 = datetime.strptime(date2, "%Y-%m-%d")
        diff = abs((d2 - d1).days)
        return f"Difference: {diff} days"
    except Exception as e:
        return f"Error: {str(e)}. Use format YYYY-MM-DD"


# ===========================================
# TEXT TOOLS
# ===========================================

@tool
def word_counter(text: str) -> str:
    """
    Count words, characters, and sentences in text.
    
    Args:
        text: The text to analyze
    
    Returns:
        Statistics about the text
    """
    words = len(text.split())
    characters = len(text)
    characters_no_space = len(text.replace(" ", ""))
    sentences = text.count('.') + text.count('!') + text.count('?')
    
    return json.dumps({
        "words": words,
        "characters": characters,
        "characters_no_spaces": characters_no_space,
        "sentences": sentences,
        "average_word_length": round(characters_no_space / max(words, 1), 2)
    })


@tool
def text_transformer(text: str, operation: str) -> str:
    """
    Transform text using various operations.
    
    Args:
        text: The text to transform
        operation: One of: uppercase, lowercase, title, reverse, remove_spaces
    
    Returns:
        Transformed text
    """
    operations = {
        "uppercase": text.upper(),
        "lowercase": text.lower(),
        "title": text.title(),
        "reverse": text[::-1],
        "remove_spaces": text.replace(" ", "")
    }
    
    if operation not in operations:
        return f"Unknown operation. Available: {list(operations.keys())}"
    
    return operations[operation]


# ===========================================
# JSON TOOL
# ===========================================

@tool
def json_formatter(data: str, operation: str = "pretty") -> str:
    """
    Format or validate JSON data.
    
    Args:
        data: JSON string to process
        operation: 'pretty' for formatted output, 'minify' for compact, 'validate' to check
    
    Returns:
        Formatted JSON or validation result
    """
    try:
        parsed = json.loads(data)
        
        if operation == "pretty":
            return json.dumps(parsed, indent=2)
        elif operation == "minify":
            return json.dumps(parsed, separators=(',', ':'))
        elif operation == "validate":
            return "Valid JSON ✓"
        else:
            return f"Unknown operation. Use: pretty, minify, validate"
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {str(e)}"


# ===========================================
# UNIT CONVERTER
# ===========================================

@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """
    Convert between common units.
    
    Args:
        value: The numeric value to convert
        from_unit: Source unit (km, m, mi, ft, kg, lb, c, f)
        to_unit: Target unit
    
    Returns:
        Converted value
    """
    conversions = {
        # Length
        ("km", "m"): lambda x: x * 1000,
        ("m", "km"): lambda x: x / 1000,
        ("km", "mi"): lambda x: x * 0.621371,
        ("mi", "km"): lambda x: x * 1.60934,
        ("m", "ft"): lambda x: x * 3.28084,
        ("ft", "m"): lambda x: x / 3.28084,
        # Weight
        ("kg", "lb"): lambda x: x * 2.20462,
        ("lb", "kg"): lambda x: x / 2.20462,
        # Temperature
        ("c", "f"): lambda x: (x * 9/5) + 32,
        ("f", "c"): lambda x: (x - 32) * 5/9,
    }
    
    key = (from_unit.lower(), to_unit.lower())
    
    if key not in conversions:
        available = list(set([k[0] for k in conversions.keys()]))
        return f"Conversion not supported. Available units: {available}"
    
    result = conversions[key](value)
    return f"{value} {from_unit} = {result:.4f} {to_unit}"


# ===========================================
# ALL TOOLS LIST
# ===========================================

ALL_TOOLS = [
    calculator,
    get_current_datetime,
    calculate_date_difference,
    word_counter,
    text_transformer,
    json_formatter,
    unit_converter
]


def get_tools():
    """Get all available tools"""
    return ALL_TOOLS


def get_tool_names() -> list[str]:
    """Get names of all tools"""
    return [tool.name for tool in ALL_TOOLS]


def get_tool_descriptions() -> dict:
    """Get tool name -> description mapping"""
    return {tool.name: tool.description for tool in ALL_TOOLS}


if __name__ == "__main__":
    # Test tools
    print("=== Calculator ===")
    print(calculator.invoke({"expression": "sqrt(16) + 2**3"}))
    
    print("\n=== DateTime ===")
    print(get_current_datetime.invoke({}))
    
    print("\n=== Word Counter ===")
    print(word_counter.invoke({"text": "Hello world! This is a test."}))
    
    print("\n=== Unit Converter ===")
    print(unit_converter.invoke({"value": 100, "from_unit": "km", "to_unit": "mi"}))
    
    print("\n=== Available Tools ===")
    for name, desc in get_tool_descriptions().items():
        print(f"  • {name}: {desc[:50]}...")
