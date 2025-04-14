def safe_value(value):
    """Handle non-numeric values for CSV output"""
    if value == 'N/A' or value == 'Error' or value == '-':
        return value
    try:
        # Try to format as float if it's numeric
        float_val = float(value)
        return f"{float_val:.6f}"
    except (ValueError, TypeError):
        # Return as is if it can't be converted to float
        return value 