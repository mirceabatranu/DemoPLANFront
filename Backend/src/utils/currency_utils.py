"""
Currency Utilities - EUR Parsing and Formatting
Handles various EUR string formats from different contractors
"""

import re
from typing import Optional
from decimal import Decimal, InvalidOperation


# ============================================================================
# EUR PARSING
# ============================================================================

def parse_eur_value(value_str: str) -> Optional[float]:
    """
    Convert EUR string to float
    
    Handles formats:
    - "€ 232,471.15"
    - "€232,471.15"
    - "€ 11,704.38"
    - "232,471.15"
    - "232.471,15" (European format)
    - "€  5,976.11 " (extra spaces)
    - "" (empty)
    - "N/A"
    - "-" (dash)
    
    Returns:
        float value or None if unparseable
        
    Examples:
        >>> parse_eur_value("€ 232,471.15")
        232471.15
        >>> parse_eur_value("€11,704.38")
        11704.38
        >>> parse_eur_value("")
        None
        >>> parse_eur_value("N/A")
        None
    """
    if not value_str or not isinstance(value_str, str):
        return None
    
    # Strip and clean
    value_str = value_str.strip()
    
    # Handle empty or N/A cases
    if not value_str or value_str.upper() in ['N/A', 'NA', '-', '–', '—']:
        return None
    
    # Remove EUR symbol and spaces
    value_str = value_str.replace('€', '').replace('EUR', '').strip()
    
    # Detect format: European (123.456,78) vs English (123,456.78)
    # If last separator is comma, it's European format
    if ',' in value_str and '.' in value_str:
        # Has both separators
        last_comma_pos = value_str.rfind(',')
        last_dot_pos = value_str.rfind('.')
        
        if last_comma_pos > last_dot_pos:
            # European format: "123.456,78"
            value_str = value_str.replace('.', '').replace(',', '.')
        else:
            # English format: "123,456.78"
            value_str = value_str.replace(',', '')
    elif ',' in value_str:
        # Only comma - check if it's decimal or thousand separator
        parts = value_str.split(',')
        if len(parts[-1]) == 2:
            # Likely decimal: "123,45"
            value_str = value_str.replace(',', '.')
        else:
            # Likely thousand separator: "123,456"
            value_str = value_str.replace(',', '')
    # If only dots, assume thousand separators for large numbers
    elif '.' in value_str:
        parts = value_str.split('.')
        if len(parts) > 2 or (len(parts) == 2 and len(parts[1]) > 2):
            # Multiple dots or large last part: "123.456" or "123.4567"
            value_str = value_str.replace('.', '')
        # else: it's a decimal point, keep it
    
    # Remove any remaining spaces
    value_str = value_str.replace(' ', '').replace('\xa0', '')  # \xa0 is non-breaking space
    
    # Try to parse
    try:
        value = float(value_str)
        # Sanity check: must be positive
        if value < 0:
            return None
        return value
    except (ValueError, InvalidOperation):
        return None


def parse_eur_value_safe(value_str: str, default: float = 0.0) -> float:
    """
    Parse EUR value with a default fallback
    
    Args:
        value_str: String to parse
        default: Default value if parsing fails (default: 0.0)
        
    Returns:
        Parsed float or default value
        
    Examples:
        >>> parse_eur_value_safe("€ 123.45")
        123.45
        >>> parse_eur_value_safe("N/A", 0.0)
        0.0
    """
    result = parse_eur_value(value_str)
    return result if result is not None else default


# ============================================================================
# EUR FORMATTING
# ============================================================================

def format_eur_display(value: float, include_symbol: bool = True, decimals: int = 2) -> str:
    """
    Format float as EUR string for display
    
    Args:
        value: Float value to format
        include_symbol: Whether to include € symbol (default: True)
        decimals: Number of decimal places (default: 2)
        
    Returns:
        Formatted EUR string
        
    Examples:
        >>> format_eur_display(232471.15)
        '€232,471.15'
        >>> format_eur_display(11704.38)
        '€11,704.38'
        >>> format_eur_display(1234.5, include_symbol=False)
        '1,234.50'
        >>> format_eur_display(1234.567, decimals=3)
        '€1,234.567'
    """
    # Format with thousand separators and decimal places
    formatted = f"{value:,.{decimals}f}"
    
    if include_symbol:
        return f"€{formatted}"
    return formatted


def format_eur_compact(value: float) -> str:
    """
    Format EUR in compact notation for large values
    
    Examples:
        >>> format_eur_compact(1234.56)
        '€1.2K'
        >>> format_eur_compact(1234567.89)
        '€1.2M'
        >>> format_eur_compact(123.45)
        '€123'
    """
    if value >= 1_000_000:
        return f"€{value / 1_000_000:.1f}M"
    elif value >= 1_000:
        return f"€{value / 1_000:.1f}K"
    else:
        return f"€{value:.0f}"


# ============================================================================
# VALIDATION
# ============================================================================

def validate_eur_value(value: float, min_value: float = 0.0, max_value: float = 100_000_000) -> bool:
    """
    Check if EUR value is reasonable
    
    Args:
        value: Value to validate
        min_value: Minimum acceptable value (default: 0.0)
        max_value: Maximum acceptable value (default: €100M)
        
    Returns:
        True if valid, False otherwise
        
    Examples:
        >>> validate_eur_value(1000.0)
        True
        >>> validate_eur_value(-100.0)
        False
        >>> validate_eur_value(150_000_000)
        False
    """
    if not isinstance(value, (int, float)):
        return False
    
    if value < min_value:
        return False
    
    if value > max_value:
        return False
    
    # Check for NaN or Inf
    if value != value or value == float('inf') or value == float('-inf'):
        return False
    
    return True


def validate_eur_string(value_str: str) -> bool:
    """
    Check if string can be parsed as valid EUR value
    
    Examples:
        >>> validate_eur_string("€ 1,234.56")
        True
        >>> validate_eur_string("invalid")
        False
        >>> validate_eur_string("")
        False
    """
    parsed = parse_eur_value(value_str)
    return parsed is not None and validate_eur_value(parsed)


# ============================================================================
# CALCULATIONS
# ============================================================================

def calculate_percentage(part: float, total: float, decimals: int = 2) -> float:
    """
    Calculate percentage with proper rounding
    
    Args:
        part: Part value
        total: Total value
        decimals: Decimal places (default: 2)
        
    Returns:
        Percentage value
        
    Examples:
        >>> calculate_percentage(232471.15, 323235.09)
        71.91
        >>> calculate_percentage(90763.94, 323235.09)
        28.09
    """
    if total == 0:
        return 0.0
    return round((part / total) * 100, decimals)


def sum_eur_values(values: list) -> float:
    """
    Sum EUR values with proper floating point handling
    
    Uses Decimal for precision to avoid floating point errors
    
    Examples:
        >>> sum_eur_values([11704.38, 78812.12, 38857.02])
        129373.52
    """
    if not values:
        return 0.0
    
    # Use Decimal for precision
    total = Decimal('0')
    for value in values:
        if value is not None:
            total += Decimal(str(value))
    
    return float(total)


def calculate_unit_price(total: float, quantity: float) -> Optional[float]:
    """
    Calculate unit price from total and quantity
    
    Args:
        total: Total value
        quantity: Quantity
        
    Returns:
        Unit price or None if quantity is 0
        
    Examples:
        >>> calculate_unit_price(184.76, 59.6)
        3.1
        >>> calculate_unit_price(204.0, 4.0)
        51.0
    """
    if quantity == 0:
        return None
    
    return round(total / quantity, 2)


def calculate_total(quantity: float, unit_price: float) -> float:
    """
    Calculate total from quantity and unit price
    
    Examples:
        >>> calculate_total(59.6, 3.1)
        184.76
        >>> calculate_total(4, 51)
        204.0
    """
    return round(quantity * unit_price, 2)


def values_match(value1: float, value2: float, tolerance_percent: float = 2.0) -> bool:
    """
    Check if two EUR values match within tolerance
    
    Args:
        value1: First value
        value2: Second value
        tolerance_percent: Tolerance percentage (default: 2%)
        
    Returns:
        True if values match within tolerance
        
    Examples:
        >>> values_match(100.0, 101.0, tolerance_percent=2.0)
        True
        >>> values_match(100.0, 105.0, tolerance_percent=2.0)
        False
    """
    if value1 == 0 and value2 == 0:
        return True
    
    if value1 == 0 or value2 == 0:
        return False
    
    diff_percent = abs(value1 - value2) / max(value1, value2) * 100
    return diff_percent <= tolerance_percent


# ============================================================================
# RANGE OPERATIONS
# ============================================================================

def get_value_range(values: list) -> tuple:
    """
    Get min and max from list of values
    
    Examples:
        >>> get_value_range([100, 200, 150, 300])
        (100, 300)
        >>> get_value_range([])
        (0.0, 0.0)
    """
    if not values:
        return (0.0, 0.0)
    
    valid_values = [v for v in values if v is not None and validate_eur_value(v)]
    if not valid_values:
        return (0.0, 0.0)
    
    return (min(valid_values), max(valid_values))


def is_within_range(value: float, value_range: tuple, tolerance_percent: float = 20.0) -> bool:
    """
    Check if value is within expected range (with tolerance)
    
    Args:
        value: Value to check
        value_range: (min, max) tuple
        tolerance_percent: Tolerance outside range (default: 20%)
        
    Returns:
        True if value is within acceptable range
        
    Examples:
        >>> is_within_range(150, (100, 200))
        True
        >>> is_within_range(250, (100, 200), tolerance_percent=30)
        True
        >>> is_within_range(300, (100, 200), tolerance_percent=20)
        False
    """
    min_val, max_val = value_range
    
    # Apply tolerance
    tolerance_amount = (max_val - min_val) * (tolerance_percent / 100)
    expanded_min = min_val - tolerance_amount
    expanded_max = max_val + tolerance_amount
    
    return expanded_min <= value <= expanded_max


def format_value_range(value_range: tuple, compact: bool = False) -> str:
    """
    Format value range for display
    
    Examples:
        >>> format_value_range((1000.0, 5000.0))
        '€1,000.00 - €5,000.00'
        >>> format_value_range((1500.0, 3500.0), compact=True)
        '€1.5K - €3.5K'
    """
    min_val, max_val = value_range
    
    if compact:
        return f"{format_eur_compact(min_val)} - {format_eur_compact(max_val)}"
    else:
        return f"{format_eur_display(min_val)} - {format_eur_display(max_val)}"