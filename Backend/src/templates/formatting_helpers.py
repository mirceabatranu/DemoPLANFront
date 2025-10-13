# -*- coding: utf-8 -*-
# src/templates/formatting_helpers.py
"""
Formatting Helper Functions for DEMOPLAN
Utilities for converting data to formatted strings
"""

from typing import List, Dict, Any, Optional
from datetime import datetime


def format_room_list(rooms: List[Dict[str, Any]], max_items: int = 10) -> str:
    """
    Format list of rooms for display
    
    Args:
        rooms: List of room dictionaries with 'name' and 'area'
        max_items: Maximum number of rooms to display
        
    Returns:
        Formatted room list string
    """
    if not rooms:
        return "  • Nicio cameră detectată\n"
    
    output = ""
    for i, room in enumerate(rooms[:max_items], 1):
        name = room.get('name', f'Cameră {i}')
        area = room.get('area', 'N/A')
        output += f"  • {name}: {area} mp\n"
    
    if len(rooms) > max_items:
        output += f"  • ... (+{len(rooms) - max_items} camere suplimentare)\n"
    
    return output


def format_bullet_list(items: List[str], indent: int = 1, max_items: int = None) -> str:
    """
    Format list of items as bullets
    
    Args:
        items: List of strings to format
        indent: Indentation level (spaces = indent * 2)
        max_items: Maximum items to show
        
    Returns:
        Formatted bullet list
    """
    if not items:
        return ""
    
    spacing = "  " * indent
    output = ""
    
    display_items = items[:max_items] if max_items else items
    
    for item in display_items:
        output += f"{spacing}• {item}\n"
    
    if max_items and len(items) > max_items:
        output += f"{spacing}• ... (+{len(items) - max_items} altele)\n"
    
    return output


def format_numbered_list(items: List[str], indent: int = 1, max_items: int = None) -> str:
    """
    Format list of items as numbered list
    
    Args:
        items: List of strings to format
        indent: Indentation level
        max_items: Maximum items to show
        
    Returns:
        Formatted numbered list
    """
    if not items:
        return ""
    
    spacing = "  " * indent
    output = ""
    
    display_items = items[:max_items] if max_items else items
    
    for i, item in enumerate(display_items, 1):
        output += f"{spacing}{i}. {item}\n"
    
    if max_items and len(items) > max_items:
        output += f"{spacing}... (+{len(items) - max_items} altele)\n"
    
    return output


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format float as percentage
    
    Args:
        value: Float value (0.0 to 1.0)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def format_currency(amount: float, currency: str = "RON", decimals: int = 2) -> str:
    """
    Format amount as currency
    
    Args:
        amount: Numeric amount
        currency: Currency code
        decimals: Decimal places
        
    Returns:
        Formatted currency string
    """
    formatted = f"{amount:,.{decimals}f}"
    return f"{formatted} {currency}"


def format_area(area: float, unit: str = "mp") -> str:
    """
    Format area with unit
    
    Args:
        area: Area value
        unit: Unit of measurement
        
    Returns:
        Formatted area string
    """
    return f"{area:.2f} {unit}"


def format_date(date: datetime, format_str: str = "%d.%m.%Y") -> str:
    """
    Format datetime object
    
    Args:
        date: Datetime object
        format_str: Format string
        
    Returns:
        Formatted date string
    """
    return date.strftime(format_str)


def format_datetime(dt: datetime, format_str: str = "%d.%m.%Y %H:%M") -> str:
    """
    Format datetime with time
    
    Args:
        dt: Datetime object
        format_str: Format string
        
    Returns:
        Formatted datetime string
    """
    return dt.strftime(format_str)


def create_table_row(columns: List[str], widths: List[int]) -> str:
    """
    Create single table row
    
    Args:
        columns: List of column values
        widths: List of column widths
        
    Returns:
        Formatted table row
    """
    row = "│"
    for col, width in zip(columns, widths):
        row += f" {col:<{width}} │"
    return row


def create_simple_table(
    headers: List[str],
    rows: List[List[str]],
    widths: Optional[List[int]] = None
) -> str:
    """
    Create simple text table
    
    Args:
        headers: Column headers
        rows: List of row data
        widths: Column widths (auto-calculated if None)
        
    Returns:
        Formatted table string
    """
    if not rows:
        return ""
    
    # Auto-calculate widths if not provided
    if widths is None:
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(str(cell)))
    
    # Create table
    table = ""
    
    # Top border
    table += "┌" + "┬".join("─" * (w + 2) for w in widths) + "┐\n"
    
    # Headers
    table += create_table_row(headers, widths) + "\n"
    
    # Separator
    table += "├" + "┼".join("─" * (w + 2) for w in widths) + "┤\n"
    
    # Rows
    for row in rows:
        table += create_table_row(row, widths) + "\n"
    
    # Bottom border
    table += "└" + "┴".join("─" * (w + 2) for w in widths) + "┘\n"
    
    return table


def create_cost_table(cost_items: List[Dict[str, Any]]) -> str:
    """
    Create formatted cost breakdown table
    
    Args:
        cost_items: List of cost item dictionaries with 'category', 'unit', 'cost'
        
    Returns:
        Formatted cost table
    """
    if not cost_items:
        return ""
    
    headers = ["Categorie", "UM", "Preț (RON)"]
    rows = []
    total = 0.0
    
    for item in cost_items:
        category = item.get('category', 'N/A')
        unit = item.get('unit', 'buc')
        cost = item.get('cost', 0.0)
        total += cost
        
        rows.append([
            category,
            unit,
            format_currency(cost, "")
        ])
    
    table = create_simple_table(headers, rows, [30, 10, 15])
    
    # Add total row
    table += f"\n**TOTAL:** {format_currency(total)}\n"
    
    return table


def format_progress_bar(value: float, width: int = 20, filled_char: str = "█", empty_char: str = "░") -> str:
    """
    Create visual progress bar
    
    Args:
        value: Progress value (0.0 to 1.0)
        width: Bar width in characters
        filled_char: Character for filled portion
        empty_char: Character for empty portion
        
    Returns:
        Progress bar string
    """
    filled = int(value * width)
    empty = width - filled
    
    bar = filled_char * filled + empty_char * empty
    return f"[{bar}] {format_percentage(value)}"


def format_confidence_indicator(confidence: float) -> str:
    """
    Format confidence with emoji indicator
    
    Args:
        confidence: Confidence value (0.0 to 1.0)
        
    Returns:
        Formatted confidence string with emoji
    """
    if confidence >= 0.90:
        return f"✅ Excelent ({format_percentage(confidence)})"
    elif confidence >= 0.75:
        return f"🟢 Ridicat ({format_percentage(confidence)})"
    elif confidence >= 0.50:
        return f"🟡 Mediu ({format_percentage(confidence)})"
    elif confidence >= 0.25:
        return f"🟠 Scăzut ({format_percentage(confidence)})"
    else:
        return f"🔴 Foarte scăzut ({format_percentage(confidence)})"


def format_status_badge(status: str) -> str:
    """
    Format status with appropriate badge
    
    Args:
        status: Status string
        
    Returns:
        Formatted status with emoji
    """
    status_lower = status.lower()
    
    if "complet" in status_lower or "gata" in status_lower:
        return f"✅ {status}"
    elif "progres" in status_lower or "procesare" in status_lower:
        return f"⏳ {status}"
    elif "eroare" in status_lower or "eșuat" in status_lower:
        return f"❌ {status}"
    elif "avertis" in status_lower:
        return f"⚠️ {status}"
    else:
        return f"ℹ️ {status}"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted file size
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def format_duration(days: int) -> str:
    """
    Format duration in days
    
    Args:
        days: Number of days
        
    Returns:
        Formatted duration string
    """
    if days == 1:
        return "1 zi"
    elif days < 30:
        return f"{days} zile"
    elif days < 365:
        months = days // 30
        if months == 1:
            return "1 lună"
        return f"{months} luni"
    else:
        years = days // 365
        if years == 1:
            return "1 an"
        return f"{years} ani"


def format_key_value_pairs(data: Dict[str, Any], indent: int = 1) -> str:
    """
    Format dictionary as key-value pairs
    
    Args:
        data: Dictionary to format
        indent: Indentation level
        
    Returns:
        Formatted key-value string
    """
    if not data:
        return ""
    
    spacing = "  " * indent
    output = ""
    
    for key, value in data.items():
        # Format key nicely
        display_key = key.replace('_', ' ').title()
        output += f"{spacing}**{display_key}:** {value}\n"
    
    return output


def wrap_in_box(text: str, width: int = 60, char: str = "─") -> str:
    """
    Wrap text in a box
    
    Args:
        text: Text to wrap
        width: Box width
        char: Border character
        
    Returns:
        Boxed text
    """
    top = "┌" + char * width + "┐"
    bottom = "└" + char * width + "┘"
    
    lines = text.split('\n')
    boxed = top + "\n"
    
    for line in lines:
        # Pad line to width
        padded = line.ljust(width)
        boxed += f"│{padded}│\n"
    
    boxed += bottom
    
    return boxed


def add_section_header(title: str, icon: str = "📋") -> str:
    """
    Create formatted section header
    
    Args:
        title: Section title
        icon: Icon to use
        
    Returns:
        Formatted header
    """
    return f"\n{icon} **{title.upper()}**\n{'━' * 40}\n\n"


def add_subsection_header(title: str) -> str:
    """
    Create formatted subsection header
    
    Args:
        title: Subsection title
        
    Returns:
        Formatted subsection header
    """
    return f"\n**{title}**\n\n"