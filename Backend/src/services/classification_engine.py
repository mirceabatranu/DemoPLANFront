"""
Classification Engine - Intelligent Item Categorization
Maps item descriptions to standard categories and types
Handles multi-language (Romanian/English) and fuzzy matching
"""

from typing import Tuple, Optional, List, Dict
import re
from difflib import SequenceMatcher

from src.models.offer_models import (
    CategoryType,
    ITEM_TYPE_KEYWORDS,
    UNIT_NORMALIZATION
)


# ============================================================================
# CATEGORY CLASSIFICATION
# ============================================================================

def classify_category(
    description: str,
    category_hint: Optional[str] = None
) -> CategoryType:
    """
    Classify item into standard category (A: Architectural, B: MEP, C: Professional)
    
    Args:
        description: Item description (any language)
        category_hint: Optional hint (e.g., "A", "B", "BUDOWLANKA", "ELEKTRYKA")
        
    Returns:
        CategoryType enum
        
    Examples:
        >>> classify_category("ARCHITECTURAL WORKS", "A")
        CategoryType.A_ARCHITECTURAL
        
        >>> classify_category("PERETI")
        CategoryType.A_ARCHITECTURAL
        
        >>> classify_category("HVAC")
        CategoryType.B_MEP
        
        >>> classify_category("Project Management")
        CategoryType.C_PROFESSIONAL
    """
    # If explicit hint provided and valid
    if category_hint:
        hint_upper = category_hint.upper().strip()
        if hint_upper == "A" or "ARCHITECTURAL" in hint_upper or "BUDOWLANKA" in hint_upper or "CONSTRUCTION" in hint_upper:
            return CategoryType.A_ARCHITECTURAL
        elif hint_upper == "B" or "MEP" in hint_upper or "ELECTRICAL" in hint_upper or "ELEKTRYKA" in hint_upper or "MECHANICAL" in hint_upper:
            return CategoryType.B_MEP
        elif hint_upper == "C" or "PROFESSIONAL" in hint_upper or "ADDITIONAL" in hint_upper or "DODATKOWE" in hint_upper:
            return CategoryType.C_PROFESSIONAL
    
    # Classify based on description
    desc_lower = description.lower()
    
    # Check for MEP keywords first (more specific)
    mep_keywords = [
        'hvac', 'electrical', 'elektryka', 'sanitary', 'sanitare',
        'plumbing', 'ventilation', 'wentylacja', 'curenti', 'current',
        'lighting', 'fire', 'bms', 'security', 'audio', 'video',
        'refrigeration', 'extraction', 'sprinkler'
    ]
    
    for keyword in mep_keywords:
        if keyword in desc_lower:
            return CategoryType.B_MEP
    
    # Check for professional services keywords
    professional_keywords = [
        'management', 'conducere', 'design', 'proiectare',
        'permits', 'autorizatii', 'preliminarii', 'preliminaries',
        'administration', 'supervision', 'insurance', 'warranty'
    ]
    
    for keyword in professional_keywords:
        if keyword in desc_lower:
            return CategoryType.C_PROFESSIONAL
    
    # Check for architectural keywords (default category)
    architectural_keywords = [
        'demolition', 'construction', 'walls', 'pereti', 'floor',
        'ceiling', 'door', 'window', 'furniture', 'painting',
        'tiles', 'glass', 'finishes', 'joinery', 'carpentry'
    ]
    
    for keyword in architectural_keywords:
        if keyword in desc_lower:
            return CategoryType.A_ARCHITECTURAL
    
    # Default to Architectural if no clear match
    return CategoryType.A_ARCHITECTURAL


def get_category_name(category_type: CategoryType) -> str:
    """
    Get human-readable category name
    
    Examples:
        >>> get_category_name(CategoryType.A_ARCHITECTURAL)
        'Architectural Works'
        >>> get_category_name(CategoryType.B_MEP)
        'MEP Works'
    """
    names = {
        CategoryType.A_ARCHITECTURAL: "Architectural Works",
        CategoryType.B_MEP: "MEP Works",
        CategoryType.C_PROFESSIONAL: "Professional Services",
        CategoryType.UNKNOWN: "Unknown"
    }
    return names.get(category_type, "Unknown")


# ============================================================================
# ITEM TYPE CLASSIFICATION
# ============================================================================

def classify_item_type(description: str) -> str:
    """
    Classify item into specific type based on description
    
    Args:
        description: Item description (any language)
        
    Returns:
        Item type string (e.g., "demolition", "hvac", "flooring")
        
    Examples:
        >>> classify_item_type("Demolition")
        'demolition'
        
        >>> classify_item_type("PERETI")
        'construction'
        
        >>> classify_item_type("Curenti Tari")
        'electrical_strong'
        
        >>> classify_item_type("Custom Made Furniture")
        'furniture'
    """
    desc_lower = description.lower()
    
    # Score each item type
    scores = {}
    
    for item_type, keywords in ITEM_TYPE_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if keyword.lower() in desc_lower:
                # Exact word match gets higher score
                if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', desc_lower):
                    score += 2
                else:
                    score += 1
        
        if score > 0:
            scores[item_type] = score
    
    # Return highest scoring type
    if scores:
        return max(scores, key=scores.get)
    
    # Fallback: try fuzzy matching for common misspellings
    best_match = None
    best_ratio = 0.0
    
    for item_type, keywords in ITEM_TYPE_KEYWORDS.items():
        for keyword in keywords:
            ratio = SequenceMatcher(None, desc_lower, keyword.lower()).ratio()
            if ratio > best_ratio and ratio > 0.6:  # 60% similarity threshold
                best_ratio = ratio
                best_match = item_type
    
    if best_match:
        return best_match
    
    # Default fallback based on category classification
    category = classify_category(description)
    
    if category == CategoryType.A_ARCHITECTURAL:
        return "construction"  # Generic architectural
    elif category == CategoryType.B_MEP:
        return "electrical_strong"  # Generic MEP
    elif category == CategoryType.C_PROFESSIONAL:
        return "management"  # Generic professional service
    
    return "other"


def classify_item(
    description: str,
    category_hint: Optional[str] = None
) -> Tuple[CategoryType, str]:
    """
    Classify item into both category and specific type
    
    Args:
        description: Item description
        category_hint: Optional category hint
        
    Returns:
        Tuple of (CategoryType, item_type_string)
        
    Examples:
        >>> classify_item("Demolition")
        (CategoryType.A_ARCHITECTURAL, 'demolition')
        
        >>> classify_item("HVAC", "B")
        (CategoryType.B_MEP, 'hvac')
    """
    category = classify_category(description, category_hint)
    item_type = classify_item_type(description)
    
    return category, item_type


# ============================================================================
# UNIT NORMALIZATION
# ============================================================================

def normalize_unit(unit: str) -> str:
    """
    Normalize unit to standard format
    
    Args:
        unit: Raw unit string (e.g., "m2", "mp", "sqm", "szt", "pcs")
        
    Returns:
        Normalized unit (e.g., "m2", "m", "unit", "set")
        
    Examples:
        >>> normalize_unit("mp")
        'm2'
        >>> normalize_unit("szt")
        'unit'
        >>> normalize_unit("mb")
        'm'
        >>> normalize_unit("kpl")
        'set'
    """
    if not unit:
        return "unit"
    
    unit_lower = unit.lower().strip()
    
    # Check each normalization group
    for normalized, variants in UNIT_NORMALIZATION.items():
        if unit_lower in [v.lower() for v in variants]:
            return normalized
    
    # If no match, return original (cleaned)
    return unit_lower


def get_unit_display_name(unit: str) -> str:
    """
    Get human-readable unit name
    
    Examples:
        >>> get_unit_display_name("m2")
        'square meters'
        >>> get_unit_display_name("m")
        'linear meters'
        >>> get_unit_display_name("unit")
        'units'
    """
    display_names = {
        'm2': 'square meters',
        'm': 'linear meters',
        'unit': 'units',
        'set': 'sets'
    }
    
    normalized = normalize_unit(unit)
    return display_names.get(normalized, unit)


# ============================================================================
# BATCH CLASSIFICATION
# ============================================================================

def classify_items_batch(
    items: List[Dict],
    category_field: str = 'description',
    category_hint_field: Optional[str] = None
) -> List[Dict]:
    """
    Classify multiple items in batch
    
    Args:
        items: List of item dicts
        category_field: Field containing description
        category_hint_field: Optional field containing category hint
        
    Returns:
        List of items with added 'category' and 'item_type' fields
    """
    classified = []
    
    for item in items:
        description = item.get(category_field, "")
        hint = item.get(category_hint_field) if category_hint_field else None
        
        category, item_type = classify_item(description, hint)
        
        # Create enriched item
        enriched = item.copy()
        enriched['category'] = category.value
        enriched['category_name'] = get_category_name(category)
        enriched['item_type'] = item_type
        
        # Normalize unit if present
        if 'unit' in item and item['unit']:
            enriched['unit_normalized'] = normalize_unit(item['unit'])
        
        classified.append(enriched)
    
    return classified


# ============================================================================
# CONFIDENCE SCORING
# ============================================================================

def calculate_classification_confidence(
    description: str,
    classified_type: str
) -> float:
    """
    Calculate confidence score for classification (0.0 - 1.0)
    
    Higher confidence = more matching keywords
    
    Examples:
        >>> calculate_classification_confidence("HVAC System Installation", "hvac")
        0.9
        >>> calculate_classification_confidence("General Works", "construction")
        0.5
    """
    if not description or not classified_type:
        return 0.0
    
    desc_lower = description.lower()
    
    # Get keywords for this type
    keywords = ITEM_TYPE_KEYWORDS.get(classified_type, [])
    if not keywords:
        return 0.5  # Medium confidence for unknown types
    
    # Count matching keywords
    matches = sum(1 for kw in keywords if kw.lower() in desc_lower)
    
    if matches == 0:
        return 0.3  # Low confidence (fallback classification)
    elif matches == 1:
        return 0.6  # Medium confidence
    elif matches == 2:
        return 0.8  # Good confidence
    else:
        return 0.95  # Very high confidence
    

def get_classification_explanation(
    description: str,
    category: CategoryType,
    item_type: str
) -> str:
    """
    Generate human-readable explanation of classification
    
    Examples:
        >>> get_classification_explanation("HVAC", CategoryType.B_MEP, "hvac")
        "Classified as MEP Works (hvac) based on keywords: hvac"
    """
    desc_lower = description.lower()
    
    # Find matching keywords
    matching_keywords = []
    if item_type in ITEM_TYPE_KEYWORDS:
        matching_keywords = [
            kw for kw in ITEM_TYPE_KEYWORDS[item_type]
            if kw.lower() in desc_lower
        ]
    
    category_name = get_category_name(category)
    
    if matching_keywords:
        keywords_str = ", ".join(matching_keywords[:3])  # Show up to 3
        return f"Classified as {category_name} ({item_type}) based on keywords: {keywords_str}"
    else:
        return f"Classified as {category_name} ({item_type}) by default category rules"


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def validate_classification(
    description: str,
    category: CategoryType,
    item_type: str
) -> Tuple[bool, List[str]]:
    """
    Validate that classification makes sense
    
    Returns:
        Tuple of (is_valid, list_of_warnings)
    """
    warnings = []
    
    # Check category-item_type consistency
    if category == CategoryType.A_ARCHITECTURAL:
        architectural_types = [
            'demolition', 'construction', 'flooring', 'ceiling',
            'doors_windows', 'furniture', 'finishes', 'glass_partitions',
            'signage_branding', 'greenery', 'blinds', 'equipment'
        ]
        if item_type not in architectural_types:
            warnings.append(
                f"Item type '{item_type}' seems inconsistent with category 'Architectural'"
            )
    
    elif category == CategoryType.B_MEP:
        mep_types = [
            'hvac', 'sanitary', 'electrical_strong', 'electrical_weak',
            'lighting', 'fire_protection', 'bms', 'security', 'av_systems',
            'kitchen_extraction', 'refrigeration'
        ]
        if item_type not in mep_types:
            warnings.append(
                f"Item type '{item_type}' seems inconsistent with category 'MEP'"
            )
    
    elif category == CategoryType.C_PROFESSIONAL:
        professional_types = [
            'management', 'design', 'permits', 'preliminaries',
            'testing', 'cleaning', 'waste'
        ]
        if item_type not in professional_types:
            warnings.append(
                f"Item type '{item_type}' seems inconsistent with category 'Professional Services'"
            )
    
    # Check if description is too generic
    generic_terms = ['works', 'items', 'other', 'various', 'miscellaneous']
    if any(term in description.lower() for term in generic_terms):
        warnings.append("Description is generic and may require manual review")
    
    is_valid = len(warnings) == 0
    
    return is_valid, warnings


# ============================================================================
# STATISTICS & REPORTING
# ============================================================================

def get_classification_stats(classified_items: List[Dict]) -> Dict:
    """
    Generate statistics about classified items
    
    Args:
        classified_items: List of items with 'category' and 'item_type' fields
        
    Returns:
        Statistics dict
    """
    if not classified_items:
        return {
            'total_items': 0,
            'categories': {},
            'item_types': {},
            'confidence': {}
        }
    
    # Count by category
    category_counts = {}
    for item in classified_items:
        cat = item.get('category', 'UNKNOWN')
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    # Count by item type
    type_counts = {}
    for item in classified_items:
        item_type = item.get('item_type', 'other')
        type_counts[item_type] = type_counts.get(item_type, 0) + 1
    
    return {
        'total_items': len(classified_items),
        'categories': category_counts,
        'item_types': type_counts,
        'unique_item_types': len(type_counts)
    }