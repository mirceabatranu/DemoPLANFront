"""
DemoPLAN Models Package
Data models for OCR and document processing
"""

from .ocr_models import (
    # Core OCR Results
    OCRResult,
    
    # Table Models
    TableCell,
    TableRow,
    TableData,
    
    # Detection Results
    HandwritingResult,
    ExtractedEntity,
    EntityExtraction,
    
    # Quality & Metrics
    DocumentQualityScore,
    OCRMetrics,
)

__all__ = [
    # Core
    "OCRResult",
    
    # Tables
    "TableCell",
    "TableRow",
    "TableData",
    
    # Detection
    "HandwritingResult",
    "ExtractedEntity",
    "EntityExtraction",
    
    # Quality
    "DocumentQualityScore",
    "OCRMetrics",
]