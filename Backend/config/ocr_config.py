"""
DemoPLAN OCR Configuration
Settings for Google Document AI integration
"""

import os
from typing import Optional, List


class OCRConfig:
    """Configuration for OCR processing with Google Document AI"""
    
    # ========================================================================
    # DOCUMENT AI SETTINGS
    # ========================================================================
    
    # Document AI Processor Configuration
    DOCUMENT_AI_PROCESSOR_ID: str = os.getenv(
        "DOCUMENT_AI_PROCESSOR_ID",
        "f6c4f619b5a674de"  # Default DemoPlanOCR processor
    )
    
    DOCUMENT_AI_PROJECT_ID: str = os.getenv(
        "DOCUMENT_AI_PROJECT_ID",
        "1041867695241"  # Numeric project ID
    )
    
    DOCUMENT_AI_LOCATION: str = os.getenv(
        "DOCUMENT_AI_LOCATION",
        "eu"  # EU region for GDPR compliance
    )
    
    # ========================================================================
    # OCR DETECTION THRESHOLDS
    # ========================================================================
    
    # Minimum characters per page to consider document as "readable"
    # Below this threshold, OCR will be triggered
    MIN_TEXT_DENSITY_THRESHOLD: int = int(os.getenv(
        "OCR_MIN_TEXT_THRESHOLD",
        "100"
    ))
    
    # Confidence thresholds for various detection types
    TABLE_CONFIDENCE_THRESHOLD: float = float(os.getenv(
        "OCR_TABLE_CONFIDENCE_THRESHOLD",
        "0.7"
    ))
    
    HANDWRITING_CONFIDENCE_THRESHOLD: float = float(os.getenv(
        "OCR_HANDWRITING_CONFIDENCE_THRESHOLD",
        "0.6"
    ))
    
    ENTITY_CONFIDENCE_THRESHOLD: float = float(os.getenv(
        "OCR_ENTITY_CONFIDENCE_THRESHOLD",
        "0.65"
    ))
    
    # Overall OCR quality threshold
    MIN_OCR_CONFIDENCE: float = float(os.getenv(
        "MIN_OCR_CONFIDENCE",
        "0.5"
    ))
    
    # ========================================================================
    # COST MANAGEMENT
    # ========================================================================
    
    # Document AI pricing (as of 2024)
    # Check: https://cloud.google.com/document-ai/pricing
    COST_PER_PAGE: float = float(os.getenv(
        "OCR_COST_PER_PAGE",
        "0.015"  # $0.015 per page for Document OCR
    ))
    
    # Maximum pages to process in a single request
    MAX_PAGES_PER_REQUEST: int = int(os.getenv(
        "OCR_MAX_PAGES_PER_DOC",
        "100"
    ))
    
    # Daily cost limit (USD) - safety measure
    DAILY_COST_LIMIT: float = float(os.getenv(
        "OCR_DAILY_COST_LIMIT",
        "50.0"
    ))
    
    # ========================================================================
    # FEATURE FLAGS
    # ========================================================================
    
    # Enable/disable OCR processing globally
    OCR_ENABLED: bool = os.getenv("OCR_ENABLED", "true").lower() == "true"
    
    # Enable table extraction
    TABLE_EXTRACTION_ENABLED: bool = os.getenv(
        "TABLE_EXTRACTION_ENABLED",
        "true"
    ).lower() == "true"
    
    # Enable handwriting detection
    HANDWRITING_DETECTION_ENABLED: bool = os.getenv(
        "HANDWRITING_DETECTION_ENABLED",
        "true"
    ).lower() == "true"
    
    # Enable entity extraction (dates, prices, measurements)
    ENTITY_EXTRACTION_ENABLED: bool = os.getenv(
        "ENTITY_EXTRACTION_ENABLED",
        "true"
    ).lower() == "true"
    
    # ========================================================================
    # LANGUAGE SETTINGS
    # ========================================================================
    
    # Supported languages for OCR (ISO 639-1 codes)
    SUPPORTED_LANGUAGES: List[str] = [
        "ro",  # Romanian - primary
        "en",  # English
        "fr",  # French
        "de",  # German
        "it",  # Italian
    ]
    
    # Default language for OCR if not specified
    DEFAULT_LANGUAGE: str = "ro"
    
    # ========================================================================
    # PROCESSING SETTINGS
    # ========================================================================
    
    # Timeout for OCR processing (seconds)
    OCR_TIMEOUT_SECONDS: int = int(os.getenv(
        "OCR_TIMEOUT_SECONDS",
        "300"  # 5 minutes
    ))
    
    # Retry attempts for failed OCR requests
    OCR_RETRY_ATTEMPTS: int = int(os.getenv(
        "OCR_RETRY_ATTEMPTS",
        "2"
    ))
    
    # Delay between retries (seconds)
    OCR_RETRY_DELAY: int = int(os.getenv(
        "OCR_RETRY_DELAY",
        "5"
    ))
    
    # ========================================================================
    # TABLE EXTRACTION SETTINGS
    # ========================================================================
    
    # Minimum rows to consider as valid table
    MIN_TABLE_ROWS: int = 2
    
    # Minimum columns to consider as valid table
    MIN_TABLE_COLUMNS: int = 2
    
    # Table classification keywords for Romanian construction
    TABLE_CLASSIFICATION_KEYWORDS = {
        "cost_breakdown": [
            "cost", "preț", "sumă", "total", "valoare",
            "tarif", "rating", "prețuri"
        ],
        "specifications": [
            "specificații", "specs", "cerințe", "requirements",
            "caracteristici", "detalii tehnice"
        ],
        "materials": [
            "material", "materiale", "cantitate", "quantity",
            "unitate", "u.m.", "furnizor"
        ],
        "timeline": [
            "durată", "timeline", "termen", "deadline",
            "planificare", "schedule", "etapă"
        ]
    }
    
    # ========================================================================
    # QUALITY CONTROL
    # ========================================================================
    
    # Minimum text density for quality warning
    LOW_DENSITY_WARNING_THRESHOLD: int = 50
    
    # Maximum acceptable processing time (ms)
    MAX_PROCESSING_TIME_MS: int = 60000  # 1 minute
    
    # Enable quality warnings in results
    ENABLE_QUALITY_WARNINGS: bool = True
    
    # ========================================================================
    # CACHING SETTINGS
    # ========================================================================
    
    # Enable caching of OCR results
    ENABLE_OCR_CACHING: bool = os.getenv(
        "ENABLE_OCR_CACHING",
        "true"
    ).lower() == "true"
    
    # Cache TTL for OCR results (hours)
    OCR_CACHE_TTL_HOURS: int = int(os.getenv(
        "OCR_CACHE_TTL_HOURS",
        "24"
    ))
    
    # ========================================================================
    # LOGGING SETTINGS
    # ========================================================================
    
    # Log OCR processing metrics
    LOG_OCR_METRICS: bool = os.getenv(
        "LOG_OCR_METRICS",
        "true"
    ).lower() == "true"
    
    # Log detailed OCR responses (for debugging)
    LOG_OCR_RESPONSES: bool = os.getenv(
        "LOG_OCR_RESPONSES",
        "false"
    ).lower() == "true"
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    @classmethod
    def get_processor_name(cls) -> str:
        """Get full Document AI processor resource name"""
        return (
            f"projects/{cls.DOCUMENT_AI_PROJECT_ID}/"
            f"locations/{cls.DOCUMENT_AI_LOCATION}/"
            f"processors/{cls.DOCUMENT_AI_PROCESSOR_ID}"
        )
    
    @classmethod
    def get_api_endpoint(cls) -> str:
        """Get Document AI API endpoint for the configured location"""
        return f"{cls.DOCUMENT_AI_LOCATION}-documentai.googleapis.com"
    
    @classmethod
    def validate_config(cls) -> dict:
        """
        Validate OCR configuration
        
        Returns:
            Dict with validation results
        """
        issues = []
        
        if not cls.DOCUMENT_AI_PROCESSOR_ID:
            issues.append("DOCUMENT_AI_PROCESSOR_ID not set")
        
        if not cls.DOCUMENT_AI_PROJECT_ID:
            issues.append("DOCUMENT_AI_PROJECT_ID not set")
        
        if cls.MIN_TEXT_DENSITY_THRESHOLD < 0:
            issues.append("MIN_TEXT_DENSITY_THRESHOLD must be >= 0")
        
        if not (0 <= cls.TABLE_CONFIDENCE_THRESHOLD <= 1):
            issues.append("TABLE_CONFIDENCE_THRESHOLD must be between 0 and 1")
        
        if cls.COST_PER_PAGE < 0:
            issues.append("COST_PER_PAGE must be >= 0")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "processor_name": cls.get_processor_name(),
            "api_endpoint": cls.get_api_endpoint()
        }
    
    @classmethod
    def get_config_summary(cls) -> dict:
        """Get summary of OCR configuration"""
        return {
            "ocr_enabled": cls.OCR_ENABLED,
            "processor_id": cls.DOCUMENT_AI_PROCESSOR_ID,
            "location": cls.DOCUMENT_AI_LOCATION,
            "text_threshold": cls.MIN_TEXT_DENSITY_THRESHOLD,
            "cost_per_page": cls.COST_PER_PAGE,
            "max_pages": cls.MAX_PAGES_PER_REQUEST,
            "features": {
                "table_extraction": cls.TABLE_EXTRACTION_ENABLED,
                "handwriting_detection": cls.HANDWRITING_DETECTION_ENABLED,
                "entity_extraction": cls.ENTITY_EXTRACTION_ENABLED
            },
            "supported_languages": cls.SUPPORTED_LANGUAGES
        }


# Global config instance
ocr_config = OCRConfig()


# Validate configuration on import
if ocr_config.OCR_ENABLED:
    validation = OCRConfig.validate_config()
    if not validation["valid"]:
        import logging
        logger = logging.getLogger("demoplan.config.ocr")
        logger.warning(
            f"⚠️ OCR configuration issues detected: "
            f"{', '.join(validation['issues'])}"
        )