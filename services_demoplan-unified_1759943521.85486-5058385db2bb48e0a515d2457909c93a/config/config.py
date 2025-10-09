"""
DemoPLAN Unified Configuration
Phase 1 settings for unified agent deployment with OCR support
"""

import os
from typing import Optional

class Settings:
    """Configuration settings for DemoPLAN Unified"""
    
    # Application settings
    app_name: str = "DemoPLAN Unified"
    version: str = "1.0.0"
    phase: int = 1
    environment: str = os.getenv("ENVIRONMENT", "production")
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Server settings
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", 8080))
    
    # Google Cloud settings
    gcp_project_id: str = os.getenv("GCP_PROJECT_ID", "demoplanfrvcxk")
    gcp_project_number: str = os.getenv("GCP_PROJECT_NUMBER", "1041867695241")
    
    # LLM settings
    gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-thinking-exp")
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4")
    
    # Unified Agent settings
    unified_agent_enabled: bool = True  # Always true for Phase 1
    min_confidence_for_offer: float = float(os.getenv("MIN_CONFIDENCE_FOR_OFFER", "75.0"))
    
    # ML settings (Phase 1 - basic)
    ml_enabled: bool = os.getenv("ML_ENABLED", "true").lower() == "true"
    
    # ========================================================================
    # OCR / DOCUMENT AI SETTINGS (NEW)
    # ========================================================================
    
    # Enable OCR processing globally
    ocr_enabled: bool = os.getenv("OCR_ENABLED", "true").lower() == "true"
    
    # Document AI Processor Configuration
    document_ai_processor_id: str = os.getenv(
        "DOCUMENT_AI_PROCESSOR_ID",
        "f6c4f619b5a674de"  # DemoPlanOCR processor
    )
    
    document_ai_project_number: str = os.getenv(
        "DOCUMENT_AI_PROJECT_NUMBER",
        gcp_project_number
    )
    
    document_ai_location: str = os.getenv(
        "DOCUMENT_AI_LOCATION",
        "eu"  # EU region for GDPR compliance
    )
    
    # OCR Detection Thresholds
    ocr_min_text_threshold: int = int(os.getenv(
        "OCR_MIN_TEXT_THRESHOLD",
        "100"  # Trigger OCR if less than 100 chars per page
    ))
    
    # Cost Management
    ocr_cost_per_page: float = float(os.getenv(
        "OCR_COST_PER_PAGE",
        "0.015"  # $0.015 per page
    ))
    
    ocr_max_pages_per_doc: int = int(os.getenv(
        "OCR_MAX_PAGES_PER_DOC",
        "100"
    ))
    
    ocr_daily_cost_limit: float = float(os.getenv(
        "OCR_DAILY_COST_LIMIT",
        "50.0"
    ))
    
    # Feature Flags
    table_extraction_enabled: bool = os.getenv(
        "TABLE_EXTRACTION_ENABLED",
        "true"
    ).lower() == "true"
    
    handwriting_detection_enabled: bool = os.getenv(
        "HANDWRITING_DETECTION_ENABLED",
        "true"
    ).lower() == "true"
    
    entity_extraction_enabled: bool = os.getenv(
        "ENTITY_EXTRACTION_ENABLED",
        "true"
    ).lower() == "true"
    
    # OCR Processing Settings
    ocr_timeout_seconds: int = int(os.getenv(
        "OCR_TIMEOUT_SECONDS",
        "300"  # 5 minutes
    ))
    
    ocr_retry_attempts: int = int(os.getenv(
        "OCR_RETRY_ATTEMPTS",
        "2"
    ))
    
    # ========================================================================
    # END OCR SETTINGS
    # ========================================================================
    
    # File processing settings
    max_file_size_mb: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    allowed_file_types: list = [".dxf", ".dwg", ".pdf", ".txt", ".jpg", ".png"]
    
    # Session settings
    session_timeout_hours: int = int(os.getenv("SESSION_TIMEOUT_HOURS", "24"))
    max_conversation_length: int = int(os.getenv("MAX_CONVERSATION_LENGTH", "50"))
    
    # Romanian construction settings
    default_currency: str = "RON"
    default_region: str = "bucuresti"
    default_units: str = "metric"
    
    # Cloud Run specific
    cloud_run_service: Optional[str] = os.getenv("K_SERVICE")
    cloud_run_revision: Optional[str] = os.getenv("K_REVISION")
    
    # ML Intelligence Configuration
    ml_learning_enabled: bool = os.getenv("ML_LEARNING_ENABLED", "true").lower() == "true"
    ml_confidence_threshold: float = float(os.getenv("ML_CONFIDENCE_THRESHOLD", "0.75"))
    
    # Pattern Matcher Configuration
    pattern_matching_enabled: bool = os.getenv("PATTERN_MATCHING_ENABLED", "true").lower() == "true"
    pattern_similarity_threshold: float = float(os.getenv("PATTERN_SIMILARITY_THRESHOLD", "0.6"))
    
    # Historical Analysis Configuration
    historical_analysis_enabled: bool = os.getenv("HISTORICAL_ANALYSIS_ENABLED", "true").lower() == "true"
    
    # Romanian Construction Market Configuration
    romanian_base_cost_sqm: float = float(os.getenv("ROMANIAN_BASE_COST_SQM", "800.0"))
    romanian_base_timeline_days: int = int(os.getenv("ROMANIAN_BASE_TIMELINE_DAYS", "30"))
    
    # Feature Flags
    enable_learning_from_outcomes: bool = os.getenv("ENABLE_LEARNING_FROM_OUTCOMES", "true").lower() == "true"
    enable_pattern_prediction: bool = os.getenv("ENABLE_PATTERN_PREDICTION", "true").lower() == "true"
    enable_intelligence_insights: bool = os.getenv("ENABLE_INTELLIGENCE_INSIGHTS", "true").lower() == "true"
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def get_intelligence_config_summary(self) -> dict:
        """Get summary of intelligence configuration"""
        return {
            "ml_enabled": self.ml_learning_enabled,
            "pattern_matching_enabled": self.pattern_matching_enabled,
            "historical_analysis_enabled": self.historical_analysis_enabled,
            "feature_flags": {
                "learning_from_outcomes": self.enable_learning_from_outcomes,
                "pattern_prediction": self.enable_pattern_prediction,
                "intelligence_insights": self.enable_intelligence_insights
            }
        }
    
    def get_ocr_config_summary(self) -> dict:
        """Get summary of OCR configuration"""
        return {
            "ocr_enabled": self.ocr_enabled,
            "processor_id": self.document_ai_processor_id,
            "location": self.document_ai_location,
            "text_threshold": self.ocr_min_text_threshold,
            "cost_per_page": self.ocr_cost_per_page,
            "max_pages": self.ocr_max_pages_per_doc,
            "features": {
                "table_extraction": self.table_extraction_enabled,
                "handwriting_detection": self.handwriting_detection_enabled,
                "entity_extraction": self.entity_extraction_enabled
            }
        }
    
    @property
    def is_cloud_run(self) -> bool:
        """Check if running on Cloud Run"""
        return self.cloud_run_service is not None
    
    @property
    def database_url(self) -> str:
        """Get database URL (Firestore for Phase 1)"""
        return f"firestore://{self.gcp_project_id}"
    
    @property
    def document_ai_processor_name(self) -> str:
        """Get full Document AI processor resource name"""
        return (
            f"projects/{self.document_ai_project_number}/"
            f"locations/{self.document_ai_location}/"
            f"processors/{self.document_ai_processor_id}"
        )
    
    def get_service_info(self) -> dict:
        """Get service information"""
        return {
            "app_name": self.app_name,
            "version": self.version,
            "phase": self.phase,
            "environment": self.environment,
            "unified_agent": self.unified_agent_enabled,
            "ml_enabled": self.ml_enabled,
            "ocr_enabled": self.ocr_enabled,
            "cloud_run": self.is_cloud_run
        }

# Global settings instance
settings = Settings()

# Validation
if not settings.gemini_api_key and settings.environment == "production":
    import logging
    logging.warning("GEMINI_API_KEY not set - LLM functionality may be limited")

if not settings.gcp_project_id:
    import logging  
    logging.warning("GCP_PROJECT_ID not set - Firestore functionality may be limited")

if settings.ocr_enabled and not settings.document_ai_processor_id:
    import logging
    logging.warning("OCR enabled but DOCUMENT_AI_PROCESSOR_ID not set - OCR functionality may fail")