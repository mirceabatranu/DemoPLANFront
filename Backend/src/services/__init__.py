"""
DemoPLAN Services Module
Core services for the unified agent
"""
# Make key services easily importable
try:
    from .llm_service import safe_construction_call, SafeConstructionLLMService
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

try:
    from .firestore_service import FirestoreService
    FIRESTORE_AVAILABLE = True
except ImportError:
    FIRESTORE_AVAILABLE = False

try:
    from .session_manager import SessionManager, get_session_manager, initialize_session_manager
    SESSION_MANAGER_AVAILABLE = True
except ImportError:
    SESSION_MANAGER_AVAILABLE = False

try:
    from .batch_processor import batch_processor, BatchProcessor, BatchStatus
    BATCH_PROCESSOR_AVAILABLE = True
except ImportError:
    BATCH_PROCESSOR_AVAILABLE = False

try:
    from .file_storage_service import file_storage_service, FileStorageService
    FILE_STORAGE_AVAILABLE = True
except ImportError:
    FILE_STORAGE_AVAILABLE = False

# ✅ NEW: OCR Storage Service
try:
    from .ocr_storage_service import OCRStorageService
    OCR_STORAGE_AVAILABLE = True
except ImportError:
    OCR_STORAGE_AVAILABLE = False

# ✅ NEW: OCR Service
try:
    from .ocr_service import OCRService
    OCR_SERVICE_AVAILABLE = True
except ImportError:
    OCR_SERVICE_AVAILABLE = False


__all__ = [
    'safe_construction_call',
    'SafeConstructionLLMService',
    'FirestoreService',
    'SessionManager',
    'get_session_manager',
    'initialize_session_manager',
    'batch_processor',
    'BatchProcessor',
    'BatchStatus',
    'file_storage_service',
    'FileStorageService',
    'OCRStorageService',  # ✅ NEW
    'OCRService',  # ✅ NEW
    'LLM_AVAILABLE',
    'FIRESTORE_AVAILABLE',
    'SESSION_MANAGER_AVAILABLE',
    'BATCH_PROCESSOR_AVAILABLE',
    'FILE_STORAGE_AVAILABLE',
    'OCR_STORAGE_AVAILABLE',  # ✅ NEW
    'OCR_SERVICE_AVAILABLE'  # ✅ NEW
]