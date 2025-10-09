"""
File Storage Service
Handles file storage in GCS for session uploads (chat/consultation)
"""

import logging
from typing import Optional
from datetime import datetime, timezone
import base64

logger = logging.getLogger("demoplan.services.file_storage")

class FileStorageService:
    """Unified file storage for session uploads"""
    
    def __init__(self):
        try:
            from google.cloud import storage
            self.storage_client = storage.Client()
            self.bucket_name = "demoplan-session-files"
            logger.info("✅ GCS storage initialized for session files")
        except Exception as e:
            logger.error(f"⚠️ GCS storage not available: {e}")
            self.storage_client = None
            self.bucket_name = None
    
    async def store_session_file(
        self, 
        session_id: str, 
        filename: str, 
        content: bytes
    ) -> dict:
        """Store uploaded file for a session"""
        try:
            content_size = len(content)
            file_key = f"{session_id}/{filename}"
            
            # Always use GCS for session files (can be large DXFs/PDFs)
            if self.storage_client:
                bucket = self.storage_client.bucket(self.bucket_name)
                blob = bucket.blob(f"sessions/{file_key}")
                blob.upload_from_string(content)
                
                storage_info = {
                    "storage_type": "gcs",
                    "bucket": self.bucket_name,
                    "blob_path": f"sessions/{file_key}",
                    "size_bytes": content_size,
                    "filename": filename,
                    "stored_at": datetime.now(timezone.utc).isoformat()
                }
                logger.info(f"📦 Stored session file {filename} in GCS ({content_size / 1024:.1f} KB)")
                
            else:
                # Fallback to memory storage (not persisted)
                storage_info = {
                    "storage_type": "memory",
                    "content": base64.b64encode(content).decode('utf-8'),
                    "size_bytes": content_size,
                    "filename": filename,
                    "stored_at": datetime.now(timezone.utc).isoformat()
                }
                logger.warning(f"⚠️ Stored {filename} in memory (no GCS available)")
            
            return storage_info
            
        except Exception as e:
            logger.error(f"❌ Failed to store session file {filename}: {e}")
            raise
    
    async def load_session_file(self, storage_info: dict) -> Optional[bytes]:
        """Load file from storage"""
        try:
            storage_type = storage_info.get("storage_type", "memory")
            
            if storage_type == "gcs" and self.storage_client:
                bucket = self.storage_client.bucket(storage_info["bucket"])
                blob = bucket.blob(storage_info["blob_path"])
                content = blob.download_as_bytes()
                logger.info(f"📥 Loaded session file from GCS ({len(content) / 1024:.1f} KB)")
                return content
                
            elif storage_type == "memory":
                content_b64 = storage_info.get("content")
                if content_b64:
                    content = base64.b64decode(content_b64)
                    logger.info(f"📥 Loaded session file from memory ({len(content) / 1024:.1f} KB)")
                    return content
                return None
                
            else:
                logger.error(f"❌ Unknown storage type: {storage_type}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to load session file: {e}")
            return None

# Singleton instance
file_storage_service = FileStorageService()