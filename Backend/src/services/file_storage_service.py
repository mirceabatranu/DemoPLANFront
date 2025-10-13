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

    async def get_file_storage_info(
        self,
        session_id: str,
        filename: str
    ) -> Optional[dict]:
        """
        Get information about where a file is stored.
        
        Args:
            session_id: Session identifier
            filename: Original filename
        
        Returns:
            Storage information dictionary with size, location, etc.
        """
        try:
            file_key = f"{session_id}/{filename}"
            
            if self.storage_client:
                bucket = self.storage_client.bucket(self.bucket_name)
                blob = bucket.blob(f"sessions/{file_key}")
                
                if blob.exists():
                    blob.reload()
                    return {
                        "storage_type": "gcs",
                        "bucket": self.bucket_name,
                        "blob_path": f"sessions/{file_key}",
                        "size_bytes": blob.size,
                        "exists": True,
                        "updated": blob.updated.isoformat() if blob.updated else None,
                        "content_type": blob.content_type
                    }
            
            return {
                "storage_type": "unknown",
                "exists": False,
                "filename": filename
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get file storage info: {e}")
            return None

    async def delete_session_file(
        self,
        session_id: str,
        filename: str
    ) -> bool:
        """
        Delete a session file from GCS storage.
        
        Args:
            session_id: Session identifier
            filename: Original filename
        
        Returns:
            True if deleted successfully
        """
        try:
            file_key = f"{session_id}/{filename}"
            
            if self.storage_client:
                bucket = self.storage_client.bucket(self.bucket_name)
                blob = bucket.blob(f"sessions/{file_key}")
                
                if blob.exists():
                    blob.delete()
                    logger.info(f"🗑️ Deleted session file: {filename}")
                    return True
                else:
                    logger.warning(f"⚠️ File not found for deletion: {filename}")
                    return False
            
            logger.warning(f"⚠️ GCS not available, cannot delete: {filename}")
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to delete session file: {e}")
            return False

    async def delete_all_session_files(
        self,
        session_id: str
    ) -> int:
        """
        Delete ALL files for a session (cleanup on session delete).
        
        Args:
            session_id: Session identifier
        
        Returns:
            Number of files deleted
        """
        try:
            if not self.storage_client:
                logger.warning("⚠️ GCS not available")
                return 0
            
            bucket = self.storage_client.bucket(self.bucket_name)
            prefix = f"sessions/{session_id}/"
            
            blobs = bucket.list_blobs(prefix=prefix)
            deleted_count = 0
            
            for blob in blobs:
                blob.delete()
                deleted_count += 1
            
            logger.info(f"🗑️ Deleted {deleted_count} session files for {session_id}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"❌ Failed to delete session files: {e}")
            return 0

    async def list_session_files(
        self,
        session_id: str
    ) -> list:
        """
        List all files stored for a session.
        
        Args:
            session_id: Session identifier
        
        Returns:
            List of file information dictionaries
        """
        try:
            if not self.storage_client:
                logger.warning("⚠️ GCS not available")
                return []
            
            bucket = self.storage_client.bucket(self.bucket_name)
            prefix = f"sessions/{session_id}/"
            
            blobs = bucket.list_blobs(prefix=prefix)
            
            files = []
            for blob in blobs:
                files.append({
                    "filename": blob.name.split("/")[-1],
                    "size_bytes": blob.size,
                    "blob_path": blob.name,
                    "content_type": blob.content_type,
                    "updated": blob.updated.isoformat() if blob.updated else None
                })
            
            logger.info(f"📋 Found {len(files)} files for session {session_id}")
            return files
            
        except Exception as e:
            logger.error(f"❌ Failed to list session files: {e}")
            return []

    async def get_session_storage_summary(
        self,
        session_id: str
    ) -> dict:
        """
        Get storage summary for a session (total size, file count, etc.).
        
        Args:
            session_id: Session identifier
        
        Returns:
            Summary dictionary with storage metrics
        """
        try:
            files = await self.list_session_files(session_id)
            
            total_size = sum(f.get("size_bytes", 0) for f in files)
            
            return {
                "session_id": session_id,
                "file_count": len(files),
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "files": files
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get storage summary: {e}")
            return {
                "session_id": session_id,
                "file_count": 0,
                "total_size_bytes": 0,
                "error": str(e)
            }

# Singleton instance
file_storage_service = FileStorageService()