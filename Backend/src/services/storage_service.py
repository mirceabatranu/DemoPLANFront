"""
Geometric Data Storage Service
Handles DXF geometric data storage in GCS separate from Firestore.
ALL geometric data (vertices, boundaries, polylines) goes to GCS.
"""

import logging
import json
from typing import Optional, Dict, Any
from datetime import datetime, timezone

logger = logging.getLogger("demoplan.services.storage")

class GeometricDataStorage:
    """
    Manages DXF geometric data in GCS.
    Rule: ALL raw geometric/spatial data → GCS
          ALL summaries/text descriptions → Firestore
    """
    
    def __init__(self):
        try:
            from google.cloud import storage
            self.storage_client = storage.Client()
            self.bucket_name = "demoplan-geometric-data"
            self._ensure_bucket_exists()
            logger.info("✅ GCS geometric data storage initialized")
        except Exception as e:
            logger.error(f"⚠️ GCS storage not available: {e}")
            self.storage_client = None
            self.bucket_name = None
    
    def _ensure_bucket_exists(self):
        """Ensure the geometric data bucket exists"""
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            if not bucket.exists():
                bucket = self.storage_client.create_bucket(
                    self.bucket_name,
                    location="europe-west1"
                )
                logger.info(f"📦 Created bucket: {self.bucket_name}")
        except Exception as e:
            logger.warning(f"Could not verify/create bucket: {e}")
    
    async def store_geometric_data(
        self,
        session_id: str,
        file_id: str,
        geometric_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Store DXF geometric data in GCS (ALWAYS, regardless of size).
        
        Args:
            session_id: Session identifier
            file_id: File identifier
            geometric_data: Complete geometric analysis (vertices, boundaries, etc.)
        
        Returns:
            Reference object with storage metadata, or None if failed
        """
        try:
            if not self.storage_client:
                logger.error("❌ GCS not available, cannot store geometric data")
                return None
            
            # Serialize to JSON
            json_data = json.dumps(geometric_data, indent=2, default=str)
            data_bytes = json_data.encode('utf-8')
            size_bytes = len(data_bytes)
            
            # Create blob path: sessions/{session_id}/geometric/{file_id}.json
            blob_path = f"sessions/{session_id}/geometric/{file_id}.json"
            
            # Upload to GCS
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob(blob_path)
            blob.upload_from_string(data_bytes, content_type='application/json')
            
            # Create reference object for Firestore
            geometric_ref = {
                "storage_type": "gcs",
                "bucket": self.bucket_name,
                "path": blob_path,
                "size_bytes": size_bytes,
                "stored_at": datetime.now(timezone.utc).isoformat(),
                "file_id": file_id,
                "session_id": session_id
            }
            
            logger.info(f"📦 Stored geometric data for {file_id} in GCS ({size_bytes / 1024:.1f} KB)")
            return geometric_ref
            
        except Exception as e:
            logger.error(f"❌ Failed to store geometric data for {file_id}: {e}")
            return None
    
    async def load_geometric_data(
        self,
        session_id: str,
        file_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Load DXF geometric data from GCS.
        
        Args:
            session_id: Session identifier
            file_id: File identifier
        
        Returns:
            Geometric data dictionary, or None if not found
        """
        try:
            if not self.storage_client:
                logger.error("❌ GCS not available, cannot load geometric data")
                return None
            
            blob_path = f"sessions/{session_id}/geometric/{file_id}.json"
            
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob(blob_path)
            
            if not blob.exists():
                logger.warning(f"⚠️ Geometric data not found: {blob_path}")
                return None
            
            # Download and parse
            json_data = blob.download_as_text()
            geometric_data = json.loads(json_data)
            
            logger.info(f"📥 Loaded geometric data for {file_id} from GCS")
            return geometric_data
            
        except Exception as e:
            logger.error(f"❌ Failed to load geometric data for {file_id}: {e}")
            return None
    
    async def load_geometric_data_from_ref(
        self,
        geometric_ref: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Load geometric data using a reference object from Firestore.
        
        Args:
            geometric_ref: Reference object with storage metadata
        
        Returns:
            Geometric data dictionary, or None if not found
        """
        try:
            if not self.storage_client:
                logger.error("❌ GCS not available")
                return None
            
            storage_type = geometric_ref.get("storage_type")
            if storage_type != "gcs":
                logger.error(f"❌ Unsupported storage type: {storage_type}")
                return None
            
            bucket_name = geometric_ref.get("bucket")
            blob_path = geometric_ref.get("path")
            
            if not bucket_name or not blob_path:
                logger.error("❌ Invalid geometric reference: missing bucket or path")
                return None
            
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            
            if not blob.exists():
                logger.warning(f"⚠️ Geometric data not found: {blob_path}")
                return None
            
            json_data = blob.download_as_text()
            geometric_data = json.loads(json_data)
            
            logger.info(f"📥 Loaded geometric data from GCS reference")
            return geometric_data
            
        except Exception as e:
            logger.error(f"❌ Failed to load geometric data from reference: {e}")
            return None
    
    async def delete_geometric_data(
        self,
        session_id: str,
        file_id: str
    ) -> bool:
        """
        Delete geometric data from GCS.
        
        Args:
            session_id: Session identifier
            file_id: File identifier
        
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            if not self.storage_client:
                logger.error("❌ GCS not available")
                return False
            
            blob_path = f"sessions/{session_id}/geometric/{file_id}.json"
            
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob(blob_path)
            
            if blob.exists():
                blob.delete()
                logger.info(f"🗑️ Deleted geometric data: {blob_path}")
                return True
            else:
                logger.warning(f"⚠️ Geometric data not found for deletion: {blob_path}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to delete geometric data: {e}")
            return False
    
    async def delete_session_geometric_data(
        self,
        session_id: str
    ) -> int:
        """
        Delete ALL geometric data for a session (cleanup on session delete).
        
        Args:
            session_id: Session identifier
        
        Returns:
            Number of files deleted
        """
        try:
            if not self.storage_client:
                return 0
            
            bucket = self.storage_client.bucket(self.bucket_name)
            prefix = f"sessions/{session_id}/geometric/"
            
            blobs = bucket.list_blobs(prefix=prefix)
            deleted_count = 0
            
            for blob in blobs:
                blob.delete()
                deleted_count += 1
            
            logger.info(f"🗑️ Deleted {deleted_count} geometric files for session {session_id}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"❌ Failed to delete session geometric data: {e}")
            return 0
    
    async def get_geometric_data_size(
        self,
        session_id: str,
        file_id: str
    ) -> Optional[int]:
        """
        Get size of geometric data in bytes.
        
        Args:
            session_id: Session identifier
            file_id: File identifier
        
        Returns:
            Size in bytes, or None if not found
        """
        try:
            if not self.storage_client:
                return None
            
            blob_path = f"sessions/{session_id}/geometric/{file_id}.json"
            
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob(blob_path)
            
            if blob.exists():
                blob.reload()
                return blob.size
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get geometric data size: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if GCS storage is available"""
        return self.storage_client is not None

# Singleton instance
geometric_storage = GeometricDataStorage()