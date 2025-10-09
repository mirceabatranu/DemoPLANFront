"""
Drawing file storage service for DXF files

This module handles storage of generated DXF drawings in Google Cloud Storage
with local fallback for development/testing.
"""

from google.cloud import storage
from typing import Optional
import logging
from datetime import datetime, timedelta
import os
import asyncio

logger = logging.getLogger("demoplan.services.drawing_storage")


class DrawingStorageService:
    """
    Handles storage of generated DXF drawings in Google Cloud Storage
    
    Features:
    - Upload DXF files to GCS
    - Generate signed URLs for secure downloads
    - Local storage fallback for development
    - Automatic cleanup of old files
    """
    
    def __init__(self, bucket_name: str = None):
        """
        Initialize drawing storage service
        
        Args:
            bucket_name: GCS bucket name (defaults to env variable)
        """
        self.bucket_name = bucket_name or os.getenv("GCS_BUCKET_NAME", "demoplan-drawings")
        
        try:
            self.storage_client = storage.Client()
            self.bucket = self.storage_client.bucket(self.bucket_name)
            logger.info(f"✅ Drawing storage initialized: gs://{self.bucket_name}")
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize Cloud Storage: {e}")
            logger.warning("📁 Falling back to local storage")
            self.storage_client = None
            self.bucket = None
    
    async def save_drawing(
        self,
        session_id: str,
        drawing_id: str,
        file_content: bytes,
        filename: str = "drawing.dxf"
    ) -> str:
        """
        Save DXF drawing to storage
        
        Args:
            session_id: Session identifier
            drawing_id: Unique drawing identifier
            file_content: DXF file content as bytes
            filename: Filename (default: drawing.dxf)
        
        Returns:
            URL to access the drawing (signed URL for GCS, file path for local)
        """
        
        if not self.storage_client:
            logger.info("💾 Saving drawing locally (Cloud Storage not available)")
            return await self._save_locally(session_id, drawing_id, file_content, filename)
        
        try:
            # Create hierarchical path: drawings/{session_id}/{drawing_id}/filename.dxf
            blob_path = f"drawings/{session_id}/{drawing_id}/{filename}"
            
            logger.info(f"☁️ Uploading to GCS: {blob_path}")
            
            blob = self.bucket.blob(blob_path)
            
            # Upload with metadata
            blob.metadata = {
                'session_id': session_id,
                'drawing_id': drawing_id,
                'uploaded_at': datetime.utcnow().isoformat(),
                'content_type': 'application/dxf'
            }
            
            blob.upload_from_string(
                file_content,
                content_type='application/dxf'
            )
            
            logger.info(f"✅ Drawing uploaded successfully: {blob_path}")
            
            # Generate signed URL (valid for 24 hours)
            url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(hours=24),
                method="GET"
            )
            
            return url
            
        except Exception as e:
            logger.error(f"❌ Failed to save drawing to Cloud Storage: {e}")
            logger.info("📁 Falling back to local storage")
            return await self._save_locally(session_id, drawing_id, file_content, filename)
    
    async def _save_locally(
        self,
        session_id: str,
        drawing_id: str,
        file_content: bytes,
        filename: str
    ) -> str:
        """
        Fallback: save to local filesystem
        
        Args:
            session_id: Session identifier
            drawing_id: Drawing identifier
            file_content: File content as bytes
            filename: Filename
        
        Returns:
            Local file path as URL
        """
        
        # Create directory structure
        local_base = os.getenv("LOCAL_DRAWING_PATH", "./temp_drawings")
        local_path = os.path.join(local_base, session_id, drawing_id)
        
        # Create directories if they don't exist
        os.makedirs(local_path, exist_ok=True)
        
        file_path = os.path.join(local_path, filename)
        
        # Write file asynchronously
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._write_file, file_path, file_content)
        
        logger.info(f"✅ Drawing saved locally: {file_path}")
        
        # Return as file:// URL
        abs_path = os.path.abspath(file_path)
        return f"file://{abs_path}"
    
    def _write_file(self, file_path: str, content: bytes):
        """Synchronous file write helper"""
        with open(file_path, 'wb') as f:
            f.write(content)
    
    async def get_drawing(
        self,
        session_id: str,
        drawing_id: str,
        filename: str = "drawing.dxf"
    ) -> Optional[bytes]:
        """
        Retrieve drawing file content
        
        Args:
            session_id: Session identifier
            drawing_id: Drawing identifier
            filename: Filename
        
        Returns:
            File content as bytes, or None if not found
        """
        
        if not self.storage_client:
            return await self._get_local(session_id, drawing_id, filename)
        
        try:
            blob_path = f"drawings/{session_id}/{drawing_id}/{filename}"
            blob = self.bucket.blob(blob_path)
            
            if blob.exists():
                logger.info(f"📥 Downloading from GCS: {blob_path}")
                return blob.download_as_bytes()
            
            logger.warning(f"⚠️ Drawing not found in GCS: {blob_path}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve drawing from GCS: {e}")
            return await self._get_local(session_id, drawing_id, filename)
    
    async def _get_local(
        self,
        session_id: str,
        drawing_id: str,
        filename: str
    ) -> Optional[bytes]:
        """Get drawing from local storage"""
        
        local_base = os.getenv("LOCAL_DRAWING_PATH", "./temp_drawings")
        file_path = os.path.join(local_base, session_id, drawing_id, filename)
        
        if not os.path.exists(file_path):
            logger.warning(f"⚠️ Drawing not found locally: {file_path}")
            return None
        
        try:
            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(None, self._read_file, file_path)
            logger.info(f"📥 Drawing loaded from local storage: {file_path}")
            return content
        except Exception as e:
            logger.error(f"❌ Failed to read local drawing: {e}")
            return None
    
    def _read_file(self, file_path: str) -> bytes:
        """Synchronous file read helper"""
        with open(file_path, 'rb') as f:
            return f.read()
    
    async def delete_drawing(
        self,
        session_id: str,
        drawing_id: str,
        filename: str = "drawing.dxf"
    ) -> bool:
        """
        Delete a drawing from storage
        
        Args:
            session_id: Session identifier
            drawing_id: Drawing identifier
            filename: Filename
        
        Returns:
            True if deleted successfully, False otherwise
        """
        
        if not self.storage_client:
            return await self._delete_local(session_id, drawing_id, filename)
        
        try:
            blob_path = f"drawings/{session_id}/{drawing_id}/{filename}"
            blob = self.bucket.blob(blob_path)
            
            if blob.exists():
                blob.delete()
                logger.info(f"🗑️ Drawing deleted from GCS: {blob_path}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to delete drawing from GCS: {e}")
            return False
    
    async def _delete_local(
        self,
        session_id: str,
        drawing_id: str,
        filename: str
    ) -> bool:
        """Delete drawing from local storage"""
        
        local_base = os.getenv("LOCAL_DRAWING_PATH", "./temp_drawings")
        file_path = os.path.join(local_base, session_id, drawing_id, filename)
        
        if not os.path.exists(file_path):
            return False
        
        try:
            os.remove(file_path)
            logger.info(f"🗑️ Drawing deleted locally: {file_path}")
            
            # Try to remove empty directories
            dir_path = os.path.dirname(file_path)
            if os.path.exists(dir_path) and not os.listdir(dir_path):
                os.rmdir(dir_path)
            
            return True
        except Exception as e:
            logger.error(f"❌ Failed to delete local drawing: {e}")
            return False
    
    async def list_session_drawings(self, session_id: str) -> list:
        """
        List all drawings for a session
        
        Args:
            session_id: Session identifier
        
        Returns:
            List of drawing metadata dictionaries
        """
        
        if not self.storage_client:
            return await self._list_local(session_id)
        
        try:
            prefix = f"drawings/{session_id}/"
            blobs = self.bucket.list_blobs(prefix=prefix)
            
            drawings = []
            for blob in blobs:
                if blob.name.endswith('.dxf'):
                    drawings.append({
                        'path': blob.name,
                        'size': blob.size,
                        'created': blob.time_created,
                        'updated': blob.updated,
                        'url': blob.generate_signed_url(
                            version="v4",
                            expiration=timedelta(hours=1),
                            method="GET"
                        )
                    })
            
            logger.info(f"📋 Found {len(drawings)} drawings for session {session_id}")
            return drawings
            
        except Exception as e:
            logger.error(f"❌ Failed to list drawings from GCS: {e}")
            return await self._list_local(session_id)
    
    async def _list_local(self, session_id: str) -> list:
        """List drawings from local storage"""
        
        local_base = os.getenv("LOCAL_DRAWING_PATH", "./temp_drawings")
        session_path = os.path.join(local_base, session_id)
        
        if not os.path.exists(session_path):
            return []
        
        drawings = []
        for drawing_id in os.listdir(session_path):
            drawing_path = os.path.join(session_path, drawing_id)
            if os.path.isdir(drawing_path):
                for filename in os.listdir(drawing_path):
                    if filename.endswith('.dxf'):
                        file_path = os.path.join(drawing_path, filename)
                        stat = os.stat(file_path)
                        drawings.append({
                            'path': file_path,
                            'size': stat.st_size,
                            'created': datetime.fromtimestamp(stat.st_ctime),
                            'updated': datetime.fromtimestamp(stat.st_mtime),
                            'url': f"file://{os.path.abspath(file_path)}"
                        })
        
        logger.info(f"📋 Found {len(drawings)} drawings locally for session {session_id}")
        return drawings
    
    def get_storage_info(self) -> dict:
        """
        Get information about storage configuration
        
        Returns:
            Dictionary with storage configuration details
        """
        return {
            'storage_type': 'gcs' if self.storage_client else 'local',
            'bucket_name': self.bucket_name if self.storage_client else None,
            'local_path': os.getenv("LOCAL_DRAWING_PATH", "./temp_drawings"),
            'available': self.storage_client is not None or True  # Local always available
        }