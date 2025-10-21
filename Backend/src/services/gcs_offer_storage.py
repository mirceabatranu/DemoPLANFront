"""
GCS Offer Storage Service
Handles all Google Cloud Storage operations for offer files
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from google.cloud import storage
from google.cloud.exceptions import NotFound, GoogleCloudError

from src.models.offer_models import ParsedOffer


logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

class GCSConfig:
    """GCS Configuration"""
    BUCKET_NAME = "demoplan-offers"
    
    # Folder structure
    ORIGINALS_PREFIX = "originals"  # originals/YYYY/MM/filename.ext
    PARSED_PREFIX = "parsed"        # parsed/OFF_YYYYMMDD_NNN.json
    ATTACHMENTS_PREFIX = "attachments"  # attachments/OFF_YYYYMMDD_NNN/
    
    # Signed URL expiration
    DEFAULT_SIGNED_URL_HOURS = 24


# ============================================================================
# GCS OFFER STORAGE SERVICE
# ============================================================================

class GCSOfferStorageService:
    """
    Manages offer file storage in Google Cloud Storage
    
    Bucket structure:
    gs://demoplan-offers/
    ├── originals/
    │   └── 2024/
    │       └── 11/
    │           └── OFF_20241120_001_imperial_brands_rev06.csv
    ├── parsed/
    │   └── OFF_20241120_001.json
    └── attachments/
        └── OFF_20241120_001/
            └── contract_signed.pdf
    """
    
    def __init__(self, bucket_name: str = GCSConfig.BUCKET_NAME):
        """
        Initialize GCS storage service
        
        Args:
            bucket_name: GCS bucket name (default: demoplan-offers)
        """
        self.bucket_name = bucket_name
        self.client = storage.Client()
        self.bucket = None
        self._initialize_bucket()
    
    def _initialize_bucket(self):
        """Initialize and verify bucket exists"""
        try:
            self.bucket = self.client.bucket(self.bucket_name)
            # Check if bucket exists
            if not self.bucket.exists():
                logger.warning(f"Bucket {self.bucket_name} does not exist. Creating...")
                self.bucket.create(location="EU")
                logger.info(f"Created bucket: {self.bucket_name}")
            else:
                logger.info(f"Connected to bucket: {self.bucket_name}")
        except Exception as e:
            logger.error(f"Failed to initialize GCS bucket: {e}")
            raise
    
    # ========================================================================
    # UPLOAD OPERATIONS
    # ========================================================================
    
    def upload_original(
        self,
        file_content: bytes,
        original_filename: str,
        offer_id: str,
        offer_date: datetime
    ) -> str:
        """
        Upload original offer file to GCS
        
        Path: originals/YYYY/MM/OFF_YYYYMMDD_NNN_original_filename.ext
        Example: originals/2024/11/OFF_20241120_001_imperial_brands_rev06.csv
        
        Args:
            file_content: Raw file bytes
            original_filename: Original filename
            offer_id: Generated offer ID
            offer_date: Date of offer
            
        Returns:
            GCS path (gs://bucket/path)
            
        Raises:
            GoogleCloudError: If upload fails
        """
        try:
            # Build path: originals/YYYY/MM/OFF_ID_filename
            year = offer_date.strftime("%Y")
            month = offer_date.strftime("%m")
            
            # Sanitize filename
            safe_filename = self._sanitize_filename(original_filename)
            
            blob_path = (
                f"{GCSConfig.ORIGINALS_PREFIX}/"
                f"{year}/"
                f"{month}/"
                f"{offer_id}_{safe_filename}"
            )
            
            # Upload
            blob = self.bucket.blob(blob_path)
            blob.upload_from_string(
                file_content,
                content_type=self._get_content_type(original_filename)
            )
            
            # Set metadata
            blob.metadata = {
                'offer_id': offer_id,
                'original_filename': original_filename,
                'upload_date': datetime.now().isoformat()
            }
            blob.patch()
            
            gcs_path = f"gs://{self.bucket_name}/{blob_path}"
            logger.info(f"Uploaded original file: {gcs_path}")
            
            return gcs_path
            
        except Exception as e:
            logger.error(f"Failed to upload original file: {e}")
            raise GoogleCloudError(f"Upload failed: {str(e)}")
    
    def upload_parsed_json(
        self,
        parsed_offer: ParsedOffer
    ) -> str:
        """
        Upload parsed offer JSON to GCS
        
        Path: parsed/OFF_YYYYMMDD_NNN.json
        Example: parsed/OFF_20241120_001.json
        
        Args:
            parsed_offer: ParsedOffer object
            
        Returns:
            GCS path (gs://bucket/path)
            
        Raises:
            GoogleCloudError: If upload fails
        """
        try:
            blob_path = f"{GCSConfig.PARSED_PREFIX}/{parsed_offer.offer_id}.json"
            
            # Convert to JSON
            json_data = parsed_offer.to_gcs_json()
            json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
            
            # Upload
            blob = self.bucket.blob(blob_path)
            blob.upload_from_string(
                json_str,
                content_type='application/json'
            )
            
            # Set metadata
            blob.metadata = {
                'offer_id': parsed_offer.offer_id,
                'project_name': parsed_offer.project.project_name,
                'total_eur': str(parsed_offer.cost_breakdown.grand_total_eur),
                'detail_level': parsed_offer.detail_level.value,
                'parsed_at': parsed_offer.parsed_at.isoformat()
            }
            blob.patch()
            
            gcs_path = f"gs://{self.bucket_name}/{blob_path}"
            logger.info(f"Uploaded parsed JSON: {gcs_path}")
            
            return gcs_path
            
        except Exception as e:
            logger.error(f"Failed to upload parsed JSON: {e}")
            raise GoogleCloudError(f"Upload failed: {str(e)}")
    
    def upload_attachment(
        self,
        file_content: bytes,
        filename: str,
        offer_id: str
    ) -> str:
        """
        Upload attachment file (contracts, photos, etc.)
        
        Path: attachments/OFF_YYYYMMDD_NNN/filename.ext
        Example: attachments/OFF_20241120_001/contract_signed.pdf
        
        Args:
            file_content: Raw file bytes
            filename: Attachment filename
            offer_id: Offer ID
            
        Returns:
            GCS path
        """
        try:
            safe_filename = self._sanitize_filename(filename)
            blob_path = f"{GCSConfig.ATTACHMENTS_PREFIX}/{offer_id}/{safe_filename}"
            
            blob = self.bucket.blob(blob_path)
            blob.upload_from_string(
                file_content,
                content_type=self._get_content_type(filename)
            )
            
            blob.metadata = {
                'offer_id': offer_id,
                'attachment_type': self._get_attachment_type(filename),
                'upload_date': datetime.now().isoformat()
            }
            blob.patch()
            
            gcs_path = f"gs://{self.bucket_name}/{blob_path}"
            logger.info(f"Uploaded attachment: {gcs_path}")
            
            return gcs_path
            
        except Exception as e:
            logger.error(f"Failed to upload attachment: {e}")
            raise GoogleCloudError(f"Upload failed: {str(e)}")
    
    # ========================================================================
    # DOWNLOAD OPERATIONS
    # ========================================================================
    
    def download_parsed_offer(self, offer_id: str) -> Dict:
        """
        Download and parse offer JSON from GCS
        
        Args:
            offer_id: Offer ID (e.g., OFF_20241120_001)
            
        Returns:
            Parsed offer data as dictionary
            
        Raises:
            NotFound: If offer doesn't exist
            GoogleCloudError: If download fails
        """
        try:
            blob_path = f"{GCSConfig.PARSED_PREFIX}/{offer_id}.json"
            blob = self.bucket.blob(blob_path)
            
            if not blob.exists():
                raise NotFound(f"Offer {offer_id} not found in GCS")
            
            # Download
            json_str = blob.download_as_string()
            offer_data = json.loads(json_str)
            
            logger.info(f"Downloaded parsed offer: {offer_id}")
            return offer_data
            
        except NotFound:
            raise
        except Exception as e:
            logger.error(f"Failed to download offer {offer_id}: {e}")
            raise GoogleCloudError(f"Download failed: {str(e)}")
    
    def download_original_file(self, gcs_path: str) -> bytes:
        """
        Download original offer file
        
        Args:
            gcs_path: Full GCS path (gs://bucket/path)
            
        Returns:
            File content as bytes
            
        Raises:
            NotFound: If file doesn't exist
            GoogleCloudError: If download fails
        """
        try:
            # Extract blob path from gs:// URL
            blob_path = gcs_path.replace(f"gs://{self.bucket_name}/", "")
            blob = self.bucket.blob(blob_path)
            
            if not blob.exists():
                raise NotFound(f"File not found: {gcs_path}")
            
            content = blob.download_as_bytes()
            logger.info(f"Downloaded original file: {gcs_path}")
            
            return content
            
        except NotFound:
            raise
        except Exception as e:
            logger.error(f"Failed to download file {gcs_path}: {e}")
            raise GoogleCloudError(f"Download failed: {str(e)}")
    
    # ========================================================================
    # SIGNED URL GENERATION
    # ========================================================================
    
    def generate_signed_url(
        self,
        gcs_path: str,
        expiration_hours: int = GCSConfig.DEFAULT_SIGNED_URL_HOURS
    ) -> str:
        """
        Generate signed URL for temporary access to file
        
        Args:
            gcs_path: GCS path (gs://bucket/path)
            expiration_hours: URL validity in hours (default: 24)
            
        Returns:
            Signed URL string
            
        Raises:
            NotFound: If file doesn't exist
            GoogleCloudError: If generation fails
        """
        try:
            # Extract blob path
            blob_path = gcs_path.replace(f"gs://{self.bucket_name}/", "")
            blob = self.bucket.blob(blob_path)
            
            if not blob.exists():
                raise NotFound(f"File not found: {gcs_path}")
            
            # Generate signed URL
            expiration = datetime.now() + timedelta(hours=expiration_hours)
            
            signed_url = blob.generate_signed_url(
                version="v4",
                expiration=expiration,
                method="GET"
            )
            
            logger.info(f"Generated signed URL for: {gcs_path} (expires in {expiration_hours}h)")
            return signed_url
            
        except NotFound:
            raise
        except Exception as e:
            logger.error(f"Failed to generate signed URL: {e}")
            raise GoogleCloudError(f"Signed URL generation failed: {str(e)}")
    
    # ========================================================================
    # LIST & SEARCH OPERATIONS
    # ========================================================================
    
    def list_offers(
        self,
        limit: int = 100,
        prefix: Optional[str] = None
    ) -> List[Dict]:
        """
        List parsed offers in bucket
        
        Args:
            limit: Maximum number of offers to return
            prefix: Optional prefix filter (e.g., "parsed/OFF_202411")
            
        Returns:
            List of offer metadata dicts
        """
        try:
            search_prefix = prefix or GCSConfig.PARSED_PREFIX
            blobs = self.bucket.list_blobs(prefix=search_prefix, max_results=limit)
            
            offers = []
            for blob in blobs:
                if blob.name.endswith('.json'):
                    offer_info = {
                        'offer_id': blob.name.split('/')[-1].replace('.json', ''),
                        'gcs_path': f"gs://{self.bucket_name}/{blob.name}",
                        'size_bytes': blob.size,
                        'created': blob.time_created.isoformat() if blob.time_created else None,
                        'updated': blob.updated.isoformat() if blob.updated else None,
                        'metadata': blob.metadata or {}
                    }
                    offers.append(offer_info)
            
            logger.info(f"Listed {len(offers)} offers")
            return offers
            
        except Exception as e:
            logger.error(f"Failed to list offers: {e}")
            raise GoogleCloudError(f"List operation failed: {str(e)}")
    
    def list_attachments(self, offer_id: str) -> List[Dict]:
        """
        List attachments for an offer
        
        Args:
            offer_id: Offer ID
            
        Returns:
            List of attachment metadata dicts
        """
        try:
            prefix = f"{GCSConfig.ATTACHMENTS_PREFIX}/{offer_id}/"
            blobs = self.bucket.list_blobs(prefix=prefix)
            
            attachments = []
            for blob in blobs:
                attachment_info = {
                    'filename': blob.name.split('/')[-1],
                    'gcs_path': f"gs://{self.bucket_name}/{blob.name}",
                    'size_bytes': blob.size,
                    'content_type': blob.content_type,
                    'created': blob.time_created.isoformat() if blob.time_created else None,
                    'metadata': blob.metadata or {}
                }
                attachments.append(attachment_info)
            
            return attachments
            
        except Exception as e:
            logger.error(f"Failed to list attachments for {offer_id}: {e}")
            raise GoogleCloudError(f"List operation failed: {str(e)}")
    
    # ========================================================================
    # DELETE OPERATIONS
    # ========================================================================
    
    def delete_offer(self, offer_id: str, delete_attachments: bool = True) -> bool:
        """
        Delete offer and optionally its attachments
        
        Args:
            offer_id: Offer ID
            delete_attachments: Whether to delete attachments too (default: True)
            
        Returns:
            True if successful
            
        Raises:
            GoogleCloudError: If deletion fails
        """
        try:
            deleted_count = 0
            
            # Delete parsed JSON
            parsed_path = f"{GCSConfig.PARSED_PREFIX}/{offer_id}.json"
            parsed_blob = self.bucket.blob(parsed_path)
            if parsed_blob.exists():
                parsed_blob.delete()
                deleted_count += 1
            
            # Delete original file (search by offer_id in metadata or path)
            originals_prefix = f"{GCSConfig.ORIGINALS_PREFIX}/"
            for blob in self.bucket.list_blobs(prefix=originals_prefix):
                if offer_id in blob.name:
                    blob.delete()
                    deleted_count += 1
                    break
            
            # Delete attachments if requested
            if delete_attachments:
                attachments_prefix = f"{GCSConfig.ATTACHMENTS_PREFIX}/{offer_id}/"
                for blob in self.bucket.list_blobs(prefix=attachments_prefix):
                    blob.delete()
                    deleted_count += 1
            
            logger.info(f"Deleted offer {offer_id} ({deleted_count} files)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete offer {offer_id}: {e}")
            raise GoogleCloudError(f"Delete operation failed: {str(e)}")
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for GCS"""
        # Remove or replace problematic characters
        safe = filename.replace(" ", "_").replace("(", "").replace(")", "")
        # Keep only alphanumeric, dots, dashes, underscores
        safe = "".join(c for c in safe if c.isalnum() or c in "._-")
        return safe
    
    def _get_content_type(self, filename: str) -> str:
        """Get content type from filename"""
        ext = filename.lower().split('.')[-1]
        
        content_types = {
            'csv': 'text/csv',
            'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'xls': 'application/vnd.ms-excel',
            'json': 'application/json',
            'pdf': 'application/pdf',
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png',
            'txt': 'text/plain'
        }
        
        return content_types.get(ext, 'application/octet-stream')
    
    def _get_attachment_type(self, filename: str) -> str:
        """Classify attachment type"""
        ext = filename.lower().split('.')[-1]
        filename_lower = filename.lower()
        
        if ext == 'pdf':
            if 'contract' in filename_lower:
                return 'contract'
            elif 'invoice' in filename_lower:
                return 'invoice'
            else:
                return 'document'
        elif ext in ['jpg', 'jpeg', 'png']:
            return 'photo'
        elif ext in ['xlsx', 'xls', 'csv']:
            return 'spreadsheet'
        else:
            return 'other'
    
    def get_storage_stats(self) -> Dict:
        """
        Get storage statistics
        
        Returns:
            Dict with storage stats
        """
        try:
            stats = {
                'originals_count': 0,
                'parsed_count': 0,
                'attachments_count': 0,
                'total_size_bytes': 0,
                'bucket_name': self.bucket_name
            }
            
            # Count originals
            for blob in self.bucket.list_blobs(prefix=GCSConfig.ORIGINALS_PREFIX):
                stats['originals_count'] += 1
                stats['total_size_bytes'] += blob.size or 0
            
            # Count parsed
            for blob in self.bucket.list_blobs(prefix=GCSConfig.PARSED_PREFIX):
                stats['parsed_count'] += 1
                stats['total_size_bytes'] += blob.size or 0
            
            # Count attachments
            for blob in self.bucket.list_blobs(prefix=GCSConfig.ATTACHMENTS_PREFIX):
                stats['attachments_count'] += 1
                stats['total_size_bytes'] += blob.size or 0
            
            # Convert to MB
            stats['total_size_mb'] = round(stats['total_size_bytes'] / (1024 * 1024), 2)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {}