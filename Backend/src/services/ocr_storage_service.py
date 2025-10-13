"""
DemoPLAN OCR Storage Service
Dedicated service for OCR result persistence and retrieval
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone, timedelta

from src.services.firestore_service import FirestoreService
from src.models.ocr_models import OCRResult

logger = logging.getLogger("demoplan.services.ocr_storage")


class OCRStorageService:
    """
    Wrapper service for OCR data storage operations
    Provides high-level interface for saving/loading OCR results
    """
    
    def __init__(self, firestore_service: FirestoreService):
        """
        Initialize OCR storage service
        
        Args:
            firestore_service: Initialized Firestore service instance
        """
        self.firestore = firestore_service
        logger.info("📦 OCR Storage Service initialized")
    
    
    async def save_ocr_result(
        self,
        ocr_result: OCRResult,
        file_id: str,
        filename: str
    ) -> bool:
        """
        Save OCR result to storage
        
        Args:
            ocr_result: Complete OCRResult object
            file_id: Unique file identifier
            filename: Original filename
            
        Returns:
            Success status
        """
        try:
            logger.info(f"💾 Saving OCR result: {file_id} ({filename})")
            
            # Serialize OCR result
            ocr_data = self._serialize_ocr_result(ocr_result)
            
            # Save to Firestore
            success = await self.firestore.save_ocr_result(
                file_id=file_id,
                ocr_data=ocr_data,
                file_name=filename
            )
            
            if success:
                logger.info(f"✅ OCR result saved: {file_id}")
            else:
                logger.error(f"❌ Failed to save OCR result: {file_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ OCR save error for {file_id}: {e}", exc_info=True)
            return False
    
    
    async def load_ocr_result(
        self,
        file_id: str
    ) -> Optional[OCRResult]:
        """
        Load OCR result from storage
        
        Args:
            file_id: Unique file identifier
            
        Returns:
            OCRResult object or None if not found
        """
        try:
            logger.info(f"📂 Loading OCR result: {file_id}")
            
            # Retrieve from Firestore
            ocr_data = await self.firestore.get_ocr_result(file_id)
            
            if not ocr_data:
                logger.warning(f"⚠️ OCR result not found: {file_id}")
                return None
            
            # Deserialize to OCRResult
            ocr_result = self._deserialize_ocr_result(ocr_data)
            
            logger.info(f"✅ OCR result loaded: {file_id}")
            return ocr_result
            
        except Exception as e:
            logger.error(f"❌ OCR load error for {file_id}: {e}", exc_info=True)
            return None
    
    
    async def delete_ocr_result(
        self,
        file_id: str
    ) -> bool:
        """
        Delete OCR result from storage
        
        Args:
            file_id: Unique file identifier
            
        Returns:
            Success status
        """
        try:
            logger.info(f"🗑️ Deleting OCR result: {file_id}")
            
            success = await self.firestore.delete_ocr_result(file_id)
            
            if success:
                logger.info(f"✅ OCR result deleted: {file_id}")
            else:
                logger.warning(f"⚠️ OCR result not found or delete failed: {file_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ OCR delete error for {file_id}: {e}", exc_info=True)
            return False
    
    
    async def cleanup_old_results(
        self,
        days: int = 90
    ) -> Dict[str, Any]:
        """
        Clean up OCR results older than specified days
        
        Args:
            days: Number of days to retain results
            
        Returns:
            Dictionary with cleanup statistics
        """
        try:
            logger.info(f"🧹 Starting OCR cleanup (older than {days} days)")
            
            result = await self.firestore.cleanup_old_ocr_results(days)
            
            deleted_count = result.get("deleted_count", 0)
            logger.info(f"✅ Cleanup complete: {deleted_count} results removed")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ OCR cleanup failed: {e}", exc_info=True)
            return {"deleted_count": 0, "error": str(e)}
    
    
    async def get_ocr_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about stored OCR results
        
        Returns:
            Dictionary with OCR storage statistics
        """
        try:
            logger.info("📊 Fetching OCR statistics")
            
            # Query all OCR results (limited sample)
            ocr_docs = await self.firestore.query_documents(
                collection="ocr_results",
                limit=1000
            )
            
            total_count = len(ocr_docs)
            
            # Calculate statistics
            total_pages = 0
            total_cost = 0.0
            avg_confidence = 0.0
            
            for doc in ocr_docs:
                ocr_data = doc.get("ocr_data", {})
                total_pages += ocr_data.get("page_count", 0)
                total_cost += ocr_data.get("cost_estimate", 0.0)
                avg_confidence += ocr_data.get("confidence", 0.0)
            
            if total_count > 0:
                avg_confidence = avg_confidence / total_count
            
            stats = {
                "total_ocr_results": total_count,
                "total_pages_processed": total_pages,
                "total_cost_usd": round(total_cost, 2),
                "average_confidence": round(avg_confidence, 2),
                "avg_pages_per_doc": round(total_pages / total_count, 1) if total_count > 0 else 0
            }
            
            logger.info(f"✅ OCR statistics: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to get OCR statistics: {e}", exc_info=True)
            return {
                "error": str(e),
                "total_ocr_results": 0
            }
    
    
    # =========================================================================
    # PRIVATE HELPER METHODS
    # =========================================================================
    
    def _serialize_ocr_result(self, ocr_result: OCRResult) -> Dict[str, Any]:
        """
        Serialize OCRResult to dictionary
        
        Args:
            ocr_result: OCRResult object
            
        Returns:
            Dictionary representation
        """
        return ocr_result.to_dict()
    
    
    def _deserialize_ocr_result(self, data: Dict[str, Any]) -> OCRResult:
        """
        Deserialize dictionary to OCRResult
        
        Args:
            data: Dictionary data
            
        Returns:
            OCRResult object
        """
        return OCRResult.from_dict(data)