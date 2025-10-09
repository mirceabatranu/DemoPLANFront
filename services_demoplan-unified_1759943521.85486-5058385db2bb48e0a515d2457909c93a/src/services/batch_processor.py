"""
DemoPLAN Batch Processing Service
Handles batch uploads and training data processing for ML learning
Enhanced with GCS storage for large files
"""

import asyncio
import logging
import json
import uuid
import base64
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import io

from src.services.firestore_service import FirestoreService
from src.services.llm_service import safe_construction_call
from src.processors.dxf_analyzer import UnifiedDocumentProcessor

logger = logging.getLogger("demoplan.services.batch_processor")

class BatchStatus(Enum):
    """Batch processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class BatchUpload:
    """Batch upload data structure"""
    batch_id: str
    files: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: BatchStatus = BatchStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    processed_at: Optional[datetime] = None
    error_message: Optional[str] = None

@dataclass
class TrainingOffer:
    """Training offer data structure"""
    project_id: str
    offer_text: str
    project_type: str
    area_sqm: Optional[float] = None
    rooms: Optional[int] = None
    final_cost_ron: Optional[float] = None
    timeline_days: Optional[int] = None
    client_satisfaction: Optional[float] = None
    materials: List[str] = field(default_factory=list)
    complexity_factors: List[str] = field(default_factory=list)
    regional_data: Dict[str, Any] = field(default_factory=dict)

class BatchProcessor:
    """Handles batch processing operations"""
    
    def __init__(self):
        self.firestore = FirestoreService()
        self.document_processor = UnifiedDocumentProcessor()
        self.learning_engine = None
        self.active_batches = {}
        
        # Initialize GCS Storage for large files
        try:
            from google.cloud import storage
            self.storage_client = storage.Client()
            self.bucket_name = "demoplan-training-files"
            logger.info("✅ GCS storage initialized for batch files")
        except Exception as e:
            logger.error(f"⚠️ GCS storage not available: {e}")
            self.storage_client = None
            self.bucket_name = None
        
    async def initialize(self, learning_engine: Optional[Any] = None):
        """Initialize batch processor"""
        try:
            await self.firestore.initialize()
            
            # Link learning engine if provided
            try:
                self.learning_engine = learning_engine
                logger.info("Learning Engine initialized for batch processing")
            except Exception as e:
                logger.warning(f"Learning Engine not available: {e}")
            
            logger.info("Batch Processor initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Batch Processor: {e}")
            raise

    async def upload_batch(
        self, 
        files: List[Dict[str, Any]], 
        metadata: Dict[str, Any] = None
    ) -> str:
        """Upload a batch of files for processing"""
        try:
            batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            
            # Process and store files
            processed_files = []
            for i, file_info in enumerate(files):
                processed_file = {
                    "file_id": f"{batch_id}_{i:03d}",
                    "filename": file_info.get("filename", f"file_{i}"),
                    "content_type": file_info.get("content_type", "unknown"),
                    "size": len(file_info.get("content", b"")),
                    "uploaded_at": datetime.now(timezone.utc).isoformat()
                }
                
                # Store file content separately for processing
                if "content" in file_info:
                    await self._store_file_content(processed_file["file_id"], file_info["content"])
                
                processed_files.append(processed_file)
            
            # Create batch record
            batch_upload = BatchUpload(
                batch_id=batch_id,
                files=processed_files,
                metadata=metadata or {},
                status=BatchStatus.PENDING
            )
            
            # Save to database
            await self.firestore.save_document(
                "training_batches",
                batch_id,
                {
                    "batch_id": batch_upload.batch_id,
                    "files": batch_upload.files,
                    "metadata": batch_upload.metadata,
                    "status": batch_upload.status.value,
                    "created_at": batch_upload.created_at,
                    "processed_at": batch_upload.processed_at,
                    "error_message": batch_upload.error_message
                }
            )
            
            self.active_batches[batch_id] = batch_upload
            logger.info(f"Batch {batch_id} uploaded with {len(processed_files)} files")
            
            return batch_id
            
        except Exception as e:
            logger.error(f"Failed to upload batch: {e}")
            raise

    async def process_batch(self, batch_id: str) -> Dict[str, Any]:
        """Process a batch of uploaded files"""
        try:
            # Load batch data
            batch_data = await self.firestore.get_document("training_batches", batch_id)
            if not batch_data:
                raise ValueError(f"Batch {batch_id} not found")
            
            # Update status to processing
            batch_data["status"] = BatchStatus.PROCESSING.value
            batch_data["processing_started_at"] = datetime.now(timezone.utc).isoformat()
            await self.firestore.update_document("training_batches", batch_id, batch_data)
            
            processing_results = {
                "batch_id": batch_id,
                "files_processed": 0,
                "training_data_extracted": 0,
                "patterns_learned": 0,
                "errors": []
            }
            
            # Process each file in the batch
            for file_info in batch_data.get("files", []):
                try:
                    file_id = file_info["file_id"]
                    filename = file_info["filename"]
                    
                    # Load file content
                    content = await self._load_file_content(file_id)
                    if not content:
                        processing_results["errors"].append(f"No content for {filename}")
                        continue
                    
                    # Process based on file type
                    if filename.lower().endswith(('.dxf', '.pdf', '.txt', '.csv', '.xlsx', '.xls', '.json')):
                        document_result = self.document_processor.process_document(filename, content)
                        
                        if document_result.combined_confidence > 0.3:
                            # Extract training data
                            training_data = await self._extract_training_data(
                                document_result, filename, batch_data.get("metadata", {})
                            )
                            
                            if training_data:
                                # Save training data
                                training_id = f"training_{batch_id}_{file_id}"
                                await self.firestore.save_document(
                                    "training_data", training_id, training_data
                                )
                                processing_results["training_data_extracted"] += 1
                    
                    elif filename.lower().endswith(('.json', '.txt')) and 'offer' in filename.lower():
                        # Process as historical offer
                        offer_data = await self._process_historical_offer(content, filename)
                        if offer_data:
                            offer_id = f"offer_{batch_id}_{file_id}"
                            await self.firestore.save_document(
                                "historical_offers", offer_id, offer_data
                            )
                            processing_results["training_data_extracted"] += 1
                    
                    processing_results["files_processed"] += 1
                    
                except Exception as e:
                    logger.error(f"Failed to process file {file_info.get('filename')}: {e}")
                    processing_results["errors"].append(f"{file_info.get('filename')}: {str(e)}")
            
            # Apply learning if learning engine is available
            if self.learning_engine and processing_results["training_data_extracted"] > 0:
                try:
                    learning_result = await self._apply_batch_learning(batch_id, processing_results)
                    processing_results["patterns_learned"] = learning_result.get("patterns_learned", 0)
                except Exception as e:
                    logger.error(f"Learning application failed: {e}")
                    processing_results["errors"].append(f"Learning failed: {str(e)}")
            
            # Update batch status
            batch_data["status"] = BatchStatus.COMPLETED.value if not processing_results["errors"] else BatchStatus.FAILED.value
            batch_data["processed_at"] = datetime.now(timezone.utc).isoformat()
            batch_data["processing_results"] = processing_results
            
            if processing_results["errors"]:
                batch_data["error_message"] = f"{len(processing_results['errors'])} processing errors"
            
            await self.firestore.update_document("training_batches", batch_id, batch_data)
            
            logger.info(f"Batch {batch_id} processed: {processing_results['files_processed']} files, "
                       f"{processing_results['training_data_extracted']} training data extracted")
            
            return processing_results
            
        except Exception as e:
            logger.error(f"Failed to process batch {batch_id}: {e}")
            # Update batch status to failed
            try:
                await self.firestore.update_document("training_batches", batch_id, {
                    "status": BatchStatus.FAILED.value,
                    "error_message": str(e),
                    "processed_at": datetime.now(timezone.utc).isoformat()
                })
            except:
                pass
            raise

    async def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """Get batch processing status"""
        try:
            batch_data = await self.firestore.get_document("training_batches", batch_id)
            if not batch_data:
                return {"error": "Batch not found"}
            
            return {
                "batch_id": batch_id,
                "status": batch_data.get("status", "unknown"),
                "created_at": batch_data.get("created_at"),
                "processed_at": batch_data.get("processed_at"),
                "files_count": len(batch_data.get("files", [])),
                "processing_results": batch_data.get("processing_results", {}),
                "error_message": batch_data.get("error_message")
            }
            
        except Exception as e:
            logger.error(f"Failed to get batch status: {e}")
            return {"error": str(e)}

    async def retrain_models(self, model_types: List[str] = None) -> Dict[str, Any]:
        """Retrain ML models with latest training data"""
        try:
            if not self.learning_engine:
                return {"error": "Learning engine not available"}
            
            # Get recent training data
            training_data = await self.firestore.query_documents(
                "training_data",
                limit=100,
                order_by="created_at"
            )
            
            historical_offers = await self.firestore.query_documents(
                "historical_offers",
                limit=50,
                order_by="created_at"
            )
            
            if not training_data and not historical_offers:
                return {"error": "No training data available"}
            
            retrain_results = {
                "training_data_used": len(training_data),
                "historical_offers_used": len(historical_offers),
                "models_updated": 0,
                "improvements": []
            }
            
            # Process training data for pattern learning
            for data in training_data:
                try:
                    # Apply learning from training data
                    if data.get("document_type") in ["dxf", "pdf", "txt"]:
                        await self._apply_document_learning(data)
                        retrain_results["models_updated"] += 1
                except Exception as e:
                    logger.error(f"Failed to apply training data: {e}")
            
            # Process historical offers
            for offer in historical_offers:
                try:
                    await self._apply_offer_learning(offer)
                    retrain_results["models_updated"] += 1
                except Exception as e:
                    logger.error(f"Failed to apply offer learning: {e}")
            
            # Update model metadata
            model_metadata = {
                "last_retrain": datetime.now(timezone.utc).isoformat(),
                "training_data_count": len(training_data),
                "historical_offers_count": len(historical_offers),
                "retrain_results": retrain_results
            }
            
            await self.firestore.save_document("model_metadata", "latest_retrain", model_metadata)
            
            logger.info(f"Model retraining completed: {retrain_results['models_updated']} updates applied")
            
            return retrain_results
            
        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
            return {"error": str(e)}

    # Private helper methods - UPDATED FOR GCS
    
    async def _store_file_content(self, file_id: str, content: bytes):
        """Store file content in GCS for large files, Firestore for small ones"""
        try:
            content_size = len(content)
            
            # Use GCS for files > 500KB (safety margin under 1MB Firestore limit)
            if content_size > 500 * 1024 and self.storage_client:
                bucket = self.storage_client.bucket(self.bucket_name)
                blob = bucket.blob(f"batch_files/{file_id}")
                blob.upload_from_string(content)
                
                # Store metadata in Firestore pointing to GCS
                await self.firestore.save_document(
                    "file_contents",
                    file_id,
                    {
                        "storage_type": "gcs",
                        "bucket": self.bucket_name,
                        "blob_path": f"batch_files/{file_id}",
                        "size_bytes": content_size,
                        "stored_at": datetime.now(timezone.utc).isoformat()
                    }
                )
                logger.info(f"📦 Stored large file {file_id} in GCS ({content_size / 1024:.1f} KB)")
                
            else:
                # Small files can go in Firestore
                await self.firestore.save_document(
                    "file_contents",
                    file_id,
                    {
                        "storage_type": "firestore",
                        "content": base64.b64encode(content).decode('utf-8'),
                        "size_bytes": content_size,
                        "stored_at": datetime.now(timezone.utc).isoformat()
                    }
                )
                logger.info(f"📦 Stored small file {file_id} in Firestore ({content_size / 1024:.1f} KB)")
                
        except Exception as e:
            logger.error(f"❌ Failed to store file content for {file_id}: {e}")
            raise

    async def _load_file_content(self, file_id: str) -> Optional[bytes]:
        """Load file content from GCS or Firestore"""
        try:
            # Get metadata from Firestore
            metadata = await self.firestore.get_document("file_contents", file_id)
            
            if not metadata:
                logger.error(f"❌ No metadata found for file {file_id}")
                return None
            
            storage_type = metadata.get("storage_type", "firestore")
            
            if storage_type == "gcs" and self.storage_client:
                # Load from GCS
                bucket = self.storage_client.bucket(metadata["bucket"])
                blob = bucket.blob(metadata["blob_path"])
                content = blob.download_as_bytes()
                logger.info(f"📥 Loaded {file_id} from GCS ({len(content) / 1024:.1f} KB)")
                return content
                
            elif storage_type == "firestore":
                # Load from Firestore
                content_b64 = metadata.get("content")
                if not content_b64:
                    logger.error(f"❌ No content in Firestore for {file_id}")
                    return None
                content = base64.b64decode(content_b64)
                logger.info(f"📥 Loaded {file_id} from Firestore ({len(content) / 1024:.1f} KB)")
                return content
                
            else:
                logger.error(f"❌ Unknown storage type '{storage_type}' for {file_id}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to load file content for {file_id}: {e}")
            return None

    async def _extract_training_data(
        self, 
        document_result, 
        filename: str, 
        metadata: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Extract training data from document analysis"""
        try:
            training_data = {
                "filename": filename,
                "document_type": document_result.document_type,
                "confidence": document_result.combined_confidence,
                "extracted_at": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata
            }
            
            # Add type-specific data
            if document_result.dxf_analysis:
                dxf_data = document_result.dxf_analysis.get("dxf_analysis", {})
                training_data["spatial_data"] = {
                    "total_rooms": dxf_data.get("total_rooms", 0),
                    "total_area": dxf_data.get("total_area", 0),
                    "room_breakdown": dxf_data.get("room_breakdown", []),
                    "has_dimensions": dxf_data.get("has_dimensions", False)
                }
            
            if document_result.pdf_analysis:
                training_data["pdf_data"] = {
                    "construction_specs": document_result.pdf_analysis.construction_specs,
                    "materials": document_result.pdf_analysis.material_references,
                    "regulatory_info": document_result.pdf_analysis.regulatory_info
                }
            
            if document_result.txt_analysis:
                training_data["text_data"] = {
                    "requirements": document_result.txt_analysis.requirements,
                    "preferences": document_result.txt_analysis.client_preferences,
                    "keywords": document_result.txt_analysis.construction_keywords
                }
            
            return training_data
            
        except Exception as e:
            logger.error(f"Failed to extract training data: {e}")
            return None

    async def _process_historical_offer(self, content: bytes, filename: str) -> Optional[Dict[str, Any]]:
        """Process historical offer data"""
        try:
            # Try to parse as JSON first
            if isinstance(content, bytes):
                content_str = content.decode('utf-8', errors='ignore')
            else:
                content_str = str(content)
            
            try:
                data = json.loads(content_str)
                if isinstance(data, dict) and "offers" in data:
                    # Batch format
                    offers = data["offers"]
                elif isinstance(data, dict):
                    # Single offer format
                    offers = [data]
                elif isinstance(data, list):
                    # List of offers
                    offers = data
                else:
                    return None
            except json.JSONDecodeError:
                # Try to extract offer data from plain text
                offers = await self._extract_offer_from_text(content_str)
                if not offers:
                    return None
            
            processed_offers = []
            for offer in offers:
                processed_offer = {
                    "project_id": offer.get("project_id", f"extracted_{uuid.uuid4().hex[:8]}"),
                    "offer_text": offer.get("offer_text", content_str[:1000]),
                    "project_type": offer.get("project_type", "unknown"),
                    "area_sqm": offer.get("area_sqm"),
                    "rooms": offer.get("rooms"),
                    "final_cost_ron": offer.get("final_cost_ron"),
                    "timeline_days": offer.get("timeline_days"),
                    "client_satisfaction": offer.get("client_satisfaction"),
                    "materials": offer.get("materials", []),
                    "complexity_factors": offer.get("complexity_factors", []),
                    "regional_data": offer.get("regional_data", {}),
                    "extracted_from": filename,
                    "processed_at": datetime.now(timezone.utc).isoformat()
                }
                processed_offers.append(processed_offer)
            
            return {
                "offers": processed_offers,
                "filename": filename,
                "processed_at": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to process historical offer: {e}")
            return None

    async def _extract_offer_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract offer data from plain text using LLM"""
        try:
            prompt = f"""Extrage informatii despre oferte de constructii din urmatorul text:
{text[:2000]}

Returneaza JSON cu urmatoarea structura:
[{{
    "project_type": "tip_proiect",
    "area_sqm": numar_sau_null,
    "rooms": numar_sau_null,
    "final_cost_ron": numar_sau_null,
    "timeline_days": numar_sau_null,
    "materials": ["lista", "materiale"],
    "offer_text": "text_relevant_oferta"
}}]"""
            
            response = await safe_construction_call(
                user_input=prompt,
                domain="offer_extraction"
            )
            
            try:
                return json.loads(response)
            except:
                return []
                
        except Exception as e:
            logger.error(f"Failed to extract offer from text: {e}")
            return []

    async def _apply_batch_learning(self, batch_id: str, processing_results: Dict[str, Any]) -> Dict[str, Any]:
        """Apply learning from batch processing results"""
        try:
            if not self.learning_engine:
                return {"patterns_learned": 0}
            
            # This would integrate with the learning engine
            # For now, return basic results
            return {
                "patterns_learned": processing_results.get("training_data_extracted", 0),
                "batch_id": batch_id
            }
            
        except Exception as e:
            logger.error(f"Failed to apply batch learning: {e}")
            return {"patterns_learned": 0, "error": str(e)}

    async def _apply_document_learning(self, training_data: Dict[str, Any]):
        """Apply learning from document training data"""
        try:
            # This would feed training data to the learning engine
            # Implementation depends on the learning engine interface
            pass
        except Exception as e:
            logger.error(f"Failed to apply document learning: {e}")

    async def _apply_offer_learning(self, offer_data: Dict[str, Any]):
        """Apply learning from historical offer data"""
        try:
            # This would feed offer data to the learning engine
            # Implementation depends on the learning engine interface
            pass
        except Exception as e:
            logger.error(f"Failed to apply offer learning: {e}")

# Global instance
batch_processor = BatchProcessor()