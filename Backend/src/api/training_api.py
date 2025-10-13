"""
DemoPLAN Training API
Endpoints for batch upload and ML training operations
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from pydantic import BaseModel

from src.services.batch_processor import batch_processor, BatchStatus

logger = logging.getLogger("demoplan.api.training")

# Request/Response models
class BatchUploadResponse(BaseModel):
    batch_id: str
    message: str
    files_uploaded: int
    
class BatchStatusResponse(BaseModel):
    batch_id: str
    status: str
    created_at: Optional[str]
    processed_at: Optional[str]
    files_count: int
    processing_results: Optional[Dict[str, Any]]
    error_message: Optional[str]

class ProcessBatchResponse(BaseModel):
    batch_id: str
    status: str
    files_processed: int
    training_data_extracted: int
    patterns_learned: int
    errors: List[str]

class RetrainResponse(BaseModel):
    success: bool
    training_data_used: int
    historical_offers_used: int
    models_updated: int
    message: str

# Router setup
training_router = APIRouter(prefix="/api/ml", tags=["training"])

@training_router.post("/training/batch-upload", response_model=BatchUploadResponse)
async def batch_upload(
    files: List[UploadFile] = File(...),
    project_type: str = Form(default="mixed"),
    region: str = Form(default="bucuresti"),
    description: str = Form(default="")
):
    """Upload historical Romanian offers for batch processing"""
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        # Validate file types
        allowed_types = {'.json', '.txt', '.dxf', '.pdf', '.csv', '.xlsx', '.xls'}
        for file_item in files:
            file_ext = '.' + file_item.filename.split('.')[-1].lower() if '.' in file_item.filename else ''
            if file_ext not in allowed_types:
                raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_ext}. Allowed: {', '.join(allowed_types)}"
            )
        
        # Prepare files for batch processing
        files_data = []
        for file in files:
            content = await file.read()
            files_data.append({
                "filename": file.filename,
                "content_type": file.content_type,
                "content": content
            })
        
        # Metadata for the batch
        metadata = {
            "project_type": project_type,
            "region": region,
            "description": description,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "upload_source": "training_api"
        }
        
        # Process batch upload
        batch_id = await batch_processor.upload_batch(files_data, metadata)
        
        logger.info(f"Batch upload successful: {batch_id} with {len(files)} files")
        
        return BatchUploadResponse(
            batch_id=batch_id,
            message=f"Batch uploaded successfully. {len(files)} files queued for processing.",
            files_uploaded=len(files)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@training_router.post("/training/process-batch", response_model=ProcessBatchResponse)
async def process_batch(
    batch_id: str = Form(...)
):
    """Process training data into templates"""
    try:
        # Check if batch exists
        batch_status = await batch_processor.get_batch_status(batch_id)
        if "error" in batch_status:
            raise HTTPException(status_code=404, detail=batch_status["error"])
        
        if batch_status["status"] == BatchStatus.PROCESSING.value:
            raise HTTPException(status_code=409, detail="Batch is already being processed")
        
        if batch_status["status"] == BatchStatus.COMPLETED.value:
            raise HTTPException(status_code=409, detail="Batch has already been processed")
        
        logger.info(f"Starting batch processing for {batch_id}")
        
        # Process the batch
        processing_results = await batch_processor.process_batch(batch_id)
        
        return ProcessBatchResponse(
            batch_id=batch_id,
            status="completed" if not processing_results.get("errors") else "completed_with_errors",
            files_processed=processing_results.get("files_processed", 0),
            training_data_extracted=processing_results.get("training_data_extracted", 0),
            patterns_learned=processing_results.get("patterns_learned", 0),
            errors=processing_results.get("errors", [])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@training_router.get("/training/status", response_model=BatchStatusResponse)
async def get_training_status(batch_id: str):
    """Monitor processing status"""
    try:
        batch_status = await batch_processor.get_batch_status(batch_id)
        
        if "error" in batch_status:
            raise HTTPException(status_code=404, detail=batch_status["error"])
        
        return BatchStatusResponse(
            batch_id=batch_status["batch_id"],
            status=batch_status["status"],
            created_at=batch_status.get("created_at"),
            processed_at=batch_status.get("processed_at"),
            files_count=batch_status.get("files_count", 0),
            processing_results=batch_status.get("processing_results"),
            error_message=batch_status.get("error_message")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@training_router.post("/models/retrain", response_model=RetrainResponse)
async def retrain_models(
    model_types: Optional[str] = Form(default="all")
):
    """Update offer generation models"""
    try:
        logger.info("Starting model retraining process")
        
        # Parse model types
        if model_types == "all":
            models_to_retrain = None
        else:
            models_to_retrain = [t.strip() for t in model_types.split(",")]
        
        # Execute retraining
        retrain_results = await batch_processor.retrain_models(models_to_retrain)
        
        if "error" in retrain_results:
            raise HTTPException(status_code=500, detail=retrain_results["error"])
        
        success_message = f"Models retrained successfully. {retrain_results.get('models_updated', 0)} updates applied."
        
        return RetrainResponse(
            success=True,
            training_data_used=retrain_results.get("training_data_used", 0),
            historical_offers_used=retrain_results.get("historical_offers_used", 0),
            models_updated=retrain_results.get("models_updated", 0),
            message=success_message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model retraining failed: {e}")
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

# Additional utility endpoints

@training_router.get("/training/batches")
async def list_batches(limit: int = 10, status: Optional[str] = None):
    """List recent training batches"""
    try:
        # This would query recent batches from Firestore
        # Implementation depends on your Firestore service query capabilities
        return {
            "batches": [],
            "message": "Batch listing not yet implemented"
        }
    except Exception as e:
        logger.error(f"Batch listing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@training_router.get("/models/status")
async def get_models_status():
    """Get current ML models status"""
    try:
        # This would return current model metadata
        return {
            "models_status": "active",
            "last_retrain": None,
            "training_data_count": 0,
            "message": "Model status tracking not yet implemented"
        }
    except Exception as e:
        logger.error(f"Model status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))