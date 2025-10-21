"""
DemoPLAN Training API
Endpoints for batch upload, ML training operations, and offer management
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from pydantic import BaseModel

from src.services.batch_processor import batch_processor, BatchStatus
from src.services.offer_learning_service import OfferLearningService

logger = logging.getLogger("demoplan.api.training")

# Initialize offer learning service
offer_learning = OfferLearningService()


# ============================================================================
# EXISTING REQUEST/RESPONSE MODELS
# ============================================================================

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


# ============================================================================
# NEW OFFER MANAGEMENT MODELS
# ============================================================================

class OfferIngestionResponse(BaseModel):
    success: bool
    offer_id: str
    message: str
    statistics: Dict[str, Any]
    storage: Dict[str, Any]
    issues: Dict[str, List[str]]

class OfferListResponse(BaseModel):
    offers: List[Dict[str, Any]]
    total: int

class OfferDetailResponse(BaseModel):
    offer_id: str
    project: Dict[str, Any]
    cost_breakdown: Dict[str, Any]
    parsed_at: str
    detail_level: str
    source_filename: str

class LearningResponse(BaseModel):
    success: bool
    message: str
    offers_processed: int
    patterns_learned: Dict[str, Any]
    confidence_score: float
    errors: List[str]

class PatternsResponse(BaseModel):
    category_patterns: Optional[Dict[str, Any]]
    item_frequencies: Optional[Dict[str, Any]]
    unit_prices: Optional[Dict[str, Any]]

class DashboardStatsResponse(BaseModel):
    total_offers_ingested: int
    offers_with_unit_pricing: int
    last_ingestion: Optional[str]
    last_learning_run: Optional[str]
    confidence_score: float
    category_patterns_count: int
    item_types_tracked: int


# ============================================================================
# ROUTER SETUP
# ============================================================================

training_router = APIRouter(prefix="/api/ml", tags=["training"])


# ============================================================================
# EXISTING BATCH TRAINING ENDPOINTS
# ============================================================================

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


# ============================================================================
# NEW OFFER MANAGEMENT ENDPOINTS
# ============================================================================

@training_router.post("/offers/ingest-csv", response_model=OfferIngestionResponse)
async def ingest_offer_csv(
    file: UploadFile = File(...),
    project_type: str = Form(default="commercial"),
    region: str = Form(default="bucuresti")
):
    """
    Upload and ingest a single offer CSV/Excel file
    
    Supports:
    - Imperial Brands format (hierarchical with A/B categories)
    - Beautik format (flat list, auto-categorized)
    - CCC format (multi-sheet Excel with unit prices)
    """
    try:
        # Validate file type
        allowed_extensions = {'.csv', '.xlsx', '.xls'}
        file_ext = '.' + file.filename.split('.')[-1].lower() if '.' in file.filename else ''
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Read file content
        content = await file.read()
        
        # Prepare metadata
        metadata = {
            "project_type": project_type,
            "region": region,
            "uploaded_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Ingest offer
        logger.info(f"Ingesting offer: {file.filename}")
        result = offer_learning.ingest_offer(content, file.filename, metadata)
        
        # Convert IngestionResult to response
        return OfferIngestionResponse(
            success=result.success,
            offer_id=result.offer_id,
            message=result.message,
            statistics={
                'categories_found': result.categories_found,
                'items_extracted': result.items_extracted,
                'items_with_unit_pricing': result.items_with_unit_pricing,
                'total_eur': result.total_eur,
                'detail_level': result.detail_level
            },
            storage={
                'gcs_original_path': result.gcs_original_path,
                'gcs_parsed_path': result.gcs_parsed_path,
                'firestore_doc_id': result.firestore_doc_id
            },
            issues={
                'errors': result.errors,
                'warnings': result.warnings
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Offer ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@training_router.get("/offers/list", response_model=OfferListResponse)
async def list_offers(
    limit: int = 50,
    detail_level: Optional[str] = None
):
    """
    List ingested offers with optional filters
    
    Query params:
    - limit: Max results (default: 50)
    - detail_level: Filter by 'summary' or 'unit_prices'
    """
    try:
        offers = offer_learning.list_offers(limit=limit, detail_level=detail_level)
        
        return OfferListResponse(
            offers=offers,
            total=len(offers)
        )
        
    except Exception as e:
        logger.error(f"List offers failed: {e}")
        raise HTTPException(status_code=500, detail=f"List failed: {str(e)}")


@training_router.get("/offers/{offer_id}", response_model=OfferDetailResponse)
async def get_offer_details(offer_id: str):
    """
    Get complete offer details by ID
    
    Returns full parsed offer data from GCS
    """
    try:
        offer_data = offer_learning.get_offer_by_id(offer_id)
        
        if not offer_data:
            raise HTTPException(status_code=404, detail=f"Offer {offer_id} not found")
        
        return OfferDetailResponse(
            offer_id=offer_data.get('offer_id'),
            project=offer_data.get('project', {}),
            cost_breakdown=offer_data.get('cost_breakdown', {}),
            parsed_at=offer_data.get('parsed_at', ''),
            detail_level=offer_data.get('detail_level', 'summary'),
            source_filename=offer_data.get('source_filename', '')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get offer details failed: {e}")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")


@training_router.post("/offers/learn", response_model=LearningResponse)
async def trigger_learning():
    """
    Manually trigger pattern learning
    
    Processes ALL ingested offers and calculates:
    - Category distribution patterns (A: 72%, B: 28%, etc.)
    - Item frequency patterns (HVAC in 95% of offers)
    - Unit price ranges (Plastering: €5-6/m²)
    - Confidence scores based on sample size
    
    YOU control when this runs - no automatic triggers
    """
    try:
        logger.info("Manual learning trigger initiated")
        
        # Run learning process
        result = offer_learning.run_learning()
        
        return LearningResponse(
            success=result['success'],
            message=result.get('message', ''),
            offers_processed=result.get('offers_processed', 0),
            patterns_learned=result.get('patterns_learned', {}),
            confidence_score=result.get('confidence_score', 0.0),
            errors=result.get('errors', [])
        )
        
    except Exception as e:
        logger.error(f"Learning process failed: {e}")
        raise HTTPException(status_code=500, detail=f"Learning failed: {str(e)}")


@training_router.get("/patterns/commercial", response_model=PatternsResponse)
async def get_commercial_patterns():
    """
    View all learned patterns for commercial projects
    
    Returns:
    - Category splits (Architectural vs MEP percentages)
    - Item frequencies (which items appear most often)
    - Unit price ranges (typical costs per unit)
    """
    try:
        patterns = offer_learning.get_patterns()
        
        return PatternsResponse(
            category_patterns=patterns.get('category_patterns'),
            item_frequencies=patterns.get('item_frequencies'),
            unit_prices=patterns.get('unit_prices')
        )
        
    except Exception as e:
        logger.error(f"Get patterns failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pattern retrieval failed: {str(e)}")


@training_router.get("/stats/dashboard", response_model=DashboardStatsResponse)
async def get_dashboard_stats():
    """
    System status overview for monitoring
    
    Returns:
    - Total offers ingested
    - Offers with detailed unit pricing
    - Last ingestion/learning dates
    - Current confidence score
    - Pattern counts
    
    Use this to monitor system health - NO automatic recommendations
    """
    try:
        stats = offer_learning.get_learning_stats()
        
        # Get last ingestion date from most recent offer
        offers = offer_learning.list_offers(limit=1)
        last_ingestion = offers[0].get('ingestion_date') if offers else None
        
        return DashboardStatsResponse(
            total_offers_ingested=stats.get('total_offers_ingested', 0),
            offers_with_unit_pricing=stats.get('offers_with_unit_pricing', 0),
            last_ingestion=last_ingestion,
            last_learning_run=stats.get('last_learning_run'),
            confidence_score=stats.get('confidence_score', 0.0),
            category_patterns_count=stats.get('category_patterns_count', 0),
            item_types_tracked=stats.get('item_types_tracked', 0)
        )
        
    except Exception as e:
        logger.error(f"Get dashboard stats failed: {e}")
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")