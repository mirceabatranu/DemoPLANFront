# src/main_api.py
"""
DEMOPLAN Unified - Main API
Phase 1 deployment with full UnifiedConstructionAgent capabilities
Fixed circular import issues without reducing functionality
Updated with GCS file storage integration
"""

import os
import sys
import logging
import asyncio
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Dict, List, Any, Optional

# Core FastAPI imports
from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Add near other imports
from src.agents.drawing_generation_agent import DrawingGenerationAgent

# Import flags - initialized properly
WORKFLOW_AVAILABLE = False
SESSION_MANAGER_AVAILABLE = False
FIRESTORE_AVAILABLE = False
LLM_AVAILABLE = False

from src.api.training_api import training_router
from src.services.batch_processor import batch_processor
from src.services.file_storage_service import file_storage_service

# Import services first (they don't depend on agents)
try:
    from src.services.firestore_service import FirestoreService  
    FIRESTORE_AVAILABLE = True
    logging.info("✅ Firestore service import successful")
except ImportError as e:
    FIRESTORE_AVAILABLE = False
    logging.warning(f"⚠️ Firestore service not available: {e}")

try:
    from src.services.llm_service import safe_construction_call
    LLM_AVAILABLE = True
    logging.info("✅ LLM service import successful")
except ImportError as e:
    LLM_AVAILABLE = False
    logging.warning(f"⚠️ LLM service not available: {e}")
    
    async def safe_construction_call(*args, **kwargs):
        return "LLM service not available"

try:
    # Import the new get_session_manager utility
    from src.services.session_manager import initialize_session_manager, get_session_manager
    SESSION_MANAGER_AVAILABLE = True
    logging.info("✅ Session manager import successful")
except ImportError as e:
    SESSION_MANAGER_AVAILABLE = False
    logging.error(f"❌ Session manager import failed: {e}")
    
    async def initialize_session_manager():
        pass
    def get_session_manager():
        raise RuntimeError("Session manager is not available.")

# Import workflow last (it depends on the agent)
try:
    from src.workflow import run_complete_workflow, initialize_workflow, get_workflow_status
    WORKFLOW_AVAILABLE = True
    logging.info("✅ Unified workflow import successful")
except ImportError as e:
    logging.error(f"❌ Unified workflow import failed: {e}")
    WORKFLOW_AVAILABLE = False
    
    # Create dummy functions
    async def run_complete_workflow(*args, **kwargs):
        return {"error": "Workflow not available"}
    async def initialize_workflow():
        pass
    async def get_workflow_status():
        return {"status": "not_available"}

# Settings
class Settings:
    environment = os.getenv("ENVIRONMENT", "production")
    log_level = os.getenv("LOG_LEVEL", "INFO")
    gcp_project_id = os.getenv("GCP_PROJECT_ID", "demoplanfrvcxk")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    min_confidence_for_offer = float(os.getenv("MIN_CONFIDENCE_FOR_OFFER", "75.0"))
    
    unified_agent_enabled = True

settings = Settings()

# Logging setup
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

# Global variables
firestore_service: Optional[FirestoreService] = None
learning_engine = None
unified_agent = None
# Add with other global variables (after unified_agent)
drawing_agent: Optional[DrawingGenerationAgent] = None

async def initialize_unified_agent():
    """Initialize the global unified agent"""
    global unified_agent
    try:
        logger.info("🔄 Starting agent initialization...")
        
        from src.agents.unified_construction_agent import UnifiedConstructionAgent
        logger.info("✅ Agent class imported successfully")
        
        unified_agent = UnifiedConstructionAgent()
        logger.info("✅ Agent instance created")
        
        await unified_agent.initialize()
        logger.info("✅ Agent initialization completed")
    except Exception as e:
        logger.error(f"❌ Agent initialization failed at step: {e}", exc_info=True)
        unified_agent = None

async def initialize_drawing_agent():
    """Initialize the drawing generation agent"""
    global drawing_agent
    try:
        logger.info("🎨 Initializing Drawing Generation Agent...")
        
        # Ensure we have dependencies
        if not unified_agent or not firestore_service:
            logger.warning("⚠️ Dependencies not available for drawing agent")
            return
        
        drawing_agent = DrawingGenerationAgent(
            llm_service=unified_agent.llm_service,  # Reuse from unified agent
            firestore_service=firestore_service,
            max_retry_attempts=3
        )
        
        logger.info("✅ Drawing Generation Agent initialized")
        
    except Exception as e:
        logger.error(f"❌ Drawing agent initialization failed: {e}")
        drawing_agent = None

# =========================================================================
# STARTUP AND SHUTDOWN
# =========================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Phase 2 startup with agent initialization"""
    logger.info("🚀 Starting DEMOPLAN Phase 2...")
    global learning_engine, firestore_service
    
    try:
        # Initialize firestore
        if FIRESTORE_AVAILABLE:
            firestore_service = FirestoreService()
            await firestore_service.initialize()
            logger.info("✅ Firestore Service initialized")

        # Initialize session manager, passing the firestore service
        if SESSION_MANAGER_AVAILABLE:
            await initialize_session_manager(firestore_service)
            logger.info("✅ Session Manager initialized")

        # Initialize intelligence components
        try:
            from src.intelligence.learning_engine import LearningEngine
            learning_engine = LearningEngine()
            await batch_processor.initialize()
            logger.info("✅ Batch Processor initialized for training")
            
            await learning_engine.initialize(enable_batch_processing=True, batch_processor=batch_processor)
            logger.info("✅ Learning Engine initialized and linked with Batch Processor")

            batch_processor.learning_engine = learning_engine
        except Exception as e:
            logger.error(f"❌ Intelligence component initialization failed: {e}")

        # Initialize unified agent
        await initialize_unified_agent()

        # Initialize drawing agent
        await initialize_drawing_agent()

    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
    
    logger.info("🎯 DEMOPLAN Phase 2 ready!")
    yield
    
    logger.info("🔥 Shutting down DEMOPLAN...")

# Initialize FastAPI
app = FastAPI(
    title="DEMOPLAN Unified - Phase 1",
    description="Romanian construction consultation with unified agent architecture - Full capabilities",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(training_router)
logger.info("✅ Training API integrated successfully")

# =========================================================================
# EXCEPTION HANDLING
# =========================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception for request {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "A apărut o eroare internă neașteptată. Echipa tehnică a fost notificată.",
                "details": str(exc)
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
    )

# =========================================================================
# API ENDPOINTS
# =========================================================================

@app.get("/health")
async def health_check():
    """Health check with component status"""
    workflow_status = await get_workflow_status() if WORKFLOW_AVAILABLE else {"status": "not_available"}
    
    return {
        "status": "healthy",
        "version": "1.0.0-mvp-phase1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": "1-mvp-drawing-generation",
        "components": {
            "unified_workflow": "available" if WORKFLOW_AVAILABLE else "disabled",
            "firestore": "available" if firestore_service else "disabled",
            "file_storage": "available" if file_storage_service.storage_client else "memory_only",
            "llm_service": "available" if LLM_AVAILABLE else "disabled",
            "session_manager": "available" if SESSION_MANAGER_AVAILABLE else "disabled",
            "drawing_agent": "available" if drawing_agent else "disabled"  # NEW
        },
        "workflow_status": workflow_status,
        "drawing_generation": {  # NEW SECTION
            "enabled": drawing_agent is not None,
            "supported_types": ["field_verification"],
            "max_retry_attempts": 3,
            "storage_type": drawing_agent.storage_service.get_storage_info()['storage_type'] if drawing_agent else "unknown"
        }
    }

@app.post("/start-session")
async def start_session():
    """Start new consultation session using the SessionManager."""
    try:
        manager = get_session_manager()
        session = await manager.create_session()
        return {
            "session_id": session.session_id,
            "message": "Sesiune nouă creată cu succes!",
            "mode": "Unified Agent Phase 1 - Full Capabilities"
        }
    except Exception as e:
        logger.error(f"❌ Session creation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Eroare la crearea sesiunii: {str(e)}")

@app.post("/session/{session_id}/upload")
async def upload_files(
    session_id: str,
    files: List[UploadFile] = File(...),
    description: str = Form(default="")
):
    """
    Upload files, run unified analysis, and update session metadata.
    File content and analysis results are handled by the agent and stored in subcollections.
    """
    if not unified_agent:
        raise HTTPException(status_code=503, detail="Agentul nu este disponibil momentan.")

    try:
        manager = get_session_manager()
        session = await manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Sesiune inexistentă")

        files_data: List[Dict[str, Any]] = []
        storage_type = "memory"
        storage_breakdown = {
            "firestore": [],
            "gcs": [],
            "memory": []
        }
        
        for file in files:
            try:
                content = await file.read()
                storage_info = await file_storage_service.store_session_file(
                    session_id, file.filename, content
                )
                storage_type = storage_info.get("storage_type", "memory")
                
                files_data.append({
                    "filename": file.filename,
                    "content_type": file.content_type,
                    "size": len(content),
                    "content": content,
                    "storage_info": storage_info,
                    "uploaded_at": datetime.now(timezone.utc).isoformat(),
                    "description": description
                })
                
                # Track storage breakdown (NEW)
                storage_breakdown[storage_type].append({
                    "filename": file.filename,
                    "size_kb": len(content) / 1024
                })
                
            except Exception as file_error:
                logger.error(f"❌ Failed to process file {file.filename}: {file_error}", exc_info=True)
                continue

        logger.info(f"📁 {len(files_data)} files processed for agent analysis in session {session_id}")

        # The agent now handles persisting file analysis to a subcollection internally.
        analysis_result = await unified_agent.analyze_project(
            files=files_data,
            user_input=description,
            session_id=session_id
        )

        # Update session metadata based on the agent's analysis.
        # The API layer no longer saves the full analysis_result or files list.
        session.confidence_score = analysis_result.get("confidence", 0)
        session.can_generate_offer = analysis_result.get("can_generate_offer", False)
        session.last_activity = datetime.now(timezone.utc)
        await manager._persist_session(session)
        # Check if any files were split to GCS (NEW)
        geometric_split_count = 0
        if firestore_service:
            file_analyses = await firestore_service.load_all_file_analyses(session_id)
            for analysis in file_analyses:
                # analysis may contain geometric_storage at top-level or inside dxf_analysis
                if analysis.get("geometric_storage") == "gcs" or (analysis.get('analysis_data', {}) or {}).get('analysis_summary', {}).get('geometric_storage') == 'gcs':
                    geometric_split_count += 1
                    filename = analysis.get("filename", "unknown")
                    # Move from firestore to gcs tracking if previously marked as firestore
                    for item in list(storage_breakdown["firestore"]):
                        if item["filename"] == filename:
                            storage_breakdown["gcs"].append(item)
                            storage_breakdown["firestore"].remove(item)
                            break

        return {
            "session_id": session_id,
            "files_uploaded": len(files_data),
            "storage_type": storage_type,
            "storage_breakdown": storage_breakdown,  # NEW
            "geometric_split_count": geometric_split_count,  # NEW
            "analysis": {
                "confidence": session.confidence_score,
                "can_generate_offer": session.can_generate_offer,
            },
            "ai_response": analysis_result.get("response", "Fișiere analizate cu succes."),
            "message": f"✅ {len(files_data)} fișiere analizate cu succes (Stocare: {storage_type}, GCS split: {geometric_split_count})"  # UPDATED
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ File upload error for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Eroare la încărcarea fișierelor: {str(e)}")

@app.post("/session/{session_id}/chat")
async def chat(
    session_id: str,
    message: str = Form(...)
):
    """
    Continue conversation. The agent now handles persisting messages to a subcollection.
    """
    if not unified_agent:
        raise HTTPException(status_code=503, detail="Agentul nu este disponibil momentan.")

    try:
        manager = get_session_manager()
        session = await manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Sesiune inexistentă")
        
        # The API no longer appends messages to the session object.
        # The agent saves both user and assistant messages to the 'messages' subcollection.
        agent_response = await unified_agent.continue_conversation(
            session_id=session_id, 
            user_input=message
        )
        
        # Update session metadata from the agent's response
        session.confidence_score = agent_response.get("confidence", session.confidence_score)
        session.can_generate_offer = agent_response.get("can_generate_offer", session.can_generate_offer)
        session.last_activity = datetime.now(timezone.utc)
        await manager._persist_session(session)
        
        return {
            "session_id": session_id,
            "response": agent_response.get("response", "Eroare în procesare"),
            "confidence": session.confidence_score,
            "can_generate_offer": session.can_generate_offer
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Chat error for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Eroare în conversație: {str(e)}")

@app.get("/session/{session_id}")
async def get_session_status(session_id: str):
    """
    Get session status. File and message counts are now retrieved from subcollections.
    """
    try:
        manager = get_session_manager()
        session = await manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Sesiune inexistentă")
        
        # Query subcollections for counts, assuming firestore_service has these new methods.
        message_count = 0
        files_analyzed = []
        if firestore_service:
            message_count = await firestore_service.get_message_count(session_id)
            files_analyzed = await firestore_service.load_all_file_analyses(session_id)

        return {
            "session_id": session.session_id,
            "status": session.status,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "files_count": len(files_analyzed),
            "conversation_length": message_count,
            "confidence": session.confidence_score,
            "can_generate_offer": session.can_generate_offer
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Session status error for {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Eroare la obținerea statusului sesiunii: {str(e)}")

@app.get("/session/{session_id}/files")
async def get_session_files(session_id: str):
    """
    List all analyzed files for a session by querying the file_analyses subcollection.
    """
    if not firestore_service:
        raise HTTPException(status_code=503, detail="Serviciul de baze de date nu este disponibil.")
        
    try:
        # Assumes firestore_service.load_all_file_analyses is implemented
        file_analyses = await firestore_service.load_all_file_analyses(session_id)
        if file_analyses is None:
             raise HTTPException(status_code=404, detail="Sesiune inexistentă sau fără fișiere analizate.")
        
        # Return a summary of the files
        return {
            "session_id": session_id,
            "files": [
                {
                    "filename": analysis.get("filename"),
                    "file_type": analysis.get("file_type"),
                    "status": analysis.get("processing_status"),
                    "uploaded_at": analysis.get("uploaded_at")
                } for analysis in file_analyses
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error fetching files for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Eroare la preluarea listei de fișiere.")


@app.get("/session/{session_id}/geometric/{file_id}")
async def get_geometric_data(session_id: str, file_id: str):
    """
    Optional debugging endpoint: Direct access to geometric data from GCS
    
    Week 4 Enhancement: Allows inspection of geometric data for debugging
    """
    if not firestore_service:
        raise HTTPException(status_code=503, detail="Serviciul de baze de date nu este disponibil.")
    
    try:
        # Load geometric data only
        geometric_data = await firestore_service.load_geometric_data_only(session_id, file_id)
        
        if not geometric_data:
            raise HTTPException(status_code=404, detail="Date geometrice nu au fost găsite.")
        
        # Return summary (not full data - too large)
        room_breakdown = geometric_data.get("room_breakdown", [])
        entity_inventory = geometric_data.get("entity_inventory", {})
        
        return {
            "session_id": session_id,
            "file_id": file_id,
            "storage": "gcs",
            "summary": {
                "total_rooms": len(room_breakdown),
                "total_area": geometric_data.get("total_area", 0),
                "has_dimensions": geometric_data.get("has_dimensions", False),
                "entities": {
                    "doors_windows": len(entity_inventory.get("doors_windows", [])),
                    "electrical": len(entity_inventory.get("electrical_components", [])),
                    "hvac": len(entity_inventory.get("hvac_components", []))
                }
            },
            "sample_room": room_breakdown[0] if room_breakdown else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error fetching geometric data for {file_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Eroare la preluarea datelor geometrice.")

@app.post("/session/{session_id}/generate-drawing")
async def generate_drawing(
    session_id: str,
    specifications: Optional[str] = Form(None)
):
    """
    Generate field verification drawing from session data
    
    Phase 1 MVP: Only supports field_verification type
    
    Args:
        session_id: Session with analyzed DXF data
        specifications: Optional custom specifications
    
    Returns:
        Drawing generation result with download URL
    """
    
    if not drawing_agent:
        raise HTTPException(
            status_code=503,
            detail="Serviciul de generare desene nu este disponibil momentan."
        )
    
    try:
        # Verify session exists
        manager = get_session_manager()
        session = await manager.get_session(session_id)
        
        if not session:
            raise HTTPException(status_code=404, detail="Sesiune inexistentă")
        
        logger.info(f"🎨 Drawing generation requested for session {session_id}")
        
        # Generate drawing
        result = await drawing_agent.generate_field_verification_drawing(
            session_id=session_id,
            custom_specifications=specifications
        )
        
        if result.success:
            return {
                "session_id": session_id,
                "success": True,
                "drawing_id": result.drawing_id,
                "drawing_url": result.drawing_url,
                "drawing_type": "field_verification",
                "message": "Desen de verificare în teren generat cu succes!",
                "processing_time_ms": result.processing_time_ms,
                "attempts": result.attempts,
                "validation_warnings": result.validation_warnings,
                "instructions": "Descărcați fișierul DXF și deschideți-l în AutoCAD sau orice vizualizator DXF."
            }
        else:
            return {
                "session_id": session_id,
                "success": False,
                "error": result.error_message,
                "message": "Eroare la generarea desenului. Verificați dacă ați încărcat fișiere DXF pentru analiză.",
                "processing_time_ms": result.processing_time_ms,
                "attempts": result.attempts
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Drawing generation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Eroare la generarea desenului: {str(e)}"
        )

@app.get("/session/{session_id}/drawings")
async def list_session_drawings(session_id: str):
    """
    Get all generated drawings for a session
    
    Args:
        session_id: Session identifier
    
    Returns:
        List of drawings with metadata and URLs
    """
    
    if not firestore_service:
        raise HTTPException(status_code=503, detail="Firestore indisponibil")
    
    try:
        # Verify session exists
        manager = get_session_manager()
        session = await manager.get_session(session_id)
        
        if not session:
            raise HTTPException(status_code=404, detail="Sesiune inexistentă")
        
        # Get all drawings
        drawings = await firestore_service.get_session_drawings(session_id)
        
        return {
            "session_id": session_id,
            "drawings_count": len(drawings),
            "drawings": drawings
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error listing drawings: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Eroare la listarea desenelor: {str(e)}"
        )

# =========================================================================
# NEW ENDPOINTS FOR REACT FRONTEND
# =========================================================================

@app.get("/sessions")
async def get_all_sessions():
    """
    Get a summary list of all sessions for the history sidebar.
    Note: This assumes a method `get_all_sessions_summary` exists in the SessionManager.
    """
    try:
        manager = get_session_manager()
        sessions_summary = await manager.get_all_sessions_summary()
        
        response_data = []
        for session in sessions_summary:
            created_at = session.get("created_at")
            if isinstance(created_at, datetime):
                created_at_str = created_at.isoformat()
                default_title = f"Consultation from {created_at.strftime('%Y-%m-%d')}"
            else:
                created_at_str = str(created_at)
                default_title = "Untitled Session"

            response_data.append({
                "session_id": session.get("session_id"),
                "created_at": created_at_str,
                "title": session.get("title", default_title)
            })
        
        return sorted(response_data, key=lambda x: x['created_at'], reverse=True)
        
    except Exception as e:
        logger.error(f"❌ Error fetching all sessions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Eroare la preluarea listei de sesiuni.")

@app.get("/session/{session_id}/messages")
async def get_session_messages(session_id: str):
    """
    Get all messages for a specific session to load into the chat window.
    Note: This assumes a method `load_all_messages` exists in the FirestoreService.
    """
    if not firestore_service:
        raise HTTPException(status_code=503, detail="Serviciul de baze de date nu este disponibil.")
        
    try:
        messages = await firestore_service.load_messages_from_subcollection(session_id)
        
        if messages is None:
            manager = get_session_manager()
            session = await manager.get_session(session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Sesiune inexistentă.")
            return []

        formatted_messages = []
        for msg in messages:
            timestamp = msg.get("timestamp")
            if isinstance(timestamp, datetime):
                timestamp = timestamp.isoformat()
            
            formatted_messages.append({
                "type": msg.get("type"),
                "content": msg.get("content"),
                "timestamp": timestamp
            })

        return formatted_messages
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error fetching messages for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Eroare la preluarea mesajelor sesiunii.")

# =========================================================================
# SERVER STARTUP
# =========================================================================

if __name__ == "__main__":
    try:
        port = int(os.getenv("PORT", 8080))
        logger.info(f"🚀 Starting DEMOPLAN Unified Phase 1 with Full Capabilities on port {port}")
        
        uvicorn.run(
            app, 
            host="0.0.0.0",
            port=port,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"❌ Server startup failed: {e}")
        sys.exit(1)