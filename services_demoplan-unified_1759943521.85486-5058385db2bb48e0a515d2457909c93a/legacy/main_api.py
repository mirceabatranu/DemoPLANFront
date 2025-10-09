"""
DEMOPLAN - Phase 2 Enhanced Main API with Automatic DXF Content Analysis
Integrates floorplan agent DXF analysis results into consultation responses
Generates automatic technical descriptions from architectural drawings
"""

import os
import sys
import logging
import asyncio
import uuid
from pathlib import Path
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Dict, List, Any, Optional, Tuple

# The Dockerfile's PYTHONPATH handles this, so manual path manipulation is removed
# for better container-based deployment.

# Import with fallback handling
# Try to import LLMService, but don't fail if it's not available
try:
    from src.services.llm_service import SafeConstructionLLMService, safe_construction_llm_service
    LLM_AVAILABLE = True
except ImportError as e:
    safe_construction_llm_service = None
    LLM_AVAILABLE = False
    logging.warning(f"LLMService not available, using fallback. Reason: {e}")
    
    # Create dummy SafeConstructionLLMService class
    class SafeConstructionLLMService:
        def __init__(self):
            pass
        async def safe_construction_call(self, *args, **kwargs):
            return "LLM service not available"
    safe_construction_llm_service = SafeConstructionLLMService()
    
def _create_dummy_workflow_function(e):
    async def run_complete_workflow(session_id, files_info, user_requirements=None, existing_session_data=None):
        return {
            "session_id": session_id,
            "overall_confidence": 0.0,
            "agent_confidence": {"floorplan": 0, "validation": 0, "estimation": 0, "offer_composer": 0},
            "agent_results": {},
            "can_generate_offer": False,
            "next_questions": ["Workflow module not available"],
            "file_analysis": {"files_processed": 0, "error": "Module not found"},
            "dxf_enhanced": False,
            "file_description": None,
            "processing_time_seconds": 0,
            "workflow_status": "failed",
            "error": str(e)
        }
    return run_complete_workflow

try:
    from src.workflow import run_complete_workflow
    logging.info("✅ Workflow import successful")
except ImportError as e:
    logging.error(f"❌ Workflow import failed: {e}")
    run_complete_workflow = _create_dummy_workflow_function(e)

try:
    from src.services.session_manager import get_session_manager, initialize_session_manager
    logging.info("✅ Session manager import successful")
except ImportError as e:
    logging.error(f"❌ Session manager import failed: {e}")
    # Fallback - create dummy functions
    def get_session_manager():
        return None
    async def initialize_session_manager(*args):
        return None

try:
    from src.api.chat_endpoints import chat_router
    CHAT_ROUTER_AVAILABLE = True
    logging.info("✅ Chat router import successful")
except ImportError as e:
    CHAT_ROUTER_AVAILABLE = False
    logging.error(f"❌ Chat router import failed: {e}")

# Core FastAPI imports
from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Enhanced Phase 2 Imports - with graceful fallback
try:
    from src.agents.agent_orchestrator import AgentOrchestrator
    ENHANCED_AGENTS_AVAILABLE = True
    logging.info("✅ Phase 2 Enhanced Agents loaded successfully")
except ImportError as e:
    ENHANCED_AGENTS_AVAILABLE = False
    logging.warning(f"⚠️ Enhanced Agents not available, using basic mode: {e}")

# ML Integration imports - with error handling
try:
    from src.intelligence.learning_engine import LearningEngine
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("⚠️ Learning Engine not available, using basic mode")

try:
    from src.services.firestore_service import FirestoreService  
    FIRESTORE_AVAILABLE = True
except ImportError:
    FIRESTORE_AVAILABLE = False
    logging.warning("⚠️ Firestore Service not available, using memory mode")

# Settings - Enhanced with Phase 2 configuration
class Settings:
    environment = os.getenv("ENVIRONMENT", "production")
    log_level = os.getenv("LOG_LEVEL", "INFO")
    gcp_project_id = os.getenv("GCP_PROJECT_ID", "demoplanfrvcxk")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    # ML Configuration
    ml_enabled = os.getenv("ML_ENABLED", "true").lower() == "true" and ML_AVAILABLE
    ml_confidence_adjustment = float(os.getenv("ML_CONFIDENCE_ADJUSTMENT", "0.5"))
    confidence_min_cap = float(os.getenv("CONFIDENCE_MIN_CAP", "0.2"))
    confidence_max_cap = float(os.getenv("CONFIDENCE_MAX_CAP", "0.95"))
    
    # Phase 2 Agent Configuration
    agent_mode_enabled = os.getenv("AGENT_MODE_ENABLED", "true").lower() == "true" and ENHANCED_AGENTS_AVAILABLE
    min_confidence_for_offer = float(os.getenv("MIN_CONFIDENCE_FOR_OFFER", "75.0"))
    
    # Agent Weights - Phase 2
    floorplan_agent_weight = float(os.getenv("FLOORPLAN_AGENT_WEIGHT", "0.25"))
    validation_agent_weight = float(os.getenv("VALIDATION_AGENT_WEIGHT", "0.15"))
    estimation_agent_weight = float(os.getenv("ESTIMATION_AGENT_WEIGHT", "0.40"))
    offer_composer_weight = float(os.getenv("OFFER_COMPOSER_WEIGHT", "0.20"))

settings = Settings()

# Logging setup
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("demoplan.main_api")

# Global variables
sessions_memory = {}
learning_engine = None
firestore_service = None
agent_orchestrator = None

# =============================================================================
# ENHANCED STARTUP WITH PHASE 2 AGENTS
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced startup with Phase 2 agent initialization"""
    global learning_engine, firestore_service, agent_orchestrator
    
    logger.info("🚀 Starting DEMOPLAN Phase 2 Enhanced Server with DXF Auto-Analysis...")
    
    # Initialize ML components
    if settings.ml_enabled and ML_AVAILABLE:
        try:
            learning_engine = LearningEngine()
            logger.info("✅ Learning Engine initialized")
        except Exception as e:
            logger.error(f"❌ Learning Engine failed: {e}")
    
    # Initialize Firestore
    if FIRESTORE_AVAILABLE:
        try:
            firestore_service = FirestoreService()
            logger.info("✅ Firestore Service initialized")
        except Exception as e:
            logger.error(f"❌ Firestore failed: {e}")
    
    # Initialize Phase 2 Agent Orchestrator
    if settings.agent_mode_enabled and ENHANCED_AGENTS_AVAILABLE:
        try:
            agent_orchestrator = AgentOrchestrator()
            logger.info("✅ Phase 2 Agent Orchestrator with DXF Analysis initialized")
            logger.info(f"   - Floorplan Agent Weight: {settings.floorplan_agent_weight}")
            logger.info(f"   - Validation Agent Weight: {settings.validation_agent_weight}")
            logger.info(f"   - Estimation Agent Weight: {settings.estimation_agent_weight}")
            logger.info(f"   - Offer Composer Weight: {settings.offer_composer_weight}")
        except Exception as e:
            logger.error(f"❌ Agent Orchestrator failed: {e}")
            settings.agent_mode_enabled = False
    
    # Initialize Session Manager
    try:
        await initialize_session_manager()
        logger.info("✅ Session Manager initialized")
    except Exception as e:
        logger.error(f"❌ Session Manager initialization failed: {e}")

    logger.info("🎯 DEMOPLAN Phase 2 Server ready with automatic DXF content analysis!")
    logger.info(f"   ML Mode: {'Enhanced' if settings.ml_enabled else 'Basic'}")
    logger.info(f"   Agent Mode: {'Phase 2 + DXF Analysis' if settings.agent_mode_enabled else 'Basic'}")
    logger.info(f"   Storage: {'Firestore' if firestore_service else 'Memory'}")
    
    yield
    
    logger.info("🔥 Shutting down DEMOPLAN...")

# Initialize FastAPI with enhanced configuration
app = FastAPI(
    title="DEMOPLAN Phase 2 - Enhanced Construction Consultation with DXF Analysis",
    description="Chat-based construction consultation with automatic DXF content extraction and Romanian expertise",
    version="2.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Integrate the chat router
if CHAT_ROUTER_AVAILABLE:
    app.include_router(chat_router)
    logging.info("✅ Chat router integrated successfully.")

# Initialize FastAPI with enhanced configuration
app = FastAPI(
    title="DEMOPLAN Phase 2 - Enhanced Construction Consultation with DXF Analysis",
    description="Chat-based construction consultation with automatic DXF content extraction and Romanian expertise",
    version="2.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Integrate the chat router
if CHAT_ROUTER_AVAILABLE:
    app.include_router(chat_router)
    logging.info("✅ Chat router integrated successfully.")

# =============================================================================
# ENHANCED SESSION MANAGEMENT
# =============================================================================

async def create_enhanced_session() -> Tuple[str, Dict]:
    """Create enhanced session with Phase 2 agent support"""
    session_id = str(uuid.uuid4())
    
    session_data = {
        "session_id": session_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "last_activity": datetime.now(timezone.utc).isoformat(),
        "status": "active",
        "files": [],
        "conversation": [],
        "analysis_result": {},
        
        # Phase 2 Enhanced Fields
        "agent_mode": settings.agent_mode_enabled,
        "agent_requirements": {},
        "agent_confidence": {
            "floorplan": 0.0,
            "validation": 0.0, 
            "estimation": 0.0,
            "offer_composer": 0.0,
            "overall": 0.0
        },
        "consultation_phase": "initial",  # initial, gathering, analysis, offer_ready
        "offer_data": None,
        "technical_questions": [],
        "missing_data_categories": [],
        
        # Enhanced DXF Analysis Fields
        "dxf_analysis_available": False,
        "auto_generated_description": None,
        "technical_content_extracted": False
    }
    
    # Save session
    await save_session_safely(session_id, session_data)
    return session_id, {k: v for k, v in session_data.items() if k != 'files'}

async def load_session_safely(session_id: str) -> Optional[Dict]:
    """Enhanced session loading with Phase 2 compatibility"""
    try:
        # Try Firestore first
        if firestore_service:
            try:
                doc = await firestore_service.get_document("engineer_chat_sessions", session_id)
                if doc:
                    # Ensure Phase 2 fields exist for backward compatibility
                    if "agent_confidence" not in doc:
                        doc["agent_confidence"] = {
                            "floorplan": 0.0, "validation": 0.0, 
                            "estimation": 0.0, "offer_composer": 0.0, "overall": 0.0
                        }
                    if "consultation_phase" not in doc:
                        doc["consultation_phase"] = "initial"
                    if "agent_requirements" not in doc:
                        doc["agent_requirements"] = {}
                    
                    # Add DXF analysis fields if missing
                    if "dxf_analysis_available" not in doc:
                        doc["dxf_analysis_available"] = False
                    if "auto_generated_description" not in doc:
                        doc["auto_generated_description"] = None
                    if "technical_content_extracted" not in doc:
                        doc["technical_content_extracted"] = False
                    
                    logger.info(f"Session {session_id} loaded from Firestore with confidence: {doc.get('analysis_result', {}).get('confidence', 0)}")
                    return doc
            except Exception as e:
                logger.error(f"⚠️ Firestore load error: {e}")
        
        # Fallback to memory
        session_data = sessions_memory.get(session_id)
        if session_data:
            # Add Phase 2 fields for backward compatibility if they don't exist
            session_data.setdefault("agent_confidence", {"floorplan": 0.0, "validation": 0.0, "estimation": 0.0, "offer_composer": 0.0, "overall": 0.0})
            session_data.setdefault("consultation_phase", "initial")
            session_data.setdefault("agent_requirements", {})
            session_data.setdefault("dxf_analysis_available", False)
            session_data.setdefault("auto_generated_description", None)
            session_data.setdefault("technical_content_extracted", False)
            logger.info(f"Session {session_id} loaded from memory")
            return session_data
            
        logger.warning(f"Session {session_id} not found in Firestore or memory")
        return None
    except Exception as e:
        logger.error(f"❌ Session load error: {e}")
        return None

async def save_session_safely(session_id: str, session_data: Dict):
    """Enhanced session saving with Phase 2 support"""
    try:
        # Always save to memory for immediate access
        sessions_memory[session_id] = session_data

        # Save to Firestore if available
        if firestore_service:
            try:
                # Create a copy to avoid modifying the in-memory session
                import copy
                data_to_save = copy.deepcopy(session_data)
                # Remove raw file content before saving
                for file_info in data_to_save.get("files", []):
                    file_info.pop("content", None)
                
                # The firestore service now uses a different collection name mapping
                await firestore_service.save_document(
                    "chat_sessions", session_id, data_to_save
                )
            except Exception as e:
                logger.error(f"⚠️ Firestore save error: {e}")
    except Exception as e:
        logger.error(f"❌ Session save error: {e}")

# =============================================================================
# HELPER FUNCTIONS FOR DXF ANALYSIS AND RESPONSE GENERATION
# =============================================================================

def get_romanian_room_name(room_type: str) -> str:
    """Convert English room type to Romanian name"""
    romanian_names = {
        'bedroom': 'dormitor',
        'kitchen': 'bucătărie',
        'bathroom': 'baie',
        'living_room': 'living',
        'hallway': 'hol',
        'storage': 'debara',
        'balcony': 'balcon',
        'entrance': 'intrare',
        'office': 'birou',
        'dining': 'sufragerie'
    }
    return romanian_names.get(room_type, room_type)

async def generate_natural_response_from_workflow(
    workflow_result: Dict[str, Any], 
    session_data: Dict[str, Any],
    description: str = ""
) -> str:
    """Generate natural response using actual workflow results"""
    
    confidence = workflow_result.get("overall_confidence", 0) * 100
    agent_confidence = workflow_result.get("agent_confidence", {})
    next_questions = workflow_result.get("next_questions", [])
    file_description = workflow_result.get("file_description", "N/A")
    
    file_info = f"**Analiză fișiere:** {file_description}"
    if description:
        file_info += f"\n**Cerința dumneavoastră:** \"{description}\""
    
    # Generate response based on confidence level
    if confidence >= 70:
        return f"""🎯 **ANALIZĂ COMPLETĂ - Pregătit pentru ofertă**

{file_info}

📊 **Încredere analiză: {confidence:.1f}%**
✅ Toate datele necesare sunt disponibile pentru generarea unei oferte profesionale.

🏗️ **Rezultate analiză:**
- Plan arhitectural: {agent_confidence.get('floorplan', 0):.1f}% încredere
- Validare tehnică: {agent_confidence.get('validation', 0):.1f}% încredere  
- Estimare costuri: {agent_confidence.get('estimation', 0):.1f}% încredere
- Compunere ofertă: {agent_confidence.get('offer_composer', 0):.1f}% încredere

💼 **Următorul pas:** 
Scrieți "generează oferta" pentru a primi oferta tehnică și comercială completă."""

    elif confidence >= 40:
        questions_text = ""
        if next_questions:
            questions_text = "\n❓ **Următorii pași pentru analiză completă:**\n"
            for q in next_questions[:3]:
                questions_text += f"- {q}\n"
        
        return f"""🚀 **CONSULTAȚIE TEHNICĂ ÎN DEZVOLTARE**

{file_info}

📊 **Progres analiză: {confidence:.1f}%**

🔧 **Status curent:**
- Date arhitecturale: {agent_confidence.get('floorplan', 0):.1f}%
- Validare tehnică: {agent_confidence.get('validation', 0):.1f}%
- Estimare costuri: {agent_confidence.get('estimation', 0):.1f}%

{questions_text}

Vă ghidez pas cu pas către o ofertă tehnică profesională!"""

    else:
        questions_text = ""
        if next_questions:
            questions_text = "\n❓ **Pentru analiza completă, vă rog să-mi spuneți:**\n"
            for q in next_questions[:3]:
                questions_text += f"- {q}\n"
        else:
            questions_text = "\n❓ **Pentru analiza completă, vă rog să-mi spuneți mai multe despre proiectul dumneavoastră.**"

        return f"""🚀 **ÎNCEPUT CONSULTAȚIE TEHNICĂ**

{file_info}

📊 **Încredere inițială: {confidence:.1f}%**

{questions_text}

Vă ghidez prin proces pentru a ajunge la o ofertă precisă și profesională."""

def detect_detailed_description_request(message: str) -> bool:
    """Detect if user is requesting detailed technical description"""
    message_lower = message.lower()
    
    detailed_description_patterns = [
        "descriere detaliata", "descriere detaliată", 
        "descriere completa", "descriere completă",
        "mai multe detalii", "detalii complete",
        "analiza completa", "analiza completă",
        "analiza detaliata", "analiza detaliată",
        "descriere cat mai completa", "descriere cât mai completă",
        "informatii detaliate", "informații detaliate",
        "specificatii tehnice", "specificații tehnice",
        "toate detaliile", "tot ce ati gasit", "tot ce ați găsit"
    ]
    
    return any(pattern in message_lower for pattern in detailed_description_patterns)

async def get_basic_cost_estimate_from_analysis(session_id: str) -> Optional[Dict[str, Any]]:
    """Get basic cost estimation data from DXF analysis stored in the session."""
    try:
        session_data = await load_session_safely(session_id)
        if not session_data:
            return None

        # The workflow result is now stored in the session after each run
        workflow_result = session_data.get("analysis_result", {})
        agent_results = workflow_result.get("agent_results", {})
        floorplan_data = agent_results.get('floorplan', {})

        if floorplan_data:
            spatial = floorplan_data.get("spatial_analysis", {})
            return {
                "area": spatial.get("estimated_total_area", 0),
                "rooms": spatial.get("total_rooms", 0),
                "room_types": spatial.get("room_types", {})
            }
        return None
    except Exception as e:
        logger.error(f"Error getting cost estimate data: {e}")
        return None

def assess_project_complexity(session_data: Dict[str, Any]) -> str:
    """Assess project complexity based on available data"""
    files_count = len(session_data.get("files", []))
    analysis_result = session_data.get("analysis_result", {})
    
    # Get room count from analysis if available
    agent_results = analysis_result.get("agent_results", {})
    floorplan_data = agent_results.get('floorplan', {})
    room_count = floorplan_data.get("spatial_analysis", {}).get("total_rooms", 0)
    
    if files_count >= 5 or room_count >= 6:
        return "complex"
    elif files_count >= 3 or room_count >= 4:
        return "mediu"
    else:
        return "simplu"

def get_execution_timeframe(complexity: str) -> str:
    """Get execution timeframe based on complexity"""
    timeframes = {
        "simplu": "2-3 săptămâni",
        "mediu": "4-6 săptămâni", 
        "complex": "6-10 săptămâni"
    }
    return timeframes.get(complexity, "4-6 săptămâni")

def get_total_timeframe(complexity: str) -> str:
    """Get total project timeframe"""
    timeframes = {
        "simplu": "1.5-2 luni",
        "mediu": "2-3 luni",
        "complex": "3-4 luni"
    }
    return timeframes.get(complexity, "2-3 luni")

def generate_detailed_dxf_description(floorplan_data: Dict) -> str:
    """Generate detailed Romanian description from DXF analysis data"""
    try:
        spatial = floorplan_data.get("spatial_analysis", {})
        rooms = floorplan_data.get("rooms", {})
        dimensions = floorplan_data.get("dimensions", [])
        
        parts = []
        parts.append("🔍 **ANALIZĂ DETALIATĂ PLAN ARHITECTURAL**\n")
        
        # Room analysis
        total_rooms = spatial.get("total_rooms", 0)
        total_area = spatial.get("estimated_total_area", 0)
        
        if total_rooms > 0:
            parts.append(f"📐 **Spații identificate:** {total_rooms} camere")
        
        if total_area > 0:
            parts.append(f"📏 **Suprafață totală:** {total_area:.1f} mp")
        
        # Room types breakdown
        room_types = spatial.get("room_types", {})
        if room_types:
            parts.append("\n🏠 **Tipuri de spații:**")
            for room_type, count in room_types.items():
                if room_type != "unknown":
                    romanian_name = get_romanian_room_name(room_type)
                    if count > 1:
                        parts.append(f"  • {count}x {romanian_name}")
                    else:
                        parts.append(f"  • {romanian_name}")
        
        # Technical details
        wall_length = spatial.get("wall_total_length", 0)
        if wall_length > 0:
            parts.append(f"\n🧱 **Lungime pereți:** {wall_length:.1f} m")
        
        parts.append(f"\n🔧 **Procesat cu sistem optimizat DXF v2**")
        
        return "\n".join(parts)
        
    except Exception as e:
        logger.error(f"Error generating detailed description: {e}")
        return "Eroare la procesarea datelor DXF."

async def get_detailed_description_from_agent(session_id: str) -> str:
    """Get detailed description from stored DXF analysis"""
    try:
        session_data = await load_session_safely(session_id)
        if not session_data:
            return "Sesiune invalidă."
        
        # Get the workflow result from session
        analysis_result = session_data.get("analysis_result", {})
        
        # Try to get detailed DXF analysis if available
        if analysis_result.get("dxf_enhanced"):
            # Access agent results from workflow
            agent_results = analysis_result.get("agent_results", {})
            floorplan_data = agent_results.get('floorplan', {})
            
            if floorplan_data:
                # Generate detailed description from stored analysis
                return generate_detailed_dxf_description(floorplan_data)
        
        # Fallback to basic description
        return analysis_result.get("auto_generated_content", "Analiză detaliată nu este disponibilă momentan.")
        
    except Exception as e:
        logger.error(f"Error getting detailed description: {e}")
        return "Eroare la generarea descrierii detaliate."

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/health")
async def health_check():
    """Enhanced health check with Phase 2 status"""
    return {
        "status": "healthy",
        "version": "2.1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": {
            "ml_engine": "available" if settings.ml_enabled else "disabled",
            "agent_orchestrator": "available" if settings.agent_mode_enabled else "disabled", 
            "firestore": "available" if firestore_service else "memory_only",
            "enhanced_agents": "available" if ENHANCED_AGENTS_AVAILABLE else "disabled",
            "dxf_analysis": "automatic" if settings.agent_mode_enabled else "disabled"
        }
    }

@app.post("/start-session")
async def start_enhanced_session():
    """Enhanced session creation with Phase 2 support"""
    try:
        session_id, session_data = await create_enhanced_session()
        return {
            "session_id": session_id,
            "message": "Sesiune nouă creată cu succes!",
            "mode": "Phase 2 Enhanced + DXF Analysis" if settings.agent_mode_enabled else "Basic"
        }
    except Exception as e:
        logger.error(f"❌ Session creation error: {e}")
        raise HTTPException(status_code=500, detail=f"Eroare la crearea sesiunii: {str(e)}")

@app.post("/session/{session_id}/upload")
async def upload_files_enhanced(
    session_id: str,
    files: List[UploadFile] = File(...),
    description: str = Form(default="")
):
    """Enhanced file upload with actual agent processing"""
    try:
        session_data = await load_session_safely(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Sesiune inexistentă")
        
        files_data: List[Dict[str, Any]] = []
        for file in files:
            content = await file.read()
            file_info = {
                "filename": file.filename,
                "content_type": file.content_type,
                "size": len(content),
                "content": content,
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
                "description": description
            }
            files_data.append(file_info)

        session_data.get("files", []).extend(files_data)
        logger.info(f"📁 {len(files)} files uploaded to session {session_id}")
        
        workflow_result = await run_complete_workflow(
            session_id=session_id,
            files_info=files_data,
            user_requirements=description,
            existing_session_data=session_data
        )
        
        analysis_result = {
            "confidence": workflow_result.get("overall_confidence", 0),
            "consultation_phase": "gathering",
            "agent_breakdown": workflow_result.get("agent_confidence", {}),
            "can_generate_offer": workflow_result.get("can_generate_offer", False),
            "technical_questions": workflow_result.get("next_questions", []),
            "file_analysis": workflow_result.get("file_analysis", {}),
            "dxf_enhanced": workflow_result.get("dxf_enhanced", False),
            "auto_generated_content": workflow_result.get("file_description")
        }
        
        ai_response = await generate_natural_response_from_workflow(
            workflow_result, session_data, description
        )
        
        session_data["analysis_result"] = analysis_result
        session_data["agent_confidence"] = workflow_result.get("agent_confidence", {})
        session_data["technical_questions"] = workflow_result.get("next_questions", [])
        session_data["consultation_phase"] = "gathering"
        
        # Clean up workflow result before saving to session
        if "state" in workflow_result:
            del workflow_result["state"]
        if "files_info" in workflow_result:
            del workflow_result["files_info"]

        # Update analysis_result with the cleaned workflow_result
        session_data["analysis_result"] = {
            **session_data.get("analysis_result", {}),
            **workflow_result
        }
        
        session_data["conversation"].append({
            "type": "assistant",
            "message": ai_response,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        
        session_data["last_activity"] = datetime.now(timezone.utc).isoformat()
        await save_session_safely(session_id, session_data)
        
        return {
            "session_id": session_id,
            "files_uploaded": len(files),
            "analysis": analysis_result,
            "ai_response": ai_response,
            "message": f"✅ {len(files)} fișiere analizate cu succes" + 
                      (" (cu analiză DXF automată)" if analysis_result.get("dxf_enhanced") else "")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ File upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Eroare la încărcarea fișierelor: {str(e)}")

@app.post("/session/{session_id}/chat")
async def chat_enhanced(
    session_id: str,
    message: str = Form(...)
):
    """Enhanced chat with Phase 2 agent consultation and DXF context"""
    try:
        session_data = await load_session_safely(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Sesiune inexistentă")
        
        session_data.get("conversation", []).append({
            "type": "user",
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        # NEW: Use the contextual response creator
        response_details = await create_contextual_response(session_id, message, session_data)
        offer_response = response_details["response"]
        response_type = response_details["response_type"]

        # If the workflow was re-run, update the session data
        if "workflow_result" in response_details:
            workflow_result = response_details["workflow_result"]
            session_data["analysis_result"].update({
                **workflow_result, # Store the entire result
                "confidence": workflow_result.get("overall_confidence", 0),
                "consultation_phase": "gathering",
                "agent_breakdown": workflow_result.get("agent_confidence", {}),
                "can_generate_offer": workflow_result.get("can_generate_offer", False),
                "technical_questions": workflow_result.get("next_questions", []),
                "file_analysis": workflow_result.get("file_analysis", {}),
                "dxf_enhanced": workflow_result.get("dxf_enhanced", False),
                "auto_generated_content": workflow_result.get("file_description")
            })
            session_data["agent_confidence"] = workflow_result.get("agent_confidence", {})
            session_data["technical_questions"] = workflow_result.get("next_questions", [])
            session_data["consultation_phase"] = "gathering"

        session_data.get("conversation", []).append({
            "type": "assistant",
            "message": offer_response,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "response_type": response_type
        })
        
        session_data["last_activity"] = datetime.now(timezone.utc).isoformat()
        await save_session_safely(session_id, session_data)
        
        return {
            "session_id": session_id,
            "response": offer_response,
            "conversation_length": len(session_data["conversation"]),
            "consultation_phase": session_data.get("consultation_phase", "initial"),
            "overall_confidence": session_data.get("analysis_result", {}).get("overall_confidence", session_data.get("analysis_result", {}).get("confidence", 0)),
            "can_generate_offer": session_data.get("analysis_result", {}).get("can_generate_offer", False)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Eroare în conversație: {str(e)}")

async def create_contextual_response(session_id: str, message: str, session_data: Dict) -> Dict[str, Any]:
    """
    Creates a contextual response by deciding the best action based on the user's message.
    """
    # 1. Check for offer generation request
    if "genereaza oferta" in message.lower():
        analysis_result = session_data.get("analysis_result", {})
        if analysis_result.get("can_generate_offer", False):
            response = await generate_professional_offer_with_dxf(session_data)
            return {"response": response, "response_type": "offer_generated"}
        else:
            overall_confidence = analysis_result.get("overall_confidence", 0)
            response = f"""⚠️ **Oferta nu poate fi generată încă**
            
📊 **Încredere actuală: {overall_confidence:.1f}%** (necesară: {settings.min_confidence_for_offer:.0f}%)

❓ **Informații lipsă:**
{chr(10).join([f'- {q}' for q in session_data.get("technical_questions", ["Mai multe detalii despre proiect"])])}

Vă rog să furnizați aceste informații pentru o ofertă precisă."""
            return {"response": response, "response_type": "insufficient_data"}

    # 2. Check for detailed description request
    if detect_detailed_description_request(message):
        detailed_description = await get_detailed_description_from_agent(session_id)
        return {"response": detailed_description, "response_type": "detailed_description"}

    # 3. Decide whether to re-run the full workflow for other messages
    if should_rerun_workflow(message, session_data):
        logger.info(f"Rerunning full workflow for session {session_id} based on new message.")
        workflow_result = await run_complete_workflow(
            session_id=session_id,
            files_info=session_data.get("files", []),
            user_requirements=message,
            existing_session_data=session_data
        )
        
        ai_response = await generate_natural_response_from_workflow(
            workflow_result, session_data, message
        )
        return {"response": ai_response, "response_type": "consultation", "workflow_result": workflow_result}
    
    # 4. Fallback for simple Q&A (if workflow re-run is not needed)
    logger.info(f"Handling simple Q&A for session {session_id}")
    context_summary = f"""Contextul actual al proiectului (Sesiune: {session_id}):
- Încredere generală: {session_data.get("analysis_result", {}).get("overall_confidence", 0):.1f}%
- Analiză fișiere: {session_data.get("analysis_result", {}).get("auto_generated_content", 'N/A')}
- Următoarele întrebări: {session_data.get('technical_questions', [])}"""

    prompt = f"Răspunde la următoarea întrebare a utilizatorului, folosind strict contextul furnizat:\n\n{context_summary}\n\nÎntrebare utilizator: \"{message}\""
    
    if LLM_AVAILABLE and safe_construction_llm_service:
        response = await safe_construction_llm_service.safe_construction_call(user_input=prompt, domain="general_construction")
    else:
        response = "Serviciul de chat nu este complet operațional pentru a răspunde la întrebări."
        
    return {"response": response, "response_type": "qa"}

def should_rerun_workflow(message: str, session_data: Dict) -> bool:
    """Determine if the chat message requires a full workflow re-execution."""
    message_lower = message.lower()
    rerun_keywords = ["buget", "materiale", "calitate", "suprafață", "suprafata", "modific", "adăug", "adaug", "schimb", "renunț", "plan", "cerințe"]
    return any(keyword in message_lower for keyword in rerun_keywords)

async def generate_professional_offer_with_dxf(session_data: Dict) -> str:
    """Generate professional Romanian construction offer with DXF insights"""
    agent_confidence = session_data.get("analysis_result", {}).get("agent_breakdown", {})
    
    files_count = len(session_data.get("files", []))
    dxf_available = session_data.get("analysis_result", {}).get("dxf_enhanced", False) or session_data.get("dxf_analysis_available", False)
    auto_description = session_data.get("analysis_result", {}).get("auto_generated_content", "")
    
    dxf_analysis_section = ""
    if dxf_available and auto_description:
        dxf_analysis_section = f"""
🏗️ **ANALIZĂ DXF AVANSATĂ**
**Conținut tehnic detectat:** {auto_description}
**Procesare:** Analiză automată cu extracție completă de date tehnice

"""
    
    return f"""🗁 **OFERTĂ TEHNICĂ ȘI COMERCIALĂ**

**DEMOPLAN CONSTRUCT SRL**
Data: {datetime.now().strftime('%d.%m.%Y')}
Sesiune: {session_data.get('session_id', '')[:8]}...

╔═══════════════════════════════════════════════════════════════════════════════════════════╗

📋 **1. ANALIZĂ TEHNICĂ COMPLETĂ**
{dxf_analysis_section}
**Încredere analiză:** {session_data.get('analysis_result', {}).get('overall_confidence', session_data.get('analysis_result', {}).get('confidence', 0)):.1f}%
- Plan arhitectural: {agent_confidence.get('floorplan', 0):.1f}%
- Validare tehnică: {agent_confidence.get('validation', 0):.1f}%  
- Estimare costuri: {agent_confidence.get('estimation', 0):.1f}%
- Compunere ofertă: {agent_confidence.get('offer_composer', 0):.1f}%

**Documente analizate:** {files_count} fișiere{"" if not dxf_available else " (incluzând analiză DXF automată)"}

╠═══════════════════════════════════════════════════════════════════════════════════════════╣

💰 **2. ESTIMARE COSTURI**

**Costuri preliminare pe categorii:**
- Materiale: 15,000 - 25,000 RON
- Manoperă: 8,000 - 12,000 RON
- Transport și utilaje: 2,000 - 3,500 RON
- TVA 19%: 4,750 - 7,695 RON

**TOTAL ESTIMAT: 29,750 - 48,195 RON**

*Prețurile finale vor fi ajustate după confirmarea tuturor specificațiilor tehnice.*

╠═══════════════════════════════════════════════════════════════════════════════════════════╣

⏱️ **3. PROGRAM DE EXECUȚIE**

**Durata estimată:** 3-5 săptămâni
- Pregătire și mobilizare: 2-3 zile
- Execuție lucrări principale: 15-25 zile  
- Finisaje și demobilizare: 3-5 zile

**Data începerii:** După semnarea contractului
**Condiții meteo:** Sezon optim Aprilie-Octombrie

╠═══════════════════════════════════════════════════════════════════════════════════════════╣

📋 **4. SPECIFICAȚII TEHNICE**

**Materiale incluse:**
- Materiale conform standardelor românești SR EN
- Transport la șantier inclus
- Garanție materiale: 2 ani

**Manoperă:**
- Echipă specializată cu experiență 5+ ani
- Responsabil tehnic cu atestare ANIF
- Garanție lucrări: 3 ani

╠═══════════════════════════════════════════════════════════════════════════════════════════╣

⚖️ **5. CONDIȚII CONTRACTUALE**

**Modalitate plată:**
- Avans: 30% la semnarea contractului  
- Tranșe progres: 40% la 50% execuție
- Rest: 30% la recepția lucrărilor

**Garanții:**
- Garanție lucrări: 3 ani
- Garanție materiale: 2 ani
- Service post-garanție disponibil

**Documente incluse:**
- Proiect tehnic de execuție
- Certificat de garanție
- Manual de întreținere

╠═══════════════════════════════════════════════════════════════════════════════════════════╣

📞 **6. CONTACT**

Pentru întrebări sau programarea unei vizite tehnice:
- Email: office@demoplan.ro  
- Telefon: 0721.XXX.XXX

**Oferta este valabilă 30 de zile de la data emiterii.**

*Ofertă generată automat prin sistemul DEMOPLAN cu analiză AI avansată{"" if not dxf_available else " și procesare DXF automată"}.*
*Pentru ofertă detaliată personalizată, vă rugăm să contactați echipa noastră tehnică.*

╚═══════════════════════════════════════════════════════════════════════════════════════════╝

✅ **Următorii pași:**
1. Revizuiți oferta și specificațiile
2. Solicitați clarificări dacă este necesar  
3. Confirmați acceptarea pentru programarea vizitei tehnice
4. Semnarea contractului după acordul final

**Vă mulțumim pentru încrederea acordată!**"""

@app.get("/session/{session_id}")
async def get_session_status(session_id: str):
    """Get enhanced session status with Phase 2 and DXF information"""
    try:
        session_data = await load_session_safely(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Sesiune inexistentă")
        
        return {
            "session_id": session_id,
            "status": session_data.get("status", "unknown"),
            "created_at": session_data.get("created_at"),
            "last_activity": session_data.get("last_activity"),
            "files_count": len(session_data.get("files", [])),
            "conversation_length": len(session_data.get("conversation", [])),
            
            # Phase 2 Enhanced fields
            "agent_mode": session_data.get("agent_mode", False),
            "consultation_phase": session_data.get("consultation_phase", "initial"),
            "overall_confidence": session_data.get("analysis_result", {}).get("overall_confidence", session_data.get("analysis_result", {}).get("confidence", 0)),
            "agent_breakdown": session_data.get("analysis_result", {}).get("agent_breakdown", {}),
            "can_generate_offer": session_data.get("analysis_result", {}).get("can_generate_offer", False),
            "technical_questions": session_data.get("technical_questions", []),
            
            # DXF Analysis fields
            "dxf_analysis_available": session_data.get("analysis_result", {}).get("dxf_enhanced", False) or session_data.get("dxf_analysis_available", False),
            "auto_generated_description": session_data.get("analysis_result", {}).get("auto_generated_content"),
            "technical_content_extracted": session_data.get("analysis_result", {}).get("dxf_enhanced", False) or session_data.get("technical_content_extracted", False)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Session status error: {e}")
        raise HTTPException(status_code=500, detail=f"Eroare la status sesiune: {str(e)}")

# =============================================================================
# SERVER STARTUP WITH CLOUD RUN COMPATIBILITY
# =============================================================================

if __name__ == "__main__":
    try:
        port = int(os.getenv("PORT", 8080))
        logger.info(f"🚀 Starting DEMOPLAN Phase 2 Enhanced Server with DXF Analysis on port {port}")
        
        uvicorn.run(
            app, 
            host="0.0.0.0",
            port=port,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"❌ Server startup failed: {e}")
        sys.exit(1)
