"""
DemoPLAN Workflow - Simplified Version
Bypasses circular imports and complex dependencies
"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger("demoplan.workflow")

async def initialize_workflow():
    """Simple initialization"""
    logger.info("✅ Simple workflow initialized")

async def run_complete_workflow(
    session_id: str,
    files_info: List[Dict[str, Any]],
    user_requirements: Optional[str] = None,
    existing_session_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Simplified workflow that actually works
    """
    try:
        logger.info(f"🔄 Processing session {session_id}")
        
        # Basic confidence calculation
        confidence = 0.3  # Base
        
        # Boost for files
        if files_info:
            confidence += len(files_info) * 0.1
        
        # Boost for user input
        if user_requirements and len(user_requirements.strip()) > 10:
            confidence += 0.2
        
        # Cap confidence
        confidence = min(confidence, 0.8)
        
        # Generate Romanian response using LLM
        romanian_response = await _generate_romanian_response(user_requirements or "Proiect de construcții")
        
        # Basic questions based on confidence
        if confidence < 0.5:
            next_questions = [
                "Care este suprafața totală a spațiului?",
                "Câte camere are proiectul?",
                "Ce tip de lucrări doriți (renovare, construcție nouă)?"
            ]
        else:
            next_questions = [
                "Aveți preferințe pentru materiale?",
                "Care este bugetul aproximativ?"
            ]
        
        return {
            "session_id": session_id,
            "overall_confidence": confidence,
            "agent_confidence": {
                "floorplan": confidence * 0.7,
                "validation": confidence * 0.8,
                "estimation": confidence * 0.6,
                "offer_composer": confidence * 0.9
            },
            "can_generate_offer": confidence > 0.6,
            "next_questions": next_questions,
            "file_analysis": {
                "files_processed": len(files_info) if files_info else 0,
                "dxf_analysis_available": False
            },
            "dxf_enhanced": False,
            "file_description": f"{len(files_info)} fișiere procesate" if files_info else "Fără fișiere",
            "processing_time_seconds": 0.1,
            "workflow_status": "simplified_success",
            "romanian_response": romanian_response
        }
        
    except Exception as e:
        logger.error(f"❌ Simple workflow failed: {e}")
        return {
            "session_id": session_id,
            "overall_confidence": 0.2,
            "agent_confidence": {"floorplan": 0.2, "validation": 0.2, "estimation": 0.2, "offer_composer": 0.2},
            "can_generate_offer": False,
            "next_questions": ["Vă rog să descrieți proiectul dumneavoastră."],
            "file_analysis": {"files_processed": 0},
            "dxf_enhanced": False,
            "file_description": "Eroare în procesare",
            "processing_time_seconds": 0.1,
            "workflow_status": "error",
            "error": str(e)
        }

async def _generate_romanian_response(user_input: str) -> str:
    """Generate Romanian response using LLM directly"""
    try:
        # Import LLM service directly
        from src.services.llm_service import safe_construction_llm_service
        
        prompt = f"""Ești un consultant român în construcții. 
        Utilizatorul a spus: "{user_input}"
        
        Răspunde în română, profesional și util pentru proiecte de construcții din România.
        Menționează costuri în RON și materiale disponibile pe piața românească."""
        
        response = await safe_construction_llm_service.safe_construction_call(
            user_input=prompt,
            domain="general_construction",
            system_prompt="Ești expert în construcții din România. Răspunde în română."
        )
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Romanian response generation failed: {e}")
        return f"Am înțeles că doriți să discutați despre: {user_input}. Vă pot ajuta cu consultanță pentru construcții în România."

async def get_workflow_status() -> Dict[str, Any]:
    """Get simple workflow status"""
    return {
        "workflow_initialized": True,
        "agent_available": False,
        "agent_ready": False,
        "version": "simplified_v1.0",
        "fallback_mode": True
    }