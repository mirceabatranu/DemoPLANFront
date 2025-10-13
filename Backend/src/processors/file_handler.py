# src/processors/file_handler.py
"""
File handling processor for the DemoPLAN Unified Agent.
This module is responsible for preparing file data for agent analysis.
"""

from typing import List, Dict, Any
from fastapi import UploadFile, HTTPException
import logging

logger = logging.getLogger("demoplan.processors.file_handler")

async def handle_file_upload(
    session_id: str,
    files: List[UploadFile],
    requirements: str,
    unified_agent: Any  # Agent is passed as a dependency
) -> Dict[str, Any]:
    """
    Processes uploaded files and triggers a holistic agent analysis.

    Changes based on subcollection migration:
    - This function no longer loads, modifies, or saves the session object.
    - It prepares file content in memory and passes it to the agent.
    - The agent is now solely responsible for analyzing the files and persisting the results
      to the appropriate 'file_analyses' subcollection in Firestore.
    - The return value is a summary of the analysis, not the full data, and may
      include references to the created analysis documents.
    """
    if not unified_agent:
        logger.warning(f"File upload attempted for session {session_id} but agent was not available.")
        raise HTTPException(status_code=503, detail="Agentul de analiză nu este disponibil.")

    try:
        # 1. Prepare file data from the UploadFile objects.
        files_data: List[Dict[str, Any]] = []
        for file in files:
            content = await file.read()
            file_info = {
                "filename": file.filename,
                "content_type": file.content_type,
                "content": content,
            }
            files_data.append(file_info)

        logger.info(f"📁 Prepared {len(files_data)} files for agent analysis in session {session_id}")

        # 2. Call the agent to perform the analysis.
        # The agent's internal logic now includes saving each file's analysis
        # to a separate document in a Firestore subcollection.
        analysis_result = await unified_agent.analyze_project(
            files=files_data,
            user_input=requirements,
            session_id=session_id
        )

        # 3. Return a summary of the operation.
        # The detailed analysis is no longer held in memory or returned here.
        # We return references or high-level results provided by the agent.
        return {
            "session_id": session_id,
            "files_processed": len(files_data),
            "ai_response": analysis_result.get("response", "Fișierele au fost procesate cu succes."),
            "confidence": analysis_result.get("confidence", 0),
            "can_generate_offer": analysis_result.get("can_generate_offer", False),
            "file_references": analysis_result.get("file_ids", [])  # Agent should return IDs of created analysis docs
        }
    except Exception as e:
        logger.error(f"❌ Unhandled error in file processing for session {session_id}: {e}", exc_info=True)
        # Re-raise as an HTTPException to be handled by the API layer.
        raise HTTPException(status_code=500, detail=f"A apărut o eroare internă la procesarea fișierelor: {str(e)}")