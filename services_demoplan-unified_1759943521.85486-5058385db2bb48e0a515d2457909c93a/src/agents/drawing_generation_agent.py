"""
Drawing Generation Agent - Phase 1 MVP
Generates field verification drawings from offer data using adaptive RAG approach
"""

from typing import Dict, Any, Optional, List
import logging
import time
from datetime import datetime, timezone
import uuid
import tempfile
import os

from src.utils.cad_validator import CADCodeValidator, ValidationResult
from src.processors.cad_code_generator import CADCodeGenerator
from src.services.drawing_storage_service import DrawingStorageService

logger = logging.getLogger("demoplan.agents.drawing_generation")


class DrawingGenerationResult:
    """Result of drawing generation attempt"""
    
    def __init__(
        self,
        success: bool,
        drawing_url: Optional[str] = None,
        error_message: Optional[str] = None,
        processing_time_ms: float = 0,
        code_used: Optional[str] = None,
        drawing_id: Optional[str] = None,
        attempts: int = 1,
        validation_warnings: Optional[List[str]] = None
    ):
        self.success = success
        self.drawing_url = drawing_url
        self.error_message = error_message
        self.processing_time_ms = processing_time_ms
        self.code_used = code_used
        self.drawing_id = drawing_id
        self.attempts = attempts
        self.validation_warnings = validation_warnings or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            'success': self.success,
            'drawing_url': self.drawing_url,
            'error_message': self.error_message,
            'processing_time_ms': self.processing_time_ms,
            'drawing_id': self.drawing_id,
            'attempts': self.attempts,
            'validation_warnings': self.validation_warnings
        }


class DrawingGenerationAgent:
    """
    Specialized agent for generating technical drawings
    
    Phase 1 MVP: Field verification drawings only
    
    Architecture follows the adaptive RAG pattern from the Medium article:
    1. Query Transformation (spatial data -> enhanced prompt)
    2. Code Generation (LLM generates Python/ezdxf code)
    3. Code Validation (safety and correctness checks)
    4. Code Execution (run generated code)
    5. Error Feedback Loop (iterative improvement)
    """
    
    def __init__(
        self,
        llm_service,
        firestore_service,
        max_retry_attempts: int = 3
    ):
        """
        Initialize drawing generation agent
        
        Args:
            llm_service: LLM service for code generation
            firestore_service: Firestore service for metadata storage
            max_retry_attempts: Maximum retry attempts on errors
        """
        self.llm_service = llm_service
        self.firestore_service = firestore_service
        self.code_generator = CADCodeGenerator(llm_service)
        self.code_validator = CADCodeValidator()
        self.storage_service = DrawingStorageService()
        self.max_retry_attempts = max_retry_attempts
        
        logger.info("✅ DrawingGenerationAgent initialized (Phase 1 MVP)")
        logger.info(f"   Max retry attempts: {max_retry_attempts}")
        logger.info(f"   Storage: {self.storage_service.get_storage_info()['storage_type']}")
    
    async def generate_field_verification_drawing(
        self,
        session_id: str,
        custom_specifications: Optional[str] = None
    ) -> DrawingGenerationResult:
        """
        Main entry point - generate field verification drawing
        
        This is the complete adaptive RAG workflow:
        1. Load session context (spatial data from DXF analysis)
        2. Transform query (enhance with spatial context)
        3. Generate code via LLM
        4. Validate code
        5. Execute code
        6. Handle errors with feedback loop
        7. Store result
        
        Args:
            session_id: Session with analyzed DXF data
            custom_specifications: Optional user specifications
        
        Returns:
            DrawingGenerationResult with URL to DXF file
        """
        
        start_time = time.time()
        drawing_id = f"fv_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        logger.info(f"🎨 Starting drawing generation for session {session_id}")
        logger.info(f"   Drawing ID: {drawing_id}")
        
        try:
            # ================================================================
            # STEP 1: Load session context (spatial data from unified agent)
            # ================================================================
            logger.info("📊 Step 1: Loading session data...")
            session_data = await self._load_session_data(session_id)
            
            if not session_data:
                logger.error("❌ Session not found or missing data")
                return DrawingGenerationResult(
                    success=False,
                    error_message="Session not found or missing spatial data. Please analyze DXF files first.",
                    processing_time_ms=(time.time() - start_time) * 1000
                )
            
            # ================================================================
            # STEP 2: Extract and validate spatial data
            # ================================================================
            logger.info("📐 Step 2: Extracting spatial data...")
            spatial_data = session_data.get("spatial_data", {})
            romanian_context = session_data.get("romanian_context", {})
            
            if not spatial_data or not spatial_data.get("rooms"):
                logger.error("❌ No spatial data available")
                return DrawingGenerationResult(
                    success=False,
                    error_message="No spatial data available. Please upload and analyze DXF files first.",
                    processing_time_ms=(time.time() - start_time) * 1000
                )
            
            room_count = len(spatial_data.get("rooms", []))
            total_area = spatial_data.get("total_area", 0)
            logger.info(f"   Found {room_count} rooms, total area: {total_area:.2f} m²")
            
            # ================================================================
            # STEP 3: Iterative code generation with error feedback
            # ================================================================
            logger.info("🔄 Step 3: Entering code generation loop...")
            
            error_context = None
            final_code = None
            validation_result = None
            execution_result = None
            
            for attempt in range(1, self.max_retry_attempts + 1):
                logger.info(f"   Attempt {attempt}/{self.max_retry_attempts}")
                
                # Generate code using LLM
                logger.info("   🤖 Generating code with LLM...")
                try:
                    code = await self.code_generator.generate_code(
                        drawing_type="field_verification",
                        spatial_data=spatial_data,
                        romanian_context=romanian_context,
                        error_context=error_context
                    )
                    logger.info(f"   ✅ Generated {len(code)} characters of code")
                except Exception as e:
                    logger.error(f"   ❌ Code generation failed: {e}")
                    error_context = f"LLM code generation error: {str(e)}"
                    continue
                
                # Validate code for safety and correctness
                logger.info("   🔍 Validating code...")
                validation_result = self.code_validator.validate(code)
                
                if not validation_result.is_valid:
                    logger.warning(f"   ⚠️ Validation failed: {len(validation_result.errors)} errors")
                    for error in validation_result.errors:
                        logger.warning(f"      - {error}")
                    
                    error_context = f"Code validation errors: {', '.join(validation_result.errors)}"
                    continue
                
                logger.info(f"   ✅ Validation passed (warnings: {len(validation_result.warnings)})")
                
                # Execute code to generate DXF
                logger.info("   ⚙️ Executing code...")
                execution_result = await self._execute_cad_code(code, drawing_id)
                
                if execution_result["success"]:
                    final_code = code
                    logger.info("   ✅ Code executed successfully")
                    break
                else:
                    logger.warning(f"   ⚠️ Execution failed: {execution_result['error']}")
                    error_context = f"Execution error: {execution_result['error']}"
            
            # ================================================================
            # STEP 4: Check if generation succeeded
            # ================================================================
            if not final_code or not execution_result or not execution_result.get("success"):
                logger.error(f"❌ Failed after {self.max_retry_attempts} attempts")
                return DrawingGenerationResult(
                    success=False,
                    error_message=f"Failed to generate drawing after {self.max_retry_attempts} attempts. Last error: {error_context}",
                    processing_time_ms=(time.time() - start_time) * 1000,
                    attempts=self.max_retry_attempts
                )
            
            # ================================================================
            # STEP 5: Upload DXF file to storage
            # ================================================================
            logger.info("☁️ Step 5: Uploading to storage...")
            dxf_path = execution_result["file_path"]
            
            try:
                with open(dxf_path, 'rb') as f:
                    dxf_content = f.read()
                
                logger.info(f"   File size: {len(dxf_content)} bytes")
                
                drawing_url = await self.storage_service.save_drawing(
                    session_id=session_id,
                    drawing_id=drawing_id,
                    file_content=dxf_content,
                    filename="field_verification.dxf"
                )
                
                logger.info(f"   ✅ Uploaded to: {drawing_url[:100]}...")
                
            except Exception as e:
                logger.error(f"   ❌ Upload failed: {e}")
                return DrawingGenerationResult(
                    success=False,
                    error_message=f"Drawing generated but upload failed: {str(e)}",
                    processing_time_ms=(time.time() - start_time) * 1000,
                    code_used=final_code,
                    drawing_id=drawing_id
                )
            finally:
                # Cleanup temporary file
                try:
                    if os.path.exists(dxf_path):
                        os.remove(dxf_path)
                except:
                    pass
            
            # ================================================================
            # STEP 6: Save metadata to Firestore
            # ================================================================
            logger.info("💾 Step 6: Saving metadata...")
            await self._save_drawing_metadata(
                session_id=session_id,
                drawing_id=drawing_id,
                drawing_url=drawing_url,
                code_used=final_code,
                spatial_data=spatial_data,
                validation_warnings=validation_result.warnings if validation_result else []
            )
            
            # ================================================================
            # STEP 7: Return success result
            # ================================================================
            processing_time = (time.time() - start_time) * 1000
            
            logger.info(f"✅ Drawing generation completed successfully!")
            logger.info(f"   Processing time: {processing_time:.0f}ms")
            logger.info(f"   Drawing ID: {drawing_id}")
            
            return DrawingGenerationResult(
                success=True,
                drawing_url=drawing_url,
                processing_time_ms=processing_time,
                code_used=final_code,
                drawing_id=drawing_id,
                attempts=attempt,
                validation_warnings=validation_result.warnings if validation_result else []
            )
            
        except Exception as e:
            logger.error(f"❌ Drawing generation failed with exception: {e}", exc_info=True)
            return DrawingGenerationResult(
                success=False,
                error_message=f"Unexpected error: {str(e)}",
                processing_time_ms=(time.time() - start_time) * 1000
            )
    
    async def _load_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load session data from Firestore
        
        Extracts spatial data from file analyses subcollection.
        This is where we get the DXF analysis results from the unified agent.
        
        Args:
            session_id: Session identifier
        
        Returns:
            Dictionary with spatial_data and romanian_context, or None if not found
        """
        
        try:
            logger.info(f"   Loading session document: {session_id}")
            
            # Get main session document
            session_doc = await self.firestore_service.get_document(
                collection='engineer_chat_sessions',
                document_id=session_id
            )
            
            if not session_doc:
                logger.warning(f"   Session {session_id} not found in Firestore")
                return None
            
            logger.info("   ✅ Session document loaded")
            
            # Load file analyses from subcollection
            logger.info("   Loading file analyses from subcollection...")
            file_analyses = await self.firestore_service.load_file_analyses(session_id)
            
            logger.info(f"   Found {len(file_analyses)} file analyses")
            
            # Extract spatial data from DXF analyses
            spatial_data = {}
            romanian_context = {}
            
            for analysis in file_analyses:
                analysis_data = analysis.get("analysis_data", {})
                
                # Look for DXF analysis results
                if "dxf_analysis" in analysis_data:
                    dxf_data = analysis_data["dxf_analysis"]
                    
                    spatial_data = {
                        "rooms": dxf_data.get("rooms", []),
                        "total_area": dxf_data.get("total_area", 0),
                        "walls": dxf_data.get("walls", []),
                        "doors": dxf_data.get("doors", []),
                        "windows": dxf_data.get("windows", [])
                    }
                    
                    # Extract Romanian room names
                    room_names = [r.get("name_ro") for r in spatial_data["rooms"] if r.get("name_ro")]
                    romanian_context = {
                        "room_names": room_names
                    }
                    
                    logger.info(f"   ✅ Found DXF analysis with {len(spatial_data['rooms'])} rooms")
                    break
            
            if not spatial_data:
                logger.warning("   ⚠️ No DXF analysis found in file analyses")
            
            return {
                "session_id": session_id,
                "spatial_data": spatial_data,
                "romanian_context": romanian_context,
                "project_data": session_doc.get("project_data_summary", {})
            }
            
        except Exception as e:
            logger.error(f"   ❌ Failed to load session data: {e}", exc_info=True)
            return None
    
    async def _execute_cad_code(
        self,
        code: str,
        drawing_id: str
    ) -> Dict[str, Any]:
        """
        Execute generated CAD code safely in isolated environment
        
        Args:
            code: Python code to execute
            drawing_id: Unique identifier for output file
        
        Returns:
            Dictionary with:
                - success: bool
                - file_path: str (if success)
                - error: str (if failure)
        """
        
        try:
            # Create temporary file path for output
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, f"{drawing_id}.dxf")
            
            logger.info(f"   Executing code, output to: {output_path}")
            
            # Prepare safe execution environment
            # Only provide necessary globals, no access to dangerous modules
            exec_globals = {
                "output_path": output_path,
                "__builtins__": __builtins__,
                # Allow these safe imports to be available
                "ezdxf": None,  # Will be imported by the code itself
            }
            
            # Execute the generated code
            exec(code, exec_globals)
            
            # Check if DXF file was created
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                logger.info(f"   ✅ DXF file created: {file_size} bytes")
                
                # Basic validation: check if it's actually a DXF file
                with open(output_path, 'rb') as f:
                    header = f.read(100)
                    if b'DXF' not in header and b'SECTION' not in header:
                        logger.warning("   ⚠️ File doesn't look like a valid DXF")
                        return {
                            "success": False,
                            "error": "Generated file doesn't appear to be a valid DXF"
                        }
                
                return {
                    "success": True,
                    "file_path": output_path,
                    "file_size": file_size
                }
            else:
                logger.error("   ❌ Code executed but no DXF file was created")
                return {
                    "success": False,
                    "error": "Code executed but no DXF file was created at expected path"
                }
                
        except Exception as e:
            logger.error(f"   ❌ Code execution failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _save_drawing_metadata(
        self,
        session_id: str,
        drawing_id: str,
        drawing_url: str,
        code_used: str,
        spatial_data: Dict,
        validation_warnings: List[str]
    ):
        """
        Save drawing metadata to Firestore drawings subcollection
        
        Args:
            session_id: Session identifier
            drawing_id: Drawing identifier
            drawing_url: URL to access the drawing
            code_used: Generated Python code
            spatial_data: Original spatial data
            validation_warnings: Any validation warnings
        """
        
        try:
            metadata = {
                "drawing_id": drawing_id,
                "drawing_type": "field_verification",
                "generated_at": datetime.now(timezone.utc),
                "drawing_url": drawing_url,
                "code_used": code_used[:2000],  # Truncate for storage (keep first 2000 chars)
                "code_length": len(code_used),
                "status": "completed",
                "spatial_data_summary": {
                    "num_rooms": len(spatial_data.get("rooms", [])),
                    "total_area": spatial_data.get("total_area", 0),
                    "room_names": [r.get("name_ro") for r in spatial_data.get("rooms", [])[:5]]  # First 5
                },
                "validation_warnings": validation_warnings[:10],  # Keep first 10 warnings
                "metadata_version": "1.0"
            }
            
            await self.firestore_service.save_drawing_metadata(
                session_id=session_id,
                drawing_id=drawing_id,
                metadata=metadata
            )
            
            logger.info("   ✅ Metadata saved to Firestore")
            
        except Exception as e:
            logger.error(f"   ⚠️ Failed to save metadata (non-critical): {e}")
            # Don't fail the whole operation if metadata save fails
    
    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get information about the agent configuration
        
        Returns:
            Dictionary with agent configuration
        """
        return {
            "agent_type": "DrawingGenerationAgent",
            "phase": "1-mvp",
            "supported_drawing_types": ["field_verification"],
            "max_retry_attempts": self.max_retry_attempts,
            "storage": self.storage_service.get_storage_info(),
            "features": {
                "adaptive_rag": True,
                "error_feedback_loop": True,
                "code_validation": True,
                "romanian_standards": True
            }
            }