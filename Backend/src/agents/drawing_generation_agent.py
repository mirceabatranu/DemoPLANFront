"""
Drawing Generation Agent - Phase 1 MVP with HYBRID Approach
Generates field verification drawings from spatial data
Supports both geometric (precise) and text-based (flexible) extraction
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
from src.services.storage_service import geometric_storage

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
        validation_warnings: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.success = success
        self.drawing_url = drawing_url
        self.error_message = error_message
        self.processing_time_ms = processing_time_ms
        self.code_used = code_used
        self.drawing_id = drawing_id
        self.attempts = attempts
        self.validation_warnings = validation_warnings or []
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        result = {
            'success': self.success,
            'drawing_url': self.drawing_url,
            'error_message': self.error_message,
            'processing_time_ms': self.processing_time_ms,
            'drawing_id': self.drawing_id,
            'attempts': self.attempts,
            'validation_warnings': self.validation_warnings
        }
        if self.metadata:
            result['metadata'] = self.metadata
        return result


class DrawingGenerationAgent:
    """
    Specialized agent for generating technical drawings with HYBRID approach
    
    Phase 1: Field verification drawings
    
    Extraction Strategy:
    1. Try GEOMETRIC extraction (precise, from closed polylines)
    2. Fall back to TEXT-BASED extraction (flexible, from schedules/descriptions)
    3. Clear error if both fail
    
    Architecture follows adaptive RAG pattern:
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
        
        logger.info("✅ DrawingGenerationAgent initialized (Phase 1 MVP - HYBRID)")
        logger.info(f"   Max retry attempts: {max_retry_attempts}")
        logger.info(f"   Storage: {self.storage_service.get_storage_info()['storage_type']}")
        logger.info(f"   Extraction: Geometric + Text-based fallback")
    
    async def generate_field_verification_drawing(
        self,
        session_id: str,
        custom_specifications: Optional[str] = None
    ) -> DrawingGenerationResult:
        """
        Generate field verification drawing with HYBRID approach
        
        Handles both geometric and text-based spatial data
        """
        
        start_time = time.time()
        drawing_id = f"fv_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        logger.info(f"🎨 Starting HYBRID drawing generation for session {session_id}")
        logger.info(f"   Drawing ID: {drawing_id}")
        
        try:
            # STEP 1: Load session data (tries geometric, falls back to text)
            logger.info("📊 Step 1: Loading session data (hybrid approach)...")
            session_data = await self._load_session_data(session_id)
            
            if not session_data:
                logger.error("❌ Session not found or no usable data")
                return DrawingGenerationResult(
                    success=False,
                    error_message="Session not found or missing spatial data. Please upload a floor plan with room boundaries or a detailed room schedule.",
                    processing_time_ms=(time.time() - start_time) * 1000
                )
            
            spatial_data = session_data.get("spatial_data", {})
            source_type = session_data.get("source_type", "unknown")
            precision = session_data.get("precision", "unknown")
            
            if not spatial_data.get("rooms"):
                logger.error("❌ No rooms found in spatial data")
                return DrawingGenerationResult(
                    success=False,
                    error_message="No rooms detected. Please provide a floor plan with visible room boundaries or a room schedule with areas.",
                    processing_time_ms=(time.time() - start_time) * 1000
                )
            
            # Log data source
            logger.info("=" * 80)
            logger.info(f"📋 DATA SOURCE: {source_type.upper()}")
            logger.info(f"   Precision: {precision}")
            logger.info(f"   Rooms: {len(spatial_data['rooms'])}")
            logger.info(f"   Total Area: {spatial_data['total_area']:.2f} m²")
            
            if source_type == "text":
                logger.warning("   ⚠️ Generated from text description, not geometric data")
                missing = session_data.get("missing_data", [])
                if missing:
                    logger.warning(f"   Missing data for higher precision:")
                    for item in missing[:3]:
                        logger.warning(f"      - {item}")
            
            logger.info("=" * 80)
            
            # STEP 2: Generate code with appropriate prompt
            logger.info("🔧 Step 2: Generating CAD code...")
            
            # Add source information to prompt context
            if source_type == "text":
                custom_specs_enhanced = (
                    f"⚠️ IMPORTANT: This drawing is generated from a room schedule/description, not geometric floor plan data.\n"
                    f"Precision: APPROXIMATE. Room positions and relationships are estimated.\n\n"
                    f"{custom_specifications or ''}"
                )
            else:
                custom_specs_enhanced = custom_specifications
            
            code = await self.code_generator.generate_cad_code(
                drawing_type="field_verification",
                spatial_data=spatial_data,
                romanian_context=session_data.get("romanian_context", {}),
                custom_specifications=custom_specs_enhanced
            )
            
            logger.info(f"   ✅ Generated {len(code)} characters of Python code")
            
            # STEP 3-6: Validate, execute, retry loop
            final_code = None
            execution_result = None
            validation_result = None
            
            for attempt in range(1, self.max_retry_attempts + 1):
                logger.info(f"🔄 Attempt {attempt}/{self.max_retry_attempts}")
                
                # Validate code
                validation_result = self.code_validator.validate_code(code)
                
                if not validation_result.is_valid:
                    logger.error(f"   ❌ Code validation failed: {validation_result.error}")
                    
                    if attempt < self.max_retry_attempts:
                        logger.info(f"   Retrying with error feedback...")
                        code = await self.code_generator.generate_cad_code(
                            drawing_type="field_verification",
                            spatial_data=spatial_data,
                            romanian_context=session_data.get("romanian_context", {}),
                            error_context=validation_result.error
                        )
                        continue
                    else:
                        return DrawingGenerationResult(
                            success=False,
                            error_message=f"Code validation failed after {attempt} attempts: {validation_result.error}",
                            processing_time_ms=(time.time() - start_time) * 1000,
                            attempts=attempt
                        )
                
                # Execute code
                logger.info("   ⚙️ Executing CAD code...")
                execution_result = await self._execute_cad_code(code, drawing_id)
                
                if execution_result["success"]:
                    logger.info("   ✅ Code executed successfully")
                    final_code = code
                    break
                else:
                    logger.error(f"   ❌ Execution failed: {execution_result['error']}")
                    
                    if attempt < self.max_retry_attempts:
                        logger.info(f"   Retrying with execution error feedback...")
                        code = await self.code_generator.generate_cad_code(
                            drawing_type="field_verification",
                            spatial_data=spatial_data,
                            romanian_context=session_data.get("romanian_context", {}),
                            error_context=execution_result["error"]
                        )
                    else:
                        return DrawingGenerationResult(
                            success=False,
                            error_message=f"Code execution failed after {attempt} attempts: {execution_result['error']}",
                            processing_time_ms=(time.time() - start_time) * 1000,
                            attempts=attempt
                        )
            
            # STEP 7: Upload to storage
            if not execution_result or not execution_result["success"]:
                return DrawingGenerationResult(
                    success=False,
                    error_message="Failed to generate drawing",
                    processing_time_ms=(time.time() - start_time) * 1000
                )
            
            logger.info("📤 Step 7: Uploading to storage...")
            drawing_url = await self.storage_service.upload_drawing(
                file_path=execution_result["file_path"],
                drawing_id=drawing_id,
                session_id=session_id
            )
            
            logger.info(f"   ✅ Drawing uploaded: {drawing_url}")
            
            # STEP 8: Save metadata
            logger.info("💾 Step 8: Saving metadata...")
            
            # Add source information to metadata
            validation_warnings = validation_result.warnings if validation_result else []
            
            if source_type == "text":
                validation_warnings.insert(0, 
                    "⚠️ Generated from text description (schedule), not geometric floor plan"
                )
                validation_warnings.insert(1,
                    "Precision: APPROXIMATE - room positions and dimensions are estimated"
                )
                if session_data.get("missing_data"):
                    validation_warnings.append(
                        f"Missing: {', '.join(session_data['missing_data'][:3])}"
                    )
            
            await self._save_drawing_metadata(
                session_id=session_id,
                drawing_id=drawing_id,
                drawing_url=drawing_url,
                code_used=final_code,
                spatial_data=spatial_data,
                validation_warnings=validation_warnings
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            logger.info("=" * 80)
            logger.info("✅ DRAWING GENERATION COMPLETE")
            logger.info(f"   Source: {source_type}")
            logger.info(f"   Precision: {precision}")
            logger.info(f"   URL: {drawing_url}")
            logger.info(f"   Time: {processing_time:.0f}ms")
            logger.info("=" * 80)
            
            return DrawingGenerationResult(
                success=True,
                drawing_url=drawing_url,
                processing_time_ms=processing_time,
                code_used=final_code,
                drawing_id=drawing_id,
                attempts=attempt,
                validation_warnings=validation_warnings,
                metadata={
                    "source_type": source_type,
                    "precision": precision,
                    "rooms_count": len(spatial_data.get("rooms", [])),
                    "total_area": spatial_data.get("total_area", 0)
                }
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
        Load session data with HYBRID extraction approach.
        
        NOW loads from both Firestore (summaries) and GCS (geometric data).
        
        Priority:
        1. Try GEOMETRIC extraction (precise, from closed polylines in GCS)
        2. Fall back to TEXT-BASED extraction (flexible, from summaries in Firestore)
        """
        try:
            logger.info("=" * 80)
            logger.info(f"LOADING SESSION DATA: {session_id}")
            logger.info("=" * 80)
            
            # STEP 1: Load session document
            logger.info("STEP 1: Loading session document from Firestore...")
            session_doc = await self.firestore_service.get_document("chat_sessions", session_id)
            
            if not session_doc:
                logger.error("❌ Session not found")
                return None
            
            logger.info(f"✅ Session loaded: {session_id}")
            
            # STEP 2: Load all file analyses (summaries from Firestore)
            logger.info("=" * 80)
            logger.info("STEP 2: Loading file analyses from Firestore (summaries)...")
            logger.info("=" * 80)
            
            file_analyses = await self.firestore_service.load_all_file_analyses(session_id)
            
            if not file_analyses:
                logger.error("❌ No file analyses found")
                return None
            
            logger.info(f"✅ Loaded {len(file_analyses)} file analyses")
            
            # Log what we have
            for idx, analysis in enumerate(file_analyses):
                filename = analysis.get("filename", "unknown")
                file_type = analysis.get("analysis_data", {}).get("file_type")
                has_geometric_ref = "geometric_ref" in analysis
                logger.info(f"   File #{idx + 1}: {filename} (type: {file_type}, has_geometric: {has_geometric_ref})")
            
            # STEP 3: Try GEOMETRIC extraction (from GCS)
            logger.info("=" * 80)
            logger.info("STEP 3: Attempting GEOMETRIC extraction (from GCS)...")
            logger.info("=" * 80)
            
            geometric_result = await self._try_geometric_extraction(session_id, file_analyses)
            
            if geometric_result:
                logger.info("✅ SUCCESS: Geometric extraction complete")
                logger.info(f"   Source: DXF closed polylines from GCS")
                logger.info(f"   Rooms found: {len(geometric_result.get('rooms', []))}")
                logger.info(f"   Total area: {geometric_result.get('total_area', 0)} m²")
                logger.info(f"   Precision: HIGH (from actual geometry)")
                
                return {
                    "session_id": session_id,
                    "spatial_data": geometric_result,
                    "source_type": "geometric",
                    "precision": "high",
                    "project_data": session_doc.get("project_data_summary", {})
                }
            
            # STEP 4: Fall back to TEXT-BASED extraction (from Firestore summaries)
            logger.warning("⚠️ Geometric extraction failed - no room boundaries found")
            logger.info("=" * 80)
            logger.info("STEP 4: Falling back to TEXT-BASED extraction...")
            logger.info("=" * 80)
            
            text_result = await self._try_text_based_extraction(session_id, file_analyses)
            
            if text_result and text_result.get("rooms"):
                logger.info("✅ SUCCESS: Text-based extraction complete")
                logger.info(f"   Source: analysis_summary from Firestore")
                logger.info(f"   Rooms found: {len(text_result['rooms'])}")
                logger.info(f"   Total area: {text_result['total_area']:.2f} m²")
                logger.warning("   ⚠️ Precision: APPROXIMATE (generated from text)")
                
                return {
                    "session_id": session_id,
                    "spatial_data": text_result,
                    "source_type": "text",
                    "precision": "approximate",
                    "warning": "Generated from summary, not geometric floor plan",
                    "missing_data": text_result.get("missing_data", []),
                    "romanian_context": self._extract_romanian_context(text_result),
                    "project_data": session_doc.get("project_data_summary", {})
                }
            
            # STEP 5: Both methods failed
            logger.error("=" * 80)
            logger.error("❌ BOTH EXTRACTION METHODS FAILED")
            logger.error("=" * 80)
            logger.error("   No geometric data (closed polylines in GCS)")
            logger.error("   No text descriptions (room_breakdown in summaries)")
            logger.error("   Please provide:")
            logger.error("   - Floor plan view with room boundaries, OR")
            logger.error("   - Detailed room schedule with areas and dimensions")
            
            return None
            
        except Exception as e:
            logger.error("=" * 80)
            logger.error(f"❌ EXCEPTION in _load_session_data: {e}", exc_info=True)
            logger.error("=" * 80)
            return None
                

    async def _try_geometric_extraction(self, session_id: str, file_analyses: List[Dict]) -> Optional[Dict[str, Any]]:
        """
        Try to extract spatial data from geometric DXF analysis (closed polylines)
        
        Returns:
            Spatial data dict or None if extraction fails
        """
        logger.info("   Searching for geometric data (closed polylines)...")
        
        for idx, analysis in enumerate(file_analyses):
            logger.info(f"   Checking file #{idx + 1}...")
            
            analysis_data = analysis.get("analysis_data", {})
            file_id = analysis.get("file_id") or analysis.get("filename")
            # If geometric reference exists (split storage), attempt to load full geometric data
            geometric_ref = analysis.get("geometric_ref")

            if not analysis_data or (not analysis_data.get("dxf_analysis_result") and geometric_ref):
                if geometric_ref:
                    logger.info(f"   Found geometric_ref for file {file_id}, attempting to load from GCS")
                    geometric_loaded = await self._load_geometric_data_from_gcs(session_id, file_id, geometric_ref)
                    if geometric_loaded:
                        # place geometric data into analysis_data for downstream logic
                        analysis_data = analysis_data or {}
                        analysis_data["dxf_analysis"] = geometric_loaded
                        # also set a synthetic dxf_analysis_result wrapper for backward compatibility
                        analysis_data["dxf_analysis_result"] = {"dxf_analysis": geometric_loaded}
            
            # Try multiple paths to find dxf_analysis
            dxf_data = None
            source = None
            
            # Path 1: dxf_analysis_result.dxf_analysis
            if "dxf_analysis_result" in analysis_data:
                dxf_result = analysis_data["dxf_analysis_result"]
                if isinstance(dxf_result, dict) and "dxf_analysis" in dxf_result:
                    dxf_data = dxf_result["dxf_analysis"]
                    source = "dxf_analysis_result.dxf_analysis"
            
            # Path 2: Direct dxf_analysis key
            elif "dxf_analysis" in analysis_data:
                dxf_data = analysis_data["dxf_analysis"]
                source = "dxf_analysis"
            
            if not dxf_data or not isinstance(dxf_data, dict):
                logger.info(f"   No DXF data in file #{idx + 1}")
                continue
            
            logger.info(f"   Found DXF data at: {source}")
            
            # Extract room_breakdown
            room_breakdown = dxf_data.get("room_breakdown", [])
            
            if not room_breakdown:
                logger.warning(f"   ⚠️ room_breakdown is empty (no closed polylines)")
                continue
            
            # Transform room_breakdown to expected format
            rooms_transformed = []
            for room in room_breakdown:
                if not isinstance(room, dict):
                    continue
                
                area = room.get("area", 0)
                if area <= 0:
                    continue
                
                transformed_room = {
                    "room_id": room.get("room_id"),
                    "name": room.get("room_type", "unknown"),
                    "name_ro": room.get("romanian_name", "Necunoscut"),
                    "area": area,
                    "dimensions": {
                        "length": round((area ** 0.5), 2),
                        "width": round((area ** 0.5), 2)
                    },
                    "location": room.get("location", [0, 0]),
                    "confidence": room.get("confidence", 0.0)
                }
                rooms_transformed.append(transformed_room)
            
            if not rooms_transformed:
                logger.warning(f"   ⚠️ No valid rooms after transformation")
                continue
            
            # Extract entity inventory
            entity_inventory = dxf_data.get("entity_inventory", {})
            doors_windows = entity_inventory.get("doors_windows", []) if isinstance(entity_inventory, dict) else []
            
            spatial_data = {
                "rooms": rooms_transformed,
                "total_area": dxf_data.get("total_area", 0),
                "walls": [],
                "doors": [dw for dw in doors_windows if isinstance(dw, dict) and dw.get("component_type") == "door"],
                "windows": [dw for dw in doors_windows if isinstance(dw, dict) and dw.get("component_type") == "window"],
                "entity_inventory": entity_inventory
            }
            
            logger.info(f"   ✅ Geometric extraction successful:")
            logger.info(f"      Rooms: {len(spatial_data['rooms'])}")
            logger.info(f"      Area: {spatial_data['total_area']} m²")
            
            return spatial_data
        
        logger.warning("   ❌ No geometric data found in any file")
        return None

    async def _load_geometric_data_from_gcs(
        self,
        session_id: str,
        file_id: str,
        geometric_ref: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Load full geometric data from GCS using reference.
        Supports both reference-based and direct loading.
        
        Args:
            session_id: Session identifier
            file_id: File identifier
            geometric_ref: Reference object from Firestore
        
        Returns:
            Full DXF geometric analysis data, or None if failed
        """
        try:
            # Check if geometric storage is available
            if not geometric_storage or not geometric_storage.is_available():
                logger.error("❌ Geometric storage (GCS) not available")
                return None
            
            # Load from GCS using reference
            geometric_data = await geometric_storage.load_geometric_data_from_ref(geometric_ref)
            
            if geometric_data:
                logger.info(f"✅ Loaded geometric data from GCS for {file_id}")
                return geometric_data
            else:
                logger.error(f"❌ Failed to load geometric data from GCS")
                return None
                
        except Exception as e:
            logger.error(f"❌ Exception loading geometric data from GCS: {e}", exc_info=True)
            return None

    async def _try_text_based_extraction(
        self, 
        session_id: str, 
        file_analyses: List[Dict]
    ) -> Optional[Dict[str, Any]]:
        """
        Extract spatial data from text descriptions (room schedules, annotations).

        Week 3 Enhancement: Now supports PDF table extraction in addition to DXF summaries.

        Priority:
        1. PDF room schedule tables (highest confidence for text-based)
        2. DXF room breakdown summaries (from Firestore)

        Returns:
            Spatial data dict or None if extraction fails
        """
        logger.info("   Searching for text-based room descriptions...")
        
        # PRIORITY 1: Check for PDF tables first (better structure than text summaries)
        for idx, analysis in enumerate(file_analyses):
            logger.info(f"   Checking file #{idx + 1} for PDF tables...")
            
            analysis_data = analysis.get("analysis_data", {})
            # Check for PDF with tables
            if "pdf_analysis_result" in analysis_data:
                pdf_result = analysis_data["pdf_analysis_result"]
                tables_extracted = pdf_result.get("tables_extracted", [])
                
                if tables_extracted:
                    logger.info(f"   ✅ Found {len(tables_extracted)} tables in PDF")
                    
                    # Parse tables for room data
                    table_result = self._parse_room_tables(tables_extracted)
                    
                    if table_result and table_result.get("rooms"):
                        logger.info(f"   ✅ Successfully extracted rooms from PDF tables")
                        return table_result
                    else:
                        logger.info(f"   ⚠️ Tables found but no room data extracted")

        # PRIORITY 2: Fall back to DXF summaries
        logger.info("   No PDF tables with room data, checking DXF summaries...")
        
        for idx, analysis in enumerate(file_analyses):
            logger.info(f"   Checking file #{idx + 1} for DXF text data...")
            
            analysis_data = analysis.get("analysis_data", {})
            analysis_summary = analysis_data.get("analysis_summary", {})
            
            # Check if this is a DXF with room breakdown info
            if analysis_summary.get("type") != "dxf":
                logger.info(f"   File #{idx + 1} is not DXF, skipping")
                continue
            
            room_breakdown = analysis_summary.get("room_breakdown", [])
            
            if not room_breakdown:
                logger.info(f"   No room_breakdown in analysis_summary")
                continue
            
            logger.info(f"   ✅ Found room_breakdown with {len(room_breakdown)} rooms")
            
            # Transform room_breakdown from summary
            rooms_transformed = []
            total_area = 0
            
            for room in room_breakdown:
                if not isinstance(room, dict):
                    continue
                
                room_name = room.get("room_name", "Unknown")
                room_area = room.get("area", 0)
                
                if room_area <= 0:
                    continue
                
                dimensions = room.get("dimensions", {})
                length = dimensions.get("length", 0) if dimensions else 0
                width = dimensions.get("width", 0) if dimensions else 0
                
                # If dimensions missing, estimate from area
                if length == 0 or width == 0:
                    length = round((room_area ** 0.5), 2)
                    width = round((room_area ** 0.5), 2)
                
                transformed_room = {
                    "room_id": f"room_{len(rooms_transformed) + 1}",
                    "name": room_name,
                    "name_ro": room_name,  # Already in Romanian from analysis
                    "area": room_area,
                    "dimensions": {
                        "length": length,
                        "width": width
                    },
                    "location": [0, 0],  # No geometric location available
                    "confidence": 0.7,  # Medium confidence for text-based
                    "source": "text_description"
                }
                
                rooms_transformed.append(transformed_room)
                total_area += room_area
            
            if not rooms_transformed:
                logger.warning(f"   ⚠️ No valid rooms after parsing text")
                continue
            
            # Check for missing data
            missing_data = []
            if not any(room.get("dimensions", {}).get("length", 0) > room["area"] ** 0.5 * 1.1 
                       for room in room_breakdown):
                missing_data.append("Exact room dimensions (using estimated square approximations)")
            
            missing_data.append("Wall thicknesses and exact positions")
            missing_data.append("Precise door and window locations")
            missing_data.append("Room spatial relationships and adjacency")
            
            spatial_data = {
                "rooms": rooms_transformed,
                "total_area": analysis_summary.get("total_area", total_area),
                "walls": [],
                "doors": [],
                "windows": [],
                "missing_data": missing_data,
                "note": "Generated from room schedule/description, not geometric boundaries"
            }
            
            logger.info(f"   ✅ Text-based extraction successful:")
            logger.info(f"      Rooms: {len(spatial_data['rooms'])}")
            logger.info(f"      Area: {spatial_data['total_area']} m²")
            logger.info(f"      Missing data items: {len(missing_data)}")
            
            return spatial_data
        
        logger.warning("   ❌ No text-based room data found in any file (no PDF tables or DXF summaries)")
        return None

    def _parse_room_tables(self, tables: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Parse room schedule tables from PDF into spatial data format.
        
        Handles multiple table formats with Romanian and English headers.
        Extracts: room name, area, dimensions (length/width if available)
        
        Args:
            tables: List of table dictionaries from PDFAnalysisResult.tables_extracted
        
        Returns:
            Spatial data dict compatible with drawing generation, or None if parsing fails
        """
        logger.info("   📊 Parsing room schedule tables from PDF...")
        
        all_rooms = []
        total_area = 0
        
        for table in tables:
            # Only process room schedule tables
            if not table.get("is_room_schedule", False):
                logger.info(f"      Skipping non-room-schedule table on page {table.get('page')}")
                continue
            
            logger.info(f"      Processing room schedule table from page {table.get('page')}...")
            
            headers = table.get("headers", [])
            rows = table.get("rows", [])
            
            # Find column indices for room data
            room_col = self._find_column_index(headers, ["cameră", "camera", "room", "spațiu", "space", "zonă", "zone"])
            area_col = self._find_column_index(headers, ["suprafață", "suprafata", "area", "arie", "mp", "m²", "m2"])
            length_col = self._find_column_index(headers, ["lungime", "length", "l"])
            width_col = self._find_column_index(headers, ["lățime", "latime", "width", "w", "lăţime"])
            
            if room_col is None or area_col is None:
                logger.warning(f"      ⚠️ Could not find room/area columns in table headers: {headers}")
                continue
            
            logger.info(f"      Found columns: room={room_col}, area={area_col}, length={length_col}, width={width_col}")
            
            # Parse each row
            for row_idx, row in enumerate(rows):
                if len(row) <= max(room_col, area_col):
                    continue
                
                room_name_raw = row[room_col].strip()
                area_raw = row[area_col].strip()
                
                if not room_name_raw or not area_raw:
                    continue
                
                # Parse area (handle various formats: "25.5", "25,5", "25.5 mp", "25.5m²")
                try:
                    area_clean = area_raw.replace(",", ".").replace("mp", "").replace("m²", "").replace("m2", "").strip()
                    area = float(area_clean)
                except (ValueError, AttributeError):
                    logger.warning(f"      ⚠️ Could not parse area: '{area_raw}'")
                    continue
                
                if area <= 0:
                    continue
                
                # Parse dimensions if available
                length = 0
                width = 0
                
                if length_col is not None and len(row) > length_col:
                    try:
                        length_raw = row[length_col].strip().replace(",", ".").replace("m", "").strip()
                        if length_raw:
                            length = float(length_raw)
                    except (ValueError, AttributeError):
                        pass
                
                if width_col is not None and len(row) > width_col:
                    try:
                        width_raw = row[width_col].strip().replace(",", ".").replace("m", "").strip()
                        if width_raw:
                            width = float(width_raw)
                    except (ValueError, AttributeError):
                        pass
                
                # If dimensions missing, estimate from area
                if length == 0 or width == 0:
                    length = round((area ** 0.5), 2)
                    width = round((area ** 0.5), 2)
                
                # Normalize room name (Romanian → English)
                room_name_normalized = self._normalize_room_name(room_name_raw)
                
                room_data = {
                    "room_id": f"pdf_room_{len(all_rooms) + 1}",
                    "name": room_name_normalized,
                    "name_ro": room_name_raw,
                    "area": round(area, 2),
                    "dimensions": {
                        "length": round(length, 2),
                        "width": round(width, 2)
                    },
                    "location": [0, 0],  # No spatial location from table
                    "confidence": 0.8,  # Higher confidence than text description
                    "source": "pdf_table"
                }
                
                all_rooms.append(room_data)
                total_area += area
                
                logger.info(f"         ✅ Parsed: {room_name_raw} ({area:.1f} m²)")
            
            logger.info(f"      Extracted {len(all_rooms)} rooms from table")
        
        if not all_rooms:
            logger.warning("   ❌ No rooms extracted from PDF tables")
            return None
        
        missing_data = [
            "Wall thicknesses and exact positions",
            "Precise door and window locations",
            "Room spatial relationships and adjacency"
        ]
        
        spatial_data = {
            "rooms": all_rooms,
            "total_area": round(total_area, 2),
            "walls": [],
            "doors": [],
            "windows": [],
            "missing_data": missing_data,
            "note": "Generated from PDF room schedule table"
        }
        
        logger.info(f"   ✅ PDF table parsing successful:")
        logger.info(f"      Rooms: {len(all_rooms)}")
        logger.info(f"      Total Area: {total_area:.2f} m²")
        
        return spatial_data

    def _find_column_index(self, headers: List[str], keywords: List[str]) -> Optional[int]:
        """
        Find column index by matching keywords (case-insensitive, multi-language).
        
        Args:
            headers: List of header strings (already lowercased)
            keywords: List of keywords to search for
        
        Returns:
            Column index (0-based) or None if not found
        """
        for idx, header in enumerate(headers):
            if not header:
                continue
            
            header_lower = header.lower().strip()
            
            for keyword in keywords:
                if keyword.lower() in header_lower:
                    return idx
        
        return None

    def _normalize_room_name(self, room_name_ro: str) -> str:
        """
        Convert Romanian room names to English standard names.
        
        This ensures consistency with the rest of the system which uses
        English room type identifiers internally.
        
        Args:
            room_name_ro: Romanian room name from table
        
        Returns:
            Normalized English room name
        """
        room_name_lower = room_name_ro.lower().strip()
        
        # Romanian → English mapping
        room_mapping = {
            # Kitchen variants
            "bucătărie": "kitchen",
            "bucatarie": "kitchen",
            "buc": "kitchen",
            
            # Living room variants
            "living": "living_room",
            "salon": "living_room",
            "sufragerie": "living_room",
            "camera de zi": "living_room",
            
            # Bedroom variants
            "dormitor": "bedroom",
            "camera": "bedroom",
            "camera de dormit": "bedroom",
            "dorm": "bedroom",
            
            # Bathroom variants
            "baie": "bathroom",
            "bai": "bathroom",
            "toaleta": "bathroom",
            "wc": "bathroom",
            "grup sanitar": "bathroom",
            
            # Hallway variants
            "hol": "hallway",
            "coridor": "hallway",
            "antreu": "hallway",
            "vestibul": "hallway",
            
            # Storage variants
            "debara": "storage",
            "cămară": "storage",
            "camara": "storage",
            "dulap": "storage",
            
            # Office variants
            "birou": "office",
            "cabinet": "office",
            "camera de lucru": "office",
            
            # Balcony variants
            "balcon": "balcony",
            "terasa": "balcony",
            "terasă": "balcony",
            "loggie": "balcony",
            
            # Other
            "garaj": "garage",
            "parcare": "garage",
            "magazie": "storage",
            "spațiu tehnic": "technical_room",
            "spatiu tehnic": "technical_room"
        }
        
        # Check for exact match first
        if room_name_lower in room_mapping:
            return room_mapping[room_name_lower]
        
        # Check for partial match (if room name contains keyword)
        for ro_name, en_name in room_mapping.items():
            if ro_name in room_name_lower:
                return en_name
        
        # If no match found, return cleaned original name
        return room_name_lower.replace(" ", "_")

    def _extract_romanian_context(self, spatial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Romanian room names and context"""
        room_names = [r.get("name_ro") for r in spatial_data.get("rooms", []) if r.get("name_ro")]
        
        return {
            "room_names": room_names,
            "primary_language": "ro"
        }
    
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
            exec_globals = {
                "output_path": output_path,
                "__builtins__": __builtins__,
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
                "code_used": code_used[:2000],  # Truncate for storage
                "code_length": len(code_used),
                "status": "completed",
                "spatial_data_summary": {
                    "num_rooms": len(spatial_data.get("rooms", [])),
                    "total_area": spatial_data.get("total_area", 0),
                    "room_names": [r.get("name_ro") for r in spatial_data.get("rooms", [])[:5]]
                },
                "validation_warnings": validation_warnings[:10],
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
            "phase": "1-mvp-hybrid",
            "supported_drawing_types": ["field_verification"],
            "max_retry_attempts": self.max_retry_attempts,
            "storage": self.storage_service.get_storage_info(),
            "extraction_methods": ["geometric", "text_based"],
            "features": {
                "adaptive_rag": True,
                "error_feedback_loop": True,
                "code_validation": True,
                "romanian_standards": True,
                "hybrid_extraction": True,
                "graceful_degradation": True
            }
        }