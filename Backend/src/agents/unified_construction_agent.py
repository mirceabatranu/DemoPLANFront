# -*- coding: utf-8 -*-
# src/agents/unified_construction_agent.py
"""
DEMOPLAN Unified Construction Agent
Single agent handling complete project lifecycle from DXF analysis to offer generation
Combines technical analysis, Romanian expertise, and ML-enhanced intelligence
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, field, is_dataclass, asdict
from enum import Enum
import uuid # For generating unique file IDs
from typing import Dict, List, Any, Optional, Tuple
import copy # For deep copying objects

# Core processors
logger = logging.getLogger("demoplan.agents.unified_construction_agent")
from src.processors.dxf_analyzer import DXFAnalyzer
from src.processors.romanian_processor import RomanianProcessor
from src.models.ocr_models import OCRResult
from src.intelligence.gap_analyzer import (
    GapAnalyzer,
    GapAnalysisResult,
    GapPriority,
    DataCategory
)
from src.intelligence.cross_reference_engine import (
    CrossReferenceEngine,
    CrossReferenceResult,
    ConflictSeverity,
    ConflictType
)
# Add to imports at the top:
from src.services.response_builder import (
    ResponseBuilder,
    ResponseType
)


# Enhanced Intelligence components with Phase 1 compatibility
try:
    from src.intelligence import (
        intelligence_manager,
        LearningEngine,
        PatternMatcher,
        HistoricalAnalyzer
    )
    INTELLIGENCE_AVAILABLE = True
    logger.info("✅ Intelligence components loaded (Phase 1 mode)")
except ImportError as e:
    logger.warning(f"⚠️ Intelligence module import failed: {e}")
    INTELLIGENCE_AVAILABLE = False
    intelligence_manager = None

# Conditional imports to avoid circular dependencies
try:
    from src.services.session_manager import get_session_manager
except ImportError:
    get_session_manager = None

# Services
from src.services.llm_service import safe_construction_call
from src.services.firestore_service import FirestoreService
from src.services.ocr_storage_service import OCRStorageService
from src.services.file_storage_service import file_storage_service
from src.services.storage_service import geometric_storage
from src.services.image_analyzer import ImageAnalyzerService, ImageAnalysisResult

class AgentMode(Enum):
    """Operating modes for the unified agent"""
    TECHNICAL_ANALYSIS = "technical_analysis"
    CONVERSATION = "conversation"
    OFFER_GENERATION = "offer_generation"

class ConfidenceLevel(Enum):
    """Confidence levels for decision making"""
    LOW = "low"           # < 40%
    MEDIUM = "medium"     # 40-75%
    HIGH = "high"         # 75-90%
    EXCELLENT = "excellent" # > 90%

@dataclass
class AgentContext:
    """Context maintained throughout agent lifecycle"""
    session_id: str
    current_mode: AgentMode
    confidence_score: float = 0.0
    project_data: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    analysis_results: Dict[str, Any] = field(default_factory=dict)
    missing_data: List[str] = field(default_factory=list)
    romanian_context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class UnifiedResponse:
    """Unified agent response structure"""
    content: str
    confidence: float
    mode: AgentMode
    next_questions: List[str] = field(default_factory=list)
    can_generate_offer: bool = False
    analysis_summary: Optional[Dict[str, Any]] = None
    processing_time_ms: float = 0.0

class UnifiedConstructionAgent:
    """
    Single unified agent for complete Romanian construction consultation
    Handles DXF analysis, conversation, ML enhancement, and offer generation
    """

    def __init__(self):
        # Core processors
        self.dxf_analyzer = DXFAnalyzer()
        self.romanian_processor = RomanianProcessor()

        # ML Intelligence components
        self.learning_engine = None
        self.pattern_matcher = None
        self.historical_analyzer = None

        # Services
        self.session_manager = None
        self.firestore_service = FirestoreService()

        # ✅ NEW: Initialize OCR Storage Service
        self.ocr_storage = OCRStorageService(self.firestore_service)
        
        # Gap Analysis Engine
        self.gap_analyzer = GapAnalyzer()
        
        # ✅ ADD THIS - Cross-Reference Engine
        self.cross_reference = CrossReferenceEngine()
        
        # ✅ ADD THIS - Response Builder
        self.response_builder = ResponseBuilder()
        
        logger.info("✅ Cross-Reference Engine initialized")
        logger.info("✅ Response Builder initialized")


        # Configuration
        self.confidence_thresholds = {
            ConfidenceLevel.LOW: 0.40,
            ConfidenceLevel.MEDIUM: 0.75,
            ConfidenceLevel.HIGH: 0.90,
            ConfidenceLevel.EXCELLENT: 0.95
        }

        self.offer_generation_threshold = 0.75
        self.ml_enabled = False

        logger.info("🤖 Unified Construction Agent initialized")

    @property
    def llm_service(self):
        """Expose LLM service for drawing agent"""
        from src.services.llm_service import safe_construction_llm_service
        return safe_construction_llm_service

    async def initialize(self, enable_ml: bool = True):
        """Initialize agent with Phase 1 basic intelligence"""
        try:
            # Initialize core services
            try:
                await self.firestore_service.initialize()
                if get_session_manager:
                    self.session_manager = get_session_manager()
            except Exception as e:
                logger.warning(f"⚠️ Core services initialization issue: {e}")

            # Phase 1: Initialize basic intelligence components
            if INTELLIGENCE_AVAILABLE and intelligence_manager:
                try:
                    await intelligence_manager.initialize(enable_ml=False)  # Phase 1: basic mode

                    self.learning_engine = intelligence_manager.learning_engine
                    self.pattern_matcher = intelligence_manager.pattern_matcher
                    self.historical_analyzer = intelligence_manager.historical_analyzer

                    self.ml_enabled = True  # Basic ML features enabled
                    logger.info("✅ Phase 1 intelligence components initialized")

                except Exception as e:
                    logger.warning(f"⚠️ Intelligence initialization failed: {e}")
                    self.ml_enabled = False
            else:
                logger.warning("⚠️ Intelligence components not available")
                self.ml_enabled = False

            logger.info(f"✅ Phase 1 Unified Agent ready (Basic ML: {'enabled' if self.ml_enabled else 'disabled'})")

        except Exception as e:
            logger.error(f"❌ Agent initialization failed: {e}")
            self.ml_enabled = False
            logger.warning("⚠️ Agent running in emergency mode")

    async def analyze_project(
        self,
        files: List[Dict[str, Any]],
        user_input: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main entry point for project analysis.
        Handles files and user input, saving each file's analysis to a subcollection.
        """
        start_time = time.time()

        try:
            # Create or get session context
            if session_id:
                context = await self._load_session_context(session_id)
            else:
                context = await self._create_session_context()
                # Create the main session document immediately
                await self.session_manager.create_session(context.session_id, {})


            context.current_mode = AgentMode.TECHNICAL_ANALYSIS

            # Process and save each file analysis to its own document
            all_file_analyses = []
            for file_info in files:
                file_id = self._generate_unique_file_id(file_info.get("filename"))
                # The analysis data for a single file
                single_file_analysis_data = await self._process_single_file(file_info, context)

                # Add file metadata to the analysis document
                analysis_doc = {
                    "file_id": file_id,
                    "filename": file_info.get("filename"),
                    "content_type": file_info.get("content_type"),
                    "size": file_info.get("size"),
                    "uploaded_at": datetime.now(timezone.utc),
                    "analysis_data": single_file_analysis_data,
                    "processing_status": "completed"
                }

                # ✅ NEW: Use geometric split method for saving
                # This will automatically split DXF geometric data to GCS
                save_success = await self.firestore_service.save_file_analysis_with_geometric_split(
                    session_id=context.session_id,
                    file_id=file_id,
                    analysis_data=self._flatten_for_firestore(analysis_doc)
                )
                
                if save_success:
                    logger.info(f"✅ Saved file analysis for {file_id} (with geometric split if DXF)")
                else:
                    logger.error(f"❌ Failed to save file analysis for {file_id}")
                
                all_file_analyses.append(analysis_doc)

            # Consolidate in-memory analysis from all files for the current response
            file_analysis = self._consolidate_file_analyses(all_file_analyses)
            context.analysis_results["file_analysis"] = file_analysis

            # Analyze user requirements from text input
            requirements_analysis = await self._analyze_requirements(user_input, context)

            # Apply ML intelligence if available
            ml_insights = None
            if self.ml_enabled:
                ml_insights = await self._apply_ml_intelligence(context)

            # Consolidate all analyses (files, text, ML)
            consolidated_analysis = await self._consolidate_analysis(
                file_analysis, requirements_analysis, ml_insights, context
            )
            
            # Generate response based on the consolidated data
            response = await self._generate_unified_response(consolidated_analysis, context)

            # Update the main session document with metadata only
            await self._update_session_context(context)

            processing_time = (time.time() - start_time) * 1000

            # Build the API response
            response_dict = {
                "session_id": context.session_id,
                "response": response.content,
                "confidence": response.confidence,
                "mode": response.mode.value,
                "can_generate_offer": response.can_generate_offer,
                "next_questions": response.next_questions,
                "analysis_summary": response.analysis_summary,
                "processing_time_ms": processing_time,
                "ml_enhanced": self.ml_enabled and ml_insights is not None
            }

            return self._flatten_analysis_for_api(response_dict)

        except Exception as e:
            logger.error(f"❌ Project analysis failed: {e}", exc_info=True)
            error_response_content = self._build_error_response(e, context="Project analysis")
            return {
                "session_id": session_id,
                "response": error_response_content,
                "confidence": 0.0,
                "mode": AgentMode.TECHNICAL_ANALYSIS.value,
                "can_generate_offer": False,
                "error": str(e)
            }

    async def continue_conversation(
        self,
        session_id: str,
        user_input: str
    ) -> Dict[str, Any]:
        """
        Continue conversation, saving each message to the 'messages' subcollection.
        """
        start_time = time.time()

        try:
            # Load session context (which now loads from subcollections)
            logger.info(f"🔄 Loading session context for {session_id}")
            context = await self._load_session_context(session_id)
            context.current_mode = AgentMode.CONVERSATION

            # Save user message to subcollection immediately
            user_message_doc = {
                "type": "user",
                "content": user_input,
                "timestamp": datetime.now(timezone.utc)
            }
            await self.firestore_service.save_message_to_subcollection(session_id, user_message_doc)
            context.conversation_history.append(user_message_doc) # Also update in-memory context

            # Check for offer generation request
            if self._is_offer_request(user_input):
                logger.info("🎯 Detected offer generation request")
                return await self._handle_offer_request(context)

            # Process user message
            logger.info("🔍 Analyzing user message")
            message_analysis = await self._analyze_user_message(user_input, context)

            # Apply ML insights for conversation
            if self.ml_enabled:
                logger.info("🤖 Applying ML insights")
                ml_insights = await self._apply_conversation_ml(user_input, context)
                if ml_insights:
                    message_analysis.update(ml_insights)

            # Generate response
            logger.info("💬 Generating conversation response")
            response = await self._generate_conversation_response(message_analysis, context)

            # Save assistant response to subcollection immediately
            assistant_message_doc = {
                "type": "assistant",
                "content": response.content,
                "timestamp": datetime.now(timezone.utc)
            }
            await self.firestore_service.save_message_to_subcollection(session_id, assistant_message_doc)
            context.conversation_history.append(assistant_message_doc)

            # Update main session metadata
            logger.info("💾 Updating session context metadata")
            await self._update_session_context(context)

            processing_time = (time.time() - start_time) * 1000
            logger.info(f"✅ Conversation processed in {processing_time:.0f}ms")

            response_dict = {
                "session_id": context.session_id,
                "response": response.content,
                "confidence": response.confidence,
                "mode": response.mode.value,
                "can_generate_offer": response.can_generate_offer,
                "next_questions": response.next_questions,
                "processing_time_ms": processing_time,
                "analysis_summary": response.analysis_summary,
            }
            return self._flatten_analysis_for_api(response_dict)

        except Exception as e:
            logger.error(f"❌ Conversation continuation failed: {e}", exc_info=True)
            error_response_content = self._build_error_response(e, context="Conversation continuation")
            return {
                "session_id": session_id,
                "response": error_response_content,
                "confidence": 0.0,
                "error": str(e)
            }
            
    # =========================================================================
    # NEW AND REFACTORED PRIVATE METHODS FOR SUBCOLLECTION ARCHITECTURE
    # =========================================================================
    
    def _generate_unique_file_id(self, filename: str) -> str:
        """Generate a unique ID for a file analysis document."""
        safe_filename = "".join(c for c in filename if c.isalnum() or c in ('-', '_')).rstrip()
        timestamp = int(time.time() * 1000)
        return f"{timestamp}_{safe_filename[:50]}_{uuid.uuid4().hex[:8]}"

    def _consolidate_file_analyses(self, file_analyses_docs: List[Dict]) -> Dict[str, Any]:
        """
        Merge multiple file analysis documents from a subcollection into a
        single analysis_results structure for in-memory agent use.

        NOTE: For chat/conversational responses, this only uses analysis_summary
        (no full geometric data). Full geometric data is loaded separately when
        needed (e.g., for drawing generation).
        """
        consolidated = {
            "files_processed": len(file_analyses_docs),
            "dxf_analysis": None,
            "pdf_analysis": None,
            "txt_analysis": None,
            "other_files": [],
            "file_references": [],
            "confidence_contribution": 0.0
        }

        for doc in file_analyses_docs:
            analysis_data = doc.get("analysis_data", {})
            file_type = analysis_data.get("file_type")

            if file_type == "dxf":
                # For DXF, we keep the summary - full geometric is in GCS
                consolidated["dxf_analysis"] = analysis_data.get("dxf_analysis_result")
            elif file_type == "pdf":
                consolidated["pdf_analysis"] = analysis_data.get("pdf_analysis_result")
            elif file_type == "txt":
                consolidated["txt_analysis"] = analysis_data.get("txt_analysis_result")
            else:
                consolidated["other_files"].append(doc.get("filename", "unknown"))

            consolidated["confidence_contribution"] += analysis_data.get("confidence_contribution", 0.0)
            
            # Rebuild file_references with analysis_summary
            file_ref = {
                "filename": doc.get("filename"),
                "content_type": doc.get("content_type"),
                "size": doc.get("size"),
                "uploaded_at": doc.get("uploaded_at"),
                "file_id": doc.get("file_id"),
                "analysis_summary": analysis_data.get("analysis_summary", {})
            }
            consolidated["file_references"].append(file_ref)
            
        return consolidated
        
    async def _process_single_file(self, file_info: Dict[str, Any], context: AgentContext) -> Dict[str, Any]:
        """
        Process a single uploaded file and return its analysis data.
        NOW with geometric data split to GCS for DXF files.
        """
        single_file_result = {
            "file_type": None,
            "confidence_contribution": 0.0,
            "analysis_summary": {}
        }
        filename = file_info.get("filename", "").lower()
        content = file_info.get("content")

        if not content:
            logger.warning(f"⚠️ File {filename} has NO CONTENT, skipping")
            return single_file_result

        try:
            # ✅ NEW: Image handling for common image MIME types
            content_type = file_info.get("content_type", "") or ""
            if content_type.startswith("image/") or filename.endswith(('.png', '.jpg', '.jpeg', '.webp', '.tiff')):
                # Delegate to image processing method
                image_result = await self._process_image_file(file_info)
                # Merge image_result into single_file_result structure
                single_file_result.update({
                    "file_type": "image",
                    "confidence_contribution": image_result.get("analysis_summary", {}).get("confidence", 0.0) if isinstance(image_result.get("analysis_summary", {}), dict) else 0.0,
                    "image_analysis_result": image_result.get("image_analysis"),
                    "analysis_summary": image_result.get("analysis_summary", {})
                })
                return single_file_result

            from src.processors.dxf_analyzer import UnifiedDocumentProcessor
            document_processor = UnifiedDocumentProcessor()
            document_result = document_processor.process_document(filename, content)

            if not document_result:
                raise ValueError("Document processor returned None")

            single_file_result["file_type"] = document_result.document_type
            single_file_result["confidence_contribution"] = document_result.combined_confidence

            # Route and store the specific analysis result
            if document_result.document_type == "dxf" and document_result.dxf_analysis:
                analysis_dict = self._analysis_to_dict(document_result.dxf_analysis)
                single_file_result["dxf_analysis_result"] = analysis_dict
                dxf_data = analysis_dict.get("dxf_analysis", {})
                
                # ✅ NEW: Create analysis_summary with room_breakdown for text-based extraction
                room_breakdown_raw = dxf_data.get("room_breakdown", [])
                room_breakdown_transformed = self._transform_room_breakdown_for_summary(room_breakdown_raw)

                single_file_result["analysis_summary"] = {
                    "type": "dxf",
                    "total_area": dxf_data.get("total_area", 0),
                    "total_rooms": dxf_data.get("total_rooms", 0),
                    "room_breakdown": room_breakdown_transformed,  # ✅ NOW INCLUDED
                    "has_dimensions": dxf_data.get("has_dimensions", False),
                    "wall_types": dxf_data.get("wall_types", [])[:3]  # Sample for summary
                }
                
            elif document_result.document_type == "pdf" and document_result.pdf_analysis:
                analysis_dict = self._analysis_to_dict(document_result.pdf_analysis)
                single_file_result["pdf_analysis_result"] = analysis_dict
                single_file_result["analysis_summary"] = {
                    "type": "pdf",
                    "page_count": analysis_dict.get('page_count', 0),
                    "ocr_used": analysis_dict.get('ocr_used', False),
                    "construction_specs": analysis_dict.get('construction_specs', [])[:5],
                    "material_references": analysis_dict.get('material_references', [])[:5]
                }
                
            elif document_result.document_type == "txt" and document_result.txt_analysis:
                analysis_dict = self._analysis_to_dict(document_result.txt_analysis)
                single_file_result["txt_analysis_result"] = analysis_dict
                single_file_result["analysis_summary"] = {
                    "type": "txt",
                    "requirements": analysis_dict.get('requirements', [])[:5],
                    "client_preferences": analysis_dict.get('client_preferences', {})
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to process single file {filename}: {e}", exc_info=True)
            single_file_result["error"] = str(e)

        return single_file_result

    def _transform_room_breakdown_for_summary(self, room_breakdown_raw: List[Dict]) -> List[Dict]:
        """
        Transform DXF room_breakdown from geometric format to text-friendly summary format.
        
        Converts:
            {room_id, room_type, romanian_name, area, location, confidence, associated_texts}
        To:
            {room_name, area, dimensions: {length, width}}
        
        This is what drawing agent expects for text-based extraction.
        """
        transformed = []
        
        for room in room_breakdown_raw:
            if not isinstance(room, dict):
                continue
            
            area = room.get("area", 0)
            if area <= 0:
                continue
            
            # Use romanian_name if available, fallback to room_type
            room_name = room.get("romanian_name") or room.get("room_type", "Unknown")
            
            # Calculate estimated dimensions from area (square approximation)
            estimated_side = round((area ** 0.5), 2)
            
            transformed.append({
                "room_name": room_name,
                "area": area,
                "dimensions": {
                    "length": estimated_side,
                    "width": estimated_side
                }
            })
        
        return transformed

    async def _process_image_file(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process image files using Google Vision AI

        Analyzes images to extract room labels and spatial information.

        Args:
            file_info: Dictionary containing file data
                - filename: Original filename
                - content: Raw image bytes
                - content_type: MIME type (image/jpeg, image/png)
                - file_id: (Optional) Unique identifier - will be generated if not present

        Returns:
            Dictionary with image analysis results (fully flattened for Firestore)
        """
        # ✅ Extract filename FIRST before any operations that might fail
        filename = file_info.get('filename', 'unknown_image')

        try:
            # ✅ Generate file_id if not present in file_info
            file_id = file_info.get('file_id')
            if not file_id:
                import uuid
                file_id = str(uuid.uuid4())
                file_info['file_id'] = file_id
                logger.info(f"📋 Generated file_id for image: {file_id}")
            
            content = file_info.get('content')
            if not content:
                logger.error(f"❌ No content provided for image: {filename}")
                return {
                    'file_id': file_id,
                    'filename': filename,
                    'status': 'failed',
                    'error': 'No image content provided',
                    'analysis_summary': {'confidence': 0.0}
                }
            
            content_type = file_info.get('content_type', 'image/png')
            
            logger.info(f"🖼️  Processing image file: {filename}")
            logger.info(f"   File ID: {file_id}")
            logger.info(f"   Content type: {content_type}")
            logger.info(f"   Size: {len(content) / 1024:.1f} KB")
            
            # Initialize image analyzer
            from src.services.image_analyzer import ImageAnalyzerService
            analyzer = ImageAnalyzerService()
            
            # Analyze floor plan image
            analysis_result = await analyzer.analyze_floor_plan(
                image_content=content,
                filename=filename,
                min_confidence=0.5
            )
            
            # ✅ FIX: Convert to dictionary AND flatten for Firestore
            # The to_dict() method should handle the initial conversion
            image_analysis_dict = analyzer.to_dict(analysis_result)
            
            # ✅ CRITICAL: Flatten the entire image_analysis_dict to ensure NO nested objects
            # This converts any remaining dataclass objects, enums, etc. to primitives
            flattened_image_analysis = self._flatten_for_firestore(image_analysis_dict)
            
            # Create analysis summary (also flatten to be safe)
            analysis_summary = {
                'type': 'image',
                'room_labels_found': len(analysis_result.room_labels),
                'total_text_annotations': len(analysis_result.all_text_annotations),
                'objects_detected': len(analysis_result.detected_objects),
                'confidence': float(analysis_result.confidence),  # Ensure it's a Python float
                'quality_score': float(analysis_result.quality_score),
                'warnings': list(analysis_result.warnings),  # Ensure it's a list
                'image_dimensions': {
                    'width': int(analysis_result.image_dimensions.width),
                    'height': int(analysis_result.image_dimensions.height),
                    'aspect_ratio': float(analysis_result.image_dimensions.aspect_ratio)
                },
                'processing_time_ms': float(analysis_result.processing_time_ms),
                'cost_estimate': float(analysis_result.cost_estimate)
            }
            
            logger.info(f"✅ Image analysis complete:")
            logger.info(f"   Room labels: {len(analysis_result.room_labels)}")
            logger.info(f"   Confidence: {analysis_result.confidence:.1%}")
            logger.info(f"   Quality: {analysis_result.quality_score:.1%}")
            
            # Log warnings if any
            if analysis_result.warnings:
                for warning in analysis_result.warnings:
                    logger.warning(f"   ⚠️ {warning}")
            
            # ✅ Return fully flattened data structure
            result = {
                'file_id': file_id,
                'filename': filename,
                'file_type': 'image',
                'content_type': content_type,
                'size': len(content),
                'status': 'processed',
                'image_analysis': flattened_image_analysis,  # ✅ Flattened
                'analysis_summary': analysis_summary,  # ✅ Already primitive types
                'uploaded_at': file_info.get('uploaded_at'),
                'processed_at': datetime.now(timezone.utc).isoformat()
            }
            
            # ✅ EXTRA SAFETY: Flatten the entire result one more time
            # This ensures absolutely nothing nested remains
            return self._flatten_for_firestore(result)
            
        except KeyError as e:
            logger.error(f"❌ Missing required field in file_info for {filename}: {e}", exc_info=True)
            return {
                'file_id': file_info.get('file_id', 'unknown'),
                'filename': filename,
                'status': 'failed',
                'error': f'Missing required field: {str(e)}',
                'analysis_summary': {'confidence': 0.0}
            }
        except Exception as e:
            logger.error(f"❌ Image processing failed for {filename}: {e}", exc_info=True)
            return {
                'file_id': file_info.get('file_id', 'unknown'),
                'filename': filename,
                'status': 'failed',
                'error': str(e),
                'analysis_summary': {'confidence': 0.0}
            }

    async def generate_offer(self, session_id: str) -> Dict[str, Any]:
        """Generate professional Romanian construction offer"""
        try:
            context = await self._load_session_context(session_id)
            context.current_mode = AgentMode.OFFER_GENERATION

            # Perform gap analysis
            gap_result = self.gap_analyzer.analyze_gaps(
                dxf_data=context.analysis_results.get('dxf_analysis'),
                rfp_data=context.analysis_results.get('rfp_data'),
                user_requirements=context.project_data,
                conversation_context=context.conversation_history
            )
            
            # ✅ ADD THIS - Perform cross-reference validation
            logger.info("🔍 Validating data consistency before offer generation")
            cross_ref_result = self.cross_reference.validate_consistency(
                dxf_data=context.analysis_results.get('dxf_analysis'),
                rfp_data=context.analysis_results.get('rfp_data'),
                user_requirements=context.project_data,
                conversation_context=context.conversation_history
            )
            
            # ✅ CHECK FOR BLOCKING CONFLICTS
            blocking_conflicts = cross_ref_result.get_blocking_conflicts()
            if blocking_conflicts:
                conflict_descriptions = [c.description_ro for c in blocking_conflicts]
                
                error_response = {
                    "success": False,
                    "message": "Conflicte critice în date - necesită rezolvare înainte de generarea ofertei",
                    "blocking_conflicts": conflict_descriptions,
                    "recommendations": cross_ref_result.recommendations
                }
                return self._flatten_analysis_for_api(error_response)
            
            # Check if ready based on gap analysis
            if not gap_result.can_generate_offer:
                blocking_issues = gap_result.get_blocking_issues()
                
                error_response = {
                    "success": False,
                    "message": f"Confidence insuficientă ({gap_result.overall_confidence:.1%}). Necesare informații suplimentare.",
                    "blocking_issues": blocking_issues,
                    "missing_data": context.missing_data,
                    "questions": gap_result.prioritized_questions
                }
                return self._flatten_analysis_for_api(error_response)
            
            # ✅ WARN ABOUT NON-BLOCKING CONFLICTS
            warnings = cross_ref_result.get_critical_warnings()
            warning_note = ""
            if warnings:
                warning_note = f"\n\n⚠️ **Notă**: Detectate {len(warnings)} avertismente în date. Oferta este generată cu date disponibile, dar recomandăm clarificări pentru acuratețe maximă."
            
            # Generate offer with high confidence
            logger.info(f"✅ Generating offer with {gap_result.overall_confidence:.1%} confidence")
            offer_data = await self._generate_professional_offer(context)
            
            # ✅ ADD WARNING NOTE TO OFFER IF NEEDED
            if warning_note:
                offer_data += warning_note

            # Update session with offer
            context.analysis_results["generated_offer"] = offer_data
            await self._update_session_context(context)

            # ✅ FIXED: Build response dict first, then flatten
            offer_response = {
                "success": True,
                "session_id": session_id,
                "offer": offer_data,
                "confidence": context.confidence_score
            }
            return self._flatten_analysis_for_api(offer_response)

        except Exception as e:
            logger.error(f"❌ Offer generation failed: {e}", exc_info=True)
            error_response_content = self._build_error_response(e, context="Offer generation")
            
            # ✅ FIXED: Flatten error response too
            error_response = {
                "success": False,
                "error": error_response_content
            }
            return self._flatten_analysis_for_api(error_response)

    # =========================================================================
    # PRIVATE METHODS - CORE PROCESSING (largely unchanged, but used differently)
    # =========================================================================

    def _analysis_to_dict(self, analysis_obj: Any) -> Dict[str, Any]:
        """Safely converts a processor's analysis result object to a dictionary."""
        if analysis_obj is None:
            return {}
        if isinstance(analysis_obj, dict):
            return analysis_obj  # Already a dict
        if hasattr(analysis_obj, 'to_dict') and callable(getattr(analysis_obj, 'to_dict')):
            return analysis_obj.to_dict()
        if is_dataclass(analysis_obj):
            return asdict(analysis_obj)
        logger.warning(f"Could not convert analysis object of type {type(analysis_obj)} to dict. Storing as string.")
        return {"unsupported_type_data": str(analysis_obj)}

    async def _process_files(self, files: List[Dict[str, Any]], context: AgentContext) -> Dict[str, Any]:
        """Legacy method stub. Logic moved to _process_single_file and analyze_project."""
        logger.warning("DEPRECATION WARNING: _process_files should not be called directly.")
        # This function's logic is now distributed. It's kept for compatibility if called by old code.
        all_analyses = []
        for file_info in files:
            single_analysis = await self._process_single_file(file_info, context)
            doc = { "analysis_data": single_analysis } # simplified doc for consolidation
            all_analyses.append(doc)
        return self._consolidate_file_analyses(all_analyses)


    async def _analyze_requirements(self, user_input: str, context: AgentContext) -> Dict[str, Any]:
        """Analyze user requirements in Romanian"""
        try:
            # Use Romanian processor for initial analysis
            analysis_result = await self.romanian_processor.analyze_construction_text(
                text=user_input,
                analysis_type="requirements_analysis",
                domain="general_construction"
            )

            # Extract key information
            requirements_analysis = {
                "raw_text": user_input,
                "extracted_info": analysis_result,
                "confidence_contribution": 0.0
            }

            # Boost confidence based on detail level
            if len(user_input.strip()) > 50:
                requirements_analysis["confidence_contribution"] += 0.2
            if any(keyword in user_input.lower() for keyword in ["mp", "camere", "bucatarie", "baie", "dormitor"]):
                requirements_analysis["confidence_contribution"] += 0.1

            # Store in context
            context.romanian_context.update({
                "user_requirements": user_input,
                "processed_requirements": analysis_result
            })

            return requirements_analysis

        except Exception as e:
            logger.error(f"❌ Requirements analysis error: {e}")
            return {
                "raw_text": user_input,
                "error": str(e),
                "confidence_contribution": 0.0
            }

    async def _apply_ml_intelligence(self, context: AgentContext) -> Optional[Dict[str, Any]]:
        """Apply ML intelligence components and ensure results are serializable."""
        if not self.ml_enabled:
            return None

        try:
            ml_insights = {
                "pattern_analysis": None,
                "historical_insights": None,
                "learning_adjustments": None,
                "confidence_boost": 0.0
            }

            # Pattern matching
            if self.pattern_matcher:
                pattern_result = await self.pattern_matcher.analyze_project_patterns(
                    context.project_data,
                    context.romanian_context.get("user_requirements")
                )
                # ✅ FIX: Convert result object to a dictionary
                ml_insights["pattern_analysis"] = self._analysis_to_dict(pattern_result)
                ml_insights["confidence_boost"] += pattern_result.confidence * 0.1

            # Historical analysis
            if self.historical_analyzer:
                similar_projects = await self.historical_analyzer.analyze_project_similarity(
                    context.project_data
                )
                # ✅ FIX: Ensure historical data is a list of dicts for safety
                ml_insights["historical_insights"] = [self._analysis_to_dict(p) for p in similar_projects]
                if similar_projects:
                    ml_insights["confidence_boost"] += min(len(similar_projects) * 0.05, 0.15)

            # Learning engine insights
            if self.learning_engine:
                intelligence_analysis = await self.learning_engine.analyze_project_intelligence(
                    context.project_data,
                    context.analysis_results
                )
                # ✅ FIX: Convert result object to a dictionary
                ml_insights["learning_adjustments"] = self._analysis_to_dict(intelligence_analysis)
                ml_insights["confidence_boost"] += 0.1  # Base boost for ML analysis
            return ml_insights
        except Exception as e:
            logger.error(f"❌ ML intelligence error: {e}")
            return None    

    async def _consolidate_analysis(
        self,
        file_analysis: Dict[str, Any],
        requirements_analysis: Dict[str, Any],
        ml_insights: Optional[Dict[str, Any]],
        context: AgentContext
    ) -> Dict[str, Any]:
        """Enhanced with cross-reference validation"""

        # Calculate overall confidence with multi-format support
        confidence_components = [
            file_analysis.get("confidence_contribution", 0.0),
            requirements_analysis.get("confidence_contribution", 0.0)
        ]

        if ml_insights:
            confidence_components.append(ml_insights.get("confidence_boost", 0.0))

        base_confidence = min(sum(confidence_components), 1.0)
        
        # Extract RFP data if available
        rfp_data = None
        if file_analysis.get('pdf_analysis'):
            pdf_data_dict = file_analysis['pdf_analysis']
            if pdf_data_dict.get('rfp_data'):
                # Assuming rfp_data is already a dict after our fix
                rfp_data = pdf_data_dict['rfp_data']
                logger.info(f"📄 RFP data available - confidence: {rfp_data.get('extraction_confidence', 0):.1%}")
                
                # Update context with RFP information
                if rfp_data.get('client_name'):
                    context.project_data['client_name'] = rfp_data['client_name']
                if rfp_data.get('location'):
                    context.project_data['location'] = rfp_data['location']
                if rfp_data.get('work_start_date'):
                    context.project_data['work_start_date'] = rfp_data['work_start_date']
                if rfp_data.get('work_end_date'):
                    context.project_data['work_end_date'] = rfp_data['work_end_date']
                if rfp_data.get('work_duration_days'):
                    context.project_data['work_duration_days'] = rfp_data['work_duration_days']
                if rfp_data.get('guarantee_period_months'):
                    context.project_data['guarantee_months'] = rfp_data['guarantee_period_months']
                if rfp_data.get('performance_bond_percentage'):
                    context.project_data['performance_bond'] = rfp_data['performance_bond_percentage']

        # ✅ ADD THIS BLOCK - Cross-Reference Validation
        logger.info("🔍 Performing cross-reference validation")
        cross_ref_result = self.cross_reference.validate_consistency(
            dxf_data=file_analysis.get('dxf_analysis'),
            rfp_data=rfp_data, # Already a dict or None
            user_inputs=context.project_data,
            conversation_context=context.conversation_history
        )
        
        # Log validation results
        if cross_ref_result.conflicts:
            logger.warning(f"⚠️ Found {len(cross_ref_result.conflicts)} conflicts: "
                           f"{cross_ref_result.error_count} errors, "
                           f"{cross_ref_result.warning_count} warnings")
        else:
            logger.info("✅ No conflicts detected - data is consistent")
        
        # Update context with validated data (data without conflicts)
        context.project_data.update(cross_ref_result.validated_data)
        
        # Adjust confidence score based on consistency
        if hasattr(context, 'confidence_score'):
            original_confidence = context.confidence_score
            context.confidence_score *= cross_ref_result.consistency_score
            
            if context.confidence_score < original_confidence:
                logger.info(f"📉 Confidence adjusted: {original_confidence:.1%} → {context.confidence_score:.1%} "
                            f"(due to {len(cross_ref_result.conflicts)} conflicts)")

        # Enhanced missing data detection with multi-format context
        missing_data = []

        # Check for essential spatial data
        if not context.project_data.get("total_area"):
            missing_data.append("Suprafața totală construită")
        if not context.project_data.get("total_rooms"):
            missing_data.append("Numărul de camere")

        # Check for requirements (can come from TXT or user input)
        has_requirements = (
            context.romanian_context.get("user_requirements") or
            context.project_data.get("txt_requirements")
        )
        if not has_requirements:
            missing_data.append("Cerințe detaliate ale proiectului")

        # Check for material specifications (can come from DXF, PDF, or TXT)
        # Add None-safety for file_analysis and nested dxf_analysis structures
        dxf_analysis = file_analysis.get("dxf_analysis") if file_analysis else None
        has_materials = False
        if context.project_data.get("pdf_materials") or context.project_data.get("txt_keywords"):
            has_materials = True
        else:
            if dxf_analysis and isinstance(dxf_analysis, dict):
                spec_analysis = dxf_analysis.get("dxf_analysis", {}).get("specification_analysis", {})
                if spec_analysis:
                    has_materials = spec_analysis.get("materials_count", 0) > 0
        if not has_materials:
            missing_data.append("Specificații materiale")

        # Adjust confidence based on missing data and multi-format availability
        format_bonus = 0.0
        if file_analysis.get("dxf_analysis"):
            format_bonus += 0.1
        if file_analysis.get("pdf_analysis"):
            format_bonus += 0.05
        if file_analysis.get("txt_analysis"):
            format_bonus += 0.05

        final_confidence = base_confidence * (1.0 - len(missing_data) * 0.1) + format_bonus
        
        if rfp_data and rfp_data.get('extraction_confidence', 0) > 0.7:
            logger.info("🎯 High-quality RFP data available - boosting confidence")
            final_confidence = min(1.0, final_confidence + 0.15)

        context.confidence_score = max(0.0, min(final_confidence, 1.0))
        context.missing_data = missing_data

        # Store context data in analysis_results for persistence
        context.analysis_results = {
            "confidence": context.confidence_score,
            "can_generate_offer": context.confidence_score >= self.offer_generation_threshold,
            "missing_data": missing_data,
            "file_analysis": file_analysis,
            "requirements_analysis": requirements_analysis,
            "ml_insights": ml_insights,
            # ✅ CRITICAL: Include these for context restoration
            "project_data": dict(context.project_data),  # Make a copy
            "romanian_context": dict(context.romanian_context),  # Make a copy
            'rfp_data': rfp_data,
            'has_rfp': rfp_data is not None,
            'cross_reference': cross_ref_result.to_dict(),
            'consistency_validated': True,
            "analysis_summary": {
                "file_analysis": file_analysis,
                "requirements_analysis": {
                    "romanian_context": context.romanian_context,
                    "confidence_contribution": requirements_analysis.get("confidence_contribution", 0.0)
                },
                "missing_data": missing_data,
                "confidence_score": context.confidence_score,
                "project_data": dict(context.project_data)
            }
        }

        return context.analysis_results

    def _extract_structured_data(self, consolidated_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract structured data from analysis for response builder
        Ensures data is in the format expected by response_builder templates
        """
        
        structured = {
            'dxf_analysis': {},
            'pdf_analysis': {},
            'txt_analysis': {},
            'rfp_data': {}
        }
        
        # Extract DXF data
        file_analysis = consolidated_analysis.get('file_analysis', {})
        dxf_raw = file_analysis.get('dxf_analysis', {})
        
        if dxf_raw:
            dxf_data = dxf_raw.get('dxf_analysis', {})
            
            structured['dxf_analysis'] = {
                'dxf_analysis': {
                    'total_area': dxf_data.get('total_area'),
                    'total_rooms': dxf_data.get('total_rooms'),
                    'rooms': dxf_data.get('rooms', []),
                    'room_breakdown': dxf_data.get('room_breakdown', []),
                    'dimensions': dxf_data.get('dimensions', {}),
                    'has_hvac': dxf_data.get('has_hvac', False),
                    'hvac_inventory': dxf_data.get('hvac_inventory', []),
                    'has_electrical': dxf_data.get('has_electrical', False),
                    'electrical_inventory': dxf_data.get('electrical_inventory', []),
                    'wall_types': dxf_data.get('wall_types', []),
                    'technical_notes': dxf_data.get('technical_notes', [])
                }
            }
        
        # Extract RFP data
        rfp_raw = consolidated_analysis.get('rfp_data', {})
        if rfp_raw:
            structured['rfp_data'] = {
                'project_info': rfp_raw.get('project_info', {}),
                'timeline': rfp_raw.get('timeline', {}),
                'financial': rfp_raw.get('financial', {}),
                'scope': rfp_raw.get('scope', {}),
                'team': rfp_raw.get('team', {})
            }
        
        # Extract PDF analysis
        pdf_raw = file_analysis.get('pdf_analysis', {})
        if pdf_raw:
            structured['pdf_analysis'] = {
                'document_type': pdf_raw.get('document_type'),
                'page_count': pdf_raw.get('page_count'),
                'key_topics': pdf_raw.get('key_topics', []),
                'tables_found': pdf_raw.get('tables_found', 0)
            }
        
        # Extract TXT analysis
        txt_raw = file_analysis.get('txt_analysis', {})
        if txt_raw:
            structured['txt_analysis'] = {
                'requirements': txt_raw.get('requirements', []),
                'keywords': txt_raw.get('keywords', [])
            }
        
        return structured

    async def _generate_unified_response(
        self,
        consolidated_analysis: Dict[str, Any],
        context: AgentContext
    ) -> UnifiedResponse:
        """
        Enhanced with better data extraction before formatting
        """
        
        start_time = time.time()
        
        # ✅ STEP 1: Perform gap analysis
        logger.info("📊 Performing gap analysis")
        gap_result = self.gap_analyzer.analyze_gaps(
            dxf_data=consolidated_analysis.get('file_analysis', {}).get('dxf_analysis'),
            rfp_data=consolidated_analysis.get('rfp_data'),
            user_requirements=context.project_data,
            conversation_context=context.conversation_history
        )
        
        # ✅ STEP 2: Extract structured data for response builder
        logger.info("🔍 Extracting structured data")
        structured_data = self._extract_structured_data(consolidated_analysis)
        logger.info(f"📊 Structured data keys: {list(structured_data.keys())}")
        logger.info(f"📊 DXF data available: {bool(structured_data.get('dxf_analysis'))}")
        
        # ✅ STEP 3: Update context
        context.confidence_score = gap_result.overall_confidence
        context.missing_data = [
            gap.display_name_ro for gap in
            (gap_result.critical_gaps + gap_result.high_priority_gaps)
        ]
        
        # ✅ STEP 4: Get cross-reference data
        cross_ref_data = consolidated_analysis.get('cross_reference')
        
        # ✅ STEP 5: Build professional response using templates
        logger.info("🎨 Building professional response with templates")
        
        session_data = {
            'session_id': context.session_id,
            'confidence_score': context.confidence_score
        }
        
        # Use response builder with structured data
        professional_response = self.response_builder.build_file_analysis_response(
            file_analysis=structured_data,  # ✅ Use structured data
            gap_analysis=gap_result,
            cross_reference=cross_ref_data,
            session_data=session_data
        )
        
        processing_time = (time.time() - start_time) * 1000
        logger.info(f"✅ Response built in {processing_time:.0f}ms")
        
        return UnifiedResponse(
            content=professional_response,
            confidence=gap_result.overall_confidence,
            mode=context.current_mode,
            next_questions=gap_result.prioritized_questions,
            can_generate_offer=gap_result.can_generate_offer,
            analysis_summary=consolidated_analysis,
            processing_time_ms=processing_time
        )
    # =========================================================================
    # PRIVATE METHODS - CONVERSATION HANDLING
    # =========================================================================

    async def _analyze_user_message(self, message: str, context: AgentContext) -> Dict[str, Any]:
        """Analyze user message in conversation context using context-aware LLM."""
        
        # Get files from context
        file_references = context.analysis_results.get("file_references", [])
        
        # Build detailed context for LLM
        context_summary = self._build_detailed_project_summary(context)
        
        # Build COMPLETE file list for LLM (no truncation)
        file_list_details = self._build_file_context_for_llm(context)

        # ✅ SIMPLIFIED: Standard analysis without mode detection
        analysis_prompt = f"""
Ești un asistent inteligent care analizează mesajele utilizatorilor în contextul unui proiect de construcții.
Extrage orice informație nouă furnizată de utilizator și actualizează starea proiectului.
Răspunde DOAR cu un obiect JSON valid, fără text suplimentar sau formatare markdown.

Context proiect:
{context_summary}

Fișiere disponibile:
{file_list_details}

Analizează acest mesaj și extrage:
1. Tip mesaj: "data_update" (furnizează date noi), "question" (pune întrebări), "general" (conversație)
2. Date noi extrase: suprafață, camere, buget, cerințe, timeline
3. Schimbare încredere: cât crește încrederea (0.0-1.0) datorită noilor informații

Mesaj utilizator: {message}

Răspunde cu JSON:
{{
  "message_type": "data_update|question|general",
  "extracted_data": {{"area": null, "rooms": null, "budget": null, "requirements": null, "timeline": null}},
  "confidence_delta": 0.0,
  "user_message": "{message}",
  "summary_of_update": "Scurtă descriere ce s-a extras"
}}
"""
        
        try:
            from src.services.llm_service import safe_construction_call
            llm_response = await safe_construction_call(
                user_input=analysis_prompt,
                system_prompt="Ești un analist tehnic. Răspunzi DOAR cu JSON valid.",
                temperature=0.1
            )
            
            # Parse LLM JSON response
            llm_response_clean = llm_response.strip()
            if llm_response_clean.startswith("```"):
                llm_response_clean = llm_response_clean.split("```")[1]
                if llm_response_clean.startswith("json"):
                    llm_response_clean = llm_response_clean[4:]
            
            llm_analysis = json.loads(llm_response_clean)
            
            # Update context with extracted data
            extracted = llm_analysis.get("extracted_data", {})
            if extracted.get("area"):
                context.project_data["total_area"] = extracted["area"]
            if extracted.get("rooms"):
                context.project_data["total_rooms"] = extracted["rooms"]
            if extracted.get("budget"):
                context.project_data["budget"] = extracted["budget"]
            if extracted.get("requirements"):
                current_reqs = context.romanian_context.get("user_requirements", "")
                context.romanian_context["user_requirements"] = f"{current_reqs} {extracted['requirements']}".strip()
            if extracted.get("timeline"):
                context.project_data["timeline"] = extracted["timeline"]
            
            # Update confidence
            confidence_delta = llm_analysis.get("confidence_delta", 0.0)
            context.confidence_score = min(context.confidence_score + confidence_delta, 1.0)
            
            analysis = {
                "message_type": llm_analysis.get("message_type", "general"),
                "extracted_data": extracted,
                "confidence_delta": confidence_delta,
                "user_message": message,
                "update_summary": llm_analysis.get("summary_of_update", "")
            }
            
            logger.info(f"✅ Message analysis: {analysis['message_type']}, confidence Δ: +{confidence_delta:.2f}")
            return analysis

        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"❌ LLM-based message analysis failed: {e}. Falling back to basic analysis.")
            return await self._fallback_analyze_user_message(message, context)

    async def _fallback_analyze_user_message(self, message: str, context: AgentContext) -> Dict[str, Any]:
        """Fallback method to analyze user message with regex if LLM fails."""
        analysis = {
            "message_type": "general", "extracted_data": {}, "confidence_delta": 0.0,
            "user_message": message
        }
        # Look for area information
        area_match = self._extract_area_from_text(message)
        if area_match:
            context.project_data["total_area"] = area_match
            analysis["extracted_data"]["area"] = area_match
            analysis["confidence_delta"] += 0.15
            analysis["message_type"] = "data_update"
        # Look for room count
        rooms_match = self._extract_room_count_from_text(message)
        if rooms_match:
            context.project_data["total_rooms"] = rooms_match
            analysis["extracted_data"]["rooms"] = rooms_match
            analysis["confidence_delta"] += 0.10
            analysis["message_type"] = "data_update"
        # Look for budget information
        budget_match = self._extract_budget_from_text(message)
        if budget_match:
            context.project_data["budget"] = budget_match
            analysis["extracted_data"]["budget"] = budget_match
            analysis["confidence_delta"] += 0.05
        # Update context confidence
        context.confidence_score = min(context.confidence_score + analysis["confidence_delta"], 1.0)
        return analysis

    async def _generate_conversation_response(
        self,
        message_analysis: Dict[str, Any],
        context: AgentContext
    ) -> UnifiedResponse:
        """Enhanced conversation response with professional formatting"""
        
        # Store initial state for progress tracking
        initial_confidence = context.confidence_score
        initial_gaps = len(context.missing_data)
        
        # Build conversation prompt
        system_prompt = self._build_conversation_prompt(message_analysis, context)
        user_message = message_analysis.get("user_message", "")
        
        # Call LLM
        from src.services.llm_service import safe_construction_call
        llm_response = await safe_construction_call(
            user_input=user_message,
            system_prompt=system_prompt,
            temperature=0.3
        )
        
        # ✅ PERFORM GAP ANALYSIS AFTER USER RESPONSE
        logger.info("🔍 Analyzing gaps after user response")
        gap_result = self.gap_analyzer.analyze_gaps(
            dxf_data=context.analysis_results.get('file_analysis', {}).get('dxf_analysis'),
            rfp_data=context.analysis_results.get('rfp_data'),
            user_requirements=context.project_data,
            conversation_context=context.conversation_history
        )
        
        # Update context
        context.confidence_score = gap_result.overall_confidence
        context.missing_data = [
            gap.display_name_ro for gap in
            (gap_result.critical_gaps + gap_result.high_priority_gaps)
        ]
        
        # ✅ CALCULATE PROGRESS
        confidence_improvement = gap_result.overall_confidence - initial_confidence
        gaps_closed = initial_gaps - len(context.missing_data)
        
        if gaps_closed > 0:
            logger.info(f"📈 Progress: +{confidence_improvement:.1%} confidence, {gaps_closed} gaps closed")
        
        # Prepare context for response builder
        conversation_context = {
            'confidence_improvement': confidence_improvement,
            'gaps_closed': gaps_closed
        }
        
        # Get cross-reference data
        cross_ref_data = context.analysis_results.get('cross_reference')
        
        # ✅ USE RESPONSE BUILDER FOR CHAT FORMATTING
        logger.info("🎨 Building chat response")
        
        # Determine if we should show full analysis or compact
        show_full = gaps_closed > 0 or confidence_improvement > 0.1
        
        professional_response = self.response_builder.build_chat_response(
            llm_response=llm_response,
            gap_analysis=gap_result,
            cross_reference=cross_ref_data,
            context=conversation_context,
            show_full_analysis=show_full
        )
        
        return UnifiedResponse(
            content=professional_response,
            confidence=gap_result.overall_confidence,
            mode=context.current_mode,
            next_questions=gap_result.prioritized_questions,
            can_generate_offer=gap_result.can_generate_offer
        )

    def _build_conversation_prompt(self, message_analysis: Dict[str, Any], context: AgentContext) -> str:
        """Builds the system prompt for the conversation LLM call."""
        # ✅ Build comprehensive project summary with FILE DETAILS
        project_summary = self._build_detailed_project_summary(context)
        
        # ✅ Build COMPLETE file context for LLM
        file_context = self._build_file_context_for_llm(context)

        # ✅ ENHANCED: System prompt ALWAYS demands exhaustive detail
        return f"""Tu ești un consultant tehnic de construcții specializat în proiecte comerciale/birouri interioare.

REGULI OBLIGATORII PENTRU RĂSPUNSURI:
1. ÎNTOTDEAUNA furnizezi descrieri tehnice COMPLETE și EXHAUSTIVE
2. NICIODATĂ nu rezumi sau omit detalii
3. NICIODATĂ nu spui "și altele" - listezi TOT ce ai găsit
4. Pentru fiecare întrebare, verifici TOATE datele disponibile

CONTEXT PROIECT ACTUAL:
{project_summary}

FIȘIERE ANALIZATE (date complete):
{file_context}

STARE PROIECT:
- Încredere actuală: {context.confidence_score:.1%}
- Date lipsă: {', '.join(context.missing_data) if context.missing_data else 'Niciuna'}

INSTRUCȚIUNI SPECIFICE PENTRU RĂSPUNSURI:

Pentru fiecare fișier DXF:
- Listează TOATE camerele cu suprafețe exacte și dimensiuni (lungime × lățime)
- Numără și detaliază TOATE componentele MEP:
  * Prize și întrerupătoare (locații și tipuri)
  * Unități HVAC (capacități, poziționate)
  * Corpuri de iluminat (tipuri, cantități)
  * Instalații sanitare (dacă aplicabil)
- Specifică TOATE tipurile de pereți cu lungimi totale și grosimi
- Enumeră TOATE materialele identificate cu specificații complete
- Detaliază feronere și uși (dimensiuni, tipuri, materiale)
- Descrie tipuri pardoseli și suprafețe
- Menționează înălțimi tavane unde sunt disponibile

Pentru fiecare fișier PDF:
- Extrage și listează TOATE specificațiile tehnice
- Prezintă TOATE cerințele reglementare identificate
- Detaliază informații cost și timeline dacă există
- Listează TOATE materialele menționate cu specificații

Pentru fiecare fișier TXT:
- Enumeră TOATE cerințele clientului
- Detaliază TOATE preferințele menționate
- Listează TOATE cuvintele cheie relevante pentru construcție

La SFÂRȘITUL fiecărui răspuns, include ÎNTOTDEAUNA analiza completitudinii:

**Analiza completitudinii pentru ofertă:**
✅ Date disponibile: [listează EXACT ce avem]
❌ Date lipsă pentru ofertă completă: [listează EXACT ce lipsește]
📋 Următorii pași: [acțiuni specifice necesare]

Răspunde în română, profesional, cu termeni tehnici corecți pentru construcții comerciale/birouri."""

    def _build_file_context_for_llm(self, context: AgentContext) -> str:
        """Build detailed file context with COMPLETE extracted analysis data - NO LIMITS"""
        
        # FIX: Correctly access file_references from within the 'file_analysis' dictionary
        file_analysis = context.analysis_results.get("file_analysis", {})
        file_references = file_analysis.get("file_references", [])
        
        if not file_references or len(file_references) == 0:
            logger.warning("⚠️ No file references found in context!")
            return "**Fișiere încărcate:** Niciun fișier detectat în context"
        
        logger.info(f"📋 Building COMPLETE file context for {len(file_references)} files (NO TRUNCATION)")
        
        file_context_parts = [f"**Fișiere încărcate și analizate:** {len(file_references)} fișiere\n"]
        
        for idx, file_ref in enumerate(file_references, 1):
            filename = file_ref.get("filename", "unknown")
            file_type = file_ref.get("content_type", "unknown")
            file_size = file_ref.get("size", 0)
            analysis_summary = file_ref.get("analysis_summary", {})
            
            file_context_parts.append(f"\n{idx}. **{filename}**")
            file_context_parts.append(f"   - Tip: {file_type}")
            file_context_parts.append(f"   - Dimensiune: {file_size / 1024:.1f} KB")
            
            # ✅ CRITICAL: Show COMPLETE data for DXF files (full spec_analysis + summary)
            if analysis_summary.get("type") == "dxf":
                file_context_parts.append(f"   - Format: Plan tehnic DXF")

                # ✅ DEBUG: Log data structure to understand what exists
                logger.info("🔍 DXF Data Structure Check:")
                logger.info(f"   - analysis_summary keys: {list(analysis_summary.keys())}")
                
                full_analysis_data = file_ref.get("analysis_data", {})
                logger.info(f"   - analysis_data keys: {list(full_analysis_data.keys())}")
                
                dxf_analysis = full_analysis_data.get("dxf_analysis", {})
                logger.info(f"   - dxf_analysis keys: {list(dxf_analysis.keys())}")
                
                spec_analysis = dxf_analysis.get("spec_analysis", {}) if isinstance(dxf_analysis, dict) else {}
                if spec_analysis:
                    logger.info(f"   - spec_analysis keys: {list(spec_analysis.keys())}")
                    logger.info(f"   - hvac_inventory count: {len(spec_analysis.get('hvac_inventory', []))}")
                    logger.info(f"   - electrical_inventory count: {len(spec_analysis.get('electrical_inventory', []))}")
                    logger.info(f"   - door_window_schedule count: {len(spec_analysis.get('door_window_schedule', []))}")

                # ========================================================
                # SECTION 1: BASIC INFO (from analysis_summary)
                # ========================================================
                total_area = analysis_summary.get("total_area", 0)
                if total_area > 0:
                    file_context_parts.append(f"   - **Suprafață totală: {total_area:.2f} mp**")
                
                total_rooms = analysis_summary.get("total_rooms", 0)
                if total_rooms > 0:
                    file_context_parts.append(f"   - **Număr spații: {total_rooms}**")
                
                # ========================================================
                # SECTION 2: ROOMS WITH DIMENSIONS (COMPLETE)
                # ========================================================
                room_breakdown = analysis_summary.get("room_breakdown", [])
                if room_breakdown:
                    file_context_parts.append(f"   - **Detalii spații (TOATE {len(room_breakdown)} spații):**")
                    for room in room_breakdown:
                        room_name = room.get("room_name", "Unknown")
                        romanian_name = room.get("romanian_name", room_name)
                        room_area = room.get("area", 0)
                        room_dims = room.get("dimensions", {})
                        
                        # Show Romanian name (more natural for Romanian users)
                        room_detail = f"     • {romanian_name}: {room_area:.1f} mp"
                        
                        # Add dimensions if available
                        if room_dims.get("length") and room_dims.get("width"):
                            room_detail += f" ({room_dims['length']:.1f}m × {room_dims['width']:.1f}m)"
                        
                        file_context_parts.append(room_detail)
                
                # ========================================================
                # SECTION 3: WALL TYPES (COMPLETE)
                # ========================================================
                wall_types = analysis_summary.get("wall_types", [])
                if wall_types:
                    file_context_parts.append(f"   - **Tipuri pereți (TOATE {len(wall_types)} tipuri):**")
                    for wall in wall_types:
                        if isinstance(wall, dict):
                            wall_desc = f"     • {wall.get('type_code', 'Unknown')}"
                            if wall.get('thickness_mm'):
                                wall_desc += f" - grosime {wall['thickness_mm']}mm"
                            if wall.get('fire_rating'):
                                wall_desc += f" - {wall['fire_rating']}"
                            file_context_parts.append(wall_desc)
                        else:
                            file_context_parts.append(f"     • {wall}")

                # ========================================================
                # SECTION 4: FINISHES (from spec_analysis)
                # ========================================================
                finishes_count = spec_analysis.get("finishes_count", 0)
                if finishes_count > 0:
                    file_context_parts.append(f"   - **Finisaje identificate: {finishes_count} finisaje**")
                    
                    # Load finishes from specification_analysis
                    specification_data = spec_analysis.get("specification_analysis", {})
                    if specification_data:
                        finishes = specification_data.get("finishing_requirements", [])
                        if finishes:
                            file_context_parts.append(f"   - **Lista completă finisaje:**")
                            for finish in finishes:
                                finish_type = finish.get("finish_type", "Unknown")
                                finish_spec = finish.get("specification", "")
                                file_context_parts.append(f"     • {finish_type}: {finish_spec}")

                # ========================================================
                # SECTION 5: HVAC INVENTORY (COMPLETE)
                # ========================================================
                hvac_inventory = spec_analysis.get("hvac_inventory", [])
                if hvac_inventory:
                    file_context_parts.append(f"   - **Inventar HVAC (TOATE {len(hvac_inventory)} unități):**")
                    for hvac in hvac_inventory:
                        hvac_type = hvac.get('type', 'Unknown')
                        hvac_desc = f"     • {hvac_type}"
                        
                        if hvac.get('model'):
                            hvac_desc += f" - {hvac['model']}"
                        if hvac.get('capacity_kw'):
                            hvac_desc += f" ({hvac['capacity_kw']}kW)"
                        if hvac.get('room'):
                            hvac_desc += f" [Cameră: {hvac['room']}]"
                        
                        file_context_parts.append(hvac_desc)
                elif analysis_summary.get("has_hvac"):
                    file_context_parts.append(f"   - **Sistem HVAC:** Detectat (fără inventory detaliat)")

                # ========================================================
                # SECTION 6: ELECTRICAL INVENTORY (COMPLETE)
                # ========================================================
                electrical_inventory = spec_analysis.get("electrical_inventory", [])
                if electrical_inventory:
                    file_context_parts.append(f"   - **Inventar Instalații Electrice (TOATE {len(electrical_inventory)} componente):**")
                    
                    # Group by component_type for better readability
                    outlets = [e for e in electrical_inventory if e.get('component_type') == 'outlet']
                    switches = [e for e in electrical_inventory if e.get('component_type') == 'switch']
                    lights = [e for e in electrical_inventory if e.get('component_type') == 'light_fixture']
                    
                    # OUTLETS
                    if outlets:
                        total_outlets = sum(e.get('quantity', 1) for e in outlets)
                        file_context_parts.append(f"     • Prize: {total_outlets} bucăți total")
                        # Show first 10 with details
                        for outlet in outlets[:10]:
                            outlet_desc = f"       - {outlet.get('quantity', 1)}x"
                            if outlet.get('power_rating'):
                                outlet_desc += f" {outlet['power_rating']}"
                            if outlet.get('room_association'):
                                outlet_desc += f" în {outlet['room_association']}"
                            file_context_parts.append(outlet_desc)
                        if len(outlets) > 10:
                            file_context_parts.append(f"       ... și {len(outlets) - 10} prize suplimentare")
                    
                    # SWITCHES
                    if switches:
                        total_switches = sum(e.get('quantity', 1) for e in switches)
                        file_context_parts.append(f"     • Întrerupătoare: {total_switches} bucăți total")
                        for switch in switches[:10]:
                            switch_desc = f"       - {switch.get('quantity', 1)}x"
                            if switch.get('room_association'):
                                switch_desc += f" în {switch['room_association']}"
                            file_context_parts.append(switch_desc)
                        if len(switches) > 10:
                            file_context_parts.append(f"       ... și {len(switches) - 10} întrerupătoare suplimentare")
                    
                    # LIGHTS
                    if lights:
                        total_lights = sum(e.get('quantity', 1) for e in lights)
                        file_context_parts.append(f"     • Corpuri iluminat: {total_lights} bucăți total")
                        for light in lights[:10]:
                            light_desc = f"       - {light.get('quantity', 1)}x"
                            if light.get('room_association'):
                                light_desc += f" în {light['room_association']}"
                            file_context_parts.append(light_desc)
                        if len(lights) > 10:
                            file_context_parts.append(f"       ... și {len(lights) - 10} corpuri suplimentare")
                elif analysis_summary.get("has_electrical"):
                    file_context_parts.append(f"   - **Instalații electrice:** Detectate (fără inventory detaliat)")

                # ========================================================
                # SECTION 7: DOORS & WINDOWS SCHEDULE
                # ========================================================
                door_window_schedule = spec_analysis.get("door_window_schedule", [])
                if door_window_schedule:
                    doors = [dw for dw in door_window_schedule if dw.get('type') == 'door']
                    windows = [dw for dw in door_window_schedule if dw.get('type') == 'window']
                    
                    if doors:
                        file_context_parts.append(f"   - **Uși (TOATE {len(doors)} bucăți):**")
                        for door in doors[:15]:
                            width = door.get('width', 0)
                            height = door.get('height', 0)
                            if width > 0 and height > 0:
                                door_desc = f"     • {width:.2f}m × {height:.2f}m"
                                if door.get('material'):
                                    door_desc += f" - {door['material']}"
                                if door.get('opening_type'):
                                    door_desc += f" ({door['opening_type']})"
                                file_context_parts.append(door_desc)
                        if len(doors) > 15:
                            file_context_parts.append(f"     ... și {len(doors) - 15} uși suplimentare")
                    
                    if windows:
                        file_context_parts.append(f"   - **Ferestre: {len(windows)} bucăți**")

                # ========================================================
                # SECTION 8: DIMENSIONS FLAG
                # ========================================================
                if analysis_summary.get("has_dimensions"):
                    file_context_parts.append(f"   - **Dimensiuni:** Plan cotat complet")
                
                # ✅ NEW: Show ALL materials
                spec_analysis = analysis_summary.get("specification_analysis", {})
                if spec_analysis:
                    materials_count = spec_analysis.get("materials_count", 0)
                    if materials_count > 0:
                        file_context_parts.append(f"   - **Materiale identificate: {materials_count} materiale**")
                        
                        specifications = spec_analysis.get("specifications", {})
                        if specifications:
                            materials = specifications.get("material_specifications", [])
                            if materials:
                                file_context_parts.append(f"   - **Lista completă materiale:**")
                                for mat in materials:  # NO limit
                                    mat_type = mat.get("material_type", "Unknown")
                                    mat_spec = mat.get("specification", "")
                                    file_context_parts.append(f"     • {mat_type}: {mat_spec}")
                        
                    # ✅ NEW: Show ALL finishes
                    finishes_count = spec_analysis.get("finishes_count", 0)
                    if finishes_count > 0:
                        file_context_parts.append(f"   - **Finisaje identificate: {finishes_count} finisaje**")
                        if specifications:
                            finishes = specifications.get("finishing_requirements", [])
                            if finishes:
                                file_context_parts.append(f"   - **Lista completă finisaje:**")
                                for finish in finishes:  # NO limit
                                    finish_type = finish.get("finish_type", "Unknown")
                                    finish_spec = finish.get("specification", "")
                                    file_context_parts.append(f"     • {finish_type}: {finish_spec}")
                
                # ✅ NEW: MEP Components - COMPLETE inventory
                if analysis_summary.get("has_electrical"):
                    file_context_parts.append(f"   - **Instalații electrice:** Detectate")
                    # TODO Medium-term: Add detailed electrical inventory here
                    
                if analysis_summary.get("has_hvac"):
                    file_context_parts.append(f"   - **Sistem HVAC:** Detectat")
                    # TODO Medium-term: Add detailed HVAC inventory here
                
                # ✅ NEW: Dimensions information
                if analysis_summary.get("has_dimensions"):
                    file_context_parts.append(f"   - **Dimensiuni:** Plan cotat complet")
            # ✅ NEW: Image file context
            elif analysis_summary.get("type") == "image":
                file_context_parts.append(f"   - Format: Floor plan image")
                
                room_labels = analysis_summary.get("room_labels_found", 0)
                if room_labels > 0:
                    file_context_parts.append(f"   - **Room labels detected: {room_labels}**")
                
                confidence = analysis_summary.get("confidence", 0)
                quality = analysis_summary.get("quality_score", 0)
                file_context_parts.append(f"   - Analysis confidence: {confidence:.0%}")
                file_context_parts.append(f"   - Image quality: {quality:.0%}")
                
                warnings = analysis_summary.get("warnings", [])
                if warnings:
                    file_context_parts.append(f"   - ⚠️ Warnings: {len(warnings)}")
                    for warning in warnings[:2]:  # Show first 2 warnings
                        file_context_parts.append(f"     • {warning}")
            
            # ✅ COMPLETE PDF analysis
            elif analysis_summary.get("type") == "pdf":
                file_context_parts.append(f"   - Format: Document PDF")
                
                # ✅ NEW: Check if detailed OCR data is available
                ocr_result_id = file_ref.get("ocr_result_id")
                has_full_ocr = file_ref.get("has_full_ocr", False)
                if has_full_ocr and ocr_result_id:
                    file_context_parts.append(f"   - **OCR Complet Disponibil:** ID {ocr_result_id}")
                    # Optional: Add logic here to load full OCR if required by a specific task
                    # For now, just indicate its availability.

                construction_specs = analysis_summary.get("construction_specs", [])
                if construction_specs:
                    file_context_parts.append(f"   - **Specificații tehnice (TOATE {len(construction_specs)}):**")
                    for spec in construction_specs:  # NO limit
                        file_context_parts.append(f"     • {spec}")
                
                material_refs = analysis_summary.get("material_references", [])
                if material_refs:
                    file_context_parts.append(f"   - **Materiale menționate (TOATE {len(material_refs)}):**")
                    for mat in material_refs:  # NO limit
                        file_context_parts.append(f"     • {mat}")
                
                page_count = analysis_summary.get("page_count", 0)
                if page_count > 0:
                    file_context_parts.append(f"   - **Pagini: {page_count}**")
            
            # ✅ COMPLETE TXT analysis
            elif analysis_summary.get("type") == "txt":
                file_context_parts.append(f"   - Format: Document text")
                
                requirements = analysis_summary.get("requirements", [])
                if requirements:
                    file_context_parts.append(f"   - **Cerințe identificate (TOATE {len(requirements)}):**")
                    for req in requirements:  # NO limit
                        file_context_parts.append(f"     • {req}")
                
                client_prefs = analysis_summary.get("client_preferences", {})
                if client_prefs:
                    file_context_parts.append(f"   - **Preferințe client:**")
                    for key, value in client_prefs.items():
                        file_context_parts.append(f"     • {key}: {value}")
        
        result = "\n".join(file_context_parts)
        logger.info(f"✅ Built COMPLETE file context: {len(result)} characters")
        return result
    
    
    def _build_detailed_project_summary(self, context: AgentContext) -> str:
        """Build comprehensive project summary including all extracted data"""
        summary_parts = []
    
        # Spatial data
        if context.project_data.get("total_area"):
            # ✅ FIX: Handle cases where total_area might be a string
            try:
                area = float(context.project_data['total_area'])
                summary_parts.append(f"Suprafață: {area:.2f} mp")
            except (ValueError, TypeError):
                # Fallback if conversion to float fails
                summary_parts.append(f"Suprafață: {context.project_data['total_area']}")
    
        if context.project_data.get("total_rooms"):
            summary_parts.append(f"Camere: {context.project_data['total_rooms']}")
        
        # Room breakdown details
        if context.project_data.get("room_breakdown"):
            rooms = context.project_data["room_breakdown"]
            room_types = [r.get("room_name", "Unknown") for r in rooms[:5]]
            summary_parts.append(f"Tipuri camere: {', '.join(room_types)}")
    
        # Budget
        if context.project_data.get("budget"):
            summary_parts.append(f"Buget: {context.project_data['budget']} RON")
    
        # Requirements
        if context.romanian_context.get("user_requirements"):
            summary_parts.append(f"Cerințe: {context.romanian_context['user_requirements'][:200]}...")
        elif context.project_data.get("txt_requirements"):
            reqs = context.project_data["txt_requirements"]
            summary_parts.append(f"Cerințe extrase: {', '.join(reqs[:3])}")
    
        # Material specifications
        if context.project_data.get("pdf_materials"):
            materials = context.project_data["pdf_materials"]
            summary_parts.append(f"Materiale specificate: {len(materials)} tipuri")
    
        return "; ".join(summary_parts) if summary_parts else "Proiect în analiză - fișiere procesate"

    async def _handle_offer_request(self, context: AgentContext) -> Dict[str, Any]:
        """Handle offer generation request"""

        if context.confidence_score < self.offer_generation_threshold:
            # Not ready for offer
            missing_info = '\n'.join([f"- {item}" for item in context.missing_data])

            response_content = f"""**Nu pot genera oferta încă**

**Încredere actuală: {context.confidence_score:.1%}** (necesară: {self.offer_generation_threshold:.0%})

**Informații lipsă:**
{missing_info}

Vă rog să furnizați aceste informații pentru o ofertă precisă."""

            return {
                "session_id": context.session_id,
                "response": response_content,
                "confidence": context.confidence_score,
                "can_generate_offer": False,
                "next_questions": context.missing_data[:3]
            }

        # Generate offer
        offer_result = await self.generate_offer(context.session_id)

        if offer_result["success"]:
            return {
                "session_id": context.session_id,
                "response": offer_result["offer"],
                "confidence": context.confidence_score,
                "can_generate_offer": True,
                "offer_generated": True
            }
        else:
            return {
                "session_id": context.session_id,
                "response": f"Eroare la generarea ofertei: {offer_result.get('message', 'Unknown error')}",
                "confidence": context.confidence_score,
                "can_generate_offer": False,
                "error": offer_result.get("error")
            }

    # =========================================================================
    # PRIVATE METHODS - OCR HANDLING
    # =========================================================================

    async def cleanup_old_ocr_data(self, days: int = 90) -> Dict[str, Any]:
        """
        Cleanup OCR data older than specified days
        
        Args:
            days: Number of days to retain OCR data
            
        Returns:
            Cleanup statistics
        """
        try:
            logger.info(f"🧹 Cleaning up OCR data older than {days} days")
            result = await self.ocr_storage.cleanup_old_results(days)
            return result
        except Exception as e:
            logger.error(f"❌ OCR cleanup failed: {e}")
            return {"error": str(e), "deleted_count": 0}
    # =========================================================================
    # PRIVATE METHODS - OFFER GENERATION
    # =========================================================================

    async def _generate_professional_offer(self, context: AgentContext) -> str:
        """Enhanced with professional offer formatting"""

        project_data = context.project_data
        romanian_context = context.romanian_context

        # Get historical insights if available
        ml_insights = context.analysis_results.get("ml_insights", {})
        historical_insights = ml_insights.get("historical_insights", []) if ml_insights else []

        # Get cost estimates (now using historical data)
        cost_estimate = await self._calculate_cost_estimate(context, historical_insights)
        timeline_estimate = await self._calculate_timeline_estimate(context, historical_insights)

        # Build historical reference section if we have similar projects
        historical_section = ""
        if historical_insights and len(historical_insights) > 0:
            avg_area = sum(p.get("area", 0) for p in historical_insights) / len(historical_insights)
            avg_cost_per_sqm = sum(p.get("cost_per_sqm", 0) for p in historical_insights) / len(historical_insights)

            historical_section = f"""

**REFERINȚE PROIECTE SIMILARE**

Am analizat {len(historical_insights)} proiecte similare din baza noastră:
- Suprafață medie: {avg_area:.1f} mp
- Cost mediu: {avg_cost_per_sqm:.0f} RON/mp
- Această estimare reflectă prețurile reale de piață bazate pe proiecte finalizate

"""

        # Extract recommended materials from historical data
        materials_section = self._build_materials_recommendations(historical_insights)

        offer_content = f"""**OFERTĂ TEHNICĂ ȘI COMERCIALĂ**

**DEMOPLAN CONSTRUCT SRL**
Data: {datetime.now().strftime('%d.%m.%Y')}
Sesiune: {context.session_id[:8]}...

**1. ANALIZĂ TEHNICĂ COMPLETĂ**

**Caracteristici proiect:**
- Suprafață totală: {project_data.get('total_area', 'N/A')} mp
- Numărul de camere: {project_data.get('total_rooms', 'N/A')}
- Tipul construcției: Rezidențială
- Încredere analiză: {context.confidence_score:.1%}

**Cerințe client:**
{romanian_context.get('user_requirements', 'Renovare/construcție conform planurilor')}
{historical_section}

**2. ESTIMARE COSTURI**

**Costuri preliminare pe categorii:**
- Materiale: {cost_estimate['materials']:,} RON
- Manoperă: {cost_estimate['labor']:,} RON
- Transport și utilaje: {cost_estimate['transport']:,} RON
- TVA 19%: {cost_estimate['vat']:,} RON

**TOTAL ESTIMAT: {cost_estimate['total']:,} RON**

{cost_estimate.get('confidence_note', '')}

*Prețurile finale vor fi ajustate după confirmarea tuturor specificațiilor tehnice.*

**3. PROGRAM DE EXECUȚIE**

**Durata estimată:** {timeline_estimate['total_days']} zile lucrătoare
- Pregătire și mobilizare: {timeline_estimate['preparation']} zile
- Execuție lucrări principale: {timeline_estimate['execution']} zile
- Finisaje și demobilizare: {timeline_estimate['finishing']} zile

{timeline_estimate.get('historical_note', '')}

**Data începerii:** După semnarea contractului
**Condiții meteo:** Sezon optim Aprilie-Octombrie

**4. SPECIFICAȚII TEHNICE**
{materials_section}

**Manoperă:**
- Echipă specializată cu experiență 5+ ani
- Responsabil tehnic cu atestare ANIF
- Garanție lucrări: 3 ani

**5. CONDIȚII CONTRACTUALE**

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

**CONTACT**

Pentru întrebări sau programarea unei vizite tehnice:
- Email: office@demoplan.ro

**Oferta este valabilă 30 de zile de la data emiterii.**

*Ofertă generată automat prin sistemul DEMOPLAN cu analiză AI avansată și date din {len(historical_insights)} proiecte similare.*

**Următorii pași:**
1. Revizuiți oferta și specificațiile
2. Solicitați clarificări dacă este necesar
3. Confirmați acceptarea pentru programarea vizitei tehnice
4. Semnarea contractului după acordul final

**Vă mulțumim pentru încrederea acordată!**"""

        # Check for warnings from cross-reference
        warnings = []
        cross_ref = context.analysis_results.get('cross_reference')
        if cross_ref:
            warning_conflicts = [
                c for c in cross_ref.get('conflicts', [])
                if c.get('severity') == 'warning'
            ]
            
            if warning_conflicts:
                warnings.append(
                    f"Detectate {len(warning_conflicts)} avertismente în date - "
                    "oferta este generată cu informațiile disponibile"
                )
        
        # ✅ USE RESPONSE BUILDER FOR OFFER FORMATTING
        professional_offer = self.response_builder.build_offer_response(
            offer_content=offer_content,
            confidence=context.confidence_score,
            warnings=warnings if warnings else None
        )
        
        return professional_offer

    # =========================================================================
    # PRIVATE METHODS - UTILITIES
    # =========================================================================

    def _build_error_response(
        self,
        error: Exception,
        context: Optional[str] = None
    ) -> str:
        """Build user-friendly error response"""
        
        error_message = str(error)
        
        # Suggest actions based on error type
        suggestions = []
        
        if "file" in error_message.lower():
            suggestions.append("Verificați că fișierul este valid și nu este corupt")
            suggestions.append("Încercați să re-uploadați fișierul")
        elif "timeout" in error_message.lower():
            suggestions.append("Încercați din nou - serverul poate fi ocupat")
        elif "parse" in error_message.lower() or "extract" in error_message.lower():
            suggestions.append("Fișierul poate avea un format neașteptat")
            suggestions.append("Verificați că fișierul conține date valide")
        else:
            suggestions.append("Încercați din nou")
            suggestions.append("Dacă problema persistă, contactați suportul")
        
        return self.response_builder.build_error_response(
            error_message=error_message,
            context=context,
            suggestions=suggestions
        )
    
    def _flatten_for_firestore(self, data: Any) -> Any:
        """
        Recursively flatten complex objects to Firestore-compatible primitives.
        Handles dataclasses, enums, datetime objects, and nested structures.
        Args:
            data: Any data structure that needs to be flattened
        Returns:
            Firestore-compatible primitive types (dict, list, str, int, float, bool, None)
        """
        # Handle None
        if data is None:
            return None
        
        # ✅ ADDED: Handle bytes objects to prevent Firestore errors
        if isinstance(data, bytes):
            logger.warning(f"⚠️ Stripping bytes object of size {len(data)} during flattening.")
            return f"<bytes content stripped (size: {len(data)})>"
    
        # Handle primitive types
        if isinstance(data, (str, int, float, bool)):
            return data
    
        # Handle datetime objects
        if isinstance(data, (datetime, )):
            return data.isoformat()
    
        # Handle Enum
        if isinstance(data, Enum):
            return data.value
    
        # Handle dataclasses
        if is_dataclass(data):
            return self._flatten_for_firestore(asdict(data))
    
        # Handle dictionaries
        if isinstance(data, dict):
            return {
                str(key): self._flatten_for_firestore(value)
                for key, value in data.items()
            }
    
        # Handle lists and tuples
        if isinstance(data, (list, tuple)):
            return [self._flatten_for_firestore(item) for item in data]
    
        # Handle objects with to_dict() method
        if hasattr(data, 'to_dict') and callable(getattr(data, 'to_dict')):
            return self._flatten_for_firestore(data.to_dict())
    
        # Handle objects with __dict__ attribute
        if hasattr(data, '__dict__'):
            return self._flatten_for_firestore(data.__dict__)
    
        # Last resort: convert to string
        logger.warning(f"⚠️ Converting unsupported type {type(data)} to string")
        return str(data)

    def _flatten_analysis_for_api(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flatten analysis results for safe API/Firestore storage.
        Converts all complex objects (dataclasses, enums) to primitive types.
        """
        return {
            "session_id": response.get("session_id"),
            "confidence": response.get("confidence", 0),
            "can_generate_offer": response.get("can_generate_offer", False),
            "mode": response.get("mode"),
            "response": response.get("response"),
            
            # Flatten nested structures
            "analysis_summary": self._flatten_for_firestore(
                response.get("analysis_summary", {})
            ),
            "file_references": self._flatten_for_firestore(
                response.get("file_references", [])
            ),
            
            # Simple fields
            "next_questions": response.get("next_questions", []),
            "ml_enhanced": response.get("ml_enhanced", False),
            "processing_time_ms": response.get("processing_time_ms", 0),
        }

    # ❌ DELETED OLD FORMATTING METHODS (replaced by ResponseBuilder)
    # _format_gap_analysis_result, _format_rfp_context, _format_conflicts_section,
    # _format_conflicts_section_from_dict, _create_response_content,
    # _create_excellent_confidence_response, _create_high_confidence_response,
    # _create_medium_confidence_response, _create_low_confidence_response

    async def _create_session_context(self) -> AgentContext:
        """Create new session context"""
        session_id = str(uuid.uuid4())

        context = AgentContext(
            session_id=session_id,
            current_mode=AgentMode.TECHNICAL_ANALYSIS
        )
        # The session document in Firestore is created in analyze_project
        return context

    async def _load_session_context(self, session_id: str) -> AgentContext:
        """
        Load session context by fetching metadata from the main document and
        detailed data from 'messages' and 'file_analyses' subcollections.
        """
        context = AgentContext(
            session_id=session_id,
            current_mode=AgentMode.CONVERSATION
        )

        if not self.session_manager:
            logger.warning("Session manager not available, cannot load context.")
            return context

        try:
            # Load the main session document for metadata
            session = await self.session_manager.get_session(session_id)
            if not session:
                logger.warning(f"No session data found for session_id: {session_id}")
                return context

            # Restore metadata from the main session object
            context.confidence_score = session.confidence_score
            
            # Load conversation history from the 'messages' subcollection
            conversation_history = await self.firestore_service.load_messages_from_subcollection(session_id)
            context.conversation_history = sorted(conversation_history, key=lambda x: x.get('timestamp'))

            # Load file analyses from the 'file_analyses' subcollection
            file_analyses_docs = await self.firestore_service.load_all_file_analyses(session_id)
            
            # Consolidate file analyses into the in-memory context.analysis_results object
            consolidated_files = self._consolidate_file_analyses(file_analyses_docs)
            context.analysis_results = { "file_analysis": consolidated_files }
            
            # FIX [AttributeError]: Access dynamic data from the session's underlying 'data' dictionary
            # The UnifiedSession object has a hybrid structure.
            if hasattr(session, 'data') and isinstance(session.data, dict):
                context.project_data = session.data.get("project_data", {})
                context.missing_data = session.data.get("missing_data", [])
            else:
                # Fallback for safety, though the primary issue is the access pattern
                context.project_data = {}
                context.missing_data = []


            logger.info("="*80)
            logger.info("📊 DEBUG: Session Context Loaded from Subcollections")
            logger.info(f"File analyses loaded: {len(file_analyses_docs)}")
            logger.info(f"Conversation history entries: {len(context.conversation_history)}")
            logger.info(f"Project data keys: {list(context.project_data.keys())}")
            logger.info("="*80)

        except Exception as e:
            logger.error(f"Failed to load session data for {session_id}: {e}", exc_info=True)

        return context


    async def _update_session_context(self, context: AgentContext):
        """
        Update the main session document with metadata ONLY.
        Messages and file analyses are saved directly to subcollections elsewhere.
        """
        if not self.session_manager:
            logger.warning("Session manager not available, cannot update session.")
            return
        
        try:
            session_metadata = {
                "last_activity": datetime.now(timezone.utc),
                "confidence_score": context.confidence_score,
                "can_generate_offer": context.confidence_score >= self.offer_generation_threshold,
                "missing_data": context.missing_data,
                "project_data_summary": { # Store only a small summary, not the whole object
                    "total_area": context.project_data.get("total_area"),
                    "total_rooms": context.project_data.get("total_rooms"),
                }
            }
            
            await self.firestore_service.update_document(
                collection='engineer_chat_sessions',
                document_id=context.session_id,
                update_data=session_metadata
            )
            logger.info(f"✅ Session {context.session_id} metadata updated successfully.")
        
        except Exception as e:
            logger.error(f"❌ Failed to update session context metadata: {e}", exc_info=True)


    async def _generate_next_questions(self, context: AgentContext) -> List[str]:
        """Generate contextual next questions with specification awareness"""
        questions = []

        # Check if we have specification data
        dxf_analysis = context.analysis_results.get("file_analysis", {}).get("dxf_analysis")
        is_specification_sheet = (dxf_analysis and
                                dxf_analysis.get("dxf_analysis", {}).get("document_type") == "specification_sheet")

        if is_specification_sheet:
            # Specification-specific questions
            spec_analysis = dxf_analysis.get("dxf_analysis", {}).get("specification_analysis", {})

            if not context.project_data.get("total_area"):
                questions.append("Care sunt suprafețele spațiilor pentru care se aplică aceste specificații?")

            if spec_analysis.get("materials_count", 0) > 0:
                questions.append("Ce cantități sunt necesare pentru materialele specificate?")

            if spec_analysis.get("wall_types_count", 0) > 0:
                questions.append("Câți metri liniari de pereți sunt necesari pentru fiecare tip specificat?")

            if not context.project_data.get("timeline_requirements"):
                questions.append("Există restricții de timp pentru implementarea acestor specificații?")

            if not context.project_data.get("site_conditions"):
                questions.append("Care sunt condițiile șantierului unde se vor aplica specificațiile?")

        else:
            # Standard project questions (existing logic)
            if not context.project_data.get("total_area"):
                questions.append("Care este suprafața totală construită în metri pătrați?")

            if not context.project_data.get("total_rooms"):
                questions.append("Câte camere are proiectul total?")

            # Budget and timeline
            if not context.project_data.get("budget"):
                questions.append("Aveți un buget estimat pentru acest proiect?")

            # Quality level
            if not context.project_data.get("quality_level"):
                questions.append("Ce nivel de finisaje doriți: standard, premium sau lux?")

            # Timeline requirements
            if not context.project_data.get("timeline_requirements"):
                questions.append("Până când trebuie să fie gata proiectul?")

        # Use ML to generate better questions if available
        if self.ml_enabled and self.pattern_matcher:
            try:
                ml_questions = await self.romanian_processor.generate_clarification_questions(
                    missing_data=context.missing_data,
                    project_context=context.project_data
                )
                questions.extend(ml_questions[:2])  # Add top 2 ML-generated questions
            except Exception as e:
                logger.warning(f"ML question generation failed: {e}")

        return questions[:3]  # Return max 3 questions

    def _create_analysis_summary(self, analysis: Dict[str, Any], context: AgentContext) -> Dict[str, Any]:
        """Create analysis summary for response"""
        return {
            "confidence_score": context.confidence_score,
            "files_processed": analysis.get("file_analysis", {}).get("files_processed", 0),
            "dxf_analysis_available": analysis.get("file_analysis", {}).get("dxf_analysis") is not None,
            "ml_enhanced": analysis.get("ml_insights") is not None,
            "missing_data_count": len(context.missing_data),
            "conversation_length": len(context.conversation_history)
        }

    def _build_project_summary(self, context: AgentContext) -> str:
        """Build project summary for LLM context"""
        summary_parts = []

        if context.project_data.get("total_area"):
            summary_parts.append(f"Suprafață: {context.project_data['total_area']} mp")

        if context.project_data.get("total_rooms"):
            summary_parts.append(f"Camere: {context.project_data['total_rooms']}")

        if context.project_data.get("budget"):
            summary_parts.append(f"Buget: {context.project_data['budget']} RON")

        if context.romanian_context.get("user_requirements"):
            summary_parts.append(f"Cerințe: {context.romanian_context['user_requirements'][:100]}...")

        return "; ".join(summary_parts) if summary_parts else "Proiect în analiză"

    def _generate_initial_response(
        self,
        response: UnifiedResponse,
        file_analysis: Dict[str, Any]
    ) -> str:
        """Generate initial response for file upload"""

        files_count = file_analysis.get("files_processed", 0)
        has_dxf = file_analysis.get("dxf_analysis") is not None

        initial_msg = f"✅ {files_count} fișiere analizate cu succes"
        if has_dxf:
            initial_msg += " (cu analiză DXF automată)"

        return initial_msg

    # =========================================================================
    # PRIVATE METHODS - TEXT EXTRACTION
    # =========================================================================

    def _extract_area_from_text(self, text: str) -> Optional[float]:
        """Extract area from Romanian text"""
        import re
        patterns = [
            r'(\d+(?:[.,]\d+)?)\s*(?:mp|m2|m²|metri\s*pătrați)',
            r'(\d+(?:[.,]\d+)?)\s*(?:square\s*m|sqm)',
            r'suprafața?\s*(?:de\s*)?(\d+(?:[.,]\d+)?)',
            r'(\d+(?:[.,]\d+)?)\s*m\s*pătrați',
            r'arie\s*(?:de\s*)?(\d+(?:[.,]\d+)?)'
        ]

        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                try:
                    return float(match.group(1).replace(',', '.'))
                except ValueError:
                    continue
        return None

    def _extract_room_count_from_text(self, text: str) -> Optional[int]:
        """Extract room count from Romanian text"""
        import re
        patterns = [
            r'(\d+)\s*(?:camere?|dormitoare?|rooms?)',
            r'(?:camere?|rooms?)\s*(\d+)',
            r'(\d+)\s*(?:cam|rom)',
            r'(\d+)\s*spații',
            r'numărul?\s*(?:de\s*)?camere?\s*(?:este\s*)?(\d+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue
        return None

    def _extract_budget_from_text(self, text: str) -> Optional[float]:
        """Extract budget from Romanian text"""
        import re
        patterns = [
            r'buget\s*(?:de\s*)?(\d+(?:[.,]\d+)?)\s*(?:k|mii|mil|euro|ron|lei)',
            r'(\d+(?:[.,]\d+)?)\s*(?:k|mii|mil)\s*(?:euro|ron|lei)',
            r'până\s*la\s*(\d+(?:[.,]\d+)?)\s*(?:k|mii|mil|euro|ron|lei)'
        ]

        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                try:
                    value = float(match.group(1).replace(',', '.'))
                    # Handle multipliers
                    if 'k' in match.group(0) or 'mii' in match.group(0):
                        value *= 1000
                    elif 'mil' in match.group(0):
                        value *= 1000000
                    return value
                except ValueError:
                    continue
        return None

    def _is_offer_request(self, message: str) -> bool:
        """Check if message is requesting offer generation"""
        message_lower = message.lower()
        offer_keywords = [
            "generează oferta", "genereaza oferta",
            "vreau oferta", "pot avea oferta",
            "oferta finala", "oferta finală",
            "generate offer", "make offer"
        ]
        return any(keyword in message_lower for keyword in offer_keywords)

    # =========================================================================
    # PRIVATE METHODS - COST AND TIMELINE ESTIMATION
    # =========================================================================

    async def _calculate_cost_estimate(
        self,
        context: AgentContext,
        historical_insights: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Calculate cost estimate using historical data when available"""

        area = context.project_data.get("total_area", 50)
        rooms = context.project_data.get("total_rooms", 2)

        # Use historical data if available
        if historical_insights and len(historical_insights) > 0:
            # Calculate average cost per sqm from similar projects
            historical_costs = [p.get("cost_per_sqm", 0) for p in historical_insights if p.get("cost_per_sqm", 0) > 0]

            if historical_costs:
                avg_historical_cost = sum(historical_costs) / len(historical_costs)
                base_cost_per_sqm = int(avg_historical_cost)
                confidence_note = f"\n*Estimare bazată pe {len(historical_costs)} proiecte similare finalizate (±10% variație)*"
            else:
                base_cost_per_sqm = 800
                confidence_note = ""
        else:
            base_cost_per_sqm = 800
            confidence_note = "\n*Estimare preliminară - va fi ajustată după evaluare tehnică detaliată*"

        # Calculate costs
        area_cost = area * base_cost_per_sqm
        room_adjustment = rooms * 500
        total_before_vat = int(area_cost + room_adjustment)

        materials = int(total_before_vat * 0.6)
        labor = int(total_before_vat * 0.25)
        transport = int(total_before_vat * 0.15)
        vat = int(total_before_vat * 0.19)
        total_with_vat = total_before_vat + vat

        return {
            "materials": materials,
            "labor": labor,
            "transport": transport,
            "subtotal": total_before_vat,
            "vat": vat,
            "total": total_with_vat,
            "base_cost_per_sqm": base_cost_per_sqm,
            "confidence_note": confidence_note
        }

    async def _calculate_timeline_estimate(
        self,
        context: AgentContext,
        historical_insights: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Calculate timeline using historical completion data"""

        area = context.project_data.get("total_area", 50)
        rooms = context.project_data.get("total_rooms", 2)

        # Adjust timeline based on historical data
        if historical_insights and len(historical_insights) > 0:
            historical_durations = [p.get("duration_days", 0) for p in historical_insights if p.get("duration_days", 0) > 0]

            if historical_durations:
                avg_duration = sum(historical_durations) / len(historical_durations)
                # Use historical average as base
                total_days = int(avg_duration * (area / 50)) # Adjust for current project size
                historical_note = f"\n*Timeline bazat pe durata reală a {len(historical_durations)} proiecte similare*"
            else:
                # Fallback to formula
                base_days = 20
                area_days = max(1, int(area / 25))
                room_days = rooms * 2
                total_days = base_days + area_days + room_days + 8
                historical_note = ""
        else:
            base_days = 20
            area_days = max(1, int(area / 25))
            room_days = rooms * 2
            total_days = base_days + area_days + room_days + 8
            historical_note = ""

        preparation = 3
        finishing = 5
        execution = total_days - preparation - finishing

        return {
            "preparation": preparation,
            "execution": execution,
            "finishing": finishing,
            "total_days": total_days,
            "historical_note": historical_note
        }

    def _build_materials_recommendations(self, historical_insights: List[Dict[str, Any]]) -> str:
        """Build materials section based on historical project data"""

        if not historical_insights or len(historical_insights) == 0:
            return """
**Materiale incluse:**
- Materiale conform standardelor românești SR EN
- Transport la șantier inclus
- Garanție materiale: 2 ani"""

        # Extract common materials from historical projects
        material_counts = {}
        for project in historical_insights:
            materials = project.get("materials_used", [])
            for material in materials:
                material_counts[material] = material_counts.get(material, 0) + 1

        # Get most common materials (used in >50% of projects)
        threshold = len(historical_insights) / 2
        recommended_materials = [m for m, count in material_counts.items() if count >= threshold]

        materials_text = """
**Materiale incluse:**
- Materiale conform standardelor românești SR EN
- Transport la șantier inclus
- Garanție materiale: 2 ani"""

        if recommended_materials:
            materials_text += f"\n\n**Materiale recomandate (bazate pe proiecte similare):**"
            for material in recommended_materials[:5]: # Top 5
                materials_text += f"\n- {material}"

        return materials_text

    async def _apply_conversation_ml(
        self,
        message: str,
        context: AgentContext
    ) -> Optional[Dict[str, Any]]:
        """Apply ML insights for conversation enhancement"""
        if not self.ml_enabled:
            return None

        try:
            # Use pattern matcher to analyze conversation patterns
            if self.pattern_matcher:
                # Analyze message for patterns
                pattern_result = await self.pattern_matcher.analyze_project_patterns(
                    context.project_data,
                    message
                )

                if pattern_result.confidence > 0.5:
                    return {
                        "pattern_insights": pattern_result,
                        "confidence_boost": pattern_result.confidence * 0.05
                    }

            return None

        except Exception as e:
            logger.error(f"❌ Conversation ML error: {e}")
            return None

    async def _load_ocr_data_for_file(self, file_id: str) -> Optional[Dict[str, Any]]:
        """
        Load full OCR data for a file when needed (lazy loading)
        
        Args:
            file_id: Unique identifier for the OCR result
            
        Returns:
            Full OCR data dictionary or None if not found
        """
        try:
            logger.info(f"📂 Loading OCR data for file_id: {file_id}")
            
            # ✅ NEW CODE - Use OCR Storage Service:
            ocr_result = await self.ocr_storage.load_ocr_result(file_id)

            if ocr_result:
                logger.info(f"✅ OCR data loaded successfully for {file_id}")
                # Convert OCRResult object to dict if needed
                return ocr_result.to_dict()
            else:
                logger.warning(f"⚠️ No OCR data found for {file_id}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to load OCR data for {file_id}: {e}", exc_info=True)
            return None