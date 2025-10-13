"""
DXF Analyzer for DemoPLAN Unified
Geometry-first approach with multi-language support
Handles residential and commercial projects across EU languages
Enhanced with CSV/XLSX/JSON processing for training data
"""

import logging
import math
import re
import io
import os
from typing import Dict, List, Any, Optional, Set, Tuple, Union, BinaryIO
from dataclasses import dataclass, field
from enum import Enum
import PyPDF2
import pdfplumber
import pandas as pd
import json
# Add these imports at the top of dxf_analyzer.py
from src.processors.rfp_parser import RFPParser, RFPStructure, is_rfp_document
from typing import Tuple, Dict, List, Any, Optional

try:
    import ezdxf
    from ezdxf.document import Drawing
    from ezdxf.recover import read as recover_read
    DXF_AVAILABLE = True
except ImportError:
    DXF_AVAILABLE = False
    Drawing = None

# Modified logger for consistency with the request
logger = logging.getLogger("demoplan.processors.pdf")


class DocumentType(Enum):
    FLOOR_PLAN = "floor_plan"
    SPECIFICATION_SHEET = "specification_sheet"
    TECHNICAL_DETAIL = "technical_detail"
    MIXED_DOCUMENT = "mixed_document"
    UNKNOWN = "unknown"

class ProjectType(Enum):
    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    INDUSTRIAL = "industrial"
    MIXED_USE = "mixed_use"
    UNKNOWN = "unknown"

@dataclass
class SpatialBoundary:
    """Represents a spatial boundary (potential room)"""
    boundary_id: str
    vertices: List[List[float]]
    area: float
    centroid: List[float]
    confidence: float
    associated_texts: List[str] = None

@dataclass
class FilteredText:
    """Filtered and classified text entity"""
    text: str
    location: List[float]
    relevance_score: float
    classification: str  # 'room_label', 'area_label', 'dimension', 'specification', 'noise'
    language: Optional[str] = None
    room_type: Optional[str] = None

@dataclass
class DocumentClassification:
    """Document type and project classification"""
    document_type: DocumentType
    project_type: ProjectType
    primary_language: Optional[str]
    confidence: float
    indicators: List[str]

@dataclass
class OptimizedFloorplanData:
    """Enhanced floorplan analysis results"""
    total_rooms: int
    total_area: float
    room_breakdown: List[Dict[str, Any]]
    has_dimensions: bool
    has_electrical: bool
    has_hvac: bool
    scale_factor: float
    confidence_score: float
    document_classification: DocumentClassification
    spatial_boundaries: List[SpatialBoundary]
    filtered_texts: List[FilteredText]

@dataclass
class SpecificationData:
    """Specification sheet extracted data"""
    wall_types: List[Dict[str, Any]] = field(default_factory=list)
    material_specifications: List[Dict[str, Any]] = field(default_factory=list)
    finishing_requirements: List[Dict[str, Any]] = field(default_factory=list)
    safety_requirements: List[Dict[str, Any]] = field(default_factory=list)
    mep_provisions: List[Dict[str, Any]] = field(default_factory=list)
    construction_methods: List[Dict[str, Any]] = field(default_factory=list)
    total_specifications: int = 0
    specification_confidence: float = 0.0

@dataclass
class PDFAnalysisResult:
    """PDF analysis result with OCR metadata"""
    extracted_text: str
    page_count: int
    construction_specs: List[str] = field(default_factory=list)
    material_references: List[str] = field(default_factory=list)
    regulatory_info: List[str] = field(default_factory=list)
    confidence: float = 0.0
    
    # OCR fields
    ocr_used: bool = False
    tables_extracted: List[Dict] = field(default_factory=list)
    handwriting_detected: bool = False
    entity_extractions: Dict[str, Any] = field(default_factory=dict)
    processing_cost: float = 0.0
    text_density_score: float = 0.0
    ocr_file_id: Optional[str] = None # ✅ NEW: Store file_id reference
    ocr_result: Optional['OCRResult'] = None  # Full OCR result for storage

@dataclass
class TXTAnalysisResult:
    """TXT analysis results"""
    content: str
    construction_keywords: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    client_preferences: List[str] = field(default_factory=list)
    confidence: float = 0.0

@dataclass
class TabularAnalysisResult:
    """Tabular (CSV/Excel) analysis results"""
    file_type: str
    filename: str
    columns: List[str]
    row_count: int
    data_preview: List[Dict[str, Any]]
    text_representation: str
    confidence: float = 0.0

@dataclass
class JSONAnalysisResult:
    """JSON analysis results"""
    filename: str
    data: Any
    text_representation: str
    confidence: float = 0.0

@dataclass
class UnifiedDocumentResult:
    """Combined analysis from all document types"""
    document_type: str
    dxf_analysis: Optional[Dict[str, Any]] = None
    pdf_analysis: Optional[PDFAnalysisResult] = None
    txt_analysis: Optional[TXTAnalysisResult] = None
    tabular_analysis: Optional[TabularAnalysisResult] = None
    json_analysis: Optional[JSONAnalysisResult] = None
    rfp_analysis: Optional[RFPStructure] = None  # ✅ ADD THIS LINE
    combined_confidence: float = 0.0
    integrated_specs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HVACComponent:
    """HVAC component details"""
    component_id: str
    component_type: str  # 'ac_unit', 'diffuser', 'grille', 'duct', 'thermostat'
    location: List[float]
    capacity_kw: Optional[float] = None
    model: Optional[str] = None
    room_association: Optional[str] = None
    specifications: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ElectricalComponent:
    """Electrical component details"""
    component_id: str
    component_type: str  # 'outlet', 'switch', 'light_fixture', 'panel', 'junction_box'
    location: List[float]
    circuit: Optional[str] = None
    power_rating: Optional[str] = None
    quantity: int = 1
    room_association: Optional[str] = None
    specifications: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DoorWindow:
    """Door or window details"""
    component_id: str
    component_type: str  # 'door', 'window', 'sliding_door', 'glass_door'
    location: List[float]
    width: float
    height: float
    material: Optional[str] = None
    opening_type: Optional[str] = None  # 'swing', 'sliding', 'casement', 'fixed'
    room_association: Optional[str] = None
    specifications: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DimensionEntity:
    """Dimension annotation with measurements"""
    dimension_id: str
    measurement_value: float
    dimension_type: str  # 'linear', 'angular', 'radial', 'diameter'
    location: List[float]
    text_override: Optional[str] = None
    associated_element: Optional[str] = None

@dataclass
class PlumbingComponent:
    """Plumbing fixture details"""
    component_id: str
    component_type: str  # 'sink', 'toilet', 'shower', 'bathtub', 'drain', 'pipe'
    location: List[float]
    fixture_type: Optional[str] = None
    room_association: Optional[str] = None
    specifications: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FlooringInfo:
    """Flooring type and surface information"""
    flooring_id: str
    flooring_type: str  # 'tile', 'wood', 'carpet', 'vinyl', 'concrete', 'epoxy'
    area: float
    material_specification: Optional[str] = None
    room_association: Optional[str] = None
    finish_level: Optional[str] = None

@dataclass
class CeilingInfo:
    """Ceiling information"""
    ceiling_id: str
    height: Optional[float] = None
    ceiling_type: Optional[str] = None  # 'suspended', 'gypsum', 'exposed', 'acoustic'
    room_association: Optional[str] = None
    specifications: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EntityInventory:
    """Complete entity-level inventory from DXF"""
    hvac_components: List[HVACComponent] = field(default_factory=list)
    electrical_components: List[ElectricalComponent] = field(default_factory=list)
    doors_windows: List[DoorWindow] = field(default_factory=list)
    dimensions: List[DimensionEntity] = field(default_factory=list)
    plumbing_components: List[PlumbingComponent] = field(default_factory=list)
    flooring_info: List[FlooringInfo] = field(default_factory=list)
    ceiling_info: List[CeilingInfo] = field(default_factory=list)
    text_annotations: List[Dict[str, Any]] = field(default_factory=list)

class SpecificationExtractor:
    """Extracts construction specifications from technical documents"""
    
    def __init__(self):
        # Wall type patterns commonly used in Romanian construction
        self.wall_type_patterns = [
            r'P\d+',  # Partition walls P1, P2, etc.
            r'S\d+',  # Structural walls S1, S2, etc.
            r'EI\d+', # Fire rated walls EI120, etc.
            r'REI\d+' # Load-bearing fire walls
        ]
        
        # Material patterns
        self.material_patterns = {
            'gypsum_panels': [r'GK\s*\d+', r'gips-carton', r'plăci\s+gips'],
            'steel_profiles': [r'CW\s*\d+', r'UW\s*\d+', r'profile\s+metalice'],
            'insulation': [r'vată\s+minerală', r'polistiren', r'izolație'],
            'finishing': [r'vopsea', r'gresie', r'faianță', r'parchet'],
            'fasteners': [r'șuruburi', r'dibluri', r'ancore']
        }
        
        # Romanian construction finishing terms
        self.finishing_patterns = [
            r'Brillux\s+\w+',  # Paint brands
            r'vopsea\s+\w+',   # Paint types
            r'gresie\s+\d+x\d+', # Tile sizes
            r'faianță\s+\d+x\d+',
            r'parchet\s+\w+',
            r'laminat\s+\w+'
        ]
        
        # Safety and compliance patterns
        self.safety_patterns = [
            r'EI\d+',  # Fire resistance
            r'REI\d+',
            r'ignifug',
            r'rezistență\s+la\s+foc',
            r'evacuare',
            'siguranță'
        ]

    def extract_specifications(self, filtered_texts: List[FilteredText]) -> SpecificationData:
        """Extract construction specifications from filtered texts"""
        
        spec_data = SpecificationData()
        
        # Process all relevant texts
        specification_texts = [
            ft for ft in filtered_texts 
            if ft.classification in ['specification', 'area_label', 'room_label']
            and ft.relevance_score > 0.2
        ]
        
        spec_data.total_specifications = len(specification_texts)
        
        for text in specification_texts:
            text_content = text.text.lower()
            
            # Extract wall types
            wall_types = self._extract_wall_types(text.text, text.location)
            spec_data.wall_types.extend(wall_types)
            
            # Extract materials
            materials = self._extract_materials(text.text, text.location)
            spec_data.material_specifications.extend(materials)
            
            # Extract finishing requirements
            finishes = self._extract_finishes(text.text, text.location)
            spec_data.finishing_requirements.extend(finishes)
            
            # Extract safety requirements
            safety = self._extract_safety_requirements(text.text, text.location)
            spec_data.safety_requirements.extend(safety)
            
            # Extract MEP provisions
            mep = self._extract_mep_provisions(text.text, text.location)
            spec_data.mep_provisions.extend(mep)
        
        # Calculate specification confidence
        spec_data.specification_confidence = self._calculate_specification_confidence(spec_data)
        
        return spec_data

    def _extract_wall_types(self, text: str, location: List[float]) -> List[Dict[str, Any]]:
        """Extract wall type specifications"""
        wall_types = []
        
        for pattern in self.wall_type_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                wall_type = {
                    'type_code': match.group(0).upper(),
                    'location': location,
                    'context': text,
                    'classification': 'wall_type'
                }
                
                # Try to extract additional details from surrounding text
                context_start = max(0, match.start() - 50)
                context_end = min(len(text), match.end() + 50)
                context = text[context_start:context_end]
                
                # Look for thickness
                thickness_match = re.search(r'(\d+)\s*mm', context)
                if thickness_match:
                    wall_type['thickness_mm'] = int(thickness_match.group(1))
                
                # Look for fire rating
                fire_match = re.search(r'EI(\d+)', context)
                if fire_match:
                    wall_type['fire_rating'] = f"EI{fire_match.group(1)}"
                
                wall_types.append(wall_type)
        
        return wall_types

    def _extract_materials(self, text: str, location: List[float]) -> List[Dict[str, Any]]:
        """Extract material specifications"""
        materials = []
        
        for material_type, patterns in self.material_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    material = {
                        'material_type': material_type,
                        'specification': match.group(0),
                        'location': location,
                        'context': text,
                        'classification': 'material'
                    }
                    
                    # Extract dimensions if present
                    dim_pattern = r'(\d+(?:[.,]\d+)?)\s*[xX×]\s*(\d+(?:[.,]\d+)?)'
                    dim_match = re.search(dim_pattern, text[max(0, match.start()-20):match.end()+20])
                    if dim_match:
                        material['dimensions'] = f"{dim_match.group(1)}x{dim_match.group(2)}"
                    
                    materials.append(material)
        
        return materials

    def _extract_finishes(self, text: str, location: List[float]) -> List[Dict[str, Any]]:
        """Extract finishing requirements"""
        finishes = []
        
        for pattern in self.finishing_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                finish = {
                    'finish_type': self._classify_finish_type(match.group(0)),
                    'specification': match.group(0),
                    'location': location,
                    'context': text,
                    'classification': 'finishing'
                }
                finishes.append(finish)
        
        return finishes

    def _extract_safety_requirements(self, text: str, location: List[float]) -> List[Dict[str, Any]]:
        """Extract safety and compliance requirements"""
        safety_reqs = []
        
        for pattern in self.safety_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                safety = {
                    'safety_type': self._classify_safety_type(match.group(0)),
                    'requirement': match.group(0),
                    'location': location,
                    'context': text,
                    'classification': 'safety'
                }
                safety_reqs.append(safety)
        
        return safety_reqs

    def _extract_mep_provisions(self, text: str, location: List[float]) -> List[Dict[str, Any]]:
        """Extract MEP (Mechanical, Electrical, Plumbing) provisions"""
        mep_provisions = []
        
        mep_patterns = [
            r'priza\s+\w+',
            r'întrerupător',
            r'ventilație',
            r'climatizare', 
            r'încălzire',
            r'instalații\s+electrice',
            r'instalații\s+sanitare'
        ]
        
        for pattern in mep_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                mep = {
                    'mep_type': self._classify_mep_type(match.group(0)),
                    'specification': match.group(0),
                    'location': location,
                    'context': text,
                    'classification': 'mep'
                }
                mep_provisions.append(mep)
        
        return mep_provisions

    def _classify_finish_type(self, specification: str) -> str:
        """Classify finish type from specification"""
        spec_lower = specification.lower()
        if any(paint in spec_lower for paint in ['vopsea', 'brillux']):
            return 'paint'
        elif any(tile in spec_lower for tile in ['gresie', 'faianță']):
            return 'ceramic_tiles'
        elif 'parchet' in spec_lower:
            return 'hardwood'
        elif 'laminat' in spec_lower:
            return 'laminate'
        else:
            return 'other'

    def _classify_safety_type(self, requirement: str) -> str:
        """Classify safety requirement type"""
        req_lower = requirement.lower()
        if 'ei' in req_lower or 'rei' in req_lower:
            return 'fire_resistance'
        elif 'evacuare' in req_lower:
            return 'evacuation'
        elif 'siguranță' in req_lower:
            return 'general_safety'
        else:
            return 'compliance'

    def _classify_mep_type(self, specification: str) -> str:
        """Classify MEP provision type"""
        spec_lower = specification.lower()
        if any(elec in spec_lower for elec in ['priza', 'întrerupător', 'electrice']):
            return 'electrical'
        elif any(hvac in spec_lower for hvac in ['ventilație', 'climatizare', 'încălzire']):
            return 'hvac'
        elif 'sanitare' in spec_lower:
            return 'plumbing'
        else:
            return 'general_mep'

    def _calculate_specification_confidence(self, spec_data: SpecificationData) -> float:
        """Calculate confidence in specification extraction"""
        confidence = 0.3  # Base confidence
        
        # Boost based on extracted specifications
        if spec_data.wall_types:
            confidence += len(spec_data.wall_types) * 0.1
        if spec_data.material_specifications:
            confidence += len(spec_data.material_specifications) * 0.08
        if spec_data.finishing_requirements:
            confidence += len(spec_data.finishing_requirements) * 0.06
        if spec_data.safety_requirements:
            confidence += len(spec_data.safety_requirements) * 0.05
        
        return min(confidence, 0.95)  # Cap at 95%

class PDFProcessor:
    """Processes PDF documents for construction information"""
    
    def __init__(self):
        self.construction_keywords = [
            'specificații', 'cerințe', 'materiale', 'finisaje', 'instalații',
            'specifications', 'requirements', 'materials', 'finishes',
            'pereți', 'pardoseală', 'tavan', 'uși', 'ferestre'
        ]
        
        self.regulatory_keywords = [
            'normativ', 'standard', 'SR EN', 'STAS', 'autorizație',
            'certificat', 'aprobare', 'ISU', 'DSP', 'ITM'
        ]
        
        self.ocr_service = None  # Lazy load to avoid startup issues
        self.ocr_enabled = os.getenv("OCR_ENABLED", "true").lower() == "true"
        self.ocr_min_text_threshold = int(os.getenv("OCR_MIN_TEXT_THRESHOLD", "100"))
        
        logger.info(f"📄 PDF Processor initialized (OCR: {'enabled' if self.ocr_enabled else 'disabled'})")

    def analyze_pdf(self, file_content: bytes, filename: str = None) -> PDFAnalysisResult:
        """Analyze PDF content for construction information with OCR support

        Week 3 Enhancement: Added table extraction for room schedules.
        """
        try:
            # Reset OCR metadata
            self._last_ocr_result = None
            self._last_ocr_file_id = None # ✅ NEW: Reset file_id
            
            # Pass filename to extraction
            extracted_text = self._extract_text_from_pdf(file_content, filename=filename)
            
            if not extracted_text:
                logger.warning("⚠️ No text extracted from PDF")
                return PDFAnalysisResult(
                    extracted_text="",
                    page_count=0,
                    confidence=0.0,
                    ocr_used=False
                )
            
            # Analyze extracted text
            construction_specs = self._extract_construction_specs(extracted_text)
            material_references = self._extract_material_references(extracted_text)
            regulatory_info = self._extract_regulatory_info(extracted_text)

            # ✅ NEW: Extract tables (room schedules)
            tables_extracted = self._extract_tables_from_pdf(file_content)
            
            # Calculate confidence
            confidence = self._calculate_pdf_confidence(
                extracted_text, construction_specs, material_references
            )
            
            # Prepare OCR metadata
            ocr_metadata = self._prepare_ocr_metadata()
            page_count = self._count_pdf_pages(file_content)
            
            result = PDFAnalysisResult(
                extracted_text=extracted_text[:5000],  # Limit for storage
                page_count=page_count,
                construction_specs=construction_specs,
                material_references=material_references,
                regulatory_info=regulatory_info,
                confidence=confidence,
                ocr_used=ocr_metadata.get('ocr_used', False),
                tables_extracted=tables_extracted or ocr_metadata.get('tables_extracted', []),
                handwriting_detected=ocr_metadata.get('handwriting_detected', False),
                entity_extractions=ocr_metadata.get('entity_extractions', {}),
                processing_cost=ocr_metadata.get('processing_cost', 0.0),
                text_density_score=ocr_metadata.get('text_density_score', 0.0)
            )

            # ✅ NEW: Attach file_id to result if OCR was used
            if hasattr(self, '_last_ocr_file_id') and self._last_ocr_file_id:
                result.ocr_file_id = self._last_ocr_file_id
            
            # Attach full OCR result for storage
            if hasattr(self, '_last_ocr_result') and self._last_ocr_result:
                result.ocr_result = self._last_ocr_result
            
            return result
            
        except Exception as e:
            logger.error(f"❌ PDF analysis failed: {e}", exc_info=True)
            return PDFAnalysisResult(
                extracted_text="",
                page_count=0,
                confidence=0.0,
                ocr_used=False
            )

    def _extract_text_from_pdf(self, file_content: bytes, filename: str = None) -> str:
        """Extract text from PDF using multiple methods with OCR fallback"""
        text = ""
        
        try:
            # Step 1: Try pdfplumber first
            with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e_pdfplumber:
            logger.warning(f"pdfplumber extraction failed: {e_pdfplumber}")
            try:
                # Step 2: Fallback to PyPDF2
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            except Exception as e_pypdf2:
                logger.error(f"PyPDF2 extraction also failed: {e_pypdf2}")
        
        # Step 3: Check if OCR is needed
        if self.ocr_enabled and self._needs_ocr(text, file_content):
            logger.info("🔍 Low text density detected - initiating OCR fallback")
            try:
                # Pass filename to OCR
                ocr_text = self._apply_ocr_fallback(file_content, filename=filename)
                if ocr_text and len(ocr_text) > len(text):
                    logger.info(f"✅ OCR extracted {len(ocr_text)} chars vs {len(text)} from traditional methods")
                    text = ocr_text
            except Exception as e_ocr:
                logger.error(f"❌ OCR fallback failed: {e_ocr}")
        
        return text.strip()

    def _extract_tables_from_pdf(self, file_content: bytes) -> List[Dict[str, Any]]:
        """
        Extract tables from PDF using pdfplumber.
        
        Week 3 Enhancement: Detects room schedule tables by headers.
        Supports both Romanian and English headers.
        
        Returns:
            List of table dictionaries with headers and rows
        """
        tables_extracted = []
        
        try:
            with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        # Extract tables from page
                        page_tables = page.extract_tables()
                        
                        if not page_tables:
                            continue
                        
                        for table_idx, table in enumerate(page_tables):
                            if not table or len(table) < 2:  # Need header + at least 1 row
                                continue
                            
                            # Parse table structure
                            headers = [str(cell).strip().lower() if cell else "" for cell in table[0]]
                            rows = [[str(cell).strip() if cell else "" for cell in row] for row in table[1:]]
                            
                            # Identify if this is a room schedule table
                            is_room_table = self._is_room_schedule_table(headers)
                            
                            if is_room_table or len(headers) >= 2:  # Store if room table OR multi-column
                                tables_extracted.append({
                                    "page": page_num + 1,
                                    "table_index": table_idx,
                                    "headers": headers,
                                    "rows": rows,
                                    "is_room_schedule": is_room_table,
                                    "row_count": len(rows)
                                })
                                
                                logger.info(f"📊 Extracted table from page {page_num + 1}: {len(rows)} rows, "
                                           f"room_schedule={is_room_table}")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to extract tables from page {page_num + 1}: {e}")
                        continue
        except Exception as e:
            logger.error(f"❌ PDF table extraction failed: {e}")
        
        return tables_extracted

    def _is_room_schedule_table(self, headers: List[str]) -> bool:
        """
        Determine if table is a room schedule based on headers.
        
        Looks for Romanian or English keywords:
        - Room: "cameră", "room", "spațiu", "space"
        - Area: "suprafață", "area", "mp", "m²"
        - Dimensions: "lungime", "length", "lățime", "width", "dimensiuni"
        
        Returns:
            True if room schedule table detected
        """
        headers_lower = " ".join(headers).lower()
        
        # Romanian indicators
        room_indicators_ro = ["cameră", "camera", "spațiu", "spatiu", "zonă", "zona"]
        area_indicators_ro = ["suprafață", "suprafata", "arie", "mp", "m²", "m2"]
        
        # English indicators
        room_indicators_en = ["room", "space", "area name", "zone"]
        area_indicators_en = ["area", "surface", "sq m", "sqm", "m²", "m2"]
        
        # Check for room + area combination
        has_room = any(indicator in headers_lower for indicator in room_indicators_ro + room_indicators_en)
        has_area = any(indicator in headers_lower for indicator in area_indicators_ro + area_indicators_en)
        
        return has_room and has_area

    def _needs_ocr(self, extracted_text: str, file_content: bytes) -> bool:
        """Determine if OCR is needed based on text density"""
        if not self.ocr_enabled:
            return False
        
        try:
            page_count = self._count_pdf_pages(file_content)
            if page_count == 0: return True
        except Exception:
            page_count = 1
        
        text_length = len(extracted_text.strip())
        text_density = text_length / page_count
        
        needs_ocr = text_density < self.ocr_min_text_threshold
        
        if needs_ocr:
            logger.info(
                f"📊 Text density too low: {text_density:.1f} chars/page "
                f"(threshold: {self.ocr_min_text_threshold})"
            )
        
        return needs_ocr

    def _apply_ocr_fallback(self, file_content: bytes, filename: str = None) -> str:
        """
        Apply Google Document AI OCR to extract text from scanned PDF
        
        Args:
            file_content: PDF file bytes
            filename: Optional filename for file_id generation
            
        Returns:
            Extracted text from OCR
        """
        try:
            if self.ocr_service is None:
                self._lazy_load_ocr()
            
            import asyncio
            
            loop = asyncio.get_event_loop()
            if loop.is_running():
                ocr_result_task = asyncio.create_task(
                    self.ocr_service.process_pdf_with_ocr(
                        file_content=file_content,
                        language_hints=['ro', 'en'],
                        filename=filename  # ✅ Pass filename
                    )
                )
                # Unpack tuple (OCRResult, file_id)
                ocr_result, file_id = loop.run_until_complete(ocr_result_task)
            else:
                # Unpack tuple
                ocr_result, file_id = asyncio.run(
                    self.ocr_service.process_pdf_with_ocr(
                        file_content=file_content,
                        language_hints=['ro', 'en'],
                        filename=filename  # ✅ Pass filename
                    )
                )
            
            logger.info(
                f"✅ OCR completed: {ocr_result.page_count} pages, "
                f"{ocr_result.confidence:.1%} confidence, "
                f"{len(ocr_result.tables)} tables extracted, "
                f"file_id: {file_id}"  # ✅ Log file_id
            )
            
            self._last_ocr_result = ocr_result
            self._last_ocr_file_id = file_id  # ✅ Store file_id
            
            return ocr_result.text
            
        except Exception as e:
            logger.error(f"❌ OCR processing failed: {e}")
            return ""
            
    def _lazy_load_ocr(self):
        """Lazy load OCR service to avoid startup issues"""
        try:
            from src.services.ocr_service import OCRService
            from config.config import settings
            
            self.ocr_service = OCRService(
                processor_id=settings.document_ai_processor_id,
                project_id=settings.document_ai_project_number,
                location=settings.document_ai_location
            )
            
            logger.info("✅ OCR service loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load OCR service: {e}")
            self.ocr_enabled = False
            raise
            
    def _prepare_ocr_metadata(self) -> dict:
        """Prepare OCR metadata from last OCR result"""
        if not hasattr(self, '_last_ocr_result') or self._last_ocr_result is None:
            return {
                'ocr_used': False,
                'tables_extracted': [],
                'handwriting_detected': False,
                'entity_extractions': {},
                'processing_cost': 0.0,
                'text_density_score': 0.0
            }
        
        ocr_result = self._last_ocr_result
        tables_data = [table.to_dict() for table in ocr_result.tables]
        entities = {}
        if ocr_result.entities:
            entities = {
                'dates': ocr_result.entities.dates,
                'prices': ocr_result.entities.prices,
                'measurements': ocr_result.entities.measurements
            }
        
        return {
            'ocr_used': True,
            'tables_extracted': tables_data,
            'handwriting_detected': (
                ocr_result.handwriting.detected 
                if ocr_result.handwriting else False
            ),
            'entity_extractions': entities,
            'processing_cost': ocr_result.cost_estimate,
            'text_density_score': ocr_result.text_density_score
        }

    def _extract_construction_specs(self, text: str) -> List[str]:
        """Extract construction specifications from text"""
        specs = []
        text_lower = text.lower()
        
        spec_patterns = [
            r'specificații?\s+tehnice?',
            r'cerințe\s+constructive',
            r'standarde?\s+aplicabile',
            r'normative?\s+românești',
            r'materiale\s+folosite'
        ]
        
        for pattern in spec_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 200)
                context = text[start:end].strip()
                if context and len(context) > 20:
                    specs.append(context)
        
        return specs[:10]

    def _extract_material_references(self, text: str) -> List[str]:
        """Extract material references from text"""
        materials = []
        text_lower = text.lower()
        
        material_keywords = [
            'gips-carton', 'profile metalice', 'vată minerală', 'polistiren',
            'gresie', 'faianță', 'parchet', 'laminat', 'vopsea', 'zugrăveli'
        ]
        
        for keyword in material_keywords:
            if keyword in text_lower:
                sentences = text.split('.')
                for sentence in sentences:
                    if keyword in sentence.lower() and len(sentence.strip()) > 10:
                        materials.append(sentence.strip())
                        break
        
        return materials[:8]

    def _extract_regulatory_info(self, text: str) -> List[str]:
        """Extract regulatory information from text"""
        regulatory = []
        
        for keyword in self.regulatory_keywords:
            if keyword.lower() in text.lower():
                pattern = rf'.{{0,50}}{re.escape(keyword)}.{{0,100}}'
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    regulatory.append(match.group(0).strip())
        
        return regulatory[:5]

    def _calculate_pdf_confidence(self, text: str, specs: List[str], materials: List[str]) -> float:
        """Calculate confidence in PDF analysis"""
        confidence = 0.2
        if len(text) > 500:
            confidence += 0.2
        keyword_count = sum(1 for keyword in self.construction_keywords if keyword in text.lower())
        confidence += min(keyword_count * 0.05, 0.3)
        confidence += min(len(specs) * 0.1, 0.2)
        confidence += min(len(materials) * 0.05, 0.1)
        return min(confidence, 0.95)

    def _count_pdf_pages(self, file_content: bytes) -> int:
        """Count PDF pages"""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            return len(pdf_reader.pages)
        except Exception:
            return 0

class TXTProcessor:
    """Processes TXT files for construction requirements"""
    
    def __init__(self):
        self.requirement_keywords = [
            'doresc', 'vreau', 'necesit', 'trebuie', 'important',
            'obligatoriu', 'preferat', 'dorit'
        ]
        
        self.construction_terms = [
            'renovare', 'construcție', 'amenajare', 'modernizare',
            'compartimentare', 'finisaje', 'instalații', 'pardoseală'
        ]

    def analyze_txt(self, file_content: Union[bytes, str]) -> TXTAnalysisResult:
        """Analyze TXT content for construction requirements"""
        try:
            if isinstance(file_content, bytes):
                content = file_content.decode('utf-8', errors='ignore')
            else:
                content = file_content
            
            construction_keywords = self._extract_construction_keywords(content)
            requirements = self._extract_requirements(content)
            client_preferences = self._extract_client_preferences(content)
            
            confidence = self._calculate_txt_confidence(
                content, construction_keywords, requirements
            )
            
            return TXTAnalysisResult(
                content=content[:2000],
                construction_keywords=construction_keywords,
                requirements=requirements,
                client_preferences=client_preferences,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"TXT analysis failed: {e}")
            return TXTAnalysisResult(content="", confidence=0.0)

    def _extract_construction_keywords(self, text: str) -> List[str]:
        """Extract construction-related keywords"""
        keywords = [term for term in self.construction_terms if term in text.lower()]
        return keywords

    def _extract_requirements(self, text: str) -> List[str]:
        """Extract specific requirements from text"""
        requirements = []
        sentences = text.split('.')
        
        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            if any(keyword in sentence_lower for keyword in self.requirement_keywords):
                if len(sentence.strip()) > 10:
                    requirements.append(sentence.strip())
        
        return requirements[:8]

    def _extract_client_preferences(self, text: str) -> List[str]:
        """Extract client preferences and special requests"""
        preferences = []
        preference_keywords = [
            'prefer', 'doresc', 'îmi place', 'vreau', 'aș dori',
            'este important', 'prioritate'
        ]
        
        sentences = text.split('.')
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in preference_keywords):
                if len(sentence.strip()) > 15:
                    preferences.append(sentence.strip())
        
        return preferences[:5]

    def _calculate_txt_confidence(self, text: str, keywords: List[str], requirements: List[str]) -> float:
        """Calculate confidence in TXT analysis"""
        confidence = 0.3
        if 50 < len(text) < 2000:
            confidence += 0.2
        confidence += min(len(keywords) * 0.1, 0.3)
        confidence += min(len(requirements) * 0.05, 0.2)
        return min(confidence, 0.9)

class UnifiedDocumentProcessor:
    """Unified processor for all document types"""
    
    def __init__(self):
        self.dxf_analyzer = DXFAnalyzer() if DXF_AVAILABLE else None
        self.pdf_processor = PDFProcessor()
        self.txt_processor = TXTProcessor()
        self.rfp_parser = RFPParser()  # ✅ ADD THIS LINE
        self.logger = logging.getLogger("demoplan.processors.pdf")

    def process_document(self, filename: str, file_content: bytes) -> UnifiedDocumentResult:
        """Process document based on file type"""
        filename_lower = filename.lower()
        file_ext = filename_lower.split('.')[-1] if '.' in filename_lower else ''
        
        if filename_lower.endswith('.dxf') and self.dxf_analyzer:
            return self._process_dxf(filename, file_content)
        elif filename_lower.endswith('.pdf'):
            return self._process_pdf(filename, file_content)
        elif filename_lower.endswith('.txt'):
            return self._process_txt(filename, file_content)
        elif file_ext == 'csv':
            return self._process_csv(filename, file_content)
        elif file_ext in ['xlsx', 'xls']:
            return self._process_excel(filename, file_content)
        elif file_ext == 'json':
            return self._process_json(filename, file_content)
        else:
            return UnifiedDocumentResult(document_type="unsupported", combined_confidence=0.0)

    def _process_dxf(self, filename: str, file_content: bytes) -> UnifiedDocumentResult:
        """Process DXF file"""
        try:
            dxf_result = self.dxf_analyzer.analyze(file_content)
            
            if dxf_result is None or "dxf_analysis" not in dxf_result:
                logger.warning(f"DXF analyzer returned invalid result for {filename}")
                return UnifiedDocumentResult(
                    document_type="dxf",
                    combined_confidence=0.0,
                    integrated_specs={"error": "DXF analysis failed"}
                )
            
            confidence = dxf_result.get("dxf_analysis", {}).get("confidence", 0) / 100.0
            
            return UnifiedDocumentResult(
                document_type="dxf",
                dxf_analysis=dxf_result,
                combined_confidence=confidence
            )
        except Exception as e:
            logger.error(f"DXF processing failed for {filename}: {e}", exc_info=True)
            return UnifiedDocumentResult(
                document_type="dxf",
                combined_confidence=0.0,
                integrated_specs={"error": f"DXF processing exception: {str(e)}"}
            )

    def _process_pdf(self, filename: str, file_content: bytes) -> UnifiedDocumentResult:
        try:
            self.logger.info(f"📄 Processing PDF: {filename}")  # Changed
            pdf_result = self.pdf_processor.analyze_pdf(file_content, filename=filename)
            extracted_text = pdf_result.extracted_text or ""
            self.logger.info(f"🔍 PDF text extracted: {len(extracted_text)} chars")  # Changed
        
            is_rfp, rfp_confidence = self.rfp_parser.is_rfp_document(extracted_text)
            self.logger.info(f"🔍 RFP detection: {is_rfp}, confidence: {rfp_confidence:.2f}")  # Changed
        
            rfp_data = None
            if is_rfp and len(extracted_text) > 200:
                try:
                    rfp_data = self.rfp_parser.parse_rfp(extracted_text)
                    self.logger.info(f"✅ RFP parsed: {rfp_data.project_name}")  # Changed
                except Exception as e:
                    self.logger.error(f"❌ RFP parsing error: {e}")  # Changed
        
            return UnifiedDocumentResult(
                document_type="RFP" if is_rfp else "pdf",
                pdf_analysis=pdf_result,
                rfp_analysis=rfp_data,
                combined_confidence=self._calculate_pdf_confidence(pdf_result, rfp_data),
                integrated_specs={
                    "document_source": "pdf",
                    "specifications_found": len(pdf_result.construction_specs),
                    "materials_identified": len(pdf_result.material_references)
                }
            )
        except Exception as e:
            self.logger.error(f"❌ PDF processing exception: {e}", exc_info=True)  # Changed
            return UnifiedDocumentResult(
                document_type="pdf",
                combined_confidence=0.0,
                integrated_specs={"error": str(e)}
            )

    def _process_txt(self, filename: str, file_content: bytes) -> UnifiedDocumentResult:
        """Process TXT file"""
        txt_result = self.txt_processor.analyze_txt(file_content)
        
        integrated_specs = {
            "document_source": "txt",
            "requirements_found": len(txt_result.requirements),
            "preferences_identified": len(txt_result.client_preferences),
            "construction_keywords": len(txt_result.construction_keywords),
            "content_length": len(txt_result.content)
        }
        
        return UnifiedDocumentResult(
            document_type="txt",
            txt_analysis=txt_result,
            combined_confidence=txt_result.confidence,
            integrated_specs=integrated_specs
        )

    def _calculate_pdf_confidence(self, pdf_result: PDFAnalysisResult, rfp_data: Optional[RFPStructure]) -> float:
        """Calculate confidence including RFP extraction quality"""
        base_confidence = pdf_result.confidence
        
        if rfp_data:
            # Boost confidence if RFP was successfully extracted
            rfp_confidence = rfp_data.extraction_confidence
            # Weighted average: 60% base PDF + 40% RFP extraction
            return (base_confidence * 0.6) + (rfp_confidence * 0.4)
        
        return base_confidence

    def _process_csv(self, filename: str, content: bytes) -> UnifiedDocumentResult:
        """Process CSV files for training data"""
        try:
            df = pd.read_csv(io.BytesIO(content))
            
            text_content = f"CSV File: {filename}\nColumns: {', '.join(df.columns.tolist())}\nRows: {len(df)}\n\n{df.to_string()}"

            analysis_result = TabularAnalysisResult(
                file_type="csv",
                filename=filename,
                columns=df.columns.tolist(),
                row_count=len(df),
                data_preview=df.to_dict('records')[:100],
                text_representation=text_content,
                confidence=0.8
            )

            return UnifiedDocumentResult(
                document_type="csv",
                tabular_analysis=analysis_result,
                combined_confidence=0.8,
                integrated_specs={"document_source": "csv", "columns": analysis_result.columns, "row_count": analysis_result.row_count}
            )
        except Exception as e:
            logger.error(f"CSV processing error: {e}")
            return UnifiedDocumentResult(document_type="csv", combined_confidence=0.1, integrated_specs={"error": f"Failed to read CSV: {e}"})

    def _process_excel(self, filename: str, content: bytes) -> UnifiedDocumentResult:
        """Process Excel files for training data"""
        try:
            df = pd.read_excel(io.BytesIO(content), sheet_name=0)
            
            text_content = f"Excel File: {filename}\nColumns: {', '.join(df.columns.tolist())}\nRows: {len(df)}\n\n{df.to_string()}"

            analysis_result = TabularAnalysisResult(
                file_type="excel",
                filename=filename,
                columns=df.columns.tolist(),
                row_count=len(df),
                data_preview=df.to_dict('records')[:100],
                text_representation=text_content,
                confidence=0.8
            )

            return UnifiedDocumentResult(
                document_type="excel",
                tabular_analysis=analysis_result,
                combined_confidence=0.8,
                integrated_specs={"document_source": "excel", "columns": analysis_result.columns, "row_count": analysis_result.row_count}
            )
        except Exception as e:
            logger.error(f"Excel processing error: {e}")
            return UnifiedDocumentResult(document_type="excel", combined_confidence=0.1, integrated_specs={"error": f"Failed to read Excel: {e}"})

    def _process_json(self, filename: str, content: bytes) -> UnifiedDocumentResult:
        """Process JSON files for training data"""
        try:
            data = json.loads(content.decode('utf-8'))
            text_content = f"JSON File: {filename}\n{json.dumps(data, indent=2, ensure_ascii=False)}"

            analysis_result = JSONAnalysisResult(
                filename=filename,
                data=data,
                text_representation=text_content,
                confidence=0.9
            )

            return UnifiedDocumentResult(
                document_type="json",
                json_analysis=analysis_result,
                combined_confidence=0.9,
                integrated_specs={"document_source": "json", "data_keys": list(data.keys()) if isinstance(data, dict) else []}
            )
        except Exception as e:
            logger.error(f"JSON processing error: {e}")
            return UnifiedDocumentResult(document_type="json", combined_confidence=0.1, integrated_specs={"error": f"Failed to parse JSON: {e}"})

class MultiLanguageRoomClassifier:
    """Multi-language room classification for EU languages"""
    
    def __init__(self):
        self.room_keywords = {
            'english': {
                'bedroom': ['bedroom', 'bed room', 'master bed', 'guest room', 'master bedroom'],
                'kitchen': ['kitchen', 'kitchenette', 'cooking area'],
                'bathroom': ['bathroom', 'toilet', 'restroom', 'lavatory', 'wc', 'wash room'],
                'living_room': ['living room', 'living', 'lounge', 'sitting room', 'family room'],
                'dining_room': ['dining room', 'dining', 'dining area'],
                'hallway': ['hallway', 'corridor', 'passage', 'entrance hall'],
                'storage': ['storage', 'closet', 'pantry', 'utility'],
                'office': ['office', 'study', 'den', 'work room'],
                'balcony': ['balcony', 'terrace', 'patio', 'deck'],
                'garage': ['garage', 'parking'],
                'basement': ['basement', 'cellar'],
                'commercial': ['sales area', 'shop', 'store', 'retail', 'back of house', 'warehouse']
            },
            'romanian': {
                'bedroom': ['dormitor', 'camera de dormit', 'camera'],
                'kitchen': ['bucatarie', 'bucătărie', 'anex bucătărie'],
                'bathroom': ['baie', 'toaleta', 'wc', 'băi'],
                'living_room': ['living', 'salon', 'sufragerie', 'camera de zi'],
                'dining_room': ['sufragerie', 'sala de mese'],
                'hallway': ['hol', 'coridor', 'antreu', 'vestibul'],
                'storage': ['debara', 'cămară', 'dulap', 'spațiu de depozitare'],
                'office': ['birou', 'cabinet', 'cameră de lucru'],
                'balcony': ['balcon', 'terasă', 'loggie'],
                'garage': ['garaj', 'parcare'],
                'basement': ['subsol', 'pivniță'],
                'commercial': ['zona de vânzare', 'magazin', 'spațiu comercial', 'depozit']
            }
        }
        
        self.language_indicators = {
            'english': ['room', 'area', 'space', 'floor', 'wall', 'door', 'window'],
            'romanian': ['cameră', 'spațiu', 'zonă', 'perete', 'ușă', 'fereastră']
        }
        
        self.noise_patterns = [
            r'płyta\s+GK', r'plasterboard', r'gips-carton', r'ceramic\s+tiles?',
            r'gres', r'faience', r'paint', r'vopsea', r'insulation', r'izolatie',
            r'thickness', r'grosime', r'profiles?\s+CW\d+', r'profile'
        ]

    def classify_text(self, text: str, location: List[float]) -> FilteredText:
        """Classify text with language detection and relevance scoring"""
        text_lower = text.lower().strip()
        
        if len(text_lower) < 2 or len(text_lower) > 200:
            return FilteredText(text, location, 0.0, 'noise')
        
        if self._is_construction_noise(text_lower):
            return FilteredText(text, location, 0.1, 'specification')
        
        if self._is_dimension(text_lower):
            return FilteredText(text, location, 0.3, 'dimension')
        
        room_analysis = self._analyze_room_label(text_lower)
        if room_analysis['is_room']:
            return FilteredText(
                text, location, room_analysis['confidence'], 'room_label',
                room_analysis['language'], room_analysis['room_type']
            )
        
        if self._is_area_indicator(text_lower):
            return FilteredText(text, location, 0.5, 'area_label')
        
        relevance = self._calculate_text_relevance(text_lower)
        classification = 'area_label' if relevance > 0.4 else 'noise'
        
        return FilteredText(text, location, relevance, classification)

    def detect_primary_language(self, all_texts: List[str]) -> Optional[str]:
        """Detect primary language from text content"""
        language_scores = {lang: 0 for lang in self.language_indicators.keys()}
        
        for text in all_texts:
            text_lower = text.lower()
            for lang, indicators in self.language_indicators.items():
                for indicator in indicators:
                    if indicator in text_lower:
                        language_scores[lang] += 1
        
        if max(language_scores.values()) > 0:
            return max(language_scores.items(), key=lambda x: x[1])[0]
        return None

    def _analyze_room_label(self, text: str) -> Dict[str, Any]:
        """Analyze if text represents a room label"""
        result = {'is_room': False, 'confidence': 0.0, 'language': None, 'room_type': None}
        
        for lang, room_types in self.room_keywords.items():
            for room_type, keywords in room_types.items():
                for keyword in keywords:
                    if keyword in text:
                        confidence = self._calculate_room_confidence(text, keyword)
                        if confidence > result['confidence']:
                            result = {
                                'is_room': True,
                                'confidence': confidence,
                                'language': lang,
                                'room_type': room_type
                            }
        return result

    def _calculate_room_confidence(self, text: str, keyword: str) -> float:
        """Calculate confidence for room label detection"""
        confidence = 0.4
        if text == keyword: confidence += 0.4
        if len(text.split()) == 1: confidence += 0.2
        if text.startswith(keyword) or text.endswith(keyword): confidence += 0.1
        if len(text) > 50: confidence -= 0.2
        return min(confidence, 1.0)

    def _is_construction_noise(self, text: str) -> bool:
        """Check if text is construction specification noise"""
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in self.noise_patterns)

    def _is_dimension(self, text: str) -> bool:
        """Check if text represents dimensions or measurements"""
        dimension_patterns = [
            r'\d+[.,]\d+\s*[xX×]\s*\d+[.,]\d+', r'\d+[.,]\d+\s*m',
            r'\d+[.,]\d+\s*cm', r'^\d+[.,]\d+$'
        ]
        return any(re.search(pattern, text) for pattern in dimension_patterns)

    def _is_area_indicator(self, text: str) -> bool:
        """Check if text indicates area or important spatial information"""
        area_patterns = [
            r'area|zone|surface|zonă', r'floor|etaj', r'level|nivel'
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in area_patterns)

    def _calculate_text_relevance(self, text: str) -> float:
        """Calculate overall text relevance for spatial analysis"""
        relevance = 0.2
        if 2 <= len(text) <= 20: relevance += 0.2
        if re.search(r'\d', text): relevance += 0.1
        spatial_keywords = ['room', 'space', 'area', 'zone', 'floor', 'level']
        if any(keyword in text.lower() for keyword in spatial_keywords): relevance += 0.3
        return min(relevance, 1.0)

class DocumentClassifier:
    """Classifies document type and project type"""
    
    def __init__(self):
        self.document_indicators = {
            DocumentType.SPECIFICATION_SHEET: ['types', 'typy', 'specifications', 'legend', 'legenda', 'wall types', 'finishing', 'materials'],
            DocumentType.FLOOR_PLAN: ['floor plan', 'plan', 'layout', 'rooms', 'spaces'],
            DocumentType.TECHNICAL_DETAIL: ['detail', 'section', 'elevation', 'cross-section']
        }
        self.project_indicators = {
            ProjectType.COMMERCIAL: ['sales area', 'shop', 'store', 'retail', 'office', 'commercial', 'magazin', 'birou'],
            ProjectType.RESIDENTIAL: ['bedroom', 'kitchen', 'bathroom', 'living room', 'apartment', 'dormitor', 'bucătărie', 'baie', 'living', 'apartament'],
            ProjectType.INDUSTRIAL: ['warehouse', 'factory', 'production', 'industrial', 'depot']
        }

    def classify_document(self, all_texts: List[str], boundaries_count: int) -> DocumentClassification:
        """Classify document and project type"""
        combined_text = ' '.join(all_texts).lower()
        
        doc_type_scores = {doc_type: sum(1 for indicator in indicators if indicator in combined_text) for doc_type, indicators in self.document_indicators.items()}
        project_type_scores = {proj_type: sum(1 for indicator in indicators if indicator in combined_text) for proj_type, indicators in self.project_indicators.items()}
        
        if doc_type_scores.get(DocumentType.SPECIFICATION_SHEET, 0) > 3:
            document_type = DocumentType.SPECIFICATION_SHEET
        elif boundaries_count > 3:
            document_type = DocumentType.FLOOR_PLAN
        elif doc_type_scores and max(doc_type_scores.values()) > 0:
            document_type = max(doc_type_scores, key=doc_type_scores.get)
        else:
            document_type = DocumentType.UNKNOWN
            
        if project_type_scores and max(project_type_scores.values()) > 0:
            project_type = max(project_type_scores, key=project_type_scores.get)
        else:
            project_type = ProjectType.UNKNOWN
        
        max_doc_score = max(doc_type_scores.values()) if doc_type_scores else 0
        max_proj_score = max(project_type_scores.values()) if project_type_scores else 0
        confidence = min((max_doc_score + max_proj_score) / 10.0, 1.0)
        
        indicators = []
        if document_type != DocumentType.UNKNOWN: indicators.append(f"Document: {document_type.value}")
        if project_type != ProjectType.UNKNOWN: indicators.append(f"Project: {project_type.value}")
        
        return DocumentClassification(
            document_type=document_type,
            project_type=project_type,
            primary_language=None,
            confidence=confidence,
            indicators=indicators
        )

class GeometryFirstExtractor:
    """Geometry-first spatial data extraction"""
    
    def __init__(self, room_classifier: MultiLanguageRoomClassifier):
        self.room_classifier = room_classifier
        self.min_room_area = 1.0
        self.max_room_area = 10000.0

    def extract_spatial_data(self, doc: Drawing) -> Tuple[List[SpatialBoundary], List[FilteredText]]:
        """Extract spatial boundaries and filtered text"""
        boundaries = self._extract_spatial_boundaries(doc)
        all_texts = self._extract_all_text_entities(doc)
        filtered_texts = [self.room_classifier.classify_text(t['text'], t['location']) for t in all_texts]
        self._associate_texts_with_boundaries(boundaries, filtered_texts)
        return boundaries, filtered_texts

    def _extract_spatial_boundaries(self, doc: Drawing) -> List[SpatialBoundary]:
        """Extract closed polylines as potential room boundaries
        
        Week 3 Enhancement: Added per-entity try/except to gracefully skip malformed entities.
        """
        boundaries = []

        # Extract LWPOLYLINE entities (query by type only, then check closed flag)
        for i, entity in enumerate(doc.modelspace().query('LWPOLYLINE')):
            try:
                is_closed = getattr(entity, 'is_closed', None)
                # Some ezdxf entities expose is_closed as attribute or method
                if is_closed is None:
                    try:
                        is_closed = bool(entity.get_closed())
                    except Exception:
                        is_closed = False

                if not is_closed:
                    continue

                vertices = [list(p)[:2] for p in entity.get_points()]
                if len(vertices) < 3:
                    continue

                area = self._calculate_polygon_area(vertices)
                if not (self.min_room_area <= area <= self.max_room_area):
                    continue

                boundaries.append(SpatialBoundary(
                    boundary_id=f"lwpoly_{i}",
                    vertices=vertices,
                    area=area,
                    centroid=self._calculate_centroid(vertices),
                    confidence=self._calculate_boundary_confidence(area, vertices),
                    associated_texts=[]
                ))
            except Exception as e:
                logger.debug(f"Skipped LWPOLYLINE entity {i}: {e}")
                continue

        # Extract legacy POLYLINE entities (query by type only)
        for i, entity in enumerate(doc.modelspace().query('POLYLINE')):
            try:
                # POLYLINE may have is_closed stored on the entity or in flags
                is_closed = getattr(entity, 'is_closed', None)
                if is_closed is None:
                    try:
                        is_closed = bool(entity.is_closed)
                    except Exception:
                        # try dxf attribute
                        is_closed = getattr(entity.dxf, 'closed', False) if hasattr(entity, 'dxf') else False

                if not is_closed:
                    continue

                # vertices for POLYLINE come from entity.vertices
                vertices = [list(v.dxf.location)[:2] for v in entity.vertices]
                if len(vertices) < 3:
                    continue

                area = self._calculate_polygon_area(vertices)
                if not (self.min_room_area <= area <= self.max_room_area):
                    continue

                boundaries.append(SpatialBoundary(
                    boundary_id=f"poly_{i}",
                    vertices=vertices,
                    area=area,
                    centroid=self._calculate_centroid(vertices),
                    confidence=self._calculate_boundary_confidence(area, vertices),
                    associated_texts=[]
                ))
            except Exception as e:
                logger.debug(f"Skipped POLYLINE entity {i}: {e}")
                continue

        # Final safety catch-all (should be rare)
        if not boundaries:
            logger.warning("No spatial boundaries extracted or all entities skipped due to errors")

        boundaries.sort(key=lambda x: x.area, reverse=True)
        return boundaries[:50]

    def _extract_all_text_entities(self, doc: Drawing) -> List[Dict[str, Any]]:
        """Extract all text entities with location information"""
        text_entities = []
        try:
            for entity in doc.modelspace().query('TEXT MTEXT'):
                if hasattr(entity.dxf, 'text') and hasattr(entity.dxf, 'insert'):
                    text = self._clean_mtext(entity.dxf.text) if entity.dxftype() == 'MTEXT' else entity.dxf.text.strip()
                    text_entities.append({
                        'text': text,
                        'location': list(entity.dxf.insert)[:2],
                        'height': getattr(entity.dxf, 'height', 0)
                    })
        except Exception as e:
            logger.warning(f"Error extracting text entities: {e}")
        return text_entities

    def _clean_mtext(self, mtext: str) -> str:
        """Clean MTEXT formatting codes"""
        clean_text = re.sub(r'\\P', ' ', mtext)
        clean_text = re.sub(r'\\[A-Za-z]\d*;', '', clean_text)
        clean_text = re.sub(r'[{}]', '', clean_text)
        return clean_text.strip()

    def _associate_texts_with_boundaries(self, boundaries: List[SpatialBoundary], texts: List[FilteredText]):
        """Associate relevant texts with spatial boundaries"""
        for text in texts:
            if text.classification in ['room_label', 'area_label'] and text.relevance_score > 0.3:
                for boundary in boundaries:
                    if self._point_in_polygon(text.location, boundary.vertices):
                        boundary.associated_texts.append(text.text)
                        break

    def _calculate_boundary_confidence(self, area: float, vertices: List[List[float]]) -> float:
        """Calculate confidence score for a boundary"""
        confidence = 0.5
        if 5 <= area <= 100: confidence += 0.3
        elif 1 <= area <= 500: confidence += 0.2
        if 4 <= len(vertices) <= 8: confidence += 0.2
        if self._is_roughly_rectangular(vertices): confidence += 0.1
        return min(confidence, 1.0)

    def _is_roughly_rectangular(self, vertices: List[List[float]]) -> bool:
        """Check if polygon is roughly rectangular"""
        if len(vertices) != 4: return False
        angles = []
        for i in range(4):
            p1, p2, p3 = vertices[i], vertices[(i + 1) % 4], vertices[(i + 2) % 4]
            v1, v2 = [p2[0] - p1[0], p2[1] - p1[1]], [p3[0] - p2[0], p3[1] - p2[1]]
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            mag1, mag2 = math.sqrt(v1[0]**2 + v1[1]**2), math.sqrt(v2[0]**2 + v2[1]**2)
            if mag1 > 0 and mag2 > 0:
                cos_angle = max(-1, min(1, dot / (mag1 * mag2)))
                angles.append(math.degrees(math.acos(cos_angle)))
        return all(80 <= angle <= 100 for angle in angles)

    def _calculate_polygon_area(self, vertices: List[List[float]]) -> float:
        """Calculate polygon area using shoelace formula"""
        if len(vertices) < 3: return 0.0
        area = 0.0
        for i in range(len(vertices)):
            j = (i + 1) % len(vertices)
            area += vertices[i][0] * vertices[j][1] - vertices[j][0] * vertices[i][1]
        return abs(area) / 2.0

    def _calculate_centroid(self, vertices: List[List[float]]) -> List[float]:
        """Calculate centroid of polygon"""
        if not vertices: return [0.0, 0.0]
        x = sum(v[0] for v in vertices) / len(vertices)
        y = sum(v[1] for v in vertices) / len(vertices)
        return [x, y]

    def _point_in_polygon(self, point: List[float], polygon: List[List[float]]) -> bool:
        """Check if point is inside polygon using ray casting algorithm"""
        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if p1x == p2x or x <= xinters:
                    inside = not inside
            p1x, p1y = p2x, p2y
        return inside

class DXFAnalyzer:
    """Enhanced DXF analyzer with geometry-first, multi-language approach"""
    
    def __init__(self):
        if not DXF_AVAILABLE:
            raise ImportError("`ezdxf` library is required for DXFAnalyzer.")
        
        self.room_classifier = MultiLanguageRoomClassifier()
        self.document_classifier = DocumentClassifier()
        self.geometry_extractor = GeometryFirstExtractor(self.room_classifier)
        self.spec_extractor = SpecificationExtractor()

    def analyze(self, file_content: bytes) -> Dict[str, Any]:
        """Enhanced DXF analysis with geometry-first approach"""
        try:
            if isinstance(file_content, str):
                file_content = file_content.encode('utf-8', errors='ignore')

            file_stream = io.BytesIO(file_content)
            doc, auditor = recover_read(file_stream)
            
            if auditor.has_errors:
                logger.warning(f"DXF file has {len(auditor.errors)} errors but was recovered")

            boundaries, filtered_texts = self.geometry_extractor.extract_spatial_data(doc)
            all_text_strings = [ft.text for ft in filtered_texts]
            doc_classification = self.document_classifier.classify_document(all_text_strings, len(boundaries))
            doc_classification.primary_language = self.room_classifier.detect_primary_language(all_text_strings)
            
            specification_data = self.spec_extractor.extract_specifications(filtered_texts)
            
            room_breakdown = self._create_room_breakdown(boundaries, filtered_texts)
            # Create lightweight summary for Firestore storage
            room_breakdown_summary = self._create_summary_room_breakdown(room_breakdown)
            total_rooms = len(room_breakdown)
            total_area = sum(room.get('area', 0) for room in room_breakdown)
            
            has_dimensions = any(ft.classification == 'dimension' for ft in filtered_texts)
            has_electrical = self._detect_electrical_elements(doc)
            has_hvac = self._detect_hvac_elements(doc)
            
            confidence_score = self._calculate_overall_confidence(
                boundaries, filtered_texts, doc_classification, total_rooms, total_area, specification_data
            )
            
            summary = self._generate_enhanced_summary(
                total_rooms, total_area, room_breakdown, doc_classification, 
                has_dimensions, has_electrical, has_hvac, specification_data
            )
            
            entity_inventory = self.extract_entity_inventory(doc, boundaries)

            # If document looks like a specification sheet, try to synthesize room data
            # from text annotations (no geometric boundaries present)
            annotations = []
            try:
                annotations = self._extract_text_annotations(doc)
            except Exception:
                annotations = []

            if doc_classification.document_type == DocumentType.SPECIFICATION_SHEET:
                synthetic_rooms = self._extract_rooms_from_text_annotations(
                    text_annotations=annotations,
                    dimension_schedule=(entity_inventory.dimensions if entity_inventory else [])
                )

                if synthetic_rooms:
                    room_breakdown = synthetic_rooms
                    room_breakdown_summary = self._create_summary_room_breakdown(synthetic_rooms)

            return {
                "dxf_analysis": {
                    "total_rooms": total_rooms,
                    "total_area": round(total_area, 2),
                    "room_breakdown": room_breakdown,
                    "room_breakdown_summary": room_breakdown_summary,
                    "has_dimensions": has_dimensions,
                    "has_electrical": has_electrical,
                    "has_hvac": has_hvac,
                    "summary": summary,
                    "confidence": confidence_score,
                    "document_type": doc_classification.document_type.value,
                    "project_type": doc_classification.project_type.value,
                    "primary_language": doc_classification.primary_language,
                    "boundaries_detected": len(boundaries),
                    "texts_filtered": len([ft for ft in filtered_texts if ft.relevance_score > 0.3]),
                    "specification_analysis": specification_data.__dict__ if specification_data else None,
                    "hvac_inventory": [c.__dict__ for c in entity_inventory.hvac_components],
                    "electrical_inventory": [c.__dict__ for c in entity_inventory.electrical_components],
                    "door_window_schedule": [c.__dict__ for c in entity_inventory.doors_windows],
                    "dimension_schedule": [c.__dict__ for c in entity_inventory.dimensions],
                    "plumbing_inventory": [c.__dict__ for c in entity_inventory.plumbing_components],
                    "flooring_schedule": [c.__dict__ for c in entity_inventory.flooring_info],
                    "ceiling_schedule": [c.__dict__ for c in entity_inventory.ceiling_info],
                    "text_annotations": entity_inventory.text_annotations
                },
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error during DXF analysis: {e}", exc_info=True)
            # Return a consistent shaped response so callers always receive dxf_analysis
            error_msg = f"Failed to analyze DXF file: {e}"
            return {
                "error": error_msg,
                "status": "error",
                "dxf_analysis": {
                    "total_rooms": 0,
                    "total_area": 0.0,
                    "room_breakdown": [],
                    "room_breakdown_summary": [],
                    "has_dimensions": False,
                    "has_electrical": False,
                    "has_hvac": False,
                    "summary": "",
                    "confidence": 0.0,
                    "document_type": None,
                    "project_type": None,
                    "primary_language": None,
                    "boundaries_detected": 0,
                    "texts_filtered": 0,
                    "specification_analysis": None,
                    "hvac_inventory": [],
                    "electrical_inventory": [],
                    "door_window_schedule": [],
                    "dimension_schedule": [],
                    "plumbing_inventory": [],
                    "flooring_schedule": [],
                    "ceiling_schedule": [],
                    "text_annotations": [],
                    "error_message": error_msg
                }
            }

    def extract_entity_inventory(self, doc: Drawing, boundaries: List[SpatialBoundary]) -> EntityInventory:
        """Extract comprehensive entity-level inventory."""
        inventory = EntityInventory()
        logger.info("🔍 Starting detailed entity inventory extraction...")
        try:
            inventory.hvac_components = self._extract_hvac_inventory(doc, boundaries)
            inventory.electrical_components = self._extract_electrical_inventory(doc, boundaries)
            inventory.doors_windows = self._extract_doors_windows_inventory(doc, boundaries)
            inventory.dimensions = self._extract_dimensions_inventory(doc)
            inventory.plumbing_components = self._extract_plumbing_inventory(doc, boundaries)
            inventory.flooring_info = self._extract_flooring_info(doc, boundaries)
            inventory.ceiling_info = self._extract_ceiling_info(doc, boundaries)
            inventory.text_annotations = self._extract_text_annotations(doc)
            logger.info("✅ Entity inventory extraction complete.")
        except Exception as e:
            logger.error(f"❌ Entity inventory extraction error: {e}", exc_info=True)
        return inventory

    def _extract_hvac_inventory(self, doc: Drawing, boundaries: List[SpatialBoundary]) -> List[HVACComponent]:
        """Extract HVAC components from blocks and entities"""
        components = []
        hvac_block_names = ['ac', 'hvac', 'air', 'vent', 'diffuser', 'grille', 'fan', 'thermostat']
        
        for i, entity in enumerate(doc.modelspace().query('INSERT')):
            if not hasattr(entity.dxf, 'name') or not hasattr(entity.dxf, 'insert'): continue
            block_name = entity.dxf.name.lower()
            if any(hvac_name in block_name for hvac_name in hvac_block_names):
                location = list(entity.dxf.insert)[:2]
                components.append(HVACComponent(
                    component_id=f"hvac_{i}",
                    component_type='diffuser' if 'diffuser' in block_name else 'ac_unit',
                    location=location,
                    capacity_kw=self._extract_capacity_from_block(entity, doc),
                    room_association=self._find_room_for_location(location, boundaries),
                    specifications={'block_name': entity.dxf.name, 'layer': entity.dxf.layer}
                ))
        return components

    def _extract_electrical_inventory(self, doc: Drawing, boundaries: List[SpatialBoundary]) -> List[ElectricalComponent]:
        """Extract electrical components from blocks and symbols"""
        components = []
        outlet_names = ['priza', 'outlet', 'socket', 'plug']
        switch_names = ['intrerupator', 'switch', 'sw']
        light_names = ['spot', 'light', 'lamp', 'fixture']
        
        for i, entity in enumerate(doc.modelspace().query('INSERT')):
            if not hasattr(entity.dxf, 'name') or not hasattr(entity.dxf, 'insert'): continue
            block_name = entity.dxf.name.lower()
            location = list(entity.dxf.insert)[:2]
            comp_type = None
            if any(n in block_name for n in outlet_names): comp_type = 'outlet'
            elif any(n in block_name for n in switch_names): comp_type = 'switch'
            elif any(n in block_name for n in light_names): comp_type = 'light_fixture'
            
            if comp_type:
                components.append(ElectricalComponent(
                    component_id=f"elec_{i}", component_type=comp_type, location=location,
                    power_rating=self._extract_power_rating_from_block(entity, doc),
                    room_association=self._find_room_for_location(location, boundaries),
                    specifications={'block_name': entity.dxf.name, 'layer': entity.dxf.layer}
                ))
        return self._group_electrical_components(components)

    def _extract_doors_windows_inventory(self, doc: Drawing, boundaries: List[SpatialBoundary]) -> List[DoorWindow]:
        """Extract doors and windows from blocks and polylines"""
        components = []
        door_kw = ['door', 'usa']
        window_kw = ['window', 'fereastra']
        
        for i, entity in enumerate(doc.modelspace().query('INSERT')):
            if not hasattr(entity.dxf, 'name') or not hasattr(entity.dxf, 'insert'): continue
            block_name = entity.dxf.name.lower()
            location = list(entity.dxf.insert)[:2]
            
            is_door, is_window = any(kw in block_name for kw in door_kw), any(kw in block_name for kw in window_kw)
            
            if is_door or is_window:
                width, height = self._extract_opening_dimensions(entity, doc)
                components.append(DoorWindow(
                    component_id=f"{'door' if is_door else 'window'}_{i}",
                    component_type='door' if is_door else 'window', location=location,
                    width=width or 0.9, height=height or 2.1,
                    room_association=self._find_room_for_location(location, boundaries),
                    specifications={'block_name': entity.dxf.name, 'layer': entity.dxf.layer}
                ))
        return components

    def _extract_dimensions_inventory(self, doc: Drawing) -> List[DimensionEntity]:
        """Extract dimension entities with actual measured values"""
        dims = []
        for i, entity in enumerate(doc.modelspace().query('DIMENSION')):
            if not hasattr(entity.dxf, 'defpoint'): continue
            dims.append(DimensionEntity(
                dimension_id=f"dim_{i}",
                measurement_value=getattr(entity.dxf, 'actual_measurement', 0.0),
                dimension_type='linear', # Simplified
                location=list(entity.dxf.defpoint)[:2],
                text_override=getattr(entity.dxf, 'text', None)
            ))
        return dims
        
    def _extract_plumbing_inventory(self, doc: Drawing, boundaries: List[SpatialBoundary]) -> List[PlumbingComponent]:
        return [] # Simplified for brevity

    def _extract_flooring_info(self, doc: Drawing, boundaries: List[SpatialBoundary]) -> List[FlooringInfo]:
        return [] # Simplified for brevity

    def _extract_ceiling_info(self, doc: Drawing, boundaries: List[SpatialBoundary]) -> List[CeilingInfo]:
        return [] # Simplified for brevity

    def _extract_text_annotations(self, doc: Drawing) -> List[Dict[str, Any]]:
        annotations = []
        for entity in doc.modelspace().query('TEXT MTEXT'):
            if hasattr(entity.dxf, 'text') and hasattr(entity.dxf, 'insert'):
                text = entity.dxf.text.strip()
                if len(text) >= 3 and not text.isdigit():
                    annotations.append({'text': text, 'location': list(entity.dxf.insert)[:2], 'layer': entity.dxf.layer})
        return annotations

    def _extract_rooms_from_text_annotations(self, text_annotations: List[Dict[str, Any]], dimension_schedule: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create a synthetic room breakdown from loose text annotations and any found dimensions.

        This is a best-effort extractor used for specification sheets where no geometry is available.
        It looks for lines that contain a room name and optionally an area (e.g., "Living - 25.5 m2").
        """
        rooms = []
        try:
            # Build a quick lookup of dimensions by nearby locations if available (not required)
            dims_by_text = {}
            for dim in (dimension_schedule or []):
                # dimension entries may contain 'text_override' or similar fields
                key = dim.get('text_override') or str(dim.get('measurement_value', ''))
                dims_by_text[key] = dim

            for i, ann in enumerate(text_annotations):
                text = ann.get('text', '').strip()
                if not text:
                    continue

                # Try to extract area in the text (e.g., '25.5', '25,5 m2')
                area = None
                match = re.search(r'(\d+[\.,]?\d*)\s*(m2|m²|mp)?', text.lower())
                if match:
                    try:
                        area = float(match.group(1).replace(',', '.'))
                    except Exception:
                        area = None

                # Heuristically pick room name as the first token(s) before a dash or number
                room_name = text
                if '-' in text:
                    parts = [p.strip() for p in text.split('-', 1)]
                    if parts and not re.match(r'^\d', parts[0]):
                        room_name = parts[0]
                else:
                    # remove trailing numbers
                    room_name = re.sub(r'\b\d+[\.,]?\d*\b', '', room_name).strip()

                if not room_name:
                    continue

                rooms.append({
                    'room_id': f"synth_{i}",
                    'room_type': 'unknown',
                    'romanian_name': room_name,
                    'area': round(area, 2) if area else 0,
                    'dimensions': {'length': 0, 'width': 0} if not area else {'length': round(area ** 0.5, 2), 'width': round(area ** 0.5, 2)},
                    'location': ann.get('location', [0.0, 0.0]),
                    'confidence': 0.5,
                    'is_validated': False,
                    'associated_texts': [text],
                    'vertices': [],
                    'centroid': ann.get('location', [0.0, 0.0]),
                    'boundary_id': f"synth_{i}"
                })

            return rooms
        except Exception as e:
            logger.debug(f"Failed to synthesize rooms from annotations: {e}")
            return []

    def _extract_capacity_from_block(self, entity, doc: Drawing) -> Optional[float]:
        try:
            location = list(entity.dxf.insert)[:2]
            for text_entity in doc.modelspace().query('TEXT MTEXT'):
                if hasattr(text_entity.dxf, 'insert') and self._distance(location, list(text_entity.dxf.insert)[:2]) < 2.0:
                    match = re.search(r'(\d+(?:[.,]\d+)?)\s*kw', text_entity.dxf.text.lower())
                    if match: return float(match.group(1).replace(',', '.'))
        except Exception: pass
        return None

    def _extract_power_rating_from_block(self, entity, doc: Drawing) -> Optional[str]:
        try:
            location = list(entity.dxf.insert)[:2]
            for text_entity in doc.modelspace().query('TEXT MTEXT'):
                if hasattr(text_entity.dxf, 'insert') and self._distance(location, list(text_entity.dxf.insert)[:2]) < 1.0:
                    match = re.search(r'(\d+(?:[.,]\d+)?)\s*[AVW]', text_entity.dxf.text, re.IGNORECASE)
                    if match: return text_entity.dxf.text.strip()
        except Exception: pass
        return None

    def _extract_opening_dimensions(self, entity, doc: Drawing) -> Tuple[Optional[float], Optional[float]]:
        try:
            location = list(entity.dxf.insert)[:2]
            for text_entity in doc.modelspace().query('TEXT MTEXT'):
                if hasattr(text_entity.dxf, 'insert') and self._distance(location, list(text_entity.dxf.insert)[:2]) < 1.5:
                    match = re.search(r'(\d+(?:[.,]\d+)?)\s*[xX×]\s*(\d+(?:[.,]\d+)?)', text_entity.dxf.text.lower())
                    if match:
                        w, h = float(match.group(1).replace(',', '.')), float(match.group(2).replace(',', '.'))
                        if w > 10: w /= 100.0
                        if h > 10: h /= 100.0
                        return w, h
        except Exception: pass
        return None, None

    def _find_room_for_location(self, location: List[float], boundaries: List[SpatialBoundary]) -> Optional[str]:
        for boundary in boundaries:
            if self.geometry_extractor._point_in_polygon(location, boundary.vertices):
                return boundary.associated_texts[0] if boundary.associated_texts else boundary.boundary_id
        return None

    def _distance(self, p1: List[float], p2: List[float]) -> float:
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _group_electrical_components(self, components: List[ElectricalComponent]) -> List[ElectricalComponent]:
        grouped, used_indices = [], set()
        for i, comp in enumerate(components):
            if i in used_indices: continue
            group = [c for j, c in enumerate(components) if j not in used_indices and c.component_type == comp.component_type and self._distance(comp.location, c.location) < 3.0]
            if len(group) > 1:
                comp.quantity = len(group)
                used_indices.update(components.index(c) for c in group)
            grouped.append(comp)
            used_indices.add(i)
        return grouped

    def _create_room_breakdown(self, boundaries: List[SpatialBoundary], texts: List[FilteredText]) -> List[Dict[str, Any]]:
        """
        Create detailed room breakdown with dimensions calculated from vertices.
        
        Week 2 Enhancement: Added dimensions field calculation with multiple methods:
        1. Bounding box from vertices (most accurate)
        2. Dimension entity detection nearby
        3. Square root estimation from area (fallback)
        """
        room_breakdown = []
        
        for boundary in boundaries:
            room_type, confidence = "unknown", boundary.confidence
            
            # Match room type from associated texts
            if boundary.associated_texts:
                match = next((ft for ft in texts if ft.text == boundary.associated_texts[0] and ft.room_type), None)
                if match:
                    room_type = match.room_type
                    confidence = max(confidence, match.relevance_score)
            
            # ✅ NEW: Calculate dimensions
            dimensions = self._calculate_room_dimensions(boundary.vertices, boundary.area)
            
            room_breakdown.append({
                "room_id": boundary.boundary_id,
                "room_type": room_type,
                "romanian_name": self._get_romanian_room_name(room_type),
                "area": round(boundary.area, 2),
                "dimensions": dimensions,  # ✅ NEW FIELD
                "location": boundary.centroid,
                "confidence": round(confidence, 2),
                "is_validated": bool(boundary.associated_texts),
                "associated_texts": boundary.associated_texts,
                # Keep geometric data for GCS storage
                "vertices": boundary.vertices,
                "centroid": boundary.centroid,
                "boundary_id": boundary.boundary_id
            })
        
        return room_breakdown

    def _calculate_room_dimensions(self, vertices: List[List[float]], area: float) -> Dict[str, float]:
        """
        Calculate room dimensions using multiple approaches.
        
        Priority:
        1. Bounding box from vertices (most accurate for rectangles)
        2. Square root estimation from area (fallback)
        
        Args:
            vertices: List of [x, y] vertex coordinates
            area: Room area in square meters
        
        Returns:
            Dictionary with length and width in meters
        """
        try:
            if not vertices or len(vertices) < 3:
                # Fallback: estimate from area
                estimated_side = round((area ** 0.5), 2)
                return {"length": estimated_side, "width": estimated_side}
            
            # Method 1: Calculate bounding box
            x_coords = [v[0] for v in vertices]
            y_coords = [v[1] for v in vertices]
            
            length = round(max(x_coords) - min(x_coords), 2)
            width = round(max(y_coords) - min(y_coords), 2)
            
            # Validate: ensure dimensions are reasonable
            if length > 0 and width > 0:
                # Check if calculated area matches actual area (within 20% tolerance)
                calculated_area = length * width
                area_ratio = abs(calculated_area - area) / area if area > 0 else 1.0
                
                if area_ratio < 0.20:  # Within 20% tolerance
                    return {"length": length, "width": width}
            
            # Method 2: Fallback to square estimation from area
            estimated_side = round((area ** 0.5), 2)
            return {"length": estimated_side, "width": estimated_side}
            
        except Exception as e:
            logger.warning(f"Failed to calculate dimensions: {e}")
            # Ultimate fallback
            estimated_side = round((area ** 0.5), 2) if area > 0 else 1.0
            return {"length": estimated_side, "width": estimated_side}

    def _create_summary_room_breakdown(self, room_breakdown: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create simplified room breakdown for Firestore storage (analysis_summary).
        
        Removes geometric details (vertices, centroids) but keeps:
        - room_name (for display)
        - area
        - dimensions {length, width}
        - confidence (for quality assessment)
        - associated_texts (for validation)
        
        This format is used by:
        1. Unified Construction Agent (conversational context)
        2. Drawing Generation Agent (text-based extraction)
        
        Returns:
            Simplified list of rooms suitable for Firestore
        """
        summary = []
        
        for room in room_breakdown:
            if not isinstance(room, dict):
                continue
            
            # Use romanian_name for user-friendly display
            room_name = room.get("romanian_name", room.get("room_type", "Necunoscut"))
            area = room.get("area", 0)
            
            if area <= 0:
                continue
            
            # Get dimensions (already calculated in _create_room_breakdown)
            dimensions = room.get("dimensions", {})
            if not dimensions:
                # Fallback if somehow missing
                estimated_side = round((area ** 0.5), 2)
                dimensions = {"length": estimated_side, "width": estimated_side}
            
            summary.append({
                "room_name": room_name,  # ✅ Changed from room_type
                "area": area,
                "dimensions": {
                    "length": dimensions.get("length", 0),
                    "width": dimensions.get("width", 0)
                },
                "confidence": room.get("confidence", 0.5),  # ✅ KEPT
                "associated_texts": room.get("associated_texts", [])  # ✅ KEPT
            })
        
        return summary

    def _detect_electrical_elements(self, doc: Drawing) -> bool:
        kw = ['electric', 'power', 'lighting', 'outlet', 'switch']
        for entity in doc.modelspace().query('INSERT TEXT MTEXT'):
            name = (getattr(entity.dxf, 'name', '') or getattr(entity.dxf, 'text', '')).lower()
            if any(k in name for k in kw): return True
        return False

    def _detect_hvac_elements(self, doc: Drawing) -> bool:
        kw = ['hvac', 'ventilation', 'air', 'conditioning', 'climate']
        for entity in doc.modelspace().query('TEXT MTEXT'):
            name = (getattr(entity.dxf, 'name', '') or getattr(entity.dxf, 'text', '')).lower()
            if any(k in name for k in kw): return True
        return False

    def _calculate_overall_confidence(self, boundaries: List[SpatialBoundary], texts: List[FilteredText], doc_class: DocumentClassification, rooms: int, area: float, specs: SpecificationData) -> float:
        confidence = doc_class.confidence * 20
        if boundaries: confidence += (sum(b.confidence for b in boundaries) / len(boundaries)) * 30
        if rooms > 0: confidence += min(rooms * 5, 25)
        relevant_texts = [t for t in texts if t.relevance_score > 0.5]
        if relevant_texts: confidence += (sum(t.relevance_score for t in relevant_texts) / len(relevant_texts)) * 15
        if 10 <= area <= 1000: confidence += 10
        if specs: confidence += specs.specification_confidence * 15
        return min(confidence, 100.0)

    def _generate_enhanced_summary(self, total_rooms: int, total_area: float, room_breakdown: List[Dict], doc_class: DocumentClassification, has_dim: bool, has_elec: bool, has_hvac: bool, spec_data: Optional[SpecificationData]) -> str:
        parts = []
        if doc_class.document_type == DocumentType.FLOOR_PLAN: parts.append("Plan arhitectural")
        elif spec_data and spec_data.total_specifications > 0: parts.append(f"Foaie specificații ({spec_data.total_specifications} elem.)")
        if doc_class.project_type == ProjectType.COMMERCIAL: parts.append("comercial")
        elif doc_class.project_type == ProjectType.RESIDENTIAL: parts.append("rezidențial")
        if total_rooms > 0: parts.append(f"{total_rooms} spații")
        if total_area > 0: parts.append(f"{total_area:.0f} mp")
        
        room_types = {r.get('romanian_name', 'necunoscut'): 0 for r in room_breakdown}
        for r in room_breakdown: room_types[r.get('romanian_name', 'necunoscut')] += 1
        if any(v > 0 for k, v in room_types.items() if k != 'necunoscut'):
            parts.append(f"tipuri: {', '.join([f'{v}x{k}' if v > 1 else k for k,v in room_types.items() if k != 'necunoscut' and v > 0][:3])}")
        
        tech = [p for p, c in [("cotat", has_dim), ("electric", has_elec), ("HVAC", has_hvac)] if c]
        if tech: parts.append(f"({', '.join(tech)})")
        
        if doc_class.primary_language: parts.append(f"[{doc_class.primary_language[:2].upper()}]")
        return " - ".join(parts) if parts else "Document analizat"

    def _get_romanian_room_name(self, room_type: str) -> str:
        romanian_names = {
            'bedroom': 'dormitor', 'kitchen': 'bucătărie', 'bathroom': 'baie',
            'living_room': 'living', 'dining_room': 'sufragerie', 'hallway': 'hol',
            'storage': 'debara', 'office': 'birou', 'balcony': 'balcon',
            'garage': 'garaj', 'basement': 'subsol', 'commercial': 'spațiu comercial',
            'unknown': 'necunoscut'
        }
        return romanian_names.get(room_type, room_type)