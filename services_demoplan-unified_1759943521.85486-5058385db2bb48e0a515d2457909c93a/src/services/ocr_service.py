"""
DemoPLAN OCR Service
Google Document AI integration for scanned PDF processing
"""

import logging
import uuid
from datetime import datetime
import io
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from google.cloud import documentai_v1 as documentai
from google.api_core.client_options import ClientOptions

logger = logging.getLogger("demoplan.services.ocr")


@dataclass
class TableCell:
    """Represents a cell in an extracted table"""
    text: str
    row_index: int
    col_index: int
    confidence: float = 0.0


@dataclass
class TableData:
    """Represents an extracted table from OCR"""
    headers: List[str] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)
    confidence: float = 0.0
    page_number: int = 0
    table_type: Optional[str] = None  # 'cost_breakdown', 'specifications', 'materials', etc.


@dataclass
class HandwritingResult:
    """Results from handwriting detection"""
    detected: bool = False
    confidence: float = 0.0
    text_blocks: List[str] = field(default_factory=list)
    requires_manual_review: bool = False


@dataclass
class EntityExtraction:
    """Extracted entities from OCR (dates, prices, measurements)"""
    dates: List[Dict[str, Any]] = field(default_factory=list)
    prices: List[Dict[str, Any]] = field(default_factory=list)
    measurements: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class OCRResult:
    """Complete OCR processing result"""
    text: str
    confidence: float
    page_count: int
    tables: List[TableData] = field(default_factory=list)
    handwriting: Optional[HandwritingResult] = None
    entities: Optional[EntityExtraction] = None
    processing_time_ms: float = 0.0
    cost_estimate: float = 0.0
    text_density_score: float = 0.0
    quality_warnings: List[str] = field(default_factory=list)


class OCRService:
    """
    Google Document AI OCR Service for processing scanned construction documents
    
    Handles:
    - Scanned PDF text extraction
    - Table extraction (cost breakdowns, specifications)
    - Handwriting detection
    - Entity extraction (dates, prices, measurements)
    - Cost estimation and monitoring
    """
    
    def __init__(
        self,
        processor_id: str = "f6c4f619b5a674de",
        project_id: str = "1041867695241",
        location: str = "eu"
    ):
        """
        Initialize OCR service with Document AI processor
        
        Args:
            processor_id: Document AI processor ID
            project_id: GCP project ID (numeric)
            location: Processor location (eu, us, etc.)
        """
        self.processor_id = processor_id
        self.project_id = project_id
        self.location = location
        
        # Document AI configuration
        self.processor_name = (
            f"projects/{project_id}/locations/{location}/"
            f"processors/{processor_id}"
        )
        
        # Create client with location-specific endpoint
        opts = ClientOptions(
            api_endpoint=f"{location}-documentai.googleapis.com"
        )
        self.client = documentai.DocumentProcessorServiceClient(
            client_options=opts
        )
        
        # OCR configuration
        self.min_text_density = 100  # chars per page
        self.table_confidence_threshold = 0.7
        self.handwriting_confidence_threshold = 0.6
        self.cost_per_page = 0.015  # USD per page
        
        logger.info(
            f"🔍 OCR Service initialized - Processor: {processor_id}, "
            f"Location: {location}"
        )
    
    def detect_scanned_document(
        self, 
        text: str, 
        page_count: int
    ) -> Tuple[bool, float]:
        """
        Detect if a document is scanned (low text density)
        
        Args:
            text: Extracted text from traditional methods
            page_count: Number of pages in document
            
        Returns:
            Tuple of (is_scanned, text_density_score)
        """
        if page_count == 0:
            return True, 0.0
        
        text_length = len(text.strip())
        density = text_length / page_count
        
        is_scanned = density < self.min_text_density
        
        logger.info(
            f"📊 Text density: {density:.1f} chars/page "
            f"(threshold: {self.min_text_density})"
        )
        
        return is_scanned, density
    
    async def process_pdf_with_ocr(
        self,
        file_content: bytes,
        language_hints: Optional[List[str]] = None,
        filename: Optional[str] = None  # ✅ ADD THIS
    ) -> Tuple[OCRResult, str]:  # ✅ CHANGE RETURN TYPE
        """
        Process PDF with OCR and return result + file_id
        
        Args:
            file_content: PDF file bytes
            language_hints: Optional language codes (e.g., ['ro', 'en'])
            
        Returns:
            A tuple containing the OCRResult and the generated file_id
        """
        start_time = datetime.now()
        
        try:
            # ✅ ADD THIS LINE at the very start
            file_id = str(uuid.uuid4())
            logger.info(f"📋 Generated file_id: {file_id}")
            logger.info("🔍 Starting Document AI OCR processing...")
            
            # Prepare document for processing
            raw_document = documentai.RawDocument(
                content=file_content,
                mime_type="application/pdf"
            )
            
            # Configure processing request
            request = documentai.ProcessRequest(
                name=self.processor_name,
                raw_document=raw_document
            )
            
            # Process document
            result = self.client.process_document(request=request)
            document = result.document
            
            # Extract text
            text = document.text
            page_count = len(document.pages)
            
            # Calculate overall confidence
            confidence = self._calculate_overall_confidence(document)
            
            # Extract tables
            tables = self._extract_tables_from_document(document)
            
            # Detect handwriting
            handwriting = self._detect_handwriting(document)
            
            # Extract entities
            entities = self._extract_entities(document)
            
            # Calculate text density
            text_density = len(text) / page_count if page_count > 0 else 0
            
            # Estimate cost
            cost = self.estimate_cost(page_count)
            
            # Quality warnings
            warnings = self._generate_quality_warnings(
                confidence, text_density, handwriting
            )
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            logger.info(
                f"✅ OCR completed: {page_count} pages, "
                f"{len(tables)} tables, "
                f"{confidence:.1%} confidence, "
                f"{processing_time:.0f}ms"
            )

            ocr_result = OCRResult(
                text=text,
                confidence=confidence,
                page_count=page_count,
                tables=tables,
                handwriting=handwriting,
                entities=entities,
                processing_time_ms=processing_time,
                cost_estimate=cost.get("total_cost", 0.0),
                text_density_score=text_density,
                quality_warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"❌ OCR processing failed: {e}")
            raise

        # ✅ CHANGE THIS LINE - return tuple instead of just result
        return ocr_result, file_id  # OLD: return ocr_result
    def _calculate_overall_confidence(
        self, 
        document: documentai.Document
    ) -> float:
        """Calculate overall document confidence from page confidences"""
        if not document.pages:
            return 0.0
        
        # Average confidence across all pages
        confidences = []
        for page in document.pages:
            if hasattr(page, 'confidence') and page.confidence:
                confidences.append(page.confidence)
        
        if not confidences:
            return 0.5  # Default confidence if not available
        
        return sum(confidences) / len(confidences)
    
    def _extract_tables_from_document(
        self, 
        document: documentai.Document
    ) -> List[TableData]:
        """Extract tables from Document AI response"""
        tables = []
        
        for page_idx, page in enumerate(document.pages):
            if not hasattr(page, 'tables'):
                continue
            
            for table in page.tables:
                table_data = self._parse_table(table, document.text, page_idx)
                if table_data:
                    tables.append(table_data)
        
        logger.info(f"📊 Extracted {len(tables)} tables from document")
        return tables
    
    def _parse_table(
        self,
        table: documentai.Document.Page.Table,
        full_text: str,
        page_number: int
    ) -> Optional[TableData]:
        """Parse a single table into structured format"""
        try:
            # Extract headers (first row)
            headers = []
            rows = []
            
            # Group cells by row
            rows_dict = {}
            for cell in table.header_rows[0].cells if table.header_rows else []:
                text = self._get_text_from_layout(cell.layout, full_text)
                headers.append(text.strip())
            
            # Extract data rows
            for row in table.body_rows:
                row_data = []
                for cell in row.cells:
                    text = self._get_text_from_layout(cell.layout, full_text)
                    row_data.append(text.strip())
                rows.append(row_data)
            
            # Calculate table confidence
            confidence = 0.8  # Default for Document AI tables
            
            # Classify table type
            table_type = self._classify_table_type(headers, rows)
            
            return TableData(
                headers=headers,
                rows=rows,
                confidence=confidence,
                page_number=page_number,
                table_type=table_type
            )
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to parse table: {e}")
            return None
    
    def _get_text_from_layout(
        self,
        layout: documentai.Document.Page.Layout,
        full_text: str
    ) -> str:
        """Extract text from a layout element"""
        if not layout.text_anchor or not layout.text_anchor.text_segments:
            return ""
        
        text = ""
        for segment in layout.text_anchor.text_segments:
            start_index = int(segment.start_index) if hasattr(segment, 'start_index') else 0
            end_index = int(segment.end_index) if hasattr(segment, 'end_index') else 0
            text += full_text[start_index:end_index]
        
        return text
    
    def _classify_table_type(
        self,
        headers: List[str],
        rows: List[List[str]]
    ) -> str:
        """Classify table type based on headers and content"""
        header_text = " ".join(headers).lower()
        
        if any(word in header_text for word in ['cost', 'preț', 'sumă', 'total']):
            return 'cost_breakdown'
        elif any(word in header_text for word in ['specificații', 'specs', 'cerințe']):
            return 'specifications'
        elif any(word in header_text for word in ['material', 'cantitate', 'quantity']):
            return 'materials'
        else:
            return 'general'
    
    def _detect_handwriting(
        self,
        document: documentai.Document
    ) -> HandwritingResult:
        """Detect handwritten content in document"""
        handwriting_blocks = []
        total_confidence = 0.0
        block_count = 0
        
        for page in document.pages:
            if not hasattr(page, 'paragraphs'):
                continue
            
            for paragraph in page.paragraphs:
                # Check if paragraph has handwriting detection
                if hasattr(paragraph, 'detected_languages'):
                    for lang in paragraph.detected_languages:
                        if hasattr(lang, 'confidence'):
                            # Low confidence might indicate handwriting
                            if lang.confidence < self.handwriting_confidence_threshold:
                                text = self._get_text_from_layout(
                                    paragraph.layout, 
                                    document.text
                                )
                                handwriting_blocks.append(text)
                                total_confidence += lang.confidence
                                block_count += 1
        
        detected = len(handwriting_blocks) > 0
        avg_confidence = total_confidence / block_count if block_count > 0 else 0.0
        requires_review = avg_confidence < 0.5
        
        if detected:
            logger.warning(
                f"✍️ Handwriting detected: {len(handwriting_blocks)} blocks, "
                f"{avg_confidence:.1%} confidence"
            )
        
        return HandwritingResult(
            detected=detected,
            confidence=avg_confidence,
            text_blocks=handwriting_blocks,
            requires_manual_review=requires_review
        )
    
    def _extract_entities(
        self,
        document: documentai.Document
    ) -> EntityExtraction:
        """Extract entities (dates, prices, measurements) from document"""
        dates = []
        prices = []
        measurements = []
        
        if not hasattr(document, 'entities'):
            return EntityExtraction()
        
        for entity in document.entities:
            entity_type = entity.type_
            confidence = entity.confidence if hasattr(entity, 'confidence') else 0.0
            mention_text = entity.mention_text if hasattr(entity, 'mention_text') else ""
            
            entity_data = {
                "text": mention_text,
                "confidence": confidence
            }
            
            # Classify entities
            if entity_type in ['date', 'DATE']:
                dates.append(entity_data)
            elif entity_type in ['money', 'MONEY', 'price', 'PRICE']:
                prices.append(entity_data)
            elif entity_type in ['measurement', 'MEASUREMENT', 'quantity']:
                measurements.append(entity_data)
        
        avg_confidence = 0.0
        total_entities = len(dates) + len(prices) + len(measurements)
        if total_entities > 0:
            all_confidences = [e['confidence'] for e in dates + prices + measurements]
            avg_confidence = sum(all_confidences) / len(all_confidences)
        
        logger.info(
            f"🔍 Entities extracted: {len(dates)} dates, "
            f"{len(prices)} prices, {len(measurements)} measurements"
        )
        
        return EntityExtraction(
            dates=dates,
            prices=prices,
            measurements=measurements,
            confidence=avg_confidence
        )
    
    def estimate_cost(self, page_count: int) -> Dict[str, float]:
        """
        Estimate processing cost for OCR
        
        Args:
            page_count: Number of pages to process
            
        Returns:
            Dict with cost breakdown
        """
        base_cost = page_count * self.cost_per_page
        
        return {
            "page_count": page_count,
            "cost_per_page": self.cost_per_page,
            "total_cost": base_cost,
            "currency": "USD"
        }
    
    def _generate_quality_warnings(
        self,
        confidence: float,
        text_density: float,
        handwriting: HandwritingResult
    ) -> List[str]:
        """Generate quality warnings based on OCR results"""
        warnings = []
        
        if confidence < 0.7:
            warnings.append(
                f"Low OCR confidence ({confidence:.1%}). "
                "Results may require manual verification."
            )
        
        if text_density < 50:
            warnings.append(
                f"Very low text density ({text_density:.0f} chars/page). "
                "Document may be mostly images or diagrams."
            )
        
        if handwriting and handwriting.requires_manual_review:
            warnings.append(
                "Handwriting detected with low confidence. "
                "Manual review recommended."
            )
        
        return warnings
    
    async def extract_tables(
        self,
        file_content: bytes
    ) -> List[TableData]:
        """
        Dedicated table extraction method
        
        Args:
            file_content: PDF file bytes
            
        Returns:
            List of extracted tables
        """
        result = await self.process_pdf_with_ocr(file_content)
        return result.tables