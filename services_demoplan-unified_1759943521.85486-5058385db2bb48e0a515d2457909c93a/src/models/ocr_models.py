"""
DemoPLAN OCR Data Models
Structured data classes for OCR processing results
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class TableCell:
    """
    Represents a single cell in an extracted table
    """
    text: str
    row_index: int
    col_index: int
    confidence: float = 0.0
    colspan: int = 1
    rowspan: int = 1
    is_header: bool = False


@dataclass
class TableRow:
    """
    Represents a row in an extracted table
    """
    cells: List[TableCell] = field(default_factory=list)
    row_index: int = 0
    is_header: bool = False
    
    def get_values(self) -> List[str]:
        """Get text values from all cells in row"""
        return [cell.text for cell in self.cells]


@dataclass
class TableData:
    """
    Represents a complete extracted table with metadata
    """
    headers: List[str] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)
    confidence: float = 0.0
    page_number: int = 0
    table_type: Optional[str] = None  # 'cost_breakdown', 'specifications', 'materials', 'timeline'
    
    # Table structure metadata
    num_rows: int = 0
    num_columns: int = 0
    
    # Romanian construction-specific classification
    is_cost_table: bool = False
    is_specification_table: bool = False
    is_material_list: bool = False
    
    def __post_init__(self):
        """Calculate metadata after initialization"""
        self.num_rows = len(self.rows)
        self.num_columns = len(self.headers) if self.headers else (
            len(self.rows[0]) if self.rows else 0
        )
        
        # Auto-classify based on table_type
        if self.table_type:
            self.is_cost_table = self.table_type == 'cost_breakdown'
            self.is_specification_table = self.table_type == 'specifications'
            self.is_material_list = self.table_type == 'materials'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert table to dictionary format"""
        return {
            "headers": self.headers,
            "rows": self.rows,
            "confidence": self.confidence,
            "page_number": self.page_number,
            "table_type": self.table_type,
            "num_rows": self.num_rows,
            "num_columns": self.num_columns,
            "is_cost_table": self.is_cost_table,
            "is_specification_table": self.is_specification_table,
            "is_material_list": self.is_material_list
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TableData':
        """Reconstruct TableData from dictionary"""
        return cls(
            headers=data.get("headers", []),
            rows=data.get("rows", []),
            confidence=data.get("confidence", 0.0),
            page_number=data.get("page_number", 0),
            table_type=data.get("table_type"),
            num_rows=data.get("num_rows", 0),
            num_columns=data.get("num_columns", 0),
            is_cost_table=data.get("is_cost_table", False),
            is_specification_table=data.get("is_specification_table", False),
            is_material_list=data.get("is_material_list", False)
        )


@dataclass
class HandwritingResult:
    """
    Results from handwriting detection in document
    """
    detected: bool = False
    confidence: float = 0.0
    text_blocks: List[str] = field(default_factory=list)
    requires_manual_review: bool = False
    page_numbers: List[int] = field(default_factory=list)
    
    # Additional metadata
    num_blocks: int = 0
    avg_confidence: float = 0.0
    
    def __post_init__(self):
        """Calculate metadata"""
        self.num_blocks = len(self.text_blocks)
        if self.num_blocks > 0:
            self.avg_confidence = self.confidence
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "detected": self.detected,
            "confidence": self.confidence,
            "text_blocks": self.text_blocks,
            "num_blocks": self.num_blocks,
            "requires_manual_review": self.requires_manual_review,
            "page_numbers": self.page_numbers,
            "avg_confidence": self.avg_confidence
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HandwritingResult':
        """Reconstruct HandwritingResult from dictionary"""
        return cls(
            detected=data.get("detected", False),
            confidence=data.get("confidence", 0.0),
            text_blocks=data.get("text_blocks", []),
            requires_manual_review=data.get("requires_manual_review", False),
            page_numbers=data.get("page_numbers", []),
            num_blocks=data.get("num_blocks", 0),
            avg_confidence=data.get("avg_confidence", 0.0)
        )


@dataclass
class ExtractedEntity:
    """
    Single extracted entity (date, price, measurement)
    """
    type: str  # 'date', 'price', 'measurement'
    text: str
    confidence: float
    normalized_value: Optional[Any] = None  # Parsed/normalized value
    page_number: int = 0
    context: Optional[str] = None  # Surrounding text for context


@dataclass
class EntityExtraction:
    """
    Collection of extracted entities from OCR
    """
    dates: List[Dict[str, Any]] = field(default_factory=list)
    prices: List[Dict[str, Any]] = field(default_factory=list)
    measurements: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    
    # Romanian construction-specific entities
    deadlines: List[Dict[str, Any]] = field(default_factory=list)
    material_quantities: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Calculate overall confidence"""
        all_entities = self.dates + self.prices + self.measurements
        if all_entities:
            confidences = [e.get('confidence', 0.0) for e in all_entities]
            self.confidence = sum(confidences) / len(confidences)
    
    @property
    def total_entities(self) -> int:
        """Total number of entities extracted"""
        return len(self.dates) + len(self.prices) + len(self.measurements)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "dates": self.dates,
            "prices": self.prices,
            "measurements": self.measurements,
            "confidence": self.confidence,
            "total_entities": self.total_entities,
            "deadlines": self.deadlines,
            "material_quantities": self.material_quantities
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EntityExtraction':
        """Reconstruct EntityExtraction from dictionary"""
        return cls(
            dates=data.get("dates", []),
            prices=data.get("prices", []),
            measurements=data.get("measurements", []),
            confidence=data.get("confidence", 0.0),
            deadlines=data.get("deadlines", []),
            material_quantities=data.get("material_quantities", [])
        )


@dataclass
class DocumentQualityScore:
    """
    Overall quality assessment of OCR processing
    """
    overall_score: float = 0.0  # 0-1
    text_clarity: float = 0.0
    table_quality: float = 0.0
    entity_confidence: float = 0.0
    
    # Quality indicators
    is_high_quality: bool = False
    is_acceptable: bool = False
    requires_review: bool = False
    
    # Specific issues
    low_text_density: bool = False
    handwriting_present: bool = False
    poor_scan_quality: bool = False
    
    def __post_init__(self):
        """Determine quality levels"""
        self.is_high_quality = self.overall_score >= 0.85
        self.is_acceptable = self.overall_score >= 0.65
        self.requires_review = self.overall_score < 0.65
    
    def get_quality_label(self) -> str:
        """Get human-readable quality label"""
        if self.is_high_quality:
            return "High Quality"
        elif self.is_acceptable:
            return "Acceptable"
        else:
            return "Requires Review"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentQualityScore':
        """Reconstruct DocumentQualityScore from dictionary"""
        return cls(
            overall_score=data.get("overall_score", 0.0),
            text_clarity=data.get("text_clarity", 0.0),
            table_quality=data.get("table_quality", 0.0),
            entity_confidence=data.get("entity_confidence", 0.0),
            is_high_quality=data.get("is_high_quality", False),
            is_acceptable=data.get("is_acceptable", False),
            requires_review=data.get("requires_review", False),
            low_text_density=data.get("low_text_density", False),
            handwriting_present=data.get("handwriting_present", False),
            poor_scan_quality=data.get("poor_scan_quality", False)
        )


@dataclass
class OCRMetrics:
    """
    Processing metrics for OCR operation
    """
    processing_time_ms: float = 0.0
    page_count: int = 0
    text_length: int = 0
    tables_extracted: int = 0
    entities_extracted: int = 0
    
    # Cost tracking
    cost_estimate_usd: float = 0.0
    cost_per_page: float = 0.015
    
    # Performance
    avg_time_per_page_ms: float = 0.0
    
    # Timestamp
    processed_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Calculate derived metrics"""
        if self.page_count > 0:
            self.avg_time_per_page_ms = self.processing_time_ms / self.page_count
        
        if not self.processed_at:
            self.processed_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "processing_time_ms": self.processing_time_ms,
            "page_count": self.page_count,
            "text_length": self.text_length,
            "tables_extracted": self.tables_extracted,
            "entities_extracted": self.entities_extracted,
            "cost_estimate_usd": self.cost_estimate_usd,
            "avg_time_per_page_ms": self.avg_time_per_page_ms,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OCRMetrics':
        """Reconstruct OCRMetrics from dictionary"""
        processed_at_str = data.get("processed_at")
        processed_at = None
        if processed_at_str:
            try:
                # Handle 'Z' for UTC timezone
                processed_at = datetime.fromisoformat(processed_at_str.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                processed_at = None
        
        return cls(
            processing_time_ms=data.get("processing_time_ms", 0.0),
            page_count=data.get("page_count", 0),
            text_length=data.get("text_length", 0),
            tables_extracted=data.get("tables_extracted", 0),
            entities_extracted=data.get("entities_extracted", 0),
            cost_estimate_usd=data.get("cost_estimate_usd", 0.0),
            avg_time_per_page_ms=data.get("avg_time_per_page_ms", 0.0),
            processed_at=processed_at
        )


@dataclass
class OCRResult:
    """
    Complete OCR processing result with all extracted data
    """
    # Core OCR output
    text: str
    confidence: float
    page_count: int
    
    # Structured data
    tables: List[TableData] = field(default_factory=list)
    handwriting: Optional[HandwritingResult] = None
    entities: Optional[EntityExtraction] = None
    
    # Metrics and quality
    processing_time_ms: float = 0.0
    cost_estimate: float = 0.0
    text_density_score: float = 0.0
    quality_score: Optional[DocumentQualityScore] = None
    
    # Warnings and issues
    quality_warnings: List[str] = field(default_factory=list)
    
    # Metadata
    ocr_method: str = "document_ai"
    language_detected: str = "ro"
    metrics: Optional[OCRMetrics] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert complete OCR result to dictionary (COMPLETE serialization)"""
        return {
            "text": self.text,
            "confidence": self.confidence,
            "page_count": self.page_count,
            "tables": [t.to_dict() for t in self.tables],
            "handwriting": self.handwriting.to_dict() if self.handwriting else None,
            "entities": self.entities.to_dict() if self.entities else None,
            "processing_time_ms": self.processing_time_ms,
            "cost_estimate": self.cost_estimate,
            "text_density_score": self.text_density_score,
            "quality_score": self.quality_score.to_dict() if self.quality_score else None,
            "quality_warnings": self.quality_warnings,
            "ocr_method": self.ocr_method,
            "language_detected": self.language_detected,
            "metrics": self.metrics.to_dict() if self.metrics else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OCRResult':
        """Reconstruct complete OCRResult from dictionary"""
        # Reconstruct nested objects
        tables = [TableData.from_dict(t) for t in data.get("tables", [])]
        
        handwriting_data = data.get("handwriting")
        handwriting = HandwritingResult.from_dict(handwriting_data) if handwriting_data else None
        
        entities_data = data.get("entities")
        entities = EntityExtraction.from_dict(entities_data) if entities_data else None
        
        quality_score_data = data.get("quality_score")
        quality_score = DocumentQualityScore.from_dict(quality_score_data) if quality_score_data else None
        
        metrics_data = data.get("metrics")
        metrics = OCRMetrics.from_dict(metrics_data) if metrics_data else None
        
        return cls(
            text=data.get("text", ""),
            confidence=data.get("confidence", 0.0),
            page_count=data.get("page_count", 0),
            tables=tables,
            handwriting=handwriting,
            entities=entities,
            processing_time_ms=data.get("processing_time_ms", 0.0),
            cost_estimate=data.get("cost_estimate", 0.0),
            text_density_score=data.get("text_density_score", 0.0),
            quality_score=quality_score,
            quality_warnings=data.get("quality_warnings", []),
            ocr_method=data.get("ocr_method", "document_ai"),
            language_detected=data.get("language_detected", "ro"),
            metrics=metrics
        )
    
    def get_summary(self) -> str:
        """Get human-readable summary of OCR results"""
        summary_parts = [
            f"OCR Processing Complete:",
            f"- Pages: {self.page_count}",
            f"- Confidence: {self.confidence:.1%}",
            f"- Text extracted: {len(self.text)} characters",
            f"- Tables found: {len(self.tables)}"
        ]
        
        if self.entities:
            summary_parts.append(
                f"- Entities: {self.entities.total_entities} "
                f"({len(self.entities.dates)} dates, "
                f"{len(self.entities.prices)} prices, "
                f"{len(self.entities.measurements)} measurements)"
            )
        
        if self.handwriting and self.handwriting.detected:
            summary_parts.append(
                f"- ⚠️ Handwriting detected ({self.handwriting.num_blocks} blocks)"
            )
        
        if self.quality_warnings:
            summary_parts.append(f"- ⚠️ {len(self.quality_warnings)} warnings")
        
        summary_parts.append(f"- Processing time: {self.processing_time_ms:.0f}ms")
        summary_parts.append(f"- Estimated cost: ${self.cost_estimate:.4f}")
        
        return "\n".join(summary_parts)