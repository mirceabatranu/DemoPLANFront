"""
Offer Data Models - Complete Type-Safe Structures
Handles both summary (Imperial/Beautik) and detailed (CCC) offer formats
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from enum import Enum


# ============================================================================
# ENUMS & CONSTANTS
# ============================================================================

class DetailLevel(str, Enum):
    """Level of detail in parsed offer"""
    SUMMARY = "summary"          # High-level totals only (Imperial, Beautik)
    UNIT_PRICES = "unit_prices"  # Full breakdown with quantities (CCC)


class CategoryType(str, Enum):
    """Standard category classifications"""
    A_ARCHITECTURAL = "A"
    B_MEP = "B"
    C_PROFESSIONAL = "C"
    UNKNOWN = "UNKNOWN"


# Validation constants
VALIDATION_RULES = {
    'min_total_eur': 1_000,
    'max_total_eur': 10_000_000,
    'min_unit_price': 0.01,
    'max_unit_price': 100_000,
    'category_tolerance': 0.02  # 2% tolerance for category sum validation
}

# Unit normalization mapping (Romanian/English only)
UNIT_NORMALIZATION = {
    'm2': ['m2', 'm²', 'sqm', 'sq m', 'mp'],
    'm': ['m', 'mb', 'lm', 'ml', 'linear m'],
    'unit': ['szt', 'pcs', 'pc', 'unit', 'ea', 'buc', 'bucata'],
    'set': ['kpl', 'set', 'kit', 'complet']
}

# Item type classification keywords (Romanian/English)
ITEM_TYPE_KEYWORDS = {
    # ARCHITECTURAL (Category A)
    'demolition': [
        'demolition', 'demolare', 'removal', 'indepartare', 
        'dismantling', 'demontare', 'hammering', 'skucie'
    ],
    'construction': [
        'construction', 'constructie', 'installation works', 
        'lucrari constructii', 'walls', 'pereti', 'zidarie', 'bricklaying'
    ],
    'flooring': [
        'flooring', 'pardoseala', 'posadzka', 'floor', 
        'tiles', 'gresie', 'faianta', 'tiling', 'levelling', 'screed'
    ],
    'ceiling': [
        'ceiling', 'tavan', 'sufit', 'suspended', 'faux plafond',
        'drop ceiling', 'plaster board', 'gips carton'
    ],
    'doors_windows': [
        'door', 'usa', 'drzwi', 'window', 'fereastra', 'okna',
        'glazing', 'sticla', 'glass wall'
    ],
    'furniture': [
        'furniture', 'mobilier', 'meble', 'cabinets', 'dulapuri',
        'custom made', 'ready made', 'joinery', 'tamplarie'
    ],
    'finishes': [
        'painting', 'vopsitorie', 'plastering', 'tencuiala',
        'finishes', 'finisaje', 'cladding', 'placare'
    ],
    'glass_partitions': [
        'glass', 'sticla', 'szklane', 'partitions', 'pereti mobili',
        'demountable', 'transparent'
    ],
    'signage_branding': [
        'signage', 'semnal', 'branding', 'logo', 'graphics', 
        'wayfinding', 'signalizare'
    ],
    'greenery': [
        'greenery', 'plante', 'plants', 'landscaping', 
        'vegetation', 'green wall'
    ],
    'blinds': [
        'blinds', 'jaluzele', 'curtains', 'perdele', 
        'shades', 'rolouri', 'window treatment'
    ],
    'equipment': [
        'equipment', 'echipamente', 'appliances', 'aparate',
        'fixtures', 'fitting'
    ],
    
    # MEP (Category B)
    'hvac': [
        'hvac', 'ventilation', 'ventilatie', 'wentylacja',
        'air conditioning', 'climatizare', 'heating', 'incalzire',
        'cooling', 'racire', 'air handling', 'vav'
    ],
    'sanitary': [
        'sanitary', 'sanitare', 'plumbing', 'instalatii sanitare',
        'drainage', 'canalizare', 'water supply', 'alimentare apa',
        'sewage', 'apa uzata'
    ],
    'electrical_strong': [
        'electrical', 'electric', 'electrica', 'elektryka',
        'power', 'energie', 'strong current', 'curenti tari',
        'distribution', 'tablou electric', 'switchgear'
    ],
    'electrical_weak': [
        'low current', 'curenti slabi', 'weak current',
        'data', 'date', 'utp', 'cat 6', 'networking', 'retea',
        'telecom', 'telecomunicatii', 'structured cabling'
    ],
    'lighting': [
        'lighting', 'iluminat', 'oswietlenie', 'lamps', 'lampi',
        'luminaires', 'corpuri iluminat', 'led', 'emergency lighting'
    ],
    'fire_protection': [
        'fire', 'incendiu', 'sprinkler', 'stingere', 
        'fire alarm', 'detectie incendiu', 'fire fighting',
        'smoke detection', 'detectie fum'
    ],
    'bms': [
        'bms', 'building management', 'management tehnic',
        'automation', 'automatizare', 'controls', 'comenzi',
        'scada', 'monitoring'
    ],
    'security': [
        'security', 'securitate', 'access control', 'control acces',
        'cctv', 'supraveghere', 'alarm', 'alarma', 'intrusion'
    ],
    'av_systems': [
        'audio visual', 'av', 'sound', 'sunet', 'video',
        'projection', 'proiectie', 'conferencing', 'multimedia'
    ],
    
    # RESTAURANT/HOSPITALITY SPECIFIC
    'kitchen_equipment': [
        'kitchen', 'bucatarie', 'cooking', 'gatit',
        'commercial kitchen', 'bucatarie profesionala',
        'catering equipment', 'echipamente catering'
    ],
    'kitchen_extraction': [
        'extraction', 'extractie', 'hood', 'hota',
        'exhaust', 'evacuare', 'kitchen ventilation',
        'grease trap', 'separator grasimi'
    ],
    'refrigeration': [
        'refrigeration', 'refrigerare', 'cold room', 'camera frigorifica',
        'freezer', 'congelator', 'chiller', 'racitor', 'cooling'
    ],
    'bar': [
        'bar', 'counter', 'tejghea', 'servery', 'zona servire',
        'buffet', 'coffee station', 'cafenea'
    ],
    'dining': [
        'dining', 'restaurant', 'seating', 'locuri',
        'tables', 'mese', 'chairs', 'scaune'
    ],
    
    # PROFESSIONAL SERVICES (Category C)
    'management': [
        'management', 'conducere', 'project management',
        'site management', 'supervision', 'supraveghere'
    ],
    'design': [
        'design', 'proiectare', 'designer', 'architect',
        'arhitect', 'engineering', 'inginerie'
    ],
    'permits': [
        'permits', 'autorizatii', 'approvals', 'avize',
        'documentation', 'documentatie', 'building permit'
    ],
    'preliminaries': [
        'preliminarii', 'preliminaries', 'mobilization', 'mobilizare',
        'site setup', 'organizare santier', 'temporary works',
        'hoarding', 'imprejmuire', 'welfare', 'vestiare'
    ],
    'testing': [
        'testing', 'testare', 'commissioning', 'punere in functiune',
        'inspection', 'inspectie', 'verification', 'verificare'
    ],
    'cleaning': [
        'cleaning', 'curatenie', 'washing', 'spalare',
        'final clean', 'curatenie finala'
    ],
    'waste': [
        'waste', 'deseuri', 'disposal', 'eliminare',
        'debris removal', 'indepartare moloz', 'skip hire'
    ]
}


# ============================================================================
# CORE DATA CLASSES
# ============================================================================

@dataclass
class ProjectMetadata:
    """Project header information from offer"""
    project_name: str
    client_name: Optional[str] = None
    address: Optional[str] = None
    location: Optional[str] = None
    lead_designer: Optional[str] = None
    project_manager: Optional[str] = None
    offer_number: Optional[str] = None
    offer_date: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary, omitting None values"""
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class CostItem:
    """
    Single line item in cost breakdown
    Handles both summary (totals only) and detailed (unit prices) formats
    """
    item_number: str
    description: str
    value_eur: float
    category_id: str  # "A", "B", "C", "UNKNOWN"
    item_type: str    # Classified type (e.g., "demolition", "hvac")
    
    # Optional detailed breakdown (populated for unit_prices detail level)
    quantity: Optional[float] = None
    unit: Optional[str] = None              # Raw unit from source
    unit_normalized: Optional[str] = None   # Standardized unit
    unit_price_eur: Optional[float] = None
    
    # Optional metadata
    description_en: Optional[str] = None    # English translation if needed
    comments: Optional[str] = None
    
    def has_unit_pricing(self) -> bool:
        """Check if item has detailed unit price information"""
        return all([
            self.quantity is not None,
            self.unit is not None,
            self.unit_price_eur is not None
        ])
    
    def validate(self) -> List[str]:
        """Validate item data, return list of errors"""
        errors = []
        
        # Value validation
        if self.value_eur < 0:
            errors.append(f"Item {self.item_number}: Negative value {self.value_eur}")
        if self.value_eur > VALIDATION_RULES['max_total_eur']:
            errors.append(f"Item {self.item_number}: Value {self.value_eur} exceeds maximum")
        
        # Unit price validation (if present)
        if self.unit_price_eur is not None:
            if self.unit_price_eur < VALIDATION_RULES['min_unit_price']:
                errors.append(f"Item {self.item_number}: Unit price too low: {self.unit_price_eur}")
            if self.unit_price_eur > VALIDATION_RULES['max_unit_price']:
                errors.append(f"Item {self.item_number}: Unit price too high: {self.unit_price_eur}")
            
            # Cross-check: quantity × unit_price ≈ value
            if self.quantity and self.quantity > 0:
                calculated = self.quantity * self.unit_price_eur
                tolerance = abs(calculated - self.value_eur) / self.value_eur
                if tolerance > 0.05:  # 5% tolerance
                    errors.append(
                        f"Item {self.item_number}: Calculated value {calculated:.2f} "
                        f"differs from stated {self.value_eur:.2f}"
                    )
        
        return errors
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage"""
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class CostCategory:
    """
    Major cost category (A: Architectural, B: MEP, C: Professional Services)
    """
    category_id: str
    name: str
    items: List[CostItem]
    total_eur: float
    
    def get_item_count(self) -> int:
        """Number of items in category"""
        return len(self.items)
    
    def get_items_with_unit_pricing(self) -> List[CostItem]:
        """Get items that have detailed unit price information"""
        return [item for item in self.items if item.has_unit_pricing()]
    
    def calculate_percentage(self, grand_total: float) -> float:
        """Calculate category percentage of grand total"""
        if grand_total == 0:
            return 0.0
        return round((self.total_eur / grand_total) * 100, 2)
    
    def validate_total(self) -> Tuple[bool, float]:
        """
        Validate that sum of items matches category total
        Returns: (is_valid, difference_eur)
        """
        items_sum = sum(item.value_eur for item in self.items)
        difference = abs(items_sum - self.total_eur)
        tolerance = self.total_eur * VALIDATION_RULES['category_tolerance']
        is_valid = difference <= tolerance
        return is_valid, difference
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'category_id': self.category_id,
            'name': self.name,
            'total_eur': self.total_eur,
            'item_count': self.get_item_count(),
            'items': [item.to_dict() for item in self.items]
        }


@dataclass
class CostBreakdown:
    """Complete cost structure for an offer"""
    categories: List[CostCategory]
    grand_total_eur: float
    
    def get_category_percentages(self) -> Dict[str, float]:
        """Calculate percentage split across categories"""
        return {
            cat.category_id: cat.calculate_percentage(self.grand_total_eur)
            for cat in self.categories
        }
    
    def get_category_by_id(self, category_id: str) -> Optional[CostCategory]:
        """Get category by ID"""
        for cat in self.categories:
            if cat.category_id == category_id:
                return cat
        return None
    
    def get_total_items(self) -> int:
        """Total number of items across all categories"""
        return sum(cat.get_item_count() for cat in self.categories)
    
    def get_all_items_with_unit_pricing(self) -> List[CostItem]:
        """Get all items that have unit pricing information"""
        items = []
        for cat in self.categories:
            items.extend(cat.get_items_with_unit_pricing())
        return items
    
    def validate_total(self) -> Tuple[bool, float]:
        """
        Validate that sum of categories matches grand total
        Returns: (is_valid, difference_eur)
        """
        categories_sum = sum(cat.total_eur for cat in self.categories)
        difference = abs(categories_sum - self.grand_total_eur)
        tolerance = self.grand_total_eur * VALIDATION_RULES['category_tolerance']
        is_valid = difference <= tolerance
        return is_valid, difference
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'grand_total_eur': self.grand_total_eur,
            'categories': [cat.to_dict() for cat in self.categories],
            'category_percentages': self.get_category_percentages(),
            'total_items': self.get_total_items()
        }


@dataclass
class ParsedOffer:
    """
    Complete parsed offer - unified structure for all formats
    """
    offer_id: str
    project: ProjectMetadata
    cost_breakdown: CostBreakdown
    parsed_at: datetime
    source_filename: str
    detail_level: DetailLevel
    
    # Optional metadata
    warnings: List[str] = field(default_factory=list)
    parser_version: str = "1.0"
    
    def get_summary_stats(self) -> dict:
        """Get high-level statistics"""
        unit_price_items = self.cost_breakdown.get_all_items_with_unit_pricing()
        
        return {
            'offer_id': self.offer_id,
            'project_name': self.project.project_name,
            'total_eur': self.cost_breakdown.grand_total_eur,
            'detail_level': self.detail_level.value,
            'category_count': len(self.cost_breakdown.categories),
            'total_items': self.cost_breakdown.get_total_items(),
            'items_with_unit_pricing': len(unit_price_items),
            'parsed_at': self.parsed_at.isoformat(),
            'has_warnings': len(self.warnings) > 0
        }
    
    def to_firestore_summary(self) -> dict:
        """
        Lightweight version for Firestore (~2-5 KB)
        Only essential metadata + category summary
        """
        return {
            'offer_id': self.offer_id,
            'project_name': self.project.project_name,
            'client_name': self.project.client_name,
            'total_eur': self.cost_breakdown.grand_total_eur,
            'ingestion_date': self.parsed_at.isoformat(),
            'detail_level': self.detail_level.value,
            'category_summary': {
                cat.category_id: {
                    'name': cat.name,
                    'total': cat.total_eur,
                    'percent': cat.calculate_percentage(self.cost_breakdown.grand_total_eur),
                    'item_count': cat.get_item_count()
                }
                for cat in self.cost_breakdown.categories
            },
            'source_filename': self.source_filename,
            'has_unit_pricing': self.detail_level == DetailLevel.UNIT_PRICES,
            'warnings': self.warnings,
            'parser_version': self.parser_version
        }
    
    def to_gcs_json(self) -> dict:
        """
        Complete version for GCS storage (any size)
        Includes all items and detailed breakdowns
        """
        return {
            'offer_id': self.offer_id,
            'project': self.project.to_dict(),
            'cost_breakdown': self.cost_breakdown.to_dict(),
            'parsed_at': self.parsed_at.isoformat(),
            'source_filename': self.source_filename,
            'detail_level': self.detail_level.value,
            'warnings': self.warnings,
            'parser_version': self.parser_version,
            'summary_stats': self.get_summary_stats()
        }
    
    def validate(self) -> List[str]:
        """
        Comprehensive validation
        Returns list of all errors found
        """
        errors = []
        
        # Basic validation
        if not self.project.project_name:
            errors.append("Missing project name")
        
        if self.cost_breakdown.grand_total_eur < VALIDATION_RULES['min_total_eur']:
            errors.append(f"Total {self.cost_breakdown.grand_total_eur} below minimum")
        
        if self.cost_breakdown.grand_total_eur > VALIDATION_RULES['max_total_eur']:
            errors.append(f"Total {self.cost_breakdown.grand_total_eur} exceeds maximum")
        
        # Category validation
        if len(self.cost_breakdown.categories) == 0:
            errors.append("No categories found")
        
        # Validate category totals match grand total
        is_valid, diff = self.cost_breakdown.validate_total()
        if not is_valid:
            errors.append(f"Category sum differs from grand total by €{diff:.2f}")
        
        # Validate each category
        for cat in self.cost_breakdown.categories:
            is_valid, diff = cat.validate_total()
            if not is_valid:
                errors.append(
                    f"Category {cat.category_id}: Item sum differs from category total by €{diff:.2f}"
                )
            
            # Validate each item
            for item in cat.items:
                errors.extend(item.validate())
        
        return errors


@dataclass
class IngestionResult:
    """Result of ingesting an offer file"""
    success: bool
    offer_id: str
    message: str
    
    # Statistics
    categories_found: int = 0
    items_extracted: int = 0
    items_with_unit_pricing: int = 0
    total_eur: float = 0.0
    detail_level: Optional[str] = None
    
    # File paths
    gcs_original_path: Optional[str] = None
    gcs_parsed_path: Optional[str] = None
    firestore_doc_id: Optional[str] = None
    
    # Issues
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API response"""
        return {
            'success': self.success,
            'offer_id': self.offer_id,
            'message': self.message,
            'statistics': {
                'categories_found': self.categories_found,
                'items_extracted': self.items_extracted,
                'items_with_unit_pricing': self.items_with_unit_pricing,
                'total_eur': self.total_eur,
                'detail_level': self.detail_level
            },
            'storage': {
                'gcs_original_path': self.gcs_original_path,
                'gcs_parsed_path': self.gcs_parsed_path,
                'firestore_doc_id': self.firestore_doc_id
            },
            'issues': {
                'errors': self.errors,
                'warnings': self.warnings
            }
        }


# ============================================================================
# PATTERN LEARNING DATA CLASSES
# ============================================================================

@dataclass
class CategoryPattern:
    """Learned pattern for category distribution"""
    project_type: str
    architectural_percent: float
    mep_percent: float
    professional_percent: float = 0.0
    sample_size: int = 0
    confidence: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            'project_type': self.project_type,
            'architectural_percent': self.architectural_percent,
            'mep_percent': self.mep_percent,
            'professional_percent': self.professional_percent,
            'sample_size': self.sample_size,
            'confidence': self.confidence,
            'last_updated': self.last_updated.isoformat()
        }


@dataclass
class ItemFrequency:
    """How often an item type appears across offers"""
    item_type: str
    frequency: float  # 0.0 to 1.0
    appears_in_offer_ids: List[str]
    avg_value_eur: float
    value_range: Tuple[float, float]  # (min, max)
    
    # Unit pricing statistics (if available)
    avg_unit_price_eur: Optional[float] = None
    unit_price_range: Optional[Tuple[float, float]] = None
    common_unit: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            'item_type': self.item_type,
            'frequency': self.frequency,
            'sample_size': len(self.appears_in_offer_ids),
            'avg_value_eur': self.avg_value_eur,
            'value_range': {'min': self.value_range[0], 'max': self.value_range[1]},
            'unit_pricing': {
                'avg_unit_price_eur': self.avg_unit_price_eur,
                'unit_price_range': {
                    'min': self.unit_price_range[0] if self.unit_price_range else None,
                    'max': self.unit_price_range[1] if self.unit_price_range else None
                } if self.unit_price_range else None,
                'common_unit': self.common_unit
            } if self.avg_unit_price_eur else None
        }


@dataclass
class LearningStats:
    """Overall learning system statistics"""
    total_offers_ingested: int
    offers_with_unit_pricing: int
    last_ingestion: datetime
    last_learning_run: Optional[datetime] = None
    category_patterns_count: int = 0
    item_types_tracked: int = 0
    confidence_score: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            'total_offers_ingested': self.total_offers_ingested,
            'offers_with_unit_pricing': self.offers_with_unit_pricing,
            'last_ingestion': self.last_ingestion.isoformat(),
            'last_learning_run': self.last_learning_run.isoformat() if self.last_learning_run else None,
            'category_patterns_count': self.category_patterns_count,
            'item_types_tracked': self.item_types_tracked,
            'confidence_score': self.confidence_score
        }