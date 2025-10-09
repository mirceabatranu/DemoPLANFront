"""
Enhanced Floorplan Interpreter Agent V2 - Complete Integration
Combines Romanian chat interface with full DXF processing capabilities
Merges all three versions into a single powerful agent
"""

import asyncio
import logging
import os
import re
import math
import tempfile
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from datetime import datetime, timezone
from pathlib import Path

# Chat integration imports
from src.agents.base_chat_agent import (
    ChatIntegratedAgent, 
    TechnicalField, 
    AgentDataRequirements,
    AgentResponse,
    AgentContext,
    dataclass,
    Enum
)
import time
import json
import gzip
import base64

# DXF processing imports (add these to your requirements.txt)
try:
    import ezdxf
    from ezdxf import readfile
    DXF_AVAILABLE = True
except ImportError:
    DXF_AVAILABLE = False
    logging.warning("ezdxf not available - DXF processing will be limited")

# Google Cloud imports (if available)
try:
    from google.cloud import storage
    from google.cloud import vision
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False

logger = logging.getLogger("demoplan.floorplan_agent")

# Phase 1: Intelligent Text Classification Functions

@dataclass
class TextClassification:
    """Classification result for text entity"""
    is_room_label: bool
    is_construction_detail: bool
    is_dimension: bool
    relevance_score: float
    room_type: Optional[str] = None
    language: Optional[str] = None

class IntelligentTextClassifier:
    """Smart text classification to separate room labels from construction specifications"""
    
    def __init__(self):
        # Room-specific keywords in multiple languages
        self.room_keywords = {
            'romanian': {
                'bedroom': ['dormitor', 'camera', 'cam'],
                'kitchen': ['bucatarie', 'bucătărie', 'buc'],
                'bathroom': ['baie', 'toaleta', 'wc'],
                'living_room': ['living', 'salon', 'sufragerie'],
                'hallway': ['hol', 'hall', 'coridor'],
                'storage': ['debara', 'camara', 'cămară'],
                'balcony': ['balcon', 'terasa', 'terasă'],
                'entrance': ['intrare', 'vestibul', 'foyer']
            },
            'english': {
                'bedroom': ['bedroom', 'bed room', 'master bed'],
                'kitchen': ['kitchen', 'kitchenette'],
                'bathroom': ['bathroom', 'toilet', 'restroom', 'lavatory'],
                'living_room': ['living room', 'living', 'lounge', 'sitting'],
                'hallway': ['hallway', 'corridor', 'passage'],
                'storage': ['storage', 'closet', 'pantry'],
                'balcony': ['balcony', 'terrace', 'patio'],
                'entrance': ['entrance', 'foyer', 'entry']
            }
        }
        
        # Construction specification patterns to filter out
        self.construction_noise_patterns = [
            r'płyta\s+GK',  # Polish plasterboard
            r'ceramic\s+tiles?\s+\d+x\d+',
            r'gres\s+\d+x\d+',
            r'tapet\s+vinyl',
            r'vinyl\s+wallpaper',
            r'floor\s+tiles?',
            r'wall\s+tiles?',
            r'paint\s+\w+',
            r'farba\s+\w+',
            r'zuziłka\s+\w+',
            r'isolation\s+\w+',
            r'insulation\s+\w+',
            r'membrane\s+\w+',
            r'foil\s+PE',
            r'folie\s+PE',
            r'concrete\s+\w+',
            r'beton\s+\w+',
            r'mortar\s+\w+',
            r'tencuiala\s+\w+',
            r'placaj\s+\w+',
            r'coating\s+\w+'
        ]
        
        # Dimension patterns to identify measurements
        self.dimension_patterns = [
            r'\d+[.,]\d+\s*[xX×]\s*\d+[.,]\d+',
            r'\d+[.,]\d+\s*m',
            r'\d+\s*[xX×]\s*\d+',
            r'\d+[.,]\d+\s*cm',
            r'\d+[.,]\d+\s*mm'
        ]
        
        # Multilingual stop words for construction
        self.construction_stopwords = {
            'polish': ['płyta', 'warstw', 'izolacja', 'membrana', 'folie'],
            'english': ['layer', 'coating', 'finish', 'material', 'specification'],
            'romanian': ['strat', 'acoperire', 'finisaj', 'material', 'specificatie']
        }

    def classify_text_entity(self, text: str, location: List[float], 
                           height: float = 0, layer: str = "") -> TextClassification:
        """Classify a single text entity with comprehensive analysis"""
        
        text_lower = text.lower().strip()
        
        # Skip very short or empty text
        if len(text_lower) < 2:
            return TextClassification(
                is_room_label=False,
                is_construction_detail=False,
                is_dimension=False,
                relevance_score=0.0
            )
        
        # Check if it's a dimension
        is_dimension = self._is_dimension_text(text_lower)
        if is_dimension:
            return TextClassification(
                is_room_label=False,
                is_construction_detail=False,
                is_dimension=True,
                relevance_score=0.2  # Low relevance for room analysis
            )
        
        # Check if it's construction specification noise
        is_construction_detail = self._is_construction_specification(text_lower)
        if is_construction_detail:
            return TextClassification(
                is_room_label=False,
                is_construction_detail=True,
                is_dimension=False,
                relevance_score=0.1  # Very low relevance for room analysis
            )
        
        # Check if it's a room label
        room_analysis = self._analyze_room_label(text_lower)
        if room_analysis['is_room']:
            return TextClassification(
                is_room_label=True,
                is_construction_detail=False,
                is_dimension=False,
                relevance_score=room_analysis['confidence'],
                room_type=room_analysis['type'],
                language=room_analysis['language']
            )
        
        # Check spatial context for relevance
        spatial_relevance = self._calculate_spatial_relevance(location, height, layer)
        
        return TextClassification(
            is_room_label=False,
            is_construction_detail=False,
            is_dimension=False,
            relevance_score=spatial_relevance
        )

    def _is_dimension_text(self, text: str) -> bool:
        """Check if text represents a dimension or measurement"""
        for pattern in self.dimension_patterns:
            if re.search(pattern, text):
                return True
        
        # Check for dimension-like number patterns
        if re.match(r'^\d+([.,]\d+)?$', text):
            return True
            
        return False

    def _is_construction_specification(self, text: str) -> bool:
        """Identify construction specifications and technical details"""
        
        # Check construction noise patterns
        for pattern in self.construction_noise_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        # Check for multilingual construction terms
        for lang, stopwords in self.construction_stopwords.items():
            for stopword in stopwords:
                if stopword in text:
                    return True
        
        # Check for technical specification formats
        technical_patterns = [
            r'\w+\s+\d+x\d+',  # material 120x30
            r'thickness?\s*[=:]\s*\d+',
            r'grubość\s*\d+',
            r'warstwa\s+\d+',
            r'layer\s+\d+',
            r'strat\s+\d+',
            r'spec\.',
            r'specif\.',
            r'conform\s+\w+'
        ]
        
        for pattern in technical_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False

    def _analyze_room_label(self, text: str) -> Dict:
        """Analyze if text represents a room label"""
        
        result = {
            'is_room': False,
            'type': None,
            'confidence': 0.0,
            'language': None
        }
        
        # Check against room keywords
        for lang, room_types in self.room_keywords.items():
            for room_type, keywords in room_types.items():
                for keyword in keywords:
                    if keyword in text:
                        # Calculate confidence based on text match quality
                        confidence = self._calculate_room_confidence(text, keyword, room_type)
                        if confidence > result['confidence']:
                            result = {
                                'is_room': True,
                                'type': room_type,
                                'confidence': confidence,
                                'language': lang
                            }
        
        return result

    def _calculate_room_confidence(self, text: str, keyword: str, room_type: str) -> float:
        """Calculate confidence score for room label detection"""
        
        confidence = 0.3  # Base confidence
        
        # Exact match bonus
        if text == keyword:
            confidence += 0.4
        
        # Single word bonus (likely a clean room label)
        if len(text.split()) == 1:
            confidence += 0.2
        
        # Room type specific bonuses
        if room_type in ['bedroom', 'kitchen', 'bathroom']:
            confidence += 0.1  # Important rooms get priority
        
        # Length penalty for very long text (likely not just a room label)
        if len(text) > 20:
            confidence -= 0.2
        
        return min(confidence, 1.0)

    def _calculate_spatial_relevance(self, location: List[float], 
                                   height: float, layer: str) -> float:
        """Calculate spatial relevance for unclassified text"""
        
        relevance = 0.1  # Base relevance
        
        # Text size relevance (larger text likely more important)
        if height > 2.0:
            relevance += 0.3
        elif height > 1.0:
            relevance += 0.1
        
        # Layer relevance
        important_layers = ['text', 'labels', 'rooms', 'annotation']
        if any(layer_name in layer.lower() for layer_name in important_layers):
            relevance += 0.2
        
        return min(relevance, 0.5)  # Cap at 0.5 for unclassified text

    def filter_relevant_text_entities(self, text_entities: List[Dict]) -> Dict[str, List[Dict]]:
        """Filter and categorize text entities by relevance"""
        
        categorized = {
            'room_labels': [],
            'dimensions': [],
            'construction_details': [],
            'other_relevant': [],
            'filtered_out': []
        }
        
        for entity in text_entities:
            text = entity.get('text', '')
            location = entity.get('location', [0, 0])
            height = entity.get('height', 0)
            layer = entity.get('layer', '')
            
            classification = self.classify_text_entity(text, location, height, layer)
            
            # Add classification info to entity
            entity['classification'] = classification
            
            # Categorize based on classification
            if classification.is_room_label:
                categorized['room_labels'].append(entity)
            elif classification.is_dimension:
                categorized['dimensions'].append(entity)
            elif classification.is_construction_detail:
                categorized['construction_details'].append(entity)
            elif classification.relevance_score > 0.3:
                categorized['other_relevant'].append(entity)
            else:
                categorized['filtered_out'].append(entity)
        
        return categorized

# Phase 2: Smart Data Extraction Functions

@dataclass
class RoomSpatialData:
    """Optimized room data structure"""
    room_id: str
    room_type: str
    area: float
    location: List[float]
    confidence: float
    is_validated: bool = False

@dataclass 
class OptimizedFloorplanData:
    """Compressed floorplan analysis results"""
    # Essential spatial information
    total_rooms: int
    total_area: float
    room_breakdown: List[RoomSpatialData]
    
    # Technical summary
    has_dimensions: bool
    has_electrical: bool
    scale_factor: float
    confidence_score: float
    
    # Optional detailed data (loaded on demand)
    _detailed_analysis: Optional[Dict] = None

class SmartDataExtractor:
    """Enhanced extraction focusing on essential spatial information"""
    
    def __init__(self, text_classifier):
        self.text_classifier = text_classifier
        self.romanian_room_types = {
            'bucatarie': 'kitchen',
            'baie': 'bathroom', 
            'dormitor': 'bedroom',
            'living': 'living_room',
            'hol': 'hallway',
            'debara': 'storage',
            'sufragerie': 'living_room',
            'vestibul': 'entrance',
            'balcon': 'balcony',
            'terasa': 'terrace'
        }

    def extract_optimized_room_data(self, doc) -> OptimizedFloorplanData:
        """Extract only essential room information with high relevance"""
        
        # Get filtered text entities
        all_text_entities = self._extract_all_text_entities(doc)
        categorized_text = self.text_classifier.filter_relevant_text_entities(all_text_entities)
        
        # Focus on room labels only
        room_labels = categorized_text['room_labels']
        dimensions = categorized_text['dimensions']
        
        # Extract room boundaries (spatial data)
        room_boundaries = self._extract_room_boundaries(doc)
        
        # Combine text labels with spatial boundaries
        validated_rooms = self._match_labels_to_spaces(room_labels, room_boundaries)
        
        # Calculate essential metrics
        total_rooms = len(validated_rooms)
        total_area = sum(room.area for room in validated_rooms if room.area > 0)
        
        # Technical indicators
        has_dimensions = len(dimensions) > 0
        has_electrical = self._detect_electrical_elements(doc)
        scale_factor = self._calculate_scale_factor(doc)
        
        # Confidence calculation
        confidence_score = self._calculate_extraction_confidence(
            validated_rooms, has_dimensions, has_electrical, scale_factor
        )
        
        return OptimizedFloorplanData(
            total_rooms=total_rooms,
            total_area=total_area,
            room_breakdown=validated_rooms,
            has_dimensions=has_dimensions,
            has_electrical=has_electrical,
            scale_factor=scale_factor,
            confidence_score=confidence_score
        )

    def _extract_all_text_entities(self, doc) -> List[Dict]:
        """Extract text entities with minimal processing"""
        text_entities = []
        
        for entity_type in ['MTEXT', 'TEXT']:
            try:
                for entity in doc.modelspace().query(entity_type):
                    text_content = entity.dxf.text.strip()
                    if len(text_content) > 1:  # Minimum length filter
                        text_entity = {
                            "text": text_content,
                            "location": list(entity.dxf.insert)[:2] if hasattr(entity.dxf, 'insert') else [0, 0],
                            "height": getattr(entity.dxf, 'height', 0),
                            "layer": getattr(entity.dxf, 'layer', 'Unknown')
                        }
                        text_entities.append(text_entity)
            except Exception:
                continue  # Skip if entity type not found
                
        return text_entities

    def _extract_room_boundaries(self, doc) -> List[Dict]:
        """Extract only significant room boundaries"""
        boundaries = []
        min_room_area = 2.0  # Minimum 2 sqm for a room
        
        try:
            for entity in doc.modelspace().query('LWPOLYLINE'):
                if entity.is_closed:
                    vertices = [list(p)[:2] for p in entity.get_points()]
                    area = self._calculate_polygon_area(vertices)
                    
                    # Only keep boundaries that could represent actual rooms
                    if area >= min_room_area:
                        boundaries.append({
                            "vertices": vertices,
                            "area": area,
                            "centroid": self._calculate_centroid(vertices),
                            "layer": getattr(entity.dxf, 'layer', 'Unknown')
                        })
        except Exception:
            pass
            
        # Sort by area (largest rooms first)
        return sorted(boundaries, key=lambda x: x['area'], reverse=True)

    def _match_labels_to_spaces(self, room_labels: List[Dict], 
                               boundaries: List[Dict]) -> List[RoomSpatialData]:
        """Match room labels with spatial boundaries"""
        validated_rooms = []
        used_boundaries = set()
        
        for i, label in enumerate(room_labels):
            label_location = label['location']
            classification = label['classification']
            
            # Find closest boundary that contains this label
            best_match = None
            best_distance = float('inf')
            
            for j, boundary in enumerate(boundaries):
                if j in used_boundaries:
                    continue
                    
                # Check if label is inside boundary
                if self._point_in_polygon(label_location, boundary['vertices']):
                    distance = self._calculate_distance(label_location, boundary['centroid'])
                    if distance < best_distance:
                        best_distance = distance
                        best_match = j
            
            # Create room data
            if best_match is not None:
                boundary = boundaries[best_match]
                used_boundaries.add(best_match)
                
                room_data = RoomSpatialData(
                    room_id=f"room_{i}",
                    room_type=classification.room_type or "unknown",
                    area=boundary['area'],
                    location=boundary['centroid'],
                    confidence=classification.relevance_score,
                    is_validated=True
                )
            else:
                # Label without clear boundary
                room_data = RoomSpatialData(
                    room_id=f"room_{i}",
                    room_type=classification.room_type or "unknown", 
                    area=0.0,  # Unknown area
                    location=label_location,
                    confidence=classification.relevance_score * 0.7,  # Penalty for no boundary
                    is_validated=False
                )
            
            validated_rooms.append(room_data)
        
        # Add unmatched boundaries as unknown rooms
        for j, boundary in enumerate(boundaries):
            if j not in used_boundaries and boundary['area'] > 5.0:  # Only significant spaces
                room_data = RoomSpatialData(
                    room_id=f"space_{j}",
                    room_type="unknown",
                    area=boundary['area'],
                    location=boundary['centroid'],
                    confidence=0.3,  # Low confidence for unlabeled spaces
                    is_validated=False
                )
                validated_rooms.append(room_data)
        
        return validated_rooms

    def _detect_electrical_elements(self, doc) -> bool:
        """Quick check for electrical installations"""
        electrical_keywords = ['prize', 'switch', 'outlet', 'intrerupator', 'electric']
        
        # Check block references (common for electrical symbols)
        try:
            for entity in doc.modelspace().query('INSERT'):
                block_name = entity.dxf.name.lower()
                if any(keyword in block_name for keyword in electrical_keywords):
                    return True
        except Exception:
            pass
            
        # Check circles (often represent outlets/fixtures)
        try:
            circle_count = len(list(doc.modelspace().query('CIRCLE')))
            return circle_count > 3  # Multiple circles suggest electrical plan
        except Exception:
            pass
            
        return False

    def _calculate_scale_factor(self, doc) -> float:
        """Determine drawing scale factor"""
        try:
            header = doc.header
            
            # Check drawing units
            insunits_code = header.get('$INSUNITS', 0)
            unit_scales = {
                4: 1000.0,  # Millimeters to meters
                5: 100.0,   # Centimeters to meters
                6: 1.0,     # Meters
                1: 39.37,   # Inches to meters (approximate)
                2: 3.28     # Feet to meters (approximate)
            }
            
            return unit_scales.get(insunits_code, 1.0)
            
        except Exception:
            return 1.0

    def _calculate_extraction_confidence(self, rooms: List[RoomSpatialData],
                                       has_dimensions: bool, has_electrical: bool,
                                       scale_factor: float) -> float:
        """Calculate confidence score for data extraction"""
        
        confidence = 0.0
        
        # Room detection confidence
        validated_rooms = sum(1 for room in rooms if room.is_validated)
        confidence += min(validated_rooms * 15, 60)  # Max 60 points
        
        # Room type identification
        typed_rooms = sum(1 for room in rooms if room.room_type != "unknown")
        if len(rooms) > 0:
            confidence += (typed_rooms / len(rooms)) * 20  # Max 20 points
        
        # Technical completeness
        if has_dimensions:
            confidence += 10
        if has_electrical:
            confidence += 5
        if scale_factor != 1.0:
            confidence += 5  # Scale detection
        
        return min(confidence, 100.0)

    # Utility geometry functions
    def _calculate_polygon_area(self, vertices: List[List[float]]) -> float:
        """Calculate polygon area using shoelace formula"""
        if len(vertices) < 3:
            return 0
        area = 0
        for i in range(len(vertices)):
            j = (i + 1) % len(vertices)
            area += vertices[i][0] * vertices[j][1]
            area -= vertices[j][0] * vertices[i][1]
        return abs(area) / 2

    def _calculate_centroid(self, vertices: List[List[float]]) -> List[float]:
        """Calculate centroid of polygon"""
        if not vertices:
            return [0, 0]
        
        cx = sum(v[0] for v in vertices) / len(vertices)
        cy = sum(v[1] for v in vertices) / len(vertices)
        return [cx, cy]

    def _point_in_polygon(self, point: List[float], polygon: List[List[float]]) -> bool:
        """Check if point is inside polygon using ray casting"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside

    def _calculate_distance(self, point1: List[float], point2: List[float]) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def generate_concise_summary(self, optimized_data: OptimizedFloorplanData) -> str:
        """Generate concise Romanian summary for API responses"""
        
        parts = []
        
        if optimized_data.total_rooms > 0:
            parts.append(f"{optimized_data.total_rooms} spatii")
        
        if optimized_data.total_area > 0:
            parts.append(f"{optimized_data.total_area:.1f} mp")
        
        # Room types summary
        room_types = {}
        for room in optimized_data.room_breakdown:
            if room.room_type != "unknown":
                romanian_name = self._get_romanian_room_name(room.room_type)
                room_types[romanian_name] = room_types.get(romanian_name, 0) + 1
        
        if room_types:
            type_list = [f"{count}x{name}" if count > 1 else name 
                        for name, count in room_types.items()]
            parts.append(f"tipuri: {', '.join(type_list)}")
        
        # Technical indicators
        tech_parts = []
        if optimized_data.has_dimensions:
            tech_parts.append("cotat")
        if optimized_data.has_electrical:
            tech_parts.append("instalatii")
        
        if tech_parts:
            parts.append(f"({', '.join(tech_parts)})")
        
        summary = f"Plan: {', '.join(parts)}" if parts else "Plan analizat"
        
        return summary

    def _get_romanian_room_name(self, room_type: str) -> str:
        """Convert English room type to Romanian"""
        romanian_names = {
            'bedroom': 'dormitor',
            'kitchen': 'bucatarie', 
            'bathroom': 'baie',
            'living_room': 'living',
            'hallway': 'hol',
            'storage': 'debara',
            'balcony': 'balcon',
            'entrance': 'intrare'
        }
        return romanian_names.get(room_type, room_type)

    def export_for_session(self, optimized_data: OptimizedFloorplanData) -> Dict[str, Any]:
        """Export minimal data for session storage"""
        return {
            "room_count": optimized_data.total_rooms,
            "total_area": optimized_data.total_area,
            "room_types": {room.room_type: 1 for room in optimized_data.room_breakdown 
                          if room.room_type != "unknown"},
            "has_technical_data": optimized_data.has_dimensions or optimized_data.has_electrical,
            "confidence": optimized_data.confidence_score,
            "summary": self.generate_concise_summary(optimized_data)
        }

# Phase 3: Optimized Storage Structure & Performance Functions

class DetailLevel(Enum):
    """Detail levels for different use cases"""
    ESSENTIAL = "essential"      # Room count, area, types - for API responses
    SUMMARY = "summary"          # + technical indicators - for session storage
    DETAILED = "detailed"        # + construction specs - for technical analysis
    COMPLETE = "complete"        # Everything - for debugging/export

@dataclass
class TieredFloorplanData:
    """Optimized tiered storage structure"""
    
    # Essential data (always loaded) - ~200 bytes
    essential: Dict[str, Any]
    
    # Summary data (loaded on demand) - ~500 bytes
    summary: Optional[Dict[str, Any]] = None
    
    # Detailed data (lazy loaded) - ~2KB compressed
    detailed_compressed: Optional[str] = None
    
    # Complete data (only for export/debug)
    complete_compressed: Optional[str] = None
    
    # Metadata
    data_version: str = "optimized_v3"
    created_at: str = ""
    processing_time_ms: float = 0.0

class OptimizedFloorplanStorage:
    """High-performance storage manager with tiered data access"""
    
    def __init__(self):
        self.compression_threshold = 1000  # Compress data > 1KB
        self.essential_fields = [
            "total_rooms", "total_area", "room_types", 
            "confidence_score", "has_technical_data"
        ]
    
    def create_tiered_storage(self, full_analysis: Dict[str, Any], 
                            processing_time: float = 0.0) -> TieredFloorplanData:
        """Create optimized tiered storage from full DXF analysis"""
        
        # Extract essential data only
        essential_data = self._extract_essential_data(full_analysis)
        
        # Create summary data
        summary_data = self._extract_summary_data(full_analysis)
        
        # Compress detailed technical data
        detailed_data = self._extract_detailed_data(full_analysis)
        detailed_compressed = self._compress_data(detailed_data) if detailed_data else None
        
        # Store complete data (compressed)
        complete_compressed = self._compress_data(full_analysis)
        
        return TieredFloorplanData(
            essential=essential_data,
            summary=summary_data,
            detailed_compressed=detailed_compressed,
            complete_compressed=complete_compressed,
            processing_time_ms=processing_time
        )
    
    def _extract_essential_data(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract only essential spatial information (~200 bytes)"""
        
        spatial = analysis.get("spatial_analysis", {})
        
        essential = {
            "total_rooms": spatial.get("total_rooms", 0),
            "total_area": round(spatial.get("estimated_total_area", 0.0), 1),
            "confidence_score": round(analysis.get("confidence_factors", {}).get("overall", 0.0), 1),
            "has_technical_data": len(analysis.get("dimensions", [])) > 0 or analysis.get("electrical_elements", {}).get("has_electrical_plan", False)
        }
        
        # Add room types summary (compressed)
        room_types = spatial.get("room_types", {})
        if room_types:
            # Only store non-unknown room types
            filtered_types = {k: v for k, v in room_types.items() if k != "unknown" and v > 0}
            if filtered_types:
                essential["room_types"] = filtered_types
        
        return essential
    
    def _extract_summary_data(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract summary technical data (~500 bytes)"""
        
        spatial = analysis.get("spatial_analysis", {})
        electrical = analysis.get("electrical_elements", {})
        openings = analysis.get("doors_windows", {})
        scale_info = analysis.get("scale_info", {})
        
        summary = {
            "wall_length": round(spatial.get("wall_total_length", 0.0), 1),
            "has_dimensions": len(analysis.get("dimensions", [])) > 0,
            "has_electrical": electrical.get("has_electrical_plan", False),
            "scale_factor": scale_info.get("detected_scale", 1.0),
            "total_openings": openings.get("total_openings", 0)
        }
        
        # Add electrical summary if present
        if electrical.get("has_electrical_plan"):
            summary["electrical_summary"] = {
                "outlets": len(electrical.get("outlets", [])),
                "switches": len(electrical.get("switches", [])),
                "lighting": len(electrical.get("lighting", []))
            }
        
        # Add room boundary data
        rooms = analysis.get("rooms", {})
        boundaries = rooms.get("boundaries", [])
        if boundaries:
            summary["room_boundaries_count"] = len(boundaries)
            summary["largest_room_area"] = max(b.get("area", 0) for b in boundaries) if boundaries else 0
        
        return summary
    
    def _extract_detailed_data(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract detailed technical data for specialist analysis (~2KB)"""
        
        detailed = {}
        
        # Construction specifications (filtered)
        construction_specs = self._extract_construction_specifications(analysis)
        if construction_specs:
            detailed["construction_specs"] = construction_specs
        
        # Layer analysis
        layers = analysis.get("layers", [])
        if layers:
            detailed["layer_analysis"] = {
                "total_layers": len(layers),
                "important_layers": [l["name"] for l in layers if not l.get("frozen", False)][:5]
            }
        
        # Dimension details
        dimensions = analysis.get("dimensions", [])
        if dimensions:
            detailed["dimension_analysis"] = {
                "total_dimensions": len(dimensions),
                "measurement_samples": [d.get("measurement") for d in dimensions if d.get("measurement")][:5]
            }
        
        # Drawing technical info
        drawing_info = analysis.get("drawing_info", {})
        if drawing_info:
            detailed["drawing_info"] = {
                "dxf_version": drawing_info.get("dxf_version"),
                "units": drawing_info.get("units"),
                "scale": drawing_info.get("scale", 1.0)
            }
        
        return detailed
    
    def _extract_construction_specifications(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant construction specifications only"""
        
        specs = {}
        
        # Get all text entities and filter for construction specs only
        rooms = analysis.get("rooms", {})
        all_text = rooms.get("all_text", [])
        
        # Count construction-related specifications
        material_specs = []
        technical_notes = []
        
        for text_entity in all_text:
            text = text_entity.get("text", "").lower()
            
            # Material specifications
            if any(keyword in text for keyword in ["ceramic", "tiles", "gres", "paint", "tencuiala"]):
                if len(material_specs) < 10:  # Limit to prevent bloat
                    material_specs.append(text_entity.get("text"))
            
            # Technical notes
            elif any(keyword in text for keyword in ["thickness", "layer", "grubość", "strat"]):
                if len(technical_notes) < 5:
                    technical_notes.append(text_entity.get("text"))
        
        if material_specs:
            specs["materials"] = material_specs
        if technical_notes:
            specs["technical_notes"] = technical_notes
        
        return specs
    
    def _compress_data(self, data: Dict[str, Any]) -> str:
        """Compress data using gzip and base64 encoding"""
        try:
            json_str = json.dumps(data, separators=(',', ':'))  # Minimal JSON
            if len(json_str) < self.compression_threshold:
                return json_str  # Don't compress small data
            
            compressed = gzip.compress(json_str.encode('utf-8'))
            return base64.b64encode(compressed).decode('ascii')
        except Exception:
            return json.dumps(data, separators=(',', ':'))  # Fallback
    
    def _decompress_data(self, compressed_data: str) -> Dict[str, Any]:
        """Decompress data from base64 gzip"""
        try:
            # Try to parse as JSON first (uncompressed)
            return json.loads(compressed_data)
        except json.JSONDecodeError:
            try:
                # Try decompression
                compressed_bytes = base64.b64decode(compressed_data.encode('ascii'))
                decompressed = gzip.decompress(compressed_bytes)
                return json.loads(decompressed.decode('utf-8'))
            except Exception:
                return {}  # Return empty dict on error
    
    def get_data_by_level(self, tiered_data: TieredFloorplanData, 
                         level: DetailLevel) -> Dict[str, Any]:
        """Get data at specified detail level"""
        
        if level == DetailLevel.ESSENTIAL:
            return tiered_data.essential
        
        elif level == DetailLevel.SUMMARY:
            result = tiered_data.essential.copy()
            if tiered_data.summary:
                result.update(tiered_data.summary)
            return result
        
        elif level == DetailLevel.DETAILED:
            result = self.get_data_by_level(tiered_data, DetailLevel.SUMMARY)
            if tiered_data.detailed_compressed:
                detailed_data = self._decompress_data(tiered_data.detailed_compressed)
                result.update(detailed_data)
            return result
        
        elif level == DetailLevel.COMPLETE:
            if tiered_data.complete_compressed:
                return self._decompress_data(tiered_data.complete_compressed)
            return self.get_data_by_level(tiered_data, DetailLevel.DETAILED)
        
        return tiered_data.essential  # Fallback
    
    def generate_api_response(self, tiered_data: TieredFloorplanData, 
                            include_summary: bool = False) -> Dict[str, Any]:
        """Generate optimized API response"""
        
        if include_summary:
            data = self.get_data_by_level(tiered_data, DetailLevel.SUMMARY)
        else:
            data = tiered_data.essential
        
        # Generate Romanian description
        description = self._generate_romanian_description(data)
        
        response = {
            "floorplan_analysis": {
                "summary": description,
                "data": data,
                "confidence": data.get("confidence_score", 0.0),
                "processing_time_ms": tiered_data.processing_time_ms
            }
        }
        
        return response
    
    def _generate_romanian_description(self, data: Dict[str, Any]) -> str:
        """Generate concise Romanian description from data"""
        
        parts = []
        
        total_rooms = data.get("total_rooms", 0)
        total_area = data.get("total_area", 0)
        
        if total_rooms > 0:
            parts.append(f"{total_rooms} spatii")
        
        if total_area > 0:
            parts.append(f"{total_area} mp")
        
        # Room types
        room_types = data.get("room_types", {})
        if room_types:
            romanian_names = {
                'bedroom': 'dormitor', 'kitchen': 'bucatarie', 'bathroom': 'baie',
                'living_room': 'living', 'hallway': 'hol', 'storage': 'debara',
                'balcony': 'balcon', 'entrance': 'intrare'
            }
            
            type_descriptions = []
            for room_type, count in room_types.items():
                romanian_name = romanian_names.get(room_type, room_type)
                if count > 1:
                    type_descriptions.append(f"{count}x{romanian_name}")
                else:
                    type_descriptions.append(romanian_name)
            
            if type_descriptions:
                parts.append(f"tipuri: {', '.join(type_descriptions)}")
        
        # Technical indicators
        tech_indicators = []
        if data.get("has_dimensions"):
            tech_indicators.append("cotat")
        if data.get("has_electrical"):
            tech_indicators.append("instalatii")
        if data.get("has_technical_data"):
            tech_indicators.append("detalii tehnice")
        
        if tech_indicators:
            parts.append(f"({', '.join(tech_indicators)})")
        
        return f"Plan: {', '.join(parts)}" if parts else "Plan analizat"
    
    def calculate_storage_efficiency(self, original_analysis: Dict[str, Any], 
                                   tiered_data: TieredFloorplanData) -> Dict[str, Any]:
        """Calculate storage efficiency metrics"""
        
        original_size = len(json.dumps(original_analysis))
        essential_size = len(json.dumps(tiered_data.essential))
        
        compressed_detailed = len(tiered_data.detailed_compressed) if tiered_data.detailed_compressed else 0
        compressed_complete = len(tiered_data.complete_compressed) if tiered_data.complete_compressed else 0
        
        total_optimized_size = essential_size + compressed_detailed + compressed_complete
        
        return {
            "original_size_bytes": original_size,
            "optimized_size_bytes": total_optimized_size,
            "essential_size_bytes": essential_size,
            "compression_ratio": round(original_size / max(total_optimized_size, 1), 2),
            "space_saved_percent": round(((original_size - total_optimized_size) / original_size) * 100, 1),
            "api_response_size": essential_size,  # What gets sent to client
            "storage_efficiency": "high" if total_optimized_size < original_size * 0.1 else "medium"
        }
    
    def export_for_other_agents(self, tiered_data: TieredFloorplanData) -> Dict[str, Any]:
        """Export essential data for other agents in the system"""
        
        export_data = tiered_data.essential.copy()
        
        # Add summary data that other agents might need
        if tiered_data.summary:
            export_data.update({
                "has_electrical": tiered_data.summary.get("has_electrical", False),
                "has_dimensions": tiered_data.summary.get("has_dimensions", False),
                "scale_factor": tiered_data.summary.get("scale_factor", 1.0),
                "wall_length": tiered_data.summary.get("wall_length", 0)
            })
        
        # Add metadata for agent coordination
        export_data["data_source"] = "floorplan_agent"
        export_data["data_version"] = tiered_data.data_version
        export_data["confidence_level"] = "high" if export_data.get("confidence_score", 0) > 75 else "medium"
        
        return export_data

class FloorplanInterpreterChatAgent(ChatIntegratedAgent):
    """Enhanced floorplan interpreter combining Romanian chat + full DXF analysis"""
    
    def __init__(self):
        super().__init__("floorplan_interpreter")
        self.supported_file_types = [".dxf", ".dwg", ".pdf"]
        self.extracted_data = {}
        
        # Romanian construction room types
        self.romanian_room_types = {
            'bucătărie': 'kitchen',
            'baie': 'bathroom', 
            'dormitor': 'bedroom',
            'living': 'living_room',
            'hol': 'hallway',
            'debara': 'storage',
            'sufragerie': 'living_room',
            'vestibul': 'entrance',
            'dresing': 'dressing_room',
            'balcon': 'balcony',
            'terasa': 'terrace'
        }
        self.text_classifier = IntelligentTextClassifier()
        self.smart_data_extractor = SmartDataExtractor(self.text_classifier)
        
        logger.info("✅ Enhanced floorplan agent initialized with full capabilities")
    
    async def analyze_initial_data(self, session_id: str, files_data: Dict[str, Any]) -> AgentDataRequirements:
        """Enhanced analysis combining file processing with chat requirements"""
        logger.info(f"🔍 Starting enhanced floorplan analysis for session {session_id}")
        
        files = files_data.get('files', [])
        
        # Initialize base confidence and missing data
        base_confidence = 20.0
        missing_data = []
        
        # Check for architectural files
        has_dxf = any(f.get('filename', '').lower().endswith('.dxf') for f in files)
        has_dwg = any(f.get('filename', '').lower().endswith('.dwg') for f in files)
        has_architectural_pdf = any('plan' in f.get('filename', '').lower() for f in files)
        
        # Enhanced file processing
        if has_dxf and DXF_AVAILABLE:
            logger.info("📐 Processing DXF file with full analysis")
            dxf_analysis = await self._process_dxf_files(files)
            if dxf_analysis:
                base_confidence += 40.0  # Major boost for actual DXF processing
                self._update_session_data_from_dxf(session_id, dxf_analysis)
            else:
                base_confidence += 25.0  # Partial boost for DXF file presence
        elif has_dxf:
            logger.info("📋 DXF file detected but ezdxf not available")
            base_confidence += 25.0
        elif has_dwg:
            base_confidence += 20.0
        elif has_architectural_pdf:
            base_confidence += 15.0
        
        # Extract basic info from filenames and content
        basic_info = self._extract_basic_info(files)
        if basic_info.get('total_area'):
            base_confidence += 15.0
        if basic_info.get('room_count'):
            base_confidence += 10.0
        
        # Define missing data requirements
        if not has_dxf and not has_dwg:
            missing_data.append(TechnicalField(
                field_name="architectural_plan",
                description="planul arhitectural în format DXF sau DWG",
                data_type="file",
                priority="critical",
                technical_context="Necesar pentru analiza precisă a spațiului",
                confidence_impact=30.0
            ))
        
        if not basic_info.get('total_area'):
            missing_data.append(TechnicalField(
                field_name="total_area",
                description="suprafața totală construită",
                data_type="measurement",
                priority="important",
                technical_context="Bază pentru calculele de cost și materiale",
                confidence_impact=15.0
            ))
        
        if not basic_info.get('room_count'):
            missing_data.append(TechnicalField(
                field_name="room_count",
                description="numărul total de camere",
                data_type="count",
                priority="important",
                technical_context="Pentru validarea planului și estimări",
                confidence_impact=10.0
            ))
        
        missing_data.append(TechnicalField(
            field_name="building_height",
            description="înălțimea clădirii",
            data_type="measurement",
            priority="optional",
            technical_context="Pentru calculele de volum și instalații",
            confidence_impact=8.0
        ))
        
        missing_data.append(TechnicalField(
            field_name="room_layout",
            description="dispunerea și funcțiunea camerelor",
            data_type="layout",
            priority="optional",
            technical_context="Pentru optimizarea instalațiilor MEP",
            confidence_impact=8.0
        ))
        
        # Generate technical questions
        technical_questions = self._generate_floorplan_questions(missing_data)
        
        can_proceed = base_confidence >= 50.0
        
        logger.info(f"📊 Enhanced floorplan analysis complete: {base_confidence:.1f}% confidence")
        
        return AgentDataRequirements(
            agent_name=self.agent_name,
            confidence_score=base_confidence,
            can_proceed=can_proceed,
            missing_data=missing_data,
            technical_questions=technical_questions
        )
    
    async def _process_dxf_files(self, files: List[Dict]) -> Optional[Dict[str, Any]]:
        """Process DXF files using enhanced capabilities"""
        if not DXF_AVAILABLE:
            logger.warning("DXF processing not available - ezdxf not installed")
            return None
        
        for file_data in files:
            filename = file_data.get('filename', '').lower()
            if filename.endswith('.dxf'):
                try:
                    # Get file content
                    file_content = file_data.get('content')
                    if not file_content:
                        continue
                    
                    # Create temporary file for processing
                    with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as temp_file:
                        if isinstance(file_content, bytes):
                            temp_file.write(file_content)
                        else:
                            temp_file.write(file_content.encode())
                        temp_path = temp_file.name
                    
                    # Process DXF file
                    analysis = await self._analyze_dxf_file(temp_path)
                    
                    # Cleanup
                    os.unlink(temp_path)
                    
                    if analysis:
                        logger.info("✅ DXF analysis completed successfully")
                        return analysis
                
                except Exception as e:
                    logger.error(f"❌ Error processing DXF file {filename}: {e}")
                    continue
        
        return None
    
    async def _analyze_dxf_file(self, file_path: str) -> Dict[str, Any]:
        """Comprehensive DXF analysis using original agent capabilities"""
        try:
            # Read DXF file
            doc = readfile(file_path)
            logger.info(f"📍 Successfully parsed DXF file")
            
            # Use SmartDataExtractor for optimized extraction
            optimized_data = self.smart_data_extractor.extract_optimized_room_data(doc)
            
            # Build the full analysis dictionary for backward compatibility and detailed views
            analysis = {
                "drawing_info": self._extract_drawing_info(doc),
                "optimized_data": optimized_data,
                "walls": self._extract_walls(doc),
                "fixtures": self._extract_fixtures(doc),
                "dimensions": self._extract_dimensions(doc),
                "notes": self._extract_notes(doc),
                "layers": self._extract_layer_info(doc),
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "analysis_version": "enhanced_v2_phase2"
            }
            
            logger.info(f"📊 DXF analysis complete: {optimized_data.total_rooms} rooms, {optimized_data.total_area:.2f} sqm area")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing DXF file: {e}")
            return {}

    def _validate_and_fix_scale(self, analysis: Dict[str, Any], doc=None) -> Dict[str, Any]:
        """
        Validate scale and fix unrealistic measurements using a multi-pronged approach:
        1. Check drawing units ($INSUNITS).
        2. Analyze drawing extents vs. area.
        3. Fallback to dimension entity analysis.
        """
        spatial = analysis.get("spatial_analysis", {})
        estimated_area = spatial.get("estimated_total_area", 0)
        drawing_info = analysis.get("drawing_info", {})
        units = drawing_info.get("units", "Unknown")

        # Realistic area bounds (10 m² to 10,000 m²)
        is_unrealistic = (estimated_area > 0) and not (10 <= estimated_area <= 10000)

        if not is_unrealistic:
            logger.info(f"Area {estimated_area:.1f} m² is within realistic bounds. No scale correction needed.")
            return analysis

        logger.warning(f"Detected unrealistic area: {estimated_area:.1f} m². Attempting scale correction.")
        spatial["original_estimated_area"] = estimated_area

        # --- Scale Correction Logic ---
        # Common scale factors (mm to m, cm to m)
        unit_corrections = {
            'Millimeters': (1000**2, "mm -> m"),
            'Centimeters': (100**2, "cm -> m"),
            'Inches': ((1 / 0.0254)**2, "in -> m"),
        }

        # 1. Check INSUNITS header
        if units in unit_corrections:
            correction_factor, reason = unit_corrections[units]
            corrected_area = estimated_area / correction_factor
            if 10 <= corrected_area <= 10000:
                logger.info(f"Applied scale correction based on INSUNITS ({units}): {correction_factor:.1f}x")
                spatial["estimated_total_area"] = corrected_area
                spatial["scale_corrected"] = True
                spatial["scale_factor_applied"] = correction_factor
                spatial["scale_correction_reason"] = f"Unitati detectate: {reason}"
                return analysis

        # 2. Fallback to common scale issues if INSUNITS is not helpful
        common_scale_factors = [100, 1000, 10000]  # Common drawing errors
        for factor in common_scale_factors:
            corrected_area = estimated_area / factor
            if 10 <= corrected_area <= 10000:
                logger.info(f"Applied fallback scale correction: {factor}x")
                spatial["estimated_total_area"] = corrected_area
                spatial["scale_corrected"] = True
                spatial["scale_factor_applied"] = factor
                spatial["scale_correction_reason"] = "Fallback based on common scale error"
                return analysis

        # 3. If still unrealistic, mark for manual validation
        spatial["requires_scale_validation"] = True
        spatial["scale_warning"] = f"Suprafața calculată ({estimated_area:.1f} mp) pare nerealistă și nu a putut fi corectată automat."
        logger.error("Scale validation failed. Manual override required.")
        return analysis
    
    def _extract_drawing_info(self, doc) -> Dict[str, Any]:
        """Extract general drawing information and metadata"""
        try:
            header = doc.header
            
            # Map INSUNITS code to a readable string
            unit_map = {0: 'Unitless', 1: 'Inches', 2: 'Feet', 3: 'Miles', 4: 'Millimeters', 5: 'Centimeters', 6: 'Meters'}
            insunits_code = header.get('$INSUNITS', 0)
            units_str = unit_map.get(insunits_code, f'Unknown code: {insunits_code}')

            drawing_info = {
                "dxf_version": header.get('$ACADVER', 'Unknown'),
                "units": units_str,
                "units_code": insunits_code,
                "scale": header.get('$DIMSCALE', 1.0),
                "extents": {
                    "min": list(header.get('$EXTMIN', (0, 0, 0)))[:2],
                    "max": list(header.get('$EXTMAX', (100, 100, 0)))[:2]
                },
                "layer_count": len(doc.layers),
                "block_count": len(doc.blocks)
            }
            return drawing_info
        except Exception as e:
            logger.error(f"Error extracting drawing info: {e}")
            return {"error": str(e)}
    
    def _extract_walls(self, doc) -> Dict[str, List]:
        """Enhanced wall extraction"""
        walls = []
        wall_entities = {
            'lines': [],
            'polylines': [],
            'lwpolylines': []
        }
        
        # Extract LINE entities
        try:
            for entity in doc.modelspace().query('LINE'):
                line_data = {
                    "type": "line",
                    "start": list(entity.dxf.start)[:2],
                    "end": list(entity.dxf.end)[:2],
                    "layer": getattr(entity.dxf, 'layer', 'Unknown'),
                    "length": math.sqrt(
                        (entity.dxf.end[0] - entity.dxf.start[0])**2 + 
                        (entity.dxf.end[1] - entity.dxf.start[1])**2
                    )
                }
                wall_entities['lines'].append(line_data)
                walls.append(line_data)
        except Exception as e:
            logger.info(f"No LINE entities found: {e}")
        
        # Extract LWPOLYLINE entities
        try:
            for entity in doc.modelspace().query('LWPOLYLINE'):
                vertices = [list(p)[:2] for p in entity.get_points()]
                polyline_data = {
                    "type": "lwpolyline",
                    "vertices": vertices,
                    "is_closed": entity.is_closed,
                    "layer": getattr(entity.dxf, 'layer', 'Unknown'),
                    "total_length": self._calculate_polyline_length(vertices)
                }
                wall_entities['lwpolylines'].append(polyline_data)
                walls.append(polyline_data)
        except Exception as e:
            logger.info(f"No LWPOLYLINE entities found: {e}")
        
        logger.info(f"Extracted {len(walls)} wall entities")
        return {"entities": walls, "by_type": wall_entities}
    
    def _extract_fixtures(self, doc) -> List[Dict]:
        """Extract fixtures, doors, windows"""
        fixtures = []
        
        # Look for block references (doors, windows often represented as blocks)
        try:
            for entity in doc.modelspace().query('INSERT'):
                fixture = {
                    "type": "block_reference",
                    "block_name": entity.dxf.name,
                    "location": list(entity.dxf.insert)[:2],
                    "rotation": getattr(entity.dxf, 'rotation', 0),
                    "scale": [
                        getattr(entity.dxf, 'xscale', 1.0),
                        getattr(entity.dxf, 'yscale', 1.0)
                    ],
                    "layer": getattr(entity.dxf, 'layer', 'Unknown')
                }
                fixtures.append(fixture)
        except Exception as e:
            logger.info(f"No INSERT entities found: {e}")
        
        # Look for circles (might represent fixtures)
        try:
            for entity in doc.modelspace().query('CIRCLE'):
                fixture = {
                    "type": "circle",
                    "center": list(entity.dxf.center)[:2],
                    "radius": entity.dxf.radius,
                    "layer": getattr(entity.dxf, 'layer', 'Unknown')
                }
                fixtures.append(fixture)
        except Exception as e:
            logger.info(f"No CIRCLE entities found: {e}")
        
        logger.info(f"Extracted {len(fixtures)} fixture entities")
        return fixtures
    
    def _extract_dimensions(self, doc) -> List[Dict]:
        """Extract dimension entities and measurements"""
        dimensions = []
        
        # Extract dimension entities
        for dim_type in ['DIMENSION', 'ALIGNED_DIMENSION', 'LINEAR_DIMENSION']:
            try:
                for entity in doc.modelspace().query(dim_type):
                    dim_info = {
                        "type": dim_type.lower(),
                        "measurement": getattr(entity.dxf, 'measurement', None),
                        "text": getattr(entity.dxf, 'text', ''),
                        "location": list(getattr(entity.dxf, 'text_midpoint', [0, 0]))[:2],
                        "layer": getattr(entity.dxf, 'layer', 'Unknown')
                    }
                    dimensions.append(dim_info)
            except Exception as e:
                logger.info(f"No {dim_type} entities found: {e}")
        
        # Extract measurements from text entities
        text_measurements = self._extract_measurements_from_text(doc)
        dimensions.extend(text_measurements)
        
        logger.info(f"Extracted {len(dimensions)} dimension entities")
        return dimensions
    
    def _extract_notes(self, doc) -> List[Dict]:
        """Extract notes, labels, and annotations"""
        notes = []
        
        for entity_type in ['MTEXT', 'TEXT']:
            try:
                for entity in doc.modelspace().query(entity_type):
                    text_content = entity.dxf.text.strip()
                    if len(text_content) > 2:  # Filter out very short text
                        note = {
                            "text": text_content,
                            "location": list(entity.dxf.insert)[:2] if hasattr(entity.dxf, 'insert') else [0, 0],
                            "type": entity_type.lower(),
                            "layer": getattr(entity.dxf, 'layer', 'Unknown'),
                            "height": getattr(entity.dxf, 'height', 0)
                        }
                        notes.append(note)
            except Exception as e:
                logger.info(f"No {entity_type} entities found: {e}")
        
        logger.info(f"Extracted {len(notes)} text notes")
        return notes
    
    def _extract_layer_info(self, doc) -> List[Dict]:
        """Extract layer information"""
        layers = []
        try:
            for layer in doc.layers:
                layer_info = {
                    "name": layer.dxf.name,
                    "color": layer.dxf.color,
                    "linetype": layer.dxf.linetype,
                    "frozen": layer.is_frozen(),
                    "locked": layer.is_locked(),
                    "on": layer.is_on()
                }
                layers.append(layer_info)
            logger.info(f"Extracted {len(layers)} layer definitions")
        except Exception as e:
            logger.error(f"Error extracting layers: {e}")
        return layers
    
    # Helper methods for DXF processing
    def _extract_room_boundaries(self, doc) -> List[Dict]:
        """Extract closed polylines that might represent room boundaries"""
        boundaries = []
        try:
            for entity in doc.modelspace().query('LWPOLYLINE'):
                if entity.is_closed:
                    vertices = [list(p)[:2] for p in entity.get_points()]
                    area = self._calculate_polygon_area(vertices)
                    if area > 1.0:  # Filter out very small polygons
                        boundary = {
                            "vertices": vertices,
                            "area": area,
                            "layer": getattr(entity.dxf, 'layer', 'Unknown')
                        }
                        boundaries.append(boundary)
        except Exception as e:
            logger.info(f"No closed polylines found: {e}")
        return boundaries
    
    def _calculate_polyline_length(self, vertices: List[List[float]]) -> float:
        """Calculate total length of a polyline"""
        if len(vertices) < 2:
            return 0
        total_length = 0
        for i in range(len(vertices) - 1):
            dx = vertices[i+1][0] - vertices[i][0]
            dy = vertices[i+1][1] - vertices[i][1]
            total_length += math.sqrt(dx*dx + dy*dy)
        return total_length
    
    def _calculate_polygon_area(self, vertices: List[List[float]]) -> float:
        """Calculate area of a polygon using shoelace formula"""
        if len(vertices) < 3:
            return 0
        area = 0
        for i in range(len(vertices)):
            j = (i + 1) % len(vertices)
            area += vertices[i][0] * vertices[j][1]
            area -= vertices[j][0] * vertices[i][1]
        return abs(area) / 2
    
    def _extract_measurements_from_text(self, doc) -> List[Dict]:
        """Extract measurements from text entities using regex patterns"""
        measurements = []
        for entity_type in ['MTEXT', 'TEXT']:
            try:
                for entity in doc.modelspace().query(entity_type):
                    text = entity.dxf.text
                    for pattern in self.measurement_patterns:
                        matches = re.finditer(pattern, text)
                        for match in matches:
                            measurements.append({
                                "type": "text_measurement",
                                "text": match.group(0),
                                "location": list(entity.dxf.insert)[:2],
                                "layer": getattr(entity.dxf, 'layer', 'Unknown'),
                                "pattern_matched": pattern
                            })
            except Exception:
                continue
        return measurements
    
    def _perform_spatial_analysis(self, rooms: Dict, walls: Dict, fixtures: List) -> Dict[str, Any]:
        """DEPRECATED: Perform spatial analysis of the floorplan. Logic moved to SmartDataExtractor."""
        analysis = {
            "total_rooms": len(rooms.get("labels", [])),
            "total_walls": len(walls.get("entities", [])),
            "total_fixtures": len(fixtures),
            "room_types": {},
            "estimated_total_area": 0,
            "wall_total_length": 0,
            "analysis_source": "legacy"
        }
        
        # Analyze room types
        for room in rooms.get("labels", []):
            room_type = room.get("potential_room_type", "unknown")
            analysis["room_types"][room_type] = analysis["room_types"].get(room_type, 0) + 1
        
        # Calculate total wall length
        for wall in walls.get("entities", []):
            if wall.get("length"):
                analysis["wall_total_length"] += wall["length"]
            elif wall.get("total_length"):
                analysis["wall_total_length"] += wall["total_length"]
        
        # Estimate area from room boundaries
        for boundary in rooms.get("boundaries", []):
            if boundary.get("area"):
                analysis["estimated_total_area"] += boundary["area"]
        
        return analysis
    
    def _calculate_dxf_confidence(self, analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calculate confidence factors based on DXF analysis"""
        factors = {"confidence_source": "legacy"}
        
        # Room detection confidence
        room_count = analysis.get("spatial_analysis", {}).get("total_rooms", 0)
        factors["room_detection"] = min(room_count * 10, 50)  # Max 50 points
        
        # Wall detection confidence
        wall_count = analysis.get("spatial_analysis", {}).get("total_walls", 0)
        factors["wall_detection"] = min(wall_count * 2, 30)  # Max 30 points
        
        # Dimension information confidence
        dimension_count = len(analysis.get("dimensions", []))
        factors["dimension_info"] = min(dimension_count * 5, 20)  # Max 20 points
        
        return factors
    
    def _update_session_data_from_dxf(self, session_id: str, analysis: Dict[str, Any]):
        """Update session data with DXF analysis results"""
        if session_id not in self.extracted_data:
            self.extracted_data[session_id] = {}
        
        # Extract tiered storage if available
        tiered_storage = analysis.get("tiered_storage")
        if tiered_storage:
            # Store only essential data for session
            essential_data = self.storage_manager.get_data_by_level(
                tiered_storage, DetailLevel.ESSENTIAL
            )
            
            self.extracted_data[session_id].update(essential_data)
            
            # Store reference to detailed data (compressed)
            self.extracted_data[session_id]["_has_detailed_analysis"] = True
            self.extracted_data[session_id]["_tiered_storage"] = tiered_storage
            
            logger.info(f"✅ Stored optimized session data: "
                       f"{len(json.dumps(essential_data))} bytes vs "
                       f"{analysis.get('storage_efficiency', {}).get('original_size_bytes', 0)} original")
        
        else:
            # Fallback to legacy method (should not happen with optimized version)
            spatial = analysis.get("spatial_analysis", {})
            
            if spatial.get("total_rooms"):
                self.extracted_data[session_id]["room_count"] = spatial["total_rooms"]
            if spatial.get("estimated_total_area"):
                self.extracted_data[session_id]["total_area"] = spatial["estimated_total_area"]
            if spatial.get("room_types"):
                self.extracted_data[session_id]["room_types"] = spatial["room_types"]

    def generate_summary_description(self, session_id: str) -> str:
        """Generate concise summary using optimized storage"""
        try:
            session_data = self.extracted_data.get(session_id, {})
            
            # Check if we have tiered storage
            tiered_storage = session_data.get("_tiered_storage")
            if tiered_storage:
                essential_data = self.storage_manager.get_data_by_level(
                    tiered_storage, DetailLevel.ESSENTIAL
                )
                return self.storage_manager._generate_romanian_description(essential_data)
            
            # Fallback to legacy data
            total_rooms = session_data.get("room_count", session_data.get("total_rooms", 0))
            total_area = session_data.get("total_area", 0)
            room_types = session_data.get("room_types", {})
            
            parts = []
            
            if total_rooms > 0:
                parts.append(f"{total_rooms} spatii")
            
            if total_area > 0:
                parts.append(f"{total_area:.1f} mp")
            
            if room_types:
                romanian_types = [self._get_romanian_room_name(rt) for rt in room_types.keys() 
                                if rt != "unknown"][:3]  # Limit to 3 types
                if romanian_types:
                    parts.append(f"tipuri: {', '.join(romanian_types)}")
            
            return f"Plan: {', '.join(parts)}" if parts else "Plan DXF analizat"
            
        except Exception as e:
            logger.error(f"Error generating optimized summary: {e}")
            return "Plan procesat cu sistem optimizat"

    def get_detailed_analysis_on_demand(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed analysis only when specifically requested"""
        try:
            session_data = self.extracted_data.get(session_id, {})
            tiered_storage = session_data.get("_tiered_storage")
            
            if not tiered_storage:
                return None
            
            # Load detailed data (decompressed)
            detailed_data = self.storage_manager.get_data_by_level(
                tiered_storage, DetailLevel.DETAILED
            )
            
            logger.info(f"📋 Loaded detailed analysis on demand for session {session_id}")
            return detailed_data
            
        except Exception as e:
            logger.error(f"Error loading detailed analysis: {e}")
            return None

    def generate_api_response_optimized(self, session_id: str, 
                                      include_details: bool = False) -> Dict[str, Any]:
        """Generate optimized API response with size control"""
        try:
            session_data = self.extracted_data.get(session_id, {})
            tiered_storage = session_data.get("_tiered_storage")
            
            if tiered_storage:
                return self.storage_manager.generate_api_response(
                    tiered_storage, include_summary=include_details
                )
            
            # Fallback for legacy data
            essential_data = {
                "total_rooms": session_data.get("room_count", 0),
                "total_area": session_data.get("total_area", 0.0),
                "room_types": session_data.get("room_types", {}),
                "confidence_score": 75.0  # Default confidence
            }
            
            description = self.storage_manager._generate_romanian_description(essential_data)
            
            return {
                "floorplan_analysis": {
                    "summary": description,
                    "data": essential_data,
                    "confidence": essential_data["confidence_score"],
                    "note": "Processed with legacy compatibility mode"
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating optimized API response: {e}")
            return {
                "floorplan_analysis": {
                    "summary": "Plan analizat cu sistem optimizat",
                    "data": {"total_rooms": 0, "total_area": 0.0},
                    "confidence": 0.0,
                    "error": str(e)
                }
            }

    def export_for_orchestrator_optimized(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Export optimized data for orchestrator with size guarantees"""
        try:
            session_data = self.extracted_data.get(session_id, {})
            tiered_storage = session_data.get("_tiered_storage")
            
            if tiered_storage:
                export_data = self.storage_manager.export_for_other_agents(tiered_storage)
                
                # Guarantee size limit for orchestrator
                export_json = json.dumps(export_data)
                if len(export_json) > 2000:  # 2KB limit for orchestrator
                    # Use only essential data if export is too large
                    export_data = self.storage_manager.get_data_by_level(
                        tiered_storage, DetailLevel.ESSENTIAL
                    )
                    export_data["size_limited"] = True
                    logger.warning(f"Export data truncated for size limit: {session_id}")
                
                return export_data
            
            return None
            
        except Exception as e:
            logger.error(f"Error creating optimized export: {e}")
            return None

    # Performance monitoring method
    def get_storage_metrics(self, session_id: str) -> Dict[str, Any]:
        """Get storage efficiency metrics for monitoring"""
        try:
            session_data = self.extracted_data.get(session_id, {})
            tiered_storage = session_data.get("_tiered_storage")
            
            if not tiered_storage:
                return {"status": "no_tiered_storage"}
            
            # Calculate current storage usage
            essential_size = len(json.dumps(tiered_storage.essential))
            summary_size = len(json.dumps(tiered_storage.summary)) if tiered_storage.summary else 0
            detailed_size = len(tiered_storage.detailed_compressed) if tiered_storage.detailed_compressed else 0
            
            return {
                "essential_bytes": essential_size,
                "summary_bytes": summary_size, 
                "detailed_compressed_bytes": detailed_size,
                "total_optimized_bytes": essential_size + summary_size + detailed_size,
                "processing_time_ms": tiered_storage.processing_time_ms,
                "data_version": tiered_storage.data_version,
                "optimization_status": "active"
            }
            
        except Exception as e:
            logger.error(f"Error getting storage metrics: {e}")
            return {"status": "error", "error": str(e)}
    
    # Enhanced Romanian chat processing methods
    async def process_engineer_input(self, engineer_input: str, context: AgentContext) -> AgentResponse:
        """Enhanced processing of engineer input with DXF context awareness"""
        logger.info(f"💬 Processing enhanced floorplan input: {engineer_input[:50]}...")
        
        input_lower = engineer_input.lower()
        confidence_delta = 0.0
        status_updates = {"resolved_fields": [], "new_missing_fields": []}
        response_content = ""
        new_questions = []
        
        # Check if we have DXF analysis data for this session
        session_data = self.extracted_data.get(context.session_id, {})
        has_dxf_analysis = "dxf_analysis" in session_data
        
        # Extract numeric values for area, rooms, dimensions
        area_match = self._extract_area(engineer_input)
        room_match = self._extract_room_count(engineer_input)
        height_match = self._extract_height(engineer_input)
        
        # Process area information
        if area_match:
            confidence_delta += 25.0
            status_updates["resolved_fields"].append("total_area")
            response_content += f"Am înregistrat suprafața de {area_match} mp. "
            
            if context.session_id not in self.extracted_data:
                self.extracted_data[context.session_id] = {}
            self.extracted_data[context.session_id]["total_area"] = area_match
            
            # Cross-validate with DXF analysis if available
            if has_dxf_analysis:
                dxf_area = session_data.get("total_area")
                if dxf_area and abs(area_match - dxf_area) / max(area_match, dxf_area) < 0.2:
                    response_content += "Această valoare este consistentă cu analiza planului DXF. "
                    confidence_delta += 10.0  # Bonus for consistency
        
        # Process room count
        if room_match:
            confidence_delta += 15.0
            status_updates["resolved_fields"].append("room_count")
            response_content += f"Am înregistrat {room_match} camere. "
            
            if context.session_id not in self.extracted_data:
                self.extracted_data[context.session_id] = {}
            self.extracted_data[context.session_id]["room_count"] = room_match
            
            # Cross-validate with DXF analysis if available
            if has_dxf_analysis:
                dxf_rooms = session_data.get("room_count")
                if dxf_rooms and room_match == dxf_rooms:
                    response_content += "Confirmă numărul de camere din planul DXF. "
                    confidence_delta += 10.0  # Bonus for consistency
        
        # Process height information
        if height_match:
            confidence_delta += 10.0
            status_updates["resolved_fields"].append("building_height")
            response_content += f"Am înregistrat înălțimea de {height_match}m. "
            
            if context.session_id not in self.extracted_data:
                self.extracted_data[context.session_id] = {}
            self.extracted_data[context.session_id]["building_height"] = height_match
        
        # Check for room layout descriptions
        room_types_mentioned = []
        for romanian_room, english_room in self.romanian_room_types.items():
            if romanian_room in input_lower:
                room_types_mentioned.append(romanian_room)
        
        if room_types_mentioned:
            confidence_delta += 8.0
            status_updates["resolved_fields"].append("room_layout")
            response_content += f"Am identificat tipurile de camere: {', '.join(room_types_mentioned)}. "
            
            # Store room layout information
            if context.session_id not in self.extracted_data:
                self.extracted_data[context.session_id] = {}
            self.extracted_data[context.session_id]["room_layout"] = room_types_mentioned
        
        # Enhanced questioning based on DXF analysis context
        if has_dxf_analysis:
            # We have DXF data, ask more specific questions
            dxf_analysis = session_data["dxf_analysis"]
            spatial = dxf_analysis.get("spatial_analysis", {})
            
            if confidence_delta > 0:
                if area_match and not room_match and spatial.get("total_rooms"):
                    new_questions.append(f"Din analiza planului DXF am detectat {spatial['total_rooms']} camere. Este corect?")
                elif room_match and not area_match and spatial.get("estimated_total_area"):
                    estimated_area = spatial["estimated_total_area"]
                    new_questions.append(f"Conform planului DXF, suprafața estimată este {estimated_area:.1f} mp. Confirmați?")
                elif area_match and room_match:
                    # Ask about specific room functions detected in DXF
                    room_types = spatial.get("room_types", {})
                    if room_types:
                        detected_types = list(room_types.keys())
                        new_questions.append(f"Am detectat următoarele tipuri de spații: {', '.join(detected_types)}. Care sunt funcțiunile exacte?")
            else:
                # No progress, but we have DXF data to help
                if spatial.get("total_rooms") and not session_data.get("room_count"):
                    new_questions.append(f"Din planul DXF am identificat {spatial['total_rooms']} spații. Câte sunt camere locuibile?")
                elif spatial.get("estimated_total_area") and not session_data.get("total_area"):
                    estimated_area = spatial["estimated_total_area"]
                    new_questions.append(f"Suprafața calculată din plan este {estimated_area:.1f} mp. Este suprafața utilă sau construită?")
        else:
            # No DXF analysis, use standard questioning
            if confidence_delta > 0:
                if area_match and not room_match:
                    new_questions.append("Câte camere are această suprafață?")
                elif room_match and not area_match:
                    new_questions.append("Care este suprafața totală a acestor camere?")
                elif area_match and room_match:
                    new_questions.append("Aveți dimensiunile individuale ale camerelor?")
            else:
                new_questions.append("Care este suprafața totală construită?")
                new_questions.append("Câte camere are proiectul?")
        
        # Add DXF-specific insights to response
        if has_dxf_analysis and not response_content:
            dxf_analysis = session_data["dxf_analysis"]
            spatial = dxf_analysis.get("spatial_analysis", {})
            
            insights = []
            if spatial.get("total_rooms"):
                insights.append(f"{spatial['total_rooms']} spații identificate în plan")
            if spatial.get("wall_total_length"):
                insights.append(f"{spatial['wall_total_length']:.1f}m lungime totală pereți")
            if spatial.get("total_fixtures"):
                insights.append(f"{spatial['total_fixtures']} elemente fixe detectate")
            
            if insights:
                response_content = f"Din analiza planului DXF: {', '.join(insights)}. "
                confidence_delta += 5.0
        
        if not response_content:
            response_content = "Am procesat informațiile despre plan. "
        
        return AgentResponse(
            content=response_content.strip(),
            confidence_delta=confidence_delta,
            technical_questions=new_questions[:2],  # Limit to 2 questions
            status_update=status_updates
        )
    
    def generate_technical_questions(self) -> List[str]:
        """Generate enhanced technical questions with DXF context"""
        return [
            "Aveți planul arhitectural în format DXF sau DWG?",
            "Care este suprafața totală construită?",
            "Câte camere are proiectul?", 
            "Care este înălțimea clădirii?",
            "Cum sunt organizate camerele în plan?",
            "Ce instalații MEP sunt necesare?"
        ]
    
    def _extract_basic_info(self, files: List[Dict]) -> Dict[str, Any]:
        """Enhanced basic information extraction from uploaded files"""
        info = {}
        
        for file_data in files:
            filename = file_data.get('filename', '').lower()
            
            # Look for area information in filename
            area_match = self._extract_area(filename)
            if area_match:
                info['total_area'] = area_match
            
            # Look for room count in filename
            room_match = self._extract_room_count(filename)
            if room_match:
                info['room_count'] = room_match
                
            # Detect file type and set flags
            if filename.endswith('.dxf'):
                info['has_dxf'] = True
            elif filename.endswith('.dwg'):
                info['has_dwg'] = True
            elif 'plan' in filename and filename.endswith('.pdf'):
                info['has_architectural_pdf'] = True
        
        return info
    
    def _extract_area(self, text: str) -> Optional[float]:
        """Extract area from text (Enhanced Romanian patterns)"""
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
    
    def _extract_room_count(self, text: str) -> Optional[int]:
        """Extract room count from text (Enhanced Romanian patterns)"""
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
    
    def _extract_height(self, text: str) -> Optional[float]:
        """Extract height from text (Enhanced Romanian patterns)"""
        patterns = [
            r'(?:înălțime|height|h)\s*(?:de\s*)?(\d+(?:[.,]\d+)?)\s*(?:m|metri)',
            r'(\d+(?:[.,]\d+)?)\s*(?:m|metri)\s*(?:înălțime|height)',
            r'înălțimea?\s*(?:este\s*|de\s*)?(\d+(?:[.,]\d+)?)',
            r'(?:h|H)\s*=\s*(\d+(?:[.,]\d+)?)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                try:
                    return float(match.group(1).replace(',', '.'))
                except ValueError:
                    continue
        return None
    
    def _generate_floorplan_questions(self, missing_data: List[TechnicalField]) -> List[str]:
        """Generate contextual floorplan questions based on missing data"""
        questions = []
        
        for field in missing_data:
            if field.field_name == "architectural_plan":
                questions.append("Aveți planul arhitectural în format DXF sau DWG? Vă rog să-l încărcați.")
            elif field.field_name == "total_area":
                questions.append("Care este suprafața totală construită în metri pătrați?")
            elif field.field_name == "room_count":
                questions.append("Câte camere are proiectul total?")
            elif field.field_name == "building_height":
                questions.append("Care este înălțimea clădirii (în metri)?")
            elif field.field_name == "room_layout":
                questions.append("Cum sunt organizate camerele (bucătărie, baie, dormitoare, living)?")
        
        return questions[:3]  # Limit to 3 most important questions
    
    def get_dxf_analysis_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of DXF analysis for this session"""
        session_data = self.extracted_data.get(session_id, {})
        if "dxf_analysis" not in session_data:
            return None
        
        analysis = session_data["dxf_analysis"]
        spatial = analysis.get("spatial_analysis", {})
        
        return {
            "has_dxf_analysis": True,
            "rooms_detected": spatial.get("total_rooms", 0),
            "walls_detected": spatial.get("total_walls", 0),
            "estimated_area": spatial.get("estimated_total_area", 0),
            "room_types": spatial.get("room_types", {}),
            "confidence_factors": analysis.get("confidence_factors", {}),
            "drawing_info": analysis.get("drawing_info", {})
        }
    
    def export_dxf_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Export complete DXF analysis data for other agents"""
        session_data = self.extracted_data.get(session_id, {})
        return session_data.get("dxf_analysis")
    
    async def validate_with_dxf(self, user_input: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Validate user input against DXF analysis data"""
        validation_results = {
            "is_consistent": True,
            "discrepancies": [],
            "confidence_boost": 0.0
        }
        
        session_data = self.extracted_data.get(session_id, {})
        if "dxf_analysis" not in session_data:
            return validation_results
        
        dxf_analysis = session_data["dxf_analysis"]
        spatial = dxf_analysis.get("spatial_analysis", {})
        
        # Validate room count
        user_rooms = user_input.get("room_count")
        dxf_rooms = spatial.get("total_rooms")
        if user_rooms and dxf_rooms:
            if abs(user_rooms - dxf_rooms) <= 1:  # Allow for 1 room difference
                validation_results["confidence_boost"] += 15.0
            else:
                validation_results["is_consistent"] = False
                validation_results["discrepancies"].append(
                    f"Numărul de camere specificat ({user_rooms}) diferă de cel detectat în plan ({dxf_rooms})"
                )
        
        # Validate area
        user_area = user_input.get("total_area")
        dxf_area = spatial.get("estimated_total_area")
        if user_area and dxf_area:
            area_diff_percent = abs(user_area - dxf_area) / max(user_area, dxf_area)
            if area_diff_percent <= 0.15:  # Allow for 15% difference
                validation_results["confidence_boost"] += 20.0
            else:
                validation_results["is_consistent"] = False
                validation_results["discrepancies"].append(
                    f"Suprafața specificată ({user_area} mp) diferă semnificativ de cea calculată ({dxf_area:.1f} mp)"
                )
        
        return validation_results

    def generate_summary_description(self, session_id: str) -> str:
        """Generate concise summary for initial upload response"""
        try:
            session_data = self.extracted_data.get(session_id, {})
            dxf_analysis = session_data.get("dxf_analysis")
            
            if not dxf_analysis:
                return "Plan DXF procesat cu succes"
            
            spatial = dxf_analysis.get("spatial_analysis", {})
            total_rooms = spatial.get("total_rooms", 0)
            total_area = spatial.get("estimated_total_area", 0)
            room_types = spatial.get("room_types", {})
            
            summary_parts = []
            
            if total_rooms > 0:
                summary_parts.append(f"{total_rooms} spații")
            
            if total_area > 0:
                if spatial.get("requires_scale_validation"):
                    summary_parts.append(f"suprafață necesită validare ({total_area:.1f} mp)")
                else:
                    summary_parts.append(f"suprafață totală {total_area:.1f} mp")
            
            if room_types:
                romanian_types = [self._get_romanian_room_name(rt) for rt in room_types.keys() if rt != "unknown"]
                if romanian_types:
                    summary_parts.append(f"tipuri camere: {', '.join(romanian_types)}")
            
            base_summary = f"Plan detectat: {', '.join(summary_parts)}" if summary_parts else "Plan DXF analizat"
            
            # Add hint about detailed analysis
            return f"{base_summary}. (Analiză detaliată disponibilă la cerere)"
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Plan DXF procesat cu informații de bază disponibile"

    def generate_detailed_description(self, session_id: str) -> str:
        """Generate comprehensive technical description on request"""
        try:
            session_data = self.extracted_data.get(session_id, {})
            dxf_analysis = session_data.get("dxf_analysis")
            
            if not dxf_analysis:
                return "Nu există analiză DXF detaliată disponibilă pentru această sesiune."
            
            detailed_parts = []
            
            # Drawing technical info
            drawing_info = dxf_analysis.get("drawing_info", {})
            spatial = dxf_analysis.get("spatial_analysis", {})
            dimensions = dxf_analysis.get("dimensions", [])
            layers = dxf_analysis.get("layers", [])
            fixtures = dxf_analysis.get("fixtures", [])
            
            # Header
            detailed_parts.append("📐 **ANALIZĂ TEHNICĂ DETALIATĂ DXF**\n")
            
            # Drawing information
            if drawing_info:
                detailed_parts.append("🗂️ **Informații desen:**")
                if drawing_info.get("dxf_version"):
                    detailed_parts.append(f"- Versiune DXF: {drawing_info['dxf_version']}")
                if drawing_info.get("units"):
                    detailed_parts.append(f"- Unități: {drawing_info['units']}")
                if drawing_info.get("scale"):
                    detailed_parts.append(f"- Scară detectată: 1:{drawing_info['scale']}")
                detailed_parts.append("")
            
            # Spatial analysis
            if spatial:
                detailed_parts.append("📏 **Analiză spațială:**")
                if spatial.get("total_rooms"):
                    detailed_parts.append(f"- Numărul total de spații: {spatial['total_rooms']}")
                
                if spatial.get("estimated_total_area"):
                    area_text = f"- Suprafața totală: {spatial['estimated_total_area']:.1f} mp"
                    if spatial.get("scale_corrected"):
                        area_text += f" (corectată cu factorul {spatial.get('scale_factor_applied', 1)})"
                    elif spatial.get("requires_scale_validation"):
                        area_text += " ⚠️ (necesită validare scară)"
                    detailed_parts.append(area_text)
                
                if spatial.get("wall_total_length"):
                    detailed_parts.append(f"- Lungime totală pereți: {spatial['wall_total_length']:.1f} m")
                
                room_types = spatial.get("room_types", {})
                if room_types:
                    detailed_parts.append("- Tipuri de spații detectate:")
                    for room_type, count in room_types.items():
                        if room_type != "unknown":
                            romanian_name = self._get_romanian_room_name(room_type)
                            detailed_parts.append(f"  • {count}x {romanian_name}")
                detailed_parts.append("")
            
            # Dimensions and measurements
            if dimensions:
                detailed_parts.append("📐 **Informații cotare:**")
                detailed_parts.append(f"- Numărul de dimensiuni detectate: {len(dimensions)}")
                
                # Extract sample measurements
                measurements = [d.get("measurement") for d in dimensions if d.get("measurement")]
                if measurements:
                    sample_measurements = measurements[:3]  # Show first 3
                    detailed_parts.append(f"- Exemple măsurători: {', '.join(map(str, sample_measurements))}")
                detailed_parts.append("")
            
            # Layer organization
            if layers:
                detailed_parts.append("🎨 **Organizare straturi (layers):**")
                detailed_parts.append(f"- Numărul total de straturi: {len(layers)}")
                
                # Show active layers
                active_layers = [l["name"] for l in layers if l.get("on", True)][:5]
                if active_layers:
                    detailed_parts.append(f"- Straturi active: {', '.join(active_layers)}")
                detailed_parts.append("")
            
            # Fixtures and elements
            if fixtures:
                detailed_parts.append("🔧 **Elemente și fixtures:**")
                detailed_parts.append(f"- Total elemente detectate: {len(fixtures)}")
                
                # Group by type
                fixture_types = {}
                for fixture in fixtures:
                    f_type = fixture.get("type", "unknown")
                    fixture_types[f_type] = fixture_types.get(f_type, 0) + 1
                
                for f_type, count in fixture_types.items():
                    detailed_parts.append(f"  • {f_type}: {count}")
                detailed_parts.append("")
            
            # Quality indicators
            detailed_parts.append("✅ **Indicatori calitate analiză:**")
            completeness = self._calculate_analysis_completeness(dxf_analysis)
            detailed_parts.append(f"- Gradul de completare: {completeness:.1f}%")
            
            confidence_factors = dxf_analysis.get("confidence_factors", {})
            if confidence_factors:
                detailed_parts.append("- Factori încredere:")
                for factor, value in confidence_factors.items():
                    detailed_parts.append(f"  • {factor}: {value:.1f}%")
            
            # Warnings if any
            if spatial.get("requires_scale_validation"):
                detailed_parts.append("\n⚠️ **Avertismente:**")
                detailed_parts.append(f"- {spatial.get('scale_warning', 'Verificare scară necesară')}")
            
            return "\n".join(detailed_parts)
            
        except Exception as e:
            logger.error(f"Error generating detailed description: {e}")
            return "Eroare la generarea descrierii detaliate. Vă rugăm să încercați din nou."

    def generate_technical_description_from_analysis(self, session_id: str) -> Optional[str]:
        """Generate Romanian technical description from DXF analysis results"""
        try:
            session_data = self.extracted_data.get(session_id, {})
            dxf_analysis = session_data.get("dxf_analysis")
            
            if not dxf_analysis:
                return None
            
            description_parts = []
            
            # Extract key data from DXF analysis
            spatial = dxf_analysis.get("spatial_analysis", {})
            rooms = dxf_analysis.get("rooms", {})
            
            # Basic plan detection
            total_rooms = spatial.get("total_rooms", 0)
            total_area = spatial.get("estimated_total_area", 0)
            
            if total_rooms > 0:
                if total_area > 0:
                    description_parts.append(f"Plan detectat: {total_rooms} spatii, suprafata totala {total_area:.1f} mp")
                else:
                    description_parts.append(f"Plan detectat: {total_rooms} spatii identificate")
            
            # Room types analysis
            room_types = spatial.get("room_types", {})
            if room_types:
                romanian_room_names = []
                for room_type, count in room_types.items():
                    if room_type != "unknown":
                        romanian_name = self._get_romanian_room_name(room_type)
                        if count > 1:
                            romanian_room_names.append(f"{count} {romanian_name}")
                        else:
                            romanian_room_names.append(romanian_name)
                
                if romanian_room_names:
                    description_parts.append(f"Tipuri camere: {', '.join(romanian_room_names)}")
            
            # Technical quality indicators
            dimensions_data = dxf_analysis.get("dimensions", [])
            if len(dimensions_data) > 0:
                description_parts.append("cu cotare")
            
            # Layer organization
            layers = dxf_analysis.get("layers", [])
            if len(layers) > 3:
                description_parts.append("plan organizat pe straturi")
            
            # Combine all parts
            if description_parts:
                main_description = description_parts[0]
                if len(description_parts) > 1:
                    main_description += ", " + ", ".join(description_parts[1:])
                return main_description
            else:
                return "Plan architectural DXF analizat cu succes"
                
        except Exception as e:
            logger.error(f"Error generating DXF description: {e}")
            return None

    def _get_romanian_room_name(self, room_type: str) -> str:
        """Convert English room type to Romanian name"""
        romanian_names = {
            'bedroom': 'dormitor',
            'kitchen': 'bucatarie',
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

    def get_analysis_summary_for_orchestrator(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Export structured DXF analysis data for the orchestrator"""
        try:
            session_data = self.extracted_data.get(session_id, {})
            dxf_analysis = session_data.get("dxf_analysis")
            
            if not dxf_analysis:
                return None
            
            # Extract and structure key data for the orchestrator
            spatial = dxf_analysis.get("spatial_analysis", {})
            rooms = dxf_analysis.get("rooms", {})
            dimensions = dxf_analysis.get("dimensions", [])
            layers = dxf_analysis.get("layers", [])
            
            summary = {
                "has_dxf_analysis": True,
                "technical_description": self.generate_technical_description_from_analysis(session_id),
                "spatial_data": {
                    "total_rooms": spatial.get("total_rooms", 0),
                    "estimated_area": spatial.get("estimated_total_area", 0),
                    "room_types": spatial.get("room_types", {}),
                    "wall_length": spatial.get("wall_total_length", 0)
                },
                "technical_quality": {
                    "has_dimensions": len(dimensions) > 0,
                    "dimension_count": len(dimensions),
                    "layer_count": len(layers),
                    "has_room_labels": len(rooms.get("labels", [])) > 0,
                    "has_room_boundaries": len(rooms.get("boundaries", [])) > 0
                },
                "confidence_factors": dxf_analysis.get("confidence_factors", {}),
                "completeness_score": self._calculate_analysis_completeness(dxf_analysis)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error creating analysis summary: {e}")
            return None

    def _calculate_analysis_completeness(self, dxf_analysis: Dict[str, Any]) -> float:
        """Calculate completeness score for DXF analysis (0-100)"""
        try:
            score = 0.0
            
            spatial = dxf_analysis.get("spatial_analysis", {})
            rooms = dxf_analysis.get("rooms", {})
            dimensions = dxf_analysis.get("dimensions", [])
            
            # Room detection (40 points max)
            if spatial.get("total_rooms", 0) > 0:
                score += 20
                if len(rooms.get("labels", [])) > 0:
                    score += 10
                if spatial.get("room_types"):
                    score += 10
            
            # Area calculation (20 points max)
            if spatial.get("estimated_total_area", 0) > 0:
                score += 20
            
            # Dimensional information (20 points max)
            if len(dimensions) > 0:
                score += 10
                if len(dimensions) >= 5:
                    score += 10
            
            # Technical organization (20 points max)
            layers = dxf_analysis.get("layers", [])
            if len(layers) > 1:
                score += 10
            if len(layers) > 3:
                score += 10
            
            return min(score, 100.0)
            
        except Exception as e:
            logger.error(f"Error calculating completeness: {e}")
            return 0.0

    def get_dxf_enhanced_questions(self, session_id: str, base_questions: List[str]) -> List[str]:
        """Generate enhanced questions based on DXF analysis results"""
        try:
            session_data = self.extracted_data.get(session_id, {})
            dxf_analysis = session_data.get("dxf_analysis")
            
            if not dxf_analysis:
                return base_questions
            
            enhanced_questions = []
            spatial = dxf_analysis.get("spatial_analysis", {})
            rooms = dxf_analysis.get("rooms", {})
            
            # Generate specific questions based on DXF findings
            total_rooms = spatial.get("total_rooms", 0)
            room_types = spatial.get("room_types", {})
            estimated_area = spatial.get("estimated_total_area", 0)
            
            if total_rooms > 0:
                if room_types:
                    detected_types = list(room_types.keys())
                    if "unknown" in detected_types:
                        detected_types.remove("unknown")
                    
                    if detected_types:
                        romanian_types = [self._get_romanian_room_name(rt) for rt in detected_types]
                        enhanced_questions.append(
                            f"Am detectat {total_rooms} spatii in plan cu tipurile: {', '.join(romanian_types)}. Confirmati functiunile?"
                        )
                    else:
                        enhanced_questions.append(
                            f"Am identificat {total_rooms} spatii in plan. Care sunt functiunile acestor camere?"
                        )
                else:
                    enhanced_questions.append(
                        f"Din planul DXF am detectat {total_rooms} spatii. Cate sunt camere locuibile?"
                    )
            
            if estimated_area > 0:
                enhanced_questions.append(
                    f"Suprafata calculata din plan este {estimated_area:.1f} mp. Este suprafata utila sau construita?"
                )
            
            # Add base questions that are still relevant
            enhanced_questions.extend(base_questions[:2])
            
            return enhanced_questions[:3]  # Limit to 3 questions
            
        except Exception as e:
            logger.error(f"Error generating enhanced questions: {e}")
            return base_questions

    def calculate_enhanced_confidence_with_dxf(self, session_id: str, base_confidence: float) -> float:
        """Calculate enhanced confidence score including DXF analysis boost"""
        try:
            session_data = self.extracted_data.get(session_id, {})
            dxf_analysis = session_data.get("dxf_analysis")
            
            if not dxf_analysis:
                return base_confidence
            
            # Calculate DXF-based confidence boost
            boost = 0.0
            
            spatial = dxf_analysis.get("spatial_analysis", {})
            confidence_factors = dxf_analysis.get("confidence_factors", {})
            
            # Room detection boost
            if spatial.get("total_rooms", 0) > 0:
                boost += 15.0
            
            # Area calculation boost
            if spatial.get("estimated_total_area", 0) > 0:
                boost += 10.0
            
            # Add confidence factors from original calculation
            boost += sum(confidence_factors.values()) * 0.1
            
            # Completeness bonus
            completeness = self._calculate_analysis_completeness(dxf_analysis)
            boost += completeness * 0.2
            
            # Apply boost with cap
            enhanced_confidence = min(base_confidence + boost, 95.0)
            
            logger.info(f"Enhanced confidence: {base_confidence:.1f}% + {boost:.1f}% = {enhanced_confidence:.1f}%")
            
            return enhanced_confidence
            
        except Exception as e:
            logger.error(f"Error calculating enhanced confidence: {e}")
            return base_confidence