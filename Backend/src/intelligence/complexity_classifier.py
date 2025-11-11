# -*- coding: utf-8 -*-
# src/intelligence/complexity_classifier.py
"""
Project Complexity Classifier
Automatically detects project complexity tier based on multiple signals
"""

import logging
import re
from enum import Enum
from typing import Dict, Any, Optional, List

logger = logging.getLogger("demoplan.intelligence.complexity_classifier")


class ProjectComplexity(Enum):
    """Project complexity tiers"""
    MICRO = "micro"      # Paint, small fixes, single-trade simple tasks
    SIMPLE = "simple"    # Single room renovation, straightforward scope
    MEDIUM = "medium"    # Multi-room, moderate complexity
    COMPLEX = "complex"  # Full fitout, multiple systems, commercial


class ComplexityClassifier:
    """
    Classifies project complexity based on multiple signals:
    - User message keywords
    - Number and types of files
    - DXF complexity (rooms, systems)
    - Project data completeness
    """
    
    def __init__(self):
        """Initialize keyword patterns for classification"""
        
        # MICRO project indicators
        self.micro_keywords = [
            'vopsire', 'vopsit', 'paint', 'zugrăvi',
            'reparații mici', 'small fix', 'patch',
            'o cameră', 'single room',
            'înlocuire', 'replace', 'schimbare',
            'montaj', 'install'
        ]
        
        # SIMPLE project indicators
        self.simple_keywords = [
            'renovare cameră', 'room renovation',
            'baie', 'bathroom', 'bucătărie', 'kitchen',
            'pardoseală', 'flooring', 'gresie', 'faianță',
            'dormitor', 'bedroom', 'living',
            'finisaje', 'finishes'
        ]
        
        # MEDIUM project indicators
        self.medium_keywords = [
            'renovare apartament', 'apartment renovation',
            'mai multe camere', 'multiple rooms',
            'instalații', 'installations',
            'electric', 'sanitar', 'plumbing',
            'amenajare', 'interior design',
            '2 camere', '3 camere', '4 camere'
        ]
        
        # COMPLEX project indicators
        self.complex_keywords = [
            'fitout', 'fit-out', 'fit out',
            'birou', 'office', 'commercial',
            'complet', 'complete', 'full',
            'hvac', 'climatizare', 'ventilație',
            'multiple sisteme', 'multiple systems',
            'warehouse', 'industrial',
            'showroom', 'retail',
            'restaurant', 'hotel'
        ]
    
    def classify_project(
        self,
        user_message: str = "",
        files_uploaded: int = 0,
        dxf_data: Optional[Dict[str, Any]] = None,
        project_data: Optional[Dict[str, Any]] = None
    ) -> ProjectComplexity:
        """
        Classify project complexity based on all available signals.
        
        Scoring system: Calculate complexity score, map to tier
        - Keywords: +points based on tier
        - Files: +points based on count and types
        - DXF complexity: +points based on rooms, systems
        - Project data: +points based on scope indicators
        
        Score ranges:
        - 0-25: MICRO
        - 26-50: SIMPLE
        - 51-75: MEDIUM
        - 76+: COMPLEX
        """
        
        score = 0
        signals = []
        
        # SIGNAL 1: User Message Keywords
        message_lower = user_message.lower()
        
        if any(kw in message_lower for kw in self.complex_keywords):
            score += 40
            signals.append("Complex keywords detected")
        elif any(kw in message_lower for kw in self.medium_keywords):
            score += 25
            signals.append("Medium keywords detected")
        elif any(kw in message_lower for kw in self.simple_keywords):
            score += 10
            signals.append("Simple keywords detected")
        elif any(kw in message_lower for kw in self.micro_keywords):
            score += 5
            signals.append("Micro keywords detected")
        
        # SIGNAL 2: Number of Files
        if files_uploaded >= 5:
            score += 20
            signals.append(f"Many files ({files_uploaded})")
        elif files_uploaded >= 3:
            score += 10
            signals.append(f"Multiple files ({files_uploaded})")
        elif files_uploaded >= 1:
            score += 5
            signals.append(f"File uploaded ({files_uploaded})")
        
        # SIGNAL 3: DXF Complexity
        if dxf_data:
            dxf_score = self._assess_dxf_complexity(dxf_data)
            score += dxf_score
            if dxf_score > 0:
                signals.append(f"DXF complexity (+{dxf_score})")
        
        # SIGNAL 4: Project Data Indicators
        if project_data:
            data_score = self._assess_project_data_complexity(project_data)
            score += data_score
            if data_score > 0:
                signals.append(f"Project data (+{data_score})")
        
        # MAP SCORE TO TIER
        if score >= 76:
            complexity = ProjectComplexity.COMPLEX
        elif score >= 51:
            complexity = ProjectComplexity.MEDIUM
        elif score >= 26:
            complexity = ProjectComplexity.SIMPLE
        else:
            complexity = ProjectComplexity.MICRO
        
        logger.info(f"🎯 Complexity classification: {complexity.value} (score: {score})")
        logger.info(f"   Signals: {', '.join(signals)}")
        
        return complexity
    
    def _assess_dxf_complexity(self, dxf_data: Dict[str, Any]) -> int:
        """
        Assess DXF complexity and return score contribution.
        
        Indicators:
        - Number of rooms/spaces
        - Presence of systems (electrical, HVAC)
        - Total area
        - Multiple layers/drawings
        """
        score = 0
        
        # Check for nested dxf_analysis (common structure)
        if 'dxf_analysis' in dxf_data:
            dxf_data = dxf_data['dxf_analysis']
        
        # Number of rooms
        rooms = dxf_data.get('rooms', [])
        num_rooms = len(rooms) if isinstance(rooms, list) else dxf_data.get('room_count', 0)
        
        if num_rooms >= 15:
            score += 30  # Large complex space
        elif num_rooms >= 8:
            score += 20  # Medium multi-room
        elif num_rooms >= 4:
            score += 10  # Several rooms
        elif num_rooms >= 2:
            score += 5   # Couple rooms
        
        # Systems detected
        has_electrical = bool(dxf_data.get('electrical_layout') or 
                            dxf_data.get('power_outlets') or
                            'electric' in str(dxf_data).lower())
        
        has_hvac = bool(dxf_data.get('hvac_layout') or 
                       dxf_data.get('hvac_zones') or
                       'hvac' in str(dxf_data).lower() or
                       'climatizare' in str(dxf_data).lower())
        
        if has_electrical:
            score += 10
        if has_hvac:
            score += 10
        
        # Total area
        total_area = dxf_data.get('total_area', 0)
        if total_area > 400:
            score += 15  # Large commercial space
        elif total_area > 150:
            score += 10  # Medium/large residential
        elif total_area > 80:
            score += 5   # Standard apartment
        
        # Document type (if specification sheet detected)
        doc_type = dxf_data.get('document_type', '')
        if doc_type == 'specification_sheet':
            score += 15  # Specs indicate detailed project
        
        return score
    
    def _assess_project_data_complexity(self, project_data: Dict[str, Any]) -> int:
        """
        Assess project data for complexity indicators.
        
        Indicators:
        - Multiple room types
        - Commercial vs residential
        - Special requirements mentioned
        """
        score = 0
        
        # Room types diversity
        room_types = project_data.get('room_types', [])
        if len(room_types) >= 5:
            score += 10
        
        # Commercial indicators
        project_type = project_data.get('project_type', '').lower()
        if any(term in project_type for term in ['commercial', 'office', 'retail', 'birou']):
            score += 15
        
        # Special systems mentioned
        if project_data.get('hvac_requirements'):
            score += 10
        if project_data.get('electrical_requirements'):
            score += 10
        if project_data.get('special_requirements'):
            score += 5
        
        return score
    
    def get_complexity_description(self, complexity: ProjectComplexity) -> str:
        """Get human-readable description of complexity tier"""
        descriptions = {
            ProjectComplexity.MICRO: "Proiect micro: Lucrare simplă, singur tip de meserie (ex: vopsire, montaj)",
            ProjectComplexity.SIMPLE: "Proiect simplu: Renovare o cameră sau lucrări straightforward",
            ProjectComplexity.MEDIUM: "Proiect mediu: Mai multe camere, complexitate moderată",
            ProjectComplexity.COMPLEX: "Proiect complex: Fitout complet, multiple sisteme, comercial"
        }
        return descriptions.get(complexity, "Unknown complexity")