# -*- coding: utf-8 -*-
# src/intelligence/gap_analyzer.py
"""
Gap Analysis Engine - Compares available data vs requirements for offer generation
Generates prioritized, contextual questions based on missing information
Supports autonomous agent operation with any input scenario
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum
from typing import Tuple, List, Dict, Any, Optional

logger = logging.getLogger("demoplan.intelligence.gap_analyzer")

class GapPriority(Enum):
    """Priority levels for missing data"""
    CRITICAL = "critical"      # Blocks offer generation completely
    HIGH = "high"              # Significantly impacts accuracy (±15%)
    MEDIUM = "medium"          # Improves offer quality (±5%)
    LOW = "low"                # Nice to have, minimal impact

class DataCategory(Enum):
    """Categories of project data"""
    FINANCIAL = "financial"
    TECHNICAL = "technical"
    TIMELINE = "timeline"
    MATERIALS = "materials"
    COMPLIANCE = "compliance"
    CLIENT = "client"

@dataclass
class DataGap:
    """Individual data gap definition"""
    field_name: str
    display_name_ro: str
    priority: GapPriority
    category: DataCategory
    question_template_ro: str
    validation_rule: Optional[str] = None
    examples: List[str] = field(default_factory=list)
    context_hint: Optional[str] = None  # Additional context for the question
    
    def format_question(self, context: Optional[Dict[str, Any]] = None) -> str:
        """Format question with context and examples"""
        question = f"**{self.display_name_ro}**: {self.question_template_ro}"
        
        if self.examples:
            examples_str = ", ".join(self.examples[:3])
            question += f"\n   _Exemple: {examples_str}_"
        
        if context and self.context_hint:
            question += f"\n   💡 {self.context_hint}"
        
        return question

@dataclass
class GapAnalysisResult:
    """Complete gap analysis results"""
    overall_confidence: float
    can_generate_offer: bool
    
    # Gaps by priority
    critical_gaps: List[DataGap] = field(default_factory=list)
    high_priority_gaps: List[DataGap] = field(default_factory=list)
    medium_priority_gaps: List[DataGap] = field(default_factory=list)
    low_priority_gaps: List[DataGap] = field(default_factory=list)
    
    # Questions for user
    prioritized_questions: List[str] = field(default_factory=list)
    
    # Completeness breakdown
    data_completeness: Dict[str, float] = field(default_factory=dict)
    
    # Available data summary
    available_data_summary: List[str] = field(default_factory=list)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    def get_total_gaps(self) -> int:
        """Get total number of gaps"""
        return (len(self.critical_gaps) + len(self.high_priority_gaps) + 
                len(self.medium_priority_gaps) + len(self.low_priority_gaps))
    
    def get_blocking_issues(self) -> List[str]:
        """Get list of issues blocking offer generation"""
        return [gap.display_name_ro for gap in self.critical_gaps]

class GapAnalyzer:
    """Analyze gaps between available data and offer requirements"""
    
    def __init__(self):
        """Initialize with offer requirements definitions"""
        self.offer_requirements = self._define_offer_requirements()
        self.confidence_threshold = 0.75  # 75% needed for offer generation
        
    def _define_offer_requirements(self) -> Dict[str, DataGap]:
        """
        Define what data is needed for offer generation
        Based on Romanian construction practice and typical RFPs
        """
        return {
            # ============================================================
            # CRITICAL - Must have for any offer (blocks generation)
            # ============================================================
            'total_area': DataGap(
                field_name='total_area',
                display_name_ro='Suprafață totală',
                priority=GapPriority.CRITICAL,
                category=DataCategory.TECHNICAL,
                question_template_ro='Confirmați suprafața exactă a spațiului pentru calculul cantităților?',
                validation_rule='numeric > 0',
                examples=['350.5 mp', '450 mp'],
                context_hint='Necesară pentru estimarea materialelor și manoperei'
            ),
            
            'budget_range': DataGap(
                field_name='budget_range',
                display_name_ro='Buget estimat',
                priority=GapPriority.CRITICAL,
                category=DataCategory.FINANCIAL,
                question_template_ro='Care este bugetul estimat sau intervalul de preț acceptabil? (în EUR sau RON, fără TVA)',
                validation_rule='numeric or range',
                examples=['50.000 - 75.000 EUR', 'maxim 100.000 EUR', '200.000 - 250.000 RON'],
                context_hint='Esențial pentru adaptarea soluțiilor tehnice la posibilitățile financiare'
            ),
            
            'finish_level': DataGap(
                field_name='finish_level',
                display_name_ro='Nivel finisaje',
                priority=GapPriority.CRITICAL,
                category=DataCategory.MATERIALS,
                question_template_ro='Ce nivel de finisaje doriți pentru proiect?',
                validation_rule='enum: standard, premium, luxury',
                examples=['Standard (materiale românești)', 'Premium (branduri europene)', 'Luxury (top brands)'],
                context_hint='Influențează direct costurile și calitatea finală'
            ),
            
            'project_type': DataGap(
                field_name='project_type',
                display_name_ro='Tip proiect',
                priority=GapPriority.CRITICAL,
                category=DataCategory.TECHNICAL,
                question_template_ro='Ce tip de proiect este? (renovare, amenajare spațiu gol, modernizare)',
                validation_rule='enum',
                examples=['Renovare completă', 'Amenajare spațiu shell&core', 'Modernizare parțială'],
                context_hint='Determină volumul lucrărilor de demolare și pregătire'
            ),
            
            # ============================================================
            # HIGH PRIORITY - Significantly impacts accuracy
            # ============================================================
            'work_timeline': DataGap(
                field_name='work_timeline',
                display_name_ro='Timeline lucrări',
                priority=GapPriority.HIGH,
                category=DataCategory.TIMELINE,
                question_template_ro='Care este perioada dorită de execuție? (dată start și dată finalizare)',
                validation_rule='date_range',
                examples=['11.10.2024 - 20.01.2025', '3 luni din momentul semnării', 'cât mai repede posibil'],
                context_hint='Influențează organizarea echipelor și posibilele costuri suplimentare'
            ),
            
            'hvac_requirements': DataGap(
                field_name='hvac_requirements',
                display_name_ro='Cerințe HVAC',
                priority=GapPriority.HIGH,
                category=DataCategory.TECHNICAL,
                question_template_ro='Ce sistem de climatizare preferați? Aveți preferințe de brand?',
                validation_rule='text',
                examples=['Daikin VRV', 'Mitsubishi Electric', 'LG Multi-Split', 'fără preferință de brand'],
                context_hint='Sistemul HVAC reprezintă 15-20% din costul total'
            ),
            
            'electrical_requirements': DataGap(
                field_name='electrical_requirements',
                display_name_ro='Cerințe instalații electrice',
                priority=GapPriority.HIGH,
                category=DataCategory.TECHNICAL,
                question_template_ro='Ce cerințe speciale aveți pentru instalațiile electrice? (putere necesară, prize suplimentare, UPS)',
                validation_rule='text',
                examples=['UPS pentru server room', 'Prize duble la fiecare birou', 'Iluminat pe senzori'],
                context_hint='Important pentru dimensionarea tablourilor și circuitelor'
            ),
            
            'flooring_preferences': DataGap(
                field_name='flooring_preferences',
                display_name_ro='Preferințe pardoseli',
                priority=GapPriority.HIGH,
                category=DataCategory.MATERIALS,
                question_template_ro='Ce tip de pardoseală preferați și în ce proporție? (mochetă, LVT, parchet)',
                validation_rule='text',
                examples=['100% mochetă trafic intens', '60% LVT + 40% mochetă', 'Parchet în birouri, LVT în zone comune'],
                context_hint='Impactează atât costul cât și termenul de execuție'
            ),
            
            'ceiling_requirements': DataGap(
                field_name='ceiling_requirements',
                display_name_ro='Cerințe tavane',
                priority=GapPriority.HIGH,
                category=DataCategory.TECHNICAL,
                question_template_ro='Ce tip de tavan doriți? (fals minerale, gips-carton, acoustic)',
                validation_rule='text',
                examples=['Plăci minerale Armstrong', 'Gips-carton cu benzi LED integrate', 'Panouri acustice suspendate'],
                context_hint='Tavanul influențează acustica și estetica spațiului'
            ),
            
            'lighting_preferences': DataGap(
                field_name='lighting_preferences',
                display_name_ro='Preferințe iluminat',
                priority=GapPriority.HIGH,
                category=DataCategory.MATERIALS,
                question_template_ro='Ce tip de iluminat preferați? (LED integrat, corpuri suspendate, spoturi)',
                validation_rule='text',
                examples=['LED panel 60x60 integrat în tavan', 'Corpuri suspendate design', 'Benzi LED + spoturi'],
                context_hint='Iluminatul corect crește productivitatea cu până la 15%'
            ),
            
            # ============================================================
            # MEDIUM PRIORITY - Improves offer quality
            # ============================================================
            'partition_requirements': DataGap(
                field_name='partition_requirements',
                display_name_ro='Cerințe pereți despărțitori',
                priority=GapPriority.MEDIUM,
                category=DataCategory.TECHNICAL,
                question_template_ro='Ce tip de pereți despărțitori preferați? (gips-carton, sticlă, mobile)',
                validation_rule='text',
                examples=['Gips-carton simplu', 'Pereți sticlă pentru săli meeting', 'Combinație gips + sticlă'],
                context_hint='Pereții din sticlă sunt cu 40% mai scumpi dar dau transparență'
            ),
            
            'door_window_specs': DataGap(
                field_name='door_window_specs',
                display_name_ro='Specificații uși și ferestre',
                priority=GapPriority.MEDIUM,
                category=DataCategory.MATERIALS,
                question_template_ro='Aveți cerințe speciale pentru uși și ferestre? (tip, finisaj, hardware)',
                validation_rule='text',
                examples=['Uși lemn furnir natural', 'Uși MDF vopsite', 'Ferestre PVC cu geam termopan'],
                context_hint='Hardware-ul calitativ (Häfele, Blum) crește durabilitatea'
            ),
            
            'sanitary_requirements': DataGap(
                field_name='sanitary_requirements',
                display_name_ro='Cerințe instalații sanitare',
                priority=GapPriority.MEDIUM,
                category=DataCategory.TECHNICAL,
                question_template_ro='Ce modificări doriți la instalațiile sanitare? (obiecte sanitare noi, relocări)',
                validation_rule='text',
                examples=['Înlocuire completă obiecte sanitare', 'Doar modernizare robinete', 'Fără modificări'],
                context_hint='Lucrările sanitare necesită aprobare ISC dacă sunt relocări'
            ),
            
            'acoustic_requirements': DataGap(
                field_name='acoustic_requirements',
                display_name_ro='Cerințe izolare fonică',
                priority=GapPriority.MEDIUM,
                category=DataCategory.TECHNICAL,
                question_template_ro='Aveți cerințe speciale pentru izolarea fonică? (săli meeting, open-space)',
                validation_rule='text',
                examples=['Izolație fonică ridicată pentru săli meeting', 'Panouri acustice în open-space', 'Standard'],
                context_hint='Izolația fonică corectă este esențială pentru productivitate'
            ),
            
            'paint_finishes': DataGap(
                field_name='paint_finishes',
                display_name_ro='Vopsea și finisaje pereți',
                priority=GapPriority.MEDIUM,
                category=DataCategory.MATERIALS,
                question_template_ro='Ce vopsea/finisaje doriți pentru pereți? (lavabilă, standard, tapet)',
                validation_rule='text',
                examples=['Vopsea lavabilă Dulux', 'Tapet texturat în zonele reprezentative', 'Vopsea standard albă'],
                context_hint='Vopseaua lavabilă premium costă cu 30% mai mult dar durează dublu'
            ),
            
            # ============================================================
            # LOW PRIORITY - Nice to have
            # ============================================================
            'branding_requirements': DataGap(
                field_name='branding_requirements',
                display_name_ro='Cerințe branding',
                priority=GapPriority.LOW,
                category=DataCategory.CLIENT,
                question_template_ro='Există cerințe de branding corporate pentru spațiu? (logo, culori specifice)',
                validation_rule='text',
                examples=['Logo recepție + culorile companiei', 'Branding discret', 'Fără cerințe speciale'],
                context_hint='Elementele de branding se pot adăuga în etapa finală'
            ),
            
            'furniture_preferences': DataGap(
                field_name='furniture_preferences',
                display_name_ro='Preferințe mobilier',
                priority=GapPriority.LOW,
                category=DataCategory.MATERIALS,
                question_template_ro='Aveți preferințe pentru mobilier? (stil, brand, buget separat)',
                validation_rule='text',
                examples=['Mobilier modern minimalist', 'IKEA Business', 'Mobilier custom la comandă'],
                context_hint='Mobilierul poate fi achiziționat separat dacă preferați'
            ),
            
            'technology_integration': DataGap(
                field_name='technology_integration',
                display_name_ro='Integrare tehnologie',
                priority=GapPriority.LOW,
                category=DataCategory.TECHNICAL,
                question_template_ro='Doriți integrare cu sisteme smart? (control iluminat, acces, climatizare)',
                validation_rule='text',
                examples=['Sistem BMS pentru climatizare', 'Control iluminat prin senzori', 'Fără automatizări'],
                context_hint='Sistemele smart cresc confortul dar adaugă 5-10% la cost'
            ),
            
            'sustainability_goals': DataGap(
                field_name='sustainability_goals',
                display_name_ro='Obiective sustenabilitate',
                priority=GapPriority.LOW,
                category=DataCategory.COMPLIANCE,
                question_template_ro='Aveți obiective de sustenabilitate? (certificări, materiale eco)',
                validation_rule='text',
                examples=['Certificare LEED', 'Materiale cu certificat FSC', 'Fără cerințe speciale'],
                context_hint='Certificările green pot fi un avantaj competitiv'
            ),
        }
    
    def analyze_gaps(
        self,
        dxf_data: Optional[Dict[str, Any]] = None,
        rfp_data: Optional[Dict[str, Any]] = None,
        user_requirements: Optional[Dict[str, Any]] = None,
        conversation_context: Optional[List[Dict[str, Any]]] = None
    ) -> GapAnalysisResult:
        """
        Comprehensive gap analysis across all data sources
        
        Args:
            dxf_data: Data extracted from DXF plans
            rfp_data: Data extracted from RFP document
            user_requirements: Explicit requirements from user
            conversation_context: Historical conversation for implicit requirements
            
        Returns:
            GapAnalysisResult with complete analysis
        """
        logger.info("🔍 Starting gap analysis")
        
        # Step 1: Inventory all available data
        available_data = self._inventory_available_data(
            dxf_data, rfp_data, user_requirements, conversation_context
        )
        
        logger.info(f"📊 Data inventory: {len(available_data)} fields with data")
        
        # Step 2: Identify gaps (missing required data)
        gaps = self._identify_gaps(available_data)
        
        # Step 3: Categorize gaps by priority
        critical_gaps = [g for g in gaps if g.priority == GapPriority.CRITICAL]
        high_gaps = [g for g in gaps if g.priority == GapPriority.HIGH]
        medium_gaps = [g for g in gaps if g.priority == GapPriority.MEDIUM]
        low_gaps = [g for g in gaps if g.priority == GapPriority.LOW]
        
        logger.info(f"📋 Gaps identified: {len(critical_gaps)} critical, {len(high_gaps)} high, {len(medium_gaps)} medium, {len(low_gaps)} low")
        
        # Step 4: Calculate confidence score
        confidence = self._calculate_confidence(available_data, gaps)
        
        # Step 5: Determine if can generate offer
        can_generate = self._can_generate_offer(critical_gaps, confidence)
        
        # Step 6: Generate prioritized questions
        questions = self._generate_prioritized_questions(
            critical_gaps, high_gaps, medium_gaps, low_gaps, available_data
        )
        
        # Step 7: Calculate category completeness
        completeness = self._calculate_category_completeness(available_data)
        
        # Step 8: Generate available data summary
        summary = self._generate_data_summary(available_data, dxf_data, rfp_data)
        
        # Step 9: Generate recommendations
        recommendations = self._generate_recommendations(gaps, available_data)
        
        result = GapAnalysisResult(
            overall_confidence=confidence,
            can_generate_offer=can_generate,
            critical_gaps=critical_gaps,
            high_priority_gaps=high_gaps,
            medium_priority_gaps=medium_gaps,
            low_priority_gaps=low_gaps,
            prioritized_questions=questions,
            data_completeness=completeness,
            available_data_summary=summary,
            recommendations=recommendations
        )
        
        logger.info(f"✅ Gap analysis complete - Confidence: {confidence:.1%}, Can generate: {can_generate}")
        
        return result
    
    def _inventory_available_data(
        self,
        dxf_data: Optional[Dict],
        rfp_data: Optional[Dict],
        user_req: Optional[Dict],
        conversation: Optional[List[Dict]]
    ) -> Dict[str, Any]:
        """
        Collect all available data from all sources
        
        Returns dict with field_name -> value mapping
        """
        inventory = {}
        
        # From DXF analysis
        if dxf_data:
            dxf_analysis = dxf_data.get('dxf_analysis', {})
            
            if dxf_analysis.get('total_area'):
                inventory['total_area'] = dxf_analysis['total_area']
            
            if dxf_analysis.get('total_rooms'):
                inventory['room_count'] = dxf_analysis['total_rooms']
            
            if dxf_analysis.get('has_hvac'):
                inventory['has_existing_hvac'] = True
                hvac_inv = dxf_analysis.get('hvac_inventory', [])
                if hvac_inv:
                    inventory['hvac_count'] = len(hvac_inv)
            
            if dxf_analysis.get('has_electrical'):
                inventory['has_existing_electrical'] = True
                elec_inv = dxf_analysis.get('electrical_inventory', [])
                if elec_inv:
                    inventory['electrical_count'] = len(elec_inv)
            
            # Room types can hint at project type
            room_breakdown = dxf_analysis.get('room_breakdown', [])
            if room_breakdown:
                inventory['room_types'] = [r.get('type') for r in room_breakdown]
        
        # From RFP data
        if rfp_data:
            project_info = rfp_data.get('project_info', {})
            timeline = rfp_data.get('timeline', {})
            financial = rfp_data.get('financial', {})
            scope = rfp_data.get('scope', {})
            
            if project_info.get('client'):
                inventory['client_name'] = project_info['client']
            
            if project_info.get('location'):
                inventory['location'] = project_info['location']
            
            if timeline.get('work_start') and timeline.get('work_end'):
                inventory['work_timeline'] = {
                    'start': timeline['work_start'],
                    'end': timeline['work_end'],
                    'duration_days': timeline.get('duration_days')
                }
            
            if financial.get('guarantee_months'):
                inventory['guarantee_period'] = financial['guarantee_months']
            
            if financial.get('performance_bond'):
                inventory['performance_bond'] = financial['performance_bond']
            
            if financial.get('currency'):
                inventory['currency'] = financial['currency']
            
            # Try to infer finish level from scope
            scope_items = scope.get('items', [])
            scope_text = ' '.join(scope_items).lower()
            if 'premium' in scope_text or 'luxury' in scope_text:
                inventory['finish_level'] = 'premium'
            elif 'standard' in scope_text:
                inventory['finish_level'] = 'standard'
            
            # Check for specific requirements in scope
            if any('hvac' in item.lower() or 'climatizare' in item.lower() for item in scope_items):
                inventory['hvac_in_scope'] = True
            
            if any('electric' in item.lower() for item in scope_items):
                inventory['electrical_in_scope'] = True
        
        # From explicit user requirements
        if user_req:
            # Direct mapping of user-provided data
            for key, value in user_req.items():
                if value is not None and value != '' and value != 0:
                    inventory[key] = value
        
        # From conversation history (extract implicit requirements)
        if conversation:
            extracted = self._extract_from_conversation(conversation)
            inventory.update(extracted)
        
        return inventory
    
    def _extract_from_conversation(self, conversation: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract implicit requirements from conversation history
        Uses keyword matching and patterns
        """
        extracted = {}
        
        # Combine all user messages
        user_messages = []
        for entry in conversation:
            if entry.get('type') == 'user':
                content = entry.get('content') or entry.get('message', '')
                if content:
                    user_messages.append(content.lower())
        
        full_text = ' '.join(user_messages)
        
        # Budget patterns
        budget_patterns = [
            r'(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)\s*(eur|ron|euro|lei)',
            r'buget[:\s]*(\d+\.?\d*)\s*(eur|ron)',
            r'maxim\s*(\d+\.?\d*)\s*(eur|ron)'
        ]
        
        import re
        for pattern in budget_patterns:
            match = re.search(pattern, full_text)
            if match:
                # Extract budget info
                extracted['budget_mentioned'] = True
                break
        
        # Finish level keywords
        if 'premium' in full_text or 'lux' in full_text:
            extracted['finish_level'] = 'premium'
        elif 'luxury' in full_text:
            extracted['finish_level'] = 'luxury'
        elif 'standard' in full_text or 'normal' in full_text:
            extracted['finish_level'] = 'standard'
        
        # Project type keywords
        if 'renovare' in full_text:
            extracted['project_type'] = 'renovare'
        elif 'amenajare' in full_text:
            extracted['project_type'] = 'amenajare'
        elif 'modernizare' in full_text:
            extracted['project_type'] = 'modernizare'
        
        # Material preferences
        if 'mochet' in full_text:
            extracted['flooring_preferences'] = 'mochetă menționată'
        if 'lvt' in full_text or 'parchet' in full_text:
            extracted['flooring_preferences'] = extracted.get('flooring_preferences', '') + ' LVT/parchet menționat'
        
        # Timeline urgency
        if 'urgent' in full_text or 'repede' in full_text or 'rapid' in full_text:
            extracted['timeline_urgency'] = 'high'
        
        return extracted
    
    def _identify_gaps(self, available_data: Dict[str, Any]) -> List[DataGap]:
        """Identify which required fields are missing"""
        gaps = []
        
        for field_name, gap_definition in self.offer_requirements.items():
            value = available_data.get(field_name)
            
            # Check if data is missing or incomplete
            is_missing = False
            
            if value is None:
                is_missing = True
            elif isinstance(value, str) and len(value.strip()) == 0:
                is_missing = True
            elif isinstance(value, (int, float)) and value == 0:
                is_missing = True
            elif isinstance(value, dict) and not any(value.values()):
                is_missing = True
            elif isinstance(value, list) and len(value) == 0:
                is_missing = True
            
            if is_missing:
                gaps.append(gap_definition)
        
        return gaps
    
    def _calculate_confidence(self, available_data: Dict[str, Any], gaps: List[DataGap]) -> float:
        """
        Calculate overall confidence score
        
        Scoring model:
        - Start with 100%
        - Each CRITICAL gap: -20%
        - Each HIGH gap: -7%
        - Each MEDIUM gap: -3%
        - Each LOW gap: -1%
        """
        total_requirements = len(self.offer_requirements)
        filled_requirements = total_requirements - len(gaps)
        
        # Base confidence from filled ratio
        base_confidence = filled_requirements / total_requirements
        
        # Penalty system
        penalty = 0.0
        
        for gap in gaps:
            if gap.priority == GapPriority.CRITICAL:
                penalty += 0.20
            elif gap.priority == GapPriority.HIGH:
                penalty += 0.07
            elif gap.priority == GapPriority.MEDIUM:
                penalty += 0.03
            elif gap.priority == GapPriority.LOW:
                penalty += 0.01
        
        confidence = max(0.0, base_confidence - penalty)
        
        # Cap at 0.95 (never 100% confident without user confirmation)
        confidence = min(0.95, confidence)
        
        return confidence
    
    def _can_generate_offer(self, critical_gaps: List[DataGap], confidence: float) -> bool:
        """
        Determine if offer can be generated
        
        Rules:
        - NO critical gaps allowed
        - Confidence must be >= 75%
        """
        if len(critical_gaps) > 0:
            logger.info(f"❌ Cannot generate offer: {len(critical_gaps)} critical gaps remain")
            return False
        
        if confidence < self.confidence_threshold:
            logger.info(f"❌ Cannot generate offer: confidence {confidence:.1%} < {self.confidence_threshold:.1%}")
            return False
        
        logger.info(f"✅ Can generate offer: no critical gaps, confidence {confidence:.1%}")
        return True
    
    def _generate_prioritized_questions(
        self,
        critical: List[DataGap],
        high: List[DataGap],
        medium: List[DataGap],
        low: List[DataGap],
        available_data: Dict[str, Any]
    ) -> List[str]:
        """
        Generate maximum 5 prioritized questions
        
        Priority order:
        1. All CRITICAL gaps (these block offer)
        2. HIGH priority gaps (up to remaining slots)
        3. MEDIUM priority gaps (if slots remain)
        """
        questions = []
        max_questions = 5
        
        # Always include ALL critical gaps
        for gap in critical:
            questions.append(gap.format_question(available_data))
        
        # Add HIGH priority until we reach max
        remaining_slots = max_questions - len(questions)
        for gap in high[:remaining_slots]:
            questions.append(gap.format_question(available_data))
        
        # Add MEDIUM priority if still have slots
        remaining_slots = max_questions - len(questions)
        if remaining_slots > 0:
            for gap in medium[:remaining_slots]:
                questions.append(gap.format_question(available_data))
        
        return questions
    
    def _calculate_category_completeness(self, available_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate data completeness by category"""
        category_scores = {}
        
        for category in DataCategory:
            category_requirements = [
                req for req in self.offer_requirements.values()
                if req.category == category
            ]
            
            if len(category_requirements) == 0:
                continue
            
            filled = sum(
                1 for req in category_requirements
                if available_data.get(req.field_name) is not None
            )
            
            category_scores[category.value] = filled / len(category_requirements)
        
        return category_scores
    
    def _generate_data_summary(
        self,
        available_data: Dict[str, Any],
        dxf_data: Optional[Dict[str, Any]],
        rfp_data: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Generate human-readable summary of available data"""
        summary = []
        
        # Area information
        if available_data.get('total_area'):
            summary.append(f"Suprafață: {available_data['total_area']} mp")
        
        # Client and location
        if available_data.get('client_name'):
            summary.append(f"Client: {available_data['client_name']}")
        if available_data.get('location'):
            summary.append(f"Locație: {available_data['location']}")
        
        # Timeline
        if available_data.get('work_timeline'):
            timeline = available_data['work_timeline']
            if isinstance(timeline, dict):
                start = timeline.get('start', '')[:10]
                end = timeline.get('end', '')[:10]
                duration = timeline.get('duration_days')
                if start and end:
                    summary.append(f"Perioadă: {start} - {end} ({duration} zile)")
        
        # Budget
        if available_data.get('budget_range'):
            summary.append(f"Buget: {available_data['budget_range']}")
        elif available_data.get('budget_mentioned'):
            summary.append("Buget: discutat în conversație")
        
        # Finish level
        if available_data.get('finish_level'):
            summary.append(f"Nivel finisaje: {available_data['finish_level']}")
        
        # Project type
        if available_data.get('project_type'):
            summary.append(f"Tip proiect: {available_data['project_type']}")
        
        # Systems
        systems = []
        if available_data.get('has_existing_hvac'):
            hvac_count = available_data.get('hvac_count', 0)
            systems.append(f"HVAC existent ({hvac_count} unități)")
        if available_data.get('has_existing_electrical'):
            elec_count = available_data.get('electrical_count', 0)
            systems.append(f"Instalații electrice existente ({elec_count} componente)")
        
        if systems:
            summary.append("Sisteme: " + ", ".join(systems))
        
        # From RFP
        if rfp_data:
            financial = rfp_data.get('financial', {})
            if financial.get('guarantee_months'):
                summary.append(f"Garanție: {financial['guarantee_months']} luni")
            if financial.get('performance_bond'):
                summary.append(f"Performance Bond: {financial['performance_bond']}%")
        
        return summary
    
    def _generate_recommendations(
        self,
        gaps: List[DataGap],
        available_data: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations based on gaps"""
        recommendations = []
        
        critical_count = len([g for g in gaps if g.priority == GapPriority.CRITICAL])
        high_count = len([g for g in gaps if g.priority == GapPriority.HIGH])
        
        if critical_count > 0:
            recommendations.append(
                f"⚠️ **Urgent**: {critical_count} informații critice lipsesc - acestea blochează generarea ofertei"
            )
        
        if high_count > 3:
            recommendations.append(
                f"📋 **Important**: {high_count} informații cu prioritate ridicată lipsesc - acestea vor afecta acuratețea ofertei"
            )
        
        # Specific recommendations based on data patterns
        if not available_data.get('budget_range') and not available_data.get('budget_mentioned'):
            recommendations.append(
                "💰 **Recomandare**: Cunoașterea bugetului ne permite să propunem soluții optime pentru nevoia dumneavoastră"
            )
        
        if available_data.get('has_existing_hvac') and not available_data.get('hvac_requirements'):
            recommendations.append(
                "🌡️ **Recomandare**: Detectăm sistem HVAC existent - specificați dacă doriți menținere, upgrade sau înlocuire completă"
            )
        
        if not available_data.get('work_timeline'):
            recommendations.append(
                "📅 **Recomandare**: Timeline-ul influențează organizarea echipelor și posibilele costuri suplimentare pentru urgențe"
            )
        
        return recommendations


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def quick_gap_check(
    total_area: Optional[float] = None,
    budget: Optional[str] = None,
    finish_level: Optional[str] = None,
    project_type: Optional[str] = None
) -> Tuple[bool, float, List[str]]:
    """
    Quick gap check for minimal data scenario
    
    Returns:
        Tuple of (can_generate, confidence, missing_critical_fields)
    """
    analyzer = GapAnalyzer()
    
    user_req = {}
    if total_area:
        user_req['total_area'] = total_area
    if budget:
        user_req['budget_range'] = budget
    if finish_level:
        user_req['finish_level'] = finish_level
    if project_type:
        user_req['project_type'] = project_type
    
    result = analyzer.analyze_gaps(user_requirements=user_req)
    
    missing = [gap.field_name for gap in result.critical_gaps]
    
    return result.can_generate_offer, result.overall_confidence, missing


def analyze_conversation_progress(
    initial_confidence: float,
    current_confidence: float,
    initial_gaps: int,
    current_gaps: int
) -> Dict[str, Any]:
    """
    Analyze progress in conversation
    
    Returns metrics about how the conversation is improving data completeness
    """
    confidence_improvement = current_confidence - initial_confidence
    gaps_closed = initial_gaps - current_gaps
    
    if initial_gaps > 0:
        progress_percentage = (gaps_closed / initial_gaps) * 100
    else:
        progress_percentage = 100.0
    
    return {
        'confidence_improvement': confidence_improvement,
        'gaps_closed': gaps_closed,
        'progress_percentage': progress_percentage,
        'is_improving': confidence_improvement > 0,
        'velocity': gaps_closed  # Number of gaps closed (velocity metric)
    }