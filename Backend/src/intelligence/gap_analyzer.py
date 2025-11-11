# -*- coding: utf-8 -*-
# src/intelligence/gap_analyzer.py
"""
Gap Analysis Engine - Compares available data vs requirements for offer generation
Generates prioritized, contextual questions based on missing information
Supports autonomous agent operation with any input scenario
PHASE 2: Enhanced with LLM-powered contextual question generation
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

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
    
    # NEW: Add expected type for validation
    expected_type: type = str  # Default to string
    
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
        SIMPLIFIED: Define only UNIVERSAL critical requirements.
        Context-specific requirements are now handled by LLM prompts.
        
        Philosophy: These are the MINIMUM fields needed for ANY offer.
        Everything else is context-dependent and asked by the agent intelligently.
        """
        return {
            # ============================================================
            # UNIVERSAL CRITICAL - Required for ALL projects
            # ============================================================
            'total_area': DataGap(
                field_name='total_area',
                display_name_ro='Suprafață totală',
                priority=GapPriority.CRITICAL,
                category=DataCategory.TECHNICAL,
                question_template_ro='Confirmați suprafața exactă a spațiului (în mp)?',
                validation_rule='numeric > 0',
                examples=['350.5 mp', '45 mp', '1200 mp'],
                context_hint='Necesară pentru orice tip de estimare',
                expected_type=float  # ✅ Indicate this should be numeric
            ),
            
            'project_scope': DataGap(
                field_name='project_scope',
                display_name_ro='Scopul lucrărilor',
                priority=GapPriority.CRITICAL,
                category=DataCategory.TECHNICAL,
                question_template_ro='Descrieți pe scurt ce lucrări doriți să realizați?',
                validation_rule='text',
                examples=['Vopsire 2 dormitoare', 'Renovare completă apartament', 'Fitout birou'],
                context_hint='Definește tipul și amploarea proiectului'
            ),
            
            'budget_range': DataGap(
                field_name='budget_range',
                display_name_ro='Buget estimat',
                priority=GapPriority.LOW,  # ✅ LOW priority - customers typically don't have budgets
                category=DataCategory.FINANCIAL,
                question_template_ro='Dacă aveți un buget stabilit, vă rugăm să îl menționați (complet opțional)',
                validation_rule='numeric or range or not_applicable',
                examples=['5.000 EUR', '50.000 - 75.000 EUR', 'nu avem buget stabilit'],
                context_hint='⚠️ NU întreba activ - acceptă doar dacă clientul oferă voluntar'
            ),
            
            # ============================================================
            # CONTEXT-DEPENDENT - Only asked if project complexity requires it
            # These serve as fallbacks if LLM doesn't extract them
            # ============================================================
            'timeline': DataGap(
                field_name='timeline',
                display_name_ro='Termen de finalizare',
                priority=GapPriority.HIGH,  # Demoted from CRITICAL
                category=DataCategory.TIMELINE,
                question_template_ro='Când doriți să fie finalizate lucrările?',
                validation_rule='date or duration',
                examples=['2 săptămâni', 'până în aprilie', 'cât mai repede'],
                context_hint='Influențează programarea și resursele necesare'
            ),
            
            'finish_level': DataGap(
                field_name='finish_level',
                display_name_ro='Nivel finisaje',
                priority=GapPriority.HIGH,  # Demoted from CRITICAL
                category=DataCategory.MATERIALS,
                question_template_ro='Ce nivel de finisaje/calitate doriți?',
                validation_rule='text',
                examples=['Standard', 'Calitate medie', 'Premium', 'Buget limitat'],
                context_hint='Determină alegerea materialelor și costurile'
            ),
        }
    
    async def analyze_gaps(
        self,
        dxf_data: Optional[Dict[str, Any]] = None,
        rfp_data: Optional[Dict[str, Any]] = None,
        user_requirements: Optional[Dict[str, Any]] = None,
        conversation_context: Optional[List[Dict[str, Any]]] = None,
        project_complexity: str = "medium"  # NEW PARAMETER
    ) -> GapAnalysisResult:
        """
        Comprehensive gap analysis across all data sources
        PHASE 2: Enhanced with LLM-powered contextual questions
        
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
        can_generate = self._can_generate_offer(critical_gaps, confidence, project_complexity)
        
        # Step 6: Generate prioritized questions
        # PHASE 2: Try LLM-powered questions first, fallback to templates
        file_context = self._extract_file_context(dxf_data, rfp_data)
        
        questions = await self._generate_contextual_questions_with_llm(
            critical_gaps, high_gaps, available_data, file_context
        )
        
        # Fallback to template-based if LLM fails
        if not questions or len(questions) < 3:
            logger.info("📝 Using template-based questions (LLM unavailable or insufficient)")
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
    
    def _safe_numeric_value(self, value: Any) -> Optional[float]:
        """
        Safely convert a value to numeric, handling strings and None.
        Returns None if conversion fails.
        """
        if value is None:
            return None
        
        if isinstance(value, (int, float)):
            return float(value)
        
        if isinstance(value, str):
            # Remove common text from numbers
            cleaned = value.strip()
            # Remove units like "mp", "m²", "EUR", "RON"
            cleaned = cleaned.replace('mp', '').replace('m²', '').replace('m2', '')
            cleaned = cleaned.replace('EUR', '').replace('RON', '').replace('€', '')
            cleaned = cleaned.strip()
            
            # Try to convert
            try:
                return float(cleaned)
            except ValueError:
                return None
        
        return None
    
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
                # ✅ Convert to numeric safely
                numeric_area = self._safe_numeric_value(dxf_analysis['total_area'])
                if numeric_area:
                    inventory['total_area'] = numeric_area
            
            if dxf_analysis.get('total_rooms'):
                inventory['total_rooms'] = dxf_analysis['total_rooms']
            
            if dxf_analysis.get('has_hvac'):
                inventory['has_existing_hvac'] = True
                inventory['hvac_count'] = len(dxf_analysis.get('hvac_inventory', []))
            
            if dxf_analysis.get('has_electrical'):
                inventory['has_existing_electrical'] = True
                inventory['electrical_count'] = len(dxf_analysis.get('electrical_inventory', []))
            
            if dxf_analysis.get('has_dimensions'):
                inventory['has_dimensions'] = True
        
        # From RFP data
        if rfp_data:
            project_info = rfp_data.get('project_info', {})
            financial = rfp_data.get('financial', {})
            timeline = rfp_data.get('timeline', {})
            
            if project_info.get('client_name'):
                inventory['client_name'] = project_info['client_name']
            
            if project_info.get('location'):
                inventory['location'] = project_info['location']
            
            if financial.get('budget_amount'):
                inventory['budget_range'] = financial['budget_amount']
            elif financial.get('budget_min') and financial.get('budget_max'):
                inventory['budget_range'] = f"{financial['budget_min']} - {financial['budget_max']}"
            
            if timeline.get('work_start_date'):
                inventory['work_timeline'] = {
                    'start': timeline['work_start_date'],
                    'end': timeline.get('work_end_date'),
                    'duration_days': timeline.get('work_duration_days')
                }
        
        # From user requirements
        if user_req:
            for key, value in user_req.items():
                if value and key not in inventory:
                    # ✅ Convert to numeric safely for total_area
                    if key == 'total_area':
                        numeric_area = self._safe_numeric_value(value)
                        if numeric_area:
                            inventory[key] = numeric_area
                    else:
                        inventory[key] = value
        
        # From conversation
        if conversation:
            # Extract implicit requirements from conversation
            # (simplified - could be enhanced with NLP)
            for message in conversation:
                content = message.get('content', '').lower()
                
                if 'buget' in content or 'pret' in content:
                    inventory['budget_mentioned'] = True
                
                if 'calendar' in content or 'termen' in content:
                    inventory['timeline_mentioned'] = True
        
        return inventory
    
    def _identify_gaps(self, available_data: Dict[str, Any]) -> List[DataGap]:
        """Identify which requirements are missing"""
        gaps = []
        
        for field_name, gap_definition in self.offer_requirements.items():
            value = available_data.get(field_name)
            
            # Check if data is missing or incomplete
            is_missing = False
            
            if value is None:
                is_missing = True
            # ✅ NEW: Treat "not_applicable" as VALID (not missing)
            elif isinstance(value, str) and value.lower() in ['not_applicable', 'n/a', 'nu există', 'nu avem', 'nu exista']:
                is_missing = False  # User explicitly said they don't have this
                logger.info(f"✅ Field '{field_name}' marked as not applicable by user")
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
    
    def _can_generate_offer(self, critical_gaps: List[DataGap], confidence: float, project_complexity: str = "medium") -> bool:
        """
        Determine if offer can be generated based on DYNAMIC thresholds.
        
        NEW: Complexity-aware thresholds
        - micro: 40% confidence, 1 critical gap allowed (can estimate)
        - simple: 55% confidence, 1 critical gap allowed
        - medium: 70% confidence, 0 critical gaps
        - complex: 85% confidence, 0 critical gaps
        
        Args:
            critical_gaps: List of critical missing data
            confidence: Overall confidence score
            project_complexity: From complexity classifier (micro/simple/medium/complex)
        """
        
        # Define thresholds by complexity
        thresholds = {
            "micro": {"confidence": 0.40, "max_critical_gaps": 1},
            "simple": {"confidence": 0.55, "max_critical_gaps": 1},
            "medium": {"confidence": 0.70, "max_critical_gaps": 0},
            "complex": {"confidence": 0.85, "max_critical_gaps": 0}
        }
        
        threshold = thresholds.get(project_complexity, thresholds["medium"])
        
        # Check critical gaps
        if len(critical_gaps) > threshold["max_critical_gaps"]:
            logger.info(f"❌ Cannot generate offer: {len(critical_gaps)} critical gaps (max allowed: {threshold['max_critical_gaps']} for {project_complexity} project)")
            return False
        
        # Check confidence threshold
        if confidence < threshold["confidence"]:
            logger.info(f"❌ Cannot generate offer: confidence {confidence:.1%} < {threshold['confidence']:.1%} (threshold for {project_complexity} project)")
            return False
        
        logger.info(f"✅ Can generate offer: {len(critical_gaps)} gaps, {confidence:.1%} confidence ({project_complexity} project)")
        return True
    
    # ============================================================================
    # PHASE 2: LLM-POWERED CONTEXTUAL QUESTION GENERATION
    # ============================================================================
    
    async def _generate_contextual_questions_with_llm(
        self,
        critical_gaps: List[DataGap],
        high_gaps: List[DataGap],
        available_data: Dict[str, Any],
        file_context: Optional[Dict[str, Any]] = None
    ) -> Optional[List[str]]:
        """
        Generate intelligent, context-aware questions using LLM
        PHASE 2: Adapts to file content and combinations
        
        Args:
            critical_gaps: Critical missing data
            high_gaps: High priority missing data
            available_data: Data we already have
            file_context: Information about uploaded files
            
        Returns:
            List of smart, contextual questions (max 5) or None if failed
        """
        try:
            from src.services.llm_service import safe_construction_call
            
            # Build context summary
            has_dxf = file_context.get('has_drawing', False) if file_context else False
            has_pdf = file_context.get('has_specification', False) if file_context else False
            has_txt = file_context.get('has_text', False) if file_context else False
            
            dxf_detail = file_context.get('drawing_detail_level', 'none') if file_context else 'none'
            pdf_detail = file_context.get('spec_detail_level', 'none') if file_context else 'none'
            
            # Build available data summary
            available_summary = []
            if available_data.get('total_area'):
                available_summary.append(f"Suprafata: {available_data['total_area']} mp")
            if available_data.get('total_rooms'):
                available_summary.append(f"Camere: {available_data['total_rooms']}")
            if available_data.get('budget_range'):
                available_summary.append(f"Buget: {available_data['budget_range']}")
            
            available_text = ", ".join(available_summary) if available_summary else "Nicio informatie de baza"
            
            # Build gaps list
            all_gaps = critical_gaps + high_gaps[:3]  # Max 5 total gaps
            gaps_text = "\n".join([
                f"- {gap.display_name_ro}: {gap.question_template_ro}"
                for gap in all_gaps
            ])
            
            # Build LLM prompt
            prompt = f"""Esti un expert in proiecte de constructii care genereaza intrebari inteligente pentru clienti.

CONTEXT FISIERE INCARCATE:
- Plan DXF: {'DA' if has_dxf else 'NU'} (nivel detaliu: {dxf_detail})
- Specificatii PDF: {'DA' if has_pdf else 'NU'} (nivel detaliu: {pdf_detail})
- Document text: {'DA' if has_txt else 'NU'}

DATE DISPONIBILE:
{available_text}

INFORMATII LIPSA (prioritare):
{gaps_text}

SARCINA:
Genereaza maxim 5 intrebari SPECIFICE si CONTEXTUALE pentru client.

REGULI OBLIGATORII:
1. Intrebarile trebuie sa fie SPECIFICE la fisierele incarcate
2. Daca exista DXF dar lipsesc specificatii -> intreaba despre finisaje/materiale
3. Daca exista PDF dar lipseste DXF -> intreaba despre dimensiuni/suprafete
4. Daca exista ambele dar lipseste buget -> intreaba despre buget cu context
5. Foloseste informatii din fisiere pentru intrebari mai bune

EXEMPLE BUNE:
- "Am vazut in plan ca aveti 5 camere. Ce nivel de finisaje doriti pentru fiecare? (standard/premium/luxury)"
- "Specificatiile mentioneaza sistem VRV. Aveti preferinta pentru brand? (Daikin, Mitsubishi, etc.)"
- "Planul arata 120mp cu 4 bai. Doriti faianta si gresie premium sau standard?"

EXEMPLE RELE (prea generice, evita-le):
- "Ce buget aveti?" (NU cere buget - clientii nu au)
- "Cand doriti sa inceapa lucrarile?" (fara legatura cu fisiere)

Raspunde DOAR cu intrebarile, cate una per linie, fara numerotare:"""

            # Call LLM
            llm_response = await safe_construction_call(
                user_input=prompt,
                system_prompt="Esti un expert in intrebari contextuale pentru constructii. Raspunzi DOAR cu intrebarile, fara explicatii.",
                temperature=0.7
            )
            
            # Parse response
            questions = [
                q.strip().lstrip('-').lstrip('•').strip()
                for q in llm_response.strip().split('\n')
                if q.strip() and len(q.strip()) > 10
            ]
            
            # Limit to 5 questions
            questions = questions[:5]
            
            if len(questions) >= 3:
                logger.info(f"✅ Generated {len(questions)} contextual questions with LLM")
                return questions
            else:
                logger.warning(f"⚠️ LLM generated only {len(questions)} questions, falling back")
                return None
                
        except Exception as e:
            logger.error(f"❌ LLM question generation failed: {e}")
            return None
    
    def _extract_file_context(
        self,
        dxf_data: Optional[Dict],
        rfp_data: Optional[Dict]
    ) -> Dict[str, Any]:
        """
        Extract file context for LLM question generation
        PHASE 2: Provides file metadata to LLM
        """
        context = {
            'has_drawing': False,
            'has_specification': False,
            'has_text': False,
            'drawing_detail_level': 'none',
            'spec_detail_level': 'none'
        }
        
        # Analyze DXF
        if dxf_data:
            dxf_analysis = dxf_data.get('dxf_analysis', {})
            context['has_drawing'] = True
            
            has_rooms = dxf_analysis.get('total_rooms', 0) > 0
            has_mep = dxf_analysis.get('has_hvac') or dxf_analysis.get('has_electrical')
            has_dimensions = dxf_analysis.get('has_dimensions', False)
            
            if has_rooms and has_mep and has_dimensions:
                context['drawing_detail_level'] = 'complete'
            elif has_rooms and has_mep:
                context['drawing_detail_level'] = 'good'
            elif has_rooms:
                context['drawing_detail_level'] = 'basic'
        
        # Analyze RFP/PDF
        if rfp_data:
            context['has_specification'] = True
            
            has_scope = bool(rfp_data.get('scope'))
            has_materials = bool(rfp_data.get('materials'))
            
            if has_scope and has_materials:
                context['spec_detail_level'] = 'complete'
            elif has_scope or has_materials:
                context['spec_detail_level'] = 'partial'
        
        return context
    
    def _generate_prioritized_questions(
        self,
        critical: List[DataGap],
        high: List[DataGap],
        medium: List[DataGap],
        low: List[DataGap],
        available_data: Dict[str, Any]
    ) -> List[str]:
        """
        SIMPLIFIED: Generate questions for critical gaps only.
        Context-specific questions now come from LLM prompts.
        
        Maximum 3 questions from gap analyzer.
        Agent will ask additional context-specific questions via LLM.
        """
        questions = []
        
        # Only ask about CRITICAL gaps (universal requirements)
        for gap in critical[:3]:  # Max 3 critical questions
            questions.append(gap.question_template_ro)
        
        logger.info(f"📝 Gap analyzer generated {len(questions)} universal questions")
        logger.info(f"💡 Agent will generate additional context-specific questions via LLM")
        
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
            summary.append(f"Suprafata: {available_data['total_area']} mp")
        
        # Client and location
        if available_data.get('client_name'):
            summary.append(f"Client: {available_data['client_name']}")
        if available_data.get('location'):
            summary.append(f"Locatie: {available_data['location']}")
        
        # Timeline
        if available_data.get('work_timeline'):
            timeline = available_data['work_timeline']
            if isinstance(timeline, dict):
                start = timeline.get('start', '')[:10]
                end = timeline.get('end', '')[:10]
                duration = timeline.get('duration_days')
                if start and end:
                    summary.append(f"Perioada: {start} - {end} ({duration} zile)")
        
        # Budget
        if available_data.get('budget_range'):
            summary.append(f"Buget: {available_data['budget_range']}")
        elif available_data.get('budget_mentioned'):
            summary.append("Buget: discutat in conversatie")
        
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
            systems.append(f"HVAC existent ({hvac_count} unitati)")
        if available_data.get('has_existing_electrical'):
            elec_count = available_data.get('electrical_count', 0)
            systems.append(f"Instalatii electrice existente ({elec_count} componente)")
        
        if systems:
            summary.append("Sisteme: " + ", ".join(systems))
        
        # From RFP
        if rfp_data:
            financial = rfp_data.get('financial', {})
            if financial.get('payment_terms'):
                summary.append(f"Termeni plata: {financial['payment_terms']}")
        
        return summary
    
    def _generate_recommendations(self, gaps: List[DataGap], available_data: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Get total area safely (handle both string and numeric)
        total_area_raw = available_data.get('total_area', 0)
        try:
            total_area = float(total_area_raw) if total_area_raw else 0
        except (ValueError, TypeError):
            total_area = 0
        
        # Get budget safely
        budget_raw = available_data.get('budget_range', '')
        budget_str = str(budget_raw) if budget_raw else ''
        
        # Recommendation 1: For large projects
        if total_area > 200:
            recommendations.append(
                "Proiect mare (>200mp) - Recomandăm planificare detaliată pe faze"
            )
        
        # ✅ REMOVED: Budget recommendation - customers typically don't have budgets
        
        # Recommendation 2: Timeline not specified
        if not available_data.get('timeline'):
            recommendations.append(
                "Specificarea termenului ajută la alocarea optimă a resurselor"
            )
        
        # Recommendation 4: Multiple critical gaps
        critical_gap_count = len([g for g in gaps if g.priority == GapPriority.CRITICAL])
        if critical_gap_count >= 2:
            recommendations.append(
                f"Completarea celor {critical_gap_count} informații critice va permite generarea ofertei"
            )
        
        # Recommendation 5: For complex projects
        finish_level = available_data.get('finish_level', '')
        if finish_level and 'premium' in str(finish_level).lower():
            recommendations.append(
                "Nivel premium detectat - Recomandăm specificații detaliate materiale"
            )
        
        # Recommendation 6: Site conditions matter for execution
        if not available_data.get('site_conditions') and total_area > 100:
            recommendations.append(
                "Pentru proiecte >100mp, condițiile șantierului influențează semnificativ execuția"
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
    Quick gap check for minimal data scenario (synchronous wrapper)
    
    Returns:
        Tuple of (can_generate, confidence, missing_critical_fields)
    """
    import asyncio
    
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
    
    # Run async function in sync context
    result = asyncio.run(analyzer.analyze_gaps(user_requirements=user_req))
    
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