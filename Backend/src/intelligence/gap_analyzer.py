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
                question_template_ro='Ce cerințe speciale aveți pentru instalațiile electrice?',
                validation_rule='text',
                examples=['Prize suplimentare pentru IT', 'Iluminat LED dimabil', 'Backup UPS pentru servere'],
                context_hint='Modificările electrice necesită aprobare ISC'
            ),
            
            'flooring_preferences': DataGap(
                field_name='flooring_preferences',
                display_name_ro='Preferințe pardoseli',
                priority=GapPriority.HIGH,
                category=DataCategory.MATERIALS,
                question_template_ro='Ce tip de pardoseală doriți în spații?',
                validation_rule='text',
                examples=['Gresie 60x60', 'Parchet laminat', 'Vinyl LVT', 'Covor modular'],
                context_hint='Pardoseala influențează confortul și costul de întreținere'
            ),
            
            # ============================================================
            # MEDIUM PRIORITY - Improves offer quality
            # ============================================================
            'ceiling_requirements': DataGap(
                field_name='ceiling_requirements',
                display_name_ro='Cerințe tavane',
                priority=GapPriority.MEDIUM,
                category=DataCategory.TECHNICAL,
                question_template_ro='Ce tip de tavan doriți? (fals, liber aparent, combinat)',
                validation_rule='text',
                examples=['Tavan fals gips-carton', 'Tavan liber aparent industrial', 'Tavan modular Armstrong'],
                context_hint='Tavanele false permit ascunderea instalațiilor dar reduc înălțimea utilă'
            ),
            
            'partition_requirements': DataGap(
                field_name='partition_requirements',
                display_name_ro='Cerințe compartimentare',
                priority=GapPriority.MEDIUM,
                category=DataCategory.TECHNICAL,
                question_template_ro='Ce tipuri de pereți doriți pentru compartimentare? (gips-carton, sticlă, mobile)',
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
    
    async def analyze_gaps(
        self,
        dxf_data: Optional[Dict[str, Any]] = None,
        rfp_data: Optional[Dict[str, Any]] = None,
        user_requirements: Optional[Dict[str, Any]] = None,
        conversation_context: Optional[List[Dict[str, Any]]] = None
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
        can_generate = self._can_generate_offer(critical_gaps, confidence)
        
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
- "Planul arata 120mp. Care este bugetul estimat pentru aceasta suprafata?"

EXEMPLE RELE (prea generice, evita-le):
- "Ce buget aveti?" (fara context)
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
        Generate maximum 5 prioritized questions (template-based fallback)
        
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
        """Generate recommendations based on gaps and available data"""
        recommendations = []
        
        # Check for critical gaps
        critical = [g for g in gaps if g.priority == GapPriority.CRITICAL]
        if critical:
            recommendations.append(
                f"Completati urgent {len(critical)} informatii critice pentru generarea ofertei"
            )
        
        # Check for high priority gaps
        high = [g for g in gaps if g.priority == GapPriority.HIGH]
        if high:
            recommendations.append(
                f"Furnizati {len(high)} informatii cu prioritate inalta pentru acuratete sporita"
            )
        
        # Specific recommendations based on available data
        if not available_data.get('has_dimensions') and available_data.get('total_area'):
            recommendations.append(
                "Plan fara cote - recomandam vizita tehnica pentru masuratori precise"
            )
        
        if available_data.get('total_area', 0) > 200:
            recommendations.append(
                "Suprafata mare - recomandam planificare in etape pentru optimizare costuri"
            )
        
        if not available_data.get('work_timeline'):
            recommendations.append(
                "Stabiliti un timeline clar pentru planificare echipe si materiale"
            )
        
        if not available_data.get('budget_range'):
            recommendations.append(
                "Un buget estimativ ajuta la personalizarea solutiilor tehnice"
            )
        
        # Timeline recommendations
        timeline = available_data.get('work_timeline')
        if timeline and isinstance(timeline, dict):
            duration = timeline.get('duration_days', 0)
            if duration < 30:
                recommendations.append(
                    "Timeline foarte strans - posibile costuri suplimentare pentru urgenta"
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