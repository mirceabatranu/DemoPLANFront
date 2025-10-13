# -*- coding: utf-8 -*-
# src/intelligence/cross_reference_engine.py
"""
Cross-Reference Engine - Validates consistency between data sources
Detects conflicts between DXF, RFP, and user inputs
Provides intelligent recommendations for conflict resolution
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger("demoplan.intelligence.cross_reference")

class ConflictSeverity(Enum):
    """Severity levels for data conflicts"""
    ERROR = "error"        # Must be resolved before offer generation
    WARNING = "warning"    # Should be clarified, may impact accuracy
    INFO = "info"          # FYI only, no action needed

class ConflictType(Enum):
    """Types of conflicts that can be detected"""
    AREA_MISMATCH = "area_mismatch"
    TIMELINE_CONFLICT = "timeline_conflict"
    SCOPE_MISMATCH = "scope_mismatch"
    BUDGET_INCONSISTENCY = "budget_inconsistency"
    SYSTEM_CONFLICT = "system_conflict"
    PHYSICAL_IMPOSSIBILITY = "physical_impossibility"
    REGULATORY_VIOLATION = "regulatory_violation"

@dataclass
class DataConflict:
    field_name: str
    severity: ConflictSeverity
    source1: str
    source2: str        # ✅ Required fields first
    value1: Any
    value2: Any
    description_ro: str = ""      # ✅ Defaults at end
    recommendation_ro: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'type': self.conflict_type.value,
            'field': self.field_name,
            'severity': self.severity.value,
            'source1': self.source1,
            'value1': str(self.value1),
            'source2': self.source2,
            'value2': str(self.value2),
            'description': self.description_ro,
            'recommendation': self.recommendation_ro,
            'impact': self.impact_assessment,
            'resolution_options': self.resolution_options
        }

@dataclass
class CrossReferenceResult:
    """Complete cross-reference validation results"""
    is_consistent: bool
    consistency_score: float
    
    conflicts: List[DataConflict] = field(default_factory=list)
    validated_data: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    # Conflict breakdown
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    
    def get_blocking_conflicts(self) -> List[DataConflict]:
        """Get conflicts that block offer generation"""
        return [c for c in self.conflicts if c.severity == ConflictSeverity.ERROR]
    
    def get_critical_warnings(self) -> List[DataConflict]:
        """Get warnings that should be addressed"""
        return [c for c in self.conflicts if c.severity == ConflictSeverity.WARNING]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'is_consistent': self.is_consistent,
            'consistency_score': self.consistency_score,
            'conflicts': [c.to_dict() for c in self.conflicts],
            'validated_data': self.validated_data,
            'recommendations': self.recommendations,
            'error_count': self.error_count,
            'warning_count': self.warning_count,
            'info_count': self.info_count
        }

class CrossReferenceEngine:
    """Validate data consistency across all sources"""
    
    def __init__(self):
        """Initialize with validation rules"""
        self.area_tolerance = 0.05  # 5% tolerance for area differences
        self.timeline_buffer_days = 3  # 3 days buffer for timeline conflicts
        
    def validate_consistency(
        self,
        dxf_data: Optional[Dict[str, Any]] = None,
        rfp_data: Optional[Dict[str, Any]] = None,
        user_inputs: Optional[Dict[str, Any]] = None,
        conversation_context: Optional[List[Dict[str, Any]]] = None
    ) -> CrossReferenceResult:
        """
        Main validation method - cross-references all data sources
        
        Args:
            dxf_data: Data extracted from DXF plans
            rfp_data: Data extracted from RFP document
            user_inputs: Explicit user requirements
            conversation_context: Historical conversation
            
        Returns:
            CrossReferenceResult with all conflicts and validated data
        """
        logger.info("🔍 Starting cross-reference validation")
        
        conflicts = []
        validated = {}
        
        # Area consistency checks
        area_conflicts = self._check_area_consistency(dxf_data, rfp_data, user_inputs)
        conflicts.extend(area_conflicts)
        
        # Timeline consistency checks
        timeline_conflicts = self._check_timeline_consistency(rfp_data, user_inputs)
        conflicts.extend(timeline_conflicts)
        
        # Scope consistency checks
        scope_conflicts = self._check_scope_consistency(dxf_data, rfp_data, user_inputs)
        conflicts.extend(scope_conflicts)
        
        # System consistency checks (HVAC, electrical)
        system_conflicts = self._check_system_consistency(dxf_data, rfp_data, user_inputs)
        conflicts.extend(system_conflicts)
        
        # Budget consistency checks
        budget_conflicts = self._check_budget_consistency(rfp_data, user_inputs)
        conflicts.extend(budget_conflicts)
        
        # Physical feasibility checks
        feasibility_conflicts = self._check_physical_feasibility(dxf_data, user_inputs)
        conflicts.extend(feasibility_conflicts)
        
        # Regulatory compliance checks
        regulatory_conflicts = self._check_regulatory_compliance(dxf_data, rfp_data, user_inputs)
        conflicts.extend(regulatory_conflicts)
        
        # Extract validated data (data that has no conflicts)
        validated = self._extract_validated_data(
            dxf_data, rfp_data, user_inputs, conflicts
        )
        
        # Calculate consistency score
        consistency_score = self._calculate_consistency_score(conflicts, validated)
        
        # Determine overall consistency
        is_consistent = len([c for c in conflicts if c.severity == ConflictSeverity.ERROR]) == 0
        
        # Generate recommendations
        recommendations = self._generate_recommendations(conflicts)
        
        # Count conflicts by severity
        error_count = len([c for c in conflicts if c.severity == ConflictSeverity.ERROR])
        warning_count = len([c for c in conflicts if c.severity == ConflictSeverity.WARNING])
        info_count = len([c for c in conflicts if c.severity == ConflictSeverity.INFO])
        
        result = CrossReferenceResult(
            is_consistent=is_consistent,
            consistency_score=consistency_score,
            conflicts=conflicts,
            validated_data=validated,
            recommendations=recommendations,
            error_count=error_count,
            warning_count=warning_count,
            info_count=info_count
        )
        
        logger.info(f"✅ Cross-reference complete: {len(conflicts)} conflicts found ({error_count} errors, {warning_count} warnings)")
        logger.info(f"📊 Consistency score: {consistency_score:.1%}")
        
        return result
    
    def _check_area_consistency(
        self,
        dxf_data: Optional[Dict],
        rfp_data: Optional[Dict],
        user_inputs: Optional[Dict]
    ) -> List[DataConflict]:
        """Check if area measurements are consistent across sources"""
        conflicts = []
        
        # Extract areas from different sources
        dxf_area = None
        if dxf_data and 'dxf_analysis' in dxf_data:
            dxf_area = dxf_data['dxf_analysis'].get('total_area')
        
        rfp_area = None
        if rfp_data and 'project_info' in rfp_data:
            # RFP might mention area in project description
            pass  # Would need text parsing for this
        
        user_area = user_inputs.get('total_area') if user_inputs else None
        
        # Compare DXF vs User input
        if dxf_area and user_area:
            difference = abs(dxf_area - user_area)
            tolerance = dxf_area * self.area_tolerance
            
            if difference > tolerance:
                percentage_diff = (difference / dxf_area) * 100
                
                severity = ConflictSeverity.ERROR if percentage_diff > 10 else ConflictSeverity.WARNING
                
                conflicts.append(DataConflict(
                    conflict_type=ConflictType.AREA_MISMATCH,
                    field_name='total_area',
                    severity=severity,
                    source1='DXF Plan',
                    value1=f"{dxf_area:.2f} mp",
                    source1_detail='Calculat automat din geometria planului',
                    source2='Cerință utilizator',
                    value2=f"{user_area:.2f} mp",
                    source2_detail='Declarat de utilizator',
                    description_ro=f"Suprafața din plan ({dxf_area:.2f} mp) diferă cu {percentage_diff:.1f}% față de cea declarată ({user_area:.2f} mp)",
                    recommendation_ro=f"Verificați măsurătorile. Diferența de {difference:.2f} mp poate indica: (1) eroare în plan, (2) zone excluse din calcul, sau (3) eroare de declarare.",
                    impact_assessment=f"Impact pe estimare: ±{percentage_diff:.0f}% cost",
                    resolution_options=[
                        f"Acceptați suprafața din plan: {dxf_area:.2f} mp",
                        f"Confirmați suprafața declarată: {user_area:.2f} mp",
                        "Solicitați verificare topografică"
                    ]
                ))
                
                logger.warning(f"⚠️ Area mismatch: DXF={dxf_area:.2f}, User={user_area:.2f}, Diff={percentage_diff:.1f}%")
        
        return conflicts
    
    def _check_timeline_consistency(
        self,
        rfp_data: Optional[Dict],
        user_inputs: Optional[Dict]
    ) -> List[DataConflict]:
        """Check timeline consistency between RFP and user requirements"""
        conflicts = []
        
        if not rfp_data or not rfp_data.get('timeline'):
            return conflicts
        
        rfp_timeline = rfp_data['timeline']
        rfp_start = rfp_timeline.get('work_start')
        rfp_end = rfp_timeline.get('work_end')
        rfp_duration = rfp_timeline.get('duration_days')
        
        # Check if user wants different timeline
        user_timeline = user_inputs.get('work_timeline') if user_inputs else None
        
        if user_timeline and isinstance(user_timeline, dict):
            user_start = user_timeline.get('start')
            user_end = user_timeline.get('end')
            
            # Parse dates if strings
            if isinstance(user_start, str):
                try:
                    user_start = datetime.fromisoformat(user_start.replace('Z', '+00:00'))
                except:
                    user_start = None
            
            if isinstance(user_end, str):
                try:
                    user_end = datetime.fromisoformat(user_end.replace('Z', '+00:00'))
                except:
                    user_end = None
            
            # Parse RFP dates
            if isinstance(rfp_start, str):
                try:
                    rfp_start = datetime.fromisoformat(rfp_start.replace('Z', '+00:00'))
                except:
                    rfp_start = None
            
            if isinstance(rfp_end, str):
                try:
                    rfp_end = datetime.fromisoformat(rfp_end.replace('Z', '+00:00'))
                except:
                    rfp_end = None
            
            # Compare timelines
            if rfp_start and user_start and rfp_start != user_start:
                days_diff = abs((user_start - rfp_start).days)
                
                if days_diff > self.timeline_buffer_days:
                    conflicts.append(DataConflict(
                        conflict_type=ConflictType.TIMELINE_CONFLICT,
                        field_name='work_start_date',
                        severity=ConflictSeverity.WARNING,
                        source1='RFP',
                        value1=rfp_start.strftime('%d.%m.%Y'),
                        source1_detail='Dată specificată în RFP',
                        source2='Cerință utilizator',
                        value2=user_start.strftime('%d.%m.%Y'),
                        source2_detail='Dată dorită de utilizator',
                        description_ro=f"Data start din RFP ({rfp_start.strftime('%d.%m.%Y')}) diferă cu {days_diff} zile față de data dorită ({user_start.strftime('%d.%m.%Y')})",
                        recommendation_ro="Clarificați data corectă de start. Modificarea timeline-ului poate necesita aprobare de la client/landlord.",
                        impact_assessment="Poate afecta disponibilitatea echipelor și cost" if days_diff > 7 else "Impact minor"
                    ))
            
            # Check duration feasibility
            if rfp_duration and user_end and rfp_start:
                user_duration = (user_end - rfp_start).days if isinstance(user_end, datetime) else None
                
                if user_duration and abs(user_duration - rfp_duration) > 7:
                    conflicts.append(DataConflict(
                        conflict_type=ConflictType.TIMELINE_CONFLICT,
                        field_name='work_duration',
                        severity=ConflictSeverity.WARNING,
                        source1='RFP',
                        value1=f"{rfp_duration} zile",
                        source2='Cerință utilizator',
                        value2=f"{user_duration} zile",
                        description_ro=f"Durata din RFP ({rfp_duration} zile) diferă de durata dorită ({user_duration} zile)",
                        recommendation_ro="Durata mai scurtă poate necesita echipe suplimentare (cost +10-20%). Durata mai lungă poate afecta deadline-ul proiectului.",
                        impact_assessment="Impact semnificativ pe cost și organizare"
                    ))
        
        # Check if timeline is realistic
        if rfp_duration:
            # Rough estimate: need ~1 day per 10mp for standard fit-out
            # This is a heuristic, actual depends on complexity
            pass  # Would need area data for this check
        
        return conflicts
    
    def _check_scope_consistency(
        self,
        dxf_data: Optional[Dict],
        rfp_data: Optional[Dict],
        user_inputs: Optional[Dict]
    ) -> List[DataConflict]:
        """Check if scope matches between sources"""
        conflicts = []
        
        if not dxf_data or not rfp_data:
            return conflicts
        
        dxf_analysis = dxf_data.get('dxf_analysis', {})
        scope_data = rfp_data.get('scope', {})
        scope_items = scope_data.get('items', [])
        scope_text = ' '.join(scope_items).lower()
        
        # Check HVAC consistency
        has_hvac_in_plan = dxf_analysis.get('has_hvac', False)
        hvac_in_scope = any(
            keyword in scope_text 
            for keyword in ['hvac', 'climatizare', 'ventilatie', 'air conditioning']
        )
        
        if hvac_in_scope and not has_hvac_in_plan:
            conflicts.append(DataConflict(
                conflict_type=ConflictType.SCOPE_MISMATCH,
                field_name='hvac_system',
                severity=ConflictSeverity.WARNING,
                source1='RFP Scope',
                value1='HVAC menționat în domeniul lucrărilor',
                source2='DXF Plan',
                value2='Sistem HVAC NU detectat în plan',
                description_ro='RFP-ul menționează lucrări HVAC, dar planul DXF nu conține sistem HVAC vizibil',
                recommendation_ro='Clarificați dacă: (1) sistemul HVAC există dar nu e în plan, (2) trebuie proiectat de la zero, sau (3) se livrează separat de contractor specializat',
                impact_assessment='Impact major - HVAC reprezintă 15-25% din costul total',
                resolution_options=[
                    'Sistem HVAC existent (nu apare în plan)',
                    'Sistem HVAC nou - proiectare necesară',
                    'HVAC contractor separat (exclus din ofertă)'
                ]
            ))
        
        # Check electrical consistency
        has_electrical_in_plan = dxf_analysis.get('has_electrical', False)
        electrical_in_scope = any(
            keyword in scope_text 
            for keyword in ['electrical', 'electric', 'instalații electrice', 'prize', 'iluminat']
        )
        
        if electrical_in_scope and not has_electrical_in_plan:
            conflicts.append(DataConflict(
                conflict_type=ConflictType.SCOPE_MISMATCH,
                field_name='electrical_system',
                severity=ConflictSeverity.INFO,
                source1='RFP Scope',
                value1='Instalații electrice în scope',
                source2='DXF Plan',
                value2='Instalații electrice NU detectate complet',
                description_ro='RFP menționează instalații electrice, dar planul nu le arată detaliat',
                recommendation_ro='Normal pentru planuri arhitecturale. Se va lucra după proiect de specialitate (instalații)',
                impact_assessment='Impact moderat - necesar proiect electrical de specialitate'
            ))
        
        # Check for demolition scope
        demolition_in_scope = any(
            keyword in scope_text 
            for keyword in ['demolish', 'demolish', 'demolare', 'evacuate']
        )
        
        if demolition_in_scope:
            # Check if plan shows existing conditions
            conflicts.append(DataConflict(
                conflict_type=ConflictType.SCOPE_MISMATCH,
                field_name='demolition_scope',
                severity=ConflictSeverity.INFO,
                source1='RFP Scope',
                value1='Lucrări de demolare menționate',
                source2='DXF Plan',
                value2='Plan arată starea finală dorită',
                description_ro='RFP-ul include demolări, dar planul arată doar starea finală',
                recommendation_ro='Necesară vizită la fața locului pentru evaluarea volumului de demolare. Costul demolării: 15-30 EUR/mp în funcție de complexitate.',
                impact_assessment='Impact pe cost: +5-10% pentru demolare și evacuare'
            ))
        
        return conflicts
    
    def _check_system_consistency(
        self,
        dxf_data: Optional[Dict],
        rfp_data: Optional[Dict],
        user_inputs: Optional[Dict]
    ) -> List[DataConflict]:
        """Check system requirements consistency"""
        conflicts = []
        
        if not dxf_data:
            return conflicts
        
        dxf_analysis = dxf_data.get('dxf_analysis', {})
        
        # Check if user requirements match plan reality
        if user_inputs:
            # HVAC consistency
            user_hvac_req = user_inputs.get('hvac_requirements')
            has_hvac_in_plan = dxf_analysis.get('has_hvac', False)
            
            if user_hvac_req and not has_hvac_in_plan:
                conflicts.append(DataConflict(
                    conflict_type=ConflictType.SYSTEM_CONFLICT,
                    field_name='hvac_requirements',
                    severity=ConflictSeverity.WARNING,
                    source1='Cerință utilizator',
                    value1=str(user_hvac_req),
                    source2='DXF Plan',
                    value2='Sistem HVAC NU detectat',
                    description_ro=f'Ați specificat cerințe HVAC ({user_hvac_req}), dar planul nu arată sistem existent',
                    recommendation_ro='Vom include sistem HVAC nou în ofertă. Verificați dacă există sistem existent care nu apare în plan.',
                    impact_assessment='Cost HVAC nou: 80-120 EUR/mp în funcție de sistem'
                ))
        
        return conflicts
    
    def _check_budget_consistency(
        self,
        rfp_data: Optional[Dict],
        user_inputs: Optional[Dict]
    ) -> List[DataConflict]:
        """Check budget consistency and feasibility"""
        conflicts = []
        
        # Extract budgets
        user_budget = user_inputs.get('budget_range') if user_inputs else None
        
        # Try to extract budget from RFP (if mentioned)
        rfp_budget = None
        if rfp_data and rfp_data.get('financial'):
            # RFPs usually don't specify budget, but sometimes hint at it
            pass
        
        # Check budget feasibility based on area and finish level
        if user_budget and user_inputs:
            area = user_inputs.get('total_area')
            finish_level = user_inputs.get('finish_level')
            
            if area and finish_level:
                # Rough cost per sqm estimates for Romania
                cost_per_sqm = {
                    'standard': (250, 350),  # EUR/mp
                    'premium': (400, 550),
                    'luxury': (600, 900)
                }
                
                if finish_level.lower() in cost_per_sqm:
                    min_cost, max_cost = cost_per_sqm[finish_level.lower()]
                    estimated_min = area * min_cost
                    estimated_max = area * max_cost
                    
                    # Parse user budget
                    import re
                    budget_numbers = re.findall(r'\d+', str(user_budget).replace('.', '').replace(',', ''))
                    
                    if budget_numbers:
                        user_budget_value = float(budget_numbers[0])
                        
                        if user_budget_value < estimated_min * 0.8:
                            conflicts.append(DataConflict(
                                conflict_type=ConflictType.BUDGET_INCONSISTENCY,
                                field_name='budget_range',
                                severity=ConflictSeverity.WARNING,
                                source1='Buget declarat',
                                value1=f"{user_budget_value:,.0f} EUR",
                                source2='Estimare bazată pe suprafață și finish level',
                                value2=f"{estimated_min:,.0f} - {estimated_max:,.0f} EUR",
                                description_ro=f"Bugetul declarat ({user_budget_value:,.0f} EUR) este sub estimarea pentru {area:.0f} mp cu finisaje {finish_level} ({estimated_min:,.0f} - {estimated_max:,.0f} EUR)",
                                recommendation_ro=f"Opțiuni: (1) Revizuiți bugetul, (2) Reduceți suprafața, (3) Coborâți nivelul finisajelor, sau (4) Prioritizați lucrările esențiale",
                                impact_assessment="Risc de neîndeplinire a așteptărilor",
                                resolution_options=[
                                    f"Creșteți bugetul la minim {estimated_min:,.0f} EUR",
                                    "Treceți la finisaje standard",
                                    "Execuție pe faze"
                                ]
                            ))
                        elif user_budget_value > estimated_max * 1.3:
                            conflicts.append(DataConflict(
                                conflict_type=ConflictType.BUDGET_INCONSISTENCY,
                                field_name='budget_range',
                                severity=ConflictSeverity.INFO,
                                source1='Buget declarat',
                                value1=f"{user_budget_value:,.0f} EUR",
                                source2='Estimare bazată pe suprafață și finish level',
                                value2=f"{estimated_min:,.0f} - {estimated_max:,.0f} EUR",
                                description_ro=f"Bugetul declarat ({user_budget_value:,.0f} EUR) depășește estimarea pentru {area:.0f} mp cu finisaje {finish_level}",
                                recommendation_ro="Excelent! Bugetul permite: materiale premium, design custom, sau extinderea scopului lucrărilor",
                                impact_assessment="Oportunitate pentru upgrade",
                                resolution_options=[
                                    "Mențineți bugetul - calitate maximă",
                                    "Upgrade la finisaje luxury",
                                    "Includeți mobilier custom"
                                ]
                            ))
        
        return conflicts
    
    def _check_physical_feasibility(
        self,
        dxf_data: Optional[Dict],
        user_inputs: Optional[Dict]
    ) -> List[DataConflict]:
        """Check if user requirements are physically feasible"""
        conflicts = []
        
        if not dxf_data or not user_inputs:
            return conflicts
        
        dxf_analysis = dxf_data.get('dxf_analysis', {})
        
        # Check for physically impossible requirements
        # Example: User wants to add wall at specific coordinates
        user_wall_request = user_inputs.get('wall_location')
        
        if user_wall_request:
            # Check if location is valid in plan
            # This would require spatial analysis of DXF
            # For now, we'll add a validation placeholder
            pass
        
        return conflicts
    
    def _check_regulatory_compliance(
        self,
        dxf_data: Optional[Dict],
        rfp_data: Optional[Dict],
        user_inputs: Optional[Dict]
    ) -> List[DataConflict]:
        """Check regulatory compliance requirements"""
        conflicts = []
        
        # Check fire safety requirements
        if rfp_data:
            scope_items = rfp_data.get('scope', {}).get('items', [])
            scope_text = ' '.join(scope_items).lower()
            
            if 'fire' in scope_text or 'psi' in scope_text or 'incendiu' in scope_text:
                deliverables = rfp_data.get('scope', {}).get('deliverables', [])
                has_fire_permit = any(
                    'fire permit' in d.lower() or 'autorizație psi' in d.lower()
                    for d in deliverables
                )
                
                if has_fire_permit:
                    conflicts.append(DataConflict(
                        conflict_type=ConflictType.REGULATORY_VIOLATION,
                        field_name='fire_permit',
                        severity=ConflictSeverity.INFO,
                        source1='RFP Requirements',
                        value1='Fire Permit obligatoriu',
                        source2='Romanian Law',
                        value2='Necesită aprobare ISU',
                        description_ro='RFP-ul cere obținerea Autorizației PSI',
                        recommendation_ro='Vom include în ofertă: (1) Proiect PSI de specialitate, (2) Taxe ISU, (3) Timp pentru aprobare (15-30 zile)',
                        impact_assessment='Timp suplimentar: 15-30 zile, Cost: 1.500-3.000 EUR'
                    ))
        
        return conflicts
    
    def _extract_validated_data(
        self,
        dxf_data: Optional[Dict],
        rfp_data: Optional[Dict],
        user_inputs: Optional[Dict],
        conflicts: List[DataConflict]
    ) -> Dict[str, Any]:
        """Extract data that has been validated (no conflicts)"""
        validated = {}
        
        # Fields that have conflicts
        conflicted_fields = {c.field_name for c in conflicts}
        
        # Add DXF data that has no conflicts
        if dxf_data and 'dxf_analysis' in dxf_data:
            dxf_analysis = dxf_data['dxf_analysis']
            
            if 'total_area' not in conflicted_fields and dxf_analysis.get('total_area'):
                validated['total_area'] = dxf_analysis['total_area']
                validated['total_area_source'] = 'DXF'
            
            if dxf_analysis.get('has_hvac'):
                validated['has_existing_hvac'] = True
                validated['hvac_count'] = len(dxf_analysis.get('hvac_inventory', []))
            
            if dxf_analysis.get('has_electrical'):
                validated['has_existing_electrical'] = True
        
        # Add RFP data that has no conflicts
        if rfp_data:
            timeline = rfp_data.get('timeline', {})
            if 'work_timeline' not in conflicted_fields:
                if timeline.get('work_start'):
                    validated['work_start_date'] = timeline['work_start']
                    validated['work_start_date_source'] = 'RFP'
                if timeline.get('work_end'):
                    validated['work_end_date'] = timeline['work_end']
                    validated['work_end_date_source'] = 'RFP'
            
            financial = rfp_data.get('financial', {})
            if financial.get('guarantee_months'):
                validated['guarantee_months'] = financial['guarantee_months']
                validated['guarantee_source'] = 'RFP'
        
        # Add user inputs that have no conflicts
        if user_inputs:
            for key, value in user_inputs.items():
                if key not in conflicted_fields and value is not None:
                    validated[key] = value
                    validated[f'{key}_source'] = 'User'
        
        return validated
    
    def _calculate_consistency_score(
        self,
        conflicts: List[DataConflict],
        validated_data: Dict[str, Any]
    ) -> float:
        """
        Calculate overall consistency score
        
        Scoring:
        - Start at 100%
        - Each ERROR: -15%
        - Each WARNING: -5%
        - Each INFO: -1%
        """
        if not conflicts:
            return 1.0
        
        score = 1.0
        
        for conflict in conflicts:
            if conflict.severity == ConflictSeverity.ERROR:
                score -= 0.15
            elif conflict.severity == ConflictSeverity.WARNING:
                score -= 0.05
            elif conflict.severity == ConflictSeverity.INFO:
                score -= 0.01
        
        # Don't go below 0
        score = max(0.0, score)
        
        # Boost score if we have validated data
        if len(validated_data) > 10:
            score = min(1.0, score + 0.05)
        
        return score
    
    def _generate_recommendations(
        self,
        conflicts: List[DataConflict]
    ) -> List[str]:
        """Generate actionable recommendations based on conflicts"""
        recommendations = []
        
        # Group conflicts by type
        error_conflicts = [c for c in conflicts if c.severity == ConflictSeverity.ERROR]
        warning_conflicts = [c for c in conflicts if c.severity == ConflictSeverity.WARNING]
        
        # Recommendations for errors
        if error_conflicts:
            recommendations.append(
                f"🔴 **CRITIC**: {len(error_conflicts)} conflicte majore detectate - acestea blochează generarea ofertei și trebuie rezolvate"
            )
            
            for conflict in error_conflicts[:2]:  # Show first 2
                recommendations.append(f"   → {conflict.recommendation_ro}")
        
        # Recommendations for warnings
        if warning_conflicts:
            recommendations.append(
                f"🟡 **ATENȚIE**: {len(warning_conflicts)} inconsistențe detectate - clarificarea acestora va îmbunătăți acuratețea ofertei"
            )
            
            # Prioritize area and timeline warnings
            area_warnings = [c for c in warning_conflicts if c.conflict_type == ConflictType.AREA_MISMATCH]
            timeline_warnings = [c for c in warning_conflicts if c.conflict_type == ConflictType.TIMELINE_CONFLICT]
            
            if area_warnings:
                recommendations.append(f"   → Prioritar: Clarificați suprafața exactă (impact direct pe cost)")
            if timeline_warnings:
                recommendations.append(f"   → Prioritar: Confirmați timeline-ul (impact pe organizare și cost)")
        
        # General recommendations based on conflict patterns
        conflict_types = [c.conflict_type for c in conflicts]
        
        if conflict_types.count(ConflictType.SCOPE_MISMATCH) > 1:
            recommendations.append(
                "💡 **Sugestie**: Detectăm multiple inconsistențe în scope - recomandăm vizită la fața locului pentru clarificări"
            )
        
        if any(c.conflict_type == ConflictType.BUDGET_INCONSISTENCY for c in conflicts):
            recommendations.append(
                "💰 **Sugestie financiară**: Verificați alinierea buget-cerințe pentru a evita surprize neplăcute"
            )
        
        return recommendations


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def quick_consistency_check(
    dxf_area: Optional[float] = None,
    user_area: Optional[float] = None,
    rfp_timeline: Optional[Dict] = None,
    user_timeline: Optional[Dict] = None
) -> Tuple[bool, List[str]]:
    """
    Quick consistency check for common conflicts
    
    Returns:
        Tuple of (is_consistent, conflict_descriptions)
    """
    engine = CrossReferenceEngine()
    
    # Prepare minimal data structures
    dxf_data = None
    if dxf_area:
        dxf_data = {'dxf_analysis': {'total_area': dxf_area}}
    
    user_inputs = {}
    if user_area:
        user_inputs['total_area'] = user_area
    if user_timeline:
        user_inputs['work_timeline'] = user_timeline
    
    rfp_data = None
    if rfp_timeline:
        rfp_data = {'timeline': rfp_timeline}
    
    result = engine.validate_consistency(
        dxf_data=dxf_data,
        rfp_data=rfp_data,
        user_inputs=user_inputs
    )
    
    descriptions = [c.description_ro for c in result.conflicts]
    
    return result.is_consistent, descriptions


def validate_user_modification(
    dxf_data: Dict[str, Any],
    modification: Dict[str, Any]
) -> Tuple[bool, Optional[str]]:
    """
    Validate if a user-requested modification is feasible
    
    Args:
        dxf_data: Current DXF plan data
        modification: User modification request (e.g., add wall, change room)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Example: Check if user wants to add wall at coordinate
    if modification.get('type') == 'add_wall':
        x = modification.get('x')
        y = modification.get('y')
        
        # Check if coordinates are within plan bounds
        dxf_analysis = dxf_data.get('dxf_analysis', {})
        dimensions = dxf_analysis.get('dimensions', {})
        
        max_x = dimensions.get('max_x', float('inf'))
        max_y = dimensions.get('max_y', float('inf'))
        
        if x and y:
            if x > max_x or y > max_y:
                return False, f"Coordonatele ({x}, {y}) sunt în afara planului (max: {max_x}, {max_y})"
        
        # Additional checks could include:
        # - Does wall intersect with existing structural elements?
        # - Is wall length reasonable?
        # - Are there utilities in the path?
        
        return True, None
    
    return True, None


def analyze_conflict_impact(
    conflicts: List[DataConflict],
    project_budget: Optional[float] = None
) -> Dict[str, Any]:
    """
    Analyze the potential impact of conflicts on project
    
    Returns:
        Dictionary with impact analysis
    """
    impact = {
        'cost_impact': 'unknown',
        'timeline_impact': 'unknown',
        'risk_level': 'low',
        'requires_immediate_action': False,
        'affected_areas': []
    }
    
    # Check for high-impact conflicts
    error_count = len([c for c in conflicts if c.severity == ConflictSeverity.ERROR])
    
    if error_count > 0:
        impact['risk_level'] = 'high'
        impact['requires_immediate_action'] = True
    
    # Check for budget impacts
    budget_conflicts = [
        c for c in conflicts 
        if c.conflict_type == ConflictType.BUDGET_INCONSISTENCY
    ]
    
    if budget_conflicts:
        impact['cost_impact'] = 'significant'
        impact['affected_areas'].append('budget')
    
    # Check for timeline impacts
    timeline_conflicts = [
        c for c in conflicts 
        if c.conflict_type == ConflictType.TIMELINE_CONFLICT
    ]
    
    if timeline_conflicts:
        impact['timeline_impact'] = 'delays possible'
        impact['affected_areas'].append('timeline')
    
    # Check for area impacts
    area_conflicts = [
        c for c in conflicts 
        if c.conflict_type == ConflictType.AREA_MISMATCH
    ]
    
    if area_conflicts and project_budget:
        # Estimate cost impact based on area difference
        for conflict in area_conflicts:
            if 'impact' in str(conflict.impact_assessment):
                impact['cost_impact'] = 'variable'
    
    return impact