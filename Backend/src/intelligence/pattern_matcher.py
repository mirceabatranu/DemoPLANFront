# src/services/pattern_matcher.py
"""
DemoPLAN Pattern Matching Service
Intelligent project categorization and requirement pattern recognition
for Romanian construction projects
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import re

from src.services.firestore_service import FirestoreService
from config.config import settings

logger = logging.getLogger("demoplan.services.pattern_matcher")

class ProjectCategory(Enum):
    """Standard Romanian construction project categories"""
    RESIDENTIAL_APARTMENT = "apartament_rezidential"
    RESIDENTIAL_HOUSE = "casa_rezidentiala"
    COMMERCIAL_OFFICE = "birou_comercial"
    COMMERCIAL_RETAIL = "spatiu_comercial"
    INDUSTRIAL_WAREHOUSE = "depozit_industrial"
    MEDICAL_CLINIC = "clinica_medicala"
    EDUCATIONAL_CLASSROOM = "sala_clasa"
    HOSPITALITY_HOTEL = "camera_hotel"
    MIXED_USE = "utilizare_mixta"
    UNKNOWN = "necunoscut"

@dataclass
class RequirementPattern:
    """Pattern for common client requirements"""
    pattern_id: str
    category: ProjectCategory
    keywords_ro: List[str]
    typical_requirements: List[str]
    complexity_score: float  # 1.0 = simple, 2.0 = medium, 3.0 = complex
    estimated_timeline_days: int
    risk_factors: List[str]
    mep_requirements: List[str]
    regulatory_constraints: List[str]

@dataclass
class ClientPersona:
    """Client behavior and requirement patterns"""
    persona_id: str
    name: str
    typical_projects: List[ProjectCategory]
    communication_style: str  # "technical", "simple", "detailed"
    budget_sensitivity: str   # "low", "medium", "high"
    timeline_flexibility: str # "strict", "flexible", "very_flexible"
    quality_expectations: str # "standard", "premium", "luxury"
    change_frequency: str     # "rare", "occasional", "frequent"
    
@dataclass
class PatternMatch:
    """Result of pattern matching analysis"""
    project_category: ProjectCategory
    confidence: float
    matching_patterns: List[RequirementPattern]
    identified_persona: Optional[ClientPersona]
    predicted_requirements: List[str]
    risk_assessment: Dict[str, float]
    timeline_estimate: int
    complexity_score: float

class PatternMatcher:
    """
    Matches current projects against known patterns to predict requirements,
    categorize projects, and identify client personas
    """
    
    def __init__(self):
        self.firestore = FirestoreService()
        self.requirement_patterns: List[RequirementPattern] = []
        self.client_personas: List[ClientPersona] = []
        self.romanian_construction_keywords = self._load_romanian_keywords()
        
    async def initialize(self):
        """Initialize pattern matcher with predefined patterns"""
        try:
            await self.firestore.initialize()
            
            # Initialize with standard Romanian construction patterns
            self._initialize_standard_patterns()
            self._initialize_client_personas()
            
            logger.info("✅ Pattern Matcher initialized with Romanian construction patterns")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Pattern Matcher: {e}")
            raise

    async def analyze_project_patterns(
        self, 
        project_data: Dict[str, Any],
        user_requirements: Optional[str] = None
    ) -> PatternMatch:
        """
        Analyze project and match against known patterns
        
        Args:
            project_data: Project data from agents
            user_requirements: User-provided requirements text (Romanian)
            
        Returns:
            Pattern matching analysis
        """
        try:
            # 1. Categorize project based on floorplan analysis
            project_category = self._categorize_project(project_data)
            
            # 2. Extract requirements from user text
            extracted_requirements = []
            if user_requirements:
                extracted_requirements = self._extract_requirements_from_text(user_requirements)
            
            # 3. Match against requirement patterns
            matching_patterns = self._match_requirement_patterns(
                project_category, 
                extracted_requirements,
                project_data
            )
            
            # 4. Calculate confidence based on matches
            confidence = self._calculate_pattern_confidence(
                project_category, 
                matching_patterns, 
                project_data
            )
            
            # 5. Identify likely client persona
            client_persona = self._identify_client_persona(
                project_category,
                extracted_requirements,
                user_requirements
            )
            
            # 6. Predict additional requirements
            predicted_requirements = self._predict_requirements(
                matching_patterns,
                client_persona,
                project_data
            )
            
            # 7. Assess risks based on patterns
            risk_assessment = self._assess_pattern_risks(matching_patterns, project_category)
            
            # 8. Estimate timeline and complexity
            timeline_estimate = self._estimate_timeline(matching_patterns, project_data)
            complexity_score = self._calculate_complexity_score(matching_patterns, project_data)
            
            result = PatternMatch(
                project_category=project_category,
                confidence=confidence,
                matching_patterns=matching_patterns,
                identified_persona=client_persona,
                predicted_requirements=predicted_requirements,
                risk_assessment=risk_assessment,
                timeline_estimate=timeline_estimate,
                complexity_score=complexity_score
            )
            
            logger.info(f"📊 Pattern analysis complete: {project_category.value} ({confidence:.1%} confidence)")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in pattern analysis: {e}")
            return self._create_default_pattern_match()

    async def learn_from_project_outcome(
        self,
        session_id: str,
        original_pattern: PatternMatch,
        actual_outcome: Dict[str, Any]
    ) -> bool:
        """
        Learn from completed projects to improve pattern matching
        
        Args:
            session_id: Project session ID
            original_pattern: Original pattern prediction
            actual_outcome: Actual project results
            
        Returns:
            Success status
        """
        try:
            learning_data = {
                'session_id': session_id,
                'predicted_category': original_pattern.project_category.value,
                'predicted_timeline': original_pattern.timeline_estimate,
                'predicted_complexity': original_pattern.complexity_score,
                'actual_timeline': actual_outcome.get('actual_timeline_days'),
                'actual_complexity': actual_outcome.get('actual_complexity_score'),
                'accuracy': self._calculate_prediction_accuracy(original_pattern, actual_outcome),
                'learned_at': datetime.now(timezone.utc),
                'improvements_suggested': self._suggest_pattern_improvements(original_pattern, actual_outcome)
            }
            
            success = await self.firestore.save_document(
                collection='pattern_learning',
                document_id=f"{session_id}_{datetime.now().timestamp()}",
                data=learning_data
            )
            
            if success:
                logger.info(f"✅ Pattern learning data saved for session {session_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error saving pattern learning data: {e}")
            return False

    async def get_category_statistics(self) -> Dict[str, Any]:
        """Get statistics about project categories and patterns"""
        try:
            # Query recent projects
            recent_projects = await self.firestore.query_documents(
                collection='offers',
                filters=[
                    ('created_at', '>=', datetime.now(timezone.utc) - timedelta(days=90))
                ],
                limit=100
            )
            
            # Analyze categories
            category_counts = {}
            total_projects = len(recent_projects)
            
            for project in recent_projects:
                category = project.get('project_category', 'unknown')
                category_counts[category] = category_counts.get(category, 0) + 1
            
            # Calculate percentages
            category_percentages = {
                category: (count / total_projects * 100) if total_projects > 0 else 0
                for category, count in category_counts.items()
            }
            
            return {
                'total_projects_analyzed': total_projects,
                'category_distribution': category_percentages,
                'most_common_category': max(category_counts.items(), key=lambda x: x[1])[0] if category_counts else 'none',
                'pattern_confidence_average': self._calculate_average_confidence(recent_projects),
                'analysis_period_days': 90
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting category statistics: {e}")
            return {'error': str(e)}

    # Private helper methods
    
    def _categorize_project(self, project_data: Dict[str, Any]) -> ProjectCategory:
        """Categorize project based on floorplan analysis"""
        
        floorplan_data = project_data.get('floorplan_analysis', {})
        rooms = floorplan_data.get('rooms', [])
        
        if not rooms:
            return ProjectCategory.UNKNOWN
        
        # Analyze room types
        room_types = [room.get('type', '').lower() for room in rooms]
        room_names = [room.get('name', '').lower() for room in rooms]
        all_rooms = room_types + room_names
        
        # Classification rules based on Romanian room terminology
        residential_indicators = ['dormitor', 'bedroom', 'living', 'bucatarie', 'kitchen', 'baie', 'bathroom']
        office_indicators = ['birou', 'office', 'sala_conferinte', 'meeting', 'secretariat']
        retail_indicators = ['magazin', 'shop', 'vitrina', 'depozit_marfa', 'casa_marcat']
        medical_indicators = ['cabinet', 'consultatii', 'sterilizare', 'asteptare', 'receptie']
        educational_indicators = ['clasa', 'classroom', 'laborator', 'biblioteca', 'cancelarie']
        
        # Count indicators
        residential_score = sum(1 for indicator in residential_indicators if any(indicator in room for room in all_rooms))
        office_score = sum(1 for indicator in office_indicators if any(indicator in room for room in all_rooms))
        retail_score = sum(1 for indicator in retail_indicators if any(indicator in room for room in all_rooms))
        medical_score = sum(1 for indicator in medical_indicators if any(indicator in room for room in all_rooms))
        educational_score = sum(1 for indicator in educational_indicators if any(indicator in room for room in all_rooms))
        
        # Determine category
        scores = {
            ProjectCategory.RESIDENTIAL_APARTMENT: residential_score,
            ProjectCategory.COMMERCIAL_OFFICE: office_score,
            ProjectCategory.COMMERCIAL_RETAIL: retail_score,
            ProjectCategory.MEDICAL_CLINIC: medical_score,
            ProjectCategory.EDUCATIONAL_CLASSROOM: educational_score
        }
        
        max_category = max(scores, key=scores.get)
        
        # Additional refinement for residential
        if max_category == ProjectCategory.RESIDENTIAL_APARTMENT:
            area = floorplan_data.get('total_area', 0)
            if area > 150:  # Larger residential = likely house
                return ProjectCategory.RESIDENTIAL_HOUSE
        
        return max_category if scores[max_category] > 0 else ProjectCategory.UNKNOWN

    def _extract_requirements_from_text(self, text: str) -> List[str]:
        """Extract requirements from Romanian text using keywords"""
        
        requirements = []
        text_lower = text.lower()
        
        # Romanian requirement patterns
        requirement_patterns = {
            'calitate_premium': ['premium', 'lux', 'high-end', 'calitate superioara'],
            'buget_limitat': ['buget mic', 'economic', 'ieftin', 'cost redus'],
            'termen_urgent': ['urgent', 'rapid', 'cat mai repede', 'termen scurt'],
            'certificare_energetica': ['certificat energetic', 'eficienta energetica', 'clasa energetica'],
            'acces_dizabilitati': ['acces dizabilitati', 'rampa', 'lift', 'accesibil'],
            'sistem_securitate': ['alarma', 'securitate', 'camere', 'control acces'],
            'climatizare': ['aer conditionat', 'climatizare', 'ventilatie', 'hvac'],
            'pardoseala_speciala': ['parchet', 'gresie', 'marmura', 'epoxidica'],
            'iluminat_led': ['led', 'iluminat modern', 'becuri economice'],
            'automatizare': ['smart home', 'automatizare', 'domotica', 'control inteligent']
        }
        
        for requirement, keywords in requirement_patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                requirements.append(requirement)
        
        return requirements

    def _match_requirement_patterns(
        self, 
        category: ProjectCategory, 
        requirements: List[str],
        project_data: Dict[str, Any]
    ) -> List[RequirementPattern]:
        """Match against predefined requirement patterns"""
        
        matching_patterns = []
        
        # Filter patterns by category
        category_patterns = [p for p in self.requirement_patterns if p.category == category]
        
        for pattern in category_patterns:
            # Check keyword matches
            keyword_matches = 0
            for keyword in pattern.keywords_ro:
                if any(keyword in req for req in requirements):
                    keyword_matches += 1
            
            # Check project data matches
            data_matches = self._check_data_matches(pattern, project_data)
            
            # Calculate pattern relevance
            relevance_score = (keyword_matches / max(len(pattern.keywords_ro), 1)) * 0.7 + data_matches * 0.3
            
            if relevance_score > 0.3:  # Minimum relevance threshold
                matching_patterns.append(pattern)
        
        # Sort by relevance (estimated by complexity and timeline match)
        matching_patterns.sort(key=lambda p: p.complexity_score, reverse=True)
        
        return matching_patterns[:5]  # Return top 5 matches

    def _calculate_pattern_confidence(
        self, 
        category: ProjectCategory,
        patterns: List[RequirementPattern],
        project_data: Dict[str, Any]
    ) -> float:
        """Calculate confidence in pattern matching"""
        
        base_confidence = 0.5  # Base confidence
        
        # Category confidence
        if category != ProjectCategory.UNKNOWN:
            base_confidence += 0.2
        
        # Pattern match confidence
        if patterns:
            pattern_bonus = min(len(patterns) * 0.1, 0.3)
            base_confidence += pattern_bonus
        
        # Data completeness confidence
        data_fields = ['floorplan_analysis', 'mep_systems', 'validation_results']
        available_fields = sum(1 for field in data_fields if project_data.get(field))
        data_confidence = (available_fields / len(data_fields)) * 0.2
        base_confidence += data_confidence
        
        return min(base_confidence, 0.95)  # Max 95% confidence

    def _identify_client_persona(
        self,
        category: ProjectCategory,
        requirements: List[str],
        user_text: Optional[str]
    ) -> Optional[ClientPersona]:
        """Identify client persona based on project and communication"""
        
        if not user_text:
            return None
        
        text_lower = user_text.lower()
        
        # Analyze communication style
        technical_indicators = ['specificatie', 'norma', 'certificat', 'clasa energetica', 'detaliu tehnic']
        budget_indicators = ['pret', 'cost', 'buget', 'economic', 'ieftin', 'scump']
        quality_indicators = ['calitate', 'premium', 'lux', 'materiale bune', 'finisaje']
        
        technical_score = sum(1 for indicator in technical_indicators if indicator in text_lower)
        budget_mentions = sum(1 for indicator in budget_indicators if indicator in text_lower)
        quality_mentions = sum(1 for indicator in quality_indicators if indicator in text_lower)
        
        # Find best matching persona
        best_persona = None
        best_score = 0
        
        for persona in self.client_personas:
            if category in persona.typical_projects:
                score = 1  # Base score for category match
                
                # Communication style match
                if technical_score > 2 and persona.communication_style == "technical":
                    score += 2
                elif budget_mentions > quality_mentions and persona.budget_sensitivity == "high":
                    score += 2
                elif quality_mentions > budget_mentions and persona.quality_expectations == "premium":
                    score += 2
                
                if score > best_score:
                    best_score = score
                    best_persona = persona
        
        return best_persona

    def _predict_requirements(
        self,
        patterns: List[RequirementPattern],
        persona: Optional[ClientPersona],
        project_data: Dict[str, Any]
    ) -> List[str]:
        """Predict additional requirements based on patterns and persona"""
        
        predicted = []
        
        # From patterns
        for pattern in patterns:
            predicted.extend(pattern.typical_requirements)
        
        # From persona
        if persona:
            if persona.quality_expectations == "premium":
                predicted.extend([
                    "Materiale de calitate superioară",
                    "Finisaje premium",
                    "Detalii arhitecturale speciale"
                ])
            
            if persona.budget_sensitivity == "low":
                predicted.extend([
                    "Automatizare și smart systems",
                    "Tehnologii moderne"
                ])
        
        # Remove duplicates and return top predictions
        unique_predicted = list(set(predicted))
        return unique_predicted[:8]

    def _assess_pattern_risks(
        self, 
        patterns: List[RequirementPattern], 
        category: ProjectCategory
    ) -> Dict[str, float]:
        """Assess risks based on matched patterns"""
        
        risks = {
            'cost_overrun': 0.2,      # Base risk
            'timeline_delay': 0.15,   # Base risk
            'scope_creep': 0.1,       # Base risk
            'technical_complexity': 0.1,  # Base risk
            'regulatory_issues': 0.05  # Base risk
        }
        
        # Increase risks based on patterns
        for pattern in patterns:
            if pattern.complexity_score > 2.5:
                risks['technical_complexity'] += 0.1
                risks['cost_overrun'] += 0.05
            
            if len(pattern.regulatory_constraints) > 3:
                risks['regulatory_issues'] += 0.1
                risks['timeline_delay'] += 0.05
        
        # Category-specific risks
        if category == ProjectCategory.MEDICAL_CLINIC:
            risks['regulatory_issues'] += 0.15
        elif category == ProjectCategory.COMMERCIAL_RETAIL:
            risks['scope_creep'] += 0.1
        
        # Cap all risks at reasonable levels
        for risk_type in risks:
            risks[risk_type] = min(risks[risk_type], 0.6)
        
        return risks

    def _estimate_timeline(
        self, 
        patterns: List[RequirementPattern], 
        project_data: Dict[str, Any]
    ) -> int:
        """Estimate project timeline based on patterns"""
        
        if not patterns:
            return 45  # Default timeline in days
        
        # Calculate weighted average of pattern timelines
        total_weight = sum(p.complexity_score for p in patterns)
        if total_weight == 0:
            return 45
        
        weighted_timeline = sum(p.estimated_timeline_days * p.complexity_score for p in patterns) / total_weight
        
        # Adjust based on project area
        area = project_data.get('floorplan_analysis', {}).get('total_area', 100)
        area_factor = min(area / 100, 2.0)  # Scale factor, max 2x
        
        final_timeline = int(weighted_timeline * area_factor)
        
        # Reasonable bounds
        return max(14, min(final_timeline, 180))  # Between 2 weeks and 6 months

    def _calculate_complexity_score(
        self, 
        patterns: List[RequirementPattern], 
        project_data: Dict[str, Any]
    ) -> float:
        """Calculate overall project complexity score"""
        
        if not patterns:
            return 1.5  # Default medium complexity
        
        avg_complexity = sum(p.complexity_score for p in patterns) / len(patterns)
        
        # Adjust based on MEP systems
        mep_systems = project_data.get('mep_systems', {})
        mep_bonus = len(mep_systems) * 0.1
        
        # Adjust based on room count
        rooms_count = len(project_data.get('floorplan_analysis', {}).get('rooms', []))
        rooms_bonus = max(0, (rooms_count - 3) * 0.05)  # Bonus for rooms over 3
        
        total_complexity = avg_complexity + mep_bonus + rooms_bonus
        
        # Bound between 1.0 and 3.0
        return max(1.0, min(total_complexity, 3.0))

    def _initialize_standard_patterns(self):
        """Initialize standard Romanian construction patterns"""
        
        self.requirement_patterns = [
            # Residential Apartment Patterns
            RequirementPattern(
                pattern_id="apartament_standard",
                category=ProjectCategory.RESIDENTIAL_APARTMENT,
                keywords_ro=["apartament", "garsoniera", "2 camere", "3 camere"],
                typical_requirements=[
                    "Instalații electrice conform normelor",
                    "Instalații sanitare complete",
                    "Pardoseală laminat/gresie", 
                    "Zugrăveli lavabile",
                    "Uși interior standard"
                ],
                complexity_score=1.5,
                estimated_timeline_days=35,
                risk_factors=["Interferențe cu vecinii", "Acces limitat"],
                mep_requirements=["Electricitate", "Sanitare", "Încălzire"],
                regulatory_constraints=["Aviz ISU", "Certificat energetic"]
            ),
            
            RequirementPattern(
                pattern_id="apartament_premium",
                category=ProjectCategory.RESIDENTIAL_APARTMENT,
                keywords_ro=["premium", "lux", "penthouse", "finisaje superioare"],
                typical_requirements=[
                    "Materiale premium (parchet masiv, faianță importată)",
                    "Instalații smart home",
                    "Climatizare individuală",
                    "Iluminat decorativ LED",
                    "Mobilier integrat"
                ],
                complexity_score=2.5,
                estimated_timeline_days=55,
                risk_factors=["Costuri materiale", "Aprovizionare materiale speciale"],
                mep_requirements=["Electricitate avansată", "Climatizare", "Automatizare"],
                regulatory_constraints=["Aviz ISU", "Certificat energetic", "Autorizații speciale"]
            ),
            
            # Commercial Office Patterns
            RequirementPattern(
                pattern_id="birou_standard", 
                category=ProjectCategory.COMMERCIAL_OFFICE,
                keywords_ro=["birou", "office", "spațiu comercial", "birouri"],
                typical_requirements=[
                    "Compartimentări modulare",
                    "Instalații IT complete",
                    "Sistem ventilație/climatizare",
                    "Pardoseală tehnică",
                    "Iluminat profesional"
                ],
                complexity_score=2.0,
                estimated_timeline_days=45,
                risk_factors=["Cerințe IT complexe", "Normele muncii"],
                mep_requirements=["Electricitate", "IT&C", "Climatizare", "Ventilație"],
                regulatory_constraints=["Aviz ISU", "Aviz ITM", "Certificat energetic"]
            ),
            
            # Medical Clinic Pattern
            RequirementPattern(
                pattern_id="clinica_medicala",
                category=ProjectCategory.MEDICAL_CLINIC, 
                keywords_ro=["clinică", "cabinet medical", "policlinică"],
                typical_requirements=[
                    "Finisaje antibacteriene", 
                    "Sistem ventilație medicală",
                    "Instalații gaze medicale",
                    "Pardoseală antistatică", 
                    "Iluminat special cabinete"
                ],
                complexity_score=3.0,
                estimated_timeline_days=65,
                risk_factors=["Reglementări stricte", "Materiale speciale", "Autorizații multiple"],
                mep_requirements=["Electricitate", "Ventilație specială", "Gaze medicale", "IT"],
                regulatory_constraints=["Aviz ISU", "Aviz DSP", "Aviz Colegiul Medicilor"]
            )
        ]

    def _initialize_client_personas(self):
        """Initialize standard client personas"""
        
        self.client_personas = [
            ClientPersona(
                persona_id="proprietar_apartament",
                name="Proprietar Apartament Rezidențial",
                typical_projects=[ProjectCategory.RESIDENTIAL_APARTMENT],
                communication_style="simple",
                budget_sensitivity="high",
                timeline_flexibility="flexible", 
                quality_expectations="standard",
                change_frequency="occasional"
            ),
            
            ClientPersona(
                persona_id="dezvoltator_premium",
                name="Dezvoltator Premium",
                typical_projects=[ProjectCategory.RESIDENTIAL_HOUSE, ProjectCategory.RESIDENTIAL_APARTMENT],
                communication_style="technical",
                budget_sensitivity="low",
                timeline_flexibility="strict",
                quality_expectations="premium", 
                change_frequency="rare"
            ),
            
            ClientPersona(
                persona_id="antreprenor_comercial",
                name="Antreprenor Comercial",
                typical_projects=[ProjectCategory.COMMERCIAL_OFFICE, ProjectCategory.COMMERCIAL_RETAIL],
                communication_style="detailed",
                budget_sensitivity="medium",
                timeline_flexibility="flexible",
                quality_expectations="standard",
                change_frequency="frequent"
            ),
            
            ClientPersona(
                persona_id="medic_clinica",
                name="Medic - Clinică Privată",
                typical_projects=[ProjectCategory.MEDICAL_CLINIC],
                communication_style="technical",
                budget_sensitivity="low",
                timeline_flexibility="very_flexible",
                quality_expectations="premium",
                change_frequency="occasional"
            )
        ]

    def _load_romanian_keywords(self) -> Dict[str, List[str]]:
        """Load Romanian construction keywords"""
        return {
            'materials': ['beton', 'caramida', 'gips-carton', 'parchet', 'gresie', 'faianță'],
            'rooms': ['dormitor', 'living', 'bucatarie', 'baie', 'hol', 'balcon'],
            'systems': ['electricitate', 'sanitare', 'încălzire', 'ventilație', 'climatizare'],
            'finishes': ['zugrăveli', 'vopsea', 'tapet', 'lambriu', 'rigips'],
            'quality': ['premium', 'standard', 'economic', 'lux', 'calitate']
        }

    def _check_data_matches(self, pattern: RequirementPattern, project_data: Dict[str, Any]) -> float:
        """Check how well project data matches pattern requirements"""
        
        matches = 0
        total_checks = 0
        
        # Check MEP requirements
        mep_systems = project_data.get('mep_systems', {})
        for mep_req in pattern.mep_requirements:
            total_checks += 1
            if mep_req.lower() in [sys.lower() for sys in mep_systems.keys()]:
                matches += 1
        
        # Check room types if available
        rooms = project_data.get('floorplan_analysis', {}).get('rooms', [])
        if rooms:
            room_types = [room.get('type', '').lower() for room in rooms]
            # Simple check - if pattern is for residential and we have bedrooms/living
            if pattern.category == ProjectCategory.RESIDENTIAL_APARTMENT:
                total_checks += 1
                if any('dormitor' in rt or 'bedroom' in rt for rt in room_types):
                    matches += 1
        
        return matches / total_checks if total_checks > 0 else 0.0

    def _create_default_pattern_match(self) -> PatternMatch:
        """Create default pattern match for error cases"""
        return PatternMatch(
            project_category=ProjectCategory.UNKNOWN,
            confidence=0.1,
            matching_patterns=[],
            identified_persona=None,
            predicted_requirements=["Evaluare detaliată necesară"],
            risk_assessment={'cost_overrun': 0.3, 'timeline_delay': 0.2, 'scope_creep': 0.15},
            timeline_estimate=45,
            complexity_score=2.0
        )

    def _calculate_prediction_accuracy(
        self, 
        original: PatternMatch, 
        actual: Dict[str, Any]
    ) -> float:
        """Calculate accuracy of original predictions"""
        
        accuracy_components = []
        
        # Timeline accuracy
        if actual.get('actual_timeline_days'):
            predicted_timeline = original.timeline_estimate
            actual_timeline = actual['actual_timeline_days']
            timeline_accuracy = 1 - abs(predicted_timeline - actual_timeline) / max(predicted_timeline, actual_timeline)
            accuracy_components.append(max(0, timeline_accuracy))
        
        # Complexity accuracy
        if actual.get('actual_complexity_score'):
            predicted_complexity = original.complexity_score
            actual_complexity = actual['actual_complexity_score']
            complexity_accuracy = 1 - abs(predicted_complexity - actual_complexity) / max(predicted_complexity, actual_complexity)
            accuracy_components.append(max(0, complexity_accuracy))
        
        return sum(accuracy_components) / len(accuracy_components) if accuracy_components else 0.5

    def _suggest_pattern_improvements(
        self, 
        original: PatternMatch, 
        actual: Dict[str, Any]
    ) -> List[str]:
        """Suggest improvements based on prediction vs actual results"""
        
        suggestions = []
        
        # Timeline suggestions
        if actual.get('actual_timeline_days'):
            predicted = original.timeline_estimate
            actual_timeline = actual['actual_timeline_days']
            
            if actual_timeline > predicted * 1.2:
                suggestions.append("Increase timeline estimates for similar projects")
            elif actual_timeline < predicted * 0.8:
                suggestions.append("Reduce timeline estimates for similar projects")
        
        # Complexity suggestions
        if actual.get('actual_complexity_score'):
            if actual['actual_complexity_score'] > original.complexity_score * 1.3:
                suggestions.append("Increase complexity assessment for similar patterns")
        
        return suggestions

    def _calculate_average_confidence(self, projects: List[Dict[str, Any]]) -> float:
        """Calculate average pattern matching confidence from recent projects"""
        
        confidences = [p.get('pattern_confidence', 0.5) for p in projects if p.get('pattern_confidence')]
        return sum(confidences) / len(confidences) if confidences else 0.5

# Export for use in other modules
__all__ = ['PatternMatcher', 'ProjectCategory', 'RequirementPattern', 'ClientPersona', 'PatternMatch']