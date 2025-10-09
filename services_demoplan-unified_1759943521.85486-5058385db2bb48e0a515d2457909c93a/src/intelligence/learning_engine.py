# src/intelligence/learning_engine.py
"""
DemoPLAN Learning Engine
Foundational scoring framework and adaptive learning system 
for Romanian construction projects
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import statistics
import math

from src.services.firestore_service import FirestoreService
from src.intelligence.historical_analyzer import HistoricalAnalyzer, SimilarityMatch
from src.intelligence.pattern_matcher import PatternMatcher, PatternMatch
from config.config import settings

logger = logging.getLogger("demoplan.intelligence.learning_engine")

class LearningMetric(Enum):
    """Types of learning metrics tracked"""
    COST_ACCURACY = "cost_accuracy"
    TIMELINE_ACCURACY = "timeline_accuracy"  
    COMPLEXITY_PREDICTION = "complexity_prediction"
    CLIENT_SATISFACTION = "client_satisfaction"
    CHANGE_REQUEST_FREQUENCY = "change_requests"
    RISK_MATERIALIZATION = "risks_realized"

@dataclass
class LearningScore:
    """Individual learning metric score"""
    metric: LearningMetric
    score: float  # 0.0 to 1.0
    confidence: float  # How confident we are in this score
    sample_size: int  # Number of projects contributing to score
    trend: str  # "improving", "stable", "declining"
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class ProjectLearning:
    """Learning data from a completed project"""
    session_id: str
    project_category: str
    predicted_cost: Optional[float]
    actual_cost: Optional[float]
    predicted_timeline: Optional[int]
    actual_timeline: Optional[int]
    predicted_complexity: Optional[float]
    actual_complexity: Optional[float]
    client_satisfaction: Optional[float]
    change_requests_count: int
    risks_materialized: List[str]
    lessons_learned: List[str]
    recorded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class IntelligenceInsight:
    """Actionable insight from learning analysis"""
    insight_type: str  # "cost_pattern", "timeline_risk", "client_behavior", etc.
    title: str
    description: str
    confidence: float
    impact_score: float  # How much this could improve estimates
    recommended_actions: List[str]
    applies_to_categories: List[str]

class LearningEngine:
    """
    Core learning engine that adapts and improves project predictions
    based on historical outcomes and patterns
    """
    
    def __init__(self):
        # Existing initialization
        self.firestore = FirestoreService()
        self.historical_analyzer = HistoricalAnalyzer()
        self.pattern_matcher = PatternMatcher()
        
        # Add batch processing support
        self.batch_processor = None
        self.batch_training_enabled = False
        
        # Training configuration
        self.learning_weights = {
            LearningMetric.COST_ACCURACY: 0.30,
            LearningMetric.TIMELINE_ACCURACY: 0.25,
            LearningMetric.CLIENT_SATISFACTION: 0.20,
            LearningMetric.COMPLEXITY_PREDICTION: 0.15,
            LearningMetric.CHANGE_REQUEST_FREQUENCY: 0.05,
            LearningMetric.RISK_MATERIALIZATION: 0.05
        }
        
        self.min_sample_size = 3
        self.learning_decay_days = 180
        
        # Batch processing configuration
        self.batch_config = {
            "batch_size": 50,
            "max_concurrent_batches": 3,
            "auto_retrain_threshold": 100
        }

    async def initialize(self, enable_batch_processing: bool = False, batch_processor: Optional[Any] = None):
        """Initialize with optional batch processing support"""
        try:
            await self.firestore.initialize()
            await self.historical_analyzer.initialize()
            await self.pattern_matcher.initialize()
            
            # Link batch processor if provided
            if enable_batch_processing and batch_processor:
                try:
                    self.batch_processor = batch_processor
                    self.batch_training_enabled = True
                    logger.info("Learning Engine initialized with batch processing")
                except Exception as e:
                    logger.warning(f"Batch processing initialization failed: {e}")
                    self.batch_training_enabled = False
            
            logger.info("Learning Engine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Learning Engine: {e}")
            raise

    async def analyze_project_intelligence(
        self,
        current_project: Dict[str, Any],
        agent_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply learned intelligence to improve current project analysis
        
        Args:
            current_project: Current project data
            agent_results: Results from all agents
            
        Returns:
            Enhanced analysis with learning-based improvements
        """
        try:
            # 1. Get historical similarities
            similar_projects = await self.historical_analyzer.analyze_project_similarity(current_project)
            
            # 2. Get pattern matches
            pattern_analysis = await self.pattern_matcher.analyze_project_patterns(
                current_project, 
                current_project.get('user_requirements')
            )
            
            # 3. Apply learned corrections
            enhanced_estimates = await self._apply_learned_corrections(
                agent_results,
                similar_projects,
                pattern_analysis
            )
            
            # 4. Generate intelligence insights
            insights = await self._generate_intelligence_insights(
                current_project,
                similar_projects,
                pattern_analysis
            )
            
            # 5. Calculate confidence adjustments
            confidence_adjustments = await self._calculate_confidence_adjustments(
                agent_results,
                similar_projects,
                pattern_analysis
            )
            
            return {
                'enhanced_estimates': enhanced_estimates,
                'intelligence_insights': insights,
                'confidence_adjustments': confidence_adjustments,
                'learning_applied': True,
                'similar_projects_count': len(similar_projects),
                'pattern_confidence': pattern_analysis.confidence,
                'learning_metadata': {
                    'historical_matches': len(similar_projects),
                    'pattern_category': pattern_analysis.project_category.value,
                    'complexity_score': pattern_analysis.complexity_score,
                    'generated_at': datetime.now(timezone.utc).isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error in intelligence analysis: {e}")
            return {
                'enhanced_estimates': agent_results,
                'intelligence_insights': [],
                'confidence_adjustments': {},
                'learning_applied': False,
                'error': str(e)
            }

    async def record_project_outcome(
        self,
        session_id: str,
        original_predictions: Dict[str, Any],
        actual_outcomes: Dict[str, Any]
    ) -> bool:
        """
        Record actual project outcomes for future learning
        
        Args:
            session_id: Project session ID
            original_predictions: What we predicted
            actual_outcomes: What actually happened
            
        Returns:
            Success status
        """
        try:
            # Create learning record
            project_learning = ProjectLearning(
                session_id=session_id,
                project_category=original_predictions.get('project_category', 'unknown'),
                predicted_cost=original_predictions.get('estimated_cost'),
                actual_cost=actual_outcomes.get('actual_cost'),
                predicted_timeline=original_predictions.get('estimated_timeline_days'),
                actual_timeline=actual_outcomes.get('actual_timeline_days'),
                predicted_complexity=original_predictions.get('complexity_score'),
                actual_complexity=actual_outcomes.get('actual_complexity_score'),
                client_satisfaction=actual_outcomes.get('client_satisfaction_score'),
                change_requests_count=actual_outcomes.get('change_requests_count', 0),
                risks_materialized=actual_outcomes.get('risks_materialized', []),
                lessons_learned=actual_outcomes.get('lessons_learned', [])
            )
            
            # Convert to dict for storage
            learning_data = {
                'session_id': project_learning.session_id,
                'project_category': project_learning.project_category,
                'predicted_cost': project_learning.predicted_cost,
                'actual_cost': project_learning.actual_cost,
                'predicted_timeline': project_learning.predicted_timeline,
                'actual_timeline': project_learning.actual_timeline,
                'predicted_complexity': project_learning.predicted_complexity,
                'actual_complexity': project_learning.actual_complexity,
                'client_satisfaction': project_learning.client_satisfaction,
                'change_requests_count': project_learning.change_requests_count,
                'risks_materialized': project_learning.risks_materialized,
                'lessons_learned': project_learning.lessons_learned,
                'recorded_at': project_learning.recorded_at,
                'source': 'learning_engine'
            }
            
            # Save to database
            success = await self.firestore.save_document(
                collection='project_learning',
                document_id=f"{session_id}_{datetime.now().timestamp()}",
                data=learning_data
            )
            
            if success:
                # Update learning metrics
                await self._update_learning_metrics(project_learning)
                logger.info(f"✅ Project learning recorded for session {session_id}")
            else:
                logger.error(f"❌ Failed to save project learning for session {session_id}")
                
            return success
            
        except Exception as e:
            logger.error(f"❌ Error recording project outcome: {e}")
            return False

    async def get_learning_dashboard(self) -> Dict[str, Any]:
        """
        Get comprehensive learning dashboard data
        
        Returns:
            Dashboard data with learning metrics and insights
        """
        try:
            # Get recent learning data
            recent_projects = await self._get_recent_learning_data(days=90)
            
            # Calculate learning scores
            learning_scores = await self._calculate_learning_scores(recent_projects)
            
            # Generate top insights
            top_insights = await self._generate_top_insights(recent_projects)
            
            # Calculate improvement trends
            improvement_trends = await self._calculate_improvement_trends(recent_projects)
            
            # Get category performance
            category_performance = await self._get_category_performance(recent_projects)
            
            return {
                'learning_scores': {metric.value: score.__dict__ for metric, score in learning_scores.items()},
                'top_insights': [insight.__dict__ for insight in top_insights],
                'improvement_trends': improvement_trends,
                'category_performance': category_performance,
                'projects_analyzed': len(recent_projects),
                'learning_period_days': 90,
                'generated_at': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating learning dashboard: {e}")
            return {'error': str(e)}

    async def predict_project_success_factors(
        self,
        project_data: Dict[str, Any],
        similar_projects: List[SimilarityMatch],
        pattern_analysis: PatternMatch
    ) -> Dict[str, Any]:
        """
        Predict key success factors for current project based on learning
        
        Args:
            project_data: Current project information
            similar_projects: Similar historical projects
            pattern_analysis: Pattern matching results
            
        Returns:
            Success factor predictions
        """
        try:
            success_factors = {
                'critical_success_factors': [],
                'risk_mitigation_actions': [],
                'client_satisfaction_drivers': [],
                'cost_optimization_opportunities': [],
                'timeline_acceleration_tactics': []
            }
            
            # Analyze similar projects for success patterns
            if similar_projects:
                high_satisfaction_projects = [
                    match for match in similar_projects 
                    if hasattr(match.historical_project, 'client_satisfaction') 
                    and match.historical_project.client_satisfaction 
                    and match.historical_project.client_satisfaction >= 4.0
                ]
                
                if high_satisfaction_projects:
                    success_factors['critical_success_factors'].extend([
                        "Comunicare frecventă cu clientul",
                        "Respectarea termenelor promise",
                        "Transparență în costuri și modificări"
                    ])
            
            # Pattern-based success factors
            if pattern_analysis.identified_persona:
                persona = pattern_analysis.identified_persona
                
                if persona.communication_style == "technical":
                    success_factors['client_satisfaction_drivers'].append(
                        "Documentație tehnică detaliată și actualizată"
                    )
                elif persona.communication_style == "simple":
                    success_factors['client_satisfaction_drivers'].append(
                        "Explicații clare și accesibile pentru progres"
                    )
                
                if persona.budget_sensitivity == "high":
                    success_factors['cost_optimization_opportunities'].extend([
                        "Identificarea alternativelor de materiale",
                        "Optimizarea fazelor de execuție"
                    ])
            
            # Risk mitigation based on pattern risks
            for risk_type, risk_level in pattern_analysis.risk_assessment.items():
                if risk_level > 0.3:
                    if risk_type == "cost_overrun":
                        success_factors['risk_mitigation_actions'].append(
                            "Stabilirea unui buget de contingență de min 15%"
                        )
                    elif risk_type == "timeline_delay":
                        success_factors['risk_mitigation_actions'].append(
                            "Buffer de timp pentru fiecare fază critică"
                        )
            
            # Timeline acceleration tactics
            if pattern_analysis.complexity_score < 2.0:
                success_factors['timeline_acceleration_tactics'].extend([
                    "Execuție paralelă a lucrărilor independente",
                    "Pre-comandarea materialelor critice"
                ])
            
            return success_factors
            
        except Exception as e:
            logger.error(f"❌ Error predicting success factors: {e}")
            return {'error': str(e)}

    # Private helper methods
    
    async def _apply_learned_corrections(
        self,
        agent_results: Dict[str, Any],
        similar_projects: List[SimilarityMatch],
        pattern_analysis: PatternMatch
    ) -> Dict[str, Any]:
        """Apply learned corrections to agent estimates"""
        
        enhanced = agent_results.copy()
        
        # Cost corrections based on historical accuracy
        if similar_projects and agent_results.get('estimation_results', {}).get('total_cost'):
            historical_accuracy = await self._get_cost_accuracy_for_category(
                pattern_analysis.project_category.value
            )
            
            if historical_accuracy < 0.8:  # Historical underestimation
                original_cost = agent_results['estimation_results']['total_cost']
                correction_factor = 1.0 + (0.8 - historical_accuracy)  # Adjust upward
                enhanced_cost = original_cost * correction_factor
                
                if 'estimation_results' not in enhanced:
                    enhanced['estimation_results'] = {}
                enhanced['estimation_results']['total_cost'] = enhanced_cost
                enhanced['estimation_results']['learning_correction_applied'] = True
                enhanced['estimation_results']['correction_factor'] = correction_factor
                enhanced['estimation_results']['correction_reason'] = f"Corecție bazată pe precizia istorică {historical_accuracy:.1%}"
        
        # Timeline corrections
        if pattern_analysis.timeline_estimate and similar_projects:
            timeline_variance = await self._get_timeline_variance_for_category(
                pattern_analysis.project_category.value
            )
            
            if timeline_variance > 0.2:  # High variance = add buffer
                buffer_days = int(pattern_analysis.timeline_estimate * 0.15)
                enhanced['timeline_estimate'] = pattern_analysis.timeline_estimate + buffer_days
                enhanced['timeline_buffer_added'] = buffer_days
                enhanced['timeline_correction_reason'] = f"Buffer adăugat din cauza varianței istorice {timeline_variance:.1%}"
        
        return enhanced

    async def _generate_intelligence_insights(
        self,
        project_data: Dict[str, Any],
        similar_projects: List[SimilarityMatch],
        pattern_analysis: PatternMatch
    ) -> List[IntelligenceInsight]:
        """Generate actionable insights from learning data"""
        
        insights = []
        
        # Cost optimization insights
        if similar_projects:
            cost_variance = self._calculate_cost_variance(similar_projects)
            if cost_variance > 0.25:
                insights.append(IntelligenceInsight(
                    insight_type="cost_pattern",
                    title="Variabilitate Mare în Costuri",
                    description=f"Proiectele similare au varianța costurilor de {cost_variance:.1%}. Recomandăm analiză detaliată.",
                    confidence=0.8,
                    impact_score=0.6,
                    recommended_actions=[
                        "Solicitați oferte de la mai mulți furnizori",
                        "Creați un buget de contingență mărit",
                        "Definiți clar specificațiile pentru a reduce variabilitatea"
                    ],
                    applies_to_categories=[pattern_analysis.project_category.value]
                ))
        
        # Timeline insights
        if pattern_analysis.timeline_estimate:
            seasonal_factor = self._get_seasonal_adjustment()
            if seasonal_factor != 1.0:
                insights.append(IntelligenceInsight(
                    insight_type="timeline_risk",
                    title="Factor Sezonier Identificat",
                    description=f"Perioada actuală afectează durata lucrărilor cu {(seasonal_factor-1)*100:+.0f}%",
                    confidence=0.7,
                    impact_score=0.4,
                    recommended_actions=[
                        "Ajustați planificarea în funcție de sezon",
                        "Considerați impactul condițiilor meteorologice"
                    ],
                    applies_to_categories=[pattern_analysis.project_category.value]
                ))
        
        # Client satisfaction insights
        if pattern_analysis.identified_persona:
            persona_insights = self._generate_persona_insights(pattern_analysis.identified_persona)
            insights.extend(persona_insights)
        
        return insights

    async def _calculate_confidence_adjustments(
        self,
        agent_results: Dict[str, Any],
        similar_projects: List[SimilarityMatch],
        pattern_analysis: PatternMatch
    ) -> Dict[str, float]:
        """Calculate confidence adjustments based on learning"""
        
        adjustments = {}
        
        # Historical data availability adjustment
        if similar_projects:
            similarity_bonus = min(len(similar_projects) * 0.05, 0.20)  # Up to 20% bonus
            avg_similarity = sum(match.similarity_score for match in similar_projects) / len(similar_projects)
            adjustments['historical_data_bonus'] = similarity_bonus * avg_similarity
        
        # Pattern matching adjustment
        if pattern_analysis.confidence > 0.7:
            adjustments['pattern_matching_bonus'] = 0.10
        elif pattern_analysis.confidence < 0.4:
            adjustments['pattern_matching_penalty'] = -0.15
        
        # Learning quality adjustment
        learning_quality = await self._assess_learning_quality(pattern_analysis.project_category.value)
        if learning_quality > 0.8:
            adjustments['learning_quality_bonus'] = 0.08
        elif learning_quality < 0.5:
            adjustments['learning_quality_penalty'] = -0.12
        
        return adjustments

    async def _update_learning_metrics(self, project_learning: ProjectLearning):
        """Update overall learning metrics based on new project outcome"""
        
        try:
            # Calculate individual metric scores
            if project_learning.predicted_cost and project_learning.actual_cost:
                cost_accuracy = min(project_learning.predicted_cost, project_learning.actual_cost) / max(project_learning.predicted_cost, project_learning.actual_cost)
                await self._update_metric_score(LearningMetric.COST_ACCURACY, cost_accuracy, project_learning.project_category)
            
            if project_learning.predicted_timeline and project_learning.actual_timeline:
                timeline_accuracy = min(project_learning.predicted_timeline, project_learning.actual_timeline) / max(project_learning.predicted_timeline, project_learning.actual_timeline)
                await self._update_metric_score(LearningMetric.TIMELINE_ACCURACY, timeline_accuracy, project_learning.project_category)
            
            if project_learning.client_satisfaction:
                satisfaction_score = project_learning.client_satisfaction / 5.0  # Normalize to 0-1
                await self._update_metric_score(LearningMetric.CLIENT_SATISFACTION, satisfaction_score, project_learning.project_category)
            
            # Change request frequency (inverse - fewer changes = better)
            change_frequency_score = max(0, 1.0 - (project_learning.change_requests_count / 10.0))  # Normalize assuming max 10 changes
            await self._update_metric_score(LearningMetric.CHANGE_REQUEST_FREQUENCY, change_frequency_score, project_learning.project_category)
            
        except Exception as e:
            logger.error(f"❌ Error updating learning metrics: {e}")

    async def _update_metric_score(self, metric: LearningMetric, score: float, category: str):
        """Update individual metric score"""
        
        try:
            metric_key = f"{category}_{metric.value}"
            
            # Get existing metric data
            existing_data = await self.firestore.get_document('learning_metrics', metric_key)
            
            if existing_data:
                # Update existing metric
                current_score = existing_data.get('average_score', 0.5)
                sample_size = existing_data.get('sample_size', 0)
                
                # Calculate new average (simple moving average)
                new_average = ((current_score * sample_size) + score) / (sample_size + 1)
                new_sample_size = sample_size + 1
                
                # Determine trend (simple comparison with previous)
                if new_average > current_score * 1.05:
                    trend = "improving"
                elif new_average < current_score * 0.95:
                    trend = "declining"
                else:
                    trend = "stable"
                
            else:
                # Create new metric
                new_average = score
                new_sample_size = 1
                trend = "new"
            
            # Save updated metric
            metric_data = {
                'metric': metric.value,
                'category': category,
                'average_score': new_average,
                'latest_score': score,
                'sample_size': new_sample_size,
                'trend': trend,
                'last_updated': datetime.now(timezone.utc),
                'confidence': min(new_sample_size / 10.0, 1.0)  # Confidence based on sample size
            }
            
            await self.firestore.save_document('learning_metrics', metric_key, metric_data)
            
        except Exception as e:
            logger.error(f"❌ Error updating metric {metric.value}: {e}")

    async def _get_recent_learning_data(self, days: int = 90) -> List[ProjectLearning]:
        """Get recent project learning data"""
        
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            docs = await self.firestore.query_documents(
                collection='project_learning',
                filters=[('recorded_at', '>=', cutoff_date)],
                limit=50
            )
            
            projects = []
            for doc in docs:
                try:
                    project = ProjectLearning(
                        session_id=doc.get('session_id', 'unknown'),
                        project_category=doc.get('project_category', 'unknown'),
                        predicted_cost=doc.get('predicted_cost'),
                        actual_cost=doc.get('actual_cost'),
                        predicted_timeline=doc.get('predicted_timeline'),
                        actual_timeline=doc.get('actual_timeline'),
                        predicted_complexity=doc.get('predicted_complexity'),
                        actual_complexity=doc.get('actual_complexity'),
                        client_satisfaction=doc.get('client_satisfaction'),
                        change_requests_count=doc.get('change_requests_count', 0),
                        risks_materialized=doc.get('risks_materialized', []),
                        lessons_learned=doc.get('lessons_learned', []),
                        recorded_at=doc.get('recorded_at', datetime.now(timezone.utc))
                    )
                    projects.append(project)
                except Exception as e:
                    logger.debug(f"Skipping invalid learning record: {e}")
                    continue
                    
            return projects
            
        except Exception as e:
            logger.error(f"❌ Error getting recent learning data: {e}")
            return []

    async def _calculate_learning_scores(self, projects: List[ProjectLearning]) -> Dict[LearningMetric, LearningScore]:
        """Calculate current learning scores from project data"""
        
        scores = {}
        
        for metric in LearningMetric:
            metric_values = []
            
            for project in projects:
                if metric == LearningMetric.COST_ACCURACY and project.predicted_cost and project.actual_cost:
                    accuracy = min(project.predicted_cost, project.actual_cost) / max(project.predicted_cost, project.actual_cost)
                    metric_values.append(accuracy)
                elif metric == LearningMetric.TIMELINE_ACCURACY and project.predicted_timeline and project.actual_timeline:
                    accuracy = min(project.predicted_timeline, project.actual_timeline) / max(project.predicted_timeline, project.actual_timeline)
                    metric_values.append(accuracy)
                elif metric == LearningMetric.CLIENT_SATISFACTION and project.client_satisfaction:
                    normalized_satisfaction = project.client_satisfaction / 5.0
                    metric_values.append(normalized_satisfaction)
            
            if metric_values:
                avg_score = statistics.mean(metric_values)
                confidence = min(len(metric_values) / 10.0, 1.0)
                
                # Simple trend analysis (compare recent vs older)
                if len(metric_values) >= 6:
                    recent_avg = statistics.mean(metric_values[-3:])
                    older_avg = statistics.mean(metric_values[:3])
                    if recent_avg > older_avg * 1.05:
                        trend = "improving"
                    elif recent_avg < older_avg * 0.95:
                        trend = "declining" 
                    else:
                        trend = "stable"
                else:
                    trend = "insufficient_data"
                
                scores[metric] = LearningScore(
                    metric=metric,
                    score=avg_score,
                    confidence=confidence,
                    sample_size=len(metric_values),
                    trend=trend
                )
            else:
                scores[metric] = LearningScore(
                    metric=metric,
                    score=0.5,  # Neutral score
                    confidence=0.0,
                    sample_size=0,
                    trend="no_data"
                )
        
        return scores

    async def _generate_top_insights(self, projects: List[ProjectLearning]) -> List[IntelligenceInsight]:
        """Generate top insights from learning data"""
        
        insights = []
        
        if len(projects) < 3:
            return insights
        
        # Cost accuracy insight
        cost_projects = [p for p in projects if p.predicted_cost and p.actual_cost]
        if cost_projects:
            accuracies = [min(p.predicted_cost, p.actual_cost) / max(p.predicted_cost, p.actual_cost) for p in cost_projects]
            avg_accuracy = statistics.mean(accuracies)
            
            if avg_accuracy < 0.8:
                insights.append(IntelligenceInsight(
                    insight_type="cost_accuracy",
                    title="Precizia Estimărilor de Cost Poate Fi Îmbunătățită",
                    description=f"Precizia medie actuală: {avg_accuracy:.1%}. Ținta: >80%",
                    confidence=0.8,
                    impact_score=0.7,
                    recommended_actions=[
                        "Revizuiți metodologia de estimare",
                        "Includeți mai multe variabile în calcul",
                        "Măriți marjele de siguranță"
                    ],
                    applies_to_categories=list(set(p.project_category for p in cost_projects))
                ))
        
        # Change request pattern insight
        high_change_projects = [p for p in projects if p.change_requests_count > 3]
        if len(high_change_projects) > len(projects) * 0.3:  # More than 30% have many changes
            insights.append(IntelligenceInsight(
                insight_type="change_pattern",
                title="Frecvența Mare de Modificări Detectată",
                description=f"{len(high_change_projects)} din {len(projects)} proiecte au avut >3 modificări",
                confidence=0.7,
                impact_score=0.5,
                recommended_actions=[
                    "Clarificați cerințele în faza inițială",
                    "Implementați un proces formal de change management",
                    "Educați clienții despre impactul modificărilor"
                ],
                applies_to_categories=list(set(p.project_category for p in high_change_projects))
            ))
        
        return insights

    async def _calculate_improvement_trends(self, projects: List[ProjectLearning]) -> Dict[str, Any]:
        """Calculate improvement trends over time"""
        
        if len(projects) < 6:
            return {'insufficient_data': True}
        
        # Sort by date
        sorted_projects = sorted(projects, key=lambda p: p.recorded_at)
        
        # Split into two halves for comparison
        mid_point = len(sorted_projects) // 2
        older_half = sorted_projects[:mid_point]
        recent_half = sorted_projects[mid_point:]
        
        trends = {}
        
        # Cost accuracy trend
        older_cost_accuracy = self._calculate_average_cost_accuracy(older_half)
        recent_cost_accuracy = self._calculate_average_cost_accuracy(recent_half)
        
        if older_cost_accuracy and recent_cost_accuracy:
            cost_improvement = recent_cost_accuracy - older_cost_accuracy
            trends['cost_accuracy'] = {
                'improvement': cost_improvement,
                'direction': 'improving' if cost_improvement > 0.02 else 'declining' if cost_improvement < -0.02 else 'stable'
            }
        
        # Client satisfaction trend
        older_satisfaction = self._calculate_average_satisfaction(older_half)
        recent_satisfaction = self._calculate_average_satisfaction(recent_half)
        
        if older_satisfaction and recent_satisfaction:
            satisfaction_improvement = recent_satisfaction - older_satisfaction
            trends['client_satisfaction'] = {
                'improvement': satisfaction_improvement,
                'direction': 'improving' if satisfaction_improvement > 0.1 else 'declining' if satisfaction_improvement < -0.1 else 'stable'
            }
        
        return trends

    async def _get_category_performance(self, projects: List[ProjectLearning]) -> Dict[str, Any]:
        """Get performance metrics by project category"""
        
        category_data = {}
        
        for project in projects:
            category = project.project_category
            if category not in category_data:
                category_data[category] = []
            category_data[category].append(project)
        
        performance = {}
        for category, category_projects in category_data.items():
            if len(category_projects) >= 2:  # Need at least 2 projects for meaningful stats
                cost_accuracy = self._calculate_average_cost_accuracy(category_projects)
                satisfaction = self._calculate_average_satisfaction(category_projects)
                avg_changes = statistics.mean([p.change_requests_count for p in category_projects])
                
                performance[category] = {
                    'project_count': len(category_projects),
                    'average_cost_accuracy': cost_accuracy,
                    'average_client_satisfaction': satisfaction,
                    'average_change_requests': avg_changes,
                    'performance_score': self._calculate_category_performance_score(cost_accuracy, satisfaction, avg_changes)
                }
        
        return performance

    # Utility helper methods
    
    def _calculate_average_cost_accuracy(self, projects: List[ProjectLearning]) -> Optional[float]:
        """Calculate average cost accuracy for a list of projects"""
        accuracies = []
        for project in projects:
            if project.predicted_cost and project.actual_cost:
                accuracy = min(project.predicted_cost, project.actual_cost) / max(project.predicted_cost, project.actual_cost)
                accuracies.append(accuracy)
        return statistics.mean(accuracies) if accuracies else None

    def _calculate_average_satisfaction(self, projects: List[ProjectLearning]) -> Optional[float]:
        """Calculate average client satisfaction for a list of projects"""
        satisfactions = [p.client_satisfaction for p in projects if p.client_satisfaction]
        return statistics.mean(satisfactions) if satisfactions else None

    def _calculate_category_performance_score(
        self, 
        cost_accuracy: Optional[float], 
        satisfaction: Optional[float], 
        avg_changes: float
    ) -> float:
        """Calculate overall performance score for a category"""
        
        score_components = []
        
        if cost_accuracy:
            score_components.append(cost_accuracy)
        
        if satisfaction:
            score_components.append(satisfaction / 5.0)  # Normalize to 0-1
        
        # Inverse of change requests (fewer changes = better)
        change_score = max(0, 1.0 - (avg_changes / 8.0))  # Assuming max 8 changes
        score_components.append(change_score)
        
        return statistics.mean(score_components) if score_components else 0.5

    def _calculate_cost_variance(self, similar_projects: List[SimilarityMatch]) -> float:
        """Calculate cost variance across similar projects"""
        
        costs = []
        for match in similar_projects:
            if match.historical_project.actual_cost:
                costs.append(match.historical_project.actual_cost)
        
        if len(costs) < 2:
            return 0.0
        
        mean_cost = statistics.mean(costs)
        variance = statistics.stdev(costs) / mean_cost if mean_cost > 0 else 0.0
        return variance

    def _get_seasonal_adjustment(self) -> float:
        """Get seasonal adjustment factor based on current month"""
        
        current_month = datetime.now().month
        
        # Romanian construction seasonality
        seasonal_factors = {
            1: 1.2,   # January - winter delays
            2: 1.2,   # February - winter delays
            3: 1.1,   # March - spring transition
            4: 1.0,   # April - optimal
            5: 1.0,   # May - optimal
            6: 1.0,   # June - optimal
            7: 1.05,  # July - summer heat
            8: 1.05,  # August - summer heat
            9: 1.0,   # September - optimal
            10: 1.0,  # October - optimal
            11: 1.1,  # November - weather deteriorating
            12: 1.15  # December - winter approaching
        }
        
        return seasonal_factors.get(current_month, 1.0)

    def _generate_persona_insights(self, persona) -> List[IntelligenceInsight]:
        """Generate insights based on identified client persona"""
        
        insights = []
        
        if persona.change_frequency == "frequent":
            insights.append(IntelligenceInsight(
                insight_type="client_behavior",
                title="Client cu Tendință de Modificări Frecvente",
                description=f"Tipul de client '{persona.name}' tinde să facă modificări frecvente",
                confidence=0.7,
                impact_score=0.4,
                recommended_actions=[
                    "Stabiliți un proces clar pentru modificări",
                    "Includeți costuri pentru modificări în oferta inițială",
                    "Programați review-uri regulate pentru a preveni modificările tardive"
                ],
                applies_to_categories=[cat.value for cat in persona.typical_projects]
            ))
        
        if persona.budget_sensitivity == "high":
            insights.append(IntelligenceInsight(
                insight_type="budget_sensitivity",
                title="Client Sensibil la Preț",
                description="Clientul este foarte atent la costuri - prezentați opțiuni multiple",
                confidence=0.8,
                impact_score=0.5,
                recommended_actions=[
                    "Oferiți variante cu diferite nivele de preț",
                    "Explicați clar valoarea fiecărei opțiuni",
                    "Identificați oportunități de economii fără compromiterea calității"
                ],
                applies_to_categories=[cat.value for cat in persona.typical_projects]
            ))
        
        return insights

    async def _get_cost_accuracy_for_category(self, category: str) -> float:
        """Get historical cost accuracy for a specific category"""
        
        try:
            metric_data = await self.firestore.get_document('learning_metrics', f"{category}_cost_accuracy")
            return metric_data.get('average_score', 0.75) if metric_data else 0.75
        except:
            return 0.75  # Default accuracy

    async def _get_timeline_variance_for_category(self, category: str) -> float:
        """Get historical timeline variance for a specific category"""
        
        try:
            metric_data = await self.firestore.get_document('learning_metrics', f"{category}_timeline_variance")
            return metric_data.get('variance', 0.15) if metric_data else 0.15
        except:
            return 0.15  # Default variance

    async def _assess_learning_quality(self, category: str) -> float:
        """Assess the quality of learning data for a category"""
        
        try:
            # Get sample sizes for different metrics
            cost_metric = await self.firestore.get_document('learning_metrics', f"{category}_cost_accuracy")
            timeline_metric = await self.firestore.get_document('learning_metrics', f"{category}_timeline_accuracy")
            satisfaction_metric = await self.firestore.get_document('learning_metrics', f"{category}_client_satisfaction")
            
            sample_sizes = []
            if cost_metric:
                sample_sizes.append(cost_metric.get('sample_size', 0))
            if timeline_metric:
                sample_sizes.append(timeline_metric.get('sample_size', 0))
            if satisfaction_metric:
                sample_sizes.append(satisfaction_metric.get('sample_size', 0))
            
            if not sample_sizes:
                return 0.3  # Low quality - no data
            
            avg_sample_size = statistics.mean(sample_sizes)
            
            # Quality score based on sample size and recency
            if avg_sample_size >= 10:
                return 0.9  # High quality
            elif avg_sample_size >= 5:
                return 0.7  # Good quality
            elif avg_sample_size >= 2:
                return 0.5  # Moderate quality
            else:
                return 0.3  # Low quality
                
        except Exception as e:
            logger.error(f"❌ Error assessing learning quality: {e}")
            return 0.5  # Default moderate quality

    async def process_batch_training(self, batch_id: str) -> Dict[str, Any]:
        """Process training batch for learning improvements"""
        if not self.batch_training_enabled:
            return {"error": "Batch training not enabled"}
        
        try:
            # Get batch training data
            training_data = await self.firestore.query_training_data_by_batch(
                batch_id, confidence_threshold=0.3
            )
            
            if not training_data:
                return {"error": "No training data found for batch"}
            
            # Apply batch learning
            batch_results = await self._apply_batch_learning(training_data)
            
            # Update learning metrics
            await self._update_batch_metrics(batch_id, batch_results)
            
            logger.info(f"Batch training completed for {batch_id}: {len(training_data)} samples processed")
            
            return {
                "batch_id": batch_id,
                "samples_processed": len(training_data),
                "learning_improvements": batch_results.get("improvements", []),
                "new_patterns": batch_results.get("new_patterns", 0),
                "confidence_adjustments": batch_results.get("confidence_adjustments", {})
            }
            
        except Exception as e:
            logger.error(f"Batch training failed: {e}")
            return {"error": str(e)}

    async def _apply_batch_learning(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply learning from batch training data"""
        improvements = []
        new_patterns = 0
        confidence_adjustments = {}
        
        try:
            # Group data by document type
            dxf_data = [d for d in training_data if d.get("document_type") == "dxf"]
            pdf_data = [d for d in training_data if d.get("document_type") == "pdf"]
            txt_data = [d for d in training_data if d.get("document_type") == "txt"]
            
            # Process DXF spatial learning
            if dxf_data:
                spatial_learning = await self._learn_from_spatial_data(dxf_data)
                new_patterns += spatial_learning.get("patterns_learned", 0)
                improvements.extend(spatial_learning.get("improvements", []))
                confidence_adjustments.update(spatial_learning.get("confidence_adjustments", {}))
            
            # Process PDF specification learning
            if pdf_data:
                spec_learning = await self._learn_from_specification_data(pdf_data)
                new_patterns += spec_learning.get("patterns_learned", 0)
                improvements.extend(spec_learning.get("improvements", []))
            
            # Process text requirement learning
            if txt_data:
                text_learning = await self._learn_from_text_data(txt_data)
                new_patterns += text_learning.get("patterns_learned", 0)
                improvements.extend(text_learning.get("improvements", []))
            
            return {
                "improvements": improvements,
                "new_patterns": new_patterns,
                "confidence_adjustments": confidence_adjustments
            }
            
        except Exception as e:
            logger.error(f"Batch learning application failed: {e}")
            return {"improvements": [], "new_patterns": 0, "confidence_adjustments": {}}

    async def _learn_from_spatial_data(self, dxf_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Learn patterns from DXF spatial data"""
        patterns_learned = 0
        improvements = []
        confidence_adjustments = {}
        
        try:
            # Analyze room area patterns
            room_area_patterns = {}
            
            for data in dxf_data:
                spatial = data.get("spatial_data", {})
                rooms = spatial.get("total_rooms", 0)
                area = spatial.get("total_area", 0)
                
                if rooms > 0 and area > 0:
                    area_per_room = area / rooms
                    room_key = f"{rooms}_rooms"
                    
                    if room_key not in room_area_patterns:
                        room_area_patterns[room_key] = []
                    room_area_patterns[room_key].append(area_per_room)
            
            # Create learning patterns
            for room_config, area_values in room_area_patterns.items():
                if len(area_values) >= 3:  # Minimum samples
                    avg_area_per_room = sum(area_values) / len(area_values)
                    pattern = {
                        "pattern_type": "spatial_room_area",
                        "room_configuration": room_config,
                        "average_area_per_room": avg_area_per_room,
                        "sample_count": len(area_values),
                        "learned_at": datetime.now(timezone.utc).isoformat()
                    }
                    
                    # Save learned pattern
                    pattern_id = f"spatial_{room_config}_{datetime.now().timestamp()}"
                    await self.firestore.save_document("learned_patterns", pattern_id, pattern)
                    
                    patterns_learned += 1
                    improvements.append(f"Learned area pattern for {room_config}: {avg_area_per_room:.1f} mp/room")
                    
                    # Update confidence adjustment
                    confidence_adjustments[room_config] = min(len(area_values) / 10.0, 0.15)
            
            return {
                "patterns_learned": patterns_learned,
                "improvements": improvements,
                "confidence_adjustments": confidence_adjustments
            }
            
        except Exception as e:
            logger.error(f"Spatial data learning failed: {e}")
            return {"patterns_learned": 0, "improvements": [], "confidence_adjustments": {}}

    async def check_auto_retrain_trigger(self) -> Dict[str, Any]:
        """Check if automatic retraining should be triggered"""
        if not self.batch_training_enabled:
            return {"should_retrain": False, "reason": "Batch training disabled"}
        
        try:
            # Count recent training samples
            recent_cutoff = datetime.now(timezone.utc) - timedelta(days=7)
            recent_samples = await self.firestore.query_documents(
                "training_data",
                filters=[("saved_at", ">=", recent_cutoff)],
                limit=self.batch_config["auto_retrain_threshold"] + 10
            )
            
            should_retrain = len(recent_samples) >= self.batch_config["auto_retrain_threshold"]
            
            return {
                "should_retrain": should_retrain,
                "recent_samples_count": len(recent_samples),
                "threshold": self.batch_config["auto_retrain_threshold"],
                "reason": f"Found {len(recent_samples)} recent samples" if should_retrain else "Insufficient samples"
            }
            
        except Exception as e:
            logger.error(f"Auto retrain check failed: {e}")
            return {"should_retrain": False, "error": str(e)}

# Export for use in other modules
__all__ = [
    'LearningEngine', 
    'LearningMetric', 
    'LearningScore', 
    'ProjectLearning', 
    'IntelligenceInsight'
]