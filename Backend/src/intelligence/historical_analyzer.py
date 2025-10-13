# src/services/historical_analyzer.py
"""
DemoPLAN Historical Project Analysis Service
Basic project comparison and pattern recognition for Romanian construction projects
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
import json
import math

from src.services.firestore_service import FirestoreService
from config.config import settings

logger = logging.getLogger("demoplan.services.historical_analyzer")

@dataclass
class ProjectProfile:
    """Basic project characteristics for comparison"""
    session_id: str
    project_type: str  # "residential", "commercial", "industrial"
    area_sqm: Optional[float] = None
    rooms_count: Optional[int] = None
    mep_complexity: str = "medium"  # "low", "medium", "high"
    construction_year: Optional[int] = None
    location_type: str = "urban"  # "urban", "suburban", "rural" 
    estimated_cost: Optional[float] = None
    actual_cost: Optional[float] = None
    completion_time_days: Optional[int] = None
    confidence_score: Optional[float] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class SimilarityMatch:
    """Project similarity comparison result"""
    historical_project: ProjectProfile
    similarity_score: float  # 0.0 to 1.0
    matching_factors: List[str]
    cost_accuracy: Optional[float] = None  # How accurate the historical estimate was

class HistoricalAnalyzer:
    """
    Analyzes historical project data to improve current project estimates
    Simple pattern matching and cost accuracy tracking
    """
    
    def __init__(self):
        self.firestore = FirestoreService()
        self.similarity_weights = {
            'project_type': 0.30,      # Most important factor
            'area_similarity': 0.25,    # Size is critical for cost
            'mep_complexity': 0.20,     # Technical complexity
            'rooms_similarity': 0.15,   # Layout complexity
            'location_type': 0.10       # Market conditions
        }
        self.min_similarity_threshold = 0.6  # Minimum for useful comparison
        
    async def initialize(self):
        """Initialize the historical analyzer"""
        try:
            await self.firestore.initialize()
            logger.info("✅ Historical Analyzer initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Historical Analyzer: {e}")
            raise

    async def analyze_project_similarity(
        self, 
        current_project: Dict[str, Any]
    ) -> List[SimilarityMatch]:
        """
        Find similar historical projects for the current project
        
        Args:
            current_project: Current project data from agents
            
        Returns:
            List of similar projects with similarity scores
        """
        try:
            # Create profile for current project
            current_profile = self._create_project_profile(current_project)
            
            # Get historical projects
            historical_projects = await self._get_historical_projects()
            
            if not historical_projects:
                logger.info("📊 No historical projects found for comparison")
                return []
            
            # Calculate similarities
            similarities = []
            for historical in historical_projects:
                similarity = self._calculate_project_similarity(current_profile, historical)
                if similarity.similarity_score >= self.min_similarity_threshold:
                    similarities.append(similarity)
            
            # Sort by similarity score (descending)
            similarities.sort(key=lambda x: x.similarity_score, reverse=True)
            
            # Return top 5 matches
            top_matches = similarities[:5]
            
            logger.info(f"📈 Found {len(top_matches)} similar projects (>{self.min_similarity_threshold:.1f} similarity)")
            
            return top_matches
            
        except Exception as e:
            logger.error(f"❌ Error analyzing project similarity: {e}")
            return []

    async def calculate_cost_accuracy_trends(
        self, 
        project_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate historical cost estimation accuracy trends
        
        Args:
            project_type: Filter by project type (optional)
            
        Returns:
            Cost accuracy analysis data
        """
        try:
            # Get completed projects with both estimated and actual costs
            completed_projects = await self._get_completed_projects(project_type)
            
            if len(completed_projects) < 3:
                return {
                    'average_accuracy': 0.0,
                    'projects_analyzed': len(completed_projects),
                    'trend': 'insufficient_data',
                    'recommendations': ['Need more historical data for accurate analysis']
                }
            
            # Calculate accuracy metrics
            accuracies = []
            cost_overruns = []
            
            for project in completed_projects:
                if project.estimated_cost and project.actual_cost and project.estimated_cost > 0:
                    accuracy = min(project.estimated_cost, project.actual_cost) / max(project.estimated_cost, project.actual_cost)
                    accuracies.append(accuracy)
                    
                    overrun_pct = ((project.actual_cost - project.estimated_cost) / project.estimated_cost) * 100
                    cost_overruns.append(overrun_pct)
            
            if not accuracies:
                return {
                    'average_accuracy': 0.0,
                    'projects_analyzed': 0,
                    'trend': 'no_valid_data'
                }
            
            # Calculate metrics
            avg_accuracy = sum(accuracies) / len(accuracies)
            avg_overrun = sum(cost_overruns) / len(cost_overruns)
            
            # Determine trend
            recent_projects = [p for p in completed_projects if (datetime.now(timezone.utc) - p.created_at).days <= 180]
            if len(recent_projects) >= 2:
                recent_accuracies = []
                for project in recent_projects:
                    if project.estimated_cost and project.actual_cost and project.estimated_cost > 0:
                        accuracy = min(project.estimated_cost, project.actual_cost) / max(project.estimated_cost, project.actual_cost)
                        recent_accuracies.append(accuracy)
                
                recent_avg = sum(recent_accuracies) / len(recent_accuracies) if recent_accuracies else avg_accuracy
                trend = "improving" if recent_avg > avg_accuracy else "stable" if abs(recent_avg - avg_accuracy) < 0.05 else "declining"
            else:
                trend = "insufficient_recent_data"
            
            return {
                'average_accuracy': round(avg_accuracy, 3),
                'average_cost_overrun_pct': round(avg_overrun, 1),
                'projects_analyzed': len(completed_projects),
                'trend': trend,
                'recommendations': self._generate_accuracy_recommendations(avg_accuracy, avg_overrun, trend)
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating cost accuracy trends: {e}")
            return {'error': str(e)}

    async def get_historical_cost_estimate(
        self, 
        similar_projects: List[SimilarityMatch],
        current_project_area: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate cost estimate based on similar historical projects
        
        Args:
            similar_projects: List of similar project matches
            current_project_area: Current project area for scaling
            
        Returns:
            Historical cost estimate with confidence
        """
        try:
            if not similar_projects:
                return {
                    'estimated_cost': None,
                    'confidence': 0.0,
                    'basis': 'no_historical_data',
                    'note': 'Nu există proiecte similare în istoric pentru comparație'
                }
            
            # Weight estimates by similarity score
            weighted_estimates = []
            total_weight = 0.0
            
            for match in similar_projects:
                if match.historical_project.actual_cost:
                    cost = match.historical_project.actual_cost
                    weight = match.similarity_score
                    
                    # Scale by area if both areas available
                    if (current_project_area and 
                        match.historical_project.area_sqm and 
                        match.historical_project.area_sqm > 0):
                        cost_per_sqm = cost / match.historical_project.area_sqm
                        scaled_cost = cost_per_sqm * current_project_area
                        weighted_estimates.append(scaled_cost * weight)
                    else:
                        weighted_estimates.append(cost * weight)
                    
                    total_weight += weight
            
            if not weighted_estimates or total_weight == 0:
                return {
                    'estimated_cost': None,
                    'confidence': 0.0,
                    'basis': 'no_cost_data',
                    'note': 'Proiectele similare nu au date complete de cost'
                }
            
            # Calculate weighted average
            historical_estimate = sum(weighted_estimates) / total_weight
            
            # Calculate confidence based on number of matches and their similarity
            base_confidence = min(len(similar_projects) * 0.15, 0.75)  # Max 75% from quantity
            similarity_bonus = sum(match.similarity_score for match in similar_projects[:3]) / 3 * 0.25  # Up to 25% from quality
            confidence = min(base_confidence + similarity_bonus, 0.95)
            
            return {
                'estimated_cost': round(historical_estimate, 2),
                'confidence': round(confidence, 3),
                'basis': f'{len(similar_projects)} proiecte similare',
                'similar_projects_used': len(similar_projects),
                'average_similarity': round(sum(match.similarity_score for match in similar_projects) / len(similar_projects), 3),
                'note': f'Estimare bazată pe {len(similar_projects)} proiecte cu similaritate {round(sum(match.similarity_score for match in similar_projects) / len(similar_projects) * 100, 1)}%'
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating historical cost estimate: {e}")
            return {
                'estimated_cost': None,
                'confidence': 0.0,
                'error': str(e)
            }

    async def save_project_outcome(
        self,
        session_id: str,
        actual_cost: float,
        completion_time_days: int,
        client_satisfaction: Optional[float] = None
    ) -> bool:
        """
        Save actual project outcome for future analysis
        
        Args:
            session_id: Project session ID
            actual_cost: Final project cost
            completion_time_days: Actual completion time
            client_satisfaction: Client satisfaction score (1-5)
            
        Returns:
            Success status
        """
        try:
            outcome_data = {
                'session_id': session_id,
                'actual_cost': actual_cost,
                'completion_time_days': completion_time_days,
                'client_satisfaction': client_satisfaction,
                'recorded_at': datetime.now(timezone.utc),
                'source': 'historical_analyzer'
            }
            
            success = await self.firestore.save_document(
                collection='project_outcomes',
                document_id=session_id,
                data=outcome_data
            )
            
            if success:
                logger.info(f"✅ Project outcome saved for session {session_id}")
            else:
                logger.error(f"❌ Failed to save project outcome for session {session_id}")
                
            return success
            
        except Exception as e:
            logger.error(f"❌ Error saving project outcome: {e}")
            return False

    # Private helper methods
    
    def _create_project_profile(self, project_data: Dict[str, Any]) -> ProjectProfile:
        """Create a project profile from agent data"""
        
        # Extract basic data
        session_id = project_data.get('session_id', 'unknown')
        
        # Try to determine project type from floorplan data
        project_type = "residential"  # Default
        if project_data.get('floorplan_analysis'):
            rooms = project_data['floorplan_analysis'].get('rooms', [])
            if any('office' in room.get('type', '').lower() for room in rooms):
                project_type = "commercial"
            elif any('warehouse' in room.get('type', '').lower() for room in rooms):
                project_type = "industrial"
        
        # Extract area and room count
        area_sqm = None
        rooms_count = None
        if project_data.get('floorplan_analysis'):
            area_sqm = project_data['floorplan_analysis'].get('total_area')
            rooms_count = len(project_data['floorplan_analysis'].get('rooms', []))
        
        # Determine MEP complexity
        mep_complexity = "medium"
        if project_data.get('mep_systems'):
            system_count = len(project_data['mep_systems'])
            if system_count <= 2:
                mep_complexity = "low"
            elif system_count >= 5:
                mep_complexity = "high"
        
        # Extract estimated cost
        estimated_cost = None
        if project_data.get('estimation_results'):
            estimated_cost = project_data['estimation_results'].get('total_cost')
        
        return ProjectProfile(
            session_id=session_id,
            project_type=project_type,
            area_sqm=area_sqm,
            rooms_count=rooms_count,
            mep_complexity=mep_complexity,
            estimated_cost=estimated_cost,
            location_type="urban"  # Default - could be enhanced with user input
        )

    async def _get_historical_projects(self) -> List[ProjectProfile]:
        """Get historical projects from database"""
        try:
            # Query multiple collections for historical data
            projects = []
            
            # Get completed projects from offers collection
            offer_docs = await self.firestore.query_documents(
                collection='offers',
                filters=[('status', '==', 'completed')],
                limit=50  # Reasonable limit for analysis
            )
            
            for doc in offer_docs:
                try:
                    profile = self._doc_to_project_profile(doc)
                    if profile:
                        projects.append(profile)
                except Exception as e:
                    logger.debug(f"Skipping invalid historical project: {e}")
                    continue
            
            return projects
            
        except Exception as e:
            logger.error(f"❌ Error getting historical projects: {e}")
            return []

    async def _get_completed_projects(self, project_type: Optional[str] = None) -> List[ProjectProfile]:
        """Get completed projects with actual cost data"""
        try:
            filters = [('status', '==', 'completed')]
            if project_type:
                filters.append(('project_type', '==', project_type))
            
            docs = await self.firestore.query_documents(
                collection='project_outcomes',
                filters=filters,
                limit=30
            )
            
            projects = []
            for doc in docs:
                try:
                    profile = self._doc_to_project_profile(doc)
                    if profile and profile.actual_cost:
                        projects.append(profile)
                except Exception as e:
                    continue
                    
            return projects
            
        except Exception as e:
            logger.error(f"❌ Error getting completed projects: {e}")
            return []

    def _doc_to_project_profile(self, doc: Dict[str, Any]) -> Optional[ProjectProfile]:
        """Convert Firestore document to ProjectProfile"""
        try:
            return ProjectProfile(
                session_id=doc.get('session_id', 'unknown'),
                project_type=doc.get('project_type', 'residential'),
                area_sqm=doc.get('area_sqm'),
                rooms_count=doc.get('rooms_count'),
                mep_complexity=doc.get('mep_complexity', 'medium'),
                construction_year=doc.get('construction_year'),
                location_type=doc.get('location_type', 'urban'),
                estimated_cost=doc.get('estimated_cost'),
                actual_cost=doc.get('actual_cost'),
                completion_time_days=doc.get('completion_time_days'),
                confidence_score=doc.get('confidence_score'),
                created_at=doc.get('created_at', datetime.now(timezone.utc))
            )
        except Exception as e:
            logger.debug(f"Invalid project profile data: {e}")
            return None

    def _calculate_project_similarity(
        self, 
        current: ProjectProfile, 
        historical: ProjectProfile
    ) -> SimilarityMatch:
        """Calculate similarity between two projects"""
        
        similarity_components = {}
        matching_factors = []
        
        # Project type similarity (exact match)
        if current.project_type == historical.project_type:
            similarity_components['project_type'] = 1.0
            matching_factors.append(f"Același tip: {current.project_type}")
        else:
            similarity_components['project_type'] = 0.0
        
        # Area similarity (if both available)
        if current.area_sqm and historical.area_sqm:
            area_diff = abs(current.area_sqm - historical.area_sqm) / max(current.area_sqm, historical.area_sqm)
            area_similarity = max(0, 1 - area_diff)
            similarity_components['area_similarity'] = area_similarity
            if area_similarity > 0.8:
                matching_factors.append(f"Suprafață similară: {current.area_sqm:.0f}m² vs {historical.area_sqm:.0f}m²")
        else:
            similarity_components['area_similarity'] = 0.3  # Neutral when data missing
        
        # MEP complexity similarity
        complexity_map = {'low': 1, 'medium': 2, 'high': 3}
        current_complexity = complexity_map.get(current.mep_complexity, 2)
        historical_complexity = complexity_map.get(historical.mep_complexity, 2)
        
        complexity_diff = abs(current_complexity - historical_complexity) / 2  # Max diff is 2
        mep_similarity = 1 - complexity_diff
        similarity_components['mep_complexity'] = mep_similarity
        
        if mep_similarity > 0.8:
            matching_factors.append(f"Complexitate MEP similară: {current.mep_complexity}")
        
        # Rooms similarity (if both available)
        if current.rooms_count and historical.rooms_count:
            rooms_diff = abs(current.rooms_count - historical.rooms_count) / max(current.rooms_count, historical.rooms_count)
            rooms_similarity = max(0, 1 - rooms_diff)
            similarity_components['rooms_similarity'] = rooms_similarity
            if rooms_similarity > 0.8:
                matching_factors.append(f"Număr camere similar: {current.rooms_count} vs {historical.rooms_count}")
        else:
            similarity_components['rooms_similarity'] = 0.3  # Neutral when data missing
        
        # Location type similarity
        if current.location_type == historical.location_type:
            similarity_components['location_type'] = 1.0
            matching_factors.append(f"Aceeași locație: {current.location_type}")
        else:
            similarity_components['location_type'] = 0.5  # Partial match for different locations
        
        # Calculate weighted overall similarity
        overall_similarity = sum(
            similarity_components[factor] * weight 
            for factor, weight in self.similarity_weights.items() 
            if factor in similarity_components
        )
        
        # Calculate cost accuracy if available
        cost_accuracy = None
        if (historical.estimated_cost and 
            historical.actual_cost and 
            historical.estimated_cost > 0):
            cost_accuracy = min(historical.estimated_cost, historical.actual_cost) / max(historical.estimated_cost, historical.actual_cost)
        
        return SimilarityMatch(
            historical_project=historical,
            similarity_score=overall_similarity,
            matching_factors=matching_factors,
            cost_accuracy=cost_accuracy
        )

    def _generate_accuracy_recommendations(
        self, 
        avg_accuracy: float, 
        avg_overrun: float, 
        trend: str
    ) -> List[str]:
        """Generate recommendations based on cost accuracy analysis"""
        
        recommendations = []
        
        if avg_accuracy < 0.7:
            recommendations.append("Precizia estimărilor este sub nivel optim - considerați mărirea marjelor de siguranță")
            
        if avg_overrun > 15:
            recommendations.append("Depășiri frecvente de buget - analizați factorii de risc suplimentari")
            
        if trend == "declining":
            recommendations.append("Tendință descrescătoare în precizie - revizuiți procesul de estimare")
        elif trend == "improving":
            recommendations.append("Progres pozitiv în precizia estimărilor - continuați strategia actuală")
            
        if not recommendations:
            recommendations.append("Performanță bună în estimări - monitorizați în continuare pentru optimizări")
            
        return recommendations

# Export for use in other modules
__all__ = ['HistoricalAnalyzer', 'ProjectProfile', 'SimilarityMatch']