"""
DemoPLAN Intelligence Module - Phase 2 Full Version
Sophisticated ML-powered intelligence with graceful fallbacks
"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger("demoplan.intelligence")

# Import sophisticated modules with fallback handling
try:
    from .learning_engine import (
        LearningEngine, 
        LearningMetric, 
        LearningScore, 
        ProjectLearning, 
        IntelligenceInsight
    )
    LEARNING_ENGINE_AVAILABLE = True
    logger.info("✅ Advanced Learning Engine loaded")
except ImportError as e:
    logger.warning(f"⚠️ Learning Engine import failed: {e}")
    LEARNING_ENGINE_AVAILABLE = False
    # Create fallback classes
    from enum import Enum
    from dataclasses import dataclass
    from datetime import datetime
    
    class LearningMetric(Enum):
        COST_ACCURACY = "cost_accuracy"
        TIMELINE_ACCURACY = "timeline_accuracy"
        CLIENT_SATISFACTION = "client_satisfaction"
    
    @dataclass
    class LearningScore:
        metric: LearningMetric
        score: float
        confidence: float
        sample_size: int
        trend: str
    
    @dataclass 
    class ProjectLearning:
        session_id: str
        project_category: str
        predicted_cost: Optional[float] = None
    
    @dataclass
    class IntelligenceInsight:
        insight_type: str
        title: str
        description: str
        confidence: float

try:
    from .pattern_matcher import (
        PatternMatcher, 
        ProjectCategory, 
        RequirementPattern, 
        ClientPersona, 
        PatternMatch
    )
    PATTERN_MATCHER_AVAILABLE = True
    logger.info("✅ Advanced Pattern Matcher loaded")
except ImportError as e:
    logger.warning(f"⚠️ Pattern Matcher import failed: {e}")
    PATTERN_MATCHER_AVAILABLE = False
    from enum import Enum
    from dataclasses import dataclass
    
    class ProjectCategory(Enum):
        RESIDENTIAL_APARTMENT = "apartament_rezidential"
        UNKNOWN = "necunoscut"
    
    @dataclass
    class PatternMatch:
        project_category: ProjectCategory
        confidence: float
        matching_patterns: List[str]

try:
    from .historical_analyzer import HistoricalAnalyzer
    HISTORICAL_ANALYZER_AVAILABLE = True
    logger.info("✅ Advanced Historical Analyzer loaded")
except ImportError as e:
    logger.warning(f"⚠️ Historical Analyzer import failed: {e}")
    HISTORICAL_ANALYZER_AVAILABLE = False

# Full Intelligence Manager for Phase 2
class IntelligenceManager:
    """Phase 2 intelligence manager with full capabilities"""
    
    def __init__(self):
        self.learning_engine = None
        self.pattern_matcher = None
        self.historical_analyzer = None
        self.initialized = False
        
    async def initialize(self, enable_ml: bool = True):
        """Initialize all available intelligence components"""
        try:
            if enable_ml and LEARNING_ENGINE_AVAILABLE:
                self.learning_engine = LearningEngine()
                await self.learning_engine.initialize()
                logger.info("✅ Learning Engine initialized")
            
            if enable_ml and PATTERN_MATCHER_AVAILABLE:
                self.pattern_matcher = PatternMatcher()
                await self.pattern_matcher.initialize()
                logger.info("✅ Pattern Matcher initialized")
            
            if enable_ml and HISTORICAL_ANALYZER_AVAILABLE:
                self.historical_analyzer = HistoricalAnalyzer()
                await self.historical_analyzer.initialize()
                logger.info("✅ Historical Analyzer initialized")
            
            self.initialized = True
            
        except Exception as e:
            logger.error(f"❌ Intelligence initialization error: {e}")
            self.initialized = True
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "learning_engine": self.learning_engine is not None,
            "pattern_matcher": self.pattern_matcher is not None,
            "historical_analyzer": self.historical_analyzer is not None,
            "modules_available": {
                "learning_engine": LEARNING_ENGINE_AVAILABLE,
                "pattern_matcher": PATTERN_MATCHER_AVAILABLE,
                "historical_analyzer": HISTORICAL_ANALYZER_AVAILABLE
            },
            "phase": 2,
            "initialized": self.initialized
        }

# Create global intelligence manager
intelligence_manager = IntelligenceManager()

# Export all
__all__ = [
    'LearningEngine',
    'LearningMetric',
    'PatternMatcher',
    'ProjectCategory',
    'PatternMatch',
    'HistoricalAnalyzer',
    'IntelligenceManager',
    'intelligence_manager',
    'LEARNING_ENGINE_AVAILABLE',
    'PATTERN_MATCHER_AVAILABLE', 
    'HISTORICAL_ANALYZER_AVAILABLE'
]