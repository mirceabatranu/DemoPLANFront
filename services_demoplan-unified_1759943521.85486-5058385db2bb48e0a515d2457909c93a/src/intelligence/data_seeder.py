"""
Intelligence Data Seeder - Phase 2
Seeds initial data for ML components
"""

import logging
from typing import Dict, List, Any
from datetime import datetime, timezone, timedelta
import random

from src.services.firestore_service import FirestoreService

logger = logging.getLogger("demoplan.intelligence.data_seeder")

class IntelligenceDataSeeder:
    """Seeds initial training data for intelligence modules"""
    
    def __init__(self):
        self.firestore = FirestoreService()
        
    async def initialize(self):
        """Initialize the data seeder"""
        await self.firestore.initialize()
        logger.info("✅ Intelligence Data Seeder initialized")
    
    async def seed_all_intelligence_data(self) -> Dict[str, int]:
        """Seed all required collections with initial data"""
        results = {
            "learning_metrics": 0,
            "project_learning": 0,
            "pattern_templates": 0
        }
        
        try:
            # Check if seeding is needed
            if await self._check_if_seeding_needed():
                logger.info("🌱 Starting intelligence data seeding...")
                
                results["learning_metrics"] = await self._seed_learning_metrics()
                results["project_learning"] = await self._seed_project_learning_data()
                results["pattern_templates"] = await self._seed_pattern_templates()
                
                logger.info(f"✅ Intelligence data seeding completed: {results}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Intelligence data seeding failed: {e}")
            return results
    
    async def _check_if_seeding_needed(self) -> bool:
        """Check if seeding is needed"""
        try:
            learning_docs = await self.firestore.query_documents("learning_metrics", limit=3)
            return len(learning_docs) < 2
        except:
            return True
    
    async def _seed_learning_metrics(self) -> int:
        """Seed baseline learning metrics"""
        try:
            metrics = {
                "residential_cost_accuracy": {
                    "metric": "cost_accuracy",
                    "category": "residential",
                    "average_score": 0.78,
                    "sample_size": 12,
                    "trend": "improving",
                    "last_updated": datetime.now(timezone.utc)
                },
                "residential_timeline_accuracy": {
                    "metric": "timeline_accuracy",
                    "category": "residential",
                    "average_score": 0.73,
                    "sample_size": 10,
                    "trend": "stable",
                    "last_updated": datetime.now(timezone.utc)
                }
            }
            
            saved = 0
            for metric_id, data in metrics.items():
                if await self.firestore.save_document("learning_metrics", metric_id, data):
                    saved += 1
                    
            return saved
        except:
            return 0
    
    async def _seed_project_learning_data(self) -> int:
        """Seed sample project data"""
        try:
            projects = {}
            for i in range(10):
                project_data = {
                    "session_id": f"seed_{i:03d}",
                    "project_category": random.choice(["apartament_renovation", "casa_renovation"]),
                    "area_sqm": random.randint(50, 120),
                    "predicted_cost": random.randint(40000, 80000),
                    "actual_cost": random.randint(45000, 90000),
                    "predicted_timeline": random.randint(20, 40),
                    "actual_timeline": random.randint(25, 50),
                    "client_satisfaction": random.uniform(3.5, 5.0),
                    "recorded_at": datetime.now(timezone.utc) - timedelta(days=random.randint(30, 180))
                }
                projects[f"seed_{i:03d}"] = project_data
            
            return await self.firestore.batch_save_documents("project_learning", projects)
        except:
            return 0
    
    async def _seed_pattern_templates(self) -> int:
        """Seed pattern templates"""
        try:
            templates = {
                "apartament_standard": {
                    "pattern_type": "residential_apartment",
                    "area_range": [50, 100],
                    "average_cost_per_sqm": 850,
                    "average_timeline_days": 35,
                    "success_rate": 0.82,
                    "created_at": datetime.now(timezone.utc)
                }
            }
            
            saved = 0
            for template_id, data in templates.items():
                if await self.firestore.save_document("pattern_templates", template_id, data):
                    saved += 1
                    
            return saved
        except:
            return 0