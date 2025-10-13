"""
Startup Service for DemoPLAN Phase 2
Handles initialization and data seeding
"""

import logging
from src.intelligence.data_seeder import IntelligenceDataSeeder
from src.intelligence import intelligence_manager

logger = logging.getLogger("demoplan.services.startup")

class StartupService:
    """Manages Phase 2 startup sequence"""
    
    def __init__(self):
        self.initialization_status = {
            "intelligence_manager": False,
            "data_seeding": False,
            "overall_ready": False
        }
        
    async def initialize_phase2_system(self) -> Dict[str, Any]:
        """Initialize Phase 2 system with intelligence"""
        logger.info("🚀 Starting Phase 2 initialization...")
        
        try:
            # Initialize intelligence
            await intelligence_manager.initialize(enable_ml=True)
            self.initialization_status["intelligence_manager"] = True
            
            # Seed data
            seeder = IntelligenceDataSeeder()
            await seeder.initialize()
            await seeder.seed_all_intelligence_data()
            self.initialization_status["data_seeding"] = True
            
            self.initialization_status["overall_ready"] = True
            
            return {
                "system_ready": True,
                "intelligence_status": intelligence_manager.get_status(),
                "phase": 2
            }
            
        except Exception as e:
            logger.error(f"❌ Phase 2 initialization failed: {e}")
            return {"system_ready": False, "error": str(e)}

startup_service = StartupService()

async def initialize_phase2_system():
    """Initialize Phase 2 system"""
    return await startup_service.initialize_phase2_system()