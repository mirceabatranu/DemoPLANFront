import json
import logging
from typing import Dict, Any, List

from src.services.llm_service import safe_construction_call

logger = logging.getLogger("demoplan.romanian_processor")

class RomanianProcessor:
    """
    Handles Romanian-specific language processing for the construction domain.
    This component is extracted from the legacy LLM service to be a focused processor.
    """

    def __init__(self):
        logger.info("✅ RomanianProcessor initialized.")

    async def analyze_construction_text(
        self,
        text: str,
        analysis_type: str = "general",
        domain: str = "general_construction"
    ) -> Dict[str, Any]:
        """
        Analyzes Romanian construction-related text.
        
        Args:
            text: Romanian text to analyze
            analysis_type: The type of analysis to perform (e.g., "materials", "costs").
            domain: The specific construction domain.
            
        Returns:
            A dictionary with structured analysis results.
        """
        try:
            system_prompt = f"""Ești un expert în analiza textelor românești pentru proiecte de construcții.
Analizează textul cu accent pe aspectele tehnice și oferă o structură clară a informațiilor relevante.
Tipul analizei: {analysis_type}
Domeniul: {domain}
Răspunde în format JSON structurat cu următoarele secțiuni:
- informații_identificate
- materiale_menționate  
- costuri_estimate
- specialiști_necesari
- etape_recomandate
- observații_tehnice"""

            prompt = f"""Analizează acest text românesc din domeniul construcțiilor:
{text}
Extrage și organizează toate informațiile relevante pentru un proiect de construcții din România."""

            response = await safe_construction_call(
                user_input=prompt,
                system_prompt=system_prompt,
                domain=domain,
                project_context={"analysis_type": analysis_type}
            )
            
            # Don't force JSON parsing - return structured response
            # The LLM is not reliably returning JSON, so we will parse it on the consumer side if possible
            # and provide a basic structured response here.
            structured_fallback = {
                 "status": "success",
                 "analysis_type": analysis_type,
                 "domain": domain,
                 "raw_analysis": response,
                 "structured_info": self._extract_basic_info(text)
            }
            return structured_fallback

        except Exception as e:
            logger.error(f"❌ Romanian construction text analysis failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "analysis_type": analysis_type,
                "domain": domain
            }
    
    async def generate_clarification_questions(
        self,
        missing_data: List[str],
        project_context: Dict[str, Any],
        domain: str = "general_construction"
    ) -> List[str]:
        """
        Generates targeted Romanian questions to clarify missing information.
        
        Args:
            missing_data: A list of categories where data is missing.
            project_context: The current context of the project.
            domain: The specific construction domain.
            
        Returns:
            A list of clear, specific questions in Romanian.
        """
        try:
            system_prompt = """Ești un expert în proiecte de construcții din România specializat în colectarea informațiilor necesare de la clienți.
Generează întrebări clare, specifice și prietenoase în română care să ajute la completarea informațiilor lipsă.
Întrebările trebuie să fie:
- Relevante pentru piața românească de construcții
- Ușor de înțeles pentru proprietari de locuințe
- Orientate spre soluții practice
- Conforme cu reglementările românești"""

            context_summary = json.dumps(project_context, indent=2, ensure_ascii=False)
            
            prompt = f"""Generează întrebări în română pentru completarea următoarelor informații lipsă:
{', '.join(missing_data)}
Context proiect existent:
{context_summary}
Domeniu specializat: {domain}
Generează maxim 5 întrebări clare și specifice care să faciliteze colectarea informațiilor necesare."""

            response = await safe_construction_call(
                user_input=prompt,
                system_prompt=system_prompt,
                domain=domain,
                project_context=project_context
            )
            
            questions = [q.strip() for q in response.split('\n') if '?' in q]
            return questions[:5]  # Limit to 5 questions
            
        except Exception as e:
            logger.error(f"❌ Failed to generate Romanian clarification questions: {e}")
            return [f"Puteți oferi mai multe detalii despre {missing_data[0]}?"] if missing_data else []
    
    async def estimate_construction_costs(
        self,
        project_description: str,
        region: str = "bucuresti",
        quality_level: str = "standard"
    ) -> Dict[str, Any]:
        """
        Generate Romanian construction cost estimates
        
        Args:
            project_description: Romanian description of construction project
            region: Romanian region for pricing context
            quality_level: Quality level (budget/standard/premium)
            
        Returns:
            Dict with structured cost estimation
        """
        try:
            system_prompt = f"""Ești expert în estimări de costuri pentru construcții în România.
Oferă estimări realiste bazate pe prețurile actuale din regiunea {region}.
Nivel calitate: {quality_level}

Structurează răspunsul cu:
- Costuri materiale (detaliat pe categorii)
- Costuri manoperă (pe specializări)
- Costuri servicii (proiectare, autorizații)
- Total estimat cu marje de eroare
- Durata estimată de execuție
- Recomandări pentru optimizare costuri"""

            response = await self.safe_construction_call(
                user_input=f"Estimare costuri pentru: {project_description}",
                domain="estimation",
                region=region,
                project_context={"quality_level": quality_level}
            )
            
            return {
                "status": "success",
                "region": region,
                "quality_level": quality_level,
                "estimation_response": response,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            logger.error(f"❌ Cost estimation failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "fallback_message": "Pentru o estimare precisă, recomand consultarea mai multor specialiști locali și solicitarea de oferte detaliate."
            }

    def _extract_basic_info(self, text: str) -> Dict[str, Any]:
        """Extract basic info without requiring JSON"""
        return {
            "has_area_mention": "mp" in text.lower() or "metri" in text.lower(),
            "has_rooms_mention": "camere" in text.lower() or "camera" in text.lower(),
            "has_budget_mention": "ron" in text.lower() or "euro" in text.lower(),
            "text_length": len(text)
    }