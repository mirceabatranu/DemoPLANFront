# src/services/llm_service.py
"""
DEMOPLAN Unified - Minimal LLM Service (Gemini Only)
Streamlined service for Phase 1 deployment with file context support
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional

# LLM Provider Imports
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    logging.warning("Google Generative AI not available")

from config.config import settings

logger = logging.getLogger("demoplan.llm_service")

class SafeConstructionLLMService:
    """
    Minimal LLM service for Romanian construction consultation - Gemini only
    Enhanced with file context support for multi-turn conversations
    """
    
    def __init__(self):
        self.initialized = False
        
        # Performance metrics
        self.metrics = {
            "total_requests": 0,
            "successful_responses": 0,
            "average_response_time_ms": 0,
        }
        
    async def initialize(self) -> bool:
        """Initialize LLM service with Gemini"""
        try:
            logger.info("🔧 Initializing LLM Service (Gemini only)...")
            
            if self._initialize_gemini():
                logger.info("✅ Gemini initialized successfully")
                self.initialized = True
                return True
            else:
                logger.error("❌ Gemini initialization failed")
                return False
            
        except Exception as e:
            logger.error(f"❌ LLM service initialization failed: {str(e)}")
            return False
    
    async def safe_construction_call(
        self,
        user_input: str,
        system_prompt: str = "Ești un consultant tehnic român specializat în construcții.",
        domain: str = "general_construction",
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        file_context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        Primary method: Safe construction-domain LLM call with file context support
        
        Args:
            user_input: User's message/question
            system_prompt: System instruction for the LLM
            domain: Construction domain category
            temperature: Response randomness (0.0-1.0)
            max_tokens: Maximum response length
            file_context: Optional dict containing file information for context
                {
                    "files": [{"filename": "plan.pdf", "summary": "...", "type": "pdf"}],
                    "project_data": {"area": 100, "rooms": 3},
                    "analysis_summary": "Brief description of what was analyzed"
                }
        """
        if not self.initialized:
            await self.initialize()
        
        start_time = time.time()
        self.metrics["total_requests"] += 1
        
        try:
            # Construct enhanced prompt with file context
            enhanced_prompt = self._construct_prompt_with_context(
                user_input, 
                file_context
            )
            
            # Call Gemini
            response = await self._call_gemini_construction_safe(
                enhanced_prompt, system_prompt, temperature, max_tokens
            )
            
            # Update success metrics
            self.metrics["successful_responses"] += 1
            self._update_performance_metrics(start_time)
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Safe construction call failed: {str(e)}")
            return self._get_fallback_response(domain)
    
    def _construct_prompt_with_context(
        self, 
        user_input: str, 
        file_context: Optional[Dict[str, Any]]
    ) -> str:
        """
        Construct enhanced prompt including file context
        
        This adds uploaded file information to the prompt so the LLM
        can reference what was previously uploaded during conversation
        """
        if not file_context:
            return user_input
        
        context_parts = []
        
        # Add uploaded files summary
        files = file_context.get("files", [])
        if files:
            context_parts.append("📁 **FIȘIERE ÎNCĂRCATE:**")
            for idx, file_info in enumerate(files, 1):
                filename = file_info.get("filename", "unknown")
                file_type = file_info.get("type", "unknown")
                summary = file_info.get("summary", "")
                
                file_desc = f"{idx}. {filename} ({file_type})"
                if summary:
                    file_desc += f"\n   Rezumat: {summary}"
                context_parts.append(file_desc)
            context_parts.append("")  # Empty line
        
        # Add project data if available
        project_data = file_context.get("project_data", {})
        if project_data:
            context_parts.append("🏗️ **DATE PROIECT:**")
            
            if project_data.get("total_area"):
                context_parts.append(f"- Suprafață: {project_data['total_area']} mp")
            if project_data.get("total_rooms"):
                context_parts.append(f"- Camere: {project_data['total_rooms']}")
            if project_data.get("document_type"):
                context_parts.append(f"- Tip document: {project_data['document_type']}")
            if project_data.get("has_electrical"):
                context_parts.append("- Conține instalații electrice")
            if project_data.get("has_hvac"):
                context_parts.append("- Conține sistem HVAC")
            
            # Add PDF specifications if available
            if project_data.get("pdf_specifications"):
                specs_count = len(project_data["pdf_specifications"])
                context_parts.append(f"- Specificații tehnice: {specs_count} identificate")
            
            # Add TXT requirements if available
            if project_data.get("txt_requirements"):
                req_count = len(project_data["txt_requirements"])
                context_parts.append(f"- Cerințe client: {req_count} identificate")
            
            context_parts.append("")  # Empty line
        
        # Add analysis summary if available
        analysis_summary = file_context.get("analysis_summary")
        if analysis_summary:
            context_parts.append(f"📊 **ANALIZĂ:** {analysis_summary}")
            context_parts.append("")  # Empty line
        
        # Combine context with user input
        if context_parts:
            context_text = "\n".join(context_parts)
            full_prompt = f"""{context_text}

**ÎNTREBAREA UTILIZATORULUI:**
{user_input}

**IMPORTANT:** Bazează-te pe informațiile din fișierele încărcate mai sus pentru a răspunde la această întrebare. Dacă fișierele conțin date relevante, folosește-le în răspunsul tău."""
            
            logger.debug(f"📝 Constructed prompt with file context: {len(files)} files, {len(project_data)} project fields")
            return full_prompt
        
        return user_input
    
    def _format_file_summary(self, file_info: Dict[str, Any]) -> str:
        """Format a single file's information for prompt context"""
        parts = []
        
        filename = file_info.get("filename", "unknown")
        parts.append(f"Fișier: {filename}")
        
        # File type specific formatting
        file_type = file_info.get("type", "").lower()
        
        if file_type == "dxf":
            dxf_data = file_info.get("dxf_analysis", {})
            if dxf_data:
                parts.append(f"  - Tip: Plan tehnic DXF")
                if dxf_data.get("total_rooms"):
                    parts.append(f"  - Spații: {dxf_data['total_rooms']}")
                if dxf_data.get("total_area"):
                    parts.append(f"  - Suprafață: {dxf_data['total_area']:.1f} mp")
                if dxf_data.get("document_type"):
                    parts.append(f"  - Document: {dxf_data['document_type']}")
        
        elif file_type == "pdf":
            pdf_data = file_info.get("pdf_analysis", {})
            if pdf_data:
                parts.append(f"  - Tip: Document PDF")
                if pdf_data.get("construction_specs"):
                    parts.append(f"  - Specificații: {len(pdf_data['construction_specs'])}")
                if pdf_data.get("material_references"):
                    parts.append(f"  - Materiale: {len(pdf_data['material_references'])}")
        
        elif file_type == "txt":
            txt_data = file_info.get("txt_analysis", {})
            if txt_data:
                parts.append(f"  - Tip: Text")
                if txt_data.get("requirements"):
                    parts.append(f"  - Cerințe: {len(txt_data['requirements'])}")
                if txt_data.get("client_preferences"):
                    parts.append(f"  - Preferințe: {len(txt_data['client_preferences'])}")
        
        return "\n".join(parts)
    
    async def _call_gemini_construction_safe(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: Optional[int]
    ) -> str:
        """Call Gemini with construction-optimized safety settings"""
        
        if not GENAI_AVAILABLE:
            raise Exception("Gemini not available")
        
        model = genai.GenerativeModel(
            model_name=settings.gemini_model,
            system_instruction=system_prompt
        )
        
        # Construction-optimized generation config
        generation_config = {
            "temperature": temperature,
            "candidate_count": 1,
        }
        
        if max_tokens:
            generation_config["max_output_tokens"] = max_tokens
        
        # Permissive safety settings for professional construction consultation
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
        ]
        
        # The `contents` should be a list. For a single prompt, it's [prompt].
        response = await model.generate_content_async(
            [prompt],
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        # Enhanced response extraction with construction context
        if response.candidates and response.candidates[0].content:
            parts = response.candidates[0].content.parts
            if parts:
                return parts[0].text
        
        # Handle various finish reasons
        if response.candidates:
            finish_reason = response.candidates[0].finish_reason
            return self._handle_gemini_finish_reason(finish_reason)
        
        return "Răspunsul nu este disponibil momentan din cauza restricțiilor tehnice."
    
    def _handle_gemini_finish_reason(self, finish_reason: int) -> str:
        """Handle Gemini finish reasons with construction context"""
        
        reason_messages = {
            2: "Din motive de siguranță, nu pot genera acest tip de răspuns. Vă rugăm să reformulați cererea în contextul consultanței tehnice pentru construcții.",
            3: "Răspunsul conține conținut protejat de drepturi de autor. Vă rugăm să încercați din nou cu o cerere mai specifică pentru consultanța în construcții.",
            4: "Răspunsul nu a putut fi completat din motive tehnice. Pentru consultanță detaliată în construcții, recomand contactarea directă a unui specialist autorizat.",
            5: "Răspunsul a fost oprit din cauza limitărilor de lungime. Pentru consultanță completă în construcții, vă rugăm să împărțiți cererea în întrebări mai specifice."
        }
        
        return reason_messages.get(
            finish_reason, 
            "Răspunsul nu este disponibil momentan. Pentru consultanță în construcții, recomand contactarea unui specialist autorizat."
        )
    
    def _initialize_gemini(self) -> bool:
        """Initialize Gemini with construction-safe configuration"""
        if not GENAI_AVAILABLE or not settings.gemini_api_key:
            return False
        
        try:
            genai.configure(api_key=settings.gemini_api_key)
            return True
        except Exception as e:
            logger.error(f"❌ Gemini initialization failed: {str(e)}")
            return False
    
    def _get_fallback_response(self, domain: str) -> str:
        """Get fallback response by domain"""
        fallbacks = {
            "electrical": "Pentru instalații electrice, consultați un electrician autorizat ANRE conform normativelor românești.",
            "plumbing": "Pentru instalații sanitare, recomand consultarea unui specialist conform normativelor românești.",
            "structural": "Pentru modificări structurale, este obligatorie consultarea unui inginer constructor autorizat.",
            "estimation": "Pentru estimări precise de costuri, recomand solicitarea de oferte de la mai mulți specialiști locali."
        }
        
        return fallbacks.get(domain, "Pentru consultanță tehnică detaliată în construcții, vă recomand contactarea unui specialist autorizat român.")
    
    def _update_performance_metrics(self, start_time: float):
        """Update performance metrics"""
        response_time = int((time.time() - start_time) * 1000)
        
        # Update average response time
        current_avg = self.metrics["average_response_time_ms"]
        total_requests = self.metrics["total_requests"]
        
        self.metrics["average_response_time_ms"] = int(
            (current_avg * (total_requests - 1) + response_time) / total_requests
        )
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status"""
        return {
            "service_name": "SafeConstructionLLMService",
            "version": "1.0.1-file-context",
            "initialized": self.initialized,
            "providers": {
                "gemini_available": GENAI_AVAILABLE and bool(settings.gemini_api_key),
            },
            "performance_metrics": self.get_performance_metrics()
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        total_requests = self.metrics["total_requests"]
        
        if total_requests == 0:
            return {
                "total_requests": 0,
                "success_rate": 0.0,
                "average_response_time_ms": 0,
            }
        
        return {
            "total_requests": total_requests,
            "successful_responses": self.metrics["successful_responses"],
            "success_rate": round(self.metrics["successful_responses"] / total_requests * 100, 2),
            "average_response_time_ms": self.metrics["average_response_time_ms"],
        }

# Global service instance
safe_construction_llm_service = SafeConstructionLLMService()

# Direct utility function for easy integration
async def safe_construction_call(
    user_input: str,
    domain: str = "general_construction",
    system_prompt: str = "Ești un asistent AI specializat în construcții.",
    file_context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> str:
    """
    Direct utility function for safe construction LLM calls with file context support
    
    Args:
        user_input: User's message
        domain: Construction domain
        system_prompt: System instruction
        file_context: Optional file context dict
        **kwargs: Additional parameters
    """
    return await safe_construction_llm_service.safe_construction_call(
        user_input=user_input,
        domain=domain,
        system_prompt=system_prompt,
        file_context=file_context,
        **kwargs
    )

logger.info("✅ Safe Construction LLM Service - File Context Support Loaded (Gemini Only)")