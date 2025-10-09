"""
DemoPLAN Utilities Module
Basic utilities for the unified agent
"""

try:
    from .safety_handler import (
        construction_safety_handler,
        ConstructionDomain,
        SafePromptResult,
        SafetyRejectionType,
        is_construction_safe_response,
        get_romanian_construction_fallback
    )
    SAFETY_AVAILABLE = True
except ImportError:
    SAFETY_AVAILABLE = False
    # Create dummy classes to prevent import errors
    
    class ConstructionDomain:
        GENERAL = "general"
        ELECTRICAL = "electrical"
        PLUMBING = "plumbing"
    
    class SafePromptResult:
        def __init__(self, success=True, safe_prompt="", confidence=0.8, domain=None):
            self.success = success
            self.safe_prompt = safe_prompt
            self.confidence = confidence
            self.domain = domain
    
    construction_safety_handler = None
    
    def is_construction_safe_response(response):
        return True
    
    def get_romanian_construction_fallback(domain):
        return "Consultați un specialist în construcții."

__all__ = [
    'construction_safety_handler',
    'ConstructionDomain', 
    'SafePromptResult',
    'is_construction_safe_response',
    'get_romanian_construction_fallback',
    'SAFETY_AVAILABLE'
]