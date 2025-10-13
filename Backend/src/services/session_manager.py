# src/services/session_manager.py
"""
DemoPLAN Unified - Session Manager
Streamlined session management for the unified agent architecture.
"""

import uuid
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional

from dataclasses import dataclass, field

from src.services.firestore_service import FirestoreService

logger = logging.getLogger("demoplan.services.session_manager")

# =============================================================================
# SESSION DATA MODELS
# =============================================================================

@dataclass
class UnifiedSession:
    """
    Represents the metadata of a single, unified consultation session.
    Large data like conversation history and file analyses are stored in Firestore subcollections.
    """
    session_id: str
    created_at: datetime
    last_activity: datetime
    status: str = "active"  # active, expired, completed
    confidence_score: float = 0.0
    can_generate_offer: bool = False

    # The following fields are loaded from subcollections on-demand
    # and are NOT part of the main session document persisted in Firestore.
    # They are included here for type hinting and in-memory representation during a request.
    conversation: List[Dict[str, Any]] = field(default_factory=list, repr=False)
    files: List[Dict[str, Any]] = field(default_factory=list, repr=False)

# =============================================================================
# SESSION MANAGER
# =============================================================================

class SessionManager:
    """Manages the lifecycle of consultation sessions."""
    
    def __init__(self, firestore_service: FirestoreService):
        self.firestore = firestore_service
        self.active_sessions: Dict[str, UnifiedSession] = {}
        self.session_timeout_hours: int = 24
        
    async def create_session(self) -> UnifiedSession:
        """Creates a new, empty session with only metadata."""
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        
        session = UnifiedSession(
            session_id=session_id,
            created_at=now,
            last_activity=now,
            status="active",
            confidence_score=0.0,
            can_generate_offer=False
        )
        
        self.active_sessions[session_id] = session
        await self._persist_session(session)
        
        logger.info(f"✅ Created new session: {session_id}")
        return session
    
    async def get_session(self, session_id: str) -> Optional[UnifiedSession]:
        """Retrieves a session by its ID, from cache or Firestore."""
        # Try memory cache first
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            if self._is_session_expired(session):
                await self._expire_session(session_id)
                return None
            return session
        
        # Load from Firestore
        session = await self._load_session_from_firestore(session_id)
        if session and not self._is_session_expired(session):
            self.active_sessions[session_id] = session
            return session
            
        return None
    
    async def cleanup_expired_sessions(self) -> int:
        """Finds and expires old sessions."""
        expired_count = 0
        current_sessions = list(self.active_sessions.keys())
        
        for session_id in current_sessions:
            session = self.active_sessions[session_id]
            if self._is_session_expired(session):
                await self._expire_session(session_id)
                expired_count += 1
        
        logger.info(f"🧹 Cleaned up {expired_count} expired sessions")
        return expired_count
     
    async def get_all_sessions_summary(self) -> List[Dict[str, Any]]:
        """
        Fetches a summary list of all sessions from Firestore.
    
        This method queries the main collection for all session documents
        and returns a lightweight summary for each.
        """
        if not self.firestore:
            logger.warning("Firestore service not available, cannot fetch sessions summary.")
            return []
    
        try:
            # This assumes your FirestoreService has a method to get all documents from a collection.
            # This is a common pattern for such a service class.
            all_sessions = await self.firestore.get_all_documents('engineer_chat_sessions')
        
            # The endpoint expects a list of dicts with session_id, created_at, and title.
            return [
                {
                    "session_id": session.get("session_id"),
                    "created_at": session.get("created_at"),
                    "title": session.get("title") 
                } for session in all_sessions if session.get("session_id")
            ]
        except Exception as e:
            logger.error(f"Failed to get all sessions summary from Firestore: {e}")
            # Return an empty list on failure to prevent the API from crashing.
            return []
        
    
    # =============================================================================
    # PRIVATE METHODS
    # =============================================================================
    
    def _is_session_expired(self, session: UnifiedSession) -> bool:
        """Checks if a session has expired based on last activity."""
        if session.status != "active":
            return True
        expiry_time = session.last_activity + timedelta(hours=self.session_timeout_hours)
        return datetime.now(timezone.utc) > expiry_time
    
    async def _persist_session(self, session: UnifiedSession) -> bool:
        """Saves the session's metadata to the main Firestore document."""
        try:
            # Create a dictionary with only the fields intended for the main document.
            # This explicitly excludes 'conversation' and 'files', which are managed
            # in subcollections to avoid the 1MB document size limit.
            session_metadata = {
                "session_id": session.session_id,
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "status": session.status,
                "confidence_score": session.confidence_score,
                "can_generate_offer": session.can_generate_offer,
            }
            
            logger.info(f"💾 Persisting session metadata for {session.session_id}")
            
            # Save to Firestore
            await self.firestore.save_document(
                collection="chat_sessions",
                document_id=session.session_id,
                data=session_metadata
            )
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to persist session {session.session_id}: {e}")
            logger.exception(e)
            return False
    
    async def _load_session_from_firestore(self, session_id: str) -> Optional[UnifiedSession]:
        """Loads session metadata from Firestore and deserializes it."""
        try:
            data = await self.firestore.get_document("chat_sessions", session_id)
            if not data:
                return None
            
            # Convert ISO strings back to datetimes
            created_at = datetime.fromisoformat(data["created_at"]) if isinstance(data.get("created_at"), str) else data.get("created_at", datetime.now(timezone.utc))
            last_activity = datetime.fromisoformat(data["last_activity"]) if isinstance(data.get("last_activity"), str) else data.get("last_activity", datetime.now(timezone.utc))
            
            # Instantiate the session object with metadata only.
            # 'conversation' and 'files' will default to empty lists and are
            # expected to be populated from subcollections by other services when needed.
            return UnifiedSession(
                session_id=data["session_id"],
                created_at=created_at,
                last_activity=last_activity,
                status=data.get("status", "active"),
                confidence_score=data.get("confidence_score", 0.0),
                can_generate_offer=data.get("can_generate_offer", False)
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to load session {session_id}: {e}")
            return None
    
    async def _expire_session(self, session_id: str) -> bool:
        """Marks a session as expired in both cache and Firestore."""
        try:
            # Remove from memory
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            # Update status in Firestore
            await self.firestore.save_document(
                collection="chat_sessions",
                document_id=session_id,
                data={"status": "expired", "last_activity": datetime.now(timezone.utc).isoformat()}
            )
            
            logger.info(f"🕒 Expired session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to expire session {session_id}: {e}")
            return False

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

# Global session manager instance
_session_manager: Optional[SessionManager] = None

async def initialize_session_manager(firestore_service=None):
    """Initializes the global session manager instance."""
    global _session_manager
    if not _session_manager:
        if firestore_service is None:
            firestore = FirestoreService()
            await firestore.initialize()
        else:
            firestore = firestore_service
        _session_manager = SessionManager(firestore)
    logger.info("✅ Session Manager initialized")
    return _session_manager

def get_session_manager() -> SessionManager:
    """Gets the global session manager instance."""
    if _session_manager is None:
        raise RuntimeError("Session manager not initialized. Call initialize_session_manager() first.")
    return _session_manager

async def load_session_safely(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Convenience function to load session metadata safely and return as a dictionary.
    Note: 'conversation' and 'files' will be empty unless populated after loading.
    """
    manager = get_session_manager()
    session = await manager.get_session(session_id)
    if not session:
        return None
        
    return {
        "session_id": session.session_id,
        "created_at": session.created_at.isoformat(),
        "last_activity": session.last_activity.isoformat(),
        "status": session.status,
        "confidence_score": session.confidence_score,
        "can_generate_offer": session.can_generate_offer,
        "files": session.files,  # Will be empty list by default
        "conversation": session.conversation, # Will be empty list by default
    }

async def save_session_safely(session_id: str, data: Dict[str, Any]):
    """
    Convenience function to save a session's metadata safely.
    Reconstructs a session object from a dictionary and persists its metadata.
    """
    manager = get_session_manager()
    
    # Convert data back to a UnifiedSession object.
    # Note: 'created_at' should ideally not be overwritten from the original session.
    created_at = datetime.fromisoformat(data["created_at"]) if isinstance(data.get("created_at"), str) else data.get("created_at", datetime.now(timezone.utc))
    
    # Always update last_activity on save to reflect recent activity.
    last_activity = datetime.now(timezone.utc)
    
    session = UnifiedSession(
        session_id=session_id,
        created_at=created_at,
        last_activity=last_activity,
        status=data.get("status", "active"),
        confidence_score=data.get("confidence_score", 0.0),
        can_generate_offer=data.get("can_generate_offer", False),
        # These are included for the in-memory object but will be stripped by _persist_session
        files=data.get("files", []),
        conversation=data.get("conversation", [])
    )   
    await manager._persist_session(session)