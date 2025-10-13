"""
DemoPLAN - Enhanced Firestore Service
Google Cloud Firestore integration with ML session storage and scalable subcollections.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta
import json

from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter
from google.api_core import exceptions
from dataclasses import is_dataclass, asdict

from config.config import settings

logger = logging.getLogger("demoplan.services.firestore")

# --- Constants for Subcollection Architecture ---
MESSAGES_SUBCOLLECTION = "messages"
FILE_ANALYSES_SUBCOLLECTION = "file_analyses"
DEFAULT_MESSAGE_LIMIT = 50

# Import geometric storage
try:
    from src.services.storage_service import geometric_storage
except ImportError:
    logger.warning("⚠️ Geometric storage not available")
    geometric_storage = None

class FirestoreService:
    """
    Enhanced Firestore service for ML-enabled session persistence.
    Handles chat sessions using a scalable subcollection model for messages and file analyses.
    """
    
    def __init__(self):
        self.db: Optional[firestore.AsyncClient] = None
        self.initialized = False
        
        # ML-specific collections
        self.ml_collections = {
            "chat_sessions": "engineer_chat_sessions",
            "training_data": "ml_training_data",
            "project_learning": "project_learning_outcomes",
            "pattern_templates": "ml_pattern_templates",
            "learning_metrics": "ml_learning_metrics",
            "enhanced_patterns": "enhanced_analysis_patterns",
            "ocr_results": "ocr_results"
        }
        
    async def initialize(self):
        """Initialize Firestore connection with ML collections"""
        try:
            if self.initialized:
                return
            logger.info("🔥 Initializing Firestore connection with ML support...")
            
            # Initialize Firestore client
            self.db = firestore.AsyncClient(
                project=settings.gcp_project_id,
                database="(default)"
            )
            
            # Test the connection
            await self._test_connection()
            
            # Setup ML collections
            await self._setup_ml_collections()
            
            self.initialized = True
            logger.info("✅ Firestore connection established with ML collections")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Firestore: {str(e)}")
            raise
    
    async def _test_connection(self):
        """Test Firestore connection"""
        try:
            # Try to read from a test collection
            test_doc = await self.db.collection("test").document("connection").get()
            logger.debug("🔥 Firestore connection test successful")
        except Exception as e:
            logger.warning(f"⚠️ Firestore connection test warning: {str(e)}")
            # Don't fail initialization for connection test issues
    
    async def _setup_ml_collections(self):
        """Setup ML-specific collections"""
        try:
            logger.info("🗄️ Setting up ML collections...")
            
            for logical_name, collection_name in self.ml_collections.items():
                test_data = {
                    "collection_type": logical_name,
                    "initialized_at": datetime.now(timezone.utc),
                    "ml_enabled": True,
                    "description": self._get_collection_description(logical_name)
                }
                
                await self._save_document_direct(
                    collection_name,
                    "_ml_initialization",
                    test_data
                )
            
            logger.info("✅ ML collections setup completed")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup ML collections: {str(e)}")
            # Don't fail initialization
    
    def _get_collection_description(self, collection_type: str) -> str:
        """Get description for collection types"""
        descriptions = {
            "chat_sessions": "Persistent chat sessions with scalable subcollections for history and analyses.",
            "training_data": "Uploaded training files for ML learning",
            "project_learning": "Historical project outcomes for learning",
            "pattern_templates": "ML-generated patterns for analysis",
            "learning_metrics": "Performance tracking for ML components",
            "enhanced_patterns": "Enhanced analysis patterns from ML"
        }
        return descriptions.get(collection_type, "ML data collection")

    # --- CORE DOCUMENT OPERATIONS (MODIFIED) ---

    async def save_document(
        self, 
        collection: str, 
        document_id: str, 
        data: Dict[str, Any]
    ) -> bool:
        """
        Save a document to Firestore with ML collection mapping.
        For 'chat_sessions', large fields are automatically stripped.
        """
        try:
            if not self.initialized:
                await self.initialize()
            
            physical_collection = self.ml_collections.get(collection, f"demoplan_{collection}")
            
            data_to_save = data.copy()

            # MODIFICATION: For chat sessions, remove large fields before saving main document.
            if collection == "chat_sessions":
                data_to_save.pop('conversation', None)
                data_to_save.pop('conversation_history', None)
                data_to_save.pop('analysis_results', None)
                logger.debug(f"Stripped large fields from session {document_id} before saving.")

            enhanced_data = {
                **data_to_save,
                "_updated_at": datetime.now(timezone.utc),
                "_collection_type": collection,
                "_ml_enabled": True
            }
            
            return await self._save_document_direct(physical_collection, document_id, enhanced_data)
            
        except Exception as e:
            logger.error(f"❌ Failed to save document {collection}/{document_id}: {str(e)}")
            return False
    
    async def get_document(
        self, 
        collection: str, 
        document_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get a document from Firestore.
        For 'chat_sessions', conversation and analyses must be loaded separately.
        """
        try:
            if not self.initialized:
                await self.initialize()
            
            physical_collection = self.ml_collections.get(collection, f"demoplan_{collection}")
            
            doc_ref = self.db.collection(physical_collection).document(document_id)
            doc = await doc_ref.get()
            
            if doc.exists:
                data = doc.to_dict()
                data['_id'] = doc.id
                logger.debug(f"📂 Document retrieved: {collection}/{document_id}")
                # MODIFICATION: Note that conversation/analysis are not loaded here.
                if collection == "chat_sessions":
                    logger.debug(f"Note: For session {document_id}, messages and analyses must be loaded from subcollections.")
                return self._deserialize_data(data)
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get document {collection}/{document_id}: {str(e)}")
            return None

    # --- NEW: MESSAGE SUBCOLLECTION METHODS ---

    async def save_message_to_subcollection(self, session_id: str, message: Dict[str, Any]) -> Optional[str]:
        """Saves a single message to the 'messages' subcollection for a given session."""
        try:
            if not self.initialized: await self.initialize()
            
            collection_name = self.ml_collections["chat_sessions"]
            session_ref = self.db.collection(collection_name).document(session_id)
            
            message_data = message.copy()
            message_data['timestamp'] = message_data.get('timestamp', datetime.now(timezone.utc))

            serialized_message = self._serialize_data(message_data)
            
            update_time, message_ref = await session_ref.collection(MESSAGES_SUBCOLLECTION).add(serialized_message)
            logger.info(f"✅ Message {message_ref.id} saved to session {session_id}")
            return message_ref.id
        except Exception as e:
            logger.error(f"❌ Failed to save message to session {session_id}: {e}")
            return None

    async def load_messages_from_subcollection(self, session_id: str, limit: int = DEFAULT_MESSAGE_LIMIT) -> List[Dict]:
        """Loads messages from the 'messages' subcollection, ordered by timestamp."""
        try:
            if not self.initialized: await self.initialize()

            collection_name = self.ml_collections["chat_sessions"]
            messages_ref = self.db.collection(collection_name).document(session_id).collection(MESSAGES_SUBCOLLECTION)
            
            query = messages_ref.order_by("timestamp", direction=firestore.Query.ASCENDING).limit(limit)
            docs = await query.get()
            
            messages = [self._deserialize_data(doc.to_dict()) for doc in docs]
            logger.info(f"📂 Loaded {len(messages)} messages for session {session_id}")
            return messages
        except Exception as e:
            logger.error(f"❌ Failed to load messages for session {session_id}: {e}")
            return []

    async def get_message_count(self, session_id: str) -> int:
        """Gets the total count of messages in a session's subcollection."""
        try:
            if not self.initialized: await self.initialize()

            collection_name = self.ml_collections["chat_sessions"]
            messages_ref = self.db.collection(collection_name).document(session_id).collection(MESSAGES_SUBCOLLECTION)
            
            # Use an aggregate query for efficient counting
            count_query = messages_ref.count()
            result = await count_query.get()
            count = result[0][0].value
            logger.debug(f"📊 Message count for session {session_id} is {count}")
            return count
        except Exception as e:
            logger.error(f"❌ Failed to get message count for session {session_id}: {e}")
            return 0

    # --- NEW: FILE ANALYSIS SUBCOLLECTION METHODS ---

    async def save_file_analysis(self, session_id: str, file_id: str, analysis_data: Dict[str, Any]) -> bool:
        """Saves or overwrites a file analysis in the 'file_analyses' subcollection."""
        try:
            if not self.initialized: await self.initialize()

            collection_name = self.ml_collections["chat_sessions"]
            doc_ref = self.db.collection(collection_name).document(session_id).collection(FILE_ANALYSES_SUBCOLLECTION).document(file_id)
            
            serialized_data = self._serialize_data(analysis_data)
            await doc_ref.set(serialized_data)
            logger.info(f"💾 File analysis {file_id} saved for session {session_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save file analysis {file_id} for session {session_id}: {e}")
            return False

    async def load_file_analysis(self, session_id: str, file_id: str) -> Optional[Dict]:
        """Loads a specific file analysis from the subcollection."""
        try:
            if not self.initialized: await self.initialize()

            collection_name = self.ml_collections["chat_sessions"]
            doc_ref = self.db.collection(collection_name).document(session_id).collection(FILE_ANALYSES_SUBCOLLECTION).document(file_id)
            doc = await doc_ref.get()

            if doc.exists:
                logger.info(f"📂 Loaded file analysis {file_id} for session {session_id}")
                return self._deserialize_data(doc.to_dict())
            return None
        except exceptions.NotFound:
            logger.warning(f"⚠️ File analysis {file_id} not found for session {session_id}")
            return None
        except Exception as e:
            logger.error(f"❌ Failed to load file analysis {file_id} for session {session_id}: {e}")
            return None

    async def load_all_file_analyses(self, session_id: str) -> List[Dict]:
        """Loads all file analyses for a given session."""
        try:
            if not self.initialized: await self.initialize()
            
            collection_name = self.ml_collections["chat_sessions"]
            analyses_ref = self.db.collection(collection_name).document(session_id).collection(FILE_ANALYSES_SUBCOLLECTION)
            docs = analyses_ref.stream()
            
            analyses = [self._deserialize_data(doc.to_dict()) async for doc in docs]
            logger.info(f"📂 Loaded all {len(analyses)} file analyses for session {session_id}")
            return analyses
        except Exception as e:
            logger.error(f"❌ Failed to load all file analyses for session {session_id}: {e}")
            return []

    async def get_file_analyses_summary(self, session_id: str) -> Dict[str, Any]:
        """Provides a summary of file analyses (count, types, statuses)."""
        analyses = await self.load_all_file_analyses(session_id)
        summary = {
            "count": len(analyses),
            "file_types": {},
            "statuses": {}
        }
        for analysis in analyses:
            file_type = analysis.get("file_type", "unknown")
            status = analysis.get("processing_status", "unknown")
            summary["file_types"][file_type] = summary["file_types"].get(file_type, 0) + 1
            summary["statuses"][status] = summary["statuses"].get(status, 0) + 1
        return summary

    async def update_file_analysis_status(self, session_id: str, file_id: str, status: str) -> bool:
        """Updates only the processing status of a file analysis document."""
        try:
            if not self.initialized: await self.initialize()

            collection_name = self.ml_collections["chat_sessions"]
            doc_ref = self.db.collection(collection_name).document(session_id).collection(FILE_ANALYSES_SUBCOLLECTION).document(file_id)
            
            await doc_ref.update({"processing_status": status, "_updated_at": datetime.now(timezone.utc)})
            logger.info(f"🔄 Status for file {file_id} in session {session_id} updated to '{status}'")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to update status for file {file_id}: {e}")
            return False

    async def delete_file_analysis(self, session_id: str, file_id: str) -> bool:
        """Deletes a specific file analysis document from the subcollection."""
        try:
            if not self.initialized: await self.initialize()

            collection_name = self.ml_collections["chat_sessions"]
            doc_ref = self.db.collection(collection_name).document(session_id).collection(FILE_ANALYSES_SUBCOLLECTION).document(file_id)
            
            await doc_ref.delete()
            logger.info(f"🗑️ Deleted file analysis {file_id} from session {session_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to delete file analysis {file_id}: {e}")
            return False

    # Add to existing FirestoreService class

    async def save_drawing_metadata(
        self,
        session_id: str,
        drawing_id: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Save drawing metadata to drawings subcollection
        
        Path: engineer_chat_sessions/{session_id}/drawings/{drawing_id}
        
        Args:
            session_id: Session identifier
            drawing_id: Drawing identifier
            metadata: Drawing metadata dictionary
        
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            doc_ref = (
                self.db.collection('engineer_chat_sessions')
                .document(session_id)
                .collection('drawings')
                .document(drawing_id)
            )
            
            await doc_ref.set(metadata)
            logger.info(f"✅ Drawing metadata saved: {drawing_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save drawing metadata: {e}")
            return False

    async def get_session_drawings(
        self,
        session_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get all drawings for a session from drawings subcollection
        
        Args:
            session_id: Session identifier
        
        Returns:
            List of drawing metadata dictionaries
        """
        try:
            drawings_ref = (
                self.db.collection('engineer_chat_sessions')
                .document(session_id)
                .collection('drawings')
            )
            
            drawings = []
            async for doc in drawings_ref.stream():
                drawing_data = doc.to_dict()
                drawing_data['drawing_id'] = doc.id
                drawings.append(drawing_data)
            
            logger.info(f"✅ Retrieved {len(drawings)} drawings for session {session_id}")
            return drawings
            
        except Exception as e:
            logger.error(f"❌ Failed to get drawings: {e}")
            return []

    # --- DEPRECATED METHODS ---

    async def save_session_with_history(self, *args, **kwargs) -> bool:
        """DEPRECATED: Use save_document and save_message_to_subcollection instead."""
        logger.warning("save_session_with_history is deprecated. Use subcollection methods.")
        # This method is now obsolete and should not be used.
        return False
    
    async def get_session_conversation_history(self, *args, **kwargs) -> List:
        """DEPRECATED: Use load_messages_from_subcollection instead."""
        logger.warning("get_session_conversation_history is deprecated. Use load_messages_from_subcollection.")
        # This method is now obsolete and should not be used.
        return []
        
    # --- EXISTING UNMODIFIED/HELPER METHODS ---
    
    async def get_all_documents(
        self,
        collection: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all documents from a collection
        
        Args:
            collection: Collection name
            limit: Optional limit on number of documents
        
        Returns:
            List of document dictionaries
        """
        try:
            if not self.initialized:
                await self.initialize()
            
            physical_collection = self.ml_collections.get(collection, f"demoplan_{collection}")
            
            query = self.db.collection(physical_collection)
            if limit:
                query = query.limit(limit)
            
            docs_stream = query.stream()
            results = []
            
            async for doc in docs_stream:
                doc_data = doc.to_dict()
                doc_data['_id'] = doc.id
                results.append(self._deserialize_data(doc_data))
            
            logger.debug(f"📂 Retrieved {len(results)} documents from {collection}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to get all documents from {collection}: {str(e)}")
            return []
    
    async def _save_document_direct(
        self, 
        collection_name: str, 
        document_id: str, 
        data: Dict[str, Any]
    ) -> bool:
        """Save document directly to physical collection"""
        try:
            serialized_data = self._serialize_data(data)
            doc_ref = self.db.collection(collection_name).document(document_id)
            await doc_ref.set(serialized_data)
            logger.debug(f"💾 Document saved: {collection_name}/{document_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Direct save failed {collection_name}/{document_id}: {str(e)}")
            return False
            
    async def delete_document(self, collection: str, document_id: str) -> bool:
        """Delete a document with ML collection mapping"""
        try:
            if not self.initialized: await self.initialize()
            physical_collection = self.ml_collections.get(collection, f"demoplan_{collection}")
            doc_ref = self.db.collection(physical_collection).document(document_id)
            await doc_ref.delete()
            logger.debug(f"🗑️ Document deleted: {collection}/{document_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to delete document {collection}/{document_id}: {str(e)}")
            return False

    async def query_documents(
        self, 
        collection: str, 
        filters: Optional[List[tuple]] = None,
        limit: Optional[int] = None,
        order_by: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Query documents with ML collection mapping and enhanced filtering"""
        try:
            if not self.initialized: await self.initialize()
            physical_collection = self.ml_collections.get(collection, f"demoplan_{collection}")
            query = self.db.collection(physical_collection)
            if filters:
                for field, operator, value in filters:
                    query = query.where(filter=FieldFilter(field, operator, value))
            if order_by:
                query = query.order_by(order_by)
            if limit:
                query = query.limit(limit)
            
            docs_stream = query.stream()
            results = [self._deserialize_data(doc.to_dict()) async for doc in docs_stream]
            logger.debug(f"🔍 Queried {len(results)} documents from {collection}")
            return results
        except Exception as e:
            logger.error(f"❌ Failed to query documents from {collection}: {str(e)}")
            return []
    
    async def update_document(
        self, 
        collection: str, 
        document_id: str, 
        update_data: Dict[str, Any]
    ) -> bool:
        """
        Update specific fields in an existing document.
        Creates the document if it doesn't exist.
        """
        try:
            if not self.initialized: await self.initialize()
            physical_collection = self.ml_collections.get(collection, f"demoplan_{collection}")
            enhanced_updates = {**update_data, "_updated_at": datetime.now(timezone.utc)}
            serialized_updates = self._serialize_data(enhanced_updates)
            doc_ref = self.db.collection(physical_collection).document(document_id)
            # FIX [404 Not Found]: Use set with merge=True to create the document if it's missing.
            await doc_ref.set(serialized_updates, merge=True)
            logger.debug(f"🔄 Document upserted: {collection}/{document_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to update document {collection}/{document_id}: {str(e)}")
            return False
    
    def _serialize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize data for Firestore storage with dataclass support"""
        serialized = {}
        for key, value in data.items():
            if isinstance(value, datetime):
                serialized[key] = value
            elif is_dataclass(value) and not isinstance(value, type):
                serialized[key] = self._serialize_data(asdict(value))
            elif isinstance(value, dict):
                serialized[key] = self._serialize_data(value)
            elif isinstance(value, list):
                serialized[key] = [
                    self._serialize_data(asdict(item)) if (is_dataclass(item) and not isinstance(item, type))
                    else self._serialize_data(item) if isinstance(item, dict)
                    else item
                    for item in value
                ]
            else:
                serialized[key] = value
        return serialized

    def _deserialize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize data from Firestore"""
        # The existing implementation is sufficient as it handles timestamps correctly.
        return data

    # ... (The rest of the file's methods like OCR, training data, etc., remain unchanged)
    
    async def cleanup_expired_sessions(self, hours: int = 24) -> int:
        """Clean up expired sessions"""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            
            expired_sessions = await self.query_documents(
                collection="chat_sessions",
                filters=[("_updated_at", "<", cutoff_time)],
                limit=100
            )
            
            deleted_count = 0
            for session in expired_sessions:
                if await self.delete_document("chat_sessions", session["_id"]):
                    deleted_count += 1
            
            logger.info(f"🧹 Cleaned up {deleted_count} expired sessions")
            return deleted_count
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup sessions: {e}")
            return 0
    
    # ... The rest of the original file content from save_learning_outcome onwards ...
    async def save_learning_outcome(self, session_id: str, learning_data: Dict[str, Any]) -> bool:
        """Save project learning outcome for ML training"""
        # This method remains valid
        try:
            learning_id = f"learning_{session_id}_{datetime.now(timezone.utc).strftime('%Y%m%d')}"
            enhanced_learning = {
                **learning_data,
                "session_id": session_id,
                "recorded_at": datetime.now(timezone.utc),
                "learning_version": "v1.0"
            }
            return await self.save_document("project_learning", learning_id, data=enhanced_learning)
        except Exception as e:
            logger.error(f"❌ Failed to save learning outcome: {e}")
            return False

    # --- Geometric split helpers ---
    async def save_file_analysis_with_geometric_split(
        self,
        session_id: str,
        file_id: str,
        analysis_data: Dict[str, Any]
    ) -> bool:
        """
        Save file analysis with UNIFORM geometric data split.
        
        Rule: For DXF files, geometric data ALWAYS goes to GCS (regardless of size).
              Summary/text data ALWAYS goes to Firestore.
        """
        try:
            if not self.initialized:
                await self.initialize()
            
            # Check if this is a DXF file with geometric data
            file_type = analysis_data.get("analysis_data", {}).get("file_type")
            dxf_analysis_result = analysis_data.get("analysis_data", {}).get("dxf_analysis_result")
            
            if file_type == "dxf" and dxf_analysis_result:
                geometric_data = dxf_analysis_result.get("dxf_analysis", {})
                
                if geometric_data and geometric_storage and geometric_storage.is_available():
                    logger.info(f"📦 DXF detected - splitting geometric data to GCS for {file_id}")
                    
                    # Store geometric data in GCS (ALWAYS)
                    geometric_ref = await geometric_storage.store_geometric_data(
                        session_id=session_id,
                        file_id=file_id,
                        geometric_data=geometric_data
                    )
                    
                    if geometric_ref:
                        # Create lightweight version for Firestore
                        analysis_data_copy = self._deep_copy_dict(analysis_data)
                        
                        # Remove large geometric data from Firestore document
                        if "analysis_data" in analysis_data_copy:
                            if "dxf_analysis_result" in analysis_data_copy["analysis_data"]:
                                # Keep only metadata, remove full geometric content
                                analysis_data_copy["analysis_data"]["dxf_analysis_result"] = {
                                    "status": "success",
                                    "storage": "gcs",
                                    "note": "Full geometric data stored in GCS"
                                }
                        
                        # Add geometric reference
                        analysis_data_copy["geometric_ref"] = geometric_ref
                        analysis_data_copy["has_geometric_data"] = True
                        analysis_data_copy["geometric_storage"] = "gcs"
                        
                        logger.info(f"✅ Geometric data split - Summary to Firestore, Geometric to GCS")
                        
                        # Save the lightweight version to Firestore
                        return await self.save_file_analysis(session_id, file_id, analysis_data_copy)
                    else:
                        logger.error(f"❌ Failed to store geometric data in GCS for {file_id}")
                        logger.warning(f"⚠️ Falling back to Firestore-only storage (may be too large)")
                        # Fall back to saving everything in Firestore (may fail if too large)
                        return await self.save_file_analysis(session_id, file_id, analysis_data)
                else:
                    # GCS not available or no geometric data
                    if not geometric_storage or not geometric_storage.is_available():
                        logger.warning(f"⚠️ GCS not available - saving DXF to Firestore only")
                    return await self.save_file_analysis(session_id, file_id, analysis_data)
            else:
                # Not a DXF file (PDF, TXT, etc.) - save normally to Firestore
                logger.info(f"📄 Non-DXF file - saving to Firestore")
                analysis_data["has_geometric_data"] = False
                analysis_data["geometric_storage"] = "none"
                return await self.save_file_analysis(session_id, file_id, analysis_data)
                
        except Exception as e:
            logger.error(f"❌ Failed to save file analysis with split: {e}", exc_info=True)
            return False

    async def load_file_analysis_with_geometric(
        self,
        session_id: str,
        file_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Load file analysis and automatically fetch geometric data from GCS if split.
        """
        try:
            # Load the Firestore document (contains summary/metadata)
            analysis_data = await self.load_file_analysis(session_id, file_id)
            
            if not analysis_data:
                logger.warning(f"⚠️ File analysis not found in Firestore: {file_id}")
                return None
            
            # Check if geometric data was split to GCS
            has_geometric_data = analysis_data.get("has_geometric_data", False)
            geometric_ref = analysis_data.get("geometric_ref")
            
            if has_geometric_data and geometric_ref:
                logger.info(f"📥 Geometric data detected in GCS for {file_id} - loading...")
                
                # Load geometric data from GCS
                if geometric_storage and geometric_storage.is_available():
                    geometric_data = await geometric_storage.load_geometric_data_from_ref(geometric_ref)
                    
                    if geometric_data:
                        # Merge geometric data back into the analysis structure
                        if "analysis_data" not in analysis_data:
                            analysis_data["analysis_data"] = {}
                        if "dxf_analysis_result" not in analysis_data["analysis_data"]:
                            analysis_data["analysis_data"]["dxf_analysis_result"] = {}
                        
                        analysis_data["analysis_data"]["dxf_analysis_result"]["dxf_analysis"] = geometric_data
                        analysis_data["analysis_data"]["dxf_analysis_result"]["status"] = "success"
                        
                        logger.info(f"✅ Merged geometric data from GCS into analysis")
                    else:
                        logger.error(f"❌ Could not load geometric data from GCS for {file_id}")
                else:
                    logger.error(f"❌ GCS not available - cannot load geometric data")
            else:
                logger.debug(f"📄 No geometric data in GCS for {file_id}")
            
            return analysis_data
            
        except Exception as e:
            logger.error(f"❌ Failed to load file analysis with geometric: {e}", exc_info=True)
            return None

    async def load_geometric_data_only(
        self,
        session_id: str,
        file_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Load ONLY the geometric data for a file (from GCS).
        """
        try:
            # First check if the file analysis has a geometric reference
            analysis_data = await self.load_file_analysis(session_id, file_id)
            
            if not analysis_data:
                logger.warning(f"⚠️ File analysis not found: {file_id}")
                return None
            
            geometric_ref = analysis_data.get("geometric_ref")
            
            if geometric_ref:
                # Load from GCS
                if geometric_storage and geometric_storage.is_available():
                    return await geometric_storage.load_geometric_data_from_ref(geometric_ref)
                else:
                    logger.error("❌ GCS not available")
                    return None
            else:
                # No geometric data in GCS (shouldn't happen for DXF files)
                logger.warning(f"⚠️ No geometric reference found for {file_id}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to load geometric data: {e}", exc_info=True)
            return None

    async def delete_file_analysis_with_geometric(
        self,
        session_id: str,
        file_id: str
    ) -> bool:
        """
        Delete file analysis and its geometric data from BOTH Firestore and GCS.
        """
        try:
            # Load to check if geometric data exists in GCS
            analysis_data = await self.load_file_analysis(session_id, file_id)
            
            if analysis_data:
                geometric_ref = analysis_data.get("geometric_ref")
                
                # Delete from GCS if exists
                if geometric_ref and geometric_storage and geometric_storage.is_available():
                    logger.info(f"🗑️ Deleting geometric data from GCS for {file_id}")
                    await geometric_storage.delete_geometric_data(session_id, file_id)
            
            # Delete from Firestore
            result = await self.delete_file_analysis(session_id, file_id)
            
            if result:
                logger.info(f"✅ Deleted file analysis with geometric data for {file_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to delete file analysis with geometric: {e}", exc_info=True)
            return False

    async def cleanup_session_with_geometric(
        self,
        session_id: str
    ) -> bool:
        """
        Clean up entire session including ALL geometric data from GCS.
        Use when deleting a session.
        """
        try:
            # Delete all file analyses (Firestore)
            file_analyses = await self.load_all_file_analyses(session_id)
            
            for analysis in file_analyses:
                file_id = analysis.get("file_id")
                if file_id:
                    await self.delete_file_analysis_with_geometric(session_id, file_id)
            
            # Delete session document
            await self.delete_document("chat_sessions", session_id)
            
            # Delete any remaining geometric data in GCS for this session
            if geometric_storage and geometric_storage.is_available():
                deleted_count = await geometric_storage.delete_session_geometric_data(session_id)
                logger.info(f"🗑️ Cleaned up {deleted_count} geometric files for session {session_id}")
            
            logger.info(f"✅ Session {session_id} cleaned up completely")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup session with geometric: {e}", exc_info=True)
            return False

    def _deep_copy_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep copy a dictionary to avoid mutating original data.
        Simple implementation for nested dicts.
        """
        import copy
        return copy.deepcopy(data)

print("✅ DemoPLAN Enhanced Firestore Service with Scalable Subcollections loaded")