"""
FirestoreService - clean, minimal, and well-formed implementation

Provides:
- class-level serialization helpers
- basic Firestore operations: save/get/update documents
- subcollection helpers for messages and file analyses
- geometric-split support for DXF files
"""

from enum import Enum
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

try:
    from google.cloud import firestore
    from google.cloud.firestore_v1.base_query import FieldFilter
except Exception:
    firestore = None
    FieldFilter = None

from dataclasses import is_dataclass, asdict
from config.config import settings

logger = logging.getLogger("demoplan.services.firestore")

MESSAGES_SUBCOLLECTION = "messages"
FILE_ANALYSES_SUBCOLLECTION = "file_analyses"
DEFAULT_MESSAGE_LIMIT = 50

# optional geometric storage
try:
    from src.services.storage_service import geometric_storage
except Exception:
    geometric_storage = None


class FirestoreService:
    def __init__(self):
        self.db: Optional[Any] = None
        self.initialized = False
        self.ml_collections = {
            "chat_sessions": "engineer_chat_sessions",
            "training_data": "ml_training_data",
            "project_learning": "project_learning_outcomes",
            "pattern_templates": "ml_pattern_templates",
            "learning_metrics": "ml_learning_metrics",
            "enhanced_patterns": "enhanced_analysis_patterns",
            "ocr_results": "ocr_results",
        }

    async def initialize(self):
        if self.initialized:
            return
        if firestore is None:
            logger.warning(
                "google-cloud-firestore not installed - running in mock mode"
            )
            self.db = None
            self.initialized = True
            return

        self.db = firestore.AsyncClient(project=settings.gcp_project_id)
        await self._test_connection()
        await self._setup_ml_collections()
        self.initialized = True

    async def _test_connection(self):
        try:
            if self.db is None:
                return
            await self.db.collection("__health_check").document("ping").get()
        except Exception:
            logger.warning("Firestore health check failed")

    async def _setup_ml_collections(self):
        logger.info("Setting up ML collections")
        for logical_name, collection_name in self.ml_collections.items():
            logger.debug(f"Mapped {logical_name} -> {collection_name}")

    # ---- Serialization helpers (class-level) ----
    def _serialize_data(self, data: Any) -> Any:
        if data is None:
            return None
        if isinstance(data, (str, int, float, bool)):
            return data
        if isinstance(data, bytes):
            return f"<bytes:{len(data)}>"
        if isinstance(data, datetime):
            return data
        if isinstance(data, Enum):
            return data.value
        if is_dataclass(data) and not isinstance(data, type):
            return self._serialize_data(asdict(data))
        if isinstance(data, dict):
            return {str(k): self._serialize_data(v) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            return [self._serialize_data(v) for v in data]
        if hasattr(data, "to_dict") and callable(getattr(data, "to_dict")):
            try:
                return self._serialize_data(data.to_dict())
            except Exception:
                pass
        if hasattr(data, "__dict__"):
            try:
                return self._serialize_data(data.__dict__)
            except Exception:
                pass
        return str(data)
    
    def _deserialize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder for deserialization logic if needed"""
        # Currently, just returns the data. Can be expanded to convert
        # ISO strings back to datetime objects if required.
        return data

    def _validate_firestore_safe(self, data: Any, path: str = "root") -> List[str]:
        problems: List[str] = []
        safe = (str, int, float, bool, type(None), datetime, list, dict)
        if not isinstance(data, safe):
            problems.append(f"{path}: {type(data).__name__}")
            return problems
        if isinstance(data, dict):
            for k, v in data.items():
                problems.extend(self._validate_firestore_safe(v, f"{path}.{k}"))
        if isinstance(data, list):
            for i, v in enumerate(data):
                problems.extend(self._validate_firestore_safe(v, f"{path}[{i}]"))
        return problems

    def _force_serialize_to_primitives(self, data: Any) -> Any:
        if data is None:
            return None
        if isinstance(data, (str, int, float, bool)):
            return data
        if isinstance(data, datetime):
            return data
        if isinstance(data, bytes):
            return f"<bytes:{len(data)}>"
        if isinstance(data, dict):
            return {str(k): self._force_serialize_to_primitives(v) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            return [self._force_serialize_to_primitives(v) for v in data]
        try:
            return str(data)
        except Exception:
            return "<unserializable>"

    # ---- Core operations ----
    async def _save_document_direct(self, collection: str, document_id: str, data: Dict[str, Any]) -> bool:
        if not self.initialized:
            await self.initialize()
        payload = self._serialize_data(data)
        problems = self._validate_firestore_safe(payload)
        if problems:
            logger.warning(f"Validation problems: {problems}, forcing serialization")
            payload = self._force_serialize_to_primitives(data)
        if self.db is None:
            logger.info(f"(mock) save {collection}/{document_id}")
            return True
        try:
            await self.db.collection(collection).document(document_id).set(payload, merge=True)
            return True
        except Exception as e:
            logger.error(f"Failed to save document {collection}/{document_id}: {e}")
            return False

    async def save_document(self, collection: str, document_id: str, data: Dict[str, Any]) -> bool:
        try:
            data_copy = dict(data)
            if collection == "chat_sessions":
                for fld in ("conversation", "conversation_history", "analysis_results"):
                    data_copy.pop(fld, None)
            enhanced = {**data_copy, "_updated_at": datetime.now(timezone.utc), "_collection_type": collection}
            physical = self.ml_collections.get(collection, f"demoplan_{collection}")
            return await self._save_document_direct(physical, document_id, enhanced)
        except Exception as e:
            logger.error(f"save_document error: {e}")
            return False

    async def get_document(self, collection: str, document_id: str) -> Optional[Dict[str, Any]]:
        try:
            if not self.initialized:
                await self.initialize()
            if self.db is None:
                return None
            physical = self.ml_collections.get(collection, f"demoplan_{collection}")
            doc = await self.db.collection(physical).document(document_id).get()
            if not doc.exists:
                return None
            data = doc.to_dict()
            data["_id"] = doc.id
            return data
        except Exception as e:
            logger.error(f"get_document error: {e}")
            return None
            
    async def update_document(self, collection: str, document_id: str, update_data: Dict[str, Any]) -> bool:
        try:
            if not self.initialized: await self.initialize()
            physical_collection = self.ml_collections.get(collection, f"demoplan_{collection}")
            enhanced_updates = {**update_data, "_updated_at": datetime.now(timezone.utc)}
            serialized_updates = self._serialize_data(enhanced_updates)
            if self.db is None:
                logger.info(f"(mock) Updated {collection}/{document_id}")
                return True
            doc_ref = self.db.collection(physical_collection).document(document_id)
            await doc_ref.set(serialized_updates, merge=True)
            logger.debug(f"🔄 Document upserted: {collection}/{document_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to update document {collection}/{document_id}: {str(e)}")
            return False
            
    # ---- NEW METHOD: get_all_documents ----
    async def get_all_documents(self, collection: str) -> List[Dict[str, Any]]:
        """Fetches all documents from a specified collection."""
        try:
            if not self.initialized:
                await self.initialize()
            if self.db is None:
                logger.info(f"(mock) getting all documents from {collection}")
                return []
            
            physical_collection = self.ml_collections.get(collection, f"demoplan_{collection}")
            query = self.db.collection(physical_collection)
            results = []
            async for doc in query.stream():
                d = doc.to_dict()
                d["_id"] = doc.id
                results.append(self._deserialize_data(d))
            return results
        except Exception as e:
            logger.error(f"❌ get_all_documents error for {collection}: {e}")
            return []

    # ---- Messages ----
    async def save_message_to_subcollection(self, session_id: str, message: Dict[str, Any]) -> Optional[str]:
        try:
            if not self.initialized:
                await self.initialize()
            if self.db is None:
                logger.info(f"(mock) saved message for {session_id}")
                return "mock-id"
            collection_name = self.ml_collections["chat_sessions"]
            session_ref = self.db.collection(collection_name).document(session_id)
            msg = dict(message)
            msg.setdefault("timestamp", datetime.now(timezone.utc))
            serialized = self._serialize_data(msg)
            _, ref = await session_ref.collection(MESSAGES_SUBCOLLECTION).add(serialized)
            return ref.id
        except Exception as e:
            logger.error(f"save_message error: {e}")
            return None

    async def load_messages_from_subcollection(self, session_id: str, limit: int = DEFAULT_MESSAGE_LIMIT) -> List[Dict[str, Any]]:
        try:
            if not self.initialized:
                await self.initialize()
            if self.db is None:
                return []
            q = self.db.collection(self.ml_collections["chat_sessions"]).document(session_id).collection(MESSAGES_SUBCOLLECTION).order_by("timestamp")
            if limit:
                q = q.limit(limit)
            results: List[Dict[str, Any]] = []
            async for doc in q.stream():
                d = doc.to_dict()
                d["_id"] = doc.id
                results.append(d)
            return results
        except Exception as e:
            logger.error(f"load_messages error: {e}")
            return []
            
    # ---- NEW METHOD: get_message_count ----
    async def get_message_count(self, session_id: str) -> int:
        """Gets the count of messages in a session's subcollection."""
        try:
            if not self.initialized:
                await self.initialize()
            if self.db is None:
                return 0
            
            collection_name = self.ml_collections["chat_sessions"]
            messages_ref = self.db.collection(collection_name).document(session_id).collection(MESSAGES_SUBCOLLECTION)
            
            # Use aggregation query to get count
            aggregate_query = messages_ref.count()
            count_result = await aggregate_query.get()
            return count_result[0][0].value
        except Exception as e:
            logger.error(f"❌ get_message_count error: {e}")
            # Fallback to slower method if aggregation fails
            try:
                messages = await self.load_messages_from_subcollection(session_id, limit=9999)
                return len(messages)
            except Exception:
                return 0

    # ---- File analyses ----
    async def save_file_analysis(self, session_id: str, file_id: str, analysis: Dict[str, Any]) -> bool:
        try:
            if not self.initialized:
                await self.initialize()
            if self.db is None:
                return True
            collection_name = self.ml_collections["chat_sessions"]
            doc_ref = self.db.collection(collection_name).document(session_id).collection(FILE_ANALYSES_SUBCOLLECTION).document(file_id)
            payload = {"file_id": file_id, "analysis_data": analysis, "saved_at": datetime.now(timezone.utc)}
            await doc_ref.set(self._serialize_data(payload), merge=True)
            return True
        except Exception as e:
            logger.error(f"save_file_analysis error: {e}")
            return False

    async def load_file_analysis(self, session_id: str, file_id: str) -> Optional[Dict[str, Any]]:
        try:
            if not self.initialized:
                await self.initialize()
            if self.db is None:
                return None
            collection_name = self.ml_collections["chat_sessions"]
            doc = await self.db.collection(collection_name).document(session_id).collection(FILE_ANALYSES_SUBCOLLECTION).document(file_id).get()
            if not doc.exists:
                return None
            return doc.to_dict()
        except Exception as e:
            logger.error(f"load_file_analysis error: {e}")
            return None
            
    # ---- NEW METHOD: load_all_file_analyses ----
    async def load_all_file_analyses(self, session_id: str) -> List[Dict[str, Any]]:
        """Loads all file analysis documents from the subcollection for a session."""
        try:
            if not self.initialized:
                await self.initialize()
            if self.db is None:
                return []
            
            collection_name = self.ml_collections["chat_sessions"]
            analyses_ref = self.db.collection(collection_name).document(session_id).collection(FILE_ANALYSES_SUBCOLLECTION)
            
            results = []
            async for doc in analyses_ref.stream():
                d = doc.to_dict()
                d["_id"] = doc.id
                # The data is nested under 'analysis_data' by save_file_analysis
                results.append(d.get("analysis_data", d))
            return results
        except Exception as e:
            logger.error(f"❌ load_all_file_analyses error: {e}")
            return []

    async def delete_file_analysis(self, session_id: str, file_id: str) -> bool:
        try:
            if not self.initialized:
                await self.initialize()
            if self.db is None:
                return True
            collection_name = self.ml_collections["chat_sessions"]
            await self.db.collection(collection_name).document(session_id).collection(FILE_ANALYSES_SUBCOLLECTION).document(file_id).delete()
            return True
        except Exception as e:
            logger.error(f"delete_file_analysis error: {e}")
            return False

    async def save_file_analysis_with_geometric_split(self, session_id: str, file_id: str, analysis_data: Dict[str, Any]) -> bool:
        try:
            if not self.initialized:
                await self.initialize()
            # If no geometric storage available or not a dxf, save whole analysis
            if geometric_storage is None:
                return await self.save_file_analysis(session_id, file_id, analysis_data)
            
            file_type = analysis_data.get("analysis_data", {}).get("file_type")
            dxf_result = analysis_data.get("analysis_data", {}).get("dxf_analysis_result")
            
            if file_type == "dxf" and dxf_result:
                geometric = dxf_result.get("dxf_analysis")
                if geometric:
                    geom_ref = await geometric_storage.store_geometric_data(session_id=session_id, file_id=file_id, geometric_data=geometric)
                    if geom_ref:
                        copy = dict(analysis_data)
                        if "analysis_data" in copy and "dxf_analysis_result" in copy["analysis_data"]:
                            copy["analysis_data"]["dxf_analysis_result"] = {"status": "stored_in_gcs"}
                        copy["geometric_ref"] = geom_ref
                        copy["has_geometric_data"] = True
                        copy["geometric_storage"] = "gcs"
                        return await self.save_file_analysis(session_id, file_id, copy)
            
            analysis_data["has_geometric_data"] = False
            analysis_data["geometric_storage"] = "none"
            return await self.save_file_analysis(session_id, file_id, analysis_data)
        except Exception as e:
            logger.error(f"save_file_analysis_with_geometric_split error: {e}")
            return False

    async def load_file_analysis_with_geometric(self, session_id: str, file_id: str) -> Optional[Dict[str, Any]]:
        try:
            analysis_data = await self.load_file_analysis(session_id, file_id)
            if not analysis_data:
                return None
            has_geometric_data = analysis_data.get("has_geometric_data", False)
            geometric_ref = analysis_data.get("geometric_ref")
            if has_geometric_data and geometric_ref and geometric_storage and geometric_storage.is_available():
                geometric_data = await geometric_storage.load_geometric_data_from_ref(geometric_ref)
                if geometric_data:
                    if "analysis_data" not in analysis_data:
                        analysis_data["analysis_data"] = {}
                    if "dxf_analysis_result" not in analysis_data["analysis_data"]:
                        analysis_data["analysis_data"]["dxf_analysis_result"] = {}
                    analysis_data["analysis_data"]["dxf_analysis_result"]["dxf_analysis"] = geometric_data
                    analysis_data["analysis_data"]["dxf_analysis_result"]["status"] = "success"
            return analysis_data
        except Exception as e:
            logger.error(f"load_file_analysis_with_geometric error: {e}")
            return None

    async def delete_file_analysis_with_geometric(self, session_id: str, file_id: str) -> bool:
        try:
            analysis_data = await self.load_file_analysis(session_id, file_id)
            if analysis_data:
                geometric_ref = analysis_data.get("geometric_ref")
                if geometric_ref and geometric_storage and geometric_storage.is_available():
                    await geometric_storage.delete_geometric_data(session_id, file_id)
            return await self.delete_file_analysis(session_id, file_id)
        except Exception as e:
            logger.error(f"delete_file_analysis_with_geometric error: {e}")
            return False

    def _deep_copy_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        import copy
        return copy.deepcopy(data)

    # ---- NEW METHOD: get_session_drawings (Example) ----
    async def get_session_drawings(self, session_id: str) -> List[Dict[str, Any]]:
        """Example method to get drawings - adjust as needed"""
        # This is a placeholder. You'll need to implement this based on
        # how you store drawing generation results.
        logger.info(f"Placeholder: Fetching drawings for session {session_id}")
        return [
            # {
            #     "drawing_id": "example-drawing-123",
            #     "drawing_url": "https://example.com/drawing.dxf",
            #     "drawing_type": "field_verification",
            #     "created_at": datetime.now(timezone.utc).isoformat()
            # }
        ]