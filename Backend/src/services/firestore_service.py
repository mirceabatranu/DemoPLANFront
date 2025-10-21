"""
DemoPLAN - Enhanced Firestore Service
Google Cloud Firestore integration with ML session storage and scalable subcollections.
This is a focused, clean implementation used by the backend. It provides class-level
serialization helpers and robust save/load operations with an optional geometric split
to external storage.
"""

from enum import Enum
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter
from google.api_core import exceptions
"""
FirestoreService - clean replacement

This file implements a compact FirestoreService with class-level helpers
for serializing complex objects into Firestore-safe primitives and a
geometric-split helper for storing large DXF geometry externally.
"""

from enum import Enum
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

try:
    from google.cloud import firestore
except Exception:
    firestore = None

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
            logger.warning("google-cloud-firestore not installed - running in mock mode")
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
        logger.info("🗄️ Setting up ML collections...")
        for logical_name, collection_name in self.ml_collections.items():
            logger.debug(f"Mapped {logical_name} → {collection_name}")
        logger.info("✅ ML collections ready")

    # class-level helpers
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
            """
            FirestoreService - clean, minimal, and well-formed implementation

            Provides:
            - class-level serialization helpers: _serialize_data, _validate_firestore_safe, _force_serialize_to_primitives
            - basic Firestore operations: save/get/update documents
            - subcollection helpers for messages and file analyses
            - geometric-split support for DXF files (stores heavy geometry in external storage)

            This file purposefully avoids duplicate or nested definitions and is formatted
            for readability and maintainability.
            """

            from enum import Enum
            import logging
            from typing import Dict, List, Any, Optional
            from datetime import datetime, timezone

            try:
                from google.cloud import firestore
            except Exception:
                firestore = None

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


            # End of file
                    # --- Message subcollection helpers ---
                    async def save_message_to_subcollection(self, session_id: str, message: Dict[str, Any]) -> Optional[str]:
                        try:
                            if not self.initialized:
                                await self.initialize()
                            if self.db is None:
                                logger.info(f"(mock) Saved message to {session_id}")
                                return "mock-id"

                            collection_name = self.ml_collections["chat_sessions"]
                            session_ref = self.db.collection(collection_name).document(session_id)
                            message_data = dict(message)
                            message_data.setdefault("timestamp", datetime.now(timezone.utc))
                            serialized = self._serialize_data(message_data)
                            _, ref = await session_ref.collection(MESSAGES_SUBCOLLECTION).add(serialized)
                            return ref.id
                        except Exception as e:
                            logger.error(f"❌ save_message error: {e}")
                            return None

                    async def load_messages_from_subcollection(self, session_id: str, limit: int = DEFAULT_MESSAGE_LIMIT) -> List[Dict[str, Any]]:
                        try:
                            if not self.initialized:
                                await self.initialize()
                            if self.db is None:
                                return []
                            collection_name = self.ml_collections["chat_sessions"]
                            messages_ref = self.db.collection(collection_name).document(session_id).collection(MESSAGES_SUBCOLLECTION)
                            query = messages_ref.order_by("timestamp")
                            if limit:
                                query = query.limit(limit)
                            results = []
                            async for doc in query.stream():
                                d = doc.to_dict()
                                d["_id"] = doc.id
                                results.append(self._deserialize_data(d))
                            return results
                        except Exception as e:
                            logger.error(f"❌ load_messages error: {e}")
                            return []

                    async def query_documents(self, collection: str, filters: Optional[List[tuple]] = None, limit: Optional[int] = None, order_by: Optional[str] = None) -> List[Dict[str, Any]]:
                        try:
                            if not self.initialized:
                                await self.initialize()
                            if self.db is None:
                                return []
                            physical_collection = self.ml_collections.get(collection, f"demoplan_{collection}")
                            query = self.db.collection(physical_collection)
                            if filters:
                                for field, op, value in filters:
                                    # Fallback if FieldFilter is unavailable
                                    if FieldFilter is not None:
                                        query = query.where(filter=FieldFilter(field, op, value))
                            if order_by:
                                query = query.order_by(order_by)
                            if limit:
                                query = query.limit(limit)
                            results = []
                            async for doc in query.stream():
                                d = doc.to_dict()
                                d["_id"] = doc.id
                                results.append(self._deserialize_data(d))
                            return results
                        except Exception as e:
                            logger.error(f"❌ query_documents error: {e}")
                            return []

                    async def update_document(self, collection: str, document_id: str, update_data: Dict[str, Any]) -> bool:
                        try:
                            if not self.initialized:
                                await self.initialize()
                            physical_collection = self.ml_collections.get(collection, f"demoplan_{collection}")
                            enhanced = {**update_data, "_updated_at": datetime.now(timezone.utc)}
                            serialized = self._serialize_data(enhanced)
                            if self.db is None:
                                logger.info(f"(mock) Updated {collection}/{document_id}")
                                return True
                            await self.db.collection(physical_collection).document(document_id).set(serialized, merge=True)
                            return True
                        except Exception as e:
                            logger.error(f"❌ update_document error: {e}")
                            return False

                    # --- File analysis helpers and geometric split ---
                    async def save_file_analysis(self, session_id: str, file_id: str, analysis: Dict[str, Any]) -> bool:
                        try:
                            if not self.initialized:
                                await self.initialize()
                            collection_name = self.ml_collections["chat_sessions"]
                            doc_ref = self.db.collection(collection_name).document(session_id).collection(FILE_ANALYSES_SUBCOLLECTION).document(file_id) if self.db else None
                            payload = {
                                "file_id": file_id,
                                "analysis_data": analysis,
                                "saved_at": datetime.now(timezone.utc)
                            }
                            if self.db is None:
                                logger.info(f"(mock) Saved file analysis {file_id} for session {session_id}")
                                return True
                            await doc_ref.set(self._serialize_data(payload), merge=True)
                            return True
                        except Exception as e:
                            logger.error(f"❌ save_file_analysis error: {e}")
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
                            logger.error(f"❌ load_file_analysis error: {e}")
                            return None

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
                            logger.error(f"❌ delete_file_analysis error: {e}")
                            return False

                    async def save_file_analysis_with_geometric_split(self, session_id: str, file_id: str, analysis_data: Dict[str, Any]) -> bool:
                        try:
                            if not self.initialized:
                                await self.initialize()

                            file_type = analysis_data.get("analysis_data", {}).get("file_type")
                            dxf_result = analysis_data.get("analysis_data", {}).get("dxf_analysis_result")

                            if file_type == "dxf" and dxf_result and geometric_storage and geometric_storage.is_available():
                                geometric = dxf_result.get("dxf_analysis")
                                if geometric:
                                    geom_ref = await geometric_storage.store_geometric_data(session_id=session_id, file_id=file_id, geometric_data=geometric)
                                    if geom_ref:
                                        copy = dict(analysis_data)
                                        # Replace the heavy payload with lightweight marker
                                        if "analysis_data" in copy and "dxf_analysis_result" in copy["analysis_data"]:
                                            copy["analysis_data"]["dxf_analysis_result"] = {"status": "stored_in_gcs"}
                                        copy["geometric_ref"] = geom_ref
                                        copy["has_geometric_data"] = True
                                        copy["geometric_storage"] = "gcs"
                                        return await self.save_file_analysis(session_id, file_id, copy)
                            # Fallback: save whole analysis
                            analysis_data["has_geometric_data"] = False
                            analysis_data["geometric_storage"] = "none"
                            return await self.save_file_analysis(session_id, file_id, analysis_data)
                        except Exception as e:
                            logger.error(f"❌ save_file_analysis_with_geometric_split error: {e}")
                            return False

                    # Utility
                    def _deep_copy_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
                        import copy
                        return copy.deepcopy(data)

        logger.info("✅ ML collections ready")

    # --- Serialization / Validation Helpers (CLASS-LEVEL) ---
    def _serialize_data(self, data: Any) -> Any:
        """Recursively serialize ANY data type for Firestore"""
        from enum import Enum

        if data is None:
            return None
        if isinstance(data, (str, int, float, bool)):
            return data
        if isinstance(data, bytes):
            logger.warning(f"⚠️  Converting bytes object (size={len(data)}) to string")
            return f"<bytes:size={len(data)}>"
        if isinstance(data, datetime):
            return data
        if isinstance(data, Enum):
            return data.value
        if is_dataclass(data) and not isinstance(data, type):
            return self._serialize_data(asdict(data))
        if isinstance(data, dict):
            return {str(k): self._serialize_data(v) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            return [self._serialize_data(item) for item in data]
        if hasattr(data, 'to_dict') and callable(getattr(data, 'to_dict')):
            try:
                return self._serialize_data(data.to_dict())
            except Exception:
                pass
        if hasattr(data, '__dict__'):
            try:
                return self._serialize_data(data.__dict__)
            except Exception:
                pass
        return str(data)

    def _validate_firestore_safe(self, data: Any, path: str = "root") -> List[str]:
        """Validate data contains only Firestore-safe types"""
        problems: List[str] = []
        safe_types = (str, int, float, bool, type(None), datetime, list, dict)

        if not isinstance(data, safe_types):
            problems.append(f"{path}: {type(data).__name__}")
            return problems

        if isinstance(data, dict):
            for key, value in data.items():
                problems.extend(self._validate_firestore_safe(value, f"{path}.{key}"))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                problems.extend(self._validate_firestore_safe(item, f"{path}[{i}]"))

        return problems

    def _force_serialize_to_primitives(self, data: Any) -> Any:
        """Emergency serializer - converts EVERYTHING to primitives"""
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
            return [self._force_serialize_to_primitives(item) for item in data]
        try:
            return str(data)
        except Exception:
            return "<unserializable>"

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
        if not self.initialized: await self.initialize()

        collection_name = self.ml_collections["chat_sessions"]
        messages_ref = self.db.collection(collection_name).document(session_id).collection(MESSAGES_SUBCOLLECTION)

        # Build query ordered by timestamp (ascending) and limited
        query = messages_ref.order_by("timestamp")
        if limit:
            query = query.limit(limit)

        docs_stream = query.stream()
        results = []
        async for doc in docs_stream:
            d = doc.to_dict()
            d["_id"] = doc.id
            results.append(self._deserialize_data(d))

        # Return the list of messages
        return results

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
    
    # duplicate serializer removed - class-level helpers above are used

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