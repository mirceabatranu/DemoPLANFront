"""
Offer Learning Service - Main Orchestrator
Coordinates ingestion workflow and pattern learning
"""

import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import statistics

from google.cloud import firestore
from google.cloud.exceptions import NotFound

from src.models.offer_models import (
    ParsedOffer,
    IngestionResult,
    CategoryPattern,
    ItemFrequency,
    LearningStats,
    DetailLevel
)
from src.services.offer_parser_service import OfferParserService
from src.services.gcs_offer_storage import GCSOfferStorageService
from src.utils.currency_utils import calculate_percentage, get_value_range


logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

class FirestoreConfig:
    """Firestore collection names"""
    LEARNED_OFFERS = "learned_offers"
    LEARNED_PATTERNS = "learned_patterns"
    
    # Pattern document IDs
    DOC_CATEGORY_PATTERNS = "category_patterns"
    DOC_ITEM_FREQUENCIES = "item_frequencies"
    DOC_UNIT_PRICES = "unit_prices"
    DOC_LEARNING_METADATA = "learning_metadata"


# Confidence score thresholds
CONFIDENCE_THRESHOLDS = {
    'low': (1, 4, 0.3),      # 1-4 offers → 0.2-0.4
    'medium': (5, 9, 0.55),  # 5-9 offers → 0.5-0.6
    'good': (10, 19, 0.75),  # 10-19 offers → 0.7-0.8
    'high': (20, 999, 0.9)   # 20+ offers → 0.85-0.95
}


# ============================================================================
# OFFER LEARNING SERVICE
# ============================================================================

class OfferLearningService:
    """
    Main orchestrator for offer ingestion and pattern learning
    """
    
    def __init__(self):
        """Initialize service with dependencies"""
        self.parser = OfferParserService()
        self.storage = GCSOfferStorageService()
        self.db = firestore.Client()
        
        # Collection references
        self.offers_collection = self.db.collection(FirestoreConfig.LEARNED_OFFERS)
        self.patterns_collection = self.db.collection(FirestoreConfig.LEARNED_PATTERNS)
    
    # ========================================================================
    # INGESTION WORKFLOW
    # ========================================================================
    
    def ingest_offer(
        self,
        file_content: bytes,
        filename: str,
        metadata: Optional[Dict] = None
    ) -> IngestionResult:
        """
        Complete ingestion workflow:
        1. Parse offer file
        2. Generate unique offer_id
        3. Upload original to GCS
        4. Upload parsed JSON to GCS
        5. Save summary to Firestore
        6. Return result
        
        Args:
            file_content: Raw file bytes
            filename: Original filename
            metadata: Optional additional metadata
            
        Returns:
            IngestionResult with paths and statistics
        """
        result = IngestionResult(
            success=False,
            offer_id="",
            message=""
        )
        
        try:
            # STEP 1: Parse offer
            logger.info(f"Parsing offer: {filename}")
            parsed_offer = self.parser.parse_offer(file_content, filename)
            
            # STEP 2: Generate unique offer_id
            offer_id = self._generate_unique_offer_id()
            parsed_offer.offer_id = offer_id
            
            logger.info(f"Generated offer ID: {offer_id}")
            
            # STEP 3: Upload original file to GCS
            try:
                original_path = self.storage.upload_original(
                    file_content=file_content,
                    original_filename=filename,
                    offer_id=offer_id,
                    offer_date=parsed_offer.parsed_at
                )
                result.gcs_original_path = original_path
            except Exception as e:
                result.errors.append(f"Failed to upload original: {str(e)}")
                logger.error(f"Original upload failed: {e}")
            
            # STEP 4: Upload parsed JSON to GCS
            try:
                parsed_path = self.storage.upload_parsed_json(parsed_offer)
                result.gcs_parsed_path = parsed_path
            except Exception as e:
                result.errors.append(f"Failed to upload parsed JSON: {str(e)}")
                logger.error(f"Parsed JSON upload failed: {e}")
                # This is critical - abort if we can't store the parsed data
                raise
            
            # STEP 5: Save summary to Firestore
            try:
                firestore_summary = parsed_offer.to_firestore_summary()
                
                # Add GCS paths to summary
                firestore_summary['gcs_original_path'] = original_path
                firestore_summary['gcs_parsed_path'] = parsed_path
                
                # Add optional metadata
                if metadata:
                    firestore_summary['custom_metadata'] = metadata
                
                # Save to Firestore
                self.offers_collection.document(offer_id).set(firestore_summary)
                result.firestore_doc_id = offer_id
                
                logger.info(f"Saved to Firestore: {offer_id}")
                
            except Exception as e:
                result.errors.append(f"Failed to save to Firestore: {str(e)}")
                logger.error(f"Firestore save failed: {e}")
                raise
            
            # STEP 6: Build result
            result.success = True
            result.offer_id = offer_id
            result.message = f"Successfully ingested offer: {parsed_offer.project.project_name}"
            result.categories_found = len(parsed_offer.cost_breakdown.categories)
            result.items_extracted = parsed_offer.cost_breakdown.get_total_items()
            result.items_with_unit_pricing = len(
                parsed_offer.cost_breakdown.get_all_items_with_unit_pricing()
            )
            result.total_eur = parsed_offer.cost_breakdown.grand_total_eur
            result.detail_level = parsed_offer.detail_level.value
            result.warnings = parsed_offer.warnings
            
            logger.info(f"Ingestion complete: {offer_id}")
            return result
            
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            result.success = False
            result.message = f"Ingestion failed: {str(e)}"
            result.errors.append(str(e))
            return result
    
    def _generate_unique_offer_id(self) -> str:
        """
        Generate unique offer ID with daily counter
        Format: OFF_YYYYMMDD_NNN
        
        Returns:
            Unique offer ID
        """
        now = datetime.now()
        date_str = now.strftime("%Y%m%d")
        
        # Query today's offers to get counter
        today_prefix = f"OFF_{date_str}_"
        
        # Get all offers starting with today's prefix
        today_offers = (
            self.offers_collection
            .where('offer_id', '>=', today_prefix + '000')
            .where('offer_id', '<=', today_prefix + '999')
            .get()
        )
        
        # Counter is current count + 1
        counter = len(list(today_offers)) + 1
        
        offer_id = f"OFF_{date_str}_{counter:03d}"
        logger.info(f"Generated offer ID: {offer_id} (counter: {counter})")
        
        return offer_id
    
    # ========================================================================
    # PATTERN LEARNING
    # ========================================================================
    
    def run_learning(self) -> Dict:
        """
        Run complete pattern learning process:
        1. Query all offers from Firestore
        2. Load detailed data from GCS
        3. Calculate category patterns
        4. Calculate item frequencies
        5. Calculate unit price ranges
        6. Compute confidence scores
        7. Save patterns to Firestore
        
        Returns:
            Learning summary dict
        """
        logger.info("Starting pattern learning...")
        
        summary = {
            'success': False,
            'offers_processed': 0,
            'patterns_learned': {},
            'errors': []
        }
        
        try:
            # STEP 1: Get all offers
            offers_data = self._get_all_offers()
            summary['offers_processed'] = len(offers_data)
            
            if len(offers_data) == 0:
                summary['success'] = True
                summary['message'] = "No offers to learn from"
                return summary
            
            logger.info(f"Processing {len(offers_data)} offers")
            
            # STEP 2: Calculate category patterns
            category_patterns = self._calculate_category_patterns(offers_data)
            summary['patterns_learned']['category_splits'] = category_patterns
            
            # STEP 3: Calculate item frequencies
            item_frequencies = self._calculate_item_frequencies(offers_data)
            summary['patterns_learned']['item_frequencies'] = len(item_frequencies)
            
            # STEP 4: Calculate unit price ranges (for detailed offers)
            unit_price_patterns = self._calculate_unit_price_patterns(offers_data)
            summary['patterns_learned']['unit_price_types'] = len(unit_price_patterns)
            
            # STEP 5: Calculate confidence
            confidence = self._calculate_confidence(len(offers_data))
            summary['confidence_score'] = confidence
            
            # STEP 6: Save patterns to Firestore
            self._save_category_patterns(category_patterns, confidence)
            self._save_item_frequencies(item_frequencies)
            self._save_unit_price_patterns(unit_price_patterns)
            self._save_learning_metadata(len(offers_data), confidence)
            
            summary['success'] = True
            summary['message'] = f"Learning complete. Processed {len(offers_data)} offers."
            
            logger.info("Pattern learning complete")
            return summary
            
        except Exception as e:
            logger.error(f"Learning failed: {e}")
            summary['success'] = False
            summary['errors'].append(str(e))
            return summary
    
    def _get_all_offers(self) -> List[Dict]:
        """
        Get all offers from Firestore and load full data from GCS
        
        Returns:
            List of full offer data dicts
        """
        offers = []
        
        # Query all offers from Firestore
        firestore_offers = self.offers_collection.stream()
        
        for doc in firestore_offers:
            offer_summary = doc.to_dict()
            offer_id = offer_summary.get('offer_id')
            
            try:
                # Load full data from GCS
                full_data = self.storage.download_parsed_offer(offer_id)
                offers.append(full_data)
            except NotFound:
                logger.warning(f"GCS data not found for {offer_id}")
                # Use Firestore summary as fallback
                offers.append(offer_summary)
            except Exception as e:
                logger.error(f"Failed to load {offer_id}: {e}")
                continue
        
        return offers
    
    def _calculate_category_patterns(self, offers: List[Dict]) -> Dict:
        """
        Calculate category distribution patterns
        
        Returns:
            Dict with category percentages
        """
        category_totals = defaultdict(list)
        
        for offer in offers:
            grand_total = offer.get('cost_breakdown', {}).get('grand_total_eur', 0)
            if grand_total == 0:
                continue
            
            categories = offer.get('cost_breakdown', {}).get('categories', [])
            
            for cat in categories:
                cat_id = cat.get('category_id', 'UNKNOWN')
                cat_total = cat.get('total_eur', 0)
                percentage = calculate_percentage(cat_total, grand_total)
                category_totals[cat_id].append(percentage)
        
        # Calculate averages
        patterns = {}
        for cat_id, percentages in category_totals.items():
            if percentages:
                patterns[cat_id] = {
                    'avg_percent': round(statistics.mean(percentages), 2),
                    'min_percent': round(min(percentages), 2),
                    'max_percent': round(max(percentages), 2),
                    'std_dev': round(statistics.stdev(percentages), 2) if len(percentages) > 1 else 0.0,
                    'sample_size': len(percentages)
                }
        
        return patterns
    
    def _calculate_item_frequencies(self, offers: List[Dict]) -> Dict:
        """
        Calculate how often each item type appears
        
        Returns:
            Dict with item frequencies and statistics
        """
        item_stats = defaultdict(lambda: {
            'count': 0,
            'offer_ids': [],
            'values': []
        })
        
        total_offers = len(offers)
        
        for offer in offers:
            offer_id = offer.get('offer_id', 'unknown')
            categories = offer.get('cost_breakdown', {}).get('categories', [])
            
            seen_items = set()  # Track items per offer
            
            for cat in categories:
                items = cat.get('items', [])
                
                for item in items:
                    item_type = item.get('item_type', 'other')
                    
                    # Only count once per offer
                    if item_type not in seen_items:
                        item_stats[item_type]['count'] += 1
                        item_stats[item_type]['offer_ids'].append(offer_id)
                        seen_items.add(item_type)
                    
                    # Always track values
                    value = item.get('value_eur', 0)
                    if value > 0:
                        item_stats[item_type]['values'].append(value)
        
        # Calculate frequencies and ranges
        frequencies = {}
        for item_type, stats in item_stats.items():
            frequency = stats['count'] / total_offers if total_offers > 0 else 0
            
            frequencies[item_type] = {
                'frequency': round(frequency, 2),
                'appears_in_offers': stats['count'],
                'avg_value_eur': round(statistics.mean(stats['values']), 2) if stats['values'] else 0,
                'value_range': {
                    'min': round(min(stats['values']), 2) if stats['values'] else 0,
                    'max': round(max(stats['values']), 2) if stats['values'] else 0
                }
            }
        
        return frequencies
    
    def _calculate_unit_price_patterns(self, offers: List[Dict]) -> Dict:
        """
        Calculate unit price patterns for detailed offers
        
        Returns:
            Dict with unit price statistics per item type
        """
        unit_price_stats = defaultdict(lambda: {
            'unit_prices': [],
            'units': [],
            'sample_size': 0
        })
        
        for offer in offers:
            # Only process detailed offers
            detail_level = offer.get('detail_level', 'summary')
            if detail_level != DetailLevel.UNIT_PRICES.value:
                continue
            
            categories = offer.get('cost_breakdown', {}).get('categories', [])
            
            for cat in categories:
                items = cat.get('items', [])
                
                for item in items:
                    # Check if has unit pricing
                    unit_price = item.get('unit_price_eur')
                    unit = item.get('unit_normalized') or item.get('unit')
                    
                    if unit_price and unit:
                        item_type = item.get('item_type', 'other')
                        
                        unit_price_stats[item_type]['unit_prices'].append(unit_price)
                        unit_price_stats[item_type]['units'].append(unit)
                        unit_price_stats[item_type]['sample_size'] += 1
        
        # Calculate patterns
        patterns = {}
        for item_type, stats in unit_price_stats.items():
            if not stats['unit_prices']:
                continue
            
            # Find most common unit
            unit_counts = defaultdict(int)
            for unit in stats['units']:
                unit_counts[unit] += 1
            common_unit = max(unit_counts, key=unit_counts.get) if unit_counts else None
            
            patterns[item_type] = {
                'avg_unit_price_eur': round(statistics.mean(stats['unit_prices']), 2),
                'min_unit_price_eur': round(min(stats['unit_prices']), 2),
                'max_unit_price_eur': round(max(stats['unit_prices']), 2),
                'common_unit': common_unit,
                'sample_size': stats['sample_size']
            }
        
        return patterns
    
    def _calculate_confidence(self, sample_size: int) -> float:
        """
        Calculate confidence score based on sample size
        
        Args:
            sample_size: Number of offers
            
        Returns:
            Confidence score (0.0 - 1.0)
        """
        for level, (min_size, max_size, base_confidence) in CONFIDENCE_THRESHOLDS.items():
            if min_size <= sample_size <= max_size:
                # Interpolate within range
                range_size = max_size - min_size
                position = (sample_size - min_size) / range_size if range_size > 0 else 1.0
                
                if level == 'low':
                    return round(0.2 + (position * 0.2), 2)  # 0.2 - 0.4
                elif level == 'medium':
                    return round(0.5 + (position * 0.1), 2)  # 0.5 - 0.6
                elif level == 'good':
                    return round(0.7 + (position * 0.1), 2)  # 0.7 - 0.8
                elif level == 'high':
                    return round(0.85 + (min(position, 1.0) * 0.1), 2)  # 0.85 - 0.95
        
        return 0.95  # Max confidence for very large samples
    
    # ========================================================================
    # SAVE PATTERNS TO FIRESTORE
    # ========================================================================
    
    def _save_category_patterns(self, patterns: Dict, confidence: float):
        """Save category patterns to Firestore"""
        doc_ref = self.patterns_collection.document(FirestoreConfig.DOC_CATEGORY_PATTERNS)
        
        data = {
            'patterns': patterns,
            'confidence': confidence,
            'updated_at': datetime.now().isoformat()
        }
        
        doc_ref.set(data)
        logger.info("Saved category patterns")
    
    def _save_item_frequencies(self, frequencies: Dict):
        """Save item frequencies to Firestore"""
        doc_ref = self.patterns_collection.document(FirestoreConfig.DOC_ITEM_FREQUENCIES)
        
        data = {
            'frequencies': frequencies,
            'updated_at': datetime.now().isoformat()
        }
        
        doc_ref.set(data)
        logger.info(f"Saved {len(frequencies)} item frequencies")
    
    def _save_unit_price_patterns(self, patterns: Dict):
        """Save unit price patterns to Firestore"""
        doc_ref = self.patterns_collection.document(FirestoreConfig.DOC_UNIT_PRICES)
        
        data = {
            'unit_prices': patterns,
            'updated_at': datetime.now().isoformat()
        }
        
        doc_ref.set(data)
        logger.info(f"Saved {len(patterns)} unit price patterns")
    
    def _save_learning_metadata(self, total_offers: int, confidence: float):
        """Save learning run metadata"""
        doc_ref = self.patterns_collection.document(FirestoreConfig.DOC_LEARNING_METADATA)
        
        data = {
            'last_learning_run': datetime.now().isoformat(),
            'total_offers_processed': total_offers,
            'confidence_score': confidence,
            'status': 'completed'
        }
        
        doc_ref.set(data)
        logger.info("Saved learning metadata")
    
    # ========================================================================
    # QUERY OPERATIONS
    # ========================================================================
    
    def get_offer_by_id(self, offer_id: str) -> Optional[Dict]:
        """
        Get full offer data by ID
        
        Args:
            offer_id: Offer ID
            
        Returns:
            Full offer dict from GCS or None
        """
        try:
            return self.storage.download_parsed_offer(offer_id)
        except NotFound:
            logger.warning(f"Offer {offer_id} not found")
            return None
    
    def list_offers(
        self,
        limit: int = 50,
        detail_level: Optional[str] = None
    ) -> List[Dict]:
        """
        List offers with optional filters
        
        Args:
            limit: Max number to return
            detail_level: Filter by detail_level (optional)
            
        Returns:
            List of offer summaries
        """
        query = self.offers_collection.limit(limit)
        
        if detail_level:
            query = query.where('detail_level', '==', detail_level)
        
        results = []
        for doc in query.stream():
            results.append(doc.to_dict())
        
        return results
    
    def get_learning_stats(self) -> Dict:
        """
        Get current learning statistics
        
        Returns:
            Statistics dict
        """
        # Count offers
        total_offers = len(list(self.offers_collection.stream()))
        
        # Count detailed offers
        detailed_query = self.offers_collection.where(
            'detail_level', '==', DetailLevel.UNIT_PRICES.value
        )
        detailed_offers = len(list(detailed_query.stream()))
        
        # Get learning metadata
        try:
            metadata_doc = self.patterns_collection.document(
                FirestoreConfig.DOC_LEARNING_METADATA
            ).get()
            
            if metadata_doc.exists:
                metadata = metadata_doc.to_dict()
            else:
                metadata = {}
        except Exception:
            metadata = {}
        
        # Get pattern counts
        try:
            cat_doc = self.patterns_collection.document(
                FirestoreConfig.DOC_CATEGORY_PATTERNS
            ).get()
            category_patterns_count = len(cat_doc.to_dict().get('patterns', {})) if cat_doc.exists else 0
        except Exception:
            category_patterns_count = 0
        
        try:
            freq_doc = self.patterns_collection.document(
                FirestoreConfig.DOC_ITEM_FREQUENCIES
            ).get()
            item_types_count = len(freq_doc.to_dict().get('frequencies', {})) if freq_doc.exists else 0
        except Exception:
            item_types_count = 0
        
        return {
            'total_offers_ingested': total_offers,
            'offers_with_unit_pricing': detailed_offers,
            'last_learning_run': metadata.get('last_learning_run'),
            'confidence_score': metadata.get('confidence_score', 0.0),
            'category_patterns_count': category_patterns_count,
            'item_types_tracked': item_types_count
        }
    
    def get_patterns(self) -> Dict:
        """
        Get all learned patterns
        
        Returns:
            Dict with all pattern data
        """
        patterns = {}
        
        # Get category patterns
        try:
            cat_doc = self.patterns_collection.document(
                FirestoreConfig.DOC_CATEGORY_PATTERNS
            ).get()
            if cat_doc.exists:
                patterns['category_patterns'] = cat_doc.to_dict()
        except Exception as e:
            logger.error(f"Failed to get category patterns: {e}")
        
        # Get item frequencies
        try:
            freq_doc = self.patterns_collection.document(
                FirestoreConfig.DOC_ITEM_FREQUENCIES
            ).get()
            if freq_doc.exists:
                patterns['item_frequencies'] = freq_doc.to_dict()
        except Exception as e:
            logger.error(f"Failed to get item frequencies: {e}")
        
        # Get unit prices
        try:
            price_doc = self.patterns_collection.document(
                FirestoreConfig.DOC_UNIT_PRICES
            ).get()
            if price_doc.exists:
                patterns['unit_prices'] = price_doc.to_dict()
        except Exception as e:
            logger.error(f"Failed to get unit prices: {e}")
        
        return patterns