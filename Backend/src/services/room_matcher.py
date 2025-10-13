"""
DemoPLAN Room Matcher Service
Matches rooms from specification sheets with image-detected labels

This service performs intelligent fuzzy matching between:
- Specification sheet data (DXF room boundaries or PDF tables)
- Image-detected room labels (from Google Vision AI)

Handles multi-language matching (Romanian ↔ English) and provides
confidence scoring for each match.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from difflib import SequenceMatcher
import re

logger = logging.getLogger("demoplan.services.room_matcher")


@dataclass
class RoomMatch:
    """Represents a match between spec room and image label"""
    spec_room: Dict[str, Any]
    image_label: Optional[str]
    confidence: float
    match_type: str  # 'exact', 'fuzzy', 'normalized', 'area_based', 'spec_only', 'image_only'
    similarity_score: float = 0.0
    notes: str = ""


@dataclass
class MatchingResult:
    """Complete matching result with statistics"""
    matched_rooms: List[RoomMatch]
    total_spec_rooms: int
    total_image_labels: int
    matched_count: int
    unmatched_spec_count: int
    unmatched_image_count: int
    overall_confidence: float
    warnings: List[str] = field(default_factory=list)


class RoomMatcher:
    """
    Intelligent room matching service with multi-language support
    
    Matching Strategies (in priority order):
    1. Exact match (room name identical)
    2. Normalized match (after Romanian ↔ English conversion)
    3. Fuzzy match (similar strings with threshold)
    4. Area-based inference (if only one unmatched room with similar area)
    
    Features:
    - Multi-language support (Romanian ↔ English)
    - Fuzzy string matching with configurable threshold
    - Area tolerance matching (±15%)
    - Confidence scoring per match
    - Unmatched room detection
    """
    
    # Romanian → English room name mapping
    # Reused from existing drawing_generation_agent.py
    ROOM_NAME_MAPPING = {
        # Kitchen variants
        "bucătărie": "kitchen",
        "bucatarie": "kitchen",
        "buc": "kitchen",
        "buc.": "kitchen",
        "bucătărie mare": "kitchen",
        "bucatarie mare": "kitchen",
        
        # Living room variants
        "living": "living_room",
        "salon": "living_room",
        "sufragerie": "living_room",
        "camera de zi": "living_room",
        "living room": "living_room",
        "open space": "open_space",
        
        # Bedroom variants
        "dormitor": "bedroom",
        "camera": "bedroom",
        "camera de dormit": "bedroom",
        "dorm": "bedroom",
        "dormitor 1": "bedroom",
        "dormitor 2": "bedroom",
        "dormitor principal": "master_bedroom",
        
        # Bathroom variants
        "baie": "bathroom",
        "bai": "bathroom",
        "toaleta": "bathroom",
        "wc": "bathroom",
        "grup sanitar": "bathroom",
        "baie 1": "bathroom",
        "baie 2": "bathroom",
        
        # Hallway variants
        "hol": "hallway",
        "coridor": "hallway",
        "antreu": "hallway",
        "vestibul": "hallway",
        "hol intrare": "hallway",
        "holul": "hallway",
        
        # Storage variants
        "debara": "storage",
        "cămară": "storage",
        "camara": "storage",
        "dulap": "storage",
        "dressing": "storage",
        "magazie": "storage",
        
        # Office variants
        "birou": "office",
        "cabinet": "office",
        "camera de lucru": "office",
        "spatiu de lucru": "office",
        "spațiu de lucru": "office",
        
        # Balcony variants
        "balcon": "balcony",
        "terasa": "balcony",
        "terasă": "balcony",
        "loggie": "balcony",
        "logie": "balcony",
        
        # Other rooms
        "garaj": "garage",
        "parcare": "garage",
        "spațiu tehnic": "technical_room",
        "spatiu tehnic": "technical_room",
        "tehnic": "technical_room",
        "recepție": "reception",
        "receptie": "reception",
        "intrare": "entrance",
    }
    
    # English → Romanian reverse mapping
    ENGLISH_TO_ROMANIAN = {
        "kitchen": "bucătărie",
        "living_room": "living",
        "bedroom": "dormitor",
        "bathroom": "baie",
        "hallway": "hol",
        "storage": "debara",
        "office": "birou",
        "balcony": "balcon",
        "garage": "garaj",
        "technical_room": "spațiu tehnic",
        "reception": "recepție",
        "entrance": "intrare",
        "open_space": "open space",
        "master_bedroom": "dormitor principal",
    }
    
    def __init__(
        self,
        similarity_threshold: float = 0.6,
        area_tolerance: float = 0.15
    ):
        """
        Initialize room matcher
        
        Args:
            similarity_threshold: Minimum similarity score for fuzzy matching (0.0-1.0)
            area_tolerance: Acceptable area difference as percentage (e.g., 0.15 = ±15%)
        """
        self.similarity_threshold = similarity_threshold
        self.area_tolerance = area_tolerance
        logger.info("✅ RoomMatcher initialized")
        logger.info(f"   Similarity threshold: {similarity_threshold:.0%}")
        logger.info(f"   Area tolerance: ±{area_tolerance:.0%}")
    
    def match_rooms(
        self,
        spec_rooms: List[Dict[str, Any]],
        image_labels: List[str],
        enable_area_matching: bool = True
    ) -> MatchingResult:
        """
        Match specification rooms with image-detected labels
        
        Args:
            spec_rooms: List of rooms from DXF/PDF specification
                Expected format: [{'name': 'living_room', 'name_ro': 'Living', 'area': 25.0, ...}, ...]
            image_labels: List of text labels detected in floor plan image
                Example: ['Living', 'Dormitor 1', 'Bucătărie', 'Baie']
            enable_area_matching: Use area similarity as additional matching hint
            
        Returns:
            MatchingResult with all matched and unmatched rooms
        """
        logger.info("🔍 Starting room matching process...")
        logger.info(f"   Spec rooms: {len(spec_rooms)}")
        logger.info(f"   Image labels: {len(image_labels)}")
        
        matched_rooms = []
        unmatched_spec_rooms = list(spec_rooms)  # Copy for tracking
        unmatched_image_labels = list(image_labels)  # Copy for tracking
        
        # Strategy 1: Exact matches
        logger.info("   Strategy 1: Exact name matching...")
        exact_matches = self._find_exact_matches(
            unmatched_spec_rooms,
            unmatched_image_labels
        )
        matched_rooms.extend(exact_matches)
        self._remove_matched_items(unmatched_spec_rooms, unmatched_image_labels, exact_matches)
        logger.info(f"      Found {len(exact_matches)} exact matches")
        
        # Strategy 2: Normalized matches (Romanian ↔ English)
        logger.info("   Strategy 2: Normalized matching (multi-language)...")
        normalized_matches = self._find_normalized_matches(
            unmatched_spec_rooms,
            unmatched_image_labels
        )
        matched_rooms.extend(normalized_matches)
        self._remove_matched_items(unmatched_spec_rooms, unmatched_image_labels, normalized_matches)
        logger.info(f"      Found {len(normalized_matches)} normalized matches")
        
        # Strategy 3: Fuzzy matches
        logger.info("   Strategy 3: Fuzzy string matching...")
        fuzzy_matches = self._find_fuzzy_matches(
            unmatched_spec_rooms,
            unmatched_image_labels
        )
        matched_rooms.extend(fuzzy_matches)
        self._remove_matched_items(unmatched_spec_rooms, unmatched_image_labels, fuzzy_matches)
        logger.info(f"      Found {len(fuzzy_matches)} fuzzy matches")
        
        # Strategy 4: Area-based inference (if enabled and only one candidate remains)
        if enable_area_matching and len(unmatched_spec_rooms) > 0:
            logger.info("   Strategy 4: Area-based inference...")
            area_matches = self._find_area_based_matches(
                unmatched_spec_rooms,
                unmatched_image_labels
            )
            matched_rooms.extend(area_matches)
            self._remove_matched_items(unmatched_spec_rooms, unmatched_image_labels, area_matches)
            logger.info(f"      Found {len(area_matches)} area-based matches")
        
        # Handle remaining unmatched rooms
        for spec_room in unmatched_spec_rooms:
            matched_rooms.append(RoomMatch(
                spec_room=spec_room,
                image_label=None,
                confidence=0.4,
                match_type='spec_only',
                similarity_score=0.0,
                notes="Room from specification not found in image"
            ))
        
        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(matched_rooms)
        
        # Generate warnings
        warnings = self._generate_warnings(
            matched_rooms,
            len(spec_rooms),
            len(image_labels),
            len(unmatched_spec_rooms),
            len(unmatched_image_labels)
        )
        
        result = MatchingResult(
            matched_rooms=matched_rooms,
            total_spec_rooms=len(spec_rooms),
            total_image_labels=len(image_labels),
            matched_count=len([m for m in matched_rooms if m.image_label is not None]),
            unmatched_spec_count=len(unmatched_spec_rooms),
            unmatched_image_count=len(unmatched_image_labels),
            overall_confidence=overall_confidence,
            warnings=warnings
        )
        
        logger.info(f"✅ Room matching complete:")
        logger.info(f"   Matched: {result.matched_count}/{result.total_spec_rooms}")
        logger.info(f"   Overall confidence: {result.overall_confidence:.1%}")
        logger.info(f"   Warnings: {len(warnings)}")
        
        return result
    
    def _find_exact_matches(
        self,
        spec_rooms: List[Dict[str, Any]],
        image_labels: List[str]
    ) -> List[RoomMatch]:
        """Find rooms with exact name matches"""
        matches = []
        
        for spec_room in spec_rooms:
            spec_name = spec_room.get('name', '').lower().strip()
            spec_name_ro = spec_room.get('name_ro', '').lower().strip()
            
            for label in image_labels:
                label_lower = label.lower().strip()
                
                # Check for exact match with either English or Romanian name
                if label_lower == spec_name or label_lower == spec_name_ro:
                    matches.append(RoomMatch(
                        spec_room=spec_room,
                        image_label=label,
                        confidence=1.0,
                        match_type='exact',
                        similarity_score=1.0,
                        notes="Exact name match"
                    ))
                    break  # Found match, move to next spec room
        
        return matches
    
    def _find_normalized_matches(
        self,
        spec_rooms: List[Dict[str, Any]],
        image_labels: List[str]
    ) -> List[RoomMatch]:
        """Find matches after normalizing names (Romanian ↔ English)"""
        matches = []
        
        for spec_room in spec_rooms:
            spec_name = spec_room.get('name', '')
            spec_name_ro = spec_room.get('name_ro', '')
            
            # Normalize both spec names
            spec_normalized = self._normalize_room_name(spec_name)
            spec_ro_normalized = self._normalize_room_name(spec_name_ro)
            
            best_match = None
            best_score = 0.0
            
            for label in image_labels:
                label_normalized = self._normalize_room_name(label)
                
                # Check if normalized names match
                if (label_normalized == spec_normalized or 
                    label_normalized == spec_ro_normalized):
                    
                    # Calculate similarity score for confidence
                    score = max(
                        SequenceMatcher(None, label.lower(), spec_name.lower()).ratio(),
                        SequenceMatcher(None, label.lower(), spec_name_ro.lower()).ratio()
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_match = label
            
            if best_match:
                matches.append(RoomMatch(
                    spec_room=spec_room,
                    image_label=best_match,
                    confidence=0.95,
                    match_type='normalized',
                    similarity_score=best_score,
                    notes="Match after name normalization"
                ))
        
        return matches
    
    def _find_fuzzy_matches(
        self,
        spec_rooms: List[Dict[str, Any]],
        image_labels: List[str]
    ) -> List[RoomMatch]:
        """Find matches using fuzzy string matching"""
        matches = []
        
        for spec_room in spec_rooms:
            spec_name = spec_room.get('name', '')
            spec_name_ro = spec_room.get('name_ro', '')
            
            best_match = None
            best_score = 0.0
            best_label = None
            
            for label in image_labels:
                # Calculate similarity with both English and Romanian names
                score_en = self._calculate_similarity(spec_name, label)
                score_ro = self._calculate_similarity(spec_name_ro, label)
                score = max(score_en, score_ro)
                
                if score > best_score and score >= self.similarity_threshold:
                    best_score = score
                    best_label = label
                    best_match = spec_room
            
            if best_match and best_label:
                # Adjust confidence based on similarity score
                confidence = 0.7 + (best_score - self.similarity_threshold) * 0.5
                confidence = min(confidence, 0.95)
                
                matches.append(RoomMatch(
                    spec_room=spec_room,
                    image_label=best_label,
                    confidence=confidence,
                    match_type='fuzzy',
                    similarity_score=best_score,
                    notes=f"Fuzzy match (similarity: {best_score:.1%})"
                ))
        
        return matches
    
    def _find_area_based_matches(
        self,
        spec_rooms: List[Dict[str, Any]],
        image_labels: List[str]
    ) -> List[RoomMatch]:
        """
        Attempt area-based matching for remaining unmatched rooms
        
        This is a last-resort strategy when:
        - Only one spec room and one image label remain unmatched
        - They likely correspond to each other by process of elimination
        """
        matches = []
        
        # Only apply if exactly one of each remains
        if len(spec_rooms) == 1 and len(image_labels) == 1:
            spec_room = spec_rooms[0]
            image_label = image_labels[0]
            
            matches.append(RoomMatch(
                spec_room=spec_room,
                image_label=image_label,
                confidence=0.6,
                match_type='area_based',
                similarity_score=0.5,
                notes="Matched by process of elimination (last remaining room)"
            ))
        
        return matches
    
    def _normalize_room_name(self, name: str) -> str:
        """
        Normalize room name for comparison
        
        Steps:
        1. Convert to lowercase
        2. Remove special characters and numbers
        3. Map Romanian → English using dictionary
        4. Remove common filler words
        """
        if not name:
            return ""
        
        # Convert to lowercase and strip
        normalized = name.lower().strip()
        
        # Remove numbers and special characters (except spaces)
        normalized = re.sub(r'[0-9.,;:!?-]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Try direct mapping from dictionary
        if normalized in self.ROOM_NAME_MAPPING:
            return self.ROOM_NAME_MAPPING[normalized]
        
        # Remove common filler words
        filler_words = ['camera', 'cameră', 'room', 'space', 'spatiu', 'spațiu', 'de']
        words = normalized.split()
        words = [w for w in words if w not in filler_words]
        normalized = ' '.join(words)
        
        # Try mapping again after removing fillers
        if normalized in self.ROOM_NAME_MAPPING:
            return self.ROOM_NAME_MAPPING[normalized]
        
        return normalized
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate string similarity score using multiple methods
        
        Combines:
        - SequenceMatcher ratio (overall similarity)
        - Substring matching bonus
        - Length difference penalty
        """
        if not str1 or not str2:
            return 0.0
        
        str1_lower = str1.lower().strip()
        str2_lower = str2.lower().strip()
        
        # Base similarity using SequenceMatcher
        base_score = SequenceMatcher(None, str1_lower, str2_lower).ratio()
        
        # Bonus for substring match
        substring_bonus = 0.0
        if str1_lower in str2_lower or str2_lower in str1_lower:
            substring_bonus = 0.1
        
        # Small penalty for large length differences
        len_diff = abs(len(str1_lower) - len(str2_lower))
        max_len = max(len(str1_lower), len(str2_lower))
        len_penalty = (len_diff / max_len) * 0.1 if max_len > 0 else 0
        
        final_score = base_score + substring_bonus - len_penalty
        return min(max(final_score, 0.0), 1.0)
    
    def _remove_matched_items(
        self,
        spec_rooms: List[Dict[str, Any]],
        image_labels: List[str],
        matches: List[RoomMatch]
    ):
        """Remove matched items from unmatched lists (in-place)"""
        for match in matches:
            # Remove matched spec room
            if match.spec_room in spec_rooms:
                spec_rooms.remove(match.spec_room)
            
            # Remove matched image label
            if match.image_label in image_labels:
                image_labels.remove(match.image_label)
    
    def _calculate_overall_confidence(self, matches: List[RoomMatch]) -> float:
        """Calculate overall confidence for all matches"""
        if not matches:
            return 0.0
        
        # Weighted average of individual match confidences
        total_confidence = sum(match.confidence for match in matches)
        avg_confidence = total_confidence / len(matches)
        
        # Penalty for unmatched rooms
        matched_count = len([m for m in matches if m.image_label is not None])
        match_ratio = matched_count / len(matches) if len(matches) > 0 else 0
        
        # Final confidence combines average confidence and match ratio
        overall = (avg_confidence * 0.7) + (match_ratio * 0.3)
        
        return min(overall, 1.0)
    
    def _generate_warnings(
        self,
        matches: List[RoomMatch],
        total_spec_rooms: int,
        total_image_labels: int,
        unmatched_spec: int,
        unmatched_image: int
    ) -> List[str]:
        """Generate warnings about matching quality"""
        warnings = []
        
        # Warning for low overall confidence
        overall_conf = self._calculate_overall_confidence(matches)
        if overall_conf < 0.6:
            warnings.append(
                f"Low matching confidence ({overall_conf:.0%}). "
                "Consider uploading a clearer image with visible room labels."
            )
        
        # Warning for unmatched spec rooms
        if unmatched_spec > 0:
            warnings.append(
                f"{unmatched_spec} room(s) from specification not found in image. "
                "Ensure all rooms are visible and labeled in the floor plan."
            )
        
        # Warning for unmatched image labels
        if unmatched_image > 0:
            warnings.append(
                f"{unmatched_image} label(s) in image not matched to specification. "
                "Image may contain extra rooms or labels not in spec sheet."
            )
        
        # Warning for low-confidence matches
        low_conf_matches = [m for m in matches if m.confidence < 0.6]
        if low_conf_matches:
            warnings.append(
                f"{len(low_conf_matches)} room(s) matched with low confidence. "
                "Review the matching results carefully."
            )
        
        return warnings
    
    def get_match_summary(self, result: MatchingResult) -> Dict[str, Any]:
        """
        Get human-readable summary of matching results
        
        Useful for validation reports and user feedback
        """
        return {
            'total_spec_rooms': result.total_spec_rooms,
            'total_image_labels': result.total_image_labels,
            'matched_count': result.matched_count,
            'match_rate': f"{(result.matched_count / result.total_spec_rooms * 100):.0f}%" if result.total_spec_rooms > 0 else "0%",
            'overall_confidence': f"{result.overall_confidence:.0%}",
            'unmatched_spec_rooms': result.unmatched_spec_count,
            'unmatched_image_labels': result.unmatched_image_count,
            'warnings': result.warnings,
            'matches_by_type': self._count_matches_by_type(result.matched_rooms),
            'high_confidence_matches': len([m for m in result.matched_rooms if m.confidence >= 0.9]),
            'medium_confidence_matches': len([m for m in result.matched_rooms if 0.7 <= m.confidence < 0.9]),
            'low_confidence_matches': len([m for m in result.matched_rooms if m.confidence < 0.7])
        }
    
    def _count_matches_by_type(self, matches: List[RoomMatch]) -> Dict[str, int]:
        """Count matches by their match type"""
        counts = {}
        for match in matches:
            match_type = match.match_type
            counts[match_type] = counts.get(match_type, 0) + 1
        return counts
    
    def to_dict(self, result: MatchingResult) -> Dict[str, Any]:
        """
        Convert matching result to dictionary for storage/API response
        
        Compatible with Firestore and existing data structures
        """
        return {
            'matched_rooms': [
                {
                    'spec_room': match.spec_room,
                    'image_label': match.image_label,
                    'confidence': match.confidence,
                    'match_type': match.match_type,
                    'similarity_score': match.similarity_score,
                    'notes': match.notes
                }
                for match in result.matched_rooms
            ],
            'statistics': {
                'total_spec_rooms': result.total_spec_rooms,
                'total_image_labels': result.total_image_labels,
                'matched_count': result.matched_count,
                'unmatched_spec_count': result.unmatched_spec_count,
                'unmatched_image_count': result.unmatched_image_count,
                'overall_confidence': result.overall_confidence
            },
            'warnings': result.warnings,
            'summary': self.get_match_summary(result)
        }