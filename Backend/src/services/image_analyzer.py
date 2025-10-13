"""
DemoPLAN Image Analyzer Service
Google Cloud Vision AI integration for floor plan image analysis

This service analyzes uploaded floor plan images to extract:
- Room labels and text annotations
- Image dimensions and quality metrics
- Detected objects and spatial information
- Confidence scoring for reliability assessment

Uses the same GCP infrastructure as the existing OCR service.
"""

import logging
import io
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from google.cloud import vision
from google.api_core import exceptions as google_exceptions

logger = logging.getLogger("demoplan.services.image_analyzer")


@dataclass
class RoomLabel:
    """Represents a room label detected in the image"""
    text: str
    confidence: float
    bounding_box: List[Tuple[int, int]]  # [(x,y), (x,y), (x,y), (x,y)]
    position: Tuple[int, int]  # Center point (x, y)


@dataclass
class ImageDimensions:
    """Image size and aspect information"""
    width: int
    height: int
    aspect_ratio: float


@dataclass
class ImageAnalysisResult:
    """Complete image analysis result"""
    room_labels: List[RoomLabel]
    image_dimensions: ImageDimensions
    all_text_annotations: List[str]
    detected_objects: List[Dict[str, Any]]
    confidence: float
    quality_score: float
    processing_time_ms: float
    warnings: List[str] = field(default_factory=list)
    cost_estimate: float = 0.0


class ImageAnalyzerService:
    """
    Google Cloud Vision AI service for floor plan image analysis
    
    Integrates with existing GCP infrastructure:
    - Same project ID as OCR service (1041867695241)
    - Same authentication mechanism
    - Similar error handling and logging patterns
    
    Features:
    - Text detection (room labels, dimensions)
    - Object localization (furniture, fixtures)
    - Label detection (categories, features)
    - Quality assessment
    - Cost estimation
    """
    
    def __init__(
        self,
        project_id: str = "1041867695241",
        cost_per_detection: float = 0.0015  # $1.50 per 1000 images
    ):
        """
        Initialize Image Analyzer with Google Vision AI
        
        Args:
            project_id: GCP project ID (same as OCR service)
            cost_per_detection: Cost per image analysis
        """
        try:
            self.client = vision.ImageAnnotatorClient()
            self.project_id = project_id
            self.cost_per_detection = cost_per_detection
            logger.info("✅ ImageAnalyzerService initialized successfully")
            logger.info(f"   Project ID: {project_id}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Vision AI client: {e}")
            raise
    
    async def analyze_floor_plan(
        self,
        image_content: bytes,
        filename: str,
        min_confidence: float = 0.5
    ) -> ImageAnalysisResult:
        """
        Analyze floor plan image using Google Vision AI
        
        This is the main entry point for image analysis. It performs:
        1. Text detection for room labels
        2. Object localization for spatial features
        3. Quality assessment
        4. Confidence scoring
        
        Args:
            image_content: Raw image bytes (JPG, PNG)
            filename: Original filename for logging
            min_confidence: Minimum confidence threshold for results
            
        Returns:
            ImageAnalysisResult with extracted data
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"🖼️  Starting image analysis for: {filename}")
            logger.info(f"   Image size: {len(image_content) / 1024:.1f} KB")
            
            # Create Vision API image object
            image = vision.Image(content=image_content)
            
            # Run comprehensive analysis
            logger.info("   Running Vision AI detection features...")
            
            # Execute multiple detection features in parallel
            response = self.client.annotate_image({
                'image': image,
                'features': [
                    {'type_': vision.Feature.Type.TEXT_DETECTION},
                    {'type_': vision.Feature.Type.OBJECT_LOCALIZATION},
                    {'type_': vision.Feature.Type.LABEL_DETECTION},
                    {'type_': vision.Feature.Type.IMAGE_PROPERTIES}
                ]
            })
            
            # Check for API errors
            if response.error.message:
                raise Exception(f"Vision API error: {response.error.message}")
            
            # Extract image dimensions
            image_dims = self._extract_image_dimensions(image_content)
            
            # Extract room labels from text annotations
            room_labels = self._extract_room_labels(response, min_confidence)
            
            # Extract all text for matching
            all_text = self._extract_all_text(response)
            
            # Extract detected objects
            detected_objects = self._extract_objects(response)
            
            # Calculate confidence score
            confidence = self._calculate_overall_confidence(response, room_labels)
            
            # Assess image quality
            quality_score = self._assess_image_quality(response, image_dims)
            
            # Generate warnings
            warnings = self._generate_warnings(
                confidence,
                quality_score,
                room_labels,
                image_dims
            )
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Estimate cost
            cost = self.cost_per_detection
            
            logger.info(f"✅ Image analysis complete:")
            logger.info(f"   Room labels: {len(room_labels)}")
            logger.info(f"   Text annotations: {len(all_text)}")
            logger.info(f"   Objects detected: {len(detected_objects)}")
            logger.info(f"   Confidence: {confidence:.1%}")
            logger.info(f"   Quality score: {quality_score:.1%}")
            logger.info(f"   Processing time: {processing_time:.0f}ms")
            logger.info(f"   Cost: ${cost:.4f}")
            
            return ImageAnalysisResult(
                room_labels=room_labels,
                image_dimensions=image_dims,
                all_text_annotations=all_text,
                detected_objects=detected_objects,
                confidence=confidence,
                quality_score=quality_score,
                processing_time_ms=processing_time,
                warnings=warnings,
                cost_estimate=cost
            )
            
        except google_exceptions.GoogleAPIError as e:
            logger.error(f"❌ Google Vision API error: {e}")
            return self._create_error_result(
                f"Vision API error: {str(e)}",
                processing_time=(datetime.now() - start_time).total_seconds() * 1000
            )
        except Exception as e:
            logger.error(f"❌ Image analysis failed: {e}", exc_info=True)
            return self._create_error_result(
                f"Analysis failed: {str(e)}",
                processing_time=(datetime.now() - start_time).total_seconds() * 1000
            )
    
    def _extract_image_dimensions(self, image_content: bytes) -> ImageDimensions:
        """Extract image dimensions from bytes"""
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(image_content))
            width, height = img.size
            aspect_ratio = width / height if height > 0 else 1.0
            
            return ImageDimensions(
                width=width,
                height=height,
                aspect_ratio=aspect_ratio
            )
        except Exception as e:
            logger.warning(f"Could not extract image dimensions: {e}")
            return ImageDimensions(width=0, height=0, aspect_ratio=1.0)
    
    def _extract_room_labels(
        self,
        response: vision.AnnotateImageResponse,
        min_confidence: float
    ) -> List[RoomLabel]:
        """
        Extract room labels from text annotations
        
        Focuses on text that likely represents room names:
        - Short text blocks (1-3 words)
        - High confidence detections
        - Romanian and English room keywords
        """
        room_labels = []
        
        # Room keywords (Romanian and English)
        room_keywords = {
            'living', 'dormitor', 'bedroom', 'bucătărie', 'bucatarie', 
            'kitchen', 'baie', 'bathroom', 'hol', 'hallway', 'birou', 
            'office', 'recepție', 'receptie', 'reception', 'cameră', 
            'camera', 'room', 'spațiu', 'spatiu', 'space', 'salon',
            'open space', 'vestibul', 'holul', 'debara', 'storage',
            'balcon', 'balcony', 'terasa', 'terrace'
        }
        
        if not response.text_annotations:
            logger.warning("   No text annotations found in image")
            return room_labels
        
        # Skip first annotation (it's the full page text)
        for annotation in response.text_annotations[1:]:
            text = annotation.description.strip()
            confidence = 0.9  # Vision API doesn't provide per-text confidence
            
            # Filter for potential room labels
            text_lower = text.lower()
            word_count = len(text.split())
            
            # Check if this looks like a room label
            is_room_label = (
                word_count <= 3 and  # Short text
                len(text) >= 2 and  # Not just single character
                (any(keyword in text_lower for keyword in room_keywords) or
                 word_count <= 2)  # Or any short text
            )
            
            if is_room_label and confidence >= min_confidence:
                # Extract bounding box
                vertices = annotation.bounding_poly.vertices
                bounding_box = [(v.x, v.y) for v in vertices]
                
                # Calculate center position
                center_x = sum(v.x for v in vertices) / len(vertices)
                center_y = sum(v.y for v in vertices) / len(vertices)
                
                room_labels.append(RoomLabel(
                    text=text,
                    confidence=confidence,
                    bounding_box=bounding_box,
                    position=(int(center_x), int(center_y))
                ))
        
        logger.info(f"   Extracted {len(room_labels)} potential room labels")
        return room_labels
    
    def _extract_all_text(self, response: vision.AnnotateImageResponse) -> List[str]:
        """Extract all text annotations for matching"""
        if not response.text_annotations:
            return []
        
        # First annotation contains all text, split it
        full_text = response.text_annotations[0].description
        
        # Split by newlines and clean
        all_text = [
            line.strip() 
            for line in full_text.split('\n') 
            if line.strip()
        ]
        
        return all_text
    
    def _extract_objects(
        self,
        response: vision.AnnotateImageResponse
    ) -> List[Dict[str, Any]]:
        """Extract detected objects (furniture, fixtures, etc.)"""
        objects = []
        
        if not response.localized_object_annotations:
            return objects
        
        for obj in response.localized_object_annotations:
            # Only include reasonably confident detections
            if obj.score >= 0.5:
                objects.append({
                    'name': obj.name,
                    'confidence': obj.score,
                    'bounding_box': [
                        (vertex.x, vertex.y) 
                        for vertex in obj.bounding_poly.normalized_vertices
                    ]
                })
        
        logger.info(f"   Detected {len(objects)} objects with confidence >= 50%")
        return objects
    
    def _calculate_overall_confidence(
        self,
        response: vision.AnnotateImageResponse,
        room_labels: List[RoomLabel]
    ) -> float:
        """
        Calculate overall confidence score for the analysis
        
        Factors:
        - Number of room labels found
        - Text detection confidence
        - Object detection scores
        - Label detection scores
        """
        confidence_factors = []
        
        # Factor 1: Room label count (more labels = higher confidence)
        if len(room_labels) > 0:
            label_score = min(len(room_labels) / 5.0, 1.0)  # Max at 5 labels
            confidence_factors.append(label_score * 0.4)  # 40% weight
        
        # Factor 2: Text detection (presence indicates readable image)
        if response.text_annotations:
            text_score = 0.9  # High score if text detected
            confidence_factors.append(text_score * 0.3)  # 30% weight
        
        # Factor 3: Object detection scores
        if response.localized_object_annotations:
            obj_scores = [obj.score for obj in response.localized_object_annotations]
            avg_obj_score = sum(obj_scores) / len(obj_scores) if obj_scores else 0
            confidence_factors.append(avg_obj_score * 0.2)  # 20% weight
        
        # Factor 4: Label detection (general image understanding)
        if response.label_annotations:
            label_scores = [label.score for label in response.label_annotations[:5]]
            avg_label_score = sum(label_scores) / len(label_scores) if label_scores else 0
            confidence_factors.append(avg_label_score * 0.1)  # 10% weight
        
        # Calculate weighted average
        overall_confidence = sum(confidence_factors) if confidence_factors else 0.3
        
        return min(overall_confidence, 1.0)
    
    def _assess_image_quality(
        self,
        response: vision.AnnotateImageResponse,
        image_dims: ImageDimensions
    ) -> float:
        """
        Assess image quality for floor plan analysis
        
        Factors:
        - Image resolution
        - Aspect ratio (floor plans usually have reasonable ratios)
        - Text detection success
        - Color properties
        """
        quality_factors = []
        
        # Factor 1: Resolution
        total_pixels = image_dims.width * image_dims.height
        if total_pixels >= 1920 * 1080:  # Full HD or better
            quality_factors.append(1.0)
        elif total_pixels >= 1280 * 720:  # HD
            quality_factors.append(0.8)
        elif total_pixels >= 640 * 480:  # SD
            quality_factors.append(0.6)
        else:
            quality_factors.append(0.4)
        
        # Factor 2: Aspect ratio (reasonable for floor plans)
        aspect = image_dims.aspect_ratio
        if 0.5 <= aspect <= 2.0:  # Reasonable range
            quality_factors.append(1.0)
        else:
            quality_factors.append(0.6)
        
        # Factor 3: Text detection (clear image has readable text)
        if response.text_annotations and len(response.text_annotations) > 1:
            quality_factors.append(0.9)
        else:
            quality_factors.append(0.5)
        
        # Calculate average
        quality_score = sum(quality_factors) / len(quality_factors) if quality_factors else 0.5
        
        return quality_score
    
    def _generate_warnings(
        self,
        confidence: float,
        quality_score: float,
        room_labels: List[RoomLabel],
        image_dims: ImageDimensions
    ) -> List[str]:
        """Generate warnings about potential issues"""
        warnings = []
        
        if confidence < 0.5:
            warnings.append("Low confidence in image analysis. Consider uploading a clearer image.")
        
        if quality_score < 0.6:
            warnings.append("Image quality is suboptimal. Higher resolution recommended.")
        
        if len(room_labels) == 0:
            warnings.append("No room labels detected. Ensure room names are visible in the image.")
        
        if image_dims.width < 800 or image_dims.height < 600:
            warnings.append("Low resolution image. Minimum 800x600 recommended.")
        
        if image_dims.aspect_ratio > 3.0 or image_dims.aspect_ratio < 0.33:
            warnings.append("Unusual aspect ratio detected. Image may be cropped incorrectly.")
        
        return warnings
    
    def _create_error_result(
        self,
        error_message: str,
        processing_time: float
    ) -> ImageAnalysisResult:
        """Create an error result with minimal data"""
        return ImageAnalysisResult(
            room_labels=[],
            image_dimensions=ImageDimensions(0, 0, 1.0),
            all_text_annotations=[],
            detected_objects=[],
            confidence=0.0,
            quality_score=0.0,
            processing_time_ms=processing_time,
            warnings=[f"ERROR: {error_message}"],
            cost_estimate=0.0
        )
    
    def to_dict(self, result: ImageAnalysisResult) -> Dict[str, Any]:
        """
        Convert analysis result to dictionary for storage/API response
        
        Compatible with existing Firestore storage patterns
        """
        return {
            'room_labels': [
                {
                    'text': label.text,
                    'confidence': label.confidence,
                    'bounding_box': label.bounding_box,
                    'position': label.position
                }
                for label in result.room_labels
            ],
            'image_dimensions': {
                'width': result.image_dimensions.width,
                'height': result.image_dimensions.height,
                'aspect_ratio': result.image_dimensions.aspect_ratio
            },
            'all_text_annotations': result.all_text_annotations,
            'detected_objects': result.detected_objects,
            'confidence': result.confidence,
            'quality_score': result.quality_score,
            'processing_time_ms': result.processing_time_ms,
            'warnings': result.warnings,
            'cost_estimate': result.cost_estimate,
            'analyzed_at': datetime.utcnow().isoformat()
        }
    
    def estimate_cost(self, num_images: int) -> float:
        """Estimate cost for batch processing"""
        return num_images * self.cost_per_detection
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get service configuration info"""
        return {
            'service': 'Google Cloud Vision AI',
            'project_id': self.project_id,
            'features': [
                'Text Detection',
                'Object Localization',
                'Label Detection',
                'Image Properties'
            ],
            'cost_per_image': self.cost_per_detection,
            'supported_formats': ['JPG', 'JPEG', 'PNG'],
            'max_file_size_mb': 10
        }