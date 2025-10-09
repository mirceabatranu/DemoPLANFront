"""
CAD code generation using LLM

This module transforms drawing requirements and spatial data into
Python code generation prompts, then uses LLM to generate executable
code for creating DXF drawings.
"""

from typing import Dict, Any, Optional, List
import logging
import re
import math
from datetime import datetime

logger = logging.getLogger("demoplan.processors.cad_code_generator")


class CADCodeGenerator:
    """
    Generates Python code for CAD drawing creation using LLM
    
    This implements the "Query Transformation" pattern from the
    Adaptive RAG approach, enhancing prompts with:
    - Spatial data from DXF analysis
    - Romanian construction context
    - Error feedback for iterative improvement
    """
    
    def __init__(self, llm_service):
        """
        Initialize code generator
        
        Args:
            llm_service: LLM service instance (from unified_construction_agent)
        """
        self.llm_service = llm_service
        logger.info("✅ CADCodeGenerator initialized")
    
    async def generate_code(
        self,
        drawing_type: str,
        spatial_data: Dict[str, Any],
        romanian_context: Dict[str, Any],
        error_context: Optional[str] = None
    ) -> str:
        """
        Generate CAD Python code using LLM
        
        Args:
            drawing_type: Type of drawing (e.g., 'field_verification')
            spatial_data: Spatial information from DXF analysis
            romanian_context: Romanian room names and context
            error_context: Previous error message for retry attempts
        
        Returns:
            Generated Python code as string
        """
        
        logger.info(f"🎨 Generating CAD code for {drawing_type}")
        
        # Build enhanced prompt
        prompt = self.build_generation_prompt(
            drawing_type,
            spatial_data,
            romanian_context,
            error_context
        )
        
        # Call LLM service
        try:
            response = await self.llm_service.safe_construction_call(
                user_input=prompt,
                domain="cad_generation",
                system_prompt=self._get_system_prompt()
            )
            
            # Extract code from response
            code = self._extract_code_from_response(response)
            
            logger.info(f"✅ Generated {len(code)} characters of code")
            
            return code
            
        except Exception as e:
            logger.error(f"❌ Code generation failed: {e}")
            raise
    
    def build_generation_prompt(
        self,
        drawing_type: str,
        spatial_data: Dict[str, Any],
        romanian_context: Dict[str, Any],
        error_context: Optional[str] = None
    ) -> str:
        """
        Build enhanced prompt for CAD code generation
        
        This is the CORE transformation step - following the Medium article's
        "Query Transformation" pattern to create a detailed, context-rich prompt.
        
        Args:
            drawing_type: Type of drawing to generate
            spatial_data: Room dimensions, walls, etc.
            romanian_context: Romanian labels and context
            error_context: Error from previous attempt (if retry)
        
        Returns:
            Complete prompt string for LLM
        """
        
        # Extract room information
        rooms = spatial_data.get("rooms", [])
        total_area = spatial_data.get("total_area", 0)
        walls = spatial_data.get("walls", [])
        
        # Build room descriptions
        room_descriptions = self._format_room_descriptions(rooms)
        
        # Build spatial data section
        spatial_info = self._format_spatial_data_for_prompt(spatial_data)
        
        # Base prompt with context
        prompt = f"""You are an expert CAD programmer specializing in Python ezdxf library and Romanian construction drawings.

TASK: Generate complete, executable Python code to create a field verification drawing (plan de verificare în teren).

PROJECT INFORMATION:
-------------------
Total Area: {total_area:.2f} m²
Number of Rooms: {len(rooms)}
Number of Walls: {len(walls)}

Rooms:
{room_descriptions}

TECHNICAL REQUIREMENTS:
-----------------------
1. Use ezdxf library to create DXF file compatible with AutoCAD R2010 or later
2. Use Romanian layer names following construction standards:
   - PERETI (walls)
   - USI (doors)
   - FERESTRE (windows)
   - DIMENSIUNI (dimensions)
   - ADNOTARI (annotations)
   - TEXT (text labels)

3. Draw each room as a rectangle with proper dimensions
4. Add Romanian room labels centered in each room
5. Add area labels for each room (format: "A = XX.XX m²")
6. Add dimension lines showing room sizes in meters
7. Use UTF-8 encoding for Romanian characters (ă, â, î, ș, ț)
8. Set proper line weights: walls=0.5mm, dimensions=0.18mm
9. Use appropriate colors: walls=7 (white/black), dimensions=1 (red)
10. Include a simple title block with project info

CODE STRUCTURE REQUIREMENTS:
----------------------------
```python
import ezdxf
from ezdxf import colors
from ezdxf.enums import TextEntityAlignment

def generate_field_verification_drawing(output_path: str) -> str:
    \"\"\"
    Generate field verification drawing for Romanian construction project
    
    Args:
        output_path: Path where DXF file will be saved
    
    Returns:
        Path to saved DXF file
    \"\"\"
    # Create new DXF document (R2010 format)
    doc = ezdxf.new('R2010', setup=True)
    msp = doc.modelspace()
    
    # Setup layers with Romanian names
    doc.layers.add(name='PERETI', color=7, linetype='CONTINUOUS')
    doc.layers.add(name='USI', color=3, linetype='CONTINUOUS')
    doc.layers.add(name='FERESTRE', color=5, linetype='CONTINUOUS')
    doc.layers.add(name='DIMENSIUNI', color=1, linetype='CONTINUOUS')
    doc.layers.add(name='ADNOTARI', color=8, linetype='CONTINUOUS')
    doc.layers.add(name='TEXT', color=7, linetype='CONTINUOUS')
    
    # Set line weights (in 1/100 mm)
    doc.layers.get('PERETI').dxf.lineweight = 50
    doc.layers.get('DIMENSIUNI').dxf.lineweight = 18
    
    # Draw rooms using spatial data
    # For each room:
    #   1. Draw rectangle on PERETI layer
    #   2. Add room label on TEXT layer (centered)
    #   3. Add area label below room name
    #   4. Add dimension lines on DIMENSIUNI layer
    
    # Add title block
    # Add title block with project info
    
    # Save DXF file
    doc.saveas(output_path)
    
    return output_path

# Execute the function
if __name__ == "__main__":
    result = generate_field_verification_drawing("field_verification.dxf")
    print(f"Drawing saved to: {{result}}")
```

SPATIAL DATA TO USE:
--------------------
{spatial_info}
"""

        # Add error context if this is a retry
        if error_context:
            prompt += f"""

⚠️ PREVIOUS ATTEMPT FAILED
--------------------------
Error encountered: {error_context}

Please fix this error in the new code. Common issues and solutions:
- Incorrect ezdxf API usage → Check ezdxf documentation for correct methods
- Missing UTF-8 encoding → Add encoding='utf-8' where needed
- Wrong coordinate calculations → Verify math for converting meters to millimeters
- Missing error handling → Add try/except blocks
- Layer not found errors → Ensure layers are created before use
- Text alignment issues → Use TextEntityAlignment enum correctly
"""

        prompt += """

CRITICAL RULES:
---------------
1. Return ONLY executable Python code, no explanations before or after
2. Code must run without modifications
3. Include all necessary imports at the top
4. Handle Romanian characters properly using UTF-8 encoding
5. Use millimeters as units (standard in Romanian construction)
6. No dangerous operations: no os.system, subprocess, eval, exec
7. All rooms must be drawn with proper dimensions from spatial data
8. Function must accept output_path parameter and return the path
9. Add error handling for file operations
10. Use the exact spatial coordinates and dimensions provided above

EXAMPLE OUTPUT FORMAT:
----------------------
Your response should contain ONLY the Python code, starting with imports and ending with the function call.
Do not include markdown code fences, explanations, or any other text.

Generate the complete Python code now:
"""
        
        return prompt
    
    def _format_room_descriptions(self, rooms: List[Dict]) -> str:
        """Format room information for prompt"""
        if not rooms:
            return "  No room data available"
        
        descriptions = []
        for idx, room in enumerate(rooms, 1):
            name = room.get("name_ro", room.get("type", f"Room {idx}"))
            area = room.get("area", 0)
            descriptions.append(f"  {idx}. {name}: {area:.2f} m²")
        
        return "\n".join(descriptions)
    
    def _format_spatial_data_for_prompt(self, spatial_data: Dict) -> str:
        """
        Format spatial data into structured, readable format for LLM
        
        This provides detailed room coordinates and dimensions
        """
        
        formatted = []
        
        rooms = spatial_data.get("rooms", [])
        
        if not rooms:
            return "No spatial data available - generate simple layout"
        
        formatted.append("Room Layout Data (use this to position and size rectangles):")
        formatted.append("")
        
        for idx, room in enumerate(rooms, 1):
            name = room.get("name_ro", room.get("type", f"Room_{idx}"))
            area = room.get("area", 0)
            
            formatted.append(f"Room {idx}: {name}")
            formatted.append(f"  Area: {area:.2f} m²")
            
            # Try to get bounds/dimensions
            bounds = room.get("bounds", {})
            if bounds:
                formatted.append(f"  Bounds: {bounds}")
            
            # Try to get center point
            center = room.get("center", {})
            if center:
                formatted.append(f"  Center: x={center.get('x', 0):.2f}, y={center.get('y', 0):.2f}")
            
            # Calculate suggested dimensions (if area is available but not bounds)
            if area > 0 and not bounds:
                # Suggest rectangular dimensions (approximate)
                width = math.sqrt(area)
                height = area / width
                formatted.append(f"  Suggested dimensions: {width:.2f}m x {height:.2f}m")
                formatted.append(f"  (Convert to mm: {width*1000:.0f}mm x {height*1000:.0f}mm)")
            
            formatted.append("")
        
        return "\n".join(formatted)
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for LLM"""
        return """You are an expert CAD programmer specializing in:
- Python programming with ezdxf library
- Romanian construction drawing standards
- DXF file format and AutoCAD compatibility
- UTF-8 encoding for Romanian diacritics

Generate clean, executable, well-documented Python code.
Always follow Romanian construction standards for layer naming and formatting.
Ensure code is safe (no system commands, no arbitrary file operations)."""
    
    def _extract_code_from_response(self, response: str) -> str:
        """
        Extract Python code from LLM response
        
        Handles various response formats:
        - Pure code
        - Code wrapped in markdown fences
        - Code with explanations
        
        Args:
            response: Raw LLM response
        
        Returns:
            Clean Python code
        """
        
        # Pattern 1: Look for markdown code blocks
        # ```python ... ``` or ``` ... ```
        code_block_pattern = r"```(?:python)?\s*\n(.*?)\n```"
        matches = re.findall(code_block_pattern, response, re.DOTALL)
        
        if matches:
            # Use the first (or largest) code block
            code = max(matches, key=len) if len(matches) > 1 else matches[0]
            logger.info("✅ Extracted code from markdown block")
            return code.strip()
        
        # Pattern 2: Look for code between import statements and end
        if "import " in response:
            # Find everything from first import to end
            start_idx = response.find("import ")
            if start_idx != -1:
                code = response[start_idx:]
                logger.info("✅ Extracted code from import statement onwards")
                return code.strip()
        
        # Pattern 3: Response is pure code (no markdown)
        # If response contains "def " and "import", likely pure code
        if "def " in response and "import" in response:
            logger.info("✅ Response appears to be pure code")
            return response.strip()
        
        # Fallback: return as-is and let validator catch issues
        logger.warning("⚠️ Could not identify clear code structure, returning as-is")
        return response.strip()
    
    def build_retry_prompt(
        self,
        original_code: str,
        error_message: str,
        spatial_data: Dict[str, Any]
    ) -> str:
        """
        Build a focused retry prompt when code execution fails
        
        Args:
            original_code: The code that failed
            error_message: Error encountered
            spatial_data: Original spatial data
        
        Returns:
            Enhanced prompt for retry
        """
        
        prompt = f"""The previous CAD code generation attempt failed. Please fix the code.

ORIGINAL CODE:
--------------
```python
{original_code[:1000]}...  # (truncated for brevity)
```

ERROR ENCOUNTERED:
------------------
{error_message}

TASK:
-----
Generate corrected Python code that fixes this error while maintaining the same functionality.

REQUIREMENTS:
-------------
1. Fix the specific error mentioned above
2. Keep using ezdxf library with Romanian layer names
3. Maintain UTF-8 encoding for Romanian characters
4. Return ONLY the corrected Python code
5. Ensure code is complete and executable

Generate the fixed code now:
"""
        
        return prompt
    
    def estimate_drawing_complexity(self, spatial_data: Dict[str, Any]) -> str:
        """
        Estimate drawing complexity based on spatial data
        
        Returns:
            'simple', 'moderate', or 'complex'
        """
        
        room_count = len(spatial_data.get("rooms", []))
        
        if room_count <= 2:
            return "simple"
        elif room_count <= 5:
            return "moderate"
        else:
            return "complex"
    
    def get_example_code_snippet(self, drawing_type: str) -> str:
        """
        Get example code snippet for specific drawing type
        (Can be used for few-shot learning in future phases)
        
        Args:
            drawing_type: Type of drawing
        
        Returns:
            Example code snippet
        """
        
        examples = {
            "field_verification": """
# Example snippet for field verification drawing
import ezdxf
from ezdxf.enums import TextEntityAlignment

doc = ezdxf.new('R2010')
msp = doc.modelspace()

# Create layer
doc.layers.add(name='PERETI', color=7)

# Draw room rectangle (example: 5m x 4m)
points = [(0, 0), (5000, 0), (5000, 4000), (0, 4000), (0, 0)]
msp.add_lwpolyline(points, dxfattribs={'layer': 'PERETI'})

# Add room label
msp.add_text(
    "Bucătărie",
    dxfattribs={'layer': 'TEXT', 'height': 250}
).set_placement((2500, 2000), align=TextEntityAlignment.MIDDLE_CENTER)

doc.saveas("output.dxf")
"""
        }
        
        return examples.get(drawing_type, "")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_simple_prompt(room_data: List[Dict], drawing_type: str = "field_verification") -> str:
    """
    Create a simplified prompt for basic drawing generation
    
    Useful for testing or simple scenarios
    
    Args:
        room_data: List of room dictionaries with name_ro and area
        drawing_type: Type of drawing
    
    Returns:
        Simple prompt string
    """
    
    room_list = "\n".join([
        f"- {r.get('name_ro', 'Room')}: {r.get('area', 0):.2f} m²"
        for r in room_data
    ])
    
    prompt = f"""Generate Python code using ezdxf to create a simple {drawing_type} drawing.

Rooms:
{room_list}

Create a DXF file with rectangles for each room, labeled in Romanian.
Use UTF-8 encoding. Return only the Python code."""
    
    return prompt


def extract_function_call(code: str) -> Optional[str]:
    """
    Extract the main function call from generated code
    
    Args:
        code: Generated Python code
    
    Returns:
        Function call line or None
    """
    
    # Look for function call pattern
    pattern = r"(\w+)\(['\"].*?\.dxf['\"]\)"
    match = re.search(pattern, code)
    
    return match.group(0) if match else None