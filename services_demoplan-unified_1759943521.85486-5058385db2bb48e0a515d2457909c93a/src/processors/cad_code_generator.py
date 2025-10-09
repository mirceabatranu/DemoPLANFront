"""
CAD code generation using LLM

This module transforms drawing requirements and spatial data into
Python code generation prompts, then uses LLM to generate executable
code for creating DXF drawings.
"""

from typing import Dict, Any, Optional, List
import logging
import re
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
    
    # TODO: Draw rooms using spatial data
    # For each room:
    #   1. Draw rectangle on PERETI layer
    #   2. Add room label on TEXT layer (centered)
    #   3. Add area label below room name
    #   4. Add dimension lines on DIMENSIUNI layer
    
    # Add title block
    # TODO: Add title block with project info
    
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