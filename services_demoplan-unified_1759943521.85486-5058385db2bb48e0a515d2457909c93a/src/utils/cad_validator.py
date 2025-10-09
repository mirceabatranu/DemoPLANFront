"""
CAD code validation and safety checks

This module validates Python code generated for CAD drawing creation,
ensuring it's safe to execute and follows proper patterns.
"""

from typing import List, Dict, Any
import re
import ast
import logging

logger = logging.getLogger("demoplan.utils.cad_validator")


class ValidationResult:
    """Result of code validation"""
    
    def __init__(
        self,
        is_valid: bool,
        errors: List[str] = None,
        warnings: List[str] = None
    ):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
    
    def __repr__(self):
        status = "VALID" if self.is_valid else "INVALID"
        return f"ValidationResult({status}, errors={len(self.errors)}, warnings={len(self.warnings)})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage"""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings)
        }


class CADCodeValidator:
    """
    Validates generated CAD code for safety and correctness
    
    This validator performs multiple checks:
    - Python syntax validation
    - Required imports presence
    - Dangerous operation detection
    - DXF save operation verification
    - Encoding checks for Romanian text
    - Function structure validation
    """
    
    # Required imports for DXF generation
    REQUIRED_IMPORTS = [
        "ezdxf"
    ]
    
    # Patterns that indicate dangerous operations
    DANGEROUS_PATTERNS = [
        (r"os\.system", "System command execution"),
        (r"subprocess\.", "Subprocess execution"),
        (r"eval\s*\(", "Eval execution"),
        (r"exec\s*\(", "Exec execution"),
        (r"__import__", "Dynamic import"),
        (r"compile\s*\(", "Code compilation"),
        (r"open\s*\([^)]*['\"]w['\"]", "Arbitrary file writing"),
        (r"open\s*\([^)]*['\"]a['\"]", "Arbitrary file appending"),
        (r"rmdir|unlink|remove", "File/directory deletion"),
        (r"import\s+socket", "Network socket usage"),
        (r"import\s+urllib", "URL fetching"),
        (r"import\s+requests", "HTTP requests"),
    ]
    
    # Allowed imports (whitelist approach)
    ALLOWED_IMPORTS = [
        "ezdxf",
        "math",
        "datetime",
        "typing",
        "dataclasses",
        "enum"
    ]
    
    # Required patterns for valid CAD code
    REQUIRED_PATTERNS = [
        (r"ezdxf\.new\(", "DXF document creation"),
        (r"\.modelspace\(\)", "Modelspace access"),
        (r"\.saveas\(|\.save\(", "DXF save operation")
    ]
    
    def __init__(self):
        self.validation_cache = {}
    
    def validate(self, code: str) -> ValidationResult:
        """
        Main validation method - performs all checks
        
        Args:
            code: Python code string to validate
        
        Returns:
            ValidationResult with validation status and messages
        """
        errors = []
        warnings = []
        
        # Check 1: Empty code
        if not code or not code.strip():
            errors.append("Code is empty")
            return ValidationResult(False, errors, warnings)
        
        # Check 2: Valid Python syntax
        syntax_errors = self._check_syntax(code)
        if syntax_errors:
            errors.extend(syntax_errors)
            # If syntax is invalid, no point in further checks
            return ValidationResult(False, errors, warnings)
        
        # Check 3: Required imports
        import_errors = self._check_required_imports(code)
        if import_errors:
            errors.extend(import_errors)
        
        # Check 4: Dangerous operations
        danger_errors = self._check_dangerous_operations(code)
        if danger_errors:
            errors.extend(danger_errors)
        
        # Check 5: Unauthorized imports
        unauthorized_warnings = self._check_unauthorized_imports(code)
        if unauthorized_warnings:
            warnings.extend(unauthorized_warnings)
        
        # Check 6: Required CAD patterns
        pattern_warnings = self._check_required_patterns(code)
        if pattern_warnings:
            warnings.extend(pattern_warnings)
        
        # Check 7: UTF-8 encoding for Romanian text
        encoding_warnings = self._check_encoding(code)
        if encoding_warnings:
            warnings.extend(encoding_warnings)
        
        # Check 8: Function structure
        structure_warnings = self._check_function_structure(code)
        if structure_warnings:
            warnings.extend(structure_warnings)
        
        # Check 9: Romanian character handling
        romanian_warnings = self._check_romanian_chars(code)
        if romanian_warnings:
            warnings.extend(romanian_warnings)
        
        # Validation passes if no errors (warnings are non-blocking)
        is_valid = len(errors) == 0
        
        if is_valid:
            logger.info(f"✅ Code validation passed (warnings: {len(warnings)})")
        else:
            logger.warning(f"❌ Code validation failed: {len(errors)} errors")
        
        return ValidationResult(is_valid, errors, warnings)
    
    def _check_syntax(self, code: str) -> List[str]:
        """Check if code has valid Python syntax"""
        errors = []
        try:
            ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
        except Exception as e:
            errors.append(f"Syntax validation error: {str(e)}")
        
        return errors
    
    def _check_required_imports(self, code: str) -> List[str]:
        """Check if all required imports are present"""
        errors = []
        
        for required_import in self.REQUIRED_IMPORTS:
            if f"import {required_import}" not in code:
                errors.append(f"Missing required import: {required_import}")
        
        return errors
    
    def _check_dangerous_operations(self, code: str) -> List[str]:
        """Check for dangerous operations that could harm the system"""
        errors = []
        
        for pattern, description in self.DANGEROUS_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                errors.append(f"Dangerous operation detected: {description} (pattern: {pattern})")
        
        return errors
    
    def _check_unauthorized_imports(self, code: str) -> List[str]:
        """Check for imports that are not in the whitelist"""
        warnings = []
        
        # Extract all import statements
        import_pattern = r"^(?:from\s+(\S+)|import\s+(\S+))"
        imports = re.findall(import_pattern, code, re.MULTILINE)
        
        for from_import, direct_import in imports:
            module = from_import or direct_import
            # Get base module name (before first dot)
            base_module = module.split('.')[0]
            
            if base_module not in self.ALLOWED_IMPORTS:
                warnings.append(f"Unauthorized import detected: {module}")
        
        return warnings
    
    def _check_required_patterns(self, code: str) -> List[str]:
        """Check if code contains required CAD patterns"""
        warnings = []
        
        for pattern, description in self.REQUIRED_PATTERNS:
            if not re.search(pattern, code):
                warnings.append(f"Missing expected pattern: {description}")
        
        return warnings
    
    def _check_encoding(self, code: str) -> List[str]:
        """Check encoding handling for Romanian characters"""
        warnings = []
        
        # If code mentions encoding, should be UTF-8
        if "encoding" in code.lower():
            if "utf-8" not in code.lower() and "utf8" not in code.lower():
                warnings.append("Code specifies encoding but not UTF-8 (required for Romanian characters)")
        
        return warnings
    
    def _check_function_structure(self, code: str) -> List[str]:
        """Check if code has proper function structure"""
        warnings = []
        
        # Check for function definition
        if "def " not in code:
            warnings.append("No function definition found - code may not be reusable")
        
        # Check for docstring
        if '"""' not in code and "'''" not in code:
            warnings.append("No docstring found - code lacks documentation")
        
        # Check for return statement (good practice)
        if "return " not in code:
            warnings.append("No return statement found - function should return file path")
        
        return warnings
    
    def _check_romanian_chars(self, code: str) -> List[str]:
        """Check if Romanian characters are likely to be handled correctly"""
        warnings = []
        
        # Check if code contains Romanian characters
        romanian_chars = ['ă', 'â', 'î', 'ș', 'ț', 'Ă', 'Â', 'Î', 'Ș', 'Ț']
        has_romanian_chars = any(char in code for char in romanian_chars)
        
        if has_romanian_chars:
            # Should use proper encoding
            if "encoding" not in code.lower():
                warnings.append("Romanian characters present but no explicit encoding specified")
        
        return warnings
    
    def extract_function_name(self, code: str) -> str:
        """
        Extract the main function name from code
        
        Args:
            code: Python code string
        
        Returns:
            Function name or default 'generate_drawing'
        """
        match = re.search(r"def\s+(\w+)\s*\(", code)
        return match.group(1) if match else "generate_drawing"
    
    def extract_output_parameter(self, code: str) -> str:
        """
        Extract the output file parameter name
        
        Args:
            code: Python code string
        
        Returns:
            Parameter name or default 'output_path'
        """
        # Look for function definition with path parameter
        match = re.search(r"def\s+\w+\s*\(\s*(\w+)\s*(?::|\))", code)
        return match.group(1) if match else "output_path"
    
    def get_validation_summary(self, result: ValidationResult) -> str:
        """
        Get human-readable validation summary
        
        Args:
            result: ValidationResult object
        
        Returns:
            Formatted summary string
        """
        if result.is_valid:
            summary = "✅ Code validation PASSED"
            if result.warnings:
                summary += f"\n⚠️  {len(result.warnings)} warning(s):"
                for warning in result.warnings:
                    summary += f"\n   - {warning}"
        else:
            summary = f"❌ Code validation FAILED with {len(result.errors)} error(s):"
            for error in result.errors:
                summary += f"\n   - {error}"
            
            if result.warnings:
                summary += f"\n⚠️  Also {len(result.warnings)} warning(s):"
                for warning in result.warnings:
                    summary += f"\n   - {warning}"
        
        return summary


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def quick_validate(code: str) -> bool:
    """
    Quick validation check - returns only pass/fail
    
    Args:
        code: Python code string
    
    Returns:
        True if valid, False otherwise
    """
    validator = CADCodeValidator()
    result = validator.validate(code)
    return result.is_valid


def validate_and_log(code: str, logger_instance=None) -> ValidationResult:
    """
    Validate code and log the results
    
    Args:
        code: Python code string
        logger_instance: Logger to use (optional)
    
    Returns:
        ValidationResult
    """
    log = logger_instance or logger
    
    validator = CADCodeValidator()
    result = validator.validate(code)
    
    summary = validator.get_validation_summary(result)
    
    if result.is_valid:
        log.info(summary)
    else:
        log.error(summary)
    
    return result