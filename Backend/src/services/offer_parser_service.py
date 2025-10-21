"""
Offer Parser Service - Adaptive Multi-Format Parser
Handles Imperial (hierarchical), Beautik (flat), and CCC (detailed) formats
"""

import csv
import io
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import openpyxl
from openpyxl import load_workbook

from src.models.offer_models import (
    ParsedOffer,
    ProjectMetadata,
    CostBreakdown,
    CostCategory,
    CostItem,
    DetailLevel,
    CategoryType
)
from src.utils.currency_utils import parse_eur_value, sum_eur_values
from src.services.classification_engine import (
    classify_item,
    normalize_unit,
    get_category_name
)


# ============================================================================
# FORMAT DETECTION
# ============================================================================

class OfferFormat:
    """Detected offer format type"""
    HIERARCHICAL_CSV = "hierarchical_csv"  # Imperial: A/B categories with items
    FLAT_CSV = "flat_csv"                  # Beautik: No categories, flat list
    DETAILED_EXCEL = "detailed_excel"      # CCC: Multi-sheet with unit prices


class FormatDetector:
    """Detects offer format from file content"""
    
    @staticmethod
    def detect(file_content: bytes, filename: str) -> str:
        """
        Detect offer format
        
        Returns:
            OfferFormat constant
        """
        # Check file extension first
        filename_lower = filename.lower()
        is_excel = filename_lower.endswith(('.xlsx', '.xls'))
        is_csv = filename_lower.endswith('.csv')
        
        if is_excel:
            # Try to detect if it's detailed (multi-sheet) or simple
            try:
                wb = load_workbook(io.BytesIO(file_content), read_only=True)
                if len(wb.sheetnames) > 2:
                    # Multi-sheet workbook = detailed format
                    return OfferFormat.DETAILED_EXCEL
            except Exception:
                pass
        
        if is_csv or not is_excel:
            # Parse CSV to detect structure
            try:
                content_str = file_content.decode('utf-8')
                reader = csv.reader(io.StringIO(content_str))
                rows = list(reader)
                
                # Look for hierarchical markers (A, B, C categories)
                has_categories = False
                for row in rows[:30]:  # Check first 30 rows
                    if len(row) > 1:
                        cell_b = row[1].strip() if len(row) > 1 else ""
                        if cell_b in ['A', 'B', 'C']:
                            has_categories = True
                            break
                
                if has_categories:
                    return OfferFormat.HIERARCHICAL_CSV
                else:
                    return OfferFormat.FLAT_CSV
            except Exception:
                pass
        
        # Default fallback
        return OfferFormat.HIERARCHICAL_CSV


# ============================================================================
# BASE PARSER CLASS
# ============================================================================

class BaseParser:
    """Base class for all parsers"""
    
    def __init__(self):
        self.warnings = []
    
    def add_warning(self, message: str):
        """Add a warning message"""
        self.warnings.append(message)
    
    def parse(self, file_content: bytes, filename: str) -> ParsedOffer:
        """Parse offer file - must be implemented by subclasses"""
        raise NotImplementedError


# ============================================================================
# HIERARCHICAL CSV PARSER (Imperial Brands)
# ============================================================================

class HierarchicalParser(BaseParser):
    """Parses Imperial Brands format: A/B/C categories with nested items"""
    
    def parse(self, file_content: bytes, filename: str) -> ParsedOffer:
        """Parse hierarchical CSV format"""
        # Decode CSV
        content_str = file_content.decode('utf-8')
        reader = csv.reader(io.StringIO(content_str))
        rows = list(reader)
        
        # Extract metadata (rows 1-7)
        metadata = self._extract_metadata(rows)
        
        # Find table start (look for SUMMARY marker)
        table_start = self._find_table_start(rows)
        
        # Parse cost data
        categories, grand_total = self._parse_cost_data(rows, table_start)
        
        # Create cost breakdown
        cost_breakdown = CostBreakdown(
            categories=categories,
            grand_total_eur=grand_total
        )
        
        # Generate offer ID
        offer_id = self._generate_offer_id()
        
        # Create parsed offer
        return ParsedOffer(
            offer_id=offer_id,
            project=metadata,
            cost_breakdown=cost_breakdown,
            parsed_at=datetime.now(),
            source_filename=filename,
            detail_level=DetailLevel.SUMMARY,
            warnings=self.warnings
        )
    
    def _extract_metadata(self, rows: List[List[str]]) -> ProjectMetadata:
        """Extract project metadata from header rows"""
        metadata = {}
        
        # Parse rows 1-7 looking for key-value pairs
        for i in range(min(10, len(rows))):
            if len(rows[i]) < 2:
                continue
            
            cell_b = rows[i][1].strip()
            
            # Look for pattern "Key: Value"
            if ':' in cell_b:
                parts = cell_b.split(':', 1)
                key = parts[0].strip().lower()
                value = parts[1].strip() if len(parts) > 1 else ""
                
                if 'project' in key:
                    metadata['project_name'] = value
                elif 'beneficary' in key or 'client' in key or 'catre' in key.lower():
                    metadata['client_name'] = value
                elif 'address' in key or 'locatie' in key.lower():
                    metadata['address'] = value
                elif 'designer' in key:
                    metadata['lead_designer'] = value
                elif 'management' in key:
                    metadata['project_manager'] = value
                elif 'investitia' in key.lower():
                    metadata['project_name'] = value
        
        # Ensure project_name exists
        if 'project_name' not in metadata:
            metadata['project_name'] = "Unknown Project"
            self.add_warning("Project name not found in metadata")
        
        return ProjectMetadata(**metadata)
    
    def _find_table_start(self, rows: List[List[str]]) -> int:
        """Find row where cost table starts"""
        for i, row in enumerate(rows):
            if len(row) > 1:
                cell_b = row[1].strip().upper()
                # Look for SUMMARY or CENTRALIZATOR marker
                if 'SUMMARY' in cell_b or 'CENTRALIZATOR' in cell_b:
                    # Table starts 2 rows after marker
                    return i + 2
        
        self.add_warning("Could not find table marker, assuming row 10")
        return 10
    
    def _parse_cost_data(
        self, 
        rows: List[List[str]], 
        start_row: int
    ) -> Tuple[List[CostCategory], float]:
        """Parse hierarchical cost structure"""
        categories = []
        current_category = None
        current_items = []
        grand_total = 0.0
        
        for i in range(start_row, len(rows)):
            row = rows[i]
            if len(row) < 3:
                continue
            
            cell_b = row[1].strip() if len(row) > 1 else ""
            cell_c = row[2].strip() if len(row) > 2 else ""
            cell_d = row[3].strip() if len(row) > 3 else ""
            
            # Check for grand total
            if 'TOTAL' in cell_b.upper() and 'w/o VAT' in cell_b:
                grand_total = parse_eur_value(cell_d) or 0.0
                break
            
            # Check for category header (single letter A, B, C)
            if cell_b in ['A', 'B', 'C']:
                # Save previous category
                if current_category and current_items:
                    category = CostCategory(
                        category_id=current_category['id'],
                        name=current_category['name'],
                        items=current_items,
                        total_eur=current_category['total']
                    )
                    categories.append(category)
                
                # Start new category
                category_total = parse_eur_value(cell_d) or 0.0
                current_category = {
                    'id': cell_b,
                    'name': cell_c,
                    'total': category_total
                }
                current_items = []
                continue
            
            # Check for item (numeric item number)
            if cell_b.isdigit() and current_category:
                value = parse_eur_value(cell_d)
                
                # Skip items with no value (intentional blanks)
                if value is None or value == 0:
                    continue
                
                # Classify item
                category_type, item_type = classify_item(cell_c, current_category['id'])
                
                item = CostItem(
                    item_number=cell_b,
                    description=cell_c,
                    value_eur=value,
                    category_id=current_category['id'],
                    item_type=item_type
                )
                current_items.append(item)
        
        # Save last category
        if current_category and current_items:
            category = CostCategory(
                category_id=current_category['id'],
                name=current_category['name'],
                items=current_items,
                total_eur=current_category['total']
            )
            categories.append(category)
        
        # Validate grand total
        if grand_total == 0:
            grand_total = sum(cat.total_eur for cat in categories)
            self.add_warning("Grand total not found, calculated from categories")
        
        return categories, grand_total
    
    def _generate_offer_id(self) -> str:
        """Generate unique offer ID: OFF_YYYYMMDD_NNN"""
        now = datetime.now()
        date_str = now.strftime("%Y%m%d")
        # TODO: Query Firestore for counter
        counter = 1
        return f"OFF_{date_str}_{counter:03d}"


# ============================================================================
# FLAT CSV PARSER (Beautik)
# ============================================================================

class FlatParser(BaseParser):
    """Parses Beautik/SL3 format: Flat list with no categories"""
    
    def parse(self, file_content: bytes, filename: str) -> ParsedOffer:
        """Parse flat CSV format"""
        content_str = file_content.decode('utf-8')
        reader = csv.reader(io.StringIO(content_str))
        rows = list(reader)
        
        # Extract metadata
        metadata = self._extract_metadata(rows)
        
        # Find table start
        table_start = self._find_table_start(rows)
        
        # Parse items and auto-categorize
        categories, grand_total = self._parse_items(rows, table_start)
        
        cost_breakdown = CostBreakdown(
            categories=categories,
            grand_total_eur=grand_total
        )
        
        offer_id = self._generate_offer_id()
        
        return ParsedOffer(
            offer_id=offer_id,
            project=metadata,
            cost_breakdown=cost_breakdown,
            parsed_at=datetime.now(),
            source_filename=filename,
            detail_level=DetailLevel.SUMMARY,
            warnings=self.warnings
        )
    
    def _extract_metadata(self, rows: List[List[str]]) -> ProjectMetadata:
        """Extract metadata from Beautik format"""
        metadata = {}
        
        for i in range(min(15, len(rows))):
            if len(rows[i]) < 2:
                continue
            
            cell_b = rows[i][1].strip()
            cell_c = rows[i][2].strip() if len(rows[i]) > 2 else ""
            
            if ':' in cell_b:
                key = cell_b.replace(':', '').strip().lower()
                
                if 'catre' in key or 'client' in key:
                    metadata['client_name'] = cell_c
                elif 'investitia' in key or 'project' in key:
                    metadata['project_name'] = cell_c
                elif 'locatie' in key or 'location' in key:
                    metadata['address'] = cell_c
                elif 'data' in key or 'date' in key:
                    metadata['offer_date'] = cell_c
                elif 'oferta' in key or 'offer' in key:
                    metadata['offer_number'] = cell_c
        
        if 'project_name' not in metadata:
            metadata['project_name'] = metadata.get('client_name', "Unknown Project")
        
        return ProjectMetadata(**metadata)
    
    def _find_table_start(self, rows: List[List[str]]) -> int:
        """Find table start in Beautik format"""
        for i, row in enumerate(rows):
            if len(row) > 2:
                cell_b = row[1].strip().upper()
                if 'CENTRALIZATOR' in cell_b or 'NR.' in cell_b:
                    return i + 1
        return 12
    
    def _parse_items(
        self,
        rows: List[List[str]],
        start_row: int
    ) -> Tuple[List[CostCategory], float]:
        """Parse flat items and auto-categorize them"""
        items_by_category = {
            'A': [],
            'B': [],
            'C': []
        }
        grand_total = 0.0
        
        for i in range(start_row, len(rows)):
            row = rows[i]
            if len(row) < 3:
                continue
            
            cell_b = row[1].strip()
            cell_c = row[2].strip()
            cell_d = row[3].strip() if len(row) > 3 else ""
            
            # Check for total row
            if 'TOTAL' in cell_b.upper():
                grand_total = parse_eur_value(cell_d) or 0.0
                break
            
            # Check for item (numeric)
            if cell_b.isdigit():
                value = parse_eur_value(cell_d)
                if value is None or value == 0:
                    continue
                
                # Classify item to determine category
                category_type, item_type = classify_item(cell_c)
                
                item = CostItem(
                    item_number=cell_b,
                    description=cell_c,
                    value_eur=value,
                    category_id=category_type.value,
                    item_type=item_type
                )
                
                items_by_category[category_type.value].append(item)
        
        # Create categories from grouped items
        categories = []
        for cat_id in ['A', 'B', 'C']:
            items = items_by_category[cat_id]
            if items:
                total = sum(item.value_eur for item in items)
                category = CostCategory(
                    category_id=cat_id,
                    name=get_category_name(CategoryType(cat_id)),
                    items=items,
                    total_eur=total
                )
                categories.append(category)
        
        if grand_total == 0:
            grand_total = sum(cat.total_eur for cat in categories)
        
        return categories, grand_total
    
    def _generate_offer_id(self) -> str:
        """Generate offer ID"""
        now = datetime.now()
        date_str = now.strftime("%Y%m%d")
        counter = 1
        return f"OFF_{date_str}_{counter:03d}"


# ============================================================================
# DETAILED EXCEL PARSER (CCC Pitesti)
# ============================================================================

class DetailedExcelParser(BaseParser):
    """Parses CCC format: Multi-sheet Excel with unit prices"""
    
    def parse(self, file_content: bytes, filename: str) -> ParsedOffer:
        """Parse detailed Excel format"""
        wb = load_workbook(io.BytesIO(file_content))
        
        # Find summary sheet
        summary_sheet = self._find_summary_sheet(wb)
        
        # Extract metadata from summary
        metadata = self._extract_metadata(summary_sheet)
        
        # Parse summary totals
        summary_totals = self._parse_summary_totals(summary_sheet)
        
        # Parse detailed sheets
        categories = self._parse_detailed_sheets(wb, summary_totals)
        
        grand_total = sum(cat.total_eur for cat in categories)
        
        cost_breakdown = CostBreakdown(
            categories=categories,
            grand_total_eur=grand_total
        )
        
        offer_id = self._generate_offer_id()
        
        return ParsedOffer(
            offer_id=offer_id,
            project=metadata,
            cost_breakdown=cost_breakdown,
            parsed_at=datetime.now(),
            source_filename=filename,
            detail_level=DetailLevel.UNIT_PRICES,
            warnings=self.warnings
        )
    
    def _find_summary_sheet(self, wb) -> object:
        """Find the summary sheet"""
        for name in wb.sheetnames:
            name_upper = name.upper()
            if 'SUMMARY' in name_upper or 'PODSUMOWANIE' in name_upper:
                return wb[name]
        
        # Fallback to first sheet
        return wb[wb.sheetnames[0]]
    
    def _extract_metadata(self, sheet) -> ProjectMetadata:
        """Extract metadata from summary sheet"""
        metadata = {}
        
        # Scan first 15 rows
        for row_idx in range(1, 16):
            for col_idx in range(1, 6):
                cell_value = sheet.cell(row_idx, col_idx).value
                if not cell_value:
                    continue
                
                cell_str = str(cell_value).strip()
                
                # Look for location/project info
                if 'location' in cell_str.lower() or 'lokalizacja' in cell_str.lower():
                    next_cell = sheet.cell(row_idx, col_idx + 1).value
                    if next_cell:
                        metadata['project_name'] = str(next_cell).strip()
                
                # Look for contractor
                if 'wykonawca' in cell_str.lower() or 'contractor' in cell_str.lower():
                    next_cell = sheet.cell(row_idx, col_idx + 1).value
                    if next_cell:
                        metadata['client_name'] = str(next_cell).strip()
        
        if 'project_name' not in metadata:
            metadata['project_name'] = "Unknown Project"
        
        return ProjectMetadata(**metadata)
    
    def _parse_summary_totals(self, sheet) -> Dict[str, float]:
        """Parse category totals from summary sheet"""
        totals = {}
        
        for row_idx in range(1, 30):
            cell_a = sheet.cell(row_idx, 1).value
            if not cell_a:
                continue
            
            cell_str = str(cell_a).lower()
            cell_d = sheet.cell(row_idx, 4).value
            
            if 'budowlanka' in cell_str or 'construction' in cell_str:
                totals['construction'] = float(cell_d) if cell_d else 0.0
            elif 'elektryka' in cell_str or 'electric' in cell_str:
                totals['electrical'] = float(cell_d) if cell_d else 0.0
            elif 'dodatkowe' in cell_str or 'additional' in cell_str:
                totals['additional'] = float(cell_d) if cell_d else 0.0
        
        return totals
    
    def _parse_detailed_sheets(self, wb, summary_totals: Dict) -> List[CostCategory]:
        """Parse detailed breakdown sheets"""
        categories = []
        
        # Look for construction sheet
        construction_sheet = self._find_sheet_by_keywords(
            wb, ['BUDOWLANKA', 'CONSTRUCTION']
        )
        if construction_sheet:
            items = self._parse_detail_sheet(construction_sheet)
            if items:
                total = summary_totals.get('construction', sum(i.value_eur for i in items))
                category = CostCategory(
                    category_id='A',
                    name='Construction Works',
                    items=items,
                    total_eur=total
                )
                categories.append(category)
        
        # Look for electrical sheet
        electrical_sheet = self._find_sheet_by_keywords(
            wb, ['ELEKTRYKA', 'ELECTRIC']
        )
        if electrical_sheet:
            items = self._parse_detail_sheet(electrical_sheet)
            if items:
                total = summary_totals.get('electrical', sum(i.value_eur for i in items))
                category = CostCategory(
                    category_id='B',
                    name='Electrical Works',
                    items=items,
                    total_eur=total
                )
                categories.append(category)
        
        return categories
    
    def _find_sheet_by_keywords(self, wb, keywords: List[str]):
        """Find sheet by keywords"""
        for sheet_name in wb.sheetnames:
            name_upper = sheet_name.upper()
            for keyword in keywords:
                if keyword.upper() in name_upper:
                    return wb[sheet_name]
        return None
    
    def _parse_detail_sheet(self, sheet) -> List[CostItem]:
        """Parse detailed sheet with unit prices"""
        items = []
        
        # Find header row (contains "OPIS", "WORKS DESCRIPTION", etc.)
        header_row = self._find_header_row(sheet)
        if not header_row:
            return items
        
        # Parse data rows
        for row_idx in range(header_row + 2, sheet.max_row + 1):
            # Column A: Item number (LP)
            item_num = sheet.cell(row_idx, 1).value
            if not item_num:
                continue
            
            # Column C: Description (English)
            desc = sheet.cell(row_idx, 3).value
            if not desc:
                continue
            
            desc_str = str(desc).strip()
            
            # Column D: Unit
            unit = sheet.cell(row_idx, 4).value
            unit_str = str(unit).strip() if unit else None
            
            # Column E: Unit price
            unit_price = sheet.cell(row_idx, 5).value
            unit_price_val = float(unit_price) if unit_price else None
            
            # Column F: Quantity
            quantity = sheet.cell(row_idx, 6).value
            quantity_val = float(quantity) if quantity else None
            
            # Column H: Total value
            total_val = sheet.cell(row_idx, 8).value
            total = float(total_val) if total_val else 0.0
            
            # Skip rows with no value
            if total == 0:
                continue
            
            # Classify item
            category_type, item_type = classify_item(desc_str)
            
            # Normalize unit
            unit_normalized = normalize_unit(unit_str) if unit_str else None
            
            item = CostItem(
                item_number=str(item_num),
                description=desc_str,
                value_eur=total,
                category_id=category_type.value,
                item_type=item_type,
                quantity=quantity_val,
                unit=unit_str,
                unit_normalized=unit_normalized,
                unit_price_eur=unit_price_val
            )
            
            items.append(item)
        
        return items
    
    def _find_header_row(self, sheet) -> Optional[int]:
        """Find header row in sheet"""
        for row_idx in range(1, 10):
            cell_value = sheet.cell(row_idx, 3).value
            if cell_value:
                cell_str = str(cell_value).upper()
                if 'DESCRIPTION' in cell_str or 'OPIS' in cell_str:
                    return row_idx
        return None
    
    def _generate_offer_id(self) -> str:
        """Generate offer ID"""
        now = datetime.now()
        date_str = now.strftime("%Y%m%d")
        counter = 1
        return f"OFF_{date_str}_{counter:03d}"


# ============================================================================
# MAIN PARSER SERVICE
# ============================================================================

class OfferParserService:
    """Main parser service - routes to appropriate parser"""
    
    def parse_offer(self, file_content: bytes, filename: str) -> ParsedOffer:
        """
        Parse offer file using adaptive format detection
        
        Args:
            file_content: Raw file bytes
            filename: Original filename
            
        Returns:
            ParsedOffer object
            
        Raises:
            ValueError: If parsing fails
        """
        # Detect format
        detected_format = FormatDetector.detect(file_content, filename)
        
        # Route to appropriate parser
        if detected_format == OfferFormat.HIERARCHICAL_CSV:
            parser = HierarchicalParser()
        elif detected_format == OfferFormat.FLAT_CSV:
            parser = FlatParser()
        elif detected_format == OfferFormat.DETAILED_EXCEL:
            parser = DetailedExcelParser()
        else:
            raise ValueError(f"Unsupported format: {detected_format}")
        
        # Parse
        try:
            parsed_offer = parser.parse(file_content, filename)
            
            # Validate
            errors = parsed_offer.validate()
            if errors:
                raise ValueError(f"Validation failed: {'; '.join(errors)}")
            
            return parsed_offer
            
        except Exception as e:
            raise ValueError(f"Parsing failed: {str(e)}")