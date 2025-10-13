# -*- coding: utf-8 -*-
# src/processors/rfp_parser.py
"""
RFP Document Parser - Extracts structured data from construction RFPs
Handles Romanian and English RFP documents with intelligent pattern matching
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger("demoplan.processors.rfp_parser")

@dataclass
class RFPStructure:
    """Structured RFP data model"""
    
    # Project Information
    project_name: Optional[str] = None
    client_name: Optional[str] = None
    location: Optional[str] = None
    building: Optional[str] = None
    floor: Optional[str] = None
    address: Optional[str] = None
    
    # Timeline
    work_start_date: Optional[datetime] = None
    work_end_date: Optional[datetime] = None
    work_duration_days: Optional[int] = None
    offer_submission_deadline: Optional[datetime] = None
    site_inspection_deadline: Optional[datetime] = None
    
    # Financial Terms
    currency: str = "EUR"
    guarantee_period_months: Optional[int] = None
    performance_bond_percentage: Optional[float] = None
    retention_percentage: Optional[float] = None
    delay_penalty_percentage: Optional[float] = None
    payment_terms: Optional[str] = None
    
    # Scope of Works
    scope_items: List[str] = field(default_factory=list)
    deliverables: List[str] = field(default_factory=list)
    mandatory_samples: List[str] = field(default_factory=list)
    excluded_works: List[str] = field(default_factory=list)
    
    # Technical Requirements
    technical_standards: List[str] = field(default_factory=list)
    compliance_requirements: List[str] = field(default_factory=list)
    building_regulations: List[str] = field(default_factory=list)
    
    # Assessment Criteria
    assessment_criteria: List[str] = field(default_factory=list)
    evaluation_weights: Dict[str, float] = field(default_factory=dict)
    
    # Team & Contacts
    project_manager: Optional[str] = None
    project_manager_contact: Optional[str] = None
    client_entity: Optional[str] = None
    lead_designer: Optional[str] = None
    general_contractor: Optional[str] = None
    
    # Document Metadata
    rfp_date: Optional[datetime] = None
    rfp_version: Optional[str] = None
    document_pages: Optional[int] = None
    
    # Extraction Quality
    extraction_confidence: float = 0.0
    missing_fields: List[str] = field(default_factory=list)
    extraction_notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "project_info": {
                "name": self.project_name,
                "client": self.client_name,
                "location": self.location,
                "building": self.building,
                "floor": self.floor,
                "address": self.address
            },
            "timeline": {
                "work_start": self.work_start_date.isoformat() if self.work_start_date else None,
                "work_end": self.work_end_date.isoformat() if self.work_end_date else None,
                "duration_days": self.work_duration_days,
                "submission_deadline": self.offer_submission_deadline.isoformat() if self.offer_submission_deadline else None,
                "inspection_deadline": self.site_inspection_deadline.isoformat() if self.site_inspection_deadline else None
            },
            "financial": {
                "currency": self.currency,
                "guarantee_months": self.guarantee_period_months,
                "performance_bond": self.performance_bond_percentage,
                "retention": self.retention_percentage,
                "delay_penalty": self.delay_penalty_percentage,
                "payment_terms": self.payment_terms
            },
            "scope": {
                "items": self.scope_items,
                "deliverables": self.deliverables,
                "mandatory_samples": self.mandatory_samples,
                "excluded": self.excluded_works
            },
            "technical": {
                "standards": self.technical_standards,
                "compliance": self.compliance_requirements,
                "regulations": self.building_regulations
            },
            "assessment": {
                "criteria": self.assessment_criteria,
                "weights": self.evaluation_weights
            },
            "team": {
                "project_manager": self.project_manager,
                "pm_contact": self.project_manager_contact,
                "client": self.client_entity,
                "designer": self.lead_designer,
                "general_contractor": self.general_contractor
            },
            "metadata": {
                "confidence": self.extraction_confidence,
                "missing_fields": self.missing_fields,
                "notes": self.extraction_notes
            }
        }


class RFPParser:
    """Extract structured data from RFP documents"""
    
    def __init__(self):
        """Initialize parser with regex patterns"""
        
        # Date patterns (multiple formats)
        self.date_patterns = [
            r'(\d{2})\.(\d{2})\.(\d{4})',  # DD.MM.YYYY
            r'(\d{2})/(\d{2})/(\d{4})',     # DD/MM/YYYY
            r'(\d{4})-(\d{2})-(\d{2})',     # YYYY-MM-DD
            r'(\d{1,2})\s+(?:ianuarie|februarie|martie|aprilie|mai|iunie|iulie|august|septembrie|octombrie|noiembrie|decembrie)\s+(\d{4})',  # Romanian month names
        ]
        
        # Financial patterns
        self.financial_patterns = {
            'guarantee': [
                r'guarantee\s*period[:\s-]*(\d+)\s*months?',
                r'garanție[:\s-]*(\d+)\s*luni',
                r'perioadă\s*garanție[:\s-]*(\d+)\s*luni'
            ],
            'performance_bond': [
                r'performance\s*bond[:\s-]*(\d+)\s*%',
                r'garanție\s*de\s*bună\s*execuție[:\s-]*(\d+)\s*%'
            ],
            'retention': [
                r'retention[:\s-]*(\d+)\s*%',
                r'retenție[:\s-]*(\d+)\s*%'
            ],
            'delay_penalty': [
                r'delay\s*(?:payment|damages)[:\s-]*([\d.]+)\s*%',
                r'penalități[:\s-]*([\d.]+)\s*%'
            ]
        }
        
        # Project identification patterns
        self.project_patterns = {
            'project_name': [
                r'(?:project|proiect)[:\s-]*([^\n]+)',
                r'REQUEST FOR PROPOSAL[:\s\n]*([^\n]+)',
                r'CERERE DE OFERTĂ[:\s\n]*([^\n]+)'
            ],
            'client': [
                r'(?:client|beneficiar)[:\s-]*([^\n]+)',
                r'(?:Client|LIENT)[:\s]*([^\n]+)',
            ],
            'location': [
                r'(?:location|locație|adresa)[:\s-]*([^\n]+)',
                r'(?:address|adresă)[:\s-]*([^\n]+)'
            ]
        }
        
        # Scope keywords
        self.scope_keywords = [
            'demolish', 'demolări', 'demolare',
            'architectural', 'arhitectural',
            'mechanical', 'mecanic', 'HVAC',
            'electrical', 'electric',
            'plumbing', 'instalații sanitare',
            'finishes', 'finisaje',
            'testing', 'testare', 'commissioning',
            'fire', 'PSI', 'protecție incendiu'
        ]
        
    def parse_rfp(self, text: str) -> RFPStructure:
        """
        Main parsing method - extracts all structured data from RFP text
        
        Args:
            text: Raw text extracted from RFP PDF
            
        Returns:
            RFPStructure with all extracted data
        """
        logger.info("🔍 Starting RFP parsing")
        rfp = RFPStructure()
        
        # Normalize text
        text = self._normalize_text(text)
        
        # Extract project information
        rfp.project_name = self._extract_project_name(text)
        rfp.client_name = self._extract_client_name(text)
        rfp.location = self._extract_location(text)
        rfp.building, rfp.floor = self._extract_building_floor(text)
        rfp.address = self._extract_address(text)
        
        # Extract timeline
        timeline = self._extract_timeline(text)
        rfp.work_start_date = timeline.get('start_date')
        rfp.work_end_date = timeline.get('end_date')
        rfp.work_duration_days = timeline.get('duration_days')
        rfp.offer_submission_deadline = timeline.get('submission_deadline')
        rfp.site_inspection_deadline = timeline.get('inspection_deadline')
        
        # Extract financial terms
        financial = self._extract_financial_terms(text)
        rfp.currency = financial.get('currency', 'EUR')
        rfp.guarantee_period_months = financial.get('guarantee_months')
        rfp.performance_bond_percentage = financial.get('performance_bond')
        rfp.retention_percentage = financial.get('retention')
        rfp.delay_penalty_percentage = financial.get('delay_penalty')
        rfp.payment_terms = financial.get('payment_terms')
        
        # Extract scope of works
        rfp.scope_items = self._extract_scope_items(text)
        rfp.deliverables = self._extract_deliverables(text)
        rfp.mandatory_samples = self._extract_mandatory_samples(text)
        rfp.excluded_works = self._extract_excluded_works(text)
        
        # Extract technical requirements
        rfp.technical_standards = self._extract_technical_standards(text)
        rfp.compliance_requirements = self._extract_compliance_requirements(text)
        rfp.building_regulations = self._extract_building_regulations(text)
        
        # Extract assessment criteria
        rfp.assessment_criteria = self._extract_assessment_criteria(text)
        
        # Extract team information
        team = self._extract_project_team(text)
        rfp.project_manager = team.get('project_manager')
        rfp.project_manager_contact = team.get('pm_contact')
        rfp.client_entity = team.get('client')
        rfp.lead_designer = team.get('designer')
        rfp.general_contractor = team.get('general_contractor')
        
        # Extract document metadata
        rfp.rfp_date = self._extract_rfp_date(text)
        
        # Calculate extraction quality
        rfp.extraction_confidence = self._calculate_confidence(rfp)
        rfp.missing_fields = self._identify_missing_fields(rfp)
        
        logger.info(f"✅ RFP parsing complete - Confidence: {rfp.extraction_confidence:.1%}")
        logger.info(f"📊 Extracted: {len(rfp.scope_items)} scope items, {len(rfp.deliverables)} deliverables")
        
        return rfp
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for better pattern matching"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        # Normalize dashes
        text = text.replace('–', '-').replace('—', '-')
        return text
    
    def _extract_project_name(self, text: str) -> Optional[str]:
        """Extract project name from RFP"""
        for pattern in self.project_patterns['project_name']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                # Clean up
                name = re.sub(r'\s+', ' ', name)
                name = name.split('\n')[0]  # Take first line only
                if len(name) > 5 and len(name) < 200:
                    logger.debug(f"Found project name: {name}")
                    return name
        return None
    
    def _extract_client_name(self, text: str) -> Optional[str]:
        """Extract client/beneficiary name"""
        for pattern in self.project_patterns['client']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                client = match.group(1).strip()
                client = client.split('\n')[0]
                # Filter out generic terms
                if 'table' not in client.lower() and len(client) > 3:
                    logger.debug(f"Found client: {client}")
                    return client
        
        # Try to find company names with SRL, SA, etc.
        company_pattern = r'\b([A-Z][a-zA-Z\s&]+(?:SRL|S\.R\.L\.|SA|S\.A\.|Ltd|Limited))\b'
        matches = re.findall(company_pattern, text)
        if matches:
            return matches[0].strip()
        
        return None
    
    def _extract_location(self, text: str) -> Optional[str]:
        """Extract project location"""
        for pattern in self.project_patterns['location']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                location = match.group(1).strip()
                location = location.split('\n')[0]
                if len(location) > 5:
                    logger.debug(f"Found location: {location}")
                    return location
        return None
    
    def _extract_building_floor(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract building and floor information"""
        building = None
        floor = None
        
        # Building patterns
        building_patterns = [
            r'(?:building|clădire|corp)[:\s]*([A-Z\d]+)',
            r'Green Court Building ([A-Z])',
            r'(?:Building|Clădirea)\s+([A-Z\d]+)'
        ]
        
        for pattern in building_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                building = match.group(1).strip()
                logger.debug(f"Found building: {building}")
                break
        
        # Floor patterns
        floor_patterns = [
            r'(\d+)(?:th|st|nd|rd)?\s*(?:floor|etaj)',
            r'(?:floor|etaj)[:\s]*(\d+)',
            r'etajul\s*(\d+)'
        ]
        
        for pattern in floor_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                floor = match.group(1).strip()
                logger.debug(f"Found floor: {floor}")
                break
        
        return building, floor
    
    def _extract_address(self, text: str) -> Optional[str]:
        """Extract full address"""
        # Romanian address pattern
        address_pattern = r'(?:address|adresa)[:\s-]*([^,\n]+,\s*[^,\n]+,\s*[^,\n]+)'
        match = re.search(address_pattern, text, re.IGNORECASE)
        if match:
            address = match.group(1).strip()
            return address
        return None
    
    def _extract_timeline(self, text: str) -> Dict[str, Any]:
        """Extract all timeline information"""
        timeline = {
            'start_date': None,
            'end_date': None,
            'duration_days': None,
            'submission_deadline': None,
            'inspection_deadline': None
        }
        
        # Work duration pattern (DD.MM.YYYY - DD.MM.YYYY)
        duration_pattern = r'(?:duration|durată|perioadă)[:\s]*(\d{2}\.\d{2}\.\d{4})\s*[-–]\s*(\d{2}\.\d{2}\.\d{4})'
        match = re.search(duration_pattern, text, re.IGNORECASE)
        if match:
            start_str = match.group(1)
            end_str = match.group(2)
            
            try:
                timeline['start_date'] = datetime.strptime(start_str, '%d.%m.%Y')
                timeline['end_date'] = datetime.strptime(end_str, '%d.%m.%Y')
                
                # Calculate duration
                duration = timeline['end_date'] - timeline['start_date']
                timeline['duration_days'] = duration.days
                
                logger.debug(f"Found work duration: {start_str} to {end_str} ({duration.days} days)")
            except ValueError as e:
                logger.warning(f"Failed to parse dates: {e}")
        
        # Submission deadline pattern
        submission_patterns = [
            r'(?:submission|submit|deadline)[:\s]*(\d{2}\.\d{2}\.\d{4})[,\s]*(\d{2}:\d{2})',
            r'(?:offers?|ofert[ăe])[:\s]*.*?(\d{2}\.\d{2}\.\d{4})[,\s]*(\d{2}:\d{2})'
        ]
        
        for pattern in submission_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                time_str = match.group(2) if len(match.groups()) > 1 else "12:00"
                
                try:
                    datetime_str = f"{date_str} {time_str}"
                    timeline['submission_deadline'] = datetime.strptime(datetime_str, '%d.%m.%Y %H:%M')
                    logger.debug(f"Found submission deadline: {datetime_str}")
                    break
                except ValueError:
                    pass
        
        # Inspection deadline pattern
        inspection_patterns = [
            r'(?:inspect|vizită|verificare)[:\s]*.*?(\d{2}\.\d{2}\.\d{4})',
            r'(?:until|până la)[:\s]*(\d{2}\.\d{2}\.\d{4})'
        ]
        
        for pattern in inspection_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                try:
                    timeline['inspection_deadline'] = datetime.strptime(date_str, '%d.%m.%Y')
                    logger.debug(f"Found inspection deadline: {date_str}")
                    break
                except ValueError:
                    pass
        
        return timeline
    
    def _extract_financial_terms(self, text: str) -> Dict[str, Any]:
        """Extract financial terms and conditions"""
        financial = {
            'currency': 'EUR',
            'guarantee_months': None,
            'performance_bond': None,
            'retention': None,
            'delay_penalty': None,
            'payment_terms': None
        }
        
        # Currency detection
        if 'EUR' in text or 'euro' in text.lower():
            financial['currency'] = 'EUR'
        elif 'RON' in text or 'lei' in text.lower():
            financial['currency'] = 'RON'
        elif 'USD' in text or 'dollar' in text.lower():
            financial['currency'] = 'USD'
        
        # Extract each financial term
        for term, patterns in self.financial_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    value_str = match.group(1)
                    try:
                        if term == 'guarantee':
                            financial['guarantee_months'] = int(value_str)
                            logger.debug(f"Found guarantee: {value_str} months")
                        else:
                            financial[term] = float(value_str)
                            logger.debug(f"Found {term}: {value_str}%")
                        break
                    except ValueError:
                        pass
        
        # Payment terms
        payment_pattern = r'(?:payment|plată)[:\s]*([^\n]{10,100})'
        match = re.search(payment_pattern, text, re.IGNORECASE)
        if match:
            financial['payment_terms'] = match.group(1).strip()
        
        return financial
    
    def _extract_scope_items(self, text: str) -> List[str]:
        """Extract scope of work items"""
        scope_items = []
        
        # Look for scope section
        scope_section_patterns = [
            r'(?:scope of works?|domeniu de aplicare|obiect)[:\s]*(.{100,2000})',
            r'(?:include|includes|cuprinde)[:\s]*(.{100,2000})'
        ]
        
        scope_text = ""
        for pattern in scope_section_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                scope_text = match.group(1)
                break
        
        # Extract bullet points or numbered items
        bullet_patterns = [
            r'[•●▪▸➤]\s*([^\n]+)',
            r'[-–]\s*([^\n]{10,200})',
            r'\d+[\.)]\s*([^\n]{10,200})'
        ]
        
        for pattern in bullet_patterns:
            matches = re.findall(pattern, scope_text if scope_text else text)
            for match in matches:
                item = match.strip()
                # Filter out short or generic items
                if len(item) > 15 and item not in scope_items:
                    scope_items.append(item)
        
        # Also look for keyword-based items
        for keyword in self.scope_keywords:
            keyword_pattern = rf'\b({keyword}[^.\n]{{10,150}})'
            matches = re.findall(keyword_pattern, text, re.IGNORECASE)
            for match in matches[:2]:  # Max 2 per keyword to avoid duplicates
                if match not in scope_items and len(match) > 20:
                    scope_items.append(match.strip())
        
        logger.debug(f"Found {len(scope_items)} scope items")
        return scope_items[:20]  # Limit to 20 items
    
    def _extract_deliverables(self, text: str) -> List[str]:
        """Extract required deliverables"""
        deliverables = []
        
        deliverable_keywords = [
            'technical book', 'carte tehnic',
            'as-built', 'as built', 'plan execuție',
            'warranty', 'garanție',
            'operation manual', 'manual operare',
            'maintenance manual', 'manual întreținere',
            'fire permit', 'autorizație PSI',
            'certificate', 'certificat',
            'approval', 'aprobare', 'aviz'
        ]
        
        for keyword in deliverable_keywords:
            pattern = rf'\b({keyword}[^.\n]{{5,100}})'
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                clean_match = match.strip()
                if clean_match not in deliverables and len(clean_match) > 10:
                    deliverables.append(clean_match)
        
        logger.debug(f"Found {len(deliverables)} deliverables")
        return deliverables[:15]
    
    def _extract_mandatory_samples(self, text: str) -> List[str]:
        """Extract mandatory material samples"""
        samples = []
        
        # Look for samples section
        samples_pattern = r'(?:samples?|mostre|eșantioane)[:\s]*(.{50,500})'
        match = re.search(samples_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if match:
            samples_text = match.group(1)
            
            # Extract items
            sample_keywords = [
                'carpet', 'mochetă',
                'LVT', 'resilient',
                'acoustic', 'acustic',
                'fabric', 'țesătură',
                'panel', 'panou',
                'lighting', 'iluminat',
                'fixture', 'corp iluminat'
            ]
            
            for keyword in sample_keywords:
                if keyword.lower() in samples_text.lower():
                    # Find the full phrase containing the keyword
                    pattern = rf'([^,\n]{{5,80}}{keyword}[^,\n]{{5,80}})'
                    matches = re.findall(pattern, samples_text, re.IGNORECASE)
                    for match in matches:
                        clean = match.strip()
                        if clean not in samples and len(clean) > 10:
                            samples.append(clean)
        
        logger.debug(f"Found {len(samples)} mandatory samples")
        return samples[:10]
    
    def _extract_excluded_works(self, text: str) -> List[str]:
        """Extract explicitly excluded works"""
        excluded = []
        
        exclusion_patterns = [
            r'(?:excluded|exclus|nu include)[:\s]*(.{50,300})',
            r'(?:not included|nu este inclus)[:\s]*(.{50,300})'
        ]
        
        for pattern in exclusion_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                exclusion_text = match.group(1)
                # Split by common delimiters
                items = re.split(r'[;,\n]', exclusion_text)
                for item in items:
                    clean = item.strip()
                    if len(clean) > 10 and clean not in excluded:
                        excluded.append(clean)
        
        return excluded[:10]
    
    def _extract_technical_standards(self, text: str) -> List[str]:
        """Extract technical standards and regulations"""
        standards = []
        
        # Common standard identifiers
        standard_patterns = [
            r'\b(ISO[\s-]?\d+)',
            r'\b(EN[\s-]?\d+)',
            r'\b(SR[\s-]?\d+)',  # Romanian standards
            r'\b(STAS[\s-]?\d+)',
            r'\b(NP[\s-]?\d+)',
        ]
        
        for pattern in standard_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            standards.extend([m.strip() for m in matches])
        
        return list(set(standards))[:15]
    
    def _extract_compliance_requirements(self, text: str) -> List[str]:
        """Extract compliance requirements"""
        requirements = []
        
        compliance_keywords = [
            'romanian law', 'legea română',
            'building code', 'cod construcții',
            'fire safety', 'siguranță incendiu',
            'health and safety', 'sănătate și securitate',
            'environmental', 'protecția mediului'
        ]
        
        for keyword in compliance_keywords:
            if keyword.lower() in text.lower():
                # Find sentence containing keyword
                pattern = rf'([^.!?]{{20,200}}{keyword}[^.!?]{{20,200}}[.!?])'
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches[:1]:  # One per keyword
                    requirements.append(match.strip())
        
        return requirements
    
    def _extract_building_regulations(self, text: str) -> List[str]:
        """Extract building-specific regulations"""
        regulations = []
        
        # Look for regulations section
        if 'building regulation' in text.lower() or 'regulament' in text.lower():
            # Extract relevant sentences
            sentences = re.split(r'[.!?]', text)
            for sentence in sentences:
                if any(word in sentence.lower() for word in ['regulament', 'regulation', 'cerință', 'requirement']):
                    clean = sentence.strip()
                    if len(clean) > 30 and len(clean) < 300:
                        regulations.append(clean)
        
        return regulations[:10]
    
    def _extract_assessment_criteria(self, text: str) -> List[str]:
        """Extract assessment/evaluation criteria"""
        criteria = []
        
        # Look for assessment section
        assessment_pattern = r'(?:assessment|evaluare|criterii)[:\s]*(.{100,800})'
        match = re.search(assessment_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if match:
            criteria_text = match.group(1)
            
            # Extract bullet points
            bullet_pattern = r'[•●▪-]\s*([^\n]{10,150})'
            matches = re.findall(bullet_pattern, criteria_text)
            criteria.extend([m.strip() for m in matches])
        
        # Common criteria keywords
        criteria_keywords = [
            'cost', 'preț',
            'experience', 'experiență',
            'timeline', 'termen',
            'quality', 'calitate',
            'compliance', 'conformitate'
        ]
        
        for keyword in criteria_keywords:
            if keyword.lower() in text.lower() and keyword not in [c.lower() for c in criteria]:
                criteria.append(keyword.title())
        
        logger.debug(f"Found {len(criteria)} assessment criteria")
        return criteria[:10]
    
    def _extract_project_team(self, text: str) -> Dict[str, Optional[str]]:
        """Extract project team members and contacts"""
        team = {
            'project_manager': None,
            'pm_contact': None,
            'client': None,
            'designer': None,
            'general_contractor': None
        }
        
        # Project Manager
        pm_patterns = [
            r'(?:project manager|manager proiect)[:\s]*([^\n]+)',
            r'(?:PM|Contact)[:\s]*([A-Z][a-z]+\s+[A-Z][a-z]+)'
        ]
        
        for pattern in pm_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                team['project_manager'] = match.group(1).strip()
                break
        
        # Extract contact info (email, phone)
        email_pattern = r'[\w\.-]+@[\w\.-]+'
        phone_pattern = r'\+?\d[\d\s\-\(\)]{8,}'
        
        emails = re.findall(email_pattern, text)
        phones = re.findall(phone_pattern, text)
        
        if emails:
            team['pm_contact'] = emails[0]
        elif phones:
            team['pm_contact'] = phones[0]
        
        # Lead Designer
        designer_patterns = [
            r'(?:lead designer|designer|architect)[:\s]*([^\n]+)',
            r'(?:proiectant)[:\s]*([^\n]+)'
        ]
        
        for pattern in designer_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                designer = match.group(1).strip()
                # Clean up
                designer = re.split(r'[,\(]', designer)[0].strip()
                if len(designer) > 3 and len(designer) < 100:
                    team['designer'] = designer
                    break
        
        # General Contractor
        gc_patterns = [
            r'(?:general contractor|GC|antreprenor general)[:\s]*([^\n]+)',
        ]
        
        for pattern in gc_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                team['general_contractor'] = match.group(1).strip()
                break
        
        logger.debug(f"Extracted team info: PM={team['project_manager']}, Designer={team['designer']}")
        return team
    
    def _extract_rfp_date(self, text: str) -> Optional[datetime]:
        """Extract RFP document date"""
        # Look near the beginning of document
        first_500_chars = text[:500]
        
        for pattern in self.date_patterns:
            match = re.search(pattern, first_500_chars)
            if match:
                try:
                    if '.' in match.group(0):
                        date_str = match.group(0)
                        return datetime.strptime(date_str, '%d.%m.%Y')
                except ValueError:
                    pass
        
        return None
    
    def _calculate_confidence(self, rfp: RFPStructure) -> float:
        """
        Calculate extraction confidence based on filled fields
        
        Scoring:
        - Critical fields (25% each): project_name, timeline, financial terms, scope
        - Important fields (10% each): client, location, team, deliverables
        - Optional fields (5% each): standards, regulations, samples
        """
        score = 0.0
        
        # Critical fields (100% of base score)
        critical_score = 0.0
        
        # Project name (25%)
        if rfp.project_name:
            critical_score += 0.25
        
        # Timeline (25%)
        timeline_fields = [
            rfp.work_start_date,
            rfp.work_end_date,
            rfp.work_duration_days
        ]
        timeline_filled = sum(1 for f in timeline_fields if f is not None)
        critical_score += 0.25 * (timeline_filled / len(timeline_fields))
        
        # Financial terms (25%)
        financial_fields = [
            rfp.guarantee_period_months,
            rfp.performance_bond_percentage,
            rfp.retention_percentage
        ]
        financial_filled = sum(1 for f in financial_fields if f is not None)
        critical_score += 0.25 * (financial_filled / len(financial_fields))
        
        # Scope (25%)
        if len(rfp.scope_items) >= 3:
            critical_score += 0.25
        elif len(rfp.scope_items) > 0:
            critical_score += 0.15
        
        score = critical_score
        
        # Important fields (bonus up to 20%)
        important_bonus = 0.0
        
        if rfp.client_name:
            important_bonus += 0.05
        if rfp.location:
            important_bonus += 0.05
        if rfp.project_manager:
            important_bonus += 0.05
        if len(rfp.deliverables) >= 3:
            important_bonus += 0.05
        
        score += important_bonus
        
        # Optional fields (bonus up to 10%)
        optional_bonus = 0.0
        
        if len(rfp.mandatory_samples) > 0:
            optional_bonus += 0.03
        if len(rfp.assessment_criteria) > 0:
            optional_bonus += 0.03
        if len(rfp.technical_standards) > 0:
            optional_bonus += 0.04
        
        score += optional_bonus
        
        # Cap at 1.0
        return min(1.0, score)
    
    def _identify_missing_fields(self, rfp: RFPStructure) -> List[str]:
        """Identify which critical/important fields are missing"""
        missing = []
        
        # Critical fields
        if not rfp.project_name:
            missing.append("project_name")
        if not rfp.work_start_date:
            missing.append("work_start_date")
        if not rfp.work_end_date:
            missing.append("work_end_date")
        if not rfp.guarantee_period_months:
            missing.append("guarantee_period")
        if not rfp.performance_bond_percentage:
            missing.append("performance_bond")
        if len(rfp.scope_items) < 3:
            missing.append("scope_items (need at least 3)")
        
        # Important fields
        if not rfp.client_name:
            missing.append("client_name")
        if not rfp.location:
            missing.append("location")
        if not rfp.project_manager:
            missing.append("project_manager")
        if len(rfp.deliverables) < 2:
            missing.append("deliverables (need at least 2)")
        
        return missing
    
    def is_rfp_document(self, text: str) -> Tuple[bool, float]:
        """
        Determine if a document is an RFP with confidence score
        
        Returns:
            Tuple of (is_rfp: bool, confidence: float)
        """
        if not text or len(text) < 100:
            return False, 0.0
        
        text_lower = text.lower()
        score = 0.0
        
        # Strong RFP indicators (20 points each)
        strong_indicators = [
            'request for proposal',
            'rfp',
            'cerere de ofertă',
            'tender',
            'licitație'
        ]
        
        for indicator in strong_indicators:
            if indicator in text_lower:
                score += 0.20
        
        # Medium indicators (10 points each)
        medium_indicators = [
            'scope of work',
            'bill of quantities',
            'performance bond',
            'contract conditions',
            'assessment criteria',
            'garanție de bună execuție'
        ]
        
        for indicator in medium_indicators:
            if indicator in text_lower:
                score += 0.10
        
        # Weak indicators (5 points each)
        weak_indicators = [
            'deadline',
            'submission',
            'deliverable',
            'contractor',
            'antreprenor'
        ]
        
        for indicator in weak_indicators:
            if indicator in text_lower:
                score += 0.05
        
        # Structural indicators
        if re.search(r'\d+\.\s+[A-Z]', text):  # Numbered sections
            score += 0.10
        
        if re.search(r'(?:project|proiect)[:\s]*[^\n]{10,}', text, re.IGNORECASE):
            score += 0.05
        
        confidence = min(1.0, score)
        is_rfp = confidence >= 0.35  # Threshold for RFP classification
        
        logger.info(f"RFP detection: is_rfp={is_rfp}, confidence={confidence:.2f}")
        return is_rfp, confidence


# Utility functions for external use
def parse_rfp_from_text(text: str) -> RFPStructure:
    """
    Convenience function to parse RFP from text
    
    Args:
        text: Raw text from PDF extraction
        
    Returns:
        RFPStructure with all extracted data
    """
    parser = RFPParser()
    return parser.parse_rfp(text)


def is_rfp_document(text: str) -> bool:
    """
    Quick check if document is an RFP
    
    Args:
        text: Raw text from document
        
    Returns:
        True if document appears to be an RFP
    """
    parser = RFPParser()
    is_rfp, _ = parser.is_rfp_document(text)
    return is_rfp