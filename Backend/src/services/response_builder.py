# -*- coding: utf-8 -*-
# src/services/response_builder.py
"""
Response Builder - Formats agent responses with professional structure
Handles all response types: file analysis, chat, offers, data updates
Provides consistent, rich, scannable formatting across all interactions
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger("demoplan.services.response_builder")

class ResponseType(Enum):
    """Types of responses that can be formatted"""
    FILE_ANALYSIS = "file_analysis"
    CHAT_RESPONSE = "chat_response"
    OFFER_GENERATION = "offer_generation"
    DATA_UPDATE = "data_update"
    ERROR = "error"
    PROGRESS_UPDATE = "progress_update"

class ResponseBuilder:
    """Build rich, contextual responses for all agent interactions"""
    
    def __init__(self):
        """Initialize response builder with formatting rules"""
        self.max_list_items = 5  # Maximum items to show in lists before truncating
        self.use_emojis = True   # Use emojis for visual hierarchy

    def _analyze_file_context(self, file_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze what files were uploaded and their content depth
        This drives contextual question generation
        """
        if not file_analysis:
            return {
                'has_any_files': False,
                'has_drawing': False,
                'has_specification': False,
                'has_text': False,
                'drawing_detail_level': 'none',
                'spec_detail_level': 'none'
            }
    
        # Check for DXF
        dxf_data = file_analysis.get('dxf_analysis', {})
        has_dxf = bool(dxf_data.get('dxf_analysis'))
    
        # Analyze DXF detail level
        drawing_detail = 'none'
        dxf_has_mep = False
        dxf_has_dimensions = False
    
        if has_dxf:
            dxf_inner = dxf_data.get('dxf_analysis', {})
            has_rooms = dxf_inner.get('total_rooms', 0) > 0
            dxf_has_mep = dxf_inner.get('has_hvac') or dxf_inner.get('has_electrical')
            dxf_has_dimensions = dxf_inner.get('has_dimensions', False)
        
            if has_rooms and dxf_has_mep and dxf_has_dimensions:
                drawing_detail = 'complete'
            elif has_rooms and dxf_has_mep:
                drawing_detail = 'good'
            elif has_rooms:
                drawing_detail = 'basic'
    
        # Check for PDF
        pdf_data = file_analysis.get('pdf_analysis', {})
        has_pdf = bool(pdf_data)
    
        # Analyze PDF detail level
        spec_detail = 'none'
        pdf_has_scope = False
    
        if has_pdf:
            # ✅ FIX: Initialize variables BEFORE using them
            has_specs = len(pdf_data.get('construction_specs', [])) > 0
            has_materials = len(pdf_data.get('material_references', [])) > 0
            pdf_has_scope = has_specs
        
            if has_specs and has_materials:
                spec_detail = 'complete'
            elif has_specs or has_materials:
                spec_detail = 'partial'
    
        # Check for TXT
        txt_data = file_analysis.get('txt_analysis', {})
        has_txt = bool(txt_data.get('requirements'))
    
        return {
            'has_any_files': has_dxf or has_pdf or has_txt,
            'has_drawing': has_dxf,
            'has_specification': has_pdf,
            'has_text': has_txt,
            'drawing_detail_level': drawing_detail,
            'spec_detail_level': spec_detail,
            'dxf_has_mep': dxf_has_mep,
            'dxf_has_dimensions': dxf_has_dimensions,
            'pdf_has_scope': pdf_has_scope
        }
    
    def build_file_analysis_response(
        self,
        file_analysis: Dict[str, Any],
        gap_analysis: Any,  # GapAnalysisResult
        cross_reference: Optional[Dict[str, Any]] = None,
        session_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build comprehensive file analysis response
        
        This is the FIRST response after file upload
        Should be impressive, complete, and actionable
        """
        sections = []
        
        # 1. Executive Summary - The "wow" moment
        sections.append(self._build_executive_summary(
            file_analysis, gap_analysis, cross_reference
        ))
        
        # 2. Project Context (from RFP if available)
        if file_analysis.get('rfp_data'):
            sections.append(self._build_rfp_context(file_analysis['rfp_data']))
        
        # 3. Technical Analysis (from DXF)
        if file_analysis.get('dxf_analysis'):
            sections.append(self._build_dxf_analysis(file_analysis['dxf_analysis']))
        
        # 4. Additional Files (PDF, TXT)
        if file_analysis.get('pdf_analysis') or file_analysis.get('txt_analysis'):
            sections.append(self._build_other_files_summary(file_analysis))
        
        # 4.5. DETAILED FILE DESCRIPTIONS (Phase 1 Enhancement)
        file_descriptions = self._build_detailed_file_descriptions(file_analysis)
        if file_descriptions:
            sections.append(file_descriptions)
        
        # 5. Data Quality Assessment
        sections.append(self._build_data_quality(gap_analysis, cross_reference))
        
        # 6. Conflicts & Validation (if any)
        if cross_reference and cross_reference.get('conflicts'):
            sections.append(self._build_conflicts_summary(cross_reference))
        
        # 7. Storage Information (NEW - optional, only if geometric split occurred)
        dxf_info = file_analysis.get('dxf_analysis', {})
        # Support both wrapper and direct dxf structure
        geometric_src = dxf_info.get('geometric_storage') or (dxf_info.get('dxf_analysis') or {}).get('geometric_storage')
        if geometric_src == 'gcs':
            sections.append(self._build_storage_info_section(file_analysis))

        # 8. Next Steps - Clear call to action
        sections.append(self._build_next_steps(gap_analysis, cross_reference, file_analysis))
        
        return "\n\n".join(sections)
    
    def build_chat_response(
        self,
        llm_response: str,
        gap_analysis: Any,
        cross_reference: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        show_full_analysis: bool = False
    ) -> str:
        """
        Build conversational response with contextual analysis
        
        Args:
            llm_response: The conversational response from LLM
            gap_analysis: Current gap analysis
            cross_reference: Cross-reference results
            context: Conversation context
            show_full_analysis: If True, show complete analysis; if False, show compact version
        """
        sections = []
        
        # 1. Main conversational response
        sections.append(llm_response)
        
        # 2. Progress indicator (if conversation is advancing)
        if context and context.get('confidence_improvement'):
            sections.append(self._build_progress_indicator(context))
        
        # 3. Compact or full analysis based on flag
        if show_full_analysis:
            # Full analysis with all details
            sections.append(self._build_data_quality(gap_analysis, cross_reference))
            
            if cross_reference and cross_reference.get('conflicts'):
                sections.append(self._build_conflicts_summary(cross_reference))
        else:
            # Compact version - just key metrics
            sections.append(self._build_compact_status(gap_analysis, cross_reference))
        
        # 4. Next questions (always show)
        if gap_analysis and gap_analysis.prioritized_questions:
            # sections.append(self._build_questions_section(gap_analysis))
            pass  #
        
        return "\n\n".join(sections)
    
    def build_offer_response(
        self,
        offer_content: str,
        confidence: float,
        warnings: Optional[List[str]] = None
    ) -> str:
        """
        Build professional offer response
        
        Args:
            offer_content: The generated offer content
            confidence: Confidence score for the offer
            warnings: Any warnings to include
        """
        sections = []
        
        # Header
        sections.append("# 📋 OFERTĂ TEHNICĂ ȘI COMERCIALĂ\n")
        sections.append(f"**Data generării**: {datetime.now().strftime('%d.%m.%Y %H:%M')}")
        sections.append(f"**Nivel confidence**: {confidence:.1%}\n")
        
        # Warnings if any
        if warnings:
            sections.append("## ⚠️ NOTIFICĂRI IMPORTANTE\n")
            for warning in warnings:
                sections.append(f"- {warning}")
            sections.append("")
        
        # Offer content
        sections.append(offer_content)
        
        # Footer
        sections.append("\n---\n")
        sections.append("*Ofertă generată automat de sistemul DEMOPLAN cu analiză AI avansată*")
        sections.append("*Vă rugăm să revizuiți și să confirmați toate detaliile înainte de acceptare*")
        
        return "\n".join(sections)
    
    def build_error_response(
        self,
        error_message: str,
        context: Optional[str] = None,
        suggestions: Optional[List[str]] = None
    ) -> str:
        """Build user-friendly error response"""
        sections = []
        
        sections.append("## ❌ EROARE\n")
        sections.append(f"**Mesaj**: {error_message}\n")
        
        if context:
            sections.append(f"**Context**: {context}\n")
        
        if suggestions:
            sections.append("### 💡 Sugestii:\n")
            for suggestion in suggestions:
                sections.append(f"- {suggestion}")
        
        sections.append("\n*Vă rugăm să încercați din nou sau să contactați suportul.*")
        
        return "\n".join(sections)
    
    # =========================================================================
    # PRIVATE METHODS - SECTION BUILDERS
    # =========================================================================
    
    def _build_executive_summary(
        self,
        file_analysis: Dict[str, Any],
        gap_analysis: Any,
        cross_reference: Optional[Dict[str, Any]]
    ) -> str:
        """Build impressive executive summary"""
        
        # Count files processed
        files_processed = 0
        if file_analysis.get('dxf_analysis'):
            files_processed += 1
        if file_analysis.get('pdf_analysis'):
            files_processed += 1
        if file_analysis.get('txt_analysis'):
            files_processed += 1
        
        # Determine overall status
        confidence = gap_analysis.overall_confidence if gap_analysis else 0.0
        can_generate = gap_analysis.can_generate_offer if gap_analysis else False
        
        if can_generate:
            status_emoji = "✅"
            status_text = "Pregătit pentru generarea ofertei"
        elif confidence > 0.5:
            status_emoji = "⏳"
            status_text = "În progres - necesită informații suplimentare"
        else:
            status_emoji = "📋"
            status_text = "Analiză inițială completă"
        
        # Build summary
        lines = []
        lines.append("# 🎯 REZUMAT EXECUTIV\n")
        lines.append(f"{status_emoji} **Status Proiect**: {status_text}")
        lines.append(f"📊 **Nivel Confidence**: {confidence:.1%}")
        lines.append(f"📁 **Fișiere analizate**: {files_processed}")
        
        # Consistency check
        if cross_reference:
            consistency = cross_reference.get('consistency_score', 1.0)
            if consistency < 1.0:
                lines.append(f"⚠️ **Consistență Date**: {consistency:.1%} (detectate inconsistențe)")
            else:
                lines.append(f"✓ **Consistență Date**: {consistency:.1%}")
        
        # Key insights
        lines.append("\n### 🔍 Constatări cheie:")
        
        # From DXF
        if file_analysis.get('dxf_analysis'):
            dxf = file_analysis['dxf_analysis'].get('dxf_analysis', {})
            if dxf.get('total_area'):
                lines.append(f"- ✓ Suprafață identificată: **{dxf['total_area']:.1f} mp**")
            if dxf.get('total_rooms'):
                lines.append(f"- ✓ Spații detectate: **{dxf['total_rooms']} camere**")
            if dxf.get('has_hvac'):
                hvac_count = len(dxf.get('hvac_inventory', []))
                lines.append(f"- ✓ Sistem HVAC: **{hvac_count} unități detectate**")
            if dxf.get('has_electrical'):
                elec_count = len(dxf.get('electrical_inventory', []))
                lines.append(f"- ✓ Instalații electrice: **{elec_count} componente**")
        
        # From RFP
        if file_analysis.get('rfp_data'):
            rfp = file_analysis['rfp_data']
            if rfp.get('project_info', {}).get('client'):
                lines.append(f"- ✓ Client identificat: **{rfp['project_info']['client']}**")
            if rfp.get('timeline', {}).get('duration_days'):
                lines.append(f"- ✓ Durată proiect: **{rfp['timeline']['duration_days']} zile**")
        
        # Critical gaps
        if gap_analysis and gap_analysis.critical_gaps:
            lines.append(f"- ⚠️ Date critice lipsă: **{len(gap_analysis.critical_gaps)} elemente**")
        
        # Call to action
        if can_generate:
            lines.append("\n✨ **Puteți solicita acum generarea ofertei finale!**")
        else:
            critical_count = len(gap_analysis.critical_gaps) if gap_analysis else 0
            if critical_count > 0:
                lines.append(f"\n📝 **Următorul pas**: Furnizați {critical_count} informații critice pentru ofertă")
        
        return "\n".join(lines)
    
    def _build_rfp_context(self, rfp_data: Dict[str, Any]) -> str:
        """Build RFP context section"""
        lines = []
        lines.append("## 📄 CONTEXT PROIECT (din RFP)\n")
        
        project_info = rfp_data.get('project_info', {})
        timeline = rfp_data.get('timeline', {})
        financial = rfp_data.get('financial', {})
        scope = rfp_data.get('scope', {})
        team = rfp_data.get('team', {})
        
        # Project basics
        if project_info.get('name'):
            lines.append(f"**Proiect**: {project_info['name']}")
        if project_info.get('client'):
            lines.append(f"**Client**: {project_info['client']}")
        if project_info.get('location'):
            lines.append(f"**Locație**: {project_info['location']}")
        if project_info.get('building') and project_info.get('floor'):
            lines.append(f"**Clădire/Etaj**: {project_info['building']}, Etajul {project_info['floor']}")
        
        # Timeline box
        if timeline.get('work_start') or timeline.get('work_end'):
            lines.append("\n### ⏱️ Timeline")
            
            if timeline.get('work_start') and timeline.get('work_end'):
                start = timeline['work_start'][:10]
                end = timeline['work_end'][:10]
                duration = timeline.get('duration_days', '?')
                lines.append(f"**Perioadă execuție**: {start} → {end}")
                lines.append(f"**Durată**: {duration} zile lucrătoare")
            
            if timeline.get('submission_deadline'):
                deadline = timeline['submission_deadline'][:16]
                lines.append(f"**⏰ Deadline ofertă**: {deadline}")
            
            if timeline.get('inspection_deadline'):
                inspection = timeline['inspection_deadline'][:10]
                lines.append(f"**Deadline inspecție**: {inspection}")
        
        # Financial terms box
        if any(financial.values()):
            lines.append("\n### 💰 Termeni Financiari")
            
            if financial.get('currency'):
                lines.append(f"**Monedă**: {financial['currency']} (fără TVA)")
            if financial.get('guarantee_months'):
                lines.append(f"**Garanție**: {financial['guarantee_months']} luni")
            if financial.get('performance_bond'):
                lines.append(f"**Performance Bond**: {financial['performance_bond']}%")
            if financial.get('retention'):
                lines.append(f"**Retenție**: {financial['retention']}%")
        
        # Scope highlights
        scope_items = scope.get('items', [])
        if scope_items:
            lines.append("\n### 🔨 Domeniu Lucrări")
            lines.append(f"*{len(scope_items)} activități identificate în RFP*\n")
            
            # Show top 5 with visual hierarchy
            for i, item in enumerate(scope_items[:5], 1):
                # Truncate long items
                display_item = item[:80] + "..." if len(item) > 80 else item
                lines.append(f"{i}. {display_item}")
            
            if len(scope_items) > 5:
                lines.append(f"   *... și încă {len(scope_items) - 5} activități*")
        
        # Deliverables
        deliverables = scope.get('deliverables', [])
        if deliverables:
            lines.append("\n### 📦 Livrabile Obligatorii")
            for deliverable in deliverables[:3]:
                lines.append(f"- ✓ {deliverable[:80]}")
            if len(deliverables) > 3:
                lines.append(f"- *... și încă {len(deliverables) - 3} livrabile*")
        
        # Team
        if team.get('project_manager') or team.get('designer'):
            lines.append("\n### 👥 Echipa Proiect")
            if team.get('project_manager'):
                pm = team['project_manager']
                contact = team.get('pm_contact', '')
                lines.append(f"**Project Manager**: {pm}")
                if contact:
                    lines.append(f"   *Contact*: {contact}")
            if team.get('designer'):
                lines.append(f"**Proiectant**: {team['designer']}")
            if team.get('general_contractor'):
                lines.append(f"**General Contractor**: {team['general_contractor']}")
        
        return "\n".join(lines)
    
    def _build_dxf_analysis(self, dxf: Dict[str, Any]) -> str:
        """
        Build DXF technical analysis section with storage information
        
        Week 4 Enhancement: Shows where data is stored (Firestore vs GCS)
        """
        # Support both wrapper shapes: either the caller passed a wrapper
        # {'dxf_analysis': {...}} or the inner dict directly. Normalize to inner dict.
        if isinstance(dxf, dict) and 'dxf_analysis' in dxf:
            dxf = dxf.get('dxf_analysis', {})

        lines = []
        lines.append("## 🏗️ ANALIZĂ TEHNICĂ DXF\n")
        
        # Storage information (NEW)
        has_geometric = dxf.get('has_geometric_data', False)
        geometric_storage = dxf.get('geometric_storage', 'unknown')

        if has_geometric and geometric_storage == 'gcs':
            lines.append("### � Informații Stocare")
            lines.append("**Datele geometrice** sunt stocate în GCS (Google Cloud Storage)")
            lines.append("**Rezumatul și metadatele** sunt în Firestore pentru acces rapid")
            lines.append("**Precizie**: Date geometrice complete disponibile pentru generare desen\n")

        # Project overview
        lines.append("### 📋 Prezentare Generală")

        total_rooms = dxf.get('total_rooms', 0)
        total_area = dxf.get('total_area', 0)

        if total_rooms > 0:
            lines.append(f"**Spații identificate**: {total_rooms}")
        if total_area > 0:
            lines.append(f"**Suprafață totală**: {total_area:.2f} mp")

        # Document metadata
        doc_type = dxf.get('document_type', 'unknown')
        project_type = dxf.get('project_type', 'unknown')

        type_labels = {
            'floor_plan': 'Plan arhitectural',
            'specification_sheet': 'Fișă specificații',
            'technical_detail': 'Detaliu tehnic',
            'residential': 'Rezidențial',
            'commercial': 'Comercial'
        }

        if doc_type != 'unknown':
            lines.append(f"**Tip document**: {type_labels.get(doc_type, doc_type)}")
        if project_type != 'unknown':
            lines.append(f"**Tip proiect**: {type_labels.get(project_type, project_type)}")

        # Dimensions info
        if dxf.get('has_dimensions'):
            dims = dxf.get('dimensions', {})
            if dims.get('length') and dims.get('width'):
                lines.append(f"**Dimensiuni**: {dims['length']:.1f}m × {dims['width']:.1f}m")

        # Room breakdown table
        room_breakdown = dxf.get('room_breakdown', [])
        if room_breakdown:
            lines.append("\n### 🚪 Detalii Spații")
            lines.append("\n| Nr | Tip Spațiu | Suprafață |")
            lines.append("|---|---|---|")
            
            for i, room in enumerate(room_breakdown[:8], 1):
                room_type = room.get('type', 'Spațiu')
                area = room.get('area', 0)
                lines.append(f"| {i} | {room_type} | {area:.1f} mp |")
            
            if len(room_breakdown) > 8:
                lines.append(f"| ... | *și încă {len(room_breakdown) - 8} spații* | ... |")

        # Systems detected
        lines.append("\n### ⚙️ Sisteme Detectate")
        
        if dxf.get('has_hvac'):
            hvac_inv = dxf.get('hvac_inventory', [])
            lines.append(f"**✓ HVAC**: {len(hvac_inv)} unități detectate")
            if hvac_inv:
                # Show types
                hvac_types = {}
                for unit in hvac_inv:
                    unit_type = unit.get('type', 'Unknown')
                    hvac_types[unit_type] = hvac_types.get(unit_type, 0) + 1
                
                for hvac_type, count in list(hvac_types.items())[:3]:
                    lines.append(f"   - {hvac_type}: {count}x")
        else:
            lines.append("**○ HVAC**: Nu detectat în plan")
        
        if dxf.get('has_electrical'):
            elec_inv = dxf.get('electrical_inventory', [])
            lines.append(f"**✓ Instalații Electrice**: {len(elec_inv)} componente")
        else:
            lines.append("**○ Instalații Electrice**: Nu detectate complet")

        # Walls and partitions
        wall_types = dxf.get('wall_types', [])
        if wall_types:
            lines.append(f"\n**Pereți**: {len(wall_types)} tipuri detectate")
            for wall in wall_types[:3]:
                if isinstance(wall, dict):
                    wall_info = wall.get('type_code', 'Unknown')
                    if wall.get('thickness_mm'):
                        wall_info += f" - grosime {wall['thickness_mm']}mm"
                    lines.append(f"   - {wall_info}")

        # Technical notes
        tech_notes = dxf.get('technical_notes', [])
        if tech_notes:
            lines.append("\n### 📝 Note Tehnice")
            for note in tech_notes[:3]:
                lines.append(f"- {note}")

        return "\n".join(lines)

    def _build_storage_info_section(self, file_analysis: Dict[str, Any]) -> str:
        """
        Build detailed storage information section for debugging and transparency.
        
        Week 4 Enhancement: Shows file sizes, storage locations, and split information.
        """
        lines = []
        lines.append("## 💾 INFORMAȚII STOCARE\n")
        
        dxf_analysis = file_analysis.get('dxf_analysis')
        pdf_analysis = file_analysis.get('pdf_analysis')
        
        total_firestore_size = 0
        total_gcs_size = 0
        files_info = []
        
        # DXF file storage info
        if dxf_analysis:
            # Support wrapper shape
            dxf = dxf_analysis.get('dxf_analysis') if isinstance(dxf_analysis, dict) and 'dxf_analysis' in dxf_analysis else dxf_analysis
            has_geometric = dxf.get('has_geometric_data', False)
            geometric_storage = dxf.get('geometric_storage', 'unknown')
            
            if has_geometric and geometric_storage == 'gcs':
                # Geometric data split to GCS
                geometric_ref = dxf.get('geometric_ref', {})
                gcs_size = geometric_ref.get('size_bytes', 0)
                total_gcs_size += gcs_size
                
                files_info.append({
                    'type': 'DXF Geometric Data',
                    'location': 'GCS (Cloud Storage)',
                    'size': gcs_size,
                    'note': 'Date geometrice complete (vertices, boundaries)'
                })
                
                files_info.append({
                    'type': 'DXF Summary',
                    'location': 'Firestore',
                    'size': 5120,  # Estimate ~5KB for summary
                    'note': 'Rezumat și metadate pentru acces rapid'
                })
                total_firestore_size += 5120
            else:
                # All data in Firestore
                files_info.append({
                    'type': 'DXF Complete',
                    'location': 'Firestore',
                    'size': 15360,  # Estimate ~15KB
                    'note': 'Date complete în Firestore'
                })
                total_firestore_size += 15360
        
        # PDF file storage info
        if pdf_analysis:
            has_tables = len(pdf_analysis.get('tables_extracted', [])) > 0 if isinstance(pdf_analysis, dict) else False
            
            files_info.append({
                'type': 'PDF Analysis',
                'location': 'Firestore',
                'size': 10240,  # Estimate ~10KB
                'note': f"Text extractat{' + tabele' if has_tables else ''}"
            })
            total_firestore_size += 10240
        
        # Display storage table
        if files_info:
            lines.append("| Tip Date | Locație | Dimensiune | Detalii |")
            lines.append("|---|---|---|---|")
            
            for info in files_info:
                size_kb = info['size'] / 1024
                lines.append(f"| {info['type']} | {info['location']} | {size_kb:.1f} KB | {info['note']} |")
            
            lines.append("")
            lines.append(f"**Total Firestore**: {total_firestore_size / 1024:.1f} KB")
            if total_gcs_size > 0:
                lines.append(f"**Total GCS**: {total_gcs_size / 1024:.1f} KB")
            
            lines.append("\n### 📊 Beneficii Split Storage")
            lines.append("- ✓ Firestore: Acces rapid la rezumate pentru conversație")
            lines.append("- ✓ GCS: Date geometrice complete pentru generare desen")
            lines.append("- ✓ Performanță optimizată: Încărcare rapidă + precizie maximă")
        
        return "\n".join(lines)
    
    def _build_other_files_summary(self, file_analysis: Dict[str, Any]) -> str:
        """Build summary for PDF and TXT files"""
        lines = []
        lines.append("## 📎 DOCUMENTE ADIȚIONALE\n")
        
        # PDF analysis
        if file_analysis.get('pdf_analysis'):
            pdf = file_analysis['pdf_analysis']
            lines.append("### 📄 Document PDF")
            
            if pdf.get('document_type'):
                lines.append(f"**Tip**: {pdf['document_type']}")
            
            if pdf.get('page_count'):
                lines.append(f"**Pagini**: {pdf['page_count']}")
            
            if pdf.get('key_topics'):
                topics = pdf['key_topics'][:5]
                lines.append(f"**Subiecte cheie**: {', '.join(topics)}")
            
            if pdf.get('tables_found'):
                lines.append(f"**Tabele detectate**: {pdf['tables_found']}")
        
        # TXT analysis
        if file_analysis.get('txt_analysis'):
            txt = file_analysis['txt_analysis']
            lines.append("\n### 📝 Fișier Text")
            
            if txt.get('requirements'):
                reqs = txt['requirements'][:5]
                lines.append(f"**Cerințe extrase**: {len(reqs)}")
                for req in reqs[:3]:
                    lines.append(f"   - {req[:80]}")
            
            if txt.get('keywords'):
                keywords = txt['keywords'][:10]
                lines.append(f"**Cuvinte cheie**: {', '.join(keywords)}")
        
        return "\n".join(lines)
    
    def _build_detailed_file_descriptions(self, file_analysis: Dict[str, Any]) -> str:
        """
        Build detailed descriptions of uploaded files
        PHASE 1: This is what users see immediately after upload
        """
        if not file_analysis:
            return ""
    
        lines = []
        lines.append("## 📄 DESCRIERE DETALIATĂ FIȘIERE\n")
    
        # DXF Description
        dxf_data = file_analysis.get('dxf_analysis', {})
        if dxf_data.get('dxf_analysis'):
            dxf_inner = dxf_data.get('dxf_analysis', {})
        
            lines.append("### 🏗️ Plan Tehnic DXF\n")
        
            # Basic info
            total_area = dxf_inner.get('total_area', 0)
            total_rooms = dxf_inner.get('total_rooms', 0)
        
            if total_area > 0:
                lines.append(f"**Suprafață totală detectată:** {total_area:.2f} mp")
            if total_rooms > 0:
                lines.append(f"**Număr spații identificate:** {total_rooms}")
        
            # Room breakdown
            room_breakdown = dxf_inner.get('room_breakdown', [])
            if room_breakdown:
                lines.append(f"\n**Detalii spații ({len(room_breakdown)} spații):**")
                for room in room_breakdown[:10]:  # Show first 10
                    room_name = room.get('romanian_name') or room.get('room_type', 'Unknown')
                    room_area = room.get('area', 0)
                    if room_area > 0:
                        lines.append(f"  • {room_name}: {room_area:.1f} mp")
            
                if len(room_breakdown) > 10:
                    lines.append(f"  • ... și încă {len(room_breakdown) - 10} spații")
        
            # MEP Systems
            has_hvac = dxf_inner.get('has_hvac', False)
            has_electrical = dxf_inner.get('has_electrical', False)
        
            if has_hvac or has_electrical:
                lines.append("\n**Sisteme instalații detectate:**")
            
                if has_hvac:
                    hvac_count = len(dxf_inner.get('hvac_inventory', []))
                    lines.append(f"  • HVAC: {hvac_count} unități")
            
                if has_electrical:
                    electrical_count = len(dxf_inner.get('electrical_inventory', []))
                    lines.append(f"  • Instalații electrice: {electrical_count} componente")
        
            # Wall types
            wall_types = dxf_inner.get('wall_types', [])
            if wall_types:
                lines.append(f"\n**Tipuri pereți:** {len(wall_types)} tipuri detectate")
        
            # Dimensions
            if dxf_inner.get('has_dimensions'):
                lines.append("\n✓ **Plan cotat** - dimensiuni disponibile pentru calcule precise")
            else:
                lines.append("\n⚠️ **Plan necotat** - recomandăm măsurători suplimentare")
    
        # PDF Description
        pdf_data = file_analysis.get('pdf_analysis', {})
        if pdf_data:
            lines.append("\n### 📋 Document PDF - Cerințe/Specificații\n")
        
            page_count = pdf_data.get('page_count', 0)
            if page_count > 0:
                lines.append(f"**Număr pagini:** {page_count}")
        
            # Construction specs
            construction_specs = pdf_data.get('construction_specs', [])
            if construction_specs:
                lines.append(f"\n**Specificații tehnice identificate ({len(construction_specs)}):**")
                for spec in construction_specs[:5]:
                    lines.append(f"  • {spec}")
            
                if len(construction_specs) > 5:
                    lines.append(f"  • ... și încă {len(construction_specs) - 5} specificații")
        
            # Materials
            materials = pdf_data.get('material_references', [])
            if materials:
                lines.append(f"\n**Materiale menționate ({len(materials)}):**")
                for mat in materials[:5]:
                    lines.append(f"  • {mat}")
            
                if len(materials) > 5:
                    lines.append(f"  • ... și încă {len(materials) - 5} materiale")
    
        # TXT Description
        txt_data = file_analysis.get('txt_analysis', {})
        if txt_data:
            lines.append("\n### 📝 Document Text - Cerințe Cliente\n")
        
            requirements = txt_data.get('requirements', [])
            if requirements:
                lines.append(f"**Cerințe identificate ({len(requirements)}):**")
                for req in requirements[:5]:
                    lines.append(f"  • {req}")
            
                if len(requirements) > 5:
                    lines.append(f"  • ... și încă {len(requirements) - 5} cerințe")
    
        if len(lines) > 1:  # If we have content beyond the header
            return "\n".join(lines)
        else:
            return ""

    def _build_data_quality(
        self,
        gap_analysis: Any,
        cross_reference: Optional[Dict[str, Any]]
    ) -> str:
        """Build data quality assessment section"""
        lines = []
        lines.append("## 📊 CALITATEA DATELOR\n")
        
        # Overall confidence
        if gap_analysis:
            confidence = gap_analysis.overall_confidence
            lines.append(f"**Nivel Confidence Global**: {confidence:.1%}")
            
            # Visual confidence bar
            bar_length = 20
            filled = int(confidence * bar_length)
            bar = "█" * filled + "░" * (bar_length - filled)
            lines.append(f"`{bar}`\n")
        
        # Completeness by category
        if gap_analysis and gap_analysis.data_completeness:
            lines.append("### 📈 Completitudine pe Categorii\n")
            
            category_names = {
                'financial': 'Financiar',
                'technical': 'Tehnic',
                'timeline': 'Timeline',
                'materials': 'Materiale',
                'compliance': 'Conformitate',
                'client': 'Client'
            }
            
            for category, score in gap_analysis.data_completeness.items():
                category_ro = category_names.get(category, category.title())
                
                # Emoji and visual bar
                if score > 0.8:
                    emoji = "✅"
                elif score > 0.5:
                    emoji = "⚠️"
                else:
                    emoji = "❌"
                
                bar_length = 10
                filled = int(score * bar_length)
                bar = "█" * filled + "░" * (bar_length - filled)
                
                lines.append(f"{emoji} **{category_ro}**: `{bar}` {score:.0%}")
        
        # Available data summary
        if gap_analysis and gap_analysis.available_data_summary:
            lines.append("\n### ✅ Date Disponibile")
            for item in gap_analysis.available_data_summary[:8]:
                lines.append(f"- ✓ {item}")
            
            if len(gap_analysis.available_data_summary) > 8:
                remaining = len(gap_analysis.available_data_summary) - 8
                lines.append(f"- *... și încă {remaining} elemente*")
        
        # Consistency score
        if cross_reference:
            consistency = cross_reference.get('consistency_score', 1.0)
            lines.append(f"\n**Consistență Date**: {consistency:.1%}")
            
            if consistency < 1.0:
                lines.append("*Detectate inconsistențe între surse - vezi secțiunea Validare*")
        
        return "\n".join(lines)
    
    def _build_conflicts_summary(self, cross_reference: Dict[str, Any]) -> str:
        """Build conflicts summary section"""
        conflicts = cross_reference.get('conflicts', [])
        
        if not conflicts:
            return ""
        
        lines = []
        lines.append("## ⚠️ VALIDARE CONSISTENȚĂ\n")
        
        error_count = cross_reference.get('error_count', 0)
        warning_count = cross_reference.get('warning_count', 0)
        
        # Status badge
        if error_count > 0:
            lines.append("🔴 **Status**: Conflicte critice detectate")
        elif warning_count > 0:
            lines.append("🟡 **Status**: Avertismente detectate")
        else:
            lines.append("ℹ️ **Status**: Doar notificări informative")
        
        lines.append(f"**Total conflicte**: {len(conflicts)} ({error_count} critice, {warning_count} avertismente)\n")
        
        # Show critical conflicts
        errors = [c for c in conflicts if c.get('severity') == 'error']
        if errors:
            lines.append("### 🔴 Conflicte Critice (Blochează Oferta)\n")
            for i, conflict in enumerate(errors, 1):
                field = conflict.get('field', 'unknown').replace('_', ' ').title()
                desc = conflict.get('description', '')
                rec = conflict.get('recommendation', '')
                
                lines.append(f"**{i}. {field}**")
                lines.append(f"   {desc}")
                lines.append(f"   💡 *{rec}*\n")
        
        # Show warnings (max 3)
        warnings = [c for c in conflicts if c.get('severity') == 'warning']
        if warnings:
            lines.append("### 🟡 Avertismente (Recomandăm Clarificări)\n")
            for i, conflict in enumerate(warnings[:3], 1):
                field = conflict.get('field', 'unknown').replace('_', ' ').title()
                desc = conflict.get('description', '')
                
                lines.append(f"**{i}. {field}**: {desc}")
            
            if len(warnings) > 3:
                lines.append(f"\n*... și încă {len(warnings) - 3} avertismente*")
        
        # Recommendations
        recommendations = cross_reference.get('recommendations', [])
        if recommendations:
            lines.append("\n### 💡 Recomandări")
            for rec in recommendations[:5]:
                lines.append(rec)
        
        return "\n".join(lines)
    
    def _build_next_steps(
        self,
        gap_analysis: Any,
        cross_reference: Optional[Dict[str, Any]],
        file_analysis: Optional[Dict[str, Any]] = None  # ✅ NEW PARAMETER
    ) -> str:
        """
        Build adaptive next steps based on files uploaded and gaps
        PHASE 1: File-aware questions
        """
        lines = []
        lines.append("## 📋 PAȘI URMĂTORI\n")
    
        # ✅ PHASE 1: Analyze file context
        file_context = self._analyze_file_context(file_analysis) if file_analysis else {}
    
        # Check for blocking issues
        has_critical_gaps = gap_analysis and len(gap_analysis.critical_gaps) > 0
        has_critical_conflicts = (cross_reference and 
                             cross_reference.get('error_count', 0) > 0)
    
        can_generate = gap_analysis and gap_analysis.can_generate_offer
    
        # CRITICAL CONFLICTS - blocks everything
        if has_critical_conflicts:
            lines.append("### 🔴 Urgent: Rezolvați Conflictele Critice")
            lines.append("Conflictele critice detectate blochează generarea ofertei.")
            lines.append("Vă rugăm să clarificați inconsistențele menționate mai sus.\n")
            return "\n".join(lines)
    
        # CRITICAL GAPS - need data
        if has_critical_gaps:
            critical_count = len(gap_analysis.critical_gaps)
            lines.append(f"### 📝 Informații Critice Necesare ({critical_count})")
        
            # ✅ PHASE 1: File-context aware message
            if file_context.get('has_drawing') and not file_context.get('has_specification'):
                lines.append("Am analizat planul tehnic. Pentru ofertă completă, mai necesit:\n")
            elif file_context.get('has_specification') and not file_context.get('has_drawing'):
                lines.append("Am analizat specificațiile. Pentru calcule precise, mai necesit:\n")
            elif file_context.get('has_drawing') and file_context.get('has_specification'):
                lines.append("Am analizat desenul și specificațiile. Mai necesit câteva detalii:\n")
            else:
                lines.append("Pentru a genera oferta, vă rugăm să furnizați:\n")
        
            # Show critical questions with file context
            if gap_analysis.prioritized_questions:
                for i, question in enumerate(gap_analysis.prioritized_questions[:5], 1):
                    lines.append(f"**{i}.** {question}\n")
        
            # ✅ PHASE 1: File-specific guidance
            lines.append(self._build_file_specific_guidance(file_context, gap_analysis))
        
        elif can_generate:
            lines.append("### ✨ Gata de Generare Ofertă!")
            lines.append("Toate informațiile necesare sunt complete.")
            lines.append("**Puteți solicita acum generarea ofertei finale.**\n")
            lines.append("*Comandă*: \"Generează oferta\" sau \"Vreau oferta\"")
        
        else:
            # Non-critical - show improvement suggestions
            if gap_analysis and gap_analysis.prioritized_questions:
                lines.append("### 💬 Întrebări pentru Îmbunătățirea Ofertei")
                lines.append("Răspunsurile la aceste întrebări vor crește acuratețea ofertei:\n")
            
                for i, question in enumerate(gap_analysis.prioritized_questions[:5], 1):
                    lines.append(f"**{i}.** {question}\n")
    
        # Progress indicator
        if gap_analysis:
            total_requirements = 20
            filled = int(gap_analysis.overall_confidence * total_requirements)
            lines.append(f"\n**Progres**: {filled}/{total_requirements} cerințe completate ({gap_analysis.overall_confidence:.0%})")
    
        return "\n".join(lines)
    
    def _build_questions_section(self, gap_analysis: Any) -> str:
        """Build standalone questions section"""
        if not gap_analysis or not gap_analysis.prioritized_questions:
            return ""
        
        lines = []
        lines.append("## ❓ ÎNTREBĂRI\n")
        
        for i, question in enumerate(gap_analysis.prioritized_questions, 1):
            lines.append(f"**{i}.** {question}\n")
        
        return "\n".join(lines)
    
    def _build_file_specific_guidance(
        self,
        file_context: Dict[str, Any],
        gap_analysis: Any
    ) -> str:
        """
        Generate file-specific guidance based on what's uploaded
        PHASE 1: Context-aware suggestions
        """
        lines = []
        lines.append("\n### 💡 Sugestii bazate pe fișierele dumneavoastră:\n")
    
        has_drawing = file_context.get('has_drawing')
        has_spec = file_context.get('has_specification')
        drawing_level = file_context.get('drawing_detail_level', 'none')
    
        # DXF only - basic
        if has_drawing and not has_spec and drawing_level == 'basic':
            lines.append("✓ Am identificat camerele din plan")
            lines.append("✗ Lipsesc detalii MEP (instalații electrice, HVAC)")
            lines.append("✗ Nu am găsit specificații de materiale\n")
            lines.append("**Recomandare:** Încărcați un PDF cu detalii despre:")
            lines.append("  • Tipuri de finisaje dorite")
            lines.append("  • Sistemul de climatizare preferat")
            lines.append("  • Nivel de finisaje (standard/premium/luxury)")
    
        # DXF only - good detail
        elif has_drawing and not has_spec and drawing_level == 'good':
            lines.append("✓ Am identificat camerele și instalațiile MEP din plan")
            lines.append("✗ Nu am găsit specificații despre finisaje și materiale\n")
            lines.append("**Recomandare:** Adăugați informații despre:")
            lines.append("  • Nivel finisaje dorit")
            lines.append("  • Timeline lucrări")
    
        # PDF/Spec only
        elif has_spec and not has_drawing:
            lines.append("✓ Am identificat cerințele din specificații")
            lines.append("✗ Lipsește desenul tehnic pentru calcul cantități\n")
            lines.append("**Recomandare:** Încărcați:")
            lines.append("  • Plan arhitectură (.DXF) pentru calcule precise")
            lines.append("  • SAU furnizați suprafețe și dimensiuni în text")
    
        # Both files but missing critical data
        elif has_drawing and has_spec:
            # Check what's actually missing from gaps
            critical_gaps = gap_analysis.critical_gaps if gap_analysis else []
            gap_names = [g.field_name for g in critical_gaps]
        
            # ✅ REMOVED: Budget gap message - customers don't have budgets
            if 'finish_level' in gap_names:
                lines.append("✓ Am desenul și specificațiile de bază")
                lines.append("✗ Lipsește nivelul de finisaje dorit\n")
    
        return "\n".join(lines) if len(lines) > 1 else ""
    
    def _build_compact_status(
        self,
        gap_analysis: Any,
        cross_reference: Optional[Dict[str, Any]]
    ) -> str:
        """Build compact status for ongoing conversations"""
        if not gap_analysis:
            return ""
        
        lines = []
        lines.append("---")
        lines.append(f"📊 **Status**: Confidence {gap_analysis.overall_confidence:.0%}")
        
        if gap_analysis.can_generate_offer:
            lines.append("| ✅ Pregătit pentru ofertă")
        else:
            critical_count = len(gap_analysis.critical_gaps)
            high_count = len(gap_analysis.high_priority_gaps)
            
            if critical_count > 0:
                lines.append(f"| ⏳ Lipsă: {critical_count} critice")
            elif high_count > 0:
                lines.append(f"| ⚠️ Lipsă: {high_count} importante")
        
        # Show consistency if issues
        if cross_reference and cross_reference.get('consistency_score', 1.0) < 0.95:
            consistency = cross_reference['consistency_score']
            lines.append(f"| ⚠️ Consistență: {consistency:.0%}")
        
        return " ".join(lines) + "\n"
    
    def _build_progress_indicator(self, context: Dict[str, Any]) -> str:
        """Build progress indicator showing improvement"""
        improvement = context.get('confidence_improvement', 0)
        gaps_closed = context.get('gaps_closed', 0)
        
        if improvement <= 0 and gaps_closed <= 0:
            return ""
        
        lines = []
        lines.append("---")
        lines.append("### 📈 PROGRES")
        
        if improvement > 0:
            lines.append(f"**Îmbunătățire confidence**: +{improvement:.1%}")
        
        if gaps_closed > 0:
            lines.append(f"**Date completate**: {gaps_closed} elemente noi")
            lines.append("✨ *Excelent! Continuați să furnizați informații.*")
        
        return "\n".join(lines) + "\n"
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def format_list(
        self,
        items: List[str],
        max_items: Optional[int] = None,
        numbered: bool = False
    ) -> str:
        """
        Format a list with optional truncation
        
        Args:
            items: List of items to format
            max_items: Maximum items to show (None = show all)
            numbered: Use numbered list (1, 2, 3) vs bullets (-)
        """
        if not items:
            return ""
        
        max_items = max_items or self.max_list_items
        lines = []
        
        for i, item in enumerate(items[:max_items], 1):
            if numbered:
                lines.append(f"{i}. {item}")
            else:
                lines.append(f"- {item}")
        
        if len(items) > max_items:
            remaining = len(items) - max_items
            lines.append(f"- *... și încă {remaining} elemente*")
        
        return "\n".join(lines)
    
    def format_table(
        self,
        headers: List[str],
        rows: List[List[str]],
        max_rows: Optional[int] = None
    ) -> str:
        """
        Format a markdown table
        
        Args:
            headers: Column headers
            rows: Table rows
            max_rows: Maximum rows to show
        """
        if not headers or not rows:
            return ""
        
        max_rows = max_rows or 10
        lines = []
        
        # Header
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---" for _ in headers]) + "|")
        
        # Rows
        for row in rows[:max_rows]:
            lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
        
        if len(rows) > max_rows:
            remaining = len(rows) - max_rows
            lines.append(f"| ... | *și încă {remaining} rânduri* | ... |")
        
        return "\n".join(lines)
    
    def format_progress_bar(
        self,
        value: float,
        length: int = 20,
        filled_char: str = "█",
        empty_char: str = "░"
    ) -> str:
        """
        Create a visual progress bar
        
        Args:
            value: Value between 0 and 1
            length: Length of the bar
            filled_char: Character for filled portion
            empty_char: Character for empty portion
        """
        value = max(0.0, min(1.0, value))
        filled = int(value * length)
        empty = length - filled
        
        return f"`{filled_char * filled}{empty_char * empty}` {value:.0%}"
    
    def format_key_value(
        self,
        key: str,
        value: Any,
        bold_key: bool = True
    ) -> str:
        """Format key-value pair"""
        if bold_key:
            return f"**{key}**: {value}"
        else:
            return f"{key}: {value}"
    
    def format_section_header(
        self,
        title: str,
        level: int = 2,
        emoji: Optional[str] = None
    ) -> str:
        """Format section header"""
        prefix = "#" * level
        
        if emoji and self.use_emojis:
            return f"{prefix} {emoji} {title}\n"
        else:
            return f"{prefix} {title}\n"
    
    def add_divider(self, style: str = "line") -> str:
        """Add visual divider"""
        if style == "line":
            return "---\n"
        elif style == "double":
            return "===\n"
        elif style == "dots":
            return "· · ·\n"
        else:
            return "\n"
    
    def truncate_text(
        self,
        text: str,
        max_length: int = 100,
        suffix: str = "..."
    ) -> str:
        """Truncate long text"""
        if len(text) <= max_length:
            return text
        
        return text[:max_length - len(suffix)] + suffix
    
    def format_date(self, date_str: str) -> str:
        """Format date string to Romanian format"""
        try:
            if 'T' in date_str:
                # ISO format with time
                dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                return dt.strftime('%d.%m.%Y %H:%M')
            else:
                # Just date
                dt = datetime.fromisoformat(date_str)
                return dt.strftime('%d.%m.%Y')
        except:
            return date_str
    
    def highlight_important(self, text: str) -> str:
        """Highlight important text"""
        return f"**{text}**"
    
    def add_callout(
        self,
        text: str,
        callout_type: str = "info"
    ) -> str:
        """
        Add callout box
        
        Types: info, warning, success, error
        """
        emoji_map = {
            'info': 'ℹ️',
            'warning': '⚠️',
            'success': '✅',
            'error': '❌',
            'tip': '💡'
        }
        
        emoji = emoji_map.get(callout_type, 'ℹ️')
        return f"{emoji} **{callout_type.upper()}**: {text}\n"


# ============================================================
# UTILITY FUNCTIONS FOR QUICK FORMATTING
# ============================================================

def quick_format_response(
    response_type: ResponseType,
    content: Dict[str, Any]
) -> str:
    """
    Quick formatting utility for simple responses
    
    Args:
        response_type: Type of response
        content: Content dictionary with required fields
        
    Returns:
        Formatted response string
    """
    builder = ResponseBuilder()
    
    if response_type == ResponseType.ERROR:
        return builder.build_error_response(
            error_message=content.get('error', 'Unknown error'),
            context=content.get('context'),
            suggestions=content.get('suggestions')
        )
    
    elif response_type == ResponseType.PROGRESS_UPDATE:
        return builder._build_progress_indicator(content)
    
    return "Response type not supported in quick format"


def format_offer_summary(
    area: float,
    budget: str,
    timeline: str,
    confidence: float
) -> str:
    """Quick format offer summary"""
    return f"""
## 📋 REZUMAT OFERTĂ

**Suprafață**: {area:.1f} mp
**Buget**: {budget}
**Timeline**: {timeline}
**Confidence**: {confidence:.1%}
"""


def format_milestone_update(
    milestone: str,
    status: str,
    date: str
) -> str:
    """Quick format milestone update"""
    status_emoji = {
        'completed': '✅',
        'in_progress': '⏳',
        'pending': '○',
        'delayed': '⚠️'
    }
    
    emoji = status_emoji.get(status, '○')
    
    return f"{emoji} **{milestone}** - {status} ({date})"


def format_cost_breakdown(
    items: Dict[str, float],
    currency: str = "EUR"
) -> str:
    """Quick format cost breakdown"""
    lines = []
    lines.append("### 💰 Detalii Cost\n")
    
    total = 0
    for item, cost in items.items():
        lines.append(f"- {item}: {cost:,.2f} {currency}")
        total += cost
    
    lines.append(f"\n**Total**: {total:,.2f} {currency}")
    
    return "\n".join(lines)


def format_timeline_gantt(
    tasks: List[Dict[str, Any]]
) -> str:
    """Quick format timeline as simple gantt"""
    lines = []
    lines.append("### 📅 Timeline Lucrări\n")
    lines.append("| Activitate | Durată | Start | Finish |")
    lines.append("|---|---|---|---|")
    
    for task in tasks:
        name = task.get('name', 'Task')
        duration = task.get('duration_days', 0)
        start = task.get('start', '-')
        finish = task.get('finish', '-')
        
        lines.append(f"| {name} | {duration} zile | {start} | {finish} |")
    
    return "\n".join(lines)


def format_comparison_table(
    options: List[Dict[str, Any]],
    criteria: List[str]
) -> str:
    """Format comparison table for multiple options"""
    lines = []
    lines.append("### 🔄 Comparație Opțiuni\n")
    
    # Header
    header = ["Opțiune"] + criteria
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---" for _ in header]) + "|")
    
    # Rows
    for option in options:
        name = option.get('name', 'Option')
        values = [str(option.get(criterion, '-')) for criterion in criteria]
        lines.append("| " + " | ".join([name] + values) + " |")
    
    return "\n".join(lines)