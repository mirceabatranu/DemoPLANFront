# -*- coding: utf-8 -*-
# src/templates/response_templates.py
"""
Romanian Response Templates for DEMOPLAN
Professional construction industry templates with consistent formatting
"""

from typing import Dict, Any, List


class ResponseTemplates:
    """
    Collection of Romanian response templates
    All templates follow professional construction industry standards
    """
    
    # =========================================================================
    # FILE ANALYSIS TEMPLATES
    # =========================================================================
    
    FILE_ANALYSIS_HEADER = """📋 **ANALIZĂ FIȘIERE COMPLETATĂ**
━━━━━━━━━━━━━━━━━━━━━━━━━━

**Sesiune:** `{session_id}`
**Data:** {date}
**Încredere:** {confidence}

"""
    
    QUICK_SUMMARY = """📊 **SUMAR RAPID**
  • Fișiere procesate: **{files_count}**
  • Încredere analiză: **{confidence}**
  • Status: **{status}**
  • Poate genera ofertă: **{can_generate}**

"""
    
    FILES_SECTION_HEADER = """📁 **FIȘIERE DETECTATE**

"""
    
    DXF_FILE_TEMPLATE = """  📄 **Plan DXF** - `{filename}`
     • Suprafață totală: **{area} mp**
     • Camere detectate: **{rooms}**
     • Instalații electrice: {electrical}
     • Sistem HVAC: {hvac}
     • Instalații sanitare: {plumbing}

"""
    
    PDF_FILE_TEMPLATE = """  📄 **Document PDF** - `{filename}`
     • Tip document: {doc_type}
     • Cerințe extrase: {requirements_count}
     • Specificații tehnice: {specs_count}
     • Materiale menționate: {materials_count}

"""
    
    TXT_FILE_TEMPLATE = """  📄 **Fișier text** - `{filename}`
     • Cerințe utilizator: {requirements_count} puncte
     • Buget menționat: {budget}
     • Timeline menționat: {timeline}

"""
    
    # =========================================================================
    # DATA EXTRACTION TEMPLATES
    # =========================================================================
    
    EXTRACTED_DATA_HEADER = """🏗️ **DATE EXTRASE**

"""
    
    ROOM_LIST_TEMPLATE = """  **Spații detectate:**
{room_list}

"""
    
    MEP_SUMMARY = """  **Instalații MEP:**
  • Electrice: {electrical_count} componente
  • HVAC: {hvac_count} unități
  • Sanitare: {plumbing_count} puncte

"""
    
    DIMENSIONS_SUMMARY = """  **Dimensiuni:**
  • Suprafață totală: {total_area} mp
  • Suprafață utilă: {usable_area} mp
  • Înălțime camere: {ceiling_height} m

"""
    
    # =========================================================================
    # GAP ANALYSIS TEMPLATES
    # =========================================================================
    
    MISSING_DATA_HEADER = """❌ **INFORMAȚII LIPSĂ**

Pentru o ofertă completă, vă rugăm să furnizați:

"""
    
    MISSING_CRITICAL = """  🔴 **Critic** - {item}
     Necesar pentru: {reason}

"""
    
    MISSING_HIGH = """  🟡 **Prioritate înaltă** - {item}
     Impact: {impact}

"""
    
    MISSING_MEDIUM = """  🟢 **Opțional** - {item}
     Recomandare: {recommendation}

"""
    
    ALL_DATA_COMPLETE = """✅ **TOATE DATELE NECESARE**

Am toate informațiile pentru a genera o ofertă completă!

"""
    
    # =========================================================================
    # CROSS-REFERENCE TEMPLATES
    # =========================================================================
    
    CROSS_REF_HEADER = """🔍 **VALIDARE CONSISTENȚĂ DATE**

"""
    
    NO_CONFLICTS = """✅ **Toate datele sunt consistente**
   Niciun conflict detectat între fișiere

"""
    
    CONFLICTS_DETECTED = """⚠️ **Detectate {count} inconsistențe:**

"""
    
    CONFLICT_CRITICAL = """  ❌ **Conflict critic:** {description}
     Sursă 1: {source1} → {value1}
     Sursă 2: {source2} → {value2}
     Recomandare: {recommendation}

"""
    
    CONFLICT_WARNING = """  ⚠️ **Avertisment:** {description}
     Diferență: {difference}

"""
    
    # =========================================================================
    # CHAT RESPONSE TEMPLATES
    # =========================================================================
    
    PROGRESS_UPDATE = """📈 **PROGRES ACTUALIZAT**

  • Încredere: {old_confidence} → {new_confidence} (↑ {delta})
  • Goluri închise: {gaps_closed}
  • Rămân: {remaining_gaps} informații

"""
    
    COMPACT_STATUS = """📊 **STATUS PROIECT**
  • Încredere: **{confidence}**
  • Date lipsă: **{missing_count}**
  • Gata ofertă: **{ready}**

"""
    
    DATA_EXTRACTED_NOTIFICATION = """📌 **INFORMAȚII NOI DETECTATE:**

{extracted_items}

"""
    
    # =========================================================================
    # NEXT STEPS TEMPLATES
    # =========================================================================
    
    NEXT_STEPS_READY = """💡 **PAȘI URMĂTORI**

✅ **Puteți genera oferta acum!**
   Scrieți "generează oferta" sau "vreau oferta" pentru a continua

"""
    
    NEXT_STEPS_NEED_INFO = """💡 **PAȘI URMĂTORI**

📝 **Vă rog să furnizați următoarele informații:**

{questions}

Răspundeți la aceste întrebări pentru a crește încrederea analizei.

"""
    
    NEXT_STEPS_TEMPLATE = """💡 **CE URMEAZĂ?**

{status_message}

{action_items}

"""
    
    # =========================================================================
    # OFFER TEMPLATES
    # =========================================================================
    
    OFFER_HEADER = """🏗️ **OFERTĂ TEHNICĂ ȘI COMERCIALĂ**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**DEMOPLAN CONSTRUCT SRL**
**Data generare:** {date}
**Număr ofertă:** {offer_number}
**Valabilitate:** 30 zile

"""
    
    OFFER_PROJECT_SUMMARY = """**1. INFORMAȚII PROIECT**

  • Client: {client_name}
  • Locație: {location}
  • Suprafață: {area} mp
  • Tip proiect: {project_type}
  • Încredere analiză: {confidence}

"""
    
    OFFER_SCOPE = """**2. DOMENIU LUCRĂRI**

{scope_description}

**Spații incluse:**
{spaces_list}

"""
    
    OFFER_COST_TABLE = """**3. ESTIMARE COSTURI**

┌─────────────────────────────────┬───────────┬──────────────┐
│ Categorie                       │ UM        │ Preț (RON)   │
├─────────────────────────────────┼───────────┼──────────────┤
{cost_rows}
├─────────────────────────────────┴───────────┼──────────────┤
│ **TOTAL ESTIMAT:**                          │ **{total}**  │
└─────────────────────────────────────────────┴──────────────┘

"""
    
    OFFER_TIMELINE = """**4. DURATĂ EXECUȚIE**

  • Start estimat: {start_date}
  • Finalizare estimată: {end_date}
  • Durată totală: **{duration} zile lucrătoare**

**Faze lucrări:**
{phases}

"""
    
    OFFER_TERMS = """**5. TERMENI ȘI CONDIȚII**

  • Plată: {payment_terms}
  • Garanție: {warranty}
  • Condiții speciale: {special_conditions}

"""
    
    OFFER_FOOTER = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**NOTĂ IMPORTANTĂ:**
Această ofertă a fost generată automat de sistemul DEMOPLAN AI pe baza 
documentelor furnizate. Pentru modificări sau clarificări, vă rugăm să 
continuați conversația.

**Validitate:** 30 zile de la data generării
**Contact:** Pentru acceptare sau întrebări, răspundeți în chat

**Vă mulțumim pentru încrederea acordată!**
DEMOPLAN CONSTRUCT SRL

"""
    
    # =========================================================================
    # ERROR TEMPLATES
    # =========================================================================
    
    ERROR_GENERAL = """❌ **EROARE ÎN PROCESARE**

**Context:** {context}
**Detalii:** {error_message}

💡 **Ce puteți face:**
{suggestions}

"""
    
    ERROR_FILE_PROCESSING = """❌ **EROARE LA PROCESAREA FIȘIERULUI**

**Fișier:** {filename}
**Problemă:** {issue}

💡 **Verificați:**
  • Fișierul este valid și necorupt
  • Formatul este suportat (DXF, PDF, TXT)
  • Dimensiunea nu depășește 50MB

"""
    
    ERROR_TIMEOUT = """⏱️ **TIMEOUT - PROCESARE ÎNTÂRZIATĂ**

Procesarea durează mai mult decât de obicei. Acest lucru poate fi cauzat de:
  • Fișiere mari sau complexe
  • Server ocupat
  • Conexiune instabilă

🔄 **Vă rugăm să:**
  1. Așteptați câteva secunde
  2. Încercați din nou
  3. Contactați suportul dacă problema persistă

"""
    
    # =========================================================================
    # RFP CONTEXT TEMPLATES
    # =========================================================================
    
    RFP_CONTEXT_HEADER = """## 📄 CONTEXT PROIECT (din RFP)

"""
    
    RFP_PROJECT_INFO = """**Proiect:** {project_name}
**Client:** {client}
**Locație:** {location}
**Clădire/Etaj:** {building}, Etajul {floor}

"""
    
    RFP_TIMELINE = """### ⏱️ Timeline

**Perioadă execuție:** {start_date} → {end_date}
**Durată:** {duration} zile lucrătoare
**⏰ Deadline ofertă:** {submission_deadline}
**Deadline inspecție:** {inspection_deadline}

"""
    
    RFP_FINANCIAL = """### 💰 Termeni Financiari

**Monedă:** {currency} (fără TVA)
**Garanție:** {guarantee_months} luni
**Performance Bond:** {performance_bond}%
**Retenție:** {retention}%

"""
    
    RFP_SCOPE = """### 🔨 Domeniu Lucrări

*{items_count} activități identificate în RFP*

{scope_items}

"""
    
    RFP_DELIVERABLES = """### 📦 Livrabile Obligatorii

{deliverables_list}

"""
    
    RFP_TEAM = """### 👥 Echipa Proiect

**Project Manager:** {project_manager}
   *Contact*: {pm_contact}
**Proiectant:** {designer}
**General Contractor:** {general_contractor}

"""
    
    # =========================================================================
    # DXF ANALYSIS TEMPLATES
    # =========================================================================
    
    DXF_ANALYSIS_HEADER = """## 🗺️ ANALIZĂ TEHNICĂ (din plan DXF)

"""
    
    DXF_DIMENSIONS = """### 📐 Dimensiuni și Spații

**Suprafață totală:** {total_area} mp
**Spații detectate:** {total_rooms} camere
**Dimensiuni generale:** {length}m × {width}m

"""
    
    DXF_ROOM_BREAKDOWN = """### 🚪 Detalii Spații

| Nr | Tip Spațiu | Suprafață |
|---|---|---|
{room_rows}

"""
    
    DXF_SYSTEMS = """### ⚙️ Sisteme Detectate

**✓ HVAC:** {hvac_count} unități detectate
{hvac_types}

**✓ Instalații Electrice:** {electrical_count} componente

**Pereți:** {wall_types_count} tipuri detectate

"""
    
    DXF_TECH_NOTES = """### 📝 Note Tehnice

{notes_list}

"""
    
    # =========================================================================
    # QUALITY ASSESSMENT TEMPLATES
    # =========================================================================
    
    DATA_QUALITY_HEADER = """## 📊 CALITATEA DATELOR

"""
    
    CONFIDENCE_VISUAL = """**Nivel Confidence Global:** {confidence}

`{progress_bar}`

"""
    
    COMPLETENESS_BY_CATEGORY = """### 📈 Completitudine pe Categorii

{category_bars}

"""
    
    AVAILABLE_DATA_SUMMARY = """### ✅ Date Disponibile

{available_items}

"""
    
    CONSISTENCY_SCORE = """
**Consistență Date:** {consistency_score}
{consistency_note}

"""
    
    # =========================================================================
    # CONFLICTS TEMPLATES
    # =========================================================================
    
    CONFLICTS_HEADER = """## ⚠️ VALIDARE CONSISTENȚĂ

"""
    
    CONFLICTS_STATUS = """{status_badge} **Status**: {status_text}
**Total conflicte:** {total} ({critical} critice, {warnings} avertismente)

"""
    
    CRITICAL_CONFLICTS_SECTION = """### 🔴 Conflicte Critice (Blochează Oferta)

{conflicts_list}

"""
    
    WARNINGS_SECTION = """### 🟡 Avertismente (Recomandăm Clarificări)

{warnings_list}

"""
    
    RECOMMENDATIONS_SECTION = """### 💡 Recomandări

{recommendations_list}

"""
    
    # =========================================================================
    # PROGRESS INDICATORS
    # =========================================================================
    
    PROGRESS_BAR_TEMPLATE = """[{filled}{empty}] {percentage}"""
    
    CONFIDENCE_BAR = """**{category}:** `{bar}` {percentage}"""
    
    STATUS_BADGE_READY = """✅ Pregătit pentru ofertă"""
    STATUS_BADGE_IN_PROGRESS = """⏳ În progres - necesită informații suplimentare"""
    STATUS_BADGE_INITIAL = """📋 Analiză inițială completă"""
    
    # =========================================================================
    # UTILITY TEMPLATES
    # =========================================================================
    
    SEPARATOR_LIGHT = "\n━━━━━━━━━━━━━━━━━━━━\n\n"
    SEPARATOR_HEAVY = "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    SEPARATOR_DOUBLE = "\n═══════════════════════════════════════════════════\n\n"
    SEPARATOR_DOTS = "\n· · · · · · · · · · · · · · · · · · · ·\n\n"
    
    BULLET_POINT = "  • "
    SUB_BULLET = "    ○ "
    CHECK_MARK = "✓ "
    CROSS_MARK = "✗ "
    
    # =========================================================================
    # EMOJI SETS
    # =========================================================================
    
    EMOJIS = {
        'success': '✅',
        'error': '❌',
        'warning': '⚠️',
        'info': 'ℹ️',
        'question': '❓',
        'progress': '⏳',
        'complete': '✓',
        'incomplete': '○',
        'critical': '🔴',
        'high': '🟡',
        'medium': '🟢',
        'low': '⚪',
        'file': '📄',
        'folder': '📁',
        'chart': '📊',
        'building': '🏗️',
        'document': '📋',
        'money': '💰',
        'calendar': '📅',
        'clock': '⏰',
        'team': '👥',
        'tools': '🔨',
        'lightbulb': '💡',
        'rocket': '🚀',
        'target': '🎯',
        'magnifying_glass': '🔍',
        'package': '📦',
        'note': '📝',
        'dimensions': '📐',
        'systems': '⚙️',
        'door': '🚪'
    }
    
    # =========================================================================
    # RESPONSE TYPE HEADERS
    # =========================================================================
    
    RESPONSE_HEADERS = {
        'file_analysis': '# 🎯 REZUMAT EXECUTIV\n',
        'chat': '',  # No header for chat
        'offer': '# 📋 OFERTĂ TEHNICĂ ȘI COMERCIALĂ\n',
        'error': '## ❌ EROARE\n',
        'progress': '### 📈 PROGRES\n'
    }
    
    # =========================================================================
    # STATUS MESSAGES
    # =========================================================================
    
    STATUS_MESSAGES = {
        'analyzing': '⏳ Analizez fișierele...',
        'processing': '⏳ Procesez informațiile...',
        'generating': '⏳ Generez oferta...',
        'complete': '✅ Procesare completă!',
        'error': '❌ A apărut o eroare',
        'waiting': '⏳ Aștept răspunsul...'
    }
    
    # =========================================================================
    # CALL TO ACTION TEMPLATES
    # =========================================================================
    
    CTA_GENERATE_OFFER = """
✨ **Sunteți pregătit să generați oferta?**
Scrieți: "generează oferta" sau "vreau oferta"
"""
    
    CTA_PROVIDE_INFO = """
📝 **Următorul pas:** Furnizați informațiile lipsă
Răspundeți la întrebările de mai sus pentru a continua
"""
    
    CTA_RESOLVE_CONFLICTS = """
🔴 **Acțiune urgentă:** Rezolvați conflictele critice
Clarificați inconsistențele pentru a putea genera oferta
"""
    
    CTA_IMPROVE_ACCURACY = """
💡 **Opțional:** Furnizați informații suplimentare
Răspunsurile vor îmbunătăți acuratețea ofertei
"""


# =========================================================================
# TEMPLATE FORMATTING HELPERS
# =========================================================================

def format_template(template: str, **kwargs) -> str:
    """
    Format template with provided kwargs
    Handles missing keys gracefully
    """
    try:
        return template.format(**kwargs)
    except KeyError as e:
        # Return template with placeholder for missing key
        return template.replace(f"{{{e.args[0]}}}", f"[{e.args[0]}]")


def get_status_emoji(confidence: float) -> str:
    """Get appropriate emoji for confidence level"""
    if confidence >= 0.90:
        return '✅'
    elif confidence >= 0.75:
        return '🟢'
    elif confidence >= 0.50:
        return '🟡'
    elif confidence >= 0.25:
        return '🟠'
    else:
        return '🔴'


def get_status_text(confidence: float, can_generate: bool) -> str:
    """Get status text based on confidence and generation capability"""
    if can_generate:
        return "Pregătit pentru generarea ofertei"
    elif confidence > 0.5:
        return "În progres - necesită informații suplimentare"
    else:
        return "Analiză inițială completă"


def create_progress_bar(value: float, length: int = 20) -> str:
    """Create visual progress bar"""
    filled = int(value * length)
    empty = length - filled
    return '█' * filled + '░' * empty


def format_list_items(items: List[str], max_items: int = 5, with_numbers: bool = False) -> str:
    """Format list of items with optional truncation"""
    if not items:
        return ""
    
    lines = []
    display_items = items[:max_items]
    
    for i, item in enumerate(display_items, 1):
        if with_numbers:
            lines.append(f"{i}. {item}")
        else:
            lines.append(f"  • {item}")
    
    if len(items) > max_items:
        remaining = len(items) - max_items
        lines.append(f"  ... și încă {remaining} {'elemente' if remaining > 1 else 'element'}")
    
    return '\n'.join(lines)


def truncate_text(text: str, max_length: int = 80, suffix: str = "...") -> str:
    """Truncate text to maximum length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix