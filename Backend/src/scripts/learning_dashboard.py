#!/usr/bin/env python3
"""
Learning Dashboard CLI
Monitor offer learning system health and trigger learning

Usage:
    python scripts/learning_dashboard.py              # View status
    python scripts/learning_dashboard.py --learn      # Trigger learning
    python scripts/learning_dashboard.py --patterns   # View patterns
    python scripts/learning_dashboard.py --offers     # List offers
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.offer_learning_service import OfferLearningService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


class LearningDashboardCLI:
    """CLI dashboard for monitoring learning system"""
    
    def __init__(self):
        """Initialize dashboard"""
        self.learning_service = OfferLearningService()
    
    def show_status(self):
        """Display system status overview"""
        logger.info("=" * 80)
        logger.info("📊 OFFER LEARNING DASHBOARD")
        logger.info("=" * 80)
        
        try:
            stats = self.learning_service.get_learning_stats()
            
            # Data status
            logger.info("\n📁 DATA STATUS")
            logger.info("─" * 80)
            logger.info(f"  Total offers ingested:        {stats['total_offers_ingested']}")
            logger.info(f"  Offers with unit pricing:     {stats['offers_with_unit_pricing']}")
            
            if stats['total_offers_ingested'] > 0:
                unit_price_percent = (stats['offers_with_unit_pricing'] / stats['total_offers_ingested']) * 100
                logger.info(f"  Unit pricing coverage:        {unit_price_percent:.1f}%")
            
            # Learning status
            logger.info("\n🎯 LEARNING STATUS")
            logger.info("─" * 80)
            
            last_learning = stats.get('last_learning_run')
            if last_learning:
                learning_date = datetime.fromisoformat(last_learning)
                days_ago = (datetime.now() - learning_date).days
                logger.info(f"  Last learning run:            {learning_date.strftime('%Y-%m-%d %H:%M')}")
                logger.info(f"  Days since learning:          {days_ago}")
            else:
                logger.info("  Last learning run:            Never")
                logger.info("  ⚠️  Learning has not been run yet")
            
            confidence = stats.get('confidence_score', 0.0)
            logger.info(f"  Confidence score:             {confidence:.2f} ({self._get_confidence_label(confidence)})")
            
            # Pattern quality
            logger.info("\n📈 PATTERN QUALITY")
            logger.info("─" * 80)
            logger.info(f"  Category patterns:            {stats['category_patterns_count']}")
            logger.info(f"  Item types tracked:           {stats['item_types_tracked']}")
            
            # Recommendations
            logger.info("\n💡 MONITORING ALERTS")
            logger.info("─" * 80)
            
            alerts = []
            
            if stats['total_offers_ingested'] == 0:
                alerts.append("  🔴 No offers ingested. Start by uploading historical offers.")
            elif stats['total_offers_ingested'] < 10:
                alerts.append(f"  🟡 Only {stats['total_offers_ingested']} offers ingested. Add more for better patterns (target: 10+)")
            
            if not last_learning:
                alerts.append("  🔴 Learning never run. Execute: --learn")
            elif last_learning and days_ago > 7:
                alerts.append(f"  🟡 {days_ago} days since last learning. Consider running: --learn")
            
            if confidence < 0.5:
                alerts.append(f"  🟡 Low confidence ({confidence:.2f}). Ingest more offers to improve accuracy.")
            
            if stats['offers_with_unit_pricing'] == 0 and stats['total_offers_ingested'] > 0:
                alerts.append("  🟡 No detailed offers with unit pricing. Upload CCC-format offers for better learning.")
            
            if not alerts:
                alerts.append("  ✅ System healthy. All metrics nominal.")
            
            for alert in alerts:
                logger.info(alert)
            
            logger.info("\n" + "=" * 80)
            
        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            sys.exit(1)
    
    def trigger_learning(self):
        """Trigger pattern learning process"""
        logger.info("=" * 80)
        logger.info("🧠 TRIGGERING PATTERN LEARNING")
        logger.info("=" * 80)
        
        # Check if there are offers to learn from
        stats = self.learning_service.get_learning_stats()
        
        if stats['total_offers_ingested'] == 0:
            logger.error("\n❌ No offers to learn from!")
            logger.info("   First ingest offers using: python scripts/ingest_offers.py")
            sys.exit(1)
        
        logger.info(f"\nProcessing {stats['total_offers_ingested']} offer(s)...\n")
        
        try:
            result = self.learning_service.run_learning()
            
            if result['success']:
                logger.info("✅ Learning complete!\n")
                
                # Show what was learned
                logger.info("📊 LEARNING RESULTS")
                logger.info("─" * 80)
                logger.info(f"  Offers processed:             {result['offers_processed']}")
                logger.info(f"  Confidence score:             {result.get('confidence_score', 0):.2f}")
                
                patterns = result.get('patterns_learned', {})
                
                if 'category_splits' in patterns:
                    logger.info(f"\n  Category Patterns:")
                    for cat_id, cat_data in patterns['category_splits'].items():
                        logger.info(f"    Category {cat_id}:            {cat_data['avg_percent']:.1f}% (±{cat_data['std_dev']:.1f}%)")
                
                if 'item_frequencies' in patterns:
                    logger.info(f"\n  Item types tracked:           {patterns['item_frequencies']}")
                
                if 'unit_price_types' in patterns:
                    logger.info(f"  Unit price patterns:          {patterns['unit_price_types']} item types")
                
                if result.get('errors'):
                    logger.warning("\n⚠️  Errors encountered:")
                    for error in result['errors']:
                        logger.warning(f"    • {error}")
                
                logger.info("\n" + "=" * 80)
                
            else:
                logger.error(f"❌ Learning failed: {result.get('message', 'Unknown error')}")
                if result.get('errors'):
                    for error in result['errors']:
                        logger.error(f"  • {error}")
                sys.exit(1)
                
        except Exception as e:
            logger.error(f"Learning process failed: {e}")
            sys.exit(1)
    
    def show_patterns(self):
        """Display learned patterns"""
        logger.info("=" * 80)
        logger.info("📈 LEARNED PATTERNS")
        logger.info("=" * 80)
        
        try:
            patterns = self.learning_service.get_patterns()
            
            # Category patterns
            if 'category_patterns' in patterns and patterns['category_patterns']:
                cat_patterns = patterns['category_patterns'].get('patterns', {})
                confidence = patterns['category_patterns'].get('confidence', 0)
                
                logger.info("\n🏗️  CATEGORY DISTRIBUTION")
                logger.info("─" * 80)
                logger.info(f"  Confidence: {confidence:.2f}\n")
                
                for cat_id, data in sorted(cat_patterns.items()):
                    logger.info(f"  Category {cat_id}:")
                    logger.info(f"    Average:        {data['avg_percent']:.1f}%")
                    logger.info(f"    Range:          {data['min_percent']:.1f}% - {data['max_percent']:.1f}%")
                    logger.info(f"    Std deviation:  ±{data['std_dev']:.1f}%")
                    logger.info(f"    Sample size:    {data['sample_size']} offers")
                    logger.info("")
            else:
                logger.warning("\n  No category patterns learned yet")
            
            # Item frequencies
            if 'item_frequencies' in patterns and patterns['item_frequencies']:
                frequencies = patterns['item_frequencies'].get('frequencies', {})
                
                logger.info("\n🔧 ITEM FREQUENCIES (Top 10)")
                logger.info("─" * 80)
                
                # Sort by frequency
                sorted_items = sorted(
                    frequencies.items(),
                    key=lambda x: x[1]['frequency'],
                    reverse=True
                )[:10]
                
                for item_type, data in sorted_items:
                    freq_pct = data['frequency'] * 100
                    logger.info(f"  {item_type:25} {freq_pct:5.1f}%  (€{data['avg_value_eur']:>10,.2f} avg)")
            else:
                logger.warning("\n  No item frequencies learned yet")
            
            # Unit prices
            if 'unit_prices' in patterns and patterns['unit_prices']:
                unit_prices = patterns['unit_prices'].get('unit_prices', {})
                
                if unit_prices:
                    logger.info("\n💰 UNIT PRICE PATTERNS (Top 10)")
                    logger.info("─" * 80)
                    
                    # Sort by sample size
                    sorted_prices = sorted(
                        unit_prices.items(),
                        key=lambda x: x[1]['sample_size'],
                        reverse=True
                    )[:10]
                    
                    for item_type, data in sorted_prices:
                        unit = data.get('common_unit', 'unit')
                        logger.info(f"  {item_type:25} €{data['avg_unit_price_eur']:>8.2f}/{unit}  "
                                  f"(€{data['min_unit_price_eur']:.2f} - €{data['max_unit_price_eur']:.2f})  "
                                  f"n={data['sample_size']}")
                else:
                    logger.warning("\n  No unit price patterns available")
            
            logger.info("\n" + "=" * 80)
            
        except Exception as e:
            logger.error(f"Failed to get patterns: {e}")
            sys.exit(1)
    
    def list_offers(self, limit: int = 20):
        """List ingested offers"""
        logger.info("=" * 80)
        logger.info("📋 INGESTED OFFERS")
        logger.info("=" * 80)
        
        try:
            offers = self.learning_service.list_offers(limit=limit)
            
            if not offers:
                logger.info("\n  No offers ingested yet")
                logger.info("  Use: python scripts/ingest_offers.py --folder <path>")
            else:
                logger.info(f"\nShowing {len(offers)} most recent offer(s):\n")
                
                for idx, offer in enumerate(offers, 1):
                    logger.info(f"{idx}. {offer.get('offer_id')}")
                    logger.info(f"   Project:      {offer.get('project_name', 'Unknown')}")
                    logger.info(f"   Total:        €{offer.get('total_eur', 0):,.2f}")
                    logger.info(f"   Detail level: {offer.get('detail_level', 'summary')}")
                    logger.info(f"   Categories:   {len(offer.get('category_summary', {}))}")
                    
                    ingestion_date = offer.get('ingestion_date')
                    if ingestion_date:
                        date = datetime.fromisoformat(ingestion_date)
                        logger.info(f"   Ingested:     {date.strftime('%Y-%m-%d %H:%M')}")
                    
                    logger.info("")
            
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Failed to list offers: {e}")
            sys.exit(1)
    
    def _get_confidence_label(self, confidence: float) -> str:
        """Get confidence level label"""
        if confidence >= 0.85:
            return "High"
        elif confidence >= 0.7:
            return "Good"
        elif confidence >= 0.5:
            return "Medium"
        else:
            return "Low"


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Monitor offer learning system and view patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View system status (default)
  python scripts/learning_dashboard.py
  
  # Trigger pattern learning
  python scripts/learning_dashboard.py --learn
  
  # View learned patterns
  python scripts/learning_dashboard.py --patterns
  
  # List ingested offers
  python scripts/learning_dashboard.py --offers
  python scripts/learning_dashboard.py --offers --limit 50
        """
    )
    
    # Arguments
    parser.add_argument(
        '--learn',
        action='store_true',
        help='Trigger pattern learning process'
    )
    
    parser.add_argument(
        '--patterns',
        action='store_true',
        help='Display learned patterns'
    )
    
    parser.add_argument(
        '--offers',
        action='store_true',
        help='List ingested offers'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=20,
        help='Limit number of offers to display (default: 20)'
    )
    
    args = parser.parse_args()
    
    # Run dashboard
    cli = LearningDashboardCLI()
    
    try:
        if args.learn:
            cli.trigger_learning()
        elif args.patterns:
            cli.show_patterns()
        elif args.offers:
            cli.list_offers(limit=args.limit)
        else:
            # Default: show status
            cli.show_status()
            
    except KeyboardInterrupt:
        logger.warning("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\n\nFatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()