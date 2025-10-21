#!/usr/bin/env python3
"""
Bulk Offer Ingestion Script
CLI tool for ingesting multiple historical offer files

Usage:
    python scripts/ingest_offers.py --folder ./historical_offers/2024/
    python scripts/ingest_offers.py --file ./offer.csv
    python scripts/ingest_offers.py --folder ./offers/ --dry-run
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.offer_learning_service import OfferLearningService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class OfferIngestionCLI:
    """CLI tool for bulk offer ingestion"""
    
    # Supported file extensions
    SUPPORTED_EXTENSIONS = {'.csv', '.xlsx', '.xls'}
    
    def __init__(self, dry_run: bool = False):
        """
        Initialize ingestion CLI
        
        Args:
            dry_run: If True, validate files but don't upload
        """
        self.dry_run = dry_run
        self.learning_service = OfferLearningService()
        
        self.stats = {
            'files_found': 0,
            'files_processed': 0,
            'files_succeeded': 0,
            'files_failed': 0,
            'total_value_eur': 0.0,
            'errors': []
        }
    
    def find_offer_files(self, path: str) -> List[Path]:
        """
        Find all offer files in directory or single file
        
        Args:
            path: Directory or file path
            
        Returns:
            List of file paths
        """
        path_obj = Path(path)
        files = []
        
        if not path_obj.exists():
            logger.error(f"Path does not exist: {path}")
            return files
        
        if path_obj.is_file():
            # Single file
            if path_obj.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                files.append(path_obj)
            else:
                logger.warning(f"Unsupported file type: {path_obj.suffix}")
        
        elif path_obj.is_dir():
            # Directory - recursively find all supported files
            for ext in self.SUPPORTED_EXTENSIONS:
                files.extend(path_obj.rglob(f"*{ext}"))
        
        self.stats['files_found'] = len(files)
        return sorted(files)
    
    def ingest_file(self, file_path: Path) -> Dict:
        """
        Ingest a single offer file
        
        Args:
            file_path: Path to offer file
            
        Returns:
            Ingestion result dict
        """
        logger.info(f"Processing: {file_path.name}")
        
        try:
            # Read file
            with open(file_path, 'rb') as f:
                content = f.read()
            
            if self.dry_run:
                logger.info(f"  [DRY RUN] Would ingest: {file_path.name} ({len(content)} bytes)")
                return {
                    'success': True,
                    'dry_run': True,
                    'filename': file_path.name,
                    'size_bytes': len(content)
                }
            
            # Ingest offer
            result = self.learning_service.ingest_offer(
                file_content=content,
                filename=file_path.name,
                metadata={
                    'source_path': str(file_path),
                    'ingested_via': 'bulk_ingestion_script'
                }
            )
            
            if result.success:
                logger.info(f"  ✓ Success: {result.offer_id}")
                logger.info(f"    - Project: {result.message}")
                logger.info(f"    - Categories: {result.categories_found}")
                logger.info(f"    - Items: {result.items_extracted}")
                logger.info(f"    - Total: €{result.total_eur:,.2f}")
                if result.items_with_unit_pricing > 0:
                    logger.info(f"    - Unit prices: {result.items_with_unit_pricing} items")
                if result.warnings:
                    logger.warning(f"    - Warnings: {len(result.warnings)}")
                    for warning in result.warnings:
                        logger.warning(f"      • {warning}")
            else:
                logger.error(f"  ✗ Failed: {result.message}")
                if result.errors:
                    for error in result.errors:
                        logger.error(f"      • {error}")
            
            return result.to_dict()
            
        except Exception as e:
            logger.error(f"  ✗ Error processing {file_path.name}: {e}")
            return {
                'success': False,
                'filename': file_path.name,
                'error': str(e)
            }
    
    def run_bulk_ingestion(self, path: str) -> Dict:
        """
        Run bulk ingestion on path
        
        Args:
            path: Directory or file path
            
        Returns:
            Summary statistics
        """
        logger.info("=" * 80)
        logger.info("OFFER BULK INGESTION")
        logger.info("=" * 80)
        
        if self.dry_run:
            logger.warning("DRY RUN MODE - No files will be uploaded")
        
        # Find files
        logger.info(f"\nScanning: {path}")
        files = self.find_offer_files(path)
        
        if not files:
            logger.warning("No offer files found!")
            return self.stats
        
        logger.info(f"Found {len(files)} offer file(s)\n")
        
        # Process each file
        for idx, file_path in enumerate(files, 1):
            logger.info(f"[{idx}/{len(files)}] {file_path.name}")
            
            result = self.ingest_file(file_path)
            
            self.stats['files_processed'] += 1
            
            if result.get('success'):
                self.stats['files_succeeded'] += 1
                if not self.dry_run:
                    self.stats['total_value_eur'] += result.get('statistics', {}).get('total_eur', 0)
            else:
                self.stats['files_failed'] += 1
                self.stats['errors'].append({
                    'file': file_path.name,
                    'error': result.get('error', 'Unknown error')
                })
            
            logger.info("")  # Blank line
        
        # Print summary
        self._print_summary()
        
        return self.stats
    
    def _print_summary(self):
        """Print ingestion summary"""
        logger.info("=" * 80)
        logger.info("INGESTION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Files found:     {self.stats['files_found']}")
        logger.info(f"Files processed: {self.stats['files_processed']}")
        logger.info(f"Success:         {self.stats['files_succeeded']}")
        logger.info(f"Failed:          {self.stats['files_failed']}")
        
        if not self.dry_run and self.stats['total_value_eur'] > 0:
            logger.info(f"Total value:     €{self.stats['total_value_eur']:,.2f}")
        
        if self.stats['errors']:
            logger.error("\nErrors:")
            for error in self.stats['errors']:
                logger.error(f"  • {error['file']}: {error['error']}")
        
        logger.info("=" * 80)
        
        if not self.dry_run and self.stats['files_succeeded'] > 0:
            logger.info("\n💡 Next step: Run pattern learning")
            logger.info("   POST /api/ml/offers/learn")
            logger.info("   or use: python scripts/learning_dashboard.py --learn")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Bulk ingest historical offer files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest all files in a folder
  python scripts/ingest_offers.py --folder ./historical_offers/2024/
  
  # Ingest a single file
  python scripts/ingest_offers.py --file ./imperial_brands_rev06.csv
  
  # Dry run (validate only, don't upload)
  python scripts/ingest_offers.py --folder ./offers/ --dry-run
  
  # Supported formats:
  - CSV (Imperial, Beautik)
  - Excel (.xlsx, .xls) (CCC format)
        """
    )
    
    # Arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--folder',
        type=str,
        help='Path to folder containing offer files (recursive)'
    )
    group.add_argument(
        '--file',
        type=str,
        help='Path to single offer file'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate files without uploading'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Get path
    path = args.folder or args.file
    
    # Run ingestion
    cli = OfferIngestionCLI(dry_run=args.dry_run)
    
    try:
        stats = cli.run_bulk_ingestion(path)
        
        # Exit code based on results
        if stats['files_failed'] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.warning("\n\nIngestion interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\n\nFatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()