#!/usr/bin/env python3
"""
Migration script to convert existing notes and chat history to the new RAG system.

This script can be run standalone or imported as a module to migrate data from
the old database format to the new fast RAG system with sqlite-vec.
"""

import sys
import logging
from pathlib import Path
from typing import Optional
import argparse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag import RagDB, BatchIndexer
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_existing_databases() -> dict:
    """Find existing notes and chat databases."""
    databases = {}
    
    # Look for notes database
    config = Config.get_instance()
    notes_path = config.notes_dir / "notes.db"
    if notes_path.exists():
        databases["notes"] = notes_path
        logger.info(f"Found notes database: {notes_path}")
    
    # Look for chat history database
    chat_path = Path.home() / ".local" / "share" / "perplexity" / "chat_history" / "chat_history.db"
    if chat_path.exists():
        databases["chat_history"] = chat_path
        logger.info(f"Found chat history database: {chat_path}")
    
    return databases


def analyze_existing_data(databases: dict) -> dict:
    """Analyze existing data to provide migration statistics."""
    import sqlite3
    
    analysis = {}
    
    for db_type, db_path in databases.items():
        try:
            with sqlite3.connect(db_path) as conn:
                if db_type == "notes":
                    cursor = conn.execute("SELECT COUNT(*) FROM notes")
                    count = cursor.fetchone()[0]
                    
                    # Get sample data
                    cursor = conn.execute("SELECT title, length(content) FROM notes LIMIT 5")
                    samples = cursor.fetchall()
                    
                    analysis[db_type] = {
                        "count": count,
                        "samples": samples,
                        "avg_content_length": 0
                    }
                    
                    if count > 0:
                        cursor = conn.execute("SELECT AVG(length(content)) FROM notes")
                        avg_length = cursor.fetchone()[0]
                        analysis[db_type]["avg_content_length"] = int(avg_length or 0)
                
                elif db_type == "chat_history":
                    cursor = conn.execute("SELECT COUNT(*) FROM messages")
                    total_messages = cursor.fetchone()[0]
                    
                    cursor = conn.execute("SELECT COUNT(DISTINCT conversation_id) FROM messages")
                    total_conversations = cursor.fetchone()[0]
                    
                    cursor = conn.execute("SELECT role, COUNT(*) FROM messages GROUP BY role")
                    role_counts = dict(cursor.fetchall())
                    
                    analysis[db_type] = {
                        "total_messages": total_messages,
                        "total_conversations": total_conversations,
                        "role_counts": role_counts
                    }
                    
                    if total_messages > 0:
                        cursor = conn.execute("SELECT AVG(length(content)) FROM messages")
                        avg_length = cursor.fetchone()[0]
                        analysis[db_type]["avg_content_length"] = int(avg_length or 0)
                
        except Exception as e:
            logger.error(f"Failed to analyze {db_type} database: {e}")
            analysis[db_type] = {"error": str(e)}
    
    return analysis


def create_rag_database() -> RagDB:
    """Create and initialize the new RAG database."""
    rag_db_path = Path.home() / ".local" / "share" / "perplexity" / "rag" / "rag.db"
    
    logger.info(f"Creating RAG database at: {rag_db_path}")
    rag_db = RagDB(rag_db_path)
    
    return rag_db


def migrate_data(
    databases: dict,
    clear_existing: bool = False,
    include_user_messages: bool = True,
    include_assistant_messages: bool = True,
    dry_run: bool = False
) -> dict:
    """
    Migrate data from existing databases to RAG system.
    
    Args:
        databases: Dictionary of database type -> path
        clear_existing: Clear existing RAG data before migration
        include_user_messages: Include user messages from chat history
        include_assistant_messages: Include assistant messages from chat history
        dry_run: Only show what would be migrated without actually doing it
        
    Returns:
        Migration results dictionary
    """
    if dry_run:
        logger.info("DRY RUN MODE - No actual migration will be performed")
        
        # Create temporary RAG DB for estimation
        temp_rag_db = create_rag_database()
        indexer = BatchIndexer(temp_rag_db, show_progress=False)
        
        estimates = indexer.estimate_migration_time(
            notes_db_path=databases.get("notes"),
            chat_db_path=databases.get("chat_history")
        )
        
        logger.info("Migration time estimates:")
        for source_type, data in estimates.items():
            if source_type != "total":
                logger.info(f"  {source_type}: {data.get('count', 0)} items, "
                          f"~{data.get('estimated_time_formatted', 'unknown')} time")
        
        if "total" in estimates:
            logger.info(f"  Total estimated time: {estimates['total']['estimated_time_formatted']}")
        
        return {"dry_run": True, "estimates": estimates}
    
    # Actual migration
    rag_db = create_rag_database()
    indexer = BatchIndexer(rag_db, show_progress=True)
    
    results = {
        "total_migrated": 0,
        "total_failed": 0,
        "by_source": {}
    }
    
    # Migrate notes
    if "notes" in databases:
        logger.info("Migrating notes...")
        migrated, failed = indexer.migrate_notes_database(
            databases["notes"], 
            clear_existing=clear_existing
        )
        
        results["by_source"]["notes"] = {
            "migrated": migrated,
            "failed": failed
        }
        results["total_migrated"] += migrated
        results["total_failed"] += failed
        
        clear_existing = False  # Only clear once
    
    # Migrate chat history
    if "chat_history" in databases:
        logger.info("Migrating chat history...")
        migrated, failed = indexer.migrate_chat_history_database(
            databases["chat_history"],
            clear_existing=clear_existing,
            include_user_messages=include_user_messages,
            include_assistant_messages=include_assistant_messages
        )
        
        results["by_source"]["chat_history"] = {
            "migrated": migrated,
            "failed": failed
        }
        results["total_migrated"] += migrated
        results["total_failed"] += failed
    
    return results


def print_migration_summary(analysis: dict, results: dict):
    """Print a summary of the migration results."""
    print("\n" + "="*60)
    print("🚀 MIGRATION SUMMARY")
    print("="*60)
    
    if results.get("dry_run"):
        print("DRY RUN - No data was actually migrated")
        return
    
    print(f"✅ Total items migrated: {results['total_migrated']}")
    print(f"❌ Total items failed: {results['total_failed']}")
    
    for source, source_results in results["by_source"].items():
        print(f"\n📊 {source.upper()}:")
        print(f"   Migrated: {source_results['migrated']}")
        print(f"   Failed: {source_results['failed']}")
        
        # Show original vs migrated counts
        if source in analysis:
            original_count = analysis[source].get("count") or analysis[source].get("total_messages", 0)
            success_rate = (source_results['migrated'] / original_count * 100) if original_count > 0 else 0
            print(f"   Success rate: {success_rate:.1f}%")
    
    if results['total_migrated'] > 0:
        print(f"\n🎉 Migration complete! You can now search your content with:")
        print(f"   perplexity rag 'your search query'")
        print(f"   perplexity rag-stats  # to see database statistics")


def main():
    """Main migration script entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate existing notes and chat history to the new RAG system"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show migration plan without actually migrating data"
    )
    
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing RAG data before migration"
    )
    
    parser.add_argument(
        "--no-user-messages",
        action="store_true",
        help="Don't migrate user messages from chat history"
    )
    
    parser.add_argument(
        "--no-assistant-messages",
        action="store_true",
        help="Don't migrate assistant messages from chat history"
    )
    
    parser.add_argument(
        "--notes-only",
        action="store_true",
        help="Migrate only notes, skip chat history"
    )
    
    parser.add_argument(
        "--chat-only",
        action="store_true",
        help="Migrate only chat history, skip notes"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Find existing databases
        print("🔍 Scanning for existing databases...")
        databases = find_existing_databases()
        
        if not databases:
            print("❌ No existing databases found to migrate.")
            print("   Make sure you have notes or chat history to migrate.")
            return 1
        
        # Apply source filters
        if args.notes_only:
            databases = {k: v for k, v in databases.items() if k == "notes"}
        elif args.chat_only:
            databases = {k: v for k, v in databases.items() if k == "chat_history"}
        
        if not databases:
            print("❌ No databases found matching the specified filters.")
            return 1
        
        # Analyze existing data
        print("\n📊 Analyzing existing data...")
        analysis = analyze_existing_data(databases)
        
        for db_type, data in analysis.items():
            if "error" in data:
                print(f"❌ {db_type}: Error - {data['error']}")
                continue
                
            if db_type == "notes":
                print(f"📝 Notes: {data['count']} items, avg length: {data['avg_content_length']} chars")
            elif db_type == "chat_history":
                print(f"💬 Chat History: {data['total_messages']} messages in {data['total_conversations']} conversations")
                for role, count in data.get("role_counts", {}).items():
                    print(f"   {role}: {count} messages")
        
        # Confirm migration (unless dry run)
        if not args.dry_run:
            print(f"\n⚠️  This will migrate your data to the new RAG system.")
            if args.clear:
                print("⚠️  Existing RAG data will be cleared first.")
            
            confirm = input("\nContinue? (y/N): ").strip().lower()
            if confirm not in ["y", "yes"]:
                print("Migration cancelled.")
                return 0
        
        # Perform migration
        print(f"\n🚀 Starting migration...")
        results = migrate_data(
            databases=databases,
            clear_existing=args.clear,
            include_user_messages=not args.no_user_messages,
            include_assistant_messages=not args.no_assistant_messages,
            dry_run=args.dry_run
        )
        
        # Print summary
        print_migration_summary(analysis, results)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n❌ Migration cancelled by user.")
        return 1
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())