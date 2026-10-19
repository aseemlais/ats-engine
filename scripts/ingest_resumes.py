import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import config
from app.workers.ingestion_pipeline import IngestionPipeline

def main():
    parser = argparse.ArgumentParser(description="Ingest resumes into ATS vector & lexical indices")
    parser.add_argument("--dir", type=str, default=str(config.resumes_dir), help="Directory of resume files")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for embeddings")
    parser.add_argument("--reset", action="store_true", help="Reset checkpoint and re-ingest all")
    args = parser.parse_args()

    pipeline = IngestionPipeline()
    print(f"Starting ingestion from: {args.dir} (batch size: {args.batch_size}, reset: {args.reset})")
    stats = pipeline.ingest_directory(directory=args.dir, batch_size=args.batch_size, reset_checkpoint=args.reset)

    print("\n" + "=" * 50)
    print("INGESTION SUMMARY:")
    print(f"Total files detected:   {stats['total_files']}")
    print(f"Newly parsed resumes:   {stats['newly_parsed']}")
    print(f"Total indexed in DB:    {stats['total_indexed']}")
    print(f"Duplicates skipped:     {stats['duplicates']}")
    print(f"Embedding time:         {stats['embedding_time']:.2f}s")
    if stats['errors']:
        print(f"Errors ({len(stats['errors'])}):")
        for err in stats['errors'][:5]:
            print(f"  - {err['file']}: {err['error']}")
    print("=" * 50)

if __name__ == "__main__":
    main()
