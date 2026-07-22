from __future__ import annotations

import argparse
import json
from pathlib import Path

from cleanbot.api.container import get_container


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cleanbot", description="CleanBot Agent maintenance commands")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init-db", help="Create tables and seed deterministic demo users/records")
    ingest = subparsers.add_parser("ingest", help="Index one document or the configured data directory")
    ingest.add_argument("path", nargs="?", help="Optional .txt/.pdf path; defaults to data directory")
    subparsers.add_parser("health", help="Print local database/vector-store counts")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    container = get_container()
    container.database.create_schema()

    if args.command == "init-db":
        users, records = container.database.seed_demo_data()
        print(json.dumps({"users_created": users, "records_upserted": records}, ensure_ascii=False))
        return
    if args.command == "ingest":
        container.database.seed_demo_data()
        if args.path:
            result = container.knowledge_base.ingest_path(Path(args.path))
            results = [result]
        else:
            results = container.knowledge_base.ingest_directory()
        container.retriever.invalidate()
        print(json.dumps([item.model_dump() for item in results], ensure_ascii=False, indent=2))
        return
    if args.command == "health":
        print(
            json.dumps(
                {
                    "users": len(container.database.list_users()),
                    "documents": container.database.knowledge_document_count(),
                    "chunks": container.knowledge_base.count(),
                    "vector_dir": str(container.settings.vector_dir),
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
