"""Exercise real local inference with external networking blocked.

Run explicitly after preparing the small BGE and MiniLM caches. Uses temporary
configuration, notes and sources; it does not modify personal CLI data.
"""

import argparse
import errno
import json
import socket
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
from typer.testing import CliRunner

from pplx_cli import config
from pplx_cli.cli import app
from pplx_cli.local import local_only_mode
from pplx_cli.notes import NotesDB
from pplx_cli.rag import embeddings
from pplx_cli.wiki import WikiWorkspace


def prove_network_blocked():
    try:
        connection = socket.create_connection(("1.1.1.1", 443), timeout=2)
    except OSError as error:
        if error.errno not in (errno.EPERM, errno.EACCES):
            raise RuntimeError("Could not establish that external networking is denied") from error
    else:
        connection.close()
        raise RuntimeError("External networking is available; run under a loopback-only test profile")


def text_pdf(path):
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                             NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 72 720 Td (The backup retention period is seven days.) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    with path.open("wb") as target:
        writer.write(target)


class Article(BaseHTTPRequestHandler):
    def do_GET(self):
        body = ("<html><head><title>Source snapshots</title></head><body><article><h1>Source snapshots</h1>"
                + "<p>Captured source snapshots remain local so readers can inspect the original research evidence. "
                  "The workspace keeps citations, source locations and captured content together.</p>" * 8
                + "</article></body></html>").encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11435")
    parser.add_argument("--model", default="gemma3:4b")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    prove_network_blocked()
    results = {"model": args.model, "external_network": "denied", "commands": []}
    with tempfile.TemporaryDirectory(prefix="pplx-local-verification-") as temporary:
        root = Path(temporary)
        config.CONFIG_DIR = root / "config"
        config.CONFIG_FILE = config.CONFIG_DIR / "config.json"
        config.CONFIG_DIR.mkdir()
        embeddings.RAG_CONFIG_PATH = config.CONFIG_DIR / "rag_config.json"
        embeddings.RAG_CONFIG_PATH.write_text(json.dumps({"embedding_provider": "local", "model": "small", "device": "cpu"}))
        config.Config._instance = None
        embeddings.reset_embedding_model()
        config.save_ollama_settings(args.endpoint, args.model, timeout=180, context=8192)
        research = root / "research"
        research.mkdir()
        (research / "architecture.md").write_text("# Storage\nThe service uses SQLite because it runs on one machine and needs no separate database server.\n")
        (research / "network.txt").write_text("Local queries never download models or contact cloud services.\n")
        text_pdf(research / "retention.pdf")
        sources = {path: path.read_bytes() for path in research.iterdir()}
        with local_only_mode():
            workspace = WikiWorkspace.initialize(research)
            outcome = workspace.sync()
            assert outcome["failed"] == 0, outcome
            notes = NotesDB(root / "notes")
            notes.add_note("SQLite decision", "We selected SQLite because the application runs on one machine and does not need a separate database server.")
        # Capture an explicit fixture page, then remove its HTTP server. Subsequent
        # questions and compilation can use only the stored web snapshot.
        server = ThreadingHTTPServer(("127.0.0.1", 0), Article)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with local_only_mode():
                captured = workspace.ingest(f"http://127.0.0.1:{server.server_port}/article")
            assert all(item.status != "error" for item in captured), captured
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        overview = workspace.wiki_dir / "overview.md"
        manual = "Manual verification note must remain unchanged."
        overview.write_text(workspace.GENERATED_START + "\nold\n" + workspace.GENERATED_END + "\n\n## Manual notes\n" + manual + "\n")
        runner = CliRunner()
        checks = [
            (["ask", "Reply exactly LOCAL_OK", "--no-save-history"], "LOCAL_OK"),
            (["ask-notes", "Why did we select SQLite?", "--dir", str(notes.directory)], "[N1]"),
            (["wiki", "query", "Why did we select SQLite?", "--dir", str(research)], "[L"),
            (["wiki", "compile", "--dir", str(research)], "Generated"),
        ]
        for arguments, expected in checks:
            result = runner.invoke(app, arguments)
            if result.exit_code != 0 or expected not in result.output:
                raise RuntimeError(f"{' '.join(arguments[:2])} failed: {result.output}") from result.exception
            label = " ".join(arguments[:2]) if arguments[0] == "wiki" else arguments[0]
            results["commands"].append(label)
            print(f"PASS {label}", flush=True)
        assert manual in overview.read_text()
        assert all(path.read_bytes() == content for path, content in sources.items())
        assert workspace.lint()["errors"] == []
        results["source_types"] = sorted(workspace.status()["sources_by_type"].keys())
        results["manual_notes"] = "preserved"
        results["source_files"] = "unchanged"
    if args.report:
        args.report.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
