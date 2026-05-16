import logging
import time
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

logger = logging.getLogger(__name__)


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


def launch_knowledge_graph(
    html_path: Path,
    port: int = 8765,
    auto_close: bool = True,
) -> str:
    html_path = html_path.resolve()

    if not html_path.exists():
        raise FileNotFoundError(f"HTML file not found: {html_path}")

    serve_dir = str(html_path.parent)
    filename = html_path.name

    class Handler(QuietHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=serve_dir, **kwargs)

    server = HTTPServer(("127.0.0.1", port), Handler)

    url = f"http://127.0.0.1:{port}/{filename}"

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    time.sleep(0.3)

    print(f"\nKnowledge graph running at: {url}")
    print("Open this URL in your browser to view the graph.")
    logger.info("Server started at %s", url)

    if auto_close:
        print("Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            server.shutdown()

    return url
