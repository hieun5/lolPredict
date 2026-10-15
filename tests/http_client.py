"""HTTP transport for local smoke tests and authenticated Vercel previews."""

from pathlib import Path
import re
import shutil
import subprocess
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
VERCEL_VERSION = "64.2.0"


class HttpClient:
    def __init__(self, base_url: str, *, vercel: bool = False):
        self.base_url = base_url.rstrip("/")
        self.vercel = vercel
        if vercel:
            parsed = urlsplit(self.base_url)
            if (parsed.scheme != "https" or not re.fullmatch(r"[a-zA-Z0-9.-]+", parsed.netloc)
                    or parsed.path or parsed.query or parsed.fragment):
                raise ValueError("Use an HTTPS deployment origin without a path, query, or credentials.")
            if not (ROOT / ".vercel/project.json").is_file():
                raise RuntimeError("Link and inspect the intended Vercel project before testing it.")
            self.npx = shutil.which("npx")
            if self.npx is None:
                raise RuntimeError("Node.js/npm is required for authenticated Vercel preview tests.")

    def fetch(self, path: str, body: bytes | None = None) -> tuple[int, bytes]:
        # Paths can come from the built HTML. Limit them to our asset/API paths,
        # including when npx is a Windows batch wrapper.
        if not re.fullmatch(r"/(?:[a-zA-Z0-9._-]+/)*[a-zA-Z0-9._-]*", path):
            raise ValueError("Expected a relative API or static asset path.")
        if self.vercel:
            command = [self.npx, "--yes", f"vercel@{VERCEL_VERSION}", "curl", path,
                       "--deployment", self.base_url, "--",
                       "--silent", "--show-error", "--max-time", "60",
                       "--write-out", r"\n__HTTP_STATUS__:%{http_code}"]
            if body is not None:
                command += ["--request", "POST", "--header", "Content-Type:application/json",
                            "--data-binary", "@-"]
            # CLI 64.2 forwards --non-interactive to system curl, which rejects
            # it. The preflight above requires an existing link; do not pass
            # curl's --yes option, which could create a link implicitly.
            result = subprocess.run(command, input=body, capture_output=True, timeout=90, cwd=ROOT)
            if result.returncode:
                raise RuntimeError("Authenticated Vercel request failed: " + result.stderr.decode(errors="replace"))
            response, separator, code = result.stdout.rpartition(b"\n__HTTP_STATUS__:")
            if not separator or not re.fullmatch(rb"[1-5][0-9]{2}\s*", code):
                raise RuntimeError("Vercel curl did not return an HTTP status; check CLI authentication.")
            return int(code), response
        headers = {"Content-Type": "application/json"} if body is not None else {}
        try:
            with urlopen(Request(self.base_url + path, data=body, headers=headers), timeout=60) as response:
                return response.status, response.read()
        except HTTPError as error:
            return error.code, error.read()
