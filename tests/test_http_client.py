"""Checks for authenticated preview test transport, including error responses."""

from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from http_client import HttpClient


class PreviewTransportTests(unittest.TestCase):
    def client(self):
        with patch.object(Path, "is_file", return_value=True), patch("http_client.shutil.which", return_value="npx"):
            return HttpClient("https://example.vercel.app", vercel=True)

    def test_post_payload_uses_stdin_and_exact_deployment(self):
        client = self.client()
        result = subprocess.CompletedProcess([], 0, b'{"ok":true}\n__HTTP_STATUS__:200', b"")
        body = b'{"league":"LCK"}'
        with patch("http_client.subprocess.run", return_value=result) as run:
            self.assertEqual(client.fetch("/api/predict", body), (200, b'{"ok":true}'))
        command = run.call_args.args[0]
        self.assertIn(client.base_url, command)
        self.assertIn("@-", command)
        self.assertNotIn(body.decode(), command)
        self.assertEqual(run.call_args.kwargs["input"], body)

    def test_error_status_and_body_are_preserved(self):
        client = self.client()
        result = subprocess.CompletedProcess([], 0, b'{"detail":"invalid"}\n__HTTP_STATUS__:422', b"")
        with patch("http_client.subprocess.run", return_value=result):
            self.assertEqual(client.fetch("/api/predict"), (422, b'{"detail":"invalid"}'))

    def test_auth_failure_is_not_reported_as_application_success(self):
        client = self.client()
        for result in (subprocess.CompletedProcess([], 1, b"", b"Login required"),
                       subprocess.CompletedProcess([], 0, b"Login required", b"")):
            with self.subTest(result=result), patch("http_client.subprocess.run", return_value=result):
                with self.assertRaises(RuntimeError):
                    client.fetch("/api/health")

    def test_unlinked_directory_and_unsafe_paths_are_rejected(self):
        with patch.object(Path, "is_file", return_value=False), self.assertRaises(RuntimeError):
            HttpClient("https://example.vercel.app", vercel=True)
        client = self.client()
        for path in ("//other.example/", "/api/health&whoami", "/api/health\ncommand", "https://other.example"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                client.fetch(path)


if __name__ == "__main__":
    unittest.main()
