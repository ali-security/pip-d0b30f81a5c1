# test the error message returned by pip when
# a bad "file:" URL is passed to it.

import urllib.parse
from typing import Any

import pytest


@pytest.mark.skipif(
    urllib.parse.urlunsplit(urllib.parse.urlsplit("file:unexistent_file"))
    == "file:unexistent_file",
    reason=(
        "Newer CPython patch releases (3.12.x+) no longer add '//' to "
        "netloc-less file: URLs in urlunsplit(), so pip 24.0 reports "
        "'No connection adapters were found for file:unexistent_file' "
        "instead of the LocalFSAdapter 404 this test expects"
    ),
)
def test_filenotfound_error_message(script: Any) -> None:
    # Test the error message returned when using a bad 'file:' URL.
    # make pip to fail and get an error message
    # by running "pip install -r file:nonexistent_file"
    proc = script.pip("install", "-r", "file:unexistent_file", expect_error=True)
    assert proc.returncode == 1
    expect = (
        "ERROR: 404 Client Error: FileNotFoundError for url: file:///unexistent_file"
    )
    assert proc.stderr.rstrip() == expect
