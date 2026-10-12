import os
import shutil
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp
from typing import Any, Dict, Optional
from unittest.mock import Mock, patch

import pytest

from pip._internal.exceptions import HashMismatch
from pip._internal.models.link import Link
from pip._internal.network.download import BatchDownloader, Downloader
from pip._internal.network.session import PipSession
from pip._internal.operations.prepare import (
    RequirementPreparer,
    _check_download_dir,
    unpack_url,
)
from pip._internal.utils.hashes import Hashes
from tests.lib import TestData
from tests.lib.requests_mocks import MockResponse


def test_unpack_url_with_urllib_response_without_content_type(data: TestData) -> None:
    """
    It should download and unpack files even if no Content-Type header exists
    """
    _real_session = PipSession()

    def _fake_session_get(*args: Any, **kwargs: Any) -> Dict[str, str]:
        resp = _real_session.get(*args, **kwargs)
        del resp.headers["Content-Type"]
        return resp

    session = Mock()
    session.get = _fake_session_get
    download = Downloader(session, progress_bar="on")

    uri = data.packages.joinpath("simple-1.0.tar.gz").as_uri()
    link = Link(uri)
    temp_dir = mkdtemp()
    try:
        unpack_url(
            link,
            temp_dir,
            download=download,
            download_dir=None,
            verbosity=0,
        )
        assert set(os.listdir(temp_dir)) == {
            "PKG-INFO",
            "setup.cfg",
            "setup.py",
            "simple",
            "simple.egg-info",
        }
    finally:
        rmtree(temp_dir)


@patch("pip._internal.network.download.raise_for_status")
def test_download_http_url__no_directory_traversal(
    mock_raise_for_status: Mock, tmpdir: Path
) -> None:
    """
    Test that directory traversal doesn't happen on download when the
    Content-Disposition header contains a filename with a ".." path part.
    """
    mock_url = "http://www.example.com/whatever.tgz"
    contents = b"downloaded"
    link = Link(mock_url)

    session = Mock()
    resp = MockResponse(contents)
    resp.url = mock_url
    resp.headers = {
        # Set the content-type to a random value to prevent
        # mimetypes.guess_extension from guessing the extension.
        "content-type": "random",
        "content-disposition": 'attachment;filename="../out_dir_file"',
    }
    session.get.return_value = resp
    download = Downloader(session, progress_bar="on")

    download_dir = os.fspath(tmpdir.joinpath("download"))
    os.mkdir(download_dir)
    file_path, content_type = download(link, download_dir)
    # The file should be downloaded to download_dir.
    actual = os.listdir(download_dir)
    assert actual == ["out_dir_file"]
    mock_raise_for_status.assert_called_once_with(resp)


def _download_with(downloader_cls: type, mock_url: str, download_dir: str) -> str:
    """Download ``mock_url`` into ``download_dir`` with a mocked session and
    return the path the downloader wrote to."""
    link = Link(mock_url)
    session = Mock()
    resp = MockResponse(b"downloaded")
    resp.url = mock_url
    # Set the content-type to a random value to prevent
    # mimetypes.guess_extension from guessing the extension.
    resp.headers = {"content-type": "random"}
    session.get.return_value = resp
    downloader = downloader_cls(session, progress_bar="on")
    if downloader_cls is BatchDownloader:
        [(_, (file_path, _))] = list(downloader([link], download_dir))
    else:
        file_path, _ = downloader(link, download_dir)
    return file_path


@pytest.mark.parametrize("downloader_cls", [Downloader, BatchDownloader])
@pytest.mark.parametrize(
    "mock_url, expected_name",
    [
        # "%252e%252e%252f" decodes once to "%2e%2e%2f" and twice to "../".
        (
            "http://www.example.com/%252e%252e%252fevil.whl",
            "%2e%2e%2fevil.whl",
        ),
        # "%255c" decodes twice to a Windows path separator.
        (
            "http://www.example.com/%252e%252e%255cevil.whl",
            "%2e%2e%5cevil.whl",
        ),
    ],
)
@patch("pip._internal.network.download.raise_for_status")
def test_download_http_url__no_directory_traversal_via_encoded_url(
    mock_raise_for_status: Mock,
    tmpdir: Path,
    downloader_cls: type,
    mock_url: str,
    expected_name: str,
) -> None:
    """
    A doubly percent-encoded "../" in the URL path must not be decoded into a
    path separator that lets the download escape the download directory.
    """
    base_dir = os.fspath(tmpdir.joinpath("base"))
    download_dir = os.path.join(base_dir, "download")
    os.makedirs(download_dir)

    file_path = _download_with(downloader_cls, mock_url, download_dir)

    assert file_path == os.path.join(download_dir, expected_name)
    assert os.listdir(download_dir) == [expected_name]
    # Nothing was written next to (i.e. outside of) the download directory.
    assert os.listdir(base_dir) == ["download"]


@pytest.mark.parametrize("downloader_cls", [Downloader, BatchDownloader])
@patch("pip._internal.network.download.raise_for_status")
def test_download_http_url__no_directory_traversal_via_netloc(
    mock_raise_for_status: Mock, tmpdir: Path, downloader_cls: type
) -> None:
    """
    A path-less URL falls back to the netloc for its file name; a netloc that
    looks like a (Windows) traversal must still be written inside the download
    directory.
    """
    base_dir = os.fspath(tmpdir.joinpath("base", "nested"))
    download_dir = os.path.join(base_dir, "download")
    os.makedirs(download_dir)

    file_path = _download_with(downloader_cls, "http://..\\..\\evil.whl/", download_dir)

    assert os.path.dirname(file_path) == download_dir
    assert os.listdir(download_dir) == [os.path.basename(file_path)]
    assert os.listdir(base_dir) == ["download"]
    assert os.listdir(os.fspath(tmpdir.joinpath("base"))) == ["nested"]


@pytest.mark.parametrize("downloader_cls", [Downloader, BatchDownloader])
@patch("pip._internal.network.download.raise_for_status")
def test_download_http_url__parent_reference_netloc_rejected(
    mock_raise_for_status: Mock, tmpdir: Path, downloader_cls: type
) -> None:
    """
    A URL whose only name is a ".." reference has no usable file name, so the
    download is refused rather than written onto the parent directory.
    """
    base_dir = os.fspath(tmpdir.joinpath("base"))
    download_dir = os.path.join(base_dir, "download")
    os.makedirs(download_dir)

    with pytest.raises(ValueError):
        _download_with(downloader_cls, "http://../", download_dir)

    assert os.listdir(download_dir) == []
    assert os.listdir(base_dir) == ["download"]


@pytest.mark.parametrize("hashes", [None, Hashes({"sha256": ["0" * 64]})])
def test_check_download_dir__no_directory_traversal(
    tmpdir: Path, hashes: Optional[Hashes]
) -> None:
    """
    A doubly percent-encoded "../" in the link must not make the download-dir
    check look at (and, on a hash mismatch, delete) a file outside of the
    download directory.
    """
    base_dir = os.fspath(tmpdir.joinpath("base"))
    download_dir = os.path.join(base_dir, "download")
    os.makedirs(download_dir)
    outside_file = os.path.join(base_dir, "evil.whl")
    with open(outside_file, "wb") as f:
        f.write(b"outside")

    link = Link("https://example.com/%252e%252e%252fevil.whl")
    result = _check_download_dir(
        link, download_dir, hashes, warn_on_hash_mismatch=False
    )

    assert result is None
    assert os.path.isfile(outside_file)


def test_save_linked_requirement__no_directory_traversal(tmpdir: Path) -> None:
    """
    Saving a downloaded requirement into the download directory must not let a
    doubly percent-encoded "../" in the link place the file outside of it.
    """
    base_dir = os.fspath(tmpdir.joinpath("base"))
    download_dir = os.path.join(base_dir, "download")
    os.makedirs(download_dir)
    source_dir = os.fspath(tmpdir.joinpath("source"))
    os.makedirs(source_dir)
    local_file_path = os.path.join(source_dir, "pkg-1.0-py3-none-any.whl")
    with open(local_file_path, "wb") as f:
        f.write(b"downloaded")

    req = Mock(
        link=Link("https://example.com/%252e%252e%252fevil.whl"),
        local_file_path=local_file_path,
        editable=False,
    )
    preparer = Mock(download_dir=download_dir)
    RequirementPreparer.save_linked_requirement(preparer, req)

    assert os.listdir(download_dir) == ["%2e%2e%2fevil.whl"]
    assert os.listdir(base_dir) == ["download"]


@pytest.fixture
def clean_project(tmpdir_factory: pytest.TempPathFactory, data: TestData) -> Path:
    tmpdir = tmpdir_factory.mktemp("clean_project")
    new_project_dir = tmpdir.joinpath("FSPkg")
    path = data.packages.joinpath("FSPkg")
    shutil.copytree(path, new_project_dir)
    return new_project_dir


class Test_unpack_url:
    def prep(self, tmpdir: Path, data: TestData) -> None:
        self.build_dir = os.fspath(tmpdir.joinpath("build"))
        self.download_dir = tmpdir.joinpath("download")
        os.mkdir(self.build_dir)
        os.mkdir(self.download_dir)
        self.dist_file = "simple-1.0.tar.gz"
        self.dist_file2 = "simple-2.0.tar.gz"
        self.dist_path = data.packages.joinpath(self.dist_file)
        self.dist_path2 = data.packages.joinpath(self.dist_file2)
        self.dist_url = Link(self.dist_path.as_uri())
        self.dist_url2 = Link(self.dist_path2.as_uri())
        self.no_download = Mock(side_effect=AssertionError)

    def test_unpack_url_no_download(self, tmpdir: Path, data: TestData) -> None:
        self.prep(tmpdir, data)
        unpack_url(self.dist_url, self.build_dir, self.no_download, verbosity=0)
        assert os.path.isdir(os.path.join(self.build_dir, "simple"))
        assert not os.path.isfile(os.path.join(self.download_dir, self.dist_file))

    def test_unpack_url_bad_hash(self, tmpdir: Path, data: TestData) -> None:
        """
        Test when the file url hash fragment is wrong
        """
        self.prep(tmpdir, data)
        url = f"{self.dist_url.url}#md5=bogus"
        dist_url = Link(url)
        with pytest.raises(HashMismatch):
            unpack_url(
                dist_url,
                self.build_dir,
                download=self.no_download,
                hashes=Hashes({"md5": ["bogus"]}),
                verbosity=0,
            )
