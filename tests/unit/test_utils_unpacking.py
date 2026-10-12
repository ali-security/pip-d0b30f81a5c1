from __future__ import annotations

import io
import os
import shutil
import stat
import sys
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path
from typing import List, Tuple
from unittest.mock import MagicMock, patch

import pytest
from _pytest.monkeypatch import MonkeyPatch

from pip._internal.exceptions import InstallationError
from pip._internal.utils.unpacking import (
    is_within_directory,
    unpack_file,
    untar_file,
    unzip_file,
)
from tests.lib import TestData


class TestUnpackArchives:
    """
    test_tar.tgz/test_tar.zip have content as follows engineered to confirm 3
    things:
     1) confirm that reg files, dirs, and symlinks get unpacked
     2) permissions are not preserved (and go by the 022 umask)
     3) reg files with *any* execute perms, get chmod +x

       file.txt         600 regular file
       symlink.txt      777 symlink to file.txt
       script_owner.sh  700 script where owner can execute
       script_group.sh  610 script where group can execute
       script_world.sh  601 script where world can execute
       dir              744 directory
       dir/dirfile      622 regular file
     4) the file contents are extracted correctly (though the content of
        each file isn't currently unique)

    """

    def setup_method(self) -> None:
        self.tempdir = tempfile.mkdtemp()
        self.old_mask = os.umask(0o022)
        self.symlink_expected_mode = None

    def teardown_method(self) -> None:
        os.umask(self.old_mask)
        shutil.rmtree(self.tempdir, ignore_errors=True)

    def mode(self, path: str) -> int:
        return stat.S_IMODE(os.stat(path).st_mode)

    def confirm_files(self) -> None:
        # expectations based on 022 umask set above and the unpack logic that
        # sets execute permissions, not preservation
        for fname, expected_mode, test, expected_contents in [
            ("file.txt", 0o644, os.path.isfile, b"file\n"),
            # We don't test the "symlink.txt" contents for now.
            ("symlink.txt", 0o644, os.path.isfile, None),
            ("script_owner.sh", 0o755, os.path.isfile, b"file\n"),
            ("script_group.sh", 0o755, os.path.isfile, b"file\n"),
            ("script_world.sh", 0o755, os.path.isfile, b"file\n"),
            ("dir", 0o755, os.path.isdir, None),
            (os.path.join("dir", "dirfile"), 0o644, os.path.isfile, b""),
        ]:
            path = os.path.join(self.tempdir, fname)
            if path.endswith("symlink.txt") and sys.platform == "win32":
                # no symlinks created on windows
                continue
            assert test(path), path
            if expected_contents is not None:
                with open(path, mode="rb") as f:
                    contents = f.read()
                assert contents == expected_contents, f"fname: {fname}"
            if sys.platform == "win32":
                # the permissions tests below don't apply in windows
                # due to os.chmod being a noop
                continue
            mode = self.mode(path)
            assert (
                mode == expected_mode
            ), f"mode: {mode}, expected mode: {expected_mode}"

    def make_zip_file(self, filename: str, file_list: List[str]) -> str:
        """
        Create a zip file for test case
        """
        test_zip = os.path.join(self.tempdir, filename)
        with zipfile.ZipFile(test_zip, "w") as myzip:
            for item in file_list:
                myzip.writestr(item, "file content")
        return test_zip

    def make_tar_file(self, filename: str, file_list: List[str]) -> str:
        """
        Create a tar file for test case
        """
        test_tar = os.path.join(self.tempdir, filename)
        with tarfile.open(test_tar, "w") as mytar:
            for item in file_list:
                file_tarinfo = tarfile.TarInfo(item)
                mytar.addfile(file_tarinfo, io.BytesIO(b"file content"))
        return test_tar

    def test_unpack_tgz(self, data: TestData) -> None:
        """
        Test unpacking a *.tgz, and setting execute permissions
        """
        test_file = data.packages.joinpath("test_tar.tgz")
        untar_file(os.fspath(test_file), self.tempdir)
        self.confirm_files()
        # Check the timestamp of an extracted file
        file_txt_path = os.path.join(self.tempdir, "file.txt")
        mtime = time.gmtime(os.stat(file_txt_path).st_mtime)
        assert mtime[0:6] == (2013, 8, 16, 5, 13, 37), mtime

    def test_unpack_zip(self, data: TestData) -> None:
        """
        Test unpacking a *.zip, and setting execute permissions
        """
        test_file = data.packages.joinpath("test_zip.zip")
        unzip_file(os.fspath(test_file), self.tempdir)
        self.confirm_files()

    def test_unpack_zip_failure(self) -> None:
        """
        Test unpacking a *.zip with file containing .. path
        and expect exception
        """
        files = ["regular_file.txt", os.path.join("..", "outside_file.txt")]
        test_zip = self.make_zip_file("test_zip.zip", files)
        with pytest.raises(InstallationError) as e:
            unzip_file(test_zip, self.tempdir)
        assert "trying to install outside target directory" in str(e.value)

    def test_unpack_zip_success(self) -> None:
        """
        Test unpacking a *.zip with regular files,
        no file will be installed outside target directory after unpack
        so no exception raised
        """
        files = [
            "regular_file1.txt",
            os.path.join("dir", "dir_file1.txt"),
            os.path.join("dir", "..", "dir_file2.txt"),
        ]
        test_zip = self.make_zip_file("test_zip.zip", files)
        unzip_file(test_zip, self.tempdir)

    def test_unpack_tar_failure(self) -> None:
        """
        Test unpacking a *.tar with file containing .. path
        and expect exception
        """
        files = ["regular_file.txt", os.path.join("..", "outside_file.txt")]
        test_tar = self.make_tar_file("test_tar.tar", files)
        with pytest.raises(InstallationError) as e:
            untar_file(test_tar, self.tempdir)
        assert "trying to install outside target directory" in str(e.value)

    def test_unpack_tar_success(self) -> None:
        """
        Test unpacking a *.tar with regular files,
        no file will be installed outside target directory after unpack
        so no exception raised
        """
        files = [
            "regular_file1.txt",
            os.path.join("dir", "dir_file1.txt"),
            os.path.join("dir", "..", "dir_file2.txt"),
        ]
        test_tar = self.make_tar_file("test_tar.tar", files)
        untar_file(test_tar, self.tempdir)

    def test_unpack_normal_tar_link1_no_data_filter(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        """
        Test unpacking a normal tar with file containing soft links, but no data_filter
        """
        if hasattr(tarfile, "data_filter"):
            monkeypatch.delattr("tarfile.data_filter")

        tar_filename = "test_tar_links_no_data_filter.tar"
        tar_filepath = os.path.join(self.tempdir, tar_filename)

        extract_path = os.path.join(self.tempdir, "extract_path")

        with tarfile.open(tar_filepath, "w") as tar:
            file_data = io.BytesIO(b"normal\n")
            normal_file_tarinfo = tarfile.TarInfo(name="normal_file")
            normal_file_tarinfo.size = len(file_data.getbuffer())
            tar.addfile(normal_file_tarinfo, fileobj=file_data)

            info = tarfile.TarInfo("normal_symlink")
            info.type = tarfile.SYMTYPE
            info.linkpath = "normal_file"
            tar.addfile(info)

        untar_file(tar_filepath, extract_path)

        assert os.path.islink(os.path.join(extract_path, "normal_symlink"))

        link_path = os.readlink(os.path.join(extract_path, "normal_symlink"))
        assert link_path == "normal_file"

        with open(os.path.join(extract_path, "normal_symlink"), "rb") as f:
            assert f.read() == b"normal\n"

    def test_unpack_normal_tar_link2_no_data_filter(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        """
        Test unpacking a normal tar with file containing soft links, but no data_filter
        """
        if hasattr(tarfile, "data_filter"):
            monkeypatch.delattr("tarfile.data_filter")

        tar_filename = "test_tar_links_no_data_filter.tar"
        tar_filepath = os.path.join(self.tempdir, tar_filename)

        extract_path = os.path.join(self.tempdir, "extract_path")

        with tarfile.open(tar_filepath, "w") as tar:
            file_data = io.BytesIO(b"normal\n")
            normal_file_tarinfo = tarfile.TarInfo(name="normal_file")
            normal_file_tarinfo.size = len(file_data.getbuffer())
            tar.addfile(normal_file_tarinfo, fileobj=file_data)

            info = tarfile.TarInfo("sub/normal_symlink")
            info.type = tarfile.SYMTYPE
            info.linkpath = ".." + os.path.sep + "normal_file"
            tar.addfile(info)

        untar_file(tar_filepath, extract_path)

        assert os.path.islink(os.path.join(extract_path, "sub", "normal_symlink"))

        link_path = os.readlink(os.path.join(extract_path, "sub", "normal_symlink"))
        assert link_path == ".." + os.path.sep + "normal_file"

        with open(os.path.join(extract_path, "sub", "normal_symlink"), "rb") as f:
            assert f.read() == b"normal\n"

    def test_unpack_evil_tar_link1_no_data_filter(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        """
        Test unpacking a evil tar with file containing soft links, but no data_filter
        """
        if hasattr(tarfile, "data_filter"):
            monkeypatch.delattr("tarfile.data_filter")

        tar_filename = "test_tar_links_no_data_filter.tar"
        tar_filepath = os.path.join(self.tempdir, tar_filename)

        import_filename = "import_file"
        import_filepath = os.path.join(self.tempdir, import_filename)
        open(import_filepath, "w").close()

        extract_path = os.path.join(self.tempdir, "extract_path")

        with tarfile.open(tar_filepath, "w") as tar:
            info = tarfile.TarInfo("evil_symlink")
            info.type = tarfile.SYMTYPE
            info.linkpath = import_filepath
            tar.addfile(info)

        with pytest.raises(InstallationError) as e:
            untar_file(tar_filepath, extract_path)

        msg = (
            "The tar file ({}) has a file ({}) trying to install outside "
            "target directory ({})"
        )
        assert msg.format(tar_filepath, "evil_symlink", import_filepath) in str(e.value)

        assert not os.path.exists(os.path.join(extract_path, "evil_symlink"))

    def test_unpack_evil_tar_link2_no_data_filter(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        """
        Test unpacking a evil tar with file containing soft links, but no data_filter
        """
        if hasattr(tarfile, "data_filter"):
            monkeypatch.delattr("tarfile.data_filter")

        tar_filename = "test_tar_links_no_data_filter.tar"
        tar_filepath = os.path.join(self.tempdir, tar_filename)

        import_filename = "import_file"
        import_filepath = os.path.join(self.tempdir, import_filename)
        open(import_filepath, "w").close()

        extract_path = os.path.join(self.tempdir, "extract_path")

        link_path = ".." + os.sep + import_filename

        with tarfile.open(tar_filepath, "w") as tar:
            info = tarfile.TarInfo("evil_symlink")
            info.type = tarfile.SYMTYPE
            info.linkpath = link_path
            tar.addfile(info)

        with pytest.raises(InstallationError) as e:
            untar_file(tar_filepath, extract_path)

        msg = (
            "The tar file ({}) has a file ({}) trying to install outside "
            "target directory ({})"
        )
        assert msg.format(tar_filepath, "evil_symlink", link_path) in str(e.value)

        assert not os.path.exists(os.path.join(extract_path, "evil_symlink"))


def test_unpack_tar_unicode(tmpdir: Path) -> None:
    test_tar = tmpdir / "test.tar"
    # tarfile tries to decode incoming
    with tarfile.open(test_tar, "w", format=tarfile.PAX_FORMAT, encoding="utf-8") as f:
        metadata = tarfile.TarInfo("dir/åäö_日本語.py")
        f.addfile(metadata, io.BytesIO(b"hello world"))

    output_dir = tmpdir / "output"
    output_dir.mkdir()

    untar_file(os.fspath(test_tar), str(output_dir))

    output_dir_name = str(output_dir)
    contents = os.listdir(output_dir_name)
    assert "åäö_日本語.py" in contents


@pytest.mark.parametrize(
    "args, expected",
    [
        # Test the second containing the first.
        (("parent/sub", "parent/"), False),
        # Test the first not ending in a trailing slash.
        (("parent", "parent/foo"), True),
        # Test target containing `..` but still inside the parent.
        (("parent/", "parent/foo/../bar"), True),
        # Test target within the parent
        (("parent/", "parent/sub"), True),
        # Test target outside parent
        (("parent/", "parent/../sub"), False),
        # Test target sub-string of parent
        (("parent/child", "parent/childfoo"), False),
    ],
)
def test_is_within_directory(args: Tuple[str, str], expected: bool) -> None:
    result = is_within_directory(*args)
    assert result == expected


@pytest.mark.parametrize(
    "is_zip, is_tar, unzip, untar, exception",
    [
        # zip file
        (True, False, True, False, False),
        # tar file
        (False, True, False, True, False),
        # neither zip nor tar
        (False, False, False, False, True),
        # ambiguous (both zip and tar)
        (True, True, False, False, True),
    ],
)
@patch("pip._internal.utils.unpacking.tarfile")
@patch("pip._internal.utils.unpacking.zipfile")
@patch("pip._internal.utils.unpacking.untar_file")
@patch("pip._internal.utils.unpacking.unzip_file")
def test_magic_signature_check_logic(
    mock_unzip: MagicMock,
    mock_untar: MagicMock,
    mock_zipfile: MagicMock,
    mock_tarfile: MagicMock,
    is_zip: bool,
    is_tar: bool,
    unzip: bool,
    untar: bool,
    exception: bool,
) -> None:
    """
    Test that pip throws an error if file is identified as both zip and tar
    and all other checks came out undeterministic.
    """
    mock_tarfile.is_tarfile.return_value = is_tar
    mock_zipfile.is_zipfile.return_value = is_zip
    filename = "ambiguous-file.unknown-extension"

    if exception:
        with pytest.raises(InstallationError):
            unpack_file(filename, "any-location", content_type=None)
    else:
        unpack_file(filename, "any-location", content_type=None)

    mock_unzip.assert_called_once() if unzip else mock_unzip.assert_not_called()
    mock_untar.assert_called_once() if untar else mock_untar.assert_not_called()
    mock_tarfile.is_tarfile.assert_called_once()
    mock_zipfile.is_zipfile.assert_called_once()


@pytest.mark.parametrize(
    "filename, content_type, unzip, untar",
    [
        # content_type check
        ("noname", "application/zip", True, False),
        ("noname", "application/x-gzip", False, True),
        # filename check
        ("ok.zip", None, True, False),
        ("ok.tar.gz", None, False, True),
    ],
)
@patch("pip._internal.utils.unpacking.tarfile")
@patch("pip._internal.utils.unpacking.zipfile")
@patch("pip._internal.utils.unpacking.untar_file")
@patch("pip._internal.utils.unpacking.unzip_file")
def test_check_priority(
    mock_unzip: MagicMock,
    mock_untar: MagicMock,
    mock_zipfile: MagicMock,
    mock_tarfile: MagicMock,
    filename: str,
    content_type: str | None,
    unzip: bool,
    untar: bool,
) -> None:
    """
    Test the order of priority of checks to ensure
    we don't use magic signature check unless we have to.
    """
    unpack_file(filename, "any-location", content_type=content_type)
    mock_unzip.assert_called_once() if unzip else mock_unzip.assert_not_called()
    mock_untar.assert_called_once() if untar else mock_untar.assert_not_called()
    mock_zipfile.is_zipfile.assert_not_called()
    mock_tarfile.is_tarfile.assert_not_called()


@pytest.mark.parametrize(
    "filename, expect_unzip",
    [
        ("pkg.zip", True),
        ("pkg.ZIP", True),
        ("pkg-1.0-py3-none-any.whl", True),
        ("pkg.tar.gz", False),
        ("pkg.TAR.GZ", False),
        ("pkg.tgz", False),
        ("pkg.tar", False),
        ("pkg.tar.bz2", False),
        ("pkg.tbz", False),
        ("pkg.tar.xz", False),
        ("pkg.txz", False),
        ("pkg.tlz", False),
        ("pkg.tar.lz", False),
        ("pkg.tar.lzma", False),
    ],
)
@patch("pip._internal.utils.unpacking.tarfile")
@patch("pip._internal.utils.unpacking.zipfile")
@patch("pip._internal.utils.unpacking.untar_file")
@patch("pip._internal.utils.unpacking.unzip_file")
def test_filename_extension_routing(
    mock_unzip: MagicMock,
    mock_untar: MagicMock,
    mock_zipfile: MagicMock,
    mock_tarfile: MagicMock,
    filename: str,
    expect_unzip: bool,
) -> None:
    unpack_file(filename, "any-location", content_type=None)
    (mock_unzip if expect_unzip else mock_untar).assert_called_once()
    (mock_untar if expect_unzip else mock_unzip).assert_not_called()
    mock_zipfile.is_zipfile.assert_not_called()
    mock_tarfile.is_tarfile.assert_not_called()


@pytest.mark.parametrize(
    "content_type, filename, expect_unzip",
    [
        ("application/zip", "pkg.tar.gz", True),
        ("application/x-gzip", "pkg.zip", False),
        ("application/x-gzip", "pkg.whl", False),
        ("application/octet-stream", "pkg.zip", True),
        ("application/octet-stream", "pkg.tar.gz", False),
    ],
)
@patch("pip._internal.utils.unpacking.tarfile")
@patch("pip._internal.utils.unpacking.zipfile")
@patch("pip._internal.utils.unpacking.untar_file")
@patch("pip._internal.utils.unpacking.unzip_file")
def test_content_type_vs_filename_priority(
    mock_unzip: MagicMock,
    mock_untar: MagicMock,
    mock_zipfile: MagicMock,
    mock_tarfile: MagicMock,
    content_type: str,
    filename: str,
    expect_unzip: bool,
) -> None:
    unpack_file(filename, "any-location", content_type=content_type)
    (mock_unzip if expect_unzip else mock_untar).assert_called_once()
    (mock_untar if expect_unzip else mock_unzip).assert_not_called()
    mock_zipfile.is_zipfile.assert_not_called()
    mock_tarfile.is_tarfile.assert_not_called()


@pytest.mark.parametrize("filename, flatten", [("pkg.whl", False), ("pkg.zip", True)])
@patch("pip._internal.utils.unpacking.unzip_file")
def test_flatten_only_for_non_whl(
    mock_unzip: MagicMock, filename: str, flatten: bool
) -> None:
    unpack_file(filename, "any-location", content_type=None)
    assert mock_unzip.call_args[1]["flatten"] is flatten


def _write_polyglot(path: Path) -> None:
    """Write a tar.gz with a zip appended; both views contain payload.txt."""
    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w:gz") as tar:
        info = tarfile.TarInfo("pkg/payload.txt")
        info.size = 8
        tar.addfile(info, io.BytesIO(b"from-tar"))
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr("pkg/payload.txt", "from-zip")
    path.write_bytes(tar_buf.getvalue() + zip_buf.getvalue())


@pytest.mark.parametrize(
    "filename, content_type, expected",
    [
        ("pkg.tar.gz", None, b"from-tar"),
        ("pkg.tgz", None, b"from-tar"),
        ("pkg.zip", None, b"from-zip"),
        ("pkg.tar.gz", "application/zip", b"from-zip"),
        ("pkg.unknown", "application/x-gzip", b"from-tar"),
    ],
)
def test_polyglot_routing(
    tmp_path: Path, filename: str, content_type: str | None, expected: bytes
) -> None:
    archive = tmp_path / filename
    _write_polyglot(archive)
    out = tmp_path / "out"
    unpack_file(str(archive), str(out), content_type=content_type)
    assert (out / "payload.txt").read_bytes() == expected


def test_polyglot_ambiguous_name_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "pkg.bin"
    _write_polyglot(archive)
    with pytest.raises(InstallationError):
        unpack_file(str(archive), str(tmp_path / "out"))
