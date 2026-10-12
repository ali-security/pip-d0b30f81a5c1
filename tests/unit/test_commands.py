import importlib
import logging
import os
import sys
from optparse import Values
from pathlib import Path
from typing import Any, Callable, List, NoReturn, Optional
from unittest import mock

import pytest
from pip._vendor.packaging.version import Version

from pip._internal import self_outdated_check
from pip._internal.cli import cmdoptions
from pip._internal.cli.base_command import Command
from pip._internal.cli.req_command import (
    IndexGroupCommand,
    RequirementCommand,
    SessionCommandMixin,
)
from pip._internal.cli.status_codes import ERROR, SUCCESS
from pip._internal.commands import commands_dict, create_command
from pip._internal.exceptions import InstallationError
from pip._internal.self_outdated_check import UpgradePrompt

# These are the expected names of the commands whose classes inherit from
# IndexGroupCommand.
EXPECTED_INDEX_GROUP_COMMANDS = ["download", "index", "install", "list", "wheel"]


def check_commands(pred: Callable[[Command], bool], expected: List[str]) -> None:
    """
    Check the commands satisfying a predicate.
    """
    commands = [create_command(name) for name in sorted(commands_dict)]
    actual = [command.name for command in commands if pred(command)]
    assert actual == expected, f"actual: {actual}"


def test_commands_dict__order() -> None:
    """
    Check the ordering of commands_dict.
    """
    names = list(commands_dict)
    # A spot-check is sufficient to check that commands_dict encodes an
    # ordering.
    assert names[0] == "install"
    assert names[-1] == "help"


@pytest.mark.parametrize("name", list(commands_dict))
def test_create_command(name: str) -> None:
    """Test creating an instance of each available command."""
    command = create_command(name)
    assert command.name == name
    assert command.summary == commands_dict[name].summary


def test_session_commands() -> None:
    """
    Test which commands inherit from SessionCommandMixin.
    """

    def is_session_command(command: Command) -> bool:
        return isinstance(command, SessionCommandMixin)

    expected = ["download", "index", "install", "list", "search", "uninstall", "wheel"]
    check_commands(is_session_command, expected)


def test_index_group_commands() -> None:
    """
    Test the commands inheriting from IndexGroupCommand.
    """

    def is_index_group_command(command: Command) -> bool:
        return isinstance(command, IndexGroupCommand)

    check_commands(is_index_group_command, EXPECTED_INDEX_GROUP_COMMANDS)

    # Also check that the commands inheriting from IndexGroupCommand are
    # exactly the commands with the --no-index option.
    def has_option_no_index(command: Command) -> bool:
        return command.parser.has_option("--no-index")

    check_commands(has_option_no_index, EXPECTED_INDEX_GROUP_COMMANDS)


@pytest.mark.parametrize("command_name", EXPECTED_INDEX_GROUP_COMMANDS)
@pytest.mark.parametrize(
    "disable_pip_version_check, no_index, expected_called",
    [
        # The fetch phase only runs when both disable_pip_version_check
        # and no_index are False.
        (False, False, True),
        (False, True, False),
        (True, False, False),
        (True, True, False),
    ],
)
@mock.patch("pip._internal.cli.req_command.pip_self_version_check_fetch")
def test_index_group_pip_version_check(
    mock_version_check: mock.Mock,
    command_name: str,
    disable_pip_version_check: bool,
    no_index: bool,
    expected_called: bool,
) -> None:
    """
    Test whether the pre-body fetch runs when ``pip_version_check()`` is
    entered, for each of the IndexGroupCommand classes.
    """
    command = create_command(command_name)
    options = command.parser.get_default_values()
    options.disable_pip_version_check = disable_pip_version_check
    options.no_index = no_index
    # Return None so the emit branch is a no-op.
    mock_version_check.return_value = None

    with command.pip_version_check(options):
        pass
    if expected_called:
        mock_version_check.assert_called_once()
    else:
        mock_version_check.assert_not_called()


class FakeIndexGroupCommand(IndexGroupCommand):
    """An IndexGroupCommand whose body is provided by the test."""

    def __init__(self, run_func: Callable[[], int]) -> None:
        self.run_func = run_func
        super().__init__("fake", "fake")

    def add_options(self) -> None:
        index_opts = cmdoptions.make_option_group(cmdoptions.index_group, self.parser)
        self.parser.insert_option_group(0, index_opts)

    def run(self, options: Values, args: List[str]) -> int:
        return self.run_func()


@pytest.fixture
def main_without_global_logging(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Allow tests to call ``Command.main()`` without leaking logging state.

    ``Command.main()`` calls ``setup_logging()``, which installs pip's console
    handlers on the root logger bound to the *current* (pytest capture)
    streams and never removes them. Tests running later on the same worker
    would then log into closed streams ("I/O operation on closed file").
    These tests only need the records, which ``caplog`` collects anyway.
    """
    monkeypatch.setattr(
        "pip._internal.cli.base_command.setup_logging",
        lambda **kwargs: logging.INFO,
    )
    caplog.set_level(logging.WARNING)
    # The autouse ``isolate`` fixture disables the check for every test.
    monkeypatch.setenv("PIP_DISABLE_PIP_VERSION_CHECK", "no")


@pytest.mark.parametrize("body_fails", [False, True])
@pytest.mark.usefixtures("main_without_global_logging")
def test_pip_version_check_fetches_before_command_body(
    body_fails: bool, tmpdir: Path
) -> None:
    """
    The remote version is fetched before the command body runs, and the
    upgrade prompt is emitted afterwards, even if the command body fails.
    """
    calls: List[str] = []

    def fetch(session: Any, options: Values) -> str:
        calls.append("fetch")
        return "99.0"

    def emit(remote_version_str: Optional[str]) -> None:
        calls.append(f"emit {remote_version_str}")

    def run() -> int:
        calls.append("run")
        if body_fails:
            raise InstallationError("boom")
        return SUCCESS

    with mock.patch(
        "pip._internal.cli.req_command.pip_self_version_check_fetch",
        side_effect=fetch,
    ), mock.patch(
        "pip._internal.cli.req_command.pip_self_version_check_emit",
        side_effect=emit,
    ):
        status = FakeIndexGroupCommand(run).main(
            ["--cache-dir", os.fspath(tmpdir / "cache")]
        )

    assert status == (ERROR if body_fails else SUCCESS)
    assert calls == ["fetch", "run", "emit 99.0"]


_PAYLOAD_MODULE = "pip_test_cve_2026_6357_payload"


@pytest.mark.usefixtures("main_without_global_logging")
def test_pip_version_check_does_not_load_code_installed_by_command_body(
    tmpdir: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """
    CVE-2026-6357: the self version check must not import code that the
    command body has just installed.

    The version check looks up a module lazily, by a well-known name, the way
    pip's network code imports optional modules such as ``keyring``. The
    command body then "installs" a module of that name onto ``sys.path``. If
    the check ran after the command body, it would import and execute it.
    """
    site_dir = tmpdir / "site-packages"
    site_dir.mkdir()
    monkeypatch.syspath_prepend(os.fspath(site_dir))
    marker = tmpdir / "payload-executed"
    lookups: List[str] = []

    def get_remote_version(session: Any, options: Values) -> str:
        try:
            importlib.import_module(_PAYLOAD_MODULE)
        except ImportError:
            lookups.append("not installed")
        else:
            lookups.append("imported")
        return "99.0"

    def install_payload() -> int:
        (site_dir / f"{_PAYLOAD_MODULE}.py").write_text(
            f"open({os.fspath(marker)!r}, 'w').close()\n", encoding="utf-8"
        )
        importlib.invalidate_caches()
        return SUCCESS

    installed_pip = mock.Mock(version=Version("1.0"), installer="pip")
    monkeypatch.setattr(
        self_outdated_check,
        "get_default_environment",
        lambda: mock.Mock(get_distribution=mock.Mock(return_value=installed_pip)),
    )
    monkeypatch.setattr(
        self_outdated_check, "_get_current_remote_pip_version", get_remote_version
    )

    try:
        status = FakeIndexGroupCommand(install_payload).main(
            ["--cache-dir", os.fspath(tmpdir / "cache")]
        )
    finally:
        sys.modules.pop(_PAYLOAD_MODULE, None)

    assert status == SUCCESS
    # The lookup happened before the command body installed the module...
    assert lookups == ["not installed"]
    # ...so the installed code never ran.
    assert not marker.exists()
    # The upgrade prompt is still emitted after the command body.
    assert (UpgradePrompt(old="1.0", new="99.0"),) in [
        record.args for record in caplog.records
    ]


@pytest.mark.parametrize("failing_step", ["build_session", "fetch", "emit"])
@pytest.mark.usefixtures("main_without_global_logging")
def test_pip_version_check_error_does_not_prevent_command_body(
    failing_step: str, tmpdir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An error in either phase of the check is logged, not raised."""
    calls: List[str] = []

    def fail(*args: Any, **kwargs: Any) -> NoReturn:
        raise RuntimeError(f"{failing_step} failed")

    def run() -> int:
        calls.append("run")
        return SUCCESS

    command = FakeIndexGroupCommand(run)
    with mock.patch.object(
        command, "_build_session", wraps=command._build_session
    ) as mock_build_session, mock.patch(
        "pip._internal.cli.req_command.pip_self_version_check_fetch",
        return_value="99.0",
    ) as mock_fetch, mock.patch(
        "pip._internal.cli.req_command.pip_self_version_check_emit",
    ) as mock_emit:
        steps = {
            "build_session": mock_build_session,
            "fetch": mock_fetch,
            "emit": mock_emit,
        }
        steps[failing_step].side_effect = fail
        status = command.main(["--cache-dir", os.fspath(tmpdir / "cache")])

    assert status == SUCCESS
    assert calls == ["run"]
    assert (
        "There was an error checking the latest version of pip." in caplog.messages
    )
    if failing_step == "emit":
        mock_emit.assert_called_once_with("99.0")
    else:
        mock_emit.assert_not_called()


def test_requirement_commands() -> None:
    """
    Test which commands inherit from RequirementCommand.
    """

    def is_requirement_command(command: Command) -> bool:
        return isinstance(command, RequirementCommand)

    check_commands(is_requirement_command, ["download", "install", "wheel"])
