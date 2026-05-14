"""Tests for the .githooks/commit-msg DCO sign-off hook.

The hook blocks any commit whose message does not carry a `Signed-off-by:`
trailer. These tests invoke the hook as a subprocess against a temp message
file so they exercise the real shell script end-to-end.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

HOOK = Path(__file__).resolve().parents[3] / ".githooks" / "commit-msg"


def _run_hook(message: str, tmp_path: Path) -> subprocess.CompletedProcess:
    msg_file = tmp_path / "COMMIT_EDITMSG"
    msg_file.write_text(message)
    return subprocess.run(
        ["sh", str(HOOK), str(msg_file)],
        capture_output=True,
        text=True,
    )


class TestDCOCommitMsgHook:
    def test_hook_file_exists_and_is_executable(self):
        assert HOOK.exists(), f"DCO hook missing at {HOOK}"
        assert HOOK.stat().st_mode & 0o111, "DCO hook is not executable"

    def test_blocks_commit_missing_signed_off_by(self, tmp_path):
        result = _run_hook("fix: a bug\n\nbody text\n", tmp_path)
        assert result.returncode == 1

    def test_accepts_commit_with_signed_off_by(self, tmp_path):
        msg = (
            "fix: a bug\n"
            "\n"
            "body text\n"
            "\n"
            "Signed-off-by: Test User <test@example.com>\n"
        )
        result = _run_hook(msg, tmp_path)
        assert result.returncode == 0, result.stdout + result.stderr

    def test_skips_merge_commit_without_trailer(self, tmp_path):
        """Merge commits don't need their own sign-off - the merged commits carry it."""
        msg = "Merge branch 'feature/x' into main\n"
        result = _run_hook(msg, tmp_path)
        assert result.returncode == 0, result.stdout + result.stderr

    def test_skips_fixup_commit_without_trailer(self, tmp_path):
        """fixup! commits are autosquashed - the target commit carries sign-off."""
        msg = "fixup! fix: a bug\n"
        result = _run_hook(msg, tmp_path)
        assert result.returncode == 0, result.stdout + result.stderr

    def test_skips_squash_commit_without_trailer(self, tmp_path):
        msg = "squash! fix: a bug\n"
        result = _run_hook(msg, tmp_path)
        assert result.returncode == 0, result.stdout + result.stderr

    def test_skips_revert_commit_without_trailer(self, tmp_path):
        """Revert commits inherit sign-off context from the commit being reverted."""
        msg = 'Revert "fix: a bug"\n\nThis reverts commit abc123.\n'
        result = _run_hook(msg, tmp_path)
        assert result.returncode == 0, result.stdout + result.stderr

    def test_blocks_malformed_signed_off_by_without_email(self, tmp_path):
        """GitHub's DCO check requires Name <email> form - a name-only trailer must not pass."""
        msg = "fix: a bug\n\nSigned-off-by: Test User\n"
        result = _run_hook(msg, tmp_path)
        assert result.returncode == 1, (
            "trailer without email should not satisfy DCO check"
        )

    def test_error_message_tells_user_how_to_fix(self, tmp_path):
        """When blocking, the hook must teach the user the remediation command."""
        result = _run_hook("fix: a bug\n", tmp_path)
        output = result.stdout + result.stderr
        assert "git commit -s" in output, "error must show -s as the fix"
        assert "--amend" in output, "error must mention amend for fixing the last commit"
