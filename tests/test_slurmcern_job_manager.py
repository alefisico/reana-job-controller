# -*- coding: utf-8 -*-
#
# This file is part of REANA.
# Copyright (C) 2026 CERN.
#
# REANA is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.

"""Tests for SlurmJobManagerCERN image handling methods."""

from unittest.mock import MagicMock, patch


def _make_manager(docker_img):
    """Instantiate SlurmJobManagerCERN with SSH mocked out."""
    with patch("reana_job_controller.slurmcern_job_manager.SSHClient"):
        from reana_job_controller.slurmcern_job_manager import SlurmJobManagerCERN
        mgr = SlurmJobManagerCERN.__new__(SlurmJobManagerCERN)
        mgr.docker_img = docker_img
        mgr.img_type_docker = mgr._is_img_type_docker()
        mgr.slurm_connection = MagicMock()
        mgr.cvmfs_mounts = "false"
        SlurmJobManagerCERN.SLURM_WORKSAPCE_PATH = "/remote/workspace"
        return mgr


class TestIsImgTypeDocker:
    def test_plain_docker_image_is_docker_type(self):
        mgr = _make_manager("docker.io/snakemake/snakemake:v9.16.3")
        assert mgr._is_img_type_docker() is True

    def test_sif_path_is_not_docker_type(self):
        mgr = _make_manager("/some/path/image.sif")
        assert mgr._is_img_type_docker() is False

    def test_cvmfs_path_is_not_docker_type(self):
        mgr = _make_manager("/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:latest")
        assert mgr._is_img_type_docker() is False


class TestGetContainer:
    def test_docker_image_becomes_sif_filename(self):
        mgr = _make_manager("docker.io/snakemake/snakemake:v9.16.3")
        assert mgr._get_container() == "snakemake_v9.16.3.sif"

    def test_sif_path_returned_unchanged(self):
        mgr = _make_manager("/some/path/image.sif")
        assert mgr._get_container() == "/some/path/image.sif"

    def test_cvmfs_path_returned_unchanged(self):
        img = "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:latest"
        mgr = _make_manager(img)
        assert mgr._get_container() == img


class TestSifPath:
    def test_sif_path_combines_workspace_and_sif_filename(self):
        mgr = _make_manager("docker.io/snakemake/snakemake:v9.16.3")
        assert mgr._sif_path() == "/remote/workspace/snakemake_v9.16.3.sif"


class TestPullImage:
    def test_pull_skipped_for_non_docker_image(self):
        """No singularity pull for .sif or cvmfs paths."""
        mgr = _make_manager("/some/path/image.sif")
        mgr._pull_image()
        mgr.slurm_connection.exec_command.assert_not_called()

    def test_pull_skipped_if_sif_already_exists(self):
        """If test -f returns a string (exit 0), singularity pull is not called."""
        mgr = _make_manager("docker.io/snakemake/snakemake:v9.16.3")
        # exec_command returns "" (non-None) → file exists
        mgr.slurm_connection.exec_command.return_value = ""
        mgr._pull_image()
        assert mgr.slurm_connection.exec_command.call_count == 1
        called_cmd = mgr.slurm_connection.exec_command.call_args[0][0]
        assert "test -f" in called_cmd
        assert "singularity pull" not in called_cmd

    def test_pull_executed_when_sif_absent(self):
        """If test -f returns None (exit non-zero), singularity pull is called."""
        mgr = _make_manager("docker.io/snakemake/snakemake:v9.16.3")
        # First call (test -f) returns None → file absent
        # Second call (singularity pull) returns "" → success
        mgr.slurm_connection.exec_command.side_effect = [None, ""]
        mgr._pull_image()
        assert mgr.slurm_connection.exec_command.call_count == 2
        pull_cmd = mgr.slurm_connection.exec_command.call_args_list[1][0][0]
        assert "singularity pull" in pull_cmd
        assert "docker://docker.io/snakemake/snakemake:v9.16.3" in pull_cmd

    def test_pull_executed_only_once_on_retry(self):
        """Second call to _pull_image() skips pull because .sif now exists."""
        mgr = _make_manager("docker.io/snakemake/snakemake:v9.16.3")
        # First invocation: file absent → pull
        mgr.slurm_connection.exec_command.side_effect = [None, ""]
        mgr._pull_image()
        # Second invocation: file now exists → skip
        mgr.slurm_connection.exec_command.side_effect = None
        mgr.slurm_connection.exec_command.return_value = ""
        mgr.slurm_connection.exec_command.reset_mock()
        mgr._pull_image()
        assert mgr.slurm_connection.exec_command.call_count == 1


class TestJobNameSanitization:
    """Tests for SBATCH job-name sanitization."""

    def test_job_name_with_wildcards_is_sanitized(self):
        """Parentheses, spaces, commas, = in job name are replaced with underscores."""
        mgr = _make_manager("/cvmfs/unpacked.cern.ch/image")
        mgr.img_type_docker = False
        mgr.job_name = "analysis_databkgs (sample=TTToHadronic, year=UL16_preVFP)"
        mgr.partition = "standard"
        mgr.timelimit = "1:00:00"
        mgr.job_file = "job.sh"
        mgr.job_description_file = "job_description.sh"
        mgr.secrets = None
        mgr.__class__.REANA_WORKSPACE_PATH = "/reana/workspace"

        mgr._dump_job_submission_file()

        written = mgr.slurm_connection.exec_command.call_args[0][0]
        assert "(sample=TTToHadronic, year=UL16_preVFP)" not in written
        assert "analysis_databkgs__sample_TTToHadronic__year_UL16_preVFP_" in written

    def test_plain_job_name_unchanged(self):
        """Job names without special chars pass through unchanged."""
        mgr = _make_manager("/cvmfs/unpacked.cern.ch/image")
        mgr.img_type_docker = False
        mgr.job_name = "analysis_databkgs"
        mgr.partition = "standard"
        mgr.timelimit = "1:00:00"
        mgr.job_file = "job.sh"
        mgr.job_description_file = "job_description.sh"
        mgr.secrets = None
        mgr.__class__.REANA_WORKSPACE_PATH = "/reana/workspace"

        mgr._dump_job_submission_file()

        written = mgr.slurm_connection.exec_command.call_args[0][0]
        assert "analysis_databkgs" in written


class TestSbatchFailure:
    """Tests for sbatch SSH failure handling."""

    def test_raises_on_sbatch_none_return(self):
        """RuntimeError raised when sbatch exec_command returns None (SSH failure)."""
        import pytest
        from unittest.mock import patch, MagicMock

        with patch("reana_job_controller.slurmcern_job_manager.SSHClient"):
            from reana_job_controller.slurmcern_job_manager import SlurmJobManagerCERN
            mgr = SlurmJobManagerCERN.__new__(SlurmJobManagerCERN)
            mgr.docker_img = "docker.io/myorg/myimage:v1.0"
            mgr.img_type_docker = True
            mgr.slurm_connection = MagicMock()
            mgr.slurm_connection.exec_command.return_value = None
            mgr.job_description_file = "job_description.sh"
            SlurmJobManagerCERN.SLURM_WORKSAPCE_PATH = "/remote/workspace"

            with pytest.raises(RuntimeError, match="sbatch returned no output"):
                mgr._execute_sbatch()

    def test_returns_job_id_on_success(self):
        """backend_job_id is stripped stdout when sbatch succeeds."""
        from unittest.mock import patch, MagicMock

        with patch("reana_job_controller.slurmcern_job_manager.SSHClient"):
            from reana_job_controller.slurmcern_job_manager import SlurmJobManagerCERN
            mgr = SlurmJobManagerCERN.__new__(SlurmJobManagerCERN)
            mgr.docker_img = "docker.io/myorg/myimage:v1.0"
            mgr.img_type_docker = True
            mgr.slurm_connection = MagicMock()
            mgr.slurm_connection.exec_command.return_value = "12345\n"
            mgr.job_description_file = "job_description.sh"
            SlurmJobManagerCERN.SLURM_WORKSAPCE_PATH = "/remote/workspace"

            result = mgr._execute_sbatch()
            assert result == "12345"


def _make_manager_with_secrets(docker_img, secrets):
    """Instantiate SlurmJobManagerCERN with secrets set."""
    mgr = _make_manager(docker_img)
    mgr.secrets = secrets
    SlurmJobManagerCERN = mgr.__class__
    SlurmJobManagerCERN.REANA_WORKSPACE_PATH = "/reana/workspace"
    mgr.job_file = "job.sh"
    return mgr


def _make_user_secrets(file_secrets=None, env_secrets=None):
    """Build a UserSecrets object with given file and env secrets."""
    from reana_commons.k8s.secrets import UserSecrets, Secret
    secrets_list = []
    for name, value in (file_secrets or {}).items():
        secrets_list.append(Secret(name, "file", value))
    for name, value in (env_secrets or {}).items():
        secrets_list.append(Secret(name, "env", value))
    return UserSecrets(user_id="test-user", k8s_secret_name="test-secret", secrets=secrets_list)


class TestSecretsBindMount:
    """Tests for _secrets_bind_mount."""

    def test_no_secrets_returns_empty(self):
        mgr = _make_manager("docker.io/org/img:v1")
        mgr.secrets = None
        mgr.job_file = "job.sh"
        from reana_job_controller.slurmcern_job_manager import SlurmJobManagerCERN
        SlurmJobManagerCERN.REANA_WORKSPACE_PATH = "/reana/workspace"
        assert mgr._secrets_bind_mount() == ""

    def test_env_only_secrets_returns_empty(self):
        mgr = _make_manager_with_secrets(
            "docker.io/org/img:v1",
            _make_user_secrets(env_secrets={"MY_VAR": "value"}),
        )
        assert mgr._secrets_bind_mount() == ""

    def test_file_secrets_returns_bind_flag(self):
        mgr = _make_manager_with_secrets(
            "docker.io/org/img:v1",
            _make_user_secrets(file_secrets={"usercert.pem": b"cert-data"}),
        )
        result = mgr._secrets_bind_mount()
        assert result.startswith(" -B ")
        assert "reana_secrets" in result
        assert "/etc/reana/secrets" in result
        assert ":ro" in result


class TestEnvSecretsExports:
    """Tests for _env_secrets_exports."""

    def test_no_secrets_returns_empty(self):
        mgr = _make_manager("docker.io/org/img:v1")
        mgr.secrets = None
        assert mgr._env_secrets_exports() == ""

    def test_file_only_secrets_returns_empty(self):
        mgr = _make_manager_with_secrets(
            "docker.io/org/img:v1",
            _make_user_secrets(file_secrets={"cert.pem": b"data"}),
        )
        assert mgr._env_secrets_exports() == ""

    def test_env_secrets_produce_export_lines(self):
        mgr = _make_manager_with_secrets(
            "docker.io/org/img:v1",
            _make_user_secrets(env_secrets={"VOMSPROXY_PASS": "mypassword"}),
        )
        result = mgr._env_secrets_exports()
        assert "export VOMSPROXY_PASS=mypassword" in result

    def test_multiple_env_secrets(self):
        mgr = _make_manager_with_secrets(
            "docker.io/org/img:v1",
            _make_user_secrets(env_secrets={"VAR_A": "aaa", "VAR_B": "bbb"}),
        )
        result = mgr._env_secrets_exports()
        assert "export VAR_A=aaa" in result
        assert "export VAR_B=bbb" in result


class TestTransferSecrets:
    """Tests for _transfer_secrets."""

    def test_no_secrets_does_nothing(self):
        mgr = _make_manager("docker.io/org/img:v1")
        mgr.secrets = None
        sftp = MagicMock()
        mgr._transfer_secrets(sftp)
        sftp.mkdir.assert_not_called()
        sftp.put.assert_not_called()

    def test_file_secrets_are_transferred(self):
        import tempfile, os
        mgr = _make_manager_with_secrets(
            "docker.io/org/img:v1",
            _make_user_secrets(file_secrets={"usercert.pem": b"cert", "userkey.pem": b"key"}),
        )
        sftp = MagicMock()
        mgr._transfer_secrets(sftp)
        sftp.mkdir.assert_called_once()
        assert sftp.put.call_count == 2
        transferred_names = {
            os.path.basename(call.args[1]) for call in sftp.put.call_args_list
        }
        assert transferred_names == {"usercert.pem", "userkey.pem"}

    def test_env_secrets_are_not_transferred(self):
        mgr = _make_manager_with_secrets(
            "docker.io/org/img:v1",
            _make_user_secrets(env_secrets={"MY_VAR": "value"}),
        )
        sftp = MagicMock()
        mgr._transfer_secrets(sftp)
        sftp.put.assert_not_called()


def _make_voms_secrets(**extra_env):
    """Build a UserSecrets with the full set of voms-proxy secrets."""
    env = {"VOMSPROXY_PASS": "cGFzc3dvcmQ=", "VONAME": "cms"}
    env.update(extra_env)
    return _make_user_secrets(
        file_secrets={"usercert.pem": b"cert", "userkey.pem": b"key"},
        env_secrets=env,
    )


class TestVomsProxy:
    """Tests for voms-proxy generation in the job submission script."""

    def test_has_voms_secrets_true_when_all_present(self):
        mgr = _make_manager_with_secrets("docker.io/org/img:v1", _make_voms_secrets())
        assert mgr._has_voms_secrets() is True

    def test_has_voms_secrets_false_when_missing_key(self):
        mgr = _make_manager_with_secrets(
            "docker.io/org/img:v1",
            _make_user_secrets(
                file_secrets={"usercert.pem": b"cert"},
                env_secrets={"VOMSPROXY_PASS": "cGFzc3dvcmQ="},
            ),
        )
        assert mgr._has_voms_secrets() is False

    def test_has_voms_secrets_false_when_no_secrets(self):
        mgr = _make_manager("docker.io/org/img:v1")
        mgr.secrets = None
        assert mgr._has_voms_secrets() is False

    def test_voms_proxy_init_cmd_contains_voms_proxy_init(self):
        mgr = _make_manager_with_secrets("docker.io/org/img:v1", _make_voms_secrets())
        cmd = mgr._voms_proxy_init_cmd()
        assert "voms-proxy-init" in cmd
        assert "--voms cms" in cmd
        assert "usercert.pem" in cmd
        assert "userkey.pem" in cmd
        assert "--pwstdin" in cmd
        assert "voms_proxy.pem" in cmd
        assert "/tmp/userkey.pem" not in cmd  # key copied to workspace, not /tmp

    def test_voms_proxy_init_cmd_empty_when_secrets_missing(self):
        mgr = _make_manager_with_secrets(
            "docker.io/org/img:v1",
            _make_user_secrets(file_secrets={"usercert.pem": b"cert"}),
        )
        assert mgr._voms_proxy_init_cmd() == ""

    def test_voms_proxy_bind_not_in_secrets_bind_mount(self):
        """Proxy bind is now a shell variable, not hardcoded in _secrets_bind_mount."""
        mgr = _make_manager_with_secrets("docker.io/org/img:v1", _make_voms_secrets())
        bind = mgr._secrets_bind_mount()
        assert "voms_proxy.pem" not in bind

    def test_voms_proxy_shell_vars_in_singularity_cmd(self):
        """Singularity cmd uses escaped shell variables for proxy bind/env.

        The variables must be written as \\$ so they survive the double-quoted
        bash assignment used to write job_description.sh on the remote host.
        """
        mgr = _make_manager_with_secrets("docker.io/org/img:v1", _make_voms_secrets())
        cmd = mgr._wrap_singularity_cmd()
        assert r"\$REANA_VOMS_PROXY_BIND" in cmd
        assert r"\$REANA_VOMS_PROXY_ENV" in cmd

    def test_voms_proxy_shell_vars_absent_without_voms_secrets(self):
        """Shell variable placeholders not added when voms secrets are absent."""
        mgr = _make_manager_with_secrets(
            "docker.io/org/img:v1",
            _make_user_secrets(env_secrets={"OTHER_VAR": "val"}),
        )
        cmd = mgr._wrap_singularity_cmd()
        assert r"\$REANA_VOMS_PROXY_BIND" not in cmd
        assert r"\$REANA_VOMS_PROXY_ENV" not in cmd

    def test_voms_proxy_init_cmd_sets_shell_vars_conditionally(self):
        """_voms_proxy_init_cmd sets shell vars only if proxy file was created."""
        mgr = _make_manager_with_secrets("docker.io/org/img:v1", _make_voms_secrets())
        cmd = mgr._voms_proxy_init_cmd()
        assert "if [ -f" in cmd
        assert "REANA_VOMS_PROXY_BIND" in cmd
        assert "REANA_VOMS_PROXY_ENV" in cmd
        assert "X509_USER_PROXY=/tmp/voms_proxy.pem" in cmd

    def test_voms_proxy_init_in_job_submission_script(self):
        mgr = _make_manager_with_secrets("docker.io/org/img:v1", _make_voms_secrets())
        mgr.job_name = "test_job"
        mgr.partition = "standard"
        mgr.timelimit = "1:00:00"
        mgr.job_description_file = "job_description.sh"
        mgr.__class__.REANA_WORKSPACE_PATH = "/reana/workspace"
        mgr._dump_job_submission_file()
        written = mgr.slurm_connection.exec_command.call_args[0][0]
        assert "voms-proxy-init" in written
        assert "REANA_VOMS_PROXY_BIND" in written
        assert "if [ -f" in written

    def test_no_voms_proxy_when_secrets_absent(self):
        mgr = _make_manager_with_secrets(
            "docker.io/org/img:v1",
            _make_user_secrets(env_secrets={"OTHER_VAR": "val"}),
        )
        mgr.job_name = "test_job"
        mgr.partition = "standard"
        mgr.timelimit = "1:00:00"
        mgr.job_description_file = "job_description.sh"
        mgr.__class__.REANA_WORKSPACE_PATH = "/reana/workspace"
        mgr._dump_job_submission_file()
        written = mgr.slurm_connection.exec_command.call_args[0][0]
        assert "voms-proxy-init" not in written


class TestNativeExecution:
    """Tests for native (no container) execution path."""

    def test_no_image_is_not_docker_type(self):
        """None docker_img → img_type_docker is False."""
        mgr = _make_manager(None)
        assert mgr.img_type_docker is False

    def test_pull_image_skipped_when_no_image(self):
        """No singularity pull when docker_img is None."""
        mgr = _make_manager(None)
        mgr._pull_image()
        mgr.slurm_connection.exec_command.assert_not_called()

    def test_wrap_singularity_cmd_returns_native_cmd_when_no_image(self):
        """_wrap_singularity_cmd() returns ./job.sh directly when no container."""
        mgr = _make_manager(None)
        mgr.job_file = "job.sh"
        result = mgr._wrap_singularity_cmd()
        assert result == "./job.sh"
        assert "singularity" not in result


class TestGetOutputs:
    """Tests for SlurmJobManagerCERN.get_outputs()."""

    def test_no_workspace_returns_immediately(self):
        """get_outputs() with no workspace arg is a no-op (no SSH connection made)."""
        with patch("reana_job_controller.slurmcern_job_manager.SSHClient") as mock_ssh_cls:
            from reana_job_controller.slurmcern_job_manager import SlurmJobManagerCERN
            SlurmJobManagerCERN.get_outputs()
            mock_ssh_cls.assert_not_called()

    def test_shared_filesystem_skips_transfer(self, tmp_path):
        """When slurm_workspace resolves to the same path as workspace, no SFTP transfer."""
        workspace = str(tmp_path / "workspace")
        with patch("reana_job_controller.slurmcern_job_manager.SSHClient") as mock_ssh_cls:
            mock_conn = MagicMock()
            mock_ssh_cls.return_value = mock_conn
            mock_conn.exec_command.return_value = ""  # pwd returns empty → home=""
            from reana_job_controller.slurmcern_job_manager import SlurmJobManagerCERN
            # Set SLURM_HOME_PATH so slurm_workspace == workspace
            original = SlurmJobManagerCERN.SLURM_HOME_PATH
            try:
                SlurmJobManagerCERN.SLURM_HOME_PATH = "/"
                SlurmJobManagerCERN.get_outputs(workspace=workspace)
                # SFTP should not be opened
                mock_conn.ssh_client.open_sftp.assert_not_called()
            finally:
                SlurmJobManagerCERN.SLURM_HOME_PATH = original

    def test_download_called_when_paths_differ(self, tmp_path):
        """When slurm and local workspaces differ, _download_dir is called via SFTP."""
        workspace = str(tmp_path / "local_workspace")
        slurm_home = "/home/export/reana-CMU"
        with patch("reana_job_controller.slurmcern_job_manager.SSHClient") as mock_ssh_cls:
            mock_conn = MagicMock()
            mock_ssh_cls.return_value = mock_conn
            mock_sftp = MagicMock()
            mock_conn.ssh_client.open_sftp.return_value = mock_sftp
            # listdir_attr returns empty list → no files to download
            mock_sftp.listdir_attr.return_value = []
            from reana_job_controller.slurmcern_job_manager import SlurmJobManagerCERN
            original = SlurmJobManagerCERN.SLURM_HOME_PATH
            try:
                SlurmJobManagerCERN.SLURM_HOME_PATH = slurm_home
                SlurmJobManagerCERN.get_outputs(workspace=workspace)
                mock_conn.ssh_client.open_sftp.assert_called_once()
                mock_sftp.close.assert_called_once()
                # The remote dir passed to listdir_attr should be the slurm workspace
                expected_remote = slurm_home + workspace
                mock_sftp.listdir_attr.assert_called_once_with(expected_remote)
            finally:
                SlurmJobManagerCERN.SLURM_HOME_PATH = original

    def test_ssh_failure_is_logged_not_raised(self):
        """get_outputs() logs errors but does not propagate exceptions."""
        with patch("reana_job_controller.slurmcern_job_manager.SSHClient") as mock_ssh_cls:
            mock_ssh_cls.side_effect = Exception("Connection refused")
            from reana_job_controller.slurmcern_job_manager import SlurmJobManagerCERN
            # Should not raise
            SlurmJobManagerCERN.get_outputs(workspace="/opt/reana/users/x/workflows/y")


class TestTransferInputsUploadOnce:
    """Tests for the upload-once sentinel + lock behaviour in _transfer_inputs."""

    def _make_transfer_manager(self, workspace):
        """Manager wired for _transfer_inputs tests."""
        with patch("reana_job_controller.slurmcern_job_manager.SSHClient"):
            from reana_job_controller.slurmcern_job_manager import SlurmJobManagerCERN
            import threading
            mgr = SlurmJobManagerCERN.__new__(SlurmJobManagerCERN)
            mgr.docker_img = "/cvmfs/unpacked.cern.ch/image:latest"
            mgr.img_type_docker = False
            mgr.slurm_connection = MagicMock()
            mgr.workflow_workspace = workspace
            mgr.secrets = None
            mgr.slurm_workspace_path = "/remote" + workspace
            mgr.slurm_home_path = "/remote"
            mgr.reana_workspace_path = workspace
            # Reset class-level state between tests
            SlurmJobManagerCERN._upload_locks = {}
            return mgr

    def test_sentinel_written_after_successful_upload(self, tmp_path):
        """After _do_transfer_inputs completes, sentinel file is written to remote."""
        from reana_job_controller.slurmcern_job_manager import SlurmJobManagerCERN
        workspace = str(tmp_path)
        mgr = self._make_transfer_manager(workspace)

        # exec_command: pwd, mkdir, sentinel check → absent (None), sentinel write
        mgr.slurm_connection.exec_command.side_effect = [
            "/remote",  # pwd
            "",         # mkdir
            None,       # test -f sentinel → absent
            "",         # touch sentinel
        ]
        sftp = MagicMock()
        sftp.listdir_attr.return_value = []
        mgr.slurm_connection.ssh_client.open_sftp.return_value = sftp

        mgr._do_transfer_inputs()

        # The last exec_command call must write the sentinel
        last_cmd = mgr.slurm_connection.exec_command.call_args_list[-1][0][0]
        assert ".reana_upload_complete" in last_cmd

    def test_upload_skipped_when_sentinel_exists(self, tmp_path):
        """When sentinel already exists on remote, SFTP put is never called."""
        from reana_job_controller.slurmcern_job_manager import SlurmJobManagerCERN
        workspace = str(tmp_path)
        # Put a real file in the workspace so os.walk would find it
        (tmp_path / "processor_HH4b.py").write_text("class analysis: pass")
        mgr = self._make_transfer_manager(workspace)

        # exec_command: pwd, mkdir, sentinel check → present ("")
        mgr.slurm_connection.exec_command.side_effect = [
            "/remote",  # pwd
            "",         # mkdir
            "",         # test -f sentinel → present (non-None = file exists)
        ]
        sftp = MagicMock()
        mgr.slurm_connection.ssh_client.open_sftp.return_value = sftp

        mgr._do_transfer_inputs()

        sftp.put.assert_not_called()

    def test_concurrent_jobs_only_upload_once(self, tmp_path):
        """With two threads calling _transfer_inputs, SFTP put runs only once.

        Both managers share the same workspace. The first thread does the full
        upload and writes the sentinel. The second thread, after waiting for the
        lock, finds the sentinel and skips the SFTP put.
        """
        import threading
        from reana_job_controller.slurmcern_job_manager import SlurmJobManagerCERN

        workspace = str(tmp_path)
        # Put a real file so os.walk finds something to upload
        (tmp_path / "processor.py").write_text("class analysis: pass")

        SlurmJobManagerCERN._upload_locks = {}

        sftp_put_count = [0]

        def make_mgr():
            mgr = self._make_transfer_manager(workspace)
            real_sftp = MagicMock()

            def counting_put(local, remote):
                sftp_put_count[0] += 1

            real_sftp.put.side_effect = counting_put
            real_sftp.listdir_attr.return_value = []

            sentinel_state = [None]  # None = absent, "" = present

            def exec_side_effect(cmd, **kwargs):
                if "test -f" in cmd and SlurmJobManagerCERN.SENTINEL_FILENAME in cmd:
                    return sentinel_state[0]
                if "touch" in cmd and SlurmJobManagerCERN.SENTINEL_FILENAME in cmd:
                    sentinel_state[0] = ""  # sentinel now exists
                    return ""
                return ""  # pwd, mkdir, etc.

            mgr.slurm_connection.exec_command.side_effect = exec_side_effect
            mgr.slurm_connection.ssh_client.open_sftp.return_value = real_sftp
            return mgr, sentinel_state

        mgr1, sentinel_state = make_mgr()
        mgr2, _ = make_mgr()
        # Share the same sentinel state between both managers
        def exec_side_effect_shared(cmd, **kwargs):
            if "test -f" in cmd and SlurmJobManagerCERN.SENTINEL_FILENAME in cmd:
                return sentinel_state[0]
            if "touch" in cmd and SlurmJobManagerCERN.SENTINEL_FILENAME in cmd:
                sentinel_state[0] = ""
                return ""
            return ""
        mgr2.slurm_connection.exec_command.side_effect = exec_side_effect_shared

        results = []
        errors = []

        def run(mgr):
            try:
                mgr._transfer_inputs()
                results.append("ok")
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=run, args=(mgr1,))
        t2 = threading.Thread(target=run, args=(mgr2,))
        t1.start(); t2.start()
        t1.join(); t2.join()

        assert not errors, errors
        assert len(results) == 2
        # SFTP put should have been called exactly once (one file, one upload)
        assert sftp_put_count[0] == 1


class TestStop:
    """Tests for SlurmJobManagerCERN.stop()."""

    def test_stop_calls_scancel_with_job_id(self):
        """stop() SSHs to head node and runs scancel <backend_job_id>."""
        with patch("reana_job_controller.slurmcern_job_manager.SSHClient") as mock_ssh_cls:
            mock_conn = MagicMock()
            mock_ssh_cls.return_value = mock_conn
            from reana_job_controller.slurmcern_job_manager import SlurmJobManagerCERN
            SlurmJobManagerCERN.stop("12345")
            mock_conn.exec_command.assert_called_once_with("scancel 12345")

    def test_stop_logs_error_on_ssh_failure(self):
        """stop() logs but does not raise when SSH connection fails."""
        with patch("reana_job_controller.slurmcern_job_manager.SSHClient") as mock_ssh_cls:
            mock_ssh_cls.side_effect = Exception("Connection refused")
            from reana_job_controller.slurmcern_job_manager import SlurmJobManagerCERN
            # Should not raise
            SlurmJobManagerCERN.stop("12345")
