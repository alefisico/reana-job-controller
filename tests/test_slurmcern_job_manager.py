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
        """Singularity cmd uses shell variables for proxy bind/env, not literal paths."""
        mgr = _make_manager_with_secrets("docker.io/org/img:v1", _make_voms_secrets())
        cmd = mgr._wrap_singularity_cmd()
        assert "$REANA_VOMS_PROXY_BIND" in cmd
        assert "$REANA_VOMS_PROXY_ENV" in cmd

    def test_voms_proxy_shell_vars_absent_without_voms_secrets(self):
        """Shell variable placeholders not added when voms secrets are absent."""
        mgr = _make_manager_with_secrets(
            "docker.io/org/img:v1",
            _make_user_secrets(env_secrets={"OTHER_VAR": "val"}),
        )
        cmd = mgr._wrap_singularity_cmd()
        assert "$REANA_VOMS_PROXY_BIND" not in cmd
        assert "$REANA_VOMS_PROXY_ENV" not in cmd

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
