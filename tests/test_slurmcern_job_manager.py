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
