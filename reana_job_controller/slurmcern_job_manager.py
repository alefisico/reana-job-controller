# This file is part of REANA.
# Copyright (C) 2019, 2020, 2021, 2022, 2023 CERN.
#
# REANA is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.

"""CERN Slurm Job Manager."""

import base64
import logging
import os
import re
import tempfile
import threading
import uuid
from stat import S_ISDIR

from reana_commons.config import REANA_USER_SECRET_MOUNT_PATH
from reana_commons.k8s.secrets import UserSecrets
from reana_job_controller.job_manager import JobManager
from reana_job_controller.utils import SSHClient, initialize_krb5_token
from reana_job_controller.config import (
    SLURM_HEADNODE_HOSTNAME,
    SLURM_HEADNODE_PORT,
    SLURM_PARTITION,
    SLURM_QOS,
    SLURM_JOB_TIMELIMIT,
    SLURM_SSH_TIMEOUT,
    SLURM_SSH_BANNER_TIMEOUT,
    SLURM_SSH_AUTH_TIMEOUT,
)


class SlurmJobManagerCERN(JobManager):
    """Slurm job management."""

    SLURM_HOME_PATH = os.getenv("SLURM_HOME_PATH", "")
    """Default SLURM home path."""
    _upload_locks = {}
    _upload_locks_mutex = threading.Lock()
    SENTINEL_FILENAME = ".reana_upload_complete"

    def __init__(
        self,
        docker_img=None,
        cmd=None,
        prettified_cmd=None,
        env_vars=None,
        workflow_uuid=None,
        workflow_workspace=None,
        cvmfs_mounts="false",
        shared_file_system=False,
        job_name=None,
        slurm_partition=SLURM_PARTITION,
        slurm_qos=SLURM_QOS,
        slurm_job_timelimit=SLURM_JOB_TIMELIMIT,
        slurm_cpus=None,
        slurm_mem=None,
        slurm_nodes=None,
        slurm_ntasks=None,
        slurm_gres=None,
        secrets: UserSecrets = None,
        **kwargs,
    ):
        """Instanciate Slurm job manager.

        :param docker_img: Docker image.
        :type docker_img: str
        :param cmd: Command to execute.
        :type cmd: list
        :param prettified_cmd: pretified version of command to execute.
        :type prettified_cmd: str
        :param env_vars: Environment variables.
        :type env_vars: dict
        :param workflow_uuid: Unique workflow id.
        :type workflow_uuid: str
        :param workflow_workspace: Workflow workspace path.
        :type workflow_workspace: str
        :param cvmfs_mounts: list of CVMFS mounts as a string.
        :type cvmfs_mounts: str
        :param shared_file_system: if shared file system is available.
        :type shared_file_system: bool
        :param job_name: Name of the job
        :type job_name: str
        :param slurm_partition: Partition of a Slurm job.
        :type slurm_partition: str
        :param slurm_job_timelimit: Maximum timelimit of a Slurm job.
        :type slurm_job_timelimit: str
        :param secrets: User secrets (file and env types).
        :type secrets: UserSecrets
        """
        super(SlurmJobManagerCERN, self).__init__(
            docker_img=docker_img,
            cmd=cmd,
            prettified_cmd=prettified_cmd,
            env_vars=env_vars,
            workflow_uuid=workflow_uuid,
            job_name=job_name,
        )
        self.compute_backend = "Slurm"
        self.workflow_workspace = workflow_workspace
        self.cvmfs_mounts = cvmfs_mounts
        self.shared_file_system = shared_file_system
        self.partition = slurm_partition
        self.qos = slurm_qos
        self.timelimit = slurm_job_timelimit
        self.cpus = slurm_cpus
        self.mem = slurm_mem
        self.nodes = slurm_nodes
        self.ntasks = slurm_ntasks
        self.gres = slurm_gres
        self.secrets = secrets
        self.img_type_docker = self._is_img_type_docker()
        self.slurm_workspace_path = ""
        self.reana_workspace_path = ""
        self.slurm_home_path = ""
        # Use a unique suffix per job instance to avoid file collisions
        # when multiple jobs from the same workflow run concurrently.
        job_suffix = str(uuid.uuid4())[:8]
        self.job_file = "job_{}.sh".format(job_suffix)
        self.job_description_file = "job_description_{}.sh".format(job_suffix)

    def _transfer_inputs(self):
        """Transfer inputs to SLURM submit node.

        Uses a per-workspace lock so that only one thread uploads when multiple
        jobs from the same workflow are submitted concurrently. The first thread
        to acquire the lock performs the full upload; subsequent threads skip
        the upload because the sentinel written by the first thread tells them
        the workspace is already complete.
        """
        workspace_key = self.workflow_workspace
        with SlurmJobManagerCERN._upload_locks_mutex:
            if workspace_key not in SlurmJobManagerCERN._upload_locks:
                SlurmJobManagerCERN._upload_locks[workspace_key] = threading.Lock()
        lock = SlurmJobManagerCERN._upload_locks[workspace_key]

        with lock:
            for attempt in range(2):
                try:
                    self._do_transfer_inputs()
                    return
                except Exception as e:
                    if attempt == 0:
                        logging.warning(
                            "Input transfer failed ({}), reconnecting and retrying.".format(e)
                        )
                        self.slurm_connection.establish_connection()
                    else:
                        raise

    def _do_transfer_inputs(self):
        """Perform the actual SFTP transfer of inputs to the SLURM submit node.

        Skips the upload entirely when a sentinel file already exists on the
        remote workspace, meaning a previous job from this workflow already
        transferred the inputs. Writes the sentinel after a successful upload.
        """
        stdout = self.slurm_connection.exec_command("pwd")
        self.slurm_home_path = SlurmJobManagerCERN.SLURM_HOME_PATH or stdout.rstrip()
        self.slurm_workspace_path = os.path.join(
            self.slurm_home_path, self.workflow_workspace[1:]
        )
        self.reana_workspace_path = self.workflow_workspace
        self.slurm_connection.exec_command(
            "mkdir -p {}".format(self.slurm_workspace_path)
        )
        sentinel_path = os.path.join(
            self.slurm_workspace_path, SlurmJobManagerCERN.SENTINEL_FILENAME
        )
        try:
            self.slurm_connection.exec_command('test -f "{}"'.format(sentinel_path))
            logging.info(
                "Input upload sentinel found — skipping transfer for workspace %s",
                self.slurm_workspace_path,
            )
            return
        except Exception:
            pass  # sentinel absent (exit 1 = file not found), proceed with upload
        sftp = self.slurm_connection.ssh_client.open_sftp()
        sftp.get_channel().settimeout(600)
        os.chdir(self.workflow_workspace)
        for dirpath, dirnames, filenames in os.walk(self.workflow_workspace):
            try:
                sftp.mkdir(os.path.join(self.slurm_home_path, dirpath[1:]))
            except Exception:
                pass
            for file in filenames:
                remote_path = os.path.join(self.slurm_home_path, dirpath[1:], file)
                try:
                    sftp.chmod(remote_path, 0o664)
                except IOError:
                    pass  # file doesn't exist yet, chmod not needed
                sftp.put(os.path.join(dirpath, file), remote_path)
        self._transfer_secrets(sftp)
        sftp.close()
        self.slurm_connection.exec_command(
            'touch "{}"'.format(sentinel_path)
        )

    def _transfer_secrets(self, sftp):
        """Transfer file-type user secrets to Slurm head node."""
        if not self.secrets:
            return
        file_secrets = [s for s in self.secrets.get_secrets() if s.type_ == "file"]
        if not file_secrets:
            return
        secrets_remote_dir = os.path.join(
            self.slurm_workspace_path, "reana_secrets"
        )
        try:
            sftp.mkdir(secrets_remote_dir)
        except Exception:
            pass
        with tempfile.TemporaryDirectory() as tmpdir:
            for secret in file_secrets:
                local_path = os.path.join(tmpdir, secret.name)
                with open(local_path, "wb") as f:
                    f.write(secret.value_bytes)
                sftp.put(local_path, os.path.join(secrets_remote_dir, secret.name))

    @JobManager.execution_hook
    def execute(self):
        """Execute / submit a job with Slurm."""
        self.cmd = self._encode_cmd(self.cmd)
        if not os.getenv("SLURM_SKIP_KRB5", "false").lower() == "true":
            initialize_krb5_token(workflow_uuid=self.workflow_uuid)
        self.slurm_connection = SSHClient(
            hostname=SLURM_HEADNODE_HOSTNAME,
            port=SLURM_HEADNODE_PORT,
            timeout=SLURM_SSH_TIMEOUT,
            banner_timeout=SLURM_SSH_BANNER_TIMEOUT,
            auth_timeout=SLURM_SSH_AUTH_TIMEOUT,
        )
        self._transfer_inputs()
        self._pull_image()
        self._dump_job_file()
        self._dump_job_submission_file()
        backend_job_id = self._execute_sbatch()
        return backend_job_id

    def _execute_sbatch(self):
        """Submit job_description_file via sbatch and return the Slurm job ID."""
        stdout = self.slurm_connection.exec_command(
            "cd {} && sbatch --parsable {}".format(
                self.slurm_workspace_path, self.job_description_file
            )
        )
        if stdout is None:
            raise RuntimeError(
                "sbatch returned no output — SSH connection to Slurm head node failed"
            )
        return stdout.rstrip()

    def _is_img_type_docker(self):
        if not self.docker_img:
            return False
        return not any(img_type in self.docker_img for img_type in [".sif", "cvmfs"])

    def _pull_image(self):
        """Pull a Docker image using Singularity, skip if .sif already exists."""
        if not self.img_type_docker:
            return
        sif_path = self._sif_path()
        result = self.slurm_connection.exec_command(f'test -f "{sif_path}"')
        if result is not None:
            return  # file exists, skip pull
        self.slurm_connection.exec_command(
            f"cd {self.slurm_workspace_path} && singularity pull docker://{self.docker_img}"
        )

    def _get_container(self):
        """Get container image."""
        if self.img_type_docker:
            return self.docker_img.split("/")[-1].replace(":", "_") + ".sif"
        return self.docker_img

    def _sif_path(self):
        """Return the absolute path of the .sif file on the Slurm head node."""
        return os.path.join(self.slurm_workspace_path, self._get_container())

    def _has_voms_secrets(self):
        """Return True if all required voms-proxy secrets are present."""
        if not self.secrets:
            return False
        names = {s.name for s in self.secrets.get_secrets()}
        return {"usercert.pem", "userkey.pem", "VOMSPROXY_PASS"}.issubset(names)

    def _voms_proxy_path(self):
        """Return the host-side path where the voms proxy will be written.

        Uses job_file stem as a unique suffix so concurrent jobs in the same
        workflow don't clobber each other's proxy files.
        """
        suffix = self.job_file.replace("job_", "").replace(".sh", "")
        return os.path.join(
            self.slurm_workspace_path, "voms_proxy_{}.pem".format(suffix)
        )

    def _voms_proxy_init_cmd(self):
        """Return bash snippet that generates a voms proxy on the Slurm worker.

        Sets REANA_VOMS_PROXY_BIND and REANA_VOMS_PROXY_ENV shell variables
        that are used by _wrap_singularity_cmd() only when the proxy file was
        successfully created — so a failed voms-proxy-init degrades gracefully
        instead of crashing the Singularity mount.
        """
        if not self._has_voms_secrets():
            return ""
        secrets_dir = os.path.join(
            self.slurm_workspace_path, "reana_secrets"
        )
        voname = ""
        if self.secrets:
            voname_secret = self.secrets.get_secret("VONAME")
            if voname_secret:
                voname = voname_secret.value_str.strip().lower()
        suffix = self.job_file.replace("job_", "").replace(".sh", "")
        userkey_tmp = os.path.join(
            self.slurm_workspace_path, "userkey_{}.pem".format(suffix)
        )
        return (
            "cp {secrets_dir}/userkey.pem {userkey_tmp}\n"
            "chmod 400 {userkey_tmp}\n"
            "echo $VOMSPROXY_PASS | base64 -d | voms-proxy-init"
            " --voms {voname}"
            " --key {userkey_tmp}"
            " --cert {secrets_dir}/usercert.pem"
            " --pwstdin"
            " --out {proxy_path}\n"
            "rm -f {userkey_tmp}\n"
            "if [ -f {proxy_path} ]; then\n"
            "  REANA_VOMS_PROXY_BIND=-B\\ {proxy_path}:/tmp/voms_proxy.pem\n"
            "  REANA_VOMS_PROXY_ENV=--env\\ X509_USER_PROXY=/tmp/voms_proxy.pem\n"
            "fi\n"
        ).format(
            secrets_dir=secrets_dir,
            userkey_tmp=userkey_tmp,
            voname=voname,
            proxy_path=self._voms_proxy_path(),
        )

    def _env_secrets_exports(self):
        """Return export statements for env-type secrets."""
        if not self.secrets:
            return ""
        exports = []
        for secret in self.secrets.get_secrets():
            if secret.type_ == "env":
                exports.append(
                    "export {}={}\n".format(secret.name, secret.value_str)
                )
        return "".join(exports)

    def _dump_job_submission_file(self):
        """Dump job submission file to the Slurm submit node."""
        safe_job_name = re.sub(r"[^\w\-.]", "_", self.job_name)
        qos_line = "#SBATCH --qos {qos} \n".format(qos=self.qos) if self.qos else ""
        cpus_line = "#SBATCH --cpus-per-task {cpus} \n".format(cpus=self.cpus) if self.cpus else ""
        mem_line = "#SBATCH --mem {mem}M \n".format(mem=self.mem) if self.mem else ""
        nodes_line = "#SBATCH --nodes {nodes} \n".format(nodes=self.nodes) if self.nodes else ""
        ntasks_line = "#SBATCH --ntasks {ntasks} \n".format(ntasks=self.ntasks) if self.ntasks else ""
        gres_line = "#SBATCH --gres {gres} \n".format(gres=self.gres) if self.gres else ""
        job_template = (
            "#!/bin/bash \n"
            "#SBATCH --job-name={job_name} \n"
            "#SBATCH --output=reana_job.%j.out \n"
            "#SBATCH --error=reana_job.%j.err \n"
            "#SBATCH --partition {partition} \n"
            "{qos_line}"
            "#SBATCH --time {time} \n"
            "{cpus_line}"
            "{mem_line}"
            "{nodes_line}"
            "{ntasks_line}"
            "{gres_line}"
            "export PATH=$PATH:/usr/sbin \n"
            "{env_secrets}"
            "{voms_proxy_init}"
            "{cvmfs_premount}"
            "srun {command}"
        ).format(
            partition=self.partition,
            qos_line=qos_line,
            time=self.timelimit,
            cpus_line=cpus_line,
            mem_line=mem_line,
            nodes_line=nodes_line,
            ntasks_line=ntasks_line,
            gres_line=gres_line,
            job_name=safe_job_name,
            env_secrets=self._env_secrets_exports(),
            voms_proxy_init=self._voms_proxy_init_cmd(),
            cvmfs_premount=self._cvmfs_premount_cmd(),
            command=self._wrap_singularity_cmd(),
        )
        desc_path = os.path.join(self.slurm_workspace_path, self.job_description_file)
        sftp = self.slurm_connection.ssh_client.open_sftp()
        try:
            with sftp.open(desc_path, "w") as f:
                f.write(job_template)
        finally:
            sftp.close()

    def _dump_job_file(self):
        """Dump job file."""
        cmd = self.cmd
        if not self.docker_img and self.slurm_workspace_path:
            # Without a container there is no bind-mount to remap the REANA
            # workspace path onto the Slurm node.  Decode, replace the cd, and
            # re-encode so bare jobs land in the NFS-accessible path on any
            # worker node (e.g. rogue01 which has no /opt/reana/users).
            import base64 as _b64
            decoded = _b64.b64decode(
                cmd.removeprefix("echo ").removesuffix("|base64 -d|bash")
            ).decode("utf-8")
            decoded = decoded.replace(
                "cd {}".format(self.workflow_workspace),
                "cd {}".format(self.slurm_workspace_path),
                1,
            )
            encoded = _b64.b64encode(decoded.encode("utf-8")).decode("utf-8")
            cmd = "echo {}|base64 -d|bash".format(encoded)
        job_content = "#!/bin/bash \n{}".format(cmd)
        job_path = os.path.join(self.slurm_workspace_path, self.job_file)
        # Write via SFTP to avoid shell quoting issues with long base64 strings.
        sftp = self.slurm_connection.ssh_client.open_sftp()
        try:
            with sftp.open(job_path, "w") as f:
                f.write(job_content)
            sftp.chmod(job_path, 0o755)
        finally:
            sftp.close()
        # Verify the file was actually written before submitting.
        # NFS attribute cache on falcon can delay visibility of a just-written file
        # for up to ~60 s; retry a few times before declaring failure.
        import time as _time
        for _attempt in range(6):
            result = self.slurm_connection.exec_command('test -f "{}"'.format(job_path))
            if result is not None:
                break
            _time.sleep(10)
        else:
            raise RuntimeError(
                "job file {} was not created on the Slurm head node".format(job_path)
            )

    def _encode_cmd(self, cmd):
        """Encode base64 cmd."""
        encoded_cmd = base64.b64encode(cmd.encode("utf-8")).decode("utf-8")
        return "echo {}|base64 -d|bash".format(encoded_cmd)

    def _secrets_bind_mount(self):
        """Return Singularity -B flag for the secrets directory if file secrets exist."""
        if not self.secrets:
            return ""
        file_secrets = [s for s in self.secrets.get_secrets() if s.type_ == "file"]
        if not file_secrets:
            return ""
        secrets_remote_dir = os.path.join(
            self.slurm_workspace_path, "reana_secrets"
        )
        return " -B {}:{}:ro".format(secrets_remote_dir, REANA_USER_SECRET_MOUNT_PATH)

    def _cvmfs_bind_mounts(self):
        """Return Singularity -B flags for CVMFS repositories.

        Parses self.cvmfs_mounts (a stringified list of repo names, e.g.
        "['atlas.cern.ch', 'sft.cern.ch']") and returns a bind-mount flag
        for each repository path under /cvmfs.  Falls back to binding the
        entire /cvmfs tree when the value cannot be parsed as a list.
        Returns an empty string when CVMFS mounting is disabled ("false" or "").
        """
        import ast

        if not self.cvmfs_mounts or self.cvmfs_mounts == "false":
            return ""
        try:
            repos = ast.literal_eval(self.cvmfs_mounts)
            return "".join(" -B /cvmfs/{}".format(r) for r in repos)
        except (ValueError, SyntaxError):
            return " -B /cvmfs"

    def _cvmfs_premount_cmd(self):
        """Return a shell line that triggers autofs to mount CVMFS repos before Singularity snapshots /cvmfs.

        Singularity bind-mounts /cvmfs at exec time; repos not yet mounted by
        autofs at that moment are invisible inside the container.  Accessing
        each repo path forces autofs to mount it first.
        """
        if not self.cvmfs_mounts or self.cvmfs_mounts == "false":
            return ""
        import ast
        try:
            repos = ast.literal_eval(self.cvmfs_mounts)
            # Known repo list: touch each path directly.
            paths = " ".join("/cvmfs/{}".format(r) for r in repos)
            return "ls {} > /dev/null 2>&1 || true\n".format(paths)
        except (ValueError, SyntaxError):
            # cvmfs_mounts is "true" or unparseable — read CVMFS_REPOSITORIES from
            # the node's config and touch each repo so autofs mounts them before
            # Singularity snapshots /cvmfs.  "ls /cvmfs" alone only lists the
            # autofs root and does not trigger lazy mounts of individual repos.
            return (
                "for _r in $(grep CVMFS_REPOSITORIES /etc/cvmfs/default.local"
                " | cut -d= -f2 | tr ',' ' '); do"
                " ls /cvmfs/$_r > /dev/null 2>&1; done || true\n"
            )

    def _wrap_singularity_cmd(self):
        """Wrap command in Singularity, or run natively if no container image.

        When voms secrets are present, $REANA_VOMS_PROXY_BIND and
        $REANA_VOMS_PROXY_ENV shell variables are expanded at runtime —
        set by _voms_proxy_init_cmd() only if voms-proxy-init succeeded.
        """
        if not self.docker_img:
            return "./" + self.job_file
        voms_args = " $REANA_VOMS_PROXY_BIND $REANA_VOMS_PROXY_ENV" if self._has_voms_secrets() else ""
        return (
            "singularity exec -B {SLURM_WORKSAPCE}:{REANA_WORKSPACE}"
            "{SECRETS_BIND}"
            "{CVMFS_BIND}"
            "{VOMS_ARGS}"
            " {IMAGE} {CMD}".format(
                SLURM_WORKSAPCE=self.slurm_workspace_path,
                REANA_WORKSPACE=self.reana_workspace_path,
                SECRETS_BIND=self._secrets_bind_mount(),
                CVMFS_BIND=self._cvmfs_bind_mounts(),
                VOMS_ARGS=voms_args,
                IMAGE=self._get_container(),
                CMD="./" + self.job_file,
            )
        )

    @classmethod
    def get_outputs(cls, workspace=None, **kwargs):
        """Transfer job outputs from the Slurm head node back to the REANA workspace.

        When SLURM_HOME_PATH differs from the REANA workspace root (i.e. the Slurm
        head node does not share a filesystem with the K8s control plane), output
        files written by the job must be copied back via SFTP so that the Snakemake
        engine can see them.  If the two roots are the same filesystem this is a
        no-op because the files are already in place.

        :param workspace: Absolute path to the REANA workflow workspace on the
            control-plane node (e.g. ``/opt/reana/users/<uid>/workflows/<wid>``).
        :type workspace: str
        """
        if not workspace:
            return
        try:
            slurm_connection = SSHClient(
                hostname=SLURM_HEADNODE_HOSTNAME,
                port=SLURM_HEADNODE_PORT,
                timeout=SLURM_SSH_TIMEOUT,
                banner_timeout=SLURM_SSH_BANNER_TIMEOUT,
                auth_timeout=SLURM_SSH_AUTH_TIMEOUT,
            )
            slurm_home = (
                cls.SLURM_HOME_PATH
                or slurm_connection.exec_command("pwd").rstrip()
            )
            slurm_workspace = os.path.join(slurm_home, workspace.lstrip("/"))
            # Skip transfer when the Slurm workspace IS the REANA workspace
            # (i.e. shared filesystem — same path resolves on both sides).
            if os.path.realpath(slurm_workspace) == os.path.realpath(workspace):
                return
            sftp = slurm_connection.ssh_client.open_sftp()
            sftp.get_channel().settimeout(600)
            SlurmJobManagerCERN._download_dir(sftp, slurm_workspace, workspace)
            sftp.close()
            logging.info(
                "Transferred outputs from %s to %s", slurm_workspace, workspace
            )
        except Exception as e:
            logging.error(
                "Failed to transfer outputs from Slurm workspace: %s", e,
                exc_info=True,
            )

    @staticmethod
    def _download_dir(sftp, remote_dir, local_dir):
        """Recursively download a remote directory via SFTP."""
        os.path.exists(local_dir) or os.makedirs(local_dir)
        dir_items = sftp.listdir_attr(remote_dir)
        for item in dir_items:
            remote_path = os.path.join(remote_dir, item.filename)
            local_path = os.path.join(local_dir, item.filename)
            if S_ISDIR(item.st_mode):
                SlurmJobManagerCERN._download_dir(sftp, remote_path, local_path)
            else:
                sftp.get(remote_path, local_path)

    @classmethod
    def get_logs(cls, backend_job_id, **kwargs):
        """Return job logs by reading them from the Slurm head node via SSH.

        :param backend_job_id: ID of the job in the backend.
        :param kwargs: Additional parameters needed to fetch logs.
            In the case of Slurm, the ``workspace`` parameter is needed.
        :return: String containing the job logs.
        """
        if "workspace" not in kwargs:
            raise ValueError("Missing 'workspace' parameter")
        workspace = kwargs["workspace"]

        try:
            slurm_connection = SSHClient(
                hostname=SLURM_HEADNODE_HOSTNAME,
                port=SLURM_HEADNODE_PORT,
                timeout=SLURM_SSH_TIMEOUT,
                banner_timeout=SLURM_SSH_BANNER_TIMEOUT,
                auth_timeout=SLURM_SSH_AUTH_TIMEOUT,
            )
            slurm_home = cls.SLURM_HOME_PATH or slurm_connection.exec_command("pwd").rstrip()
            slurm_workspace = os.path.join(slurm_home, workspace.lstrip("/"))
            stderr_file = os.path.join(
                slurm_workspace, "reana_job." + str(backend_job_id) + ".err"
            )
            stdout_file = os.path.join(
                slurm_workspace, "reana_job." + str(backend_job_id) + ".out"
            )
            job_log = ""
            for log_file in [stderr_file, stdout_file]:
                job_log += slurm_connection.exec_command(
                    "cat {}".format(log_file)
                )
            return job_log
        except Exception as e:
            msg = "Job logs of {} were not found. {}".format(backend_job_id, e)
            logging.error(msg, exc_info=True)
            return msg

    def stop(backend_job_id):
        """Stop Slurm job execution by running scancel via SSH.

        :param backend_job_id: Slurm job ID to cancel.
        """
        try:
            slurm_connection = SSHClient(
                hostname=SLURM_HEADNODE_HOSTNAME,
                port=SLURM_HEADNODE_PORT,
                timeout=SLURM_SSH_TIMEOUT,
                banner_timeout=SLURM_SSH_BANNER_TIMEOUT,
                auth_timeout=SLURM_SSH_AUTH_TIMEOUT,
            )
            output = slurm_connection.exec_command("scancel {}".format(backend_job_id))
            if output:
                logging.debug(
                    "scancel output for job {}: {}".format(backend_job_id, output)
                )
        except Exception as e:
            logging.error(
                "Failed to stop Slurm job {}: {}".format(backend_job_id, e),
                exc_info=True,
            )
