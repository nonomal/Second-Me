import os
import subprocess
import sys
from typing import List, Optional, Dict, Any
from lpm_kernel.common.logging import logger

# Test if logging works properly
logger.debug("DEBUG: ScriptExecutor module loaded")
logger.info("INFO: ScriptExecutor module loaded")
logger.warning("WARNING: ScriptExecutor module loaded")
logger.error("ERROR: ScriptExecutor module loaded")


class ScriptExecutor:
    def __init__(self):
        self.environment = self._detect_environment()

    def _detect_environment(self) -> Dict[str, str]:
        """
        Detect the current execution environment (conda, docker, system)
        
        Returns:
            Dict with environment type and details
        """
        env_info = {
            "type": "system",
            "details": "unknown"
        }
        
        # Check if in conda environment
        conda_env = os.getenv("CONDA_DEFAULT_ENV")
        if conda_env:
            env_info["type"] = "conda"
            env_info["details"] = conda_env
            return env_info
        
        # Check if in docker environment
        if os.path.exists("/.dockerenv"):
            env_info["type"] = "docker"
            try:
                with open("/proc/self/cgroup", "r") as f:
                    for line in f:
                        if "docker" in line:
                            container_id = line.split("/")[-1].strip()
                            env_info["details"] = f"container:{container_id}"
                            break
            except:
                env_info["details"] = "unknown-container"
            return env_info
            
        # Regular system environment
        try:
            import platform
            env_info["details"] = platform.platform()
        except:
            pass
            
        return env_info

    def execute(
        self,
        script_path: str,
        script_type: str,
        args: Optional[List[str]] = None,
        shell: bool = False,
        log_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute scripts in the appropriate environment (conda, docker, or system)

        Args:
            script_path: Script path or command
            script_type: Script type, used for logging
            args: Command line arguments
            shell: Whether to use shell for execution
            log_file: Log file path, if provided will redirect output to this file

        Returns:
            Execution result
        """
        try:
            # Build the complete command based on environment
            if script_path.endswith(".py"):
                # Python script execution varies by environment
                if self.environment["type"] == "conda":
                    cmd = [
                        "conda",
                        "run",
                        "-n",
                        self.environment["details"],
                        "python",
                        script_path,
                    ]
                else:  # docker or system
                    cmd = [
                        "python",
                        script_path,
                    ]
            elif script_path.endswith(".sh"):
                # Shell script execution varies by environment
                if self.environment["type"] == "conda":
                    cmd = [
                        "conda",
                        "run",
                        "-n",
                        self.environment["details"],
                        "bash",
                        script_path,
                    ]
                else:  # docker or system
                    cmd = [
                        "bash",
                        script_path,
                    ]
            else:
                # Other commands
                if self.environment["type"] == "conda":
                    cmd = [
                        "conda",
                        "run",
                        "-n",
                        self.environment["details"],
                        script_path,
                    ]
                else:  # docker or system
                    cmd = [
                        script_path,
                    ]

            # Add additional parameters
            if args:
                cmd.extend(args)

            logger.info(f"Executing command: {' '.join(cmd)}")

            # If logging is needed, ensure the log directory exists
            if log_file:
                os.makedirs(os.path.dirname(log_file), exist_ok=True)

            # Execute command and capture output
            process = subprocess.Popen(
                cmd,
                shell=shell,
                env=os.environ.copy(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Redirect error output to standard output
                bufsize=1,  # Line buffering
                universal_newlines=True,  # Use text mode
            )

            # Read and process output in real-time
            while True:
                output = process.stdout.readline()
                if output == "" and process.poll() is not None:
                    break
                if output:
                    # Output to console
                    print(output.strip())
                    # If needed, write to log file
                    if log_file:
                        with open(log_file, "a", encoding="utf-8") as f:
                            f.write(output)

            # Wait for the process to end and get the return code
            return_code = process.wait()

            if return_code != 0:
                logger.error(f"Command execution failed, return code: {return_code}")
            else:
                logger.info(f"Command execution successful, return code: {return_code}")

            return {
                "returncode": return_code,
                "error": f"Execution failed, return code: {return_code}"
                if return_code != 0
                else None,
            }

        except Exception as e:
            error_msg = f"Error occurred while executing {script_type} script: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg, "returncode": -1}
