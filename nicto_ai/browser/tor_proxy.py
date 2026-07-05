"""
NICTO AI - Tor Proxy Integration
Routes browser traffic through the Tor network for anonymous browsing
"""

import asyncio
import logging
import shutil
import subprocess
import time
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class TorProxy:
    """
    Manages a Tor SOCKS5 proxy for anonymous browsing.

    Lifecycle:
    1. start() - Launches tor process (if installed) and waits for bootstrap
    2. get_proxy_url() - Returns socks5://127.0.0.1:9050 for browser routing
    3. new_identity() - Requests a new Tor circuit (new IP address)
    4. stop() - Shuts down the tor process

    Works by launching the system tor binary and waiting for it to
    establish a SOCKS5 proxy on port 9050.
    """

    def __init__(
        self,
        socks_port: int = 9050,
        control_port: int = 9051,
        data_dir: Optional[str] = None,
    ):
        self.socks_port = socks_port
        self.control_port = control_port
        self.data_dir = Path(data_dir) if data_dir else None
        self._process: Optional[subprocess.Popen] = None
        self._tor_path: Optional[str] = None

    def find_tor(self) -> Optional[str]:
        """Find the tor binary on the system."""
        # Check common locations
        candidates = [
            "tor",
            "tor.exe",
            r"C:\Users\BYU\Desktop\NICTO\tor\tor.exe",
            r"C:\Program Files\Tor Browser\Tor\tor.exe",
            r"C:\Program Files (x86)\Tor Browser\Tor\tor.exe",
            "/usr/bin/tor",
            "/usr/local/bin/tor",
        ]

        for path in candidates:
            if shutil.which(path) or Path(path).exists():
                self._tor_path = path
                logger.info("Found tor at: %s", path)
                return path

        logger.warning("tor not found on system")
        return None

    def start(self, timeout: int = 30) -> bool:
        """
        Start the Tor proxy.

        Args:
            timeout: Seconds to wait for bootstrap

        Returns:
            True if Tor started successfully
        """
        tor_path = self.find_tor()
        if not tor_path:
            logger.error("Cannot start Tor: binary not found")
            return False

        cmd = [
            tor_path,
            "--SocksPort", str(self.socks_port),
            "--DataDirectory", str(self.data_dir or Path.home() / ".nicto_tor"),
        ]

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except Exception as e:
            logger.error("Failed to start tor: %s", e)
            return False

        # Wait for bootstrap
        start = time.time()
        while time.time() - start < timeout:
            if self._process.poll() is not None:
                stderr = self._process.stderr.read().decode()
                logger.error("Tor exited with code %d: %s", self._process.returncode, stderr[:500])
                return False

            if self._is_ready():
                logger.info("Tor proxy ready on port %d", self.socks_port)
                return True

            time.sleep(0.5)

        logger.error("Tor bootstrap timed out after %ds", timeout)
        self.stop()
        return False

    def stop(self):
        """Stop the Tor proxy."""
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
            logger.info("Tor proxy stopped")

    def new_identity(self) -> bool:
        """
        Request a new Tor circuit (new exit node / IP address).

        Uses the Tor control protocol to signal NEWNYM.
        Requires cookie authentication or password.
        """
        try:
            from stem import Signal
            from stem.control import Controller

            with Controller.from_port(port=self.control_port) as controller:
                controller.authenticate()
                controller.signal(Signal.NEWNYM)
                logger.info("New Tor identity requested")
                return True
        except Exception as e:
            logger.warning("Failed to get new identity: %s (control port may not be available)", e)
            return False

    def get_proxy_url(self) -> str:
        """Get the SOCKS5 proxy URL for browser routing."""
        return f"socks5://127.0.0.1:{self.socks_port}"

    def _is_ready(self) -> bool:
        """Check if the SOCKS5 port is accepting connections."""
        import socket
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(("127.0.0.1", self.socks_port))
            sock.close()
            return result == 0
        except Exception:
            return False

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
