"""
NICTO AI - Hash Tool
Compute cryptographic hashes and checksums.
"""

import hashlib
import zlib
import logging
from typing import Dict
from .base import Tool, ToolResult, ToolParameter

logger = logging.getLogger(__name__)


class HashTool(Tool):
    """
    Cryptographic hash and checksum engine.

    Supports: MD5, SHA-1, SHA-256, SHA-512, Blake2b, CRC32, Adler32
    """

    name = "hash_tool"
    description = "Compute hashes: MD5, SHA-1, SHA-256, SHA-512, Blake2b, CRC32. Supports file path or text input."
    parameters = [
        ToolParameter(name="data", type="string", description="Text to hash or file path", required=True),
        ToolParameter(name="algorithm", type="string", description="Hash algorithm", required=False, default="sha256", enum=[
            "md5", "sha1", "sha256", "sha512", "blake2b", "crc32", "adler32",
        ]),
        ToolParameter(name="input_type", type="string", description="Input type", required=False, default="text", enum=["text", "file"]),
    ]
    tags = ["hash", "crypto", "checksum", "security"]
    timeout_seconds = 10.0

    def _execute(self, data: str, algorithm: str = "sha256", input_type: str = "text") -> ToolResult:
        try:
            if input_type == "file":
                import os
                if not os.path.exists(data):
                    return ToolResult(success=False, error=f"File not found: {data}")
                with open(data, "rb") as f:
                    content = f.read()
            else:
                content = data.encode("utf-8")

            result = self._compute_hash(content, algorithm)

            output = {
                "algorithm": algorithm,
                "hash": result,
                "input_type": input_type,
            }
            if algorithm in ("crc32", "adler32"):
                output["hash_decimal"] = int(result, 16)

            return ToolResult(success=True, output=output)
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _compute_hash(self, content: bytes, algorithm: str) -> str:
        if algorithm == "md5":
            return hashlib.md5(content).hexdigest()
        elif algorithm == "sha1":
            return hashlib.sha1(content).hexdigest()
        elif algorithm == "sha256":
            return hashlib.sha256(content).hexdigest()
        elif algorithm == "sha512":
            return hashlib.sha512(content).hexdigest()
        elif algorithm == "blake2b":
            return hashlib.blake2b(content).hexdigest()
        elif algorithm == "crc32":
            return format(zlib.crc32(content) & 0xFFFFFFFF, '08x')
        elif algorithm == "adler32":
            return format(zlib.adler32(content) & 0xFFFFFFFF, '08x')
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
