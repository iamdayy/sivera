"""File storage abstraction (local filesystem / S3 compatible)."""
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO, Optional
from uuid import uuid4

import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
from api.core.config import settings, STORAGE_ROOT


class StorageBackend(ABC):
    """Abstract storage backend."""
    
    @abstractmethod
    def save(self, key: str, data: BinaryIO, content_type: str = "application/octet-stream") -> str:
        """Save file and return public URL or path."""
        pass
    
    @abstractmethod
    def get(self, key: str) -> Optional[BinaryIO]:
        """Get file as stream."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete file. Returns True if existed."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if file exists."""
        pass
    
    @abstractmethod
    def get_url(self, key: str, expires_in: int = 3600) -> str:
        """Get public/presigned URL."""
        pass


class LocalStorage(StorageBackend):
    """Local filesystem storage."""
    
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
    
    def _resolve(self, key: str) -> Path:
        """Resolve a storage key to a filesystem path, preventing traversal."""
        # Reject absolute keys and any component that escapes the root
        root = self.root.resolve()
        # Normalise the key: strip leading slashes, reject backslashes
        safe_key = key.lstrip("/").replace("\\", "/")
        # Reject any ".." segment after normalisation
        if ".." in Path(safe_key).parts:
            raise ValueError(f"Unsafe storage key: {key!r}")
        candidate = (root / safe_key).resolve()
        # Ensure the resolved path stays within the storage root
        try:
            candidate.relative_to(root)
        except ValueError:
            raise ValueError(f"Unsafe storage key: {key!r}")
        return candidate
    
    def save(self, key: str, data: BinaryIO, content_type: str = "application/octet-stream") -> str:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
                temporary = Path(f.name)
                shutil.copyfileobj(data, f)
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return f"/files/{key}"
    
    def get(self, key: str) -> Optional[BinaryIO]:
        path = self._resolve(key)
        if not path.exists():
            return None
        return open(path, "rb")
    
    def delete(self, key: str) -> bool:
        path = self._resolve(key)
        if path.exists():
            path.unlink()
            # Clean up empty parent directories
            try:
                parent = path.parent
                while parent != self.root and not any(parent.iterdir()):
                    parent.rmdir()
                    parent = parent.parent
            except OSError:
                pass
            return True
        return False
    
    def exists(self, key: str) -> bool:
        return self._resolve(key).exists()
    
    def get_url(self, key: str, expires_in: int = 3600) -> str:
        return f"/files/{key}"


class S3Storage(StorageBackend):
    """S3-compatible storage (AWS S3, MinIO, etc.)."""
    
    def __init__(
        self,
        bucket: str,
        region: str,
        access_key: str,
        secret_key: str,
        endpoint_url: Optional[str] = None,
    ):
        self.bucket = bucket
        self.client = boto3.client(
            "s3",
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            endpoint_url=endpoint_url,
            config=Config(connect_timeout=3, read_timeout=10, retries={"max_attempts": 2}),
        )
        # Ensure bucket exists
        try:
            self.client.head_bucket(Bucket=bucket)
        except ClientError:
            self.client.create_bucket(Bucket=bucket)
    
    def save(self, key: str, data: BinaryIO, content_type: str = "application/octet-stream") -> str:
        self.client.upload_fileobj(
            data,
            self.bucket,
            key,
            ExtraArgs={"ContentType": content_type},
        )
        return self.get_url(key)
    
    def get(self, key: str) -> Optional[BinaryIO]:
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            return response["Body"]
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return None
            raise
    
    def delete(self, key: str) -> bool:
        try:
            self.client.delete_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False
    
    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise
    
    def get_url(self, key: str, expires_in: int = 3600) -> str:
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires_in,
        )


def get_storage() -> StorageBackend:
    """Factory function to get configured storage backend."""
    if settings.STORAGE_BACKEND.lower() == "s3":
        return S3Storage(
            bucket=settings.STORAGE_S3_BUCKET,
            region=settings.STORAGE_S3_REGION,
            access_key=settings.STORAGE_S3_ACCESS_KEY,
            secret_key=settings.STORAGE_S3_SECRET_KEY,
            endpoint_url=settings.STORAGE_S3_ENDPOINT_URL,
        )
    return LocalStorage(STORAGE_ROOT)


def generate_storage_key(
    camera_id: str,
    job_id: Optional[str] = None,
    output_type: str = "output",
    extension: str = "jpg",
) -> str:
    """Generate a structured storage key."""
    parts = ["cameras", str(camera_id), output_type]
    if job_id:
        parts.append(str(job_id))
    filename = f"{uuid4().hex}.{extension}"
    parts.append(filename)
    return "/".join(parts)


def validate_image_content(
    content: bytes, filename: str = "", max_size_mb: int = 10
) -> None:
    """Validate uploaded image content bytes.
    
    Checks that content is non-empty, within size limit, and has valid
    image magic bytes (JPEG, PNG, BMP, GIF). Raises ValueError on failure.
    """
    if not content:
        raise ValueError("Empty file uploaded.")
    
    if len(content) > max_size_mb * 1024 * 1024:
        raise ValueError(
            f"File '{filename}' too large: {len(content)} bytes. "
            f"Maximum {max_size_mb}MB."
        )
    
    # Magic byte signatures
    if content[:2] == b"\xff\xd8":
        return  # JPEG
    if len(content) >= 8 and content[:8] == b"\x89PNG\r\n\x1a\n":
        return  # PNG
    if content[:2] == b"BM":
        return  # BMP
    if content[:4] in (b"GIF87a", b"GIF89a"):
        return  # GIF
    
    raise ValueError(
        f"Invalid image file '{filename}': unrecognized magic bytes "
        f"{content[:4]!r}"
    )
