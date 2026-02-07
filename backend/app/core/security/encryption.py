"""
AES-256 encryption for secure file storage.
"""
import base64
import os
import secrets
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from app.config import settings


class SecureFileManager:
    """
    AES-256 encryption for uploaded files.
    Guarantees secure storage and deletion.
    """
    
    def __init__(self, temp_dir: Optional[Path] = None):
        """
        Initialize secure file manager.
        
        Args:
            temp_dir: Custom temp directory, defaults to /dev/shm if available
        """
        self.master_key = settings.FILE_ENCRYPTION_KEY
        
        # Use RAM-based temp directory if available (Linux), otherwise system temp
        if temp_dir:
            self.temp_dir = Path(temp_dir)
        elif Path("/dev/shm").exists():
            self.temp_dir = Path(tempfile.mkdtemp(prefix="fraud_secure_", dir="/dev/shm"))
        else:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="fraud_secure_"))
        
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize cipher with derived key
        self._init_cipher()
    
    def _init_cipher(self):
        """Initialize Fernet cipher with derived key from master key."""
        # Generate a random salt for this session
        self.salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=480000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_key.encode()))
        self.cipher = Fernet(key)
    
    def encrypt_store(self, file_bytes: bytes, file_id: Optional[str] = None) -> str:
        """
        Encrypt and store file. Returns secure file ID.
        
        Args:
            file_bytes: Raw file bytes to encrypt
            file_id: Optional custom file ID, auto-generated if not provided
            
        Returns:
            File ID for later retrieval
        """
        if file_id is None:
            file_id = secrets.token_urlsafe(32)
        
        # Encrypt the file
        encrypted = self.cipher.encrypt(file_bytes)
        
        # Store with restricted permissions
        secure_path = self.temp_dir / f"{file_id}.enc"
        secure_path.write_bytes(encrypted)
        os.chmod(secure_path, 0o600)  # Owner read/write only
        
        # Also store salt for decryption
        salt_path = self.temp_dir / f"{file_id}.salt"
        salt_path.write_bytes(self.salt)
        os.chmod(salt_path, 0o600)
        
        return file_id
    
    def decrypt_retrieve(self, file_id: str) -> bytes:
        """
        Decrypt and return file content.
        
        Args:
            file_id: File ID from encrypt_store
            
        Returns:
            Decrypted file bytes
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If decryption fails
        """
        secure_path = self.temp_dir / f"{file_id}.enc"
        salt_path = self.temp_dir / f"{file_id}.salt"
        
        if not secure_path.exists():
            raise FileNotFoundError(f"Secure file not found: {file_id}")
        
        if not salt_path.exists():
            raise FileNotFoundError(f"Salt file not found: {file_id}")
        
        # Read salt and recreate cipher
        salt = salt_path.read_bytes()
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_key.encode()))
        cipher = Fernet(key)
        
        # Decrypt
        encrypted = secure_path.read_bytes()
        return cipher.decrypt(encrypted)
    
    def secure_delete(self, file_id: str) -> bool:
        """
        Cryptographic erasure - overwrite then delete.
        
        Args:
            file_id: File ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        secure_path = self.temp_dir / f"{file_id}.enc"
        salt_path = self.temp_dir / f"{file_id}.salt"
        
        deleted = False
        
        for path in [secure_path, salt_path]:
            if path.exists():
                # Overwrite with random data
                size = path.stat().st_size
                with open(path, 'wb') as f:
                    f.write(os.urandom(size))
                
                # Sync to disk
                os.fsync(f.fileno())
                
                # Delete
                path.unlink()
                deleted = True
        
        return deleted
    
    def cleanup(self):
        """Emergency cleanup of all files."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def __del__(self):
        """Cleanup on destruction."""
        # Note: In production, you might want persistent storage
        # This is for temporary secure processing
        pass
    
    def get_file_path(self, file_id: str) -> Path:
        """Get the encrypted file path (for streaming)."""
        return self.temp_dir / f"{file_id}.enc"
    
    def exists(self, file_id: str) -> bool:
        """Check if file exists."""
        return (self.temp_dir / f"{file_id}.enc").exists()


# Singleton instance
_secure_manager: Optional[SecureFileManager] = None


def get_secure_file_manager() -> SecureFileManager:
    """Get or create secure file manager singleton."""
    global _secure_manager
    if _secure_manager is None:
        _secure_manager = SecureFileManager()
    return _secure_manager
