"""Security utilities: API key authentication."""
from typing import Optional
from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from api.core.config import settings
from api.core.database import get_db
from api.models.orm import ApiKey


async def verify_api_key(
    api_key: str = Header(..., alias="X-API-Key"),
    db: AsyncSession = Depends(get_db),
) -> ApiKey:
    """Verify API key from header."""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    # Check against configured keys (for simple deployments)
    if api_key in settings.valid_api_keys:
        # Return a dummy ApiKey object for configured keys
        return ApiKey(
            id="configured",
            key_hash="",
            name="configured",
            is_active=True,
            scopes=["*"],
        )
    
    # Check database
    result = await db.execute(
        select(ApiKey).where(ApiKey.key_hash == hash_key(api_key), ApiKey.is_active == True)
    )
    db_key = result.scalar_one_or_none()
    
    if not db_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    return db_key


def hash_key(key: str) -> str:
    """Hash API key for storage."""
    import hashlib
    return hashlib.sha256(key.encode()).hexdigest()


def generate_api_key() -> tuple[str, str]:
    """Generate a new API key. Returns (plain_key, hashed_key)."""
    import secrets
    plain = f"wst_{secrets.token_urlsafe(32)}"
    return plain, hash_key(plain)


def require_scope(required_scope: str):
    """Dependency factory to require specific scope."""
    async def check_scope(api_key: ApiKey = Depends(verify_api_key)) -> ApiKey:
        if "*" not in api_key.scopes and required_scope not in api_key.scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Scope '{required_scope}' required",
            )
        return api_key
    return check_scope