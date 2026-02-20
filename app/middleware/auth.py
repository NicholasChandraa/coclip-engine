"""
JWT authentication dependency untuk FastAPI.

Memvalidasi token yang diissue oleh auth-service menggunakan
shared secret key (HMAC-SHA256).
"""

from dataclasses import dataclass, field
from typing import List

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import settings

security = HTTPBearer()


@dataclass
class CurrentUser:
    user_id: str
    username: str
    email: str
    roles: List[str] = field(default_factory=list)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> CurrentUser:
    """
    FastAPI dependency untuk validasi JWT dari auth-service.

    Usage:
        @router.get("/endpoint")
        async def endpoint(current_user: CurrentUser = Depends(get_current_user)):
            ...
    """
    token = credentials.credentials

    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=["HS256"],
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("user_id")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token: missing user_id",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return CurrentUser(
        user_id=str(user_id),
        username=payload.get("username", ""),
        email=payload.get("email", ""),
        roles=payload.get("roles", []),
    )
