"""
JWT authentication dependency untuk FastAPI.

Memvalidasi token yang diissue oleh auth-service menggunakan
shared secret key (HMAC-SHA256). Token dibaca dari HttpOnly cookie
`access_token` yang di-set oleh auth-service.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import jwt
from fastapi import HTTPException, Query, Request, status

from app.core.config import settings


@dataclass
class CurrentUser:
    user_id: str
    username: str
    email: str
    roles: List[str] = field(default_factory=list)


async def get_current_user(
    request: Request,
    token_query: Optional[str] = Query(None, alias="token"),
) -> CurrentUser:
    """
    FastAPI dependency untuk validasi JWT.
    Urutan cek: Authorization header → query param ?token= → cookie.

    Usage:
        @router.get("/endpoint")
        async def endpoint(current_user: CurrentUser = Depends(get_current_user)):
            ...
    """
    # 1. Authorization header (engineFetch dengan cross-origin)
    token = None
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header[7:]

    # 2. Query param ?token= (untuk <video src> dan <img src> yang tidak bisa set header)
    if not token and token_query:
        token = token_query

    # 3. Cookie (same-origin / local dev)
    if not token:
        token = request.cookies.get("access_token")

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )

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
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )

    user_id = payload.get("user_id")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token: missing user_id",
        )

    return CurrentUser(
        user_id=str(user_id),
        username=payload.get("username", ""),
        email=payload.get("email", ""),
        roles=payload.get("roles", []),
    )
