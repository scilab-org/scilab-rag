"""
Keycloak authentication middleware for FastAPI.

Provides JWT token validation and role-based authorization using Keycloak.
"""

from functools import lru_cache
from typing import Annotated, List, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2AuthorizationCodeBearer
from pydantic import BaseModel

from app.core.config import settings
from app.core.dependencies import get_keycloak_openid

# Build Keycloak OpenID Connect URLs from config
_keycloak_base = f"{settings.KEYCLOAK_SERVER_URL.rstrip('/')}/realms/{settings.KEYCLOAK_REALM}/protocol/openid-connect"
_authorization_url = f"{_keycloak_base}/auth"
_token_url = f"{_keycloak_base}/token"

# OAuth2 Authorization Code flow with PKCE for Swagger UI
oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl=_authorization_url,
    tokenUrl=_token_url,
    auto_error=True,
)

oauth2_scheme_optional = OAuth2AuthorizationCodeBearer(
    authorizationUrl=_authorization_url,
    tokenUrl=_token_url,
    auto_error=False,
)

class AuthenticatedUser(BaseModel):
    """Authenticated user information extracted from Keycloak JWT token."""
    
    user_id: str
    username: str
    email: Optional[str] = None
    email_verified: bool = False
    roles: List[str] = []
    realm_roles: List[str] = []
    client_roles: dict[str, List[str]] = {}
    
def _extract_user_from_token(token_info: dict) -> AuthenticatedUser:
    """Extract user information from decoded Keycloak token payload."""
    
    # Extract realm roles
    realm_access = token_info.get("realm_access", {})
    realm_roles = realm_access.get("roles", [])
    
    # Extract client-specific roles
    resource_access = token_info.get("resource_access", {})
    client_roles = {
        client: access.get("roles", [])
        for client, access in resource_access.items()
    }
    
    # Combine all roles for convenience
    all_roles = list(realm_roles)
    for roles in client_roles.values():
        all_roles.extend(roles)
    
    return AuthenticatedUser(
        user_id=token_info.get("sub", ""),
        username=token_info.get("preferred_username", ""),
        email=token_info.get("email"),
        email_verified=token_info.get("email_verified", False),
        roles=list(set(all_roles)),  # Deduplicate
        realm_roles=realm_roles,
        client_roles=client_roles,
    )

async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
) -> AuthenticatedUser:
    """
    Validate JWT token and return authenticated user.
    
    Extracts Bearer token from Authorization header, validates it against
    Keycloak, and returns user information.
    
    Raises:
        HTTPException: 401 if token is missing, invalid, or expired
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        keycloak_openid = get_keycloak_openid()
        
        # Decode and validate token using Keycloak
        # This verifies signature, expiration, and issuer
        token_info = keycloak_openid.decode_token(
            token,
            validate=True,
        )
        
        if not token_info:
            raise credentials_exception
        
        return _extract_user_from_token(token_info)
        
    except Exception as e:
        # Log the error for debugging (don't expose details to client)
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Token validation failed: {type(e).__name__}: {e}")
        raise credentials_exception

async def get_current_user_optional(
    token: Annotated[Optional[str], Depends(oauth2_scheme_optional)],
) -> Optional[AuthenticatedUser]:
    """
    Optionally validate JWT token and return authenticated user.
    
    Returns None if no token is provided, allowing endpoints to work
    both authenticated and unauthenticated.
    
    Raises:
        HTTPException: 401 if token is provided but invalid
    """
    if token is None:
        return None
    
    return await get_current_user(token)

def require_role(role: str):
    """
    Factory function that creates a dependency requiring a specific role.
    
    Usage:
        @app.get("/admin")
        async def admin_endpoint(user: Annotated[AuthenticatedUser, Depends(require_role("admin"))]):
            ...
    
    Args:
        role: The role name required to access the endpoint
        
    Returns:
        A dependency function that validates the user has the required role
    """
    async def role_checker(
        user: Annotated[AuthenticatedUser, Depends(get_current_user)],
    ) -> AuthenticatedUser:
        if role not in user.roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{role}' required",
            )
        return user
    
    return role_checker

def require_any_role(roles: List[str]):
    """
    Factory function that creates a dependency requiring any of the specified roles.
    
    Usage:
        @app.get("/moderator")
        async def mod_endpoint(user: Annotated[AuthenticatedUser, Depends(require_any_role(["admin", "moderator"]))]):
            ...
    
    Args:
        roles: List of role names, user must have at least one
        
    Returns:
        A dependency function that validates the user has at least one required role
    """
    async def role_checker(
        user: Annotated[AuthenticatedUser, Depends(get_current_user)],
    ) -> AuthenticatedUser:
        if not any(role in user.roles for role in roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"One of roles {roles} required",
            )
        return user
    
    return role_checker

# Type aliases for cleaner endpoint signatures
CurrentUser = Annotated[AuthenticatedUser, Depends(get_current_user)]
OptionalUser = Annotated[Optional[AuthenticatedUser], Depends(get_current_user_optional)]
