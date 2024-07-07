from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from gotrue.errors import AuthApiError

from app.config import supabase

oauth2_scheme = HTTPBearer()


async def get_current_user(token: HTTPAuthorizationCredentials = Depends(oauth2_scheme)):
    try:
        user = supabase.auth.get_user(token.credentials)
        return user
    except AuthApiError as e:
        raise HTTPException(status_code=401, detail=e.message)
