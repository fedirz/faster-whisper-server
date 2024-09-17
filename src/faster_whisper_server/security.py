from typing import Annotated

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from faster_whisper_server.config import config

auth_scheme = HTTPBearer(scheme_name="API key")
API_KEY = config.api_key

if not API_KEY:

    def check_api_key():
        # if No API key is set, so we don't check for it.
        pass

else:

    def check_api_key(api_key: Annotated[HTTPAuthorizationCredentials, Depends(auth_scheme)]):
        if api_key.scheme != "Bearer":
            raise HTTPException(status_code=403, detail="Invalid authentication scheme")
        if api_key.credentials != API_KEY:
            raise HTTPException(status_code=403, detail="Unauthorized")

        return api_key.credentials
