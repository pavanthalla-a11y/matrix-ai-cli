import google.auth
import logging
from .config import GCP_PROJECT_ID

logger = logging.getLogger(__name__)

class HTTPException(Exception):
    def __init__(self, status_code, detail):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")

def setup_google_auth():
    """Setup Google Cloud authentication with proper error handling"""
    try:
        # Try to get default credentials
        credentials, project = google.auth.default()
        
        # Set quota project if not already set
        if hasattr(credentials, 'quota_project_id') and not credentials.quota_project_id:
            credentials = credentials.with_quota_project(GCP_PROJECT_ID)
        
        return credentials, project
    except Exception as e:
        logger.error(f"Authentication setup failed: {e}")
        raise HTTPException(500, f"Authentication failed: {e}")
