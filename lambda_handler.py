"""
AWS Lambda handler for FastAPI application
"""

from mangum import Mangum
from fastapi_app import app

# Create Lambda handler
handler = Mangum(app)