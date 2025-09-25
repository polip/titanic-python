"""
AWS Elastic Beanstalk WSGI application entry point
"""

from fastapi_app import app

# Elastic Beanstalk expects 'application' variable
application = app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(application, host="0.0.0.0", port=8000)