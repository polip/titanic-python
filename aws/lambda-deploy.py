import boto3
import zipfile
import os
import subprocess
import shutil
from pathlib import Path
import json
import time

def create_lambda_layer():
    """Create or get existing Lambda layer"""
    lambda_client = boto3.client('lambda', region_name='eu-central-1')
    
    layer_name = 'titanic-dependencies'
    
    try:
        # Try to get existing layer
        response = lambda_client.list_layer_versions(LayerName=layer_name)
        if response['LayerVersions']:
            layer_arn = response['LayerVersions'][0]['LayerVersionArn']
            print(f"Using existing layer: {layer_arn}")
            return layer_arn
    except:
        pass
    
    # Create new layer if not exists
    print("Creating Lambda layer...")
    
    # Clean up
    if os.path.exists('layer_package'):
        shutil.rmtree('layer_package')
    
    os.makedirs('layer_package/python')
    
    # Install heavy dependencies to layer
    heavy_deps = [
        "numpy>=1.21.0",
        "pandas>=1.5.0", 
        "scikit-learn>=1.1.0",
        "joblib>=1.2.0"
    ]
    
    for dep in heavy_deps:
        subprocess.run([
            "pip", "install", dep, "-t", "layer_package/python", 
            "--no-compile", "--no-cache-dir"
        ])
    
    # Create layer ZIP
    with zipfile.ZipFile('lambda_layer.zip', 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for root, dirs, files in os.walk('layer_package'):
            for file in files:
                # Skip unnecessary files
                if any(skip in file for skip in ['.pyc', '__pycache__', '.dist-info']):
                    continue
                    
                file_path = os.path.join(root, file)
                arc_path = os.path.relpath(file_path, 'layer_package')
                zip_file.write(file_path, arc_path)
    
    # Deploy layer
    with open('lambda_layer.zip', 'rb') as zip_file:
        response = lambda_client.publish_layer_version(
            LayerName=layer_name,
            Description='Heavy dependencies for Titanic API',
            Content={'ZipFile': zip_file.read()},
            CompatibleRuntimes=['python3.9']
        )
    
    layer_arn = response['LayerVersionArn']
    print(f"Created layer: {layer_arn}")
    return layer_arn

def create_deployment_package():
    """Create lightweight deployment package"""
    
    # Clean up previous deployment
    if os.path.exists('lambda_package'):
        shutil.rmtree('lambda_package')
    if os.path.exists('lambda_function.zip'):
        os.remove('lambda_function.zip')
    
    os.makedirs('lambda_package')
    
    # Install only lightweight dependencies
    light_deps = [
        "mangum>=0.17.0",
        "fastapi>=0.68.0", 
        "pydantic>=1.8.0",
        "python-multipart>=0.0.5"
    ]
    
    print("Installing lightweight dependencies...")
    for dep in light_deps:
        subprocess.run([
            "pip", "install", dep, "-t", "lambda_package", 
            "--no-compile", "--no-cache-dir"
        ])
    
    # Copy application files
    app_files = [
        'lambda_handler.py',
        'fastapi_app.py'
    ]
    
    for file in app_files:
        if os.path.exists(file):
            shutil.copy2(file, 'lambda_package/')
            print(f"Copied {file}")
    
    # Copy model files if they exist (compress if large)
    if os.path.exists('models'):
        shutil.copytree('models', 'lambda_package/models')
    
    # Create ZIP file with compression
    print("Creating optimized ZIP package...")
    with zipfile.ZipFile('lambda_function.zip', 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zip_file:
        for root, dirs, files in os.walk('lambda_package'):
            for file in files:
                # Skip unnecessary files
                if any(skip in file for skip in ['.pyc', '__pycache__', '.dist-info', 'tests']):
                    continue
                    
                file_path = os.path.join(root, file)
                arc_path = os.path.relpath(file_path, 'lambda_package')
                zip_file.write(file_path, arc_path)
    
    # Check ZIP size
    zip_size = os.path.getsize('lambda_function.zip') / (1024 * 1024)  # MB
    print(f"Deployment package size: {zip_size:.2f} MB")
    
    return 'lambda_function.zip'

def deploy_to_lambda(function_name="titanic-fastapi", region="eu-central-1"):
    """Deploy to AWS Lambda with layer"""
    
    lambda_client = boto3.client('lambda', region_name=region)
    
    # Create or get layer
    layer_arn = create_lambda_layer()
    
    # Create or get IAM role (your existing function)
    role_arn = "arn:aws:iam::308402043233:role/lambda-execution-role"
    with open('lambda_function.zip', 'rb') as zip_file:
        zip_content = zip_file.read()
    
    try:
        # Try to update existing function
        response = lambda_client.update_function_code(
            FunctionName=function_name,
            ZipFile=zip_content
        )
        print(f"Updated existing function: {response['FunctionArn']}")
        
        # Update function configuration with layer
        lambda_client.update_function_configuration(
            FunctionName=function_name,
            Timeout=30,
            MemorySize=1024,  # Increased memory for ML models
            Layers=[layer_arn],  # Add the layer
            Environment={
                'Variables': {
                    'MODEL_PATH': 'models/titanic_model.pkl',
                    'FEATURES_PATH': 'models/titanic_model_features.pkl',
                    'ENVIRONMENT': 'prod'
                }
            }
        )
        
    except lambda_client.exceptions.ResourceNotFoundException:
        # Create new function
        print(f"Creating new function: {function_name}")
        response = lambda_client.create_function(
            FunctionName=function_name,
            Runtime='python3.9',  # Changed from 3.12 to 3.9 (more stable)
            Role=role_arn,
            Handler='lambda_handler.handler',
            Code={'ZipFile': zip_content},
            Timeout=30,
            MemorySize=1024,
            Layers=[layer_arn],  # Add the layer
            Environment={
                'Variables': {
                    'MODEL_PATH': 'models/titanic_model.pkl',
                    'FEATURES_PATH': 'models/titanic_model_features.pkl',
                    'ENVIRONMENT': 'prod'
                }
            }
        )
        print(f"Created function: {response['FunctionArn']}")
    
    return response

def create_api_gateway(function_name="titanic-fastapi", region="eu-central-1"):
    """Create API Gateway for Lambda function"""
    
    apigateway = boto3.client('apigatewayv2', region_name=region)
    lambda_client = boto3.client('lambda', region_name=region)
    
    # Get account ID
    account_id = boto3.client('sts').get_caller_identity()['Account']
    function_arn = f"arn:aws:lambda:{region}:{account_id}:function:{function_name}"
    
    try:
        # Create HTTP API
        api_response = apigateway.create_api(
            Name=f'{function_name}-api',
            ProtocolType='HTTP',
            Target=function_arn
        )
        
        api_id = api_response['ApiId']
        api_endpoint = api_response['ApiEndpoint']
        
        # Add Lambda permission for API Gateway
        try:
            lambda_client.add_permission(
                FunctionName=function_name,
                StatementId='api-gateway-invoke',
                Action='lambda:InvokeFunction',
                Principal='apigateway.amazonaws.com',
                SourceArn=f'arn:aws:execute-api:{region}:{account_id}:{api_id}/*/*'
            )
        except lambda_client.exceptions.ResourceConflictException:
            print("Permission already exists")
        
        print(f"API Gateway created: {api_endpoint}")
        return api_endpoint
        
    except Exception as e:
        print(f"Error creating API Gateway: {e}")
        return None

if __name__ == "__main__":
    try:
        # Create deployment package
        zip_file = create_deployment_package()  # UNCOMMENTED THIS
        
        if zip_file is None:
            print("Failed to create deployment package")
            exit(1)
        
        # Deploy to Lambda
        response = deploy_to_lambda()
        
        # Create API Gateway
        api_endpoint = create_api_gateway()
        
        print("\n✅ Deployment completed!")
        if api_endpoint:
            print(f"🌐 API Endpoint: {api_endpoint}")
            print(f"📖 API Docs: {api_endpoint}/docs")
            print(f"🔍 Health Check: {api_endpoint}/")
            
    except Exception as e:
        print(f"❌ Deployment failed: {str(e)}")
        import traceback
        traceback.print_exc()