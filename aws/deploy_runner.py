import boto3
import json
import base64
import subprocess
import time
from botocore.exceptions import ClientError

class AppRunnerDeployer:
    def __init__(self, region='eu-central-1'):
        self.region = region
        self.apprunner_client = boto3.client('apprunner', region_name=region)
        self.ecr_client = boto3.client('ecr', region_name=region)
        self.sts_client = boto3.client('sts', region_name=region)
        
        # Get account ID
        self.account_id = self.sts_client.get_caller_identity()['Account']
        self.repository_name = 'titanic-fastapi'
        self.service_name = 'titanic-fastapi-service'
    
    def create_ecr_repository(self):
        """Create ECR repository if it doesn't exist"""
        try:
            response = self.ecr_client.create_repository(
                repositoryName=self.repository_name,
                imageScanningConfiguration={'scanOnPush': True}
            )
            repository_uri = response['repository']['repositoryUri']
            print(f"✅ Created ECR repository: {repository_uri}")
            return repository_uri
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'RepositoryAlreadyExistsException':
                # Get existing repository URI
                response = self.ecr_client.describe_repositories(
                    repositoryNames=[self.repository_name]
                )
                repository_uri = response['repositories'][0]['repositoryUri']
                print(f"✅ Using existing ECR repository: {repository_uri}")
                return repository_uri
            else:
                raise e
    
    def build_and_push_image(self):
        """Build Docker image and push to ECR"""
        repository_uri = self.create_ecr_repository()
        
        try:
            # Get ECR login token
            token_response = self.ecr_client.get_authorization_token()
            token = token_response['authorizationData'][0]['authorizationToken']
            endpoint = token_response['authorizationData'][0]['proxyEndpoint']
            
            # Decode token
            username, password = base64.b64decode(token).decode().split(':')
            
            print("🔨 Building and pushing Docker image...")
            
            # Docker commands
            commands = [
                # Login to ECR
                f"echo '{password}' | docker login --username {username} --password-stdin {endpoint}",
                
                # Build image
                f"docker build -t {self.repository_name}:latest -f Dockerfile.fastapi .",
                
                # Tag image
                f"docker tag {self.repository_name}:latest {repository_uri}:latest",
                
                # Push image
                f"docker push {repository_uri}:latest"
            ]
            
            for i, cmd in enumerate(commands, 1):
                print(f"Step {i}/{len(commands)}: Running Docker command...")
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"❌ Error in step {i}: {result.stderr}")
                    return None
                    
                print(f"✅ Step {i} completed successfully")
            
            print(f"🎉 Docker image pushed successfully: {repository_uri}:latest")
            return f"{repository_uri}:latest"
            
        except Exception as e:
            print(f"❌ Error building/pushing image: {str(e)}")
            return None
    
    def create_app_runner_service_role(self):
        """Create IAM role for App Runner if it doesn't exist"""
        iam_client = boto3.client('iam', region_name=self.region)
        role_name = 'AppRunnerECRAccessRole'
        
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "build.apprunner.amazonaws.com"
                    },
                    "Action": "sts:AssumeRole"
                }
            ]
        }
        
        try:
            # Try to get existing role
            response = iam_client.get_role(RoleName=role_name)
            role_arn = response['Role']['Arn']
            print(f"✅ Using existing App Runner role: {role_arn}")
            return role_arn
            
        except iam_client.exceptions.NoSuchEntityException:
            # Create new role
            print(f"🔧 Creating IAM role for App Runner: {role_name}")
            
            # Create role
            create_response = iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description='App Runner ECR access role'
            )
            
            role_arn = create_response['Role']['Arn']
            
            # Attach ECR read policy
            iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn='arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess'
            )
            
            print(f"✅ Created App Runner role: {role_arn}")
            
            # Wait for role to propagate
            print("⏳ Waiting for role to propagate...")
            time.sleep(10)
            
            return role_arn
    
    def deploy_app_runner_service(self, image_uri):
        """Deploy FastAPI app to App Runner"""
        try:
            # Get or create IAM role for ECR access
            access_role_arn = self.create_app_runner_service_role()
            
            print(f"🚀 Deploying to App Runner: {self.service_name}")
            
            service_config = {
                'ServiceName': self.service_name,
                'SourceConfiguration': {
                    'ImageRepository': {
                        'ImageIdentifier': image_uri,
                        'ImageConfiguration': {
                            'Port': '8000',
                            'RuntimeEnvironmentVariables': {
                                'ENVIRONMENT': 'production',
                                'AWS_DEFAULT_REGION': self.region
                            },
                            'StartCommand': 'uvicorn fastapi_app:app --host 0.0.0.0 --port 8000'
                        },
                        'ImageRepositoryType': 'ECR'
                    },
                    'AccessRoleArn': access_role_arn,
                    'AutoDeploymentsEnabled': True
                },
                'InstanceConfiguration': {
                    'Cpu': '1 vCPU',
                    'Memory': '2 GB'
                },
                'HealthCheckConfiguration': {
                    'Protocol': 'HTTP',
                    'Path': '/health',
                    'Interval': 10,
                    'Timeout': 5,
                    'HealthyThreshold': 1,
                    'UnhealthyThreshold': 5
                }
            }
            
            # Check if service already exists
            try:
                existing_services = self.apprunner_client.list_services()
                service_exists = any(
                    service['ServiceName'] == self.service_name 
                    for service in existing_services['ServiceSummaryList']
                )
                
                if service_exists:
                    print(f"🔄 Updating existing App Runner service...")
                    # Update existing service
                    response = self.apprunner_client.update_service(
                        ServiceArn=f"arn:aws:apprunner:{self.region}:{self.account_id}:service/{self.service_name}",
                        SourceConfiguration=service_config['SourceConfiguration'],
                        InstanceConfiguration=service_config['InstanceConfiguration'],
                        HealthCheckConfiguration=service_config['HealthCheckConfiguration']
                    )
                else:
                    print(f"🆕 Creating new App Runner service...")
                    # Create new service
                    response = self.apprunner_client.create_service(**service_config)
                
                service_arn = response['Service']['ServiceArn']
                service_id = response['Service']['ServiceId']
                service_url = response['Service']['ServiceUrl']
                
                print(f"✅ App Runner service deployed successfully!")
                print(f"📍 Service ARN: {service_arn}")
                print(f"🆔 Service ID: {service_id}")
                print(f"🌐 Service URL: https://{service_url}")
                print(f"📖 API Docs: https://{service_url}/docs")
                print(f"🔍 Health Check: https://{service_url}/health")
                
                # Wait for service to become running
                print("⏳ Waiting for service to become ready...")
                self.wait_for_service_ready(service_arn)
                
                return {
                    'service_arn': service_arn,
                    'service_url': f"https://{service_url}",
                    'service_id': service_id
                }
                
            except Exception as e:
                print(f"❌ Error with App Runner service: {str(e)}")
                return None
                
        except Exception as e:
            print(f"❌ App Runner deployment failed: {str(e)}")
            return None
    
    def wait_for_service_ready(self, service_arn, timeout_minutes=10):
        """Wait for App Runner service to become ready"""
        timeout_seconds = timeout_minutes * 60
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            try:
                response = self.apprunner_client.describe_service(ServiceArn=service_arn)
                status = response['Service']['Status']
                
                print(f"⏳ Service status: {status}")
                
                if status == 'RUNNING':
                    print("🎉 Service is now running!")
                    return True
                elif status in ['CREATE_FAILED', 'UPDATE_FAILED']:
                    print(f"❌ Service deployment failed with status: {status}")
                    return False
                
                time.sleep(30)  # Wait 30 seconds before checking again
                
            except Exception as e:
                print(f"Error checking service status: {str(e)}")
                time.sleep(30)
        
        print(f"⚠️  Service did not become ready within {timeout_minutes} minutes")
        return False
    
    def deploy(self):
        """Complete deployment process"""
        try:
            print("🚀 Starting App Runner deployment...")
            
            # Step 1: Build and push Docker image
            image_uri = self.build_and_push_image()
            if not image_uri:
                print("❌ Failed to build/push Docker image")
                return False
            
            # Step 2: Deploy to App Runner
            result = self.deploy_app_runner_service(image_uri)
            if not result:
                print("❌ Failed to deploy to App Runner")
                return False
            
            print("\n🎊 Deployment completed successfully!")
            print(f"Your FastAPI app is now running at: {result['service_url']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Deployment failed: {str(e)}")
            return False

if __name__ == "__main__":
    deployer = AppRunnerDeployer()
    deployer.deploy()