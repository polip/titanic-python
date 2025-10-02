# AWS Deployment Guide - Titanic API

## Overview
This guide covers three deployment options for the Titanic Survival Prediction API on AWS.

## Prerequisites
- AWS CLI configured (`aws configure`)
- Node.js and npm installed
- Python 3.9+ with pip
- Docker (for containerized deployments)

## Deployment Options

### Option 1: AWS Lambda (Serverless) - Recommended
**Best for:** Low cost, auto-scaling, event-driven workloads

#### Setup:
```bash
# Install dependencies
npm install

# Deploy to development
./deploy.sh dev

# Deploy to production
./deploy.sh prod
```

#### Features:
- Auto-scaling from 0 to 1000+ concurrent executions
- Pay-per-request pricing
- No server management
- Built-in monitoring via CloudWatch

#### Limitations:
- 15-minute execution timeout
- 512MB-10GB memory limit
- Cold start latency

### Option 2: AWS Elastic Beanstalk
**Best for:** Quick deployment, managed infrastructure

#### Setup:
```bash
# Install EB CLI
pip install awsebcli

# Initialize EB application
eb init titanic-api --region us-east-1 --platform python-3.9

# Create environment and deploy
eb create titanic-api-dev
eb deploy
```

#### Configuration:
- Uses `application.py` as WSGI entry point
- Configuration in `.ebextensions/01_python.config`
- Auto-scaling group: 1-4 t3.micro instances

### Option 3: AWS ECS/Fargate (Container)
**Best for:** Containerized applications, predictable workloads

#### Setup:
```bash
# Build and push Docker image
docker build -f Dockerfile.fastapi -t titanic-api .

# Tag for ECR
docker tag titanic-api:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/titanic-api:latest

# Push to ECR
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/titanic-api:latest

# Deploy using CloudFormation or ECS CLI
aws cloudformation deploy --template-file cloudformation.yml --stack-name titanic-api-infra
```

## Environment Variables

### Production Environment:
```bash
ENVIRONMENT=prod
MODEL_PATH=/opt/models/titanic_model.pkl
FEATURES_PATH=/opt/models/titanic_model_features.pkl
```

### Development Environment:
```bash
ENVIRONMENT=dev
MODEL_PATH=models/titanic_model.pkl
FEATURES_PATH=models/titanic_model_features.pkl
```

## API Endpoints
After deployment, your API will be available at:

- **Health Check**: `GET /`
- **Model Info**: `GET /model/info`
- **Single Prediction**: `POST /predict`
- **Batch Prediction**: `POST /predict/batch`
- **Custom Predictions**: `POST /predict/custom`

## Monitoring & Logging

### CloudWatch Logs:
- Lambda: `/aws/lambda/titanic-api-{stage}`
- ECS: `/ecs/titanic-api`
- Elastic Beanstalk: `/aws/elasticbeanstalk/titanic-api-{env}`

### Metrics:
- API Gateway: Request count, latency, errors
- Lambda: Invocations, duration, errors
- ECS: CPU, memory utilization

## Cost Estimation

### Lambda (1000 requests/day):
- **Free tier**: First 1M requests free
- **Paid**: ~$0.20/month for 1000 requests

### Elastic Beanstalk (t3.micro):
- **Instance**: ~$8.50/month
- **Load Balancer**: ~$16/month
- **Total**: ~$25/month

### ECS Fargate (0.25 vCPU, 0.5GB):
- **Compute**: ~$6/month (running 24/7)
- **Additional**: ALB, storage costs

## Security Best Practices

1. **Environment Variables**: Store sensitive data in AWS Systems Manager Parameter Store
2. **API Keys**: Implement API key authentication for production
3. **CORS**: Configure appropriate CORS settings
4. **HTTPS**: Always use HTTPS in production
5. **VPC**: Deploy in private subnets for ECS/EB

## Testing Deployment

```bash
# Test health endpoint
curl https://your-api-url/

# Test prediction
curl -X POST https://your-api-url/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pclass": "3rd",
    "sex": "male",
    "age": 22,
    "sibsp": 1,
    "parch": 0,
    "fare": 7.25,
    "embarked": "S",
    "title": "Mr"
  }'
```

## Troubleshooting

### Common Issues:
1. **Model not found**: Ensure model files are included in deployment package
2. **Memory issues**: Increase Lambda memory or ECS task memory
3. **Timeout**: Increase timeout settings for Lambda/ALB
4. **Dependencies**: Check requirements.txt compatibility with target runtime

### Logs Access:
```bash
# Lambda logs
aws logs tail /aws/lambda/titanic-api-dev --follow

# ECS logs
aws logs tail /ecs/titanic-api --follow

# Elastic Beanstalk logs
eb logs
```

## Scaling & Performance

### Lambda:
- **Concurrent executions**: Default 1000, can request increase
- **Memory**: 128MB-10GB, affects CPU allocation
- **Provisioned concurrency**: Reduce cold starts

### ECS:
- **Auto Scaling**: CPU/memory-based scaling
- **Service**: 1-10 tasks recommended
- **Health checks**: Configure ALB health checks

### Elastic Beanstalk:
- **Auto Scaling**: Time/metric-based scaling
- **Instance types**: Scale up for CPU-intensive workloads
- **Load balancing**: Application Load Balancer included