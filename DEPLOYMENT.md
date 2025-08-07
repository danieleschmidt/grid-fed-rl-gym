# Production Deployment Guide

This guide provides comprehensive instructions for deploying Grid-Fed-RL-Gym in production environments worldwide.

## 🌟 Quick Start

```bash
# Install with all dependencies
pip install grid-fed-rl-gym[full]

# Run basic demo
grid-fed-rl demo

# Start training
grid-fed-rl train --config config/production.json --federated
```

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Production Architecture                   │
├─────────────────────────────────────────────────────────────┤
│ Load Balancer → API Gateway → Microservices                │
│                                     ↓                       │
│ Grid Simulation ← Message Queue ← Training Scheduler        │
│      ↓                ↓                ↓                    │
│ Monitoring    ←→ Database ←→ Distributed Cache              │
│      ↓                                                      │
│ Alerts & Compliance Reports                                 │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Deployment Options

### 1. Container Deployment (Recommended)

**Docker**
```bash
# Build production image
docker build -t grid-fed-rl:production .

# Run with environment variables
docker run -d \
  -e GRID_FED_RL_ENV=production \
  -e REDIS_URL=redis://cache:6379 \
  -e DATABASE_URL=postgresql://db:5432/gridfedrl \
  -p 8000:8000 \
  grid-fed-rl:production
```

**Docker Compose**
```yaml
version: '3.8'
services:
  app:
    image: grid-fed-rl:production
    environment:
      - GRID_FED_RL_ENV=production
      - REDIS_URL=redis://cache:6379
    ports:
      - "8000:8000"
  
  cache:
    image: redis:alpine
    
  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=gridfedrl
```

**Kubernetes**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grid-fed-rl
spec:
  replicas: 3
  selector:
    matchLabels:
      app: grid-fed-rl
  template:
    metadata:
      labels:
        app: grid-fed-rl
    spec:
      containers:
      - name: grid-fed-rl
        image: grid-fed-rl:production
        ports:
        - containerPort: 8000
        env:
        - name: GRID_FED_RL_ENV
          value: "production"
```

### 2. Cloud Platform Deployment

**AWS**
- ECS/Fargate for containerized deployment
- Lambda for serverless functions
- RDS for database
- ElastiCache for Redis
- CloudWatch for monitoring

**Google Cloud Platform**
- Google Kubernetes Engine (GKE)
- Cloud Run for serverless
- Cloud SQL for database
- Cloud Memorystore for Redis
- Cloud Monitoring

**Microsoft Azure**
- Azure Container Instances
- Azure Functions
- Azure Database for PostgreSQL
- Azure Cache for Redis
- Azure Monitor

### 3. On-Premises Deployment

**Requirements**
- Docker or Kubernetes cluster
- PostgreSQL 12+
- Redis 6+
- Monitoring stack (Prometheus/Grafana)
- Load balancer (nginx/HAProxy)

## ⚙️ Configuration

### Environment Variables

```bash
# Core configuration
export GRID_FED_RL_ENV=production
export GRID_FED_RL_DEBUG=false
export GRID_FED_RL_LOG_LEVEL=INFO

# Database
export DATABASE_URL=postgresql://user:pass@host:5432/gridfedrl
export REDIS_URL=redis://host:6379

# Security
export SECRET_KEY=your-secret-key
export ENCRYPTION_KEY=your-encryption-key

# Regional settings
export DEFAULT_REGION=north_america
export DEFAULT_LOCALE=en_US
export TIMEZONE=UTC

# Performance
export MAX_WORKERS=8
export CACHE_SIZE=1024
export BATCH_SIZE=32

# Monitoring
export PROMETHEUS_URL=http://prometheus:9090
export GRAFANA_URL=http://grafana:3000
```

### Configuration File

```json
{
  "environment": "production",
  "debug": false,
  "logging": {
    "level": "INFO",
    "format": "json",
    "handlers": ["console", "file", "syslog"]
  },
  "database": {
    "url": "${DATABASE_URL}",
    "pool_size": 20,
    "max_overflow": 0,
    "pool_timeout": 30
  },
  "cache": {
    "url": "${REDIS_URL}",
    "default_timeout": 300,
    "key_prefix": "gridfedrl:"
  },
  "security": {
    "secret_key": "${SECRET_KEY}",
    "encryption_key": "${ENCRYPTION_KEY}",
    "csrf_protection": true,
    "cors_origins": ["https://yourdomain.com"]
  },
  "performance": {
    "max_workers": 8,
    "cache_size": 1024,
    "batch_size": 32,
    "enable_optimization": true
  },
  "monitoring": {
    "prometheus_url": "${PROMETHEUS_URL}",
    "enable_metrics": true,
    "health_check_interval": 30
  },
  "compliance": {
    "regions": ["north_america", "europe", "asia_pacific"],
    "data_retention_days": 90,
    "audit_logging": true
  }
}
```

## 🔒 Security Configuration

### SSL/TLS Setup

```nginx
server {
    listen 443 ssl http2;
    server_name yourdomain.com;
    
    ssl_certificate /etc/ssl/certs/yourdomain.crt;
    ssl_certificate_key /etc/ssl/private/yourdomain.key;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-CHACHA20-POLY1305;
    ssl_prefer_server_ciphers off;
    
    location / {
        proxy_pass http://grid-fed-rl:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Firewall Rules

```bash
# Allow only necessary ports
ufw allow 22/tcp   # SSH
ufw allow 80/tcp   # HTTP
ufw allow 443/tcp  # HTTPS
ufw deny incoming
ufw allow outgoing
ufw enable
```

## 📊 Monitoring & Observability

### Metrics Collection

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'grid-fed-rl'
    static_configs:
      - targets: ['grid-fed-rl:8000']
    metrics_path: /metrics
    scrape_interval: 10s
```

### Alerting Rules

```yaml
# alerts.yml
groups:
- name: grid-fed-rl
  rules:
  - alert: HighErrorRate
    expr: rate(grid_errors_total[5m]) > 0.1
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: High error rate detected
      
  - alert: PowerFlowFailure
    expr: grid_power_flow_failures_total > 10
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: Multiple power flow failures
```

### Health Checks

```python
# Custom health check endpoint
@app.route('/health')
def health_check():
    checks = {
        'database': check_database(),
        'cache': check_cache(),
        'power_flow_solver': check_solver(),
        'compliance': check_compliance()
    }
    
    all_healthy = all(checks.values())
    status_code = 200 if all_healthy else 503
    
    return jsonify({
        'status': 'healthy' if all_healthy else 'unhealthy',
        'checks': checks,
        'timestamp': datetime.now().isoformat()
    }), status_code
```

## 🌍 Multi-Region Deployment

### Region-Specific Configurations

```yaml
# North America
regions:
  north_america:
    frequency_standard: 60.0
    voltage_standards: [120, 240, 480, 4160, 13800]
    compliance: ["NERC", "IEEE", "FERC"]
    locale: "en_US"
    timezone: "America/New_York"
    
# Europe  
  europe:
    frequency_standard: 50.0
    voltage_standards: [230, 400, 690, 6600, 11000]
    compliance: ["GDPR", "IEC", "CENELEC"]
    locale: "en_GB"
    timezone: "Europe/London"
    
# Asia Pacific
  asia_pacific:
    frequency_standard: 50.0
    voltage_standards: [220, 380, 660, 6600, 11000]
    compliance: ["PDPA", "IEC"]
    locale: "en_US"
    timezone: "Asia/Singapore"
```

### Data Residency

```python
# Region-specific data storage
REGIONAL_DATABASES = {
    'north_america': 'postgresql://na-db:5432/gridfedrl',
    'europe': 'postgresql://eu-db:5432/gridfedrl',
    'asia_pacific': 'postgresql://ap-db:5432/gridfedrl'
}

def get_database_url(region):
    return REGIONAL_DATABASES.get(region, REGIONAL_DATABASES['north_america'])
```

## 🔄 CI/CD Pipeline

### GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]
    tags: ['v*']

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        pip install -e .[dev]
    - name: Run tests
      run: |
        pytest tests/ --cov=grid_fed_rl --cov-report=xml
    - name: Security scan
      run: |
        bandit -r grid_fed_rl/
    
  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Build Docker image
      run: |
        docker build -t grid-fed-rl:${{ github.sha }} .
    - name: Push to registry
      run: |
        echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
        docker push grid-fed-rl:${{ github.sha }}
  
  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: startsWith(github.ref, 'refs/tags/')
    steps:
    - name: Deploy to production
      run: |
        kubectl set image deployment/grid-fed-rl grid-fed-rl=grid-fed-rl:${{ github.sha }}
        kubectl rollout status deployment/grid-fed-rl
```

## 📋 Pre-Deployment Checklist

### Infrastructure
- [ ] Load balancer configured
- [ ] SSL certificates installed
- [ ] Database migrations run
- [ ] Redis cache available
- [ ] Monitoring stack deployed

### Security
- [ ] Security scan passed
- [ ] Secrets properly configured
- [ ] Firewall rules applied
- [ ] Backup strategy implemented
- [ ] Incident response plan ready

### Performance
- [ ] Load testing completed
- [ ] Performance benchmarks met
- [ ] Caching configured
- [ ] Auto-scaling policies set
- [ ] Resource limits defined

### Compliance
- [ ] Privacy policy updated
- [ ] Data retention policies configured
- [ ] Regional compliance verified
- [ ] Audit logging enabled
- [ ] GDPR/CCPA compliance confirmed

### Monitoring
- [ ] Metrics collection configured
- [ ] Alerting rules deployed
- [ ] Dashboards created
- [ ] Health checks enabled
- [ ] Log aggregation setup

## 🆘 Troubleshooting

### Common Issues

**Container fails to start**
```bash
# Check logs
docker logs grid-fed-rl

# Common issues:
# - Missing environment variables
# - Database connection failed  
# - Port conflicts
```

**High memory usage**
```bash
# Monitor memory
docker stats grid-fed-rl

# Solutions:
# - Reduce batch_size
# - Decrease cache_size
# - Scale horizontally
```

**Power flow failures**
```bash
# Check solver configuration
grid-fed-rl test-solver --verbose

# Common causes:
# - Invalid network topology
# - Extreme loading conditions
# - Numerical instability
```

### Performance Optimization

```python
# Optimize for production
OPTIMIZATION_CONFIG = {
    'enable_vectorization': True,
    'enable_caching': True, 
    'enable_parallel': True,
    'max_workers': min(8, cpu_count()),
    'cache_size': 1024,
    'batch_size': 64
}
```

## 📞 Support

- **Documentation**: https://grid-fed-rl-gym.readthedocs.io
- **Issues**: https://github.com/terragonlabs/grid-fed-rl-gym/issues
- **Security**: security@terragonlabs.com
- **Commercial Support**: support@terragonlabs.com

## 📄 License

MIT License - see LICENSE file for details.

---

**Generated with [Claude Code](https://claude.ai/code)**

Co-Authored-By: Claude <noreply@anthropic.com>