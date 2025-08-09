# Production Deployment Guide

This guide covers deploying Grid-Fed-RL-Gym to production environments with Docker, Kubernetes, and cloud platforms.

## Quick Start

### Docker Deployment (Recommended for small scale)

```bash
# Build and deploy with Docker Compose
./scripts/deploy.sh deploy

# Or manually:
docker-compose up -d
```

### Kubernetes Deployment (Recommended for production scale)

```bash
# Deploy to Kubernetes cluster
./scripts/deploy.sh --type kubernetes deploy

# Or manually:
kubectl apply -f kubernetes/
```

## Prerequisites

### System Requirements

- **CPU**: Minimum 2 cores, Recommended 4+ cores
- **Memory**: Minimum 4GB, Recommended 8GB+  
- **Storage**: 50GB+ SSD storage
- **Network**: 1Gbps+ for federated learning scenarios

### Software Dependencies

- Docker 20.10+ and Docker Compose 2.0+
- Kubernetes 1.20+ (for K8s deployment)
- kubectl configured with cluster access
- Python 3.9+ (for development)

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `METRICS_ENABLED` | `true` | Enable Prometheus metrics |
| `CACHE_SIZE` | `1000` | In-memory cache size |
| `MAX_WORKERS` | `4` | Maximum worker processes |
| `DB_URL` | `sqlite:///data/grid.db` | Database connection URL |
| `REDIS_URL` | `redis://redis:6379` | Redis cache URL |

### Configuration Files

- `deployment_config.yaml` - Main deployment configuration
- `.env.production` - Production environment variables
- `.env.development` - Development environment variables

## Docker Deployment

### Production Setup

1. **Build the production image**:
   ```bash
   docker build --target production -t grid-fed-rl-gym:latest .
   ```

2. **Start all services**:
   ```bash
   docker-compose up -d
   ```

3. **Verify deployment**:
   ```bash
   curl http://localhost:8080/health
   ```

### Services Included

- **grid-fed-rl**: Main application (port 8080)
- **redis**: Caching and session storage (port 6379)  
- **prometheus**: Metrics collection (port 9091)
- **grafana**: Monitoring dashboard (port 3000)
- **nginx**: Load balancer (ports 80/443)

### Monitoring

- **Application**: http://localhost:8080
- **Health Check**: http://localhost:8080/health
- **Metrics**: http://localhost:8080/metrics
- **Grafana Dashboard**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9091

## Kubernetes Deployment

### Cluster Requirements

- Kubernetes 1.20+
- Ingress controller (nginx recommended)
- StorageClass for persistent volumes
- cert-manager for TLS certificates (optional)

### Deployment Steps

1. **Apply Kubernetes manifests**:
   ```bash
   kubectl apply -f kubernetes/
   ```

2. **Check deployment status**:
   ```bash
   kubectl get pods -l app=grid-fed-rl
   kubectl get services
   ```

3. **Access the application**:
   ```bash
   kubectl port-forward service/grid-fed-rl-service 8080:80
   ```

### Scaling

The deployment includes Horizontal Pod Autoscaler (HPA) that automatically scales based on:
- CPU utilization (target: 70%)
- Memory utilization (target: 80%)
- Min replicas: 2
- Max replicas: 10

### Persistent Storage

The deployment creates a PersistentVolumeClaim for:
- Application data: 10GB
- Logs and cache: EmptyDir volumes

## Cloud Platform Deployment

### AWS EKS

```bash
# Create EKS cluster
eksctl create cluster --name grid-fed-rl-cluster --version 1.21 --region us-west-2 --nodegroup-name workers --node-type m5.large --nodes 3

# Deploy application
kubectl apply -f kubernetes/

# Configure ALB ingress
kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/aws-load-balancer-controller/v2.4.1/docs/install/iam_policy.json
```

### Google GKE

```bash
# Create GKE cluster  
gcloud container clusters create grid-fed-rl-cluster --zone us-central1-a --num-nodes 3

# Get cluster credentials
gcloud container clusters get-credentials grid-fed-rl-cluster --zone us-central1-a

# Deploy application
kubectl apply -f kubernetes/
```

### Azure AKS

```bash
# Create AKS cluster
az aks create --resource-group myResourceGroup --name grid-fed-rl-cluster --node-count 3 --enable-addons monitoring

# Get cluster credentials  
az aks get-credentials --resource-group myResourceGroup --name grid-fed-rl-cluster

# Deploy application
kubectl apply -f kubernetes/
```

## Security Considerations

### Container Security

- Runs as non-root user (UID 1000)
- Read-only root filesystem where possible
- No privilege escalation allowed
- Security scanning enabled in CI/CD

### Network Security

- All services communicate over internal networks
- TLS encryption for external communications
- Ingress controller with SSL/TLS termination
- Network policies for pod-to-pod communication

### Secrets Management

- Use Kubernetes secrets for sensitive data
- External secret management (AWS Secrets Manager, etc.)
- Regular credential rotation
- No secrets in container images

## Performance Tuning

### Resource Limits

```yaml
resources:
  requests:
    cpu: "500m"
    memory: "1Gi"
  limits:
    cpu: "2000m"
    memory: "4Gi"
```

### Caching Strategy

- Redis for distributed caching
- Local in-memory caching
- Power flow computation caching
- Result memoization

### Database Optimization

- Connection pooling enabled
- Query optimization
- Proper indexing
- Read replicas for scaling

## Monitoring and Observability

### Metrics

- **Application Metrics**: Custom business metrics via Prometheus
- **System Metrics**: CPU, memory, disk, network via node-exporter
- **Container Metrics**: Container resource usage via cAdvisor

### Logging

- Structured JSON logging
- Log aggregation with ELK stack or similar
- Log rotation and retention policies
- Centralized logging for distributed deployments

### Tracing

- Distributed tracing with Jaeger or Zipkin
- Request flow monitoring
- Performance bottleneck identification
- Error tracking and debugging

### Alerting

- Critical service health alerts
- Resource utilization alerts  
- Business metric alerts
- Custom federated learning alerts

## Backup and Recovery

### Data Backup

```bash
# Backup persistent volumes
kubectl exec -n grid-fed-rl deployment/grid-fed-rl-deployment -- tar -czf - /app/data > backup-$(date +%Y%m%d).tar.gz

# Backup configuration
kubectl get configmaps,secrets -n grid-fed-rl -o yaml > config-backup.yaml
```

### Disaster Recovery

1. **Infrastructure**: Infrastructure as Code with Terraform/CloudFormation
2. **Data**: Regular automated backups to cloud storage
3. **Configuration**: GitOps approach with version control
4. **Testing**: Regular disaster recovery testing

## Troubleshooting

### Common Issues

1. **Pod CrashLoopBackOff**:
   ```bash
   kubectl logs -l app=grid-fed-rl --previous
   kubectl describe pod <pod-name>
   ```

2. **Service Not Accessible**:
   ```bash
   kubectl get endpoints
   kubectl get ingress
   ```

3. **High Resource Usage**:
   ```bash
   kubectl top pods
   kubectl top nodes
   ```

### Debug Mode

```bash
# Enable debug logging
kubectl set env deployment/grid-fed-rl-deployment LOG_LEVEL=DEBUG

# Access debug endpoints
curl http://localhost:8080/debug/health
curl http://localhost:8080/debug/metrics
```

### Performance Issues

1. Check resource limits and requests
2. Monitor cache hit rates  
3. Review database query performance
4. Analyze distributed training communication

## Maintenance

### Updates

```bash
# Rolling update
kubectl set image deployment/grid-fed-rl-deployment grid-fed-rl=grid-fed-rl-gym:v1.1.0

# Monitor rollout
kubectl rollout status deployment/grid-fed-rl-deployment

# Rollback if needed
kubectl rollout undo deployment/grid-fed-rl-deployment
```

### Health Checks

- Liveness probe: Ensures container is running
- Readiness probe: Ensures container is ready to serve traffic
- Startup probe: Handles slow-starting containers

### Scaling

```bash
# Manual scaling
kubectl scale deployment grid-fed-rl-deployment --replicas=5

# Autoscaling configuration
kubectl autoscale deployment grid-fed-rl-deployment --min=2 --max=10 --cpu-percent=70
```

## Production Checklist

### Pre-deployment

- [ ] Security scan completed
- [ ] Performance tests passed  
- [ ] Load tests executed
- [ ] Backup strategy implemented
- [ ] Monitoring configured
- [ ] Alerts configured
- [ ] Documentation updated
- [ ] Team training completed

### Post-deployment

- [ ] Health checks passing
- [ ] Metrics collection working
- [ ] Alerts functional
- [ ] Performance within SLA
- [ ] Security scan clean
- [ ] User acceptance testing
- [ ] Backup verification
- [ ] Documentation updated

### Ongoing Maintenance

- [ ] Regular security updates
- [ ] Performance monitoring
- [ ] Capacity planning
- [ ] Disaster recovery testing
- [ ] Documentation maintenance
- [ ] Team knowledge sharing

## Support

For production support:

- **Documentation**: See [docs/](docs/) directory
- **Issues**: https://github.com/terragonlabs/grid-fed-rl-gym/issues
- **Discussions**: https://github.com/terragonlabs/grid-fed-rl-gym/discussions
- **Email**: support@terragonlabs.com

## Compliance

This deployment guide ensures compliance with:

- **GDPR**: Data privacy and user rights
- **CCPA**: California privacy regulations  
- **PDPA**: Singapore data protection
- **SOC 2**: Security and availability controls
- **ISO 27001**: Information security management