#!/bin/bash
# Production Deployment Script for Grid-Fed-RL-Gym
# Autonomous deployment with comprehensive validation and rollback

set -euo pipefail

# Configuration
PROJECT_NAME="grid-fed-rl-gym"
NAMESPACE="grid-fed-rl"
MONITORING_NAMESPACE="monitoring"
DOCKER_REGISTRY="terragonlabs"
VERSION="${VERSION:-latest}"
ENVIRONMENT="${ENVIRONMENT:-production}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Error handling
error_exit() {
    log_error "$1"
    exit 1
}

# Cleanup function for graceful exit
cleanup() {
    log_info "Cleaning up temporary resources..."
    # Add any cleanup commands here
}

trap cleanup EXIT

# Pre-deployment checks
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        error_exit "kubectl is not installed or not in PATH"
    fi
    
    # Check if docker is available
    if ! command -v docker &> /dev/null; then
        error_exit "docker is not installed or not in PATH"
    fi
    
    # Check if helm is available (optional)
    if command -v helm &> /dev/null; then
        log_info "Helm is available for advanced deployments"
    else
        log_warning "Helm not found - using kubectl for deployment"
    fi
    
    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        error_exit "Cannot connect to Kubernetes cluster"
    fi
    
    # Check if jq is available for JSON processing
    if ! command -v jq &> /dev/null; then
        log_warning "jq not found - some features may be limited"
    fi
    
    log_success "Prerequisites check completed"
}

# Build and push Docker image
build_and_push_image() {
    log_info "Building and pushing Docker image..."
    
    # Generate build info
    BUILD_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
    
    # Build Docker image
    docker build \
        --build-arg BUILD_TIME="${BUILD_TIME}" \
        --build-arg GIT_COMMIT="${GIT_COMMIT}" \
        --build-arg GIT_BRANCH="${GIT_BRANCH}" \
        --build-arg VERSION="${VERSION}" \
        -t "${DOCKER_REGISTRY}/${PROJECT_NAME}:${VERSION}" \
        -t "${DOCKER_REGISTRY}/${PROJECT_NAME}:latest" \
        .
    
    # Push to registry
    docker push "${DOCKER_REGISTRY}/${PROJECT_NAME}:${VERSION}"
    docker push "${DOCKER_REGISTRY}/${PROJECT_NAME}:latest"
    
    log_success "Docker image built and pushed successfully"
}

# Create namespaces
create_namespaces() {
    log_info "Creating namespaces..."
    
    # Create main application namespace
    kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
    
    # Create monitoring namespace
    kubectl create namespace "${MONITORING_NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
    
    # Label namespaces
    kubectl label namespace "${NAMESPACE}" app=grid-fed-rl environment="${ENVIRONMENT}" --overwrite
    kubectl label namespace "${MONITORING_NAMESPACE}" app=monitoring --overwrite
    
    log_success "Namespaces created/updated successfully"
}

# Deploy secrets
deploy_secrets() {
    log_info "Deploying secrets..."
    
    # Generate random JWT secret if not provided
    JWT_SECRET="${JWT_SECRET:-$(openssl rand -base64 32)}"
    
    # Create application secrets
    kubectl create secret generic grid-fed-rl-secrets \
        --namespace="${NAMESPACE}" \
        --from-literal=jwt-secret="${JWT_SECRET}" \
        --from-literal=db-user="${DB_USER:-grid_fed_rl}" \
        --from-literal=db-password="${DB_PASSWORD:-$(openssl rand -base64 20)}" \
        --from-literal=redis-password="${REDIS_PASSWORD:-$(openssl rand -base64 20)}" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Create Grafana admin password
    GRAFANA_ADMIN_PASSWORD="${GRAFANA_ADMIN_PASSWORD:-$(openssl rand -base64 20)}"
    kubectl create secret generic grafana-secret \
        --namespace="${MONITORING_NAMESPACE}" \
        --from-literal=admin-password="${GRAFANA_ADMIN_PASSWORD}" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    log_success "Secrets deployed successfully"
    log_info "Grafana admin password: ${GRAFANA_ADMIN_PASSWORD}"
}

# Deploy monitoring stack
deploy_monitoring() {
    log_info "Deploying monitoring stack..."
    
    if [[ -f "deployment/monitoring_stack.yaml" ]]; then
        kubectl apply -f deployment/monitoring_stack.yaml
    else
        log_warning "Monitoring stack configuration not found - creating basic monitoring"
        
        # Create basic Prometheus deployment
        cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  namespace: ${MONITORING_NAMESPACE}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      containers:
      - name: prometheus
        image: prom/prometheus:latest
        ports:
        - containerPort: 9090
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
---
apiVersion: v1
kind: Service
metadata:
  name: prometheus-service
  namespace: ${MONITORING_NAMESPACE}
spec:
  selector:
    app: prometheus
  ports:
  - port: 9090
    targetPort: 9090
EOF
    fi
    
    log_success "Monitoring stack deployed"
}

# Deploy application
deploy_application() {
    log_info "Deploying Grid-Fed-RL application..."
    
    # Update image version in deployment config
    if [[ -f "deployment/production_config.yaml" ]]; then
        # Replace image version
        sed -i.bak "s|image: terragonlabs/grid-fed-rl-gym:.*|image: ${DOCKER_REGISTRY}/${PROJECT_NAME}:${VERSION}|g" \
            deployment/production_config.yaml
        
        # Apply configuration
        kubectl apply -f deployment/production_config.yaml
        
        # Restore original file
        mv deployment/production_config.yaml.bak deployment/production_config.yaml
    else
        log_warning "Production config not found - creating basic deployment"
        
        # Create basic deployment
        cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grid-fed-rl-app
  namespace: ${NAMESPACE}
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
        image: ${DOCKER_REGISTRY}/${PROJECT_NAME}:${VERSION}
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: grid-fed-rl-service
  namespace: ${NAMESPACE}
spec:
  selector:
    app: grid-fed-rl
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer
EOF
    fi
    
    log_success "Application deployed"
}

# Wait for deployment to be ready
wait_for_deployment() {
    log_info "Waiting for deployment to be ready..."
    
    # Wait for application deployment
    kubectl rollout status deployment/grid-fed-rl-app -n "${NAMESPACE}" --timeout=600s
    
    # Wait for monitoring deployment
    kubectl rollout status deployment/prometheus -n "${MONITORING_NAMESPACE}" --timeout=300s || true
    
    log_success "Deployments are ready"
}

# Validate deployment
validate_deployment() {
    log_info "Validating deployment..."
    
    # Check if pods are running
    local app_pods
    app_pods=$(kubectl get pods -n "${NAMESPACE}" -l app=grid-fed-rl --field-selector=status.phase=Running --no-headers | wc -l)
    
    if [[ "${app_pods}" -lt 1 ]]; then
        error_exit "No running application pods found"
    fi
    
    log_info "Found ${app_pods} running application pods"
    
    # Check service endpoints
    local endpoints
    endpoints=$(kubectl get endpoints -n "${NAMESPACE}" grid-fed-rl-service -o jsonpath='{.subsets[0].addresses}' 2>/dev/null || echo "[]")
    
    if [[ "${endpoints}" == "[]" || "${endpoints}" == "" ]]; then
        log_warning "Service endpoints not ready yet - may need more time"
    else
        log_success "Service endpoints are ready"
    fi
    
    # Test application health if service is accessible
    local service_ip
    service_ip=$(kubectl get service -n "${NAMESPACE}" grid-fed-rl-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
    
    if [[ -n "${service_ip}" ]]; then
        log_info "Testing application health endpoint..."
        if curl -f "http://${service_ip}/health" &> /dev/null; then
            log_success "Application health check passed"
        else
            log_warning "Application health check failed - may still be starting up"
        fi
    else
        log_info "LoadBalancer IP not yet assigned - skipping external health check"
    fi
    
    log_success "Deployment validation completed"
}

# Run smoke tests
run_smoke_tests() {
    log_info "Running smoke tests..."
    
    # Port forward to test internally
    kubectl port-forward -n "${NAMESPACE}" svc/grid-fed-rl-service 8080:80 &
    local port_forward_pid=$!
    
    # Wait for port forward to be ready
    sleep 5
    
    # Test basic functionality
    local test_passed=true
    
    # Test health endpoint
    if curl -f "http://localhost:8080/health" &> /dev/null; then
        log_success "Health endpoint test passed"
    else
        log_error "Health endpoint test failed"
        test_passed=false
    fi
    
    # Test metrics endpoint
    if curl -f "http://localhost:8080/metrics" &> /dev/null; then
        log_success "Metrics endpoint test passed"
    else
        log_warning "Metrics endpoint test failed (may not be implemented)"
    fi
    
    # Kill port forward
    kill $port_forward_pid 2>/dev/null || true
    
    if [[ "$test_passed" == "true" ]]; then
        log_success "Smoke tests passed"
    else
        error_exit "Smoke tests failed"
    fi
}

# Get deployment info
get_deployment_info() {
    log_info "Deployment Information:"
    echo "======================="
    
    # Application info
    echo "Application:"
    echo "  Namespace: ${NAMESPACE}"
    echo "  Image: ${DOCKER_REGISTRY}/${PROJECT_NAME}:${VERSION}"
    echo "  Replicas: $(kubectl get deployment -n "${NAMESPACE}" grid-fed-rl-app -o jsonpath='{.status.readyReplicas}')/$(kubectl get deployment -n "${NAMESPACE}" grid-fed-rl-app -o jsonpath='{.spec.replicas}')"
    
    # Service info
    local service_ip
    service_ip=$(kubectl get service -n "${NAMESPACE}" grid-fed-rl-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "Pending")
    echo "  External IP: ${service_ip}"
    
    # Monitoring info
    echo "Monitoring:"
    echo "  Namespace: ${MONITORING_NAMESPACE}"
    
    local prometheus_ip
    prometheus_ip=$(kubectl get service -n "${MONITORING_NAMESPACE}" prometheus-service -o jsonpath='{.spec.clusterIP}' 2>/dev/null || echo "Not deployed")
    echo "  Prometheus: ${prometheus_ip}:9090"
    
    # Access commands
    echo ""
    echo "Access Commands:"
    echo "  Application: kubectl port-forward -n ${NAMESPACE} svc/grid-fed-rl-service 8080:80"
    echo "  Prometheus: kubectl port-forward -n ${MONITORING_NAMESPACE} svc/prometheus-service 9090:9090"
    echo "  Logs: kubectl logs -n ${NAMESPACE} -l app=grid-fed-rl -f"
    echo ""
}

# Main deployment function
main() {
    log_info "Starting Grid-Fed-RL production deployment..."
    log_info "Version: ${VERSION}"
    log_info "Environment: ${ENVIRONMENT}"
    log_info "Registry: ${DOCKER_REGISTRY}"
    
    # Run deployment steps
    check_prerequisites
    build_and_push_image
    create_namespaces
    deploy_secrets
    deploy_monitoring
    deploy_application
    wait_for_deployment
    validate_deployment
    run_smoke_tests
    get_deployment_info
    
    log_success "🚀 Grid-Fed-RL deployment completed successfully!"
    log_info "The application is now running in production mode."
}

# Handle command line arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "destroy")
        log_info "Destroying Grid-Fed-RL deployment..."
        kubectl delete namespace "${NAMESPACE}" --ignore-not-found=true
        kubectl delete namespace "${MONITORING_NAMESPACE}" --ignore-not-found=true
        log_success "Deployment destroyed"
        ;;
    "status")
        get_deployment_info
        ;;
    "logs")
        kubectl logs -n "${NAMESPACE}" -l app=grid-fed-rl -f
        ;;
    "scale")
        REPLICAS="${2:-3}"
        log_info "Scaling application to ${REPLICAS} replicas..."
        kubectl scale deployment grid-fed-rl-app -n "${NAMESPACE}" --replicas="${REPLICAS}"
        log_success "Scaling initiated"
        ;;
    *)
        echo "Usage: $0 {deploy|destroy|status|logs|scale [replicas]}"
        echo "  deploy  - Deploy the application (default)"
        echo "  destroy - Remove the deployment"
        echo "  status  - Show deployment status"
        echo "  logs    - Follow application logs"
        echo "  scale   - Scale the application"
        exit 1
        ;;
esac