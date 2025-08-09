#!/bin/bash

# Grid-Fed-RL-Gym Deployment Script
# Supports Docker Compose and Kubernetes deployments

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DOCKER_IMAGE="grid-fed-rl-gym"
VERSION=${VERSION:-"latest"}
DEPLOYMENT_TYPE=${DEPLOYMENT_TYPE:-"docker"}
ENVIRONMENT=${ENVIRONMENT:-"production"}

# Functions
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

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if [ "$DEPLOYMENT_TYPE" == "docker" ]; then
        if ! command -v docker &> /dev/null; then
            log_error "Docker is not installed"
            exit 1
        fi
        
        if ! command -v docker-compose &> /dev/null; then
            log_error "Docker Compose is not installed"
            exit 1
        fi
        log_success "Docker and Docker Compose are available"
    
    elif [ "$DEPLOYMENT_TYPE" == "kubernetes" ]; then
        if ! command -v kubectl &> /dev/null; then
            log_error "kubectl is not installed"
            exit 1
        fi
        
        if ! kubectl cluster-info &> /dev/null; then
            log_error "Cannot connect to Kubernetes cluster"
            exit 1
        fi
        log_success "Kubernetes cluster is accessible"
    fi
}

build_image() {
    log_info "Building Docker image..."
    docker build -t ${DOCKER_IMAGE}:${VERSION} .
    
    if [ $? -eq 0 ]; then
        log_success "Docker image built successfully"
    else
        log_error "Failed to build Docker image"
        exit 1
    fi
}

run_tests() {
    log_info "Running tests..."
    
    # Build test image
    docker build --target testing -t ${DOCKER_IMAGE}:test .
    
    # Run tests in container
    docker run --rm ${DOCKER_IMAGE}:test
    
    if [ $? -eq 0 ]; then
        log_success "All tests passed"
    else
        log_error "Tests failed"
        exit 1
    fi
}

deploy_docker() {
    log_info "Deploying with Docker Compose..."
    
    # Set environment variables
    export GRID_IMAGE="${DOCKER_IMAGE}:${VERSION}"
    export ENV_FILE=".env.${ENVIRONMENT}"
    
    # Create environment file if it doesn't exist
    if [ ! -f "$ENV_FILE" ]; then
        log_warning "Environment file $ENV_FILE not found, creating default..."
        cat > "$ENV_FILE" << EOF
LOG_LEVEL=INFO
METRICS_ENABLED=true
CACHE_SIZE=1000
MAX_WORKERS=4
EOF
    fi
    
    # Deploy services
    docker-compose --env-file="$ENV_FILE" up -d
    
    if [ $? -eq 0 ]; then
        log_success "Services deployed successfully"
        log_info "Application available at http://localhost:8080"
        log_info "Grafana dashboard at http://localhost:3000 (admin/admin)"
    else
        log_error "Deployment failed"
        exit 1
    fi
}

deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    # Create namespace if it doesn't exist
    kubectl create namespace grid-fed-rl --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply Kubernetes manifests
    kubectl apply -f kubernetes/ -n grid-fed-rl
    
    if [ $? -eq 0 ]; then
        log_success "Kubernetes manifests applied successfully"
        
        # Wait for deployment to be ready
        log_info "Waiting for deployment to be ready..."
        kubectl wait --for=condition=available --timeout=300s deployment/grid-fed-rl-deployment -n grid-fed-rl
        
        # Show service information
        kubectl get services -n grid-fed-rl
        
        log_success "Deployment completed successfully"
    else
        log_error "Kubernetes deployment failed"
        exit 1
    fi
}

health_check() {
    log_info "Performing health check..."
    
    if [ "$DEPLOYMENT_TYPE" == "docker" ]; then
        # Wait for services to be ready
        sleep 30
        
        # Check main service
        if curl -f http://localhost:8080/health &> /dev/null; then
            log_success "Health check passed"
        else
            log_error "Health check failed"
            docker-compose logs grid-fed-rl
            exit 1
        fi
        
    elif [ "$DEPLOYMENT_TYPE" == "kubernetes" ]; then
        # Check pod status
        kubectl get pods -n grid-fed-rl
        
        # Port forward for health check
        kubectl port-forward service/grid-fed-rl-service 8080:80 -n grid-fed-rl &
        PORT_FORWARD_PID=$!
        sleep 10
        
        if curl -f http://localhost:8080/health &> /dev/null; then
            log_success "Health check passed"
        else
            log_error "Health check failed"
            kubectl logs -l app=grid-fed-rl -n grid-fed-rl
            kill $PORT_FORWARD_PID
            exit 1
        fi
        
        kill $PORT_FORWARD_PID
    fi
}

cleanup() {
    log_info "Cleaning up..."
    
    if [ "$DEPLOYMENT_TYPE" == "docker" ]; then
        docker-compose down
        docker system prune -f
        
    elif [ "$DEPLOYMENT_TYPE" == "kubernetes" ]; then
        kubectl delete namespace grid-fed-rl --ignore-not-found=true
    fi
    
    log_success "Cleanup completed"
}

show_help() {
    echo "Grid-Fed-RL-Gym Deployment Script"
    echo ""
    echo "Usage: $0 [OPTIONS] COMMAND"
    echo ""
    echo "Commands:"
    echo "  build       Build Docker image"
    echo "  test        Run tests"
    echo "  deploy      Deploy application"
    echo "  health      Run health check"
    echo "  cleanup     Clean up deployment"
    echo "  help        Show this help message"
    echo ""
    echo "Options:"
    echo "  --type TYPE       Deployment type (docker|kubernetes) [default: docker]"
    echo "  --env ENV         Environment (development|production) [default: production]"
    echo "  --version VER     Image version [default: latest]"
    echo ""
    echo "Environment variables:"
    echo "  DEPLOYMENT_TYPE   Same as --type"
    echo "  ENVIRONMENT      Same as --env"
    echo "  VERSION          Same as --version"
    echo ""
    echo "Examples:"
    echo "  $0 build"
    echo "  $0 --type docker deploy"
    echo "  $0 --type kubernetes --env production deploy"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --type)
            DEPLOYMENT_TYPE="$2"
            shift 2
            ;;
        --env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        build|test|deploy|health|cleanup|help)
            COMMAND="$1"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate inputs
if [ -z "$COMMAND" ]; then
    log_error "No command specified"
    show_help
    exit 1
fi

if [ "$DEPLOYMENT_TYPE" != "docker" ] && [ "$DEPLOYMENT_TYPE" != "kubernetes" ]; then
    log_error "Invalid deployment type: $DEPLOYMENT_TYPE"
    show_help
    exit 1
fi

# Execute command
case $COMMAND in
    build)
        check_prerequisites
        build_image
        ;;
    test)
        check_prerequisites
        run_tests
        ;;
    deploy)
        check_prerequisites
        build_image
        if [ "$DEPLOYMENT_TYPE" == "docker" ]; then
            deploy_docker
        else
            deploy_kubernetes
        fi
        health_check
        ;;
    health)
        health_check
        ;;
    cleanup)
        cleanup
        ;;
    help)
        show_help
        ;;
    *)
        log_error "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac

log_success "Command '$COMMAND' completed successfully"