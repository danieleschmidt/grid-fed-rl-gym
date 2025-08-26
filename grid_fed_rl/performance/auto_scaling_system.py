"""Advanced auto-scaling system with container orchestration and cloud-native scaling patterns."""

import asyncio
import time
import threading
import logging
import json
import subprocess
import psutil
import socket
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque, defaultdict
import yaml
import requests
from datetime import datetime, timedelta
import concurrent.futures
import weakref

try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    docker = None
    DOCKER_AVAILABLE = False

try:
    import kubernetes
    from kubernetes import client, config
    KUBERNETES_AVAILABLE = True
except ImportError:
    kubernetes = None
    client = None
    config = None
    KUBERNETES_AVAILABLE = False

try:
    import boto3
    AWS_AVAILABLE = True
except ImportError:
    boto3 = None
    AWS_AVAILABLE = False

logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Scaling direction."""
    UP = "up"
    DOWN = "down"
    MAINTAIN = "maintain"


class ScalingTrigger(Enum):
    """Triggers for scaling decisions."""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    QUEUE_DEPTH = "queue_depth"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    CUSTOM_METRIC = "custom_metric"
    PREDICTIVE = "predictive"


class OrchestrationPlatform(Enum):
    """Supported orchestration platforms."""
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    DOCKER_SWARM = "docker_swarm"
    AWS_ECS = "aws_ecs"
    AWS_LAMBDA = "aws_lambda"
    LOCAL_PROCESS = "local_process"


class ScalingPolicy(Enum):
    """Scaling policy types."""
    REACTIVE = "reactive"          # React to current conditions
    PREDICTIVE = "predictive"      # Predict future needs
    SCHEDULED = "scheduled"        # Time-based scaling
    HYBRID = "hybrid"             # Combination approach


@dataclass
class ScalingMetrics:
    """Metrics used for scaling decisions."""
    timestamp: float
    cpu_utilization: float
    memory_utilization: float
    memory_available_gb: float
    active_connections: int
    queue_depth: int
    avg_response_time_ms: float
    requests_per_second: float
    error_rate: float
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ScalingThresholds:
    """Thresholds for scaling triggers."""
    scale_up_cpu: float = 70.0        # CPU % to trigger scale up
    scale_down_cpu: float = 30.0      # CPU % to trigger scale down
    scale_up_memory: float = 80.0     # Memory % to trigger scale up
    scale_down_memory: float = 40.0   # Memory % to trigger scale down
    scale_up_queue_depth: int = 50    # Queue depth to trigger scale up
    scale_down_queue_depth: int = 5   # Queue depth to trigger scale down
    scale_up_response_time: float = 1000.0  # Response time (ms) to trigger scale up
    scale_down_response_time: float = 200.0  # Response time (ms) to trigger scale down
    min_instances: int = 1
    max_instances: int = 20
    cooldown_period: float = 300.0    # Seconds between scaling actions


@dataclass
class ContainerSpec:
    """Specification for container deployment."""
    image: str
    name: str
    ports: List[int] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)
    resource_limits: Dict[str, str] = field(default_factory=dict)
    resource_requests: Dict[str, str] = field(default_factory=dict)
    command: Optional[List[str]] = None
    volumes: List[Dict[str, str]] = field(default_factory=list)
    health_check: Optional[Dict[str, Any]] = None


@dataclass
class ScalingEvent:
    """Record of a scaling event."""
    timestamp: float
    direction: ScalingDirection
    trigger: ScalingTrigger
    old_count: int
    new_count: int
    trigger_value: float
    threshold: float
    duration: float = 0.0
    success: bool = True
    error_message: Optional[str] = None


class MetricsCollector:
    """Collects system and application metrics for scaling decisions."""
    
    def __init__(self, collection_interval: float = 10.0):
        self.collection_interval = collection_interval
        self.metrics_history = deque(maxlen=1000)
        self.custom_collectors: List[Callable[[], Dict[str, float]]] = []
        self.collecting = False
        self.collection_thread: Optional[threading.Thread] = None
        
    def start_collection(self) -> None:
        """Start metrics collection."""
        if self.collecting:
            return
        
        self.collecting = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        logger.info("Metrics collection started")
    
    def stop_collection(self) -> None:
        """Stop metrics collection."""
        self.collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5.0)
    
    def _collection_loop(self) -> None:
        """Main metrics collection loop."""
        while self.collecting:
            try:
                metrics = self._collect_system_metrics()
                
                # Collect custom metrics
                for collector in self.custom_collectors:
                    try:
                        custom = collector()
                        metrics.custom_metrics.update(custom)
                    except Exception as e:
                        logger.warning(f"Custom metrics collection failed: {e}")
                
                self.metrics_history.append(metrics)
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self) -> ScalingMetrics:
        """Collect system-level metrics."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Network connections (approximate active connections)
        try:
            connections = len(psutil.net_connections(kind='inet'))
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            connections = 0
        
        return ScalingMetrics(
            timestamp=time.time(),
            cpu_utilization=cpu_percent,
            memory_utilization=memory.percent,
            memory_available_gb=memory.available / (1024**3),
            active_connections=connections,
            queue_depth=0,  # Will be set by application
            avg_response_time_ms=0.0,  # Will be set by application
            requests_per_second=0.0,  # Will be set by application
            error_rate=0.0  # Will be set by application
        )
    
    def add_custom_collector(self, collector: Callable[[], Dict[str, float]]) -> None:
        """Add custom metrics collector."""
        self.custom_collectors.append(collector)
    
    def update_app_metrics(
        self,
        queue_depth: int = 0,
        avg_response_time_ms: float = 0.0,
        requests_per_second: float = 0.0,
        error_rate: float = 0.0
    ) -> None:
        """Update application-specific metrics."""
        if self.metrics_history:
            latest = self.metrics_history[-1]
            latest.queue_depth = queue_depth
            latest.avg_response_time_ms = avg_response_time_ms
            latest.requests_per_second = requests_per_second
            latest.error_rate = error_rate
    
    def get_latest_metrics(self) -> Optional[ScalingMetrics]:
        """Get the most recent metrics."""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_average_metrics(self, window_seconds: float = 300.0) -> Optional[ScalingMetrics]:
        """Get average metrics over a time window."""
        if not self.metrics_history:
            return None
        
        cutoff_time = time.time() - window_seconds
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return None
        
        # Calculate averages
        avg_metrics = ScalingMetrics(
            timestamp=time.time(),
            cpu_utilization=sum(m.cpu_utilization for m in recent_metrics) / len(recent_metrics),
            memory_utilization=sum(m.memory_utilization for m in recent_metrics) / len(recent_metrics),
            memory_available_gb=sum(m.memory_available_gb for m in recent_metrics) / len(recent_metrics),
            active_connections=int(sum(m.active_connections for m in recent_metrics) / len(recent_metrics)),
            queue_depth=int(sum(m.queue_depth for m in recent_metrics) / len(recent_metrics)),
            avg_response_time_ms=sum(m.avg_response_time_ms for m in recent_metrics) / len(recent_metrics),
            requests_per_second=sum(m.requests_per_second for m in recent_metrics) / len(recent_metrics),
            error_rate=sum(m.error_rate for m in recent_metrics) / len(recent_metrics)
        )
        
        return avg_metrics


class ScalingDecisionEngine:
    """Makes intelligent scaling decisions based on metrics and policies."""
    
    def __init__(self, thresholds: ScalingThresholds, policy: ScalingPolicy = ScalingPolicy.REACTIVE):
        self.thresholds = thresholds
        self.policy = policy
        self.scaling_history = deque(maxlen=1000)
        self.last_scaling_time = 0.0
        self.prediction_model = None
        
    def evaluate_scaling_need(
        self, 
        current_metrics: ScalingMetrics, 
        current_instances: int
    ) -> Tuple[ScalingDirection, ScalingTrigger, float]:
        """Evaluate if scaling is needed and determine direction."""
        
        # Check cooldown period
        if time.time() - self.last_scaling_time < self.thresholds.cooldown_period:
            return ScalingDirection.MAINTAIN, ScalingTrigger.CPU_UTILIZATION, 0.0
        
        # Evaluate different triggers
        scale_decisions = []
        
        # CPU utilization
        if current_metrics.cpu_utilization > self.thresholds.scale_up_cpu:
            if current_instances < self.thresholds.max_instances:
                scale_decisions.append((ScalingDirection.UP, ScalingTrigger.CPU_UTILIZATION, current_metrics.cpu_utilization))
        elif current_metrics.cpu_utilization < self.thresholds.scale_down_cpu:
            if current_instances > self.thresholds.min_instances:
                scale_decisions.append((ScalingDirection.DOWN, ScalingTrigger.CPU_UTILIZATION, current_metrics.cpu_utilization))
        
        # Memory utilization
        if current_metrics.memory_utilization > self.thresholds.scale_up_memory:
            if current_instances < self.thresholds.max_instances:
                scale_decisions.append((ScalingDirection.UP, ScalingTrigger.MEMORY_UTILIZATION, current_metrics.memory_utilization))
        elif current_metrics.memory_utilization < self.thresholds.scale_down_memory:
            if current_instances > self.thresholds.min_instances:
                scale_decisions.append((ScalingDirection.DOWN, ScalingTrigger.MEMORY_UTILIZATION, current_metrics.memory_utilization))
        
        # Queue depth
        if current_metrics.queue_depth > self.thresholds.scale_up_queue_depth:
            if current_instances < self.thresholds.max_instances:
                scale_decisions.append((ScalingDirection.UP, ScalingTrigger.QUEUE_DEPTH, current_metrics.queue_depth))
        elif current_metrics.queue_depth < self.thresholds.scale_down_queue_depth:
            if current_instances > self.thresholds.min_instances:
                scale_decisions.append((ScalingDirection.DOWN, ScalingTrigger.QUEUE_DEPTH, current_metrics.queue_depth))
        
        # Response time
        if current_metrics.avg_response_time_ms > self.thresholds.scale_up_response_time:
            if current_instances < self.thresholds.max_instances:
                scale_decisions.append((ScalingDirection.UP, ScalingTrigger.RESPONSE_TIME, current_metrics.avg_response_time_ms))
        elif current_metrics.avg_response_time_ms < self.thresholds.scale_down_response_time:
            if current_instances > self.thresholds.min_instances:
                scale_decisions.append((ScalingDirection.DOWN, ScalingTrigger.RESPONSE_TIME, current_metrics.avg_response_time_ms))
        
        # Apply policy-specific logic
        if self.policy == ScalingPolicy.PREDICTIVE:
            return self._apply_predictive_scaling(scale_decisions, current_metrics, current_instances)
        elif self.policy == ScalingPolicy.SCHEDULED:
            return self._apply_scheduled_scaling(scale_decisions, current_metrics, current_instances)
        else:
            return self._apply_reactive_scaling(scale_decisions)
    
    def _apply_reactive_scaling(
        self, 
        scale_decisions: List[Tuple[ScalingDirection, ScalingTrigger, float]]
    ) -> Tuple[ScalingDirection, ScalingTrigger, float]:
        """Apply reactive scaling logic."""
        if not scale_decisions:
            return ScalingDirection.MAINTAIN, ScalingTrigger.CPU_UTILIZATION, 0.0
        
        # Prioritize scale up decisions
        up_decisions = [d for d in scale_decisions if d[0] == ScalingDirection.UP]
        if up_decisions:
            return up_decisions[0]  # Take the first scale up decision
        
        # Otherwise take scale down decisions
        down_decisions = [d for d in scale_decisions if d[0] == ScalingDirection.DOWN]
        if down_decisions:
            return down_decisions[0]
        
        return ScalingDirection.MAINTAIN, ScalingTrigger.CPU_UTILIZATION, 0.0
    
    def _apply_predictive_scaling(
        self,
        scale_decisions: List[Tuple[ScalingDirection, ScalingTrigger, float]],
        current_metrics: ScalingMetrics,
        current_instances: int
    ) -> Tuple[ScalingDirection, ScalingTrigger, float]:
        """Apply predictive scaling using historical trends."""
        # Simple trend analysis - in production would use ML models
        if len(self.scaling_history) < 10:
            return self._apply_reactive_scaling(scale_decisions)
        
        # Analyze recent CPU trend
        recent_events = list(self.scaling_history)[-10:]
        cpu_trend = []
        for event in recent_events:
            if hasattr(event, 'trigger_value'):
                cpu_trend.append(event.trigger_value)
        
        if len(cpu_trend) >= 3:
            # Simple linear trend
            trend = (cpu_trend[-1] - cpu_trend[0]) / len(cpu_trend)
            
            # If trend is increasing and we're not scaling up, consider proactive scaling
            if trend > 5.0 and current_instances < self.thresholds.max_instances:
                return ScalingDirection.UP, ScalingTrigger.PREDICTIVE, trend
            elif trend < -5.0 and current_instances > self.thresholds.min_instances:
                return ScalingDirection.DOWN, ScalingTrigger.PREDICTIVE, trend
        
        return self._apply_reactive_scaling(scale_decisions)
    
    def _apply_scheduled_scaling(
        self,
        scale_decisions: List[Tuple[ScalingDirection, ScalingTrigger, float]],
        current_metrics: ScalingMetrics,
        current_instances: int
    ) -> Tuple[ScalingDirection, ScalingTrigger, float]:
        """Apply scheduled scaling based on time patterns."""
        current_hour = datetime.now().hour
        
        # Simple schedule - scale up during business hours, down at night
        if 8 <= current_hour <= 18:  # Business hours
            if current_instances < self.thresholds.max_instances // 2:
                return ScalingDirection.UP, ScalingTrigger.SCHEDULED, current_hour
        else:  # Off hours
            if current_instances > self.thresholds.min_instances:
                return ScalingDirection.DOWN, ScalingTrigger.SCHEDULED, current_hour
        
        # Fall back to reactive scaling
        return self._apply_reactive_scaling(scale_decisions)
    
    def record_scaling_event(self, event: ScalingEvent) -> None:
        """Record a scaling event."""
        self.scaling_history.append(event)
        self.last_scaling_time = event.timestamp
    
    def get_scaling_recommendations(self) -> List[str]:
        """Get scaling recommendations based on history."""
        if not self.scaling_history:
            return []
        
        recommendations = []
        recent_events = list(self.scaling_history)[-20:]
        
        # Check for frequent scaling
        scale_ups = len([e for e in recent_events if e.direction == ScalingDirection.UP])
        scale_downs = len([e for e in recent_events if e.direction == ScalingDirection.DOWN])
        
        if scale_ups > 5 and scale_downs > 5:
            recommendations.append("Consider adjusting thresholds - frequent scaling detected")
        
        # Check for failed scaling events
        failed_events = [e for e in recent_events if not e.success]
        if len(failed_events) > 3:
            recommendations.append("Multiple scaling failures detected - check infrastructure")
        
        return recommendations


class ContainerOrchestrator:
    """Manages container orchestration across different platforms."""
    
    def __init__(self, platform: OrchestrationPlatform):
        self.platform = platform
        self.client = None
        self._initialize_client()
        
    def _initialize_client(self) -> None:
        """Initialize orchestration client."""
        try:
            if self.platform == OrchestrationPlatform.DOCKER and DOCKER_AVAILABLE:
                self.client = docker.from_env()
                logger.info("Docker client initialized")
                
            elif self.platform == OrchestrationPlatform.KUBERNETES and KUBERNETES_AVAILABLE:
                try:
                    config.load_incluster_config()
                except config.ConfigException:
                    config.load_kube_config()
                
                self.client = {
                    'apps_v1': client.AppsV1Api(),
                    'core_v1': client.CoreV1Api(),
                    'autoscaling_v1': client.AutoscalingV1Api()
                }
                logger.info("Kubernetes client initialized")
                
            elif self.platform == OrchestrationPlatform.AWS_ECS and AWS_AVAILABLE:
                self.client = boto3.client('ecs')
                logger.info("AWS ECS client initialized")
                
            else:
                logger.warning(f"Platform {self.platform.value} not available or not supported")
                
        except Exception as e:
            logger.error(f"Failed to initialize {self.platform.value} client: {e}")
    
    def deploy_containers(self, spec: ContainerSpec, count: int) -> bool:
        """Deploy containers according to specification."""
        try:
            if self.platform == OrchestrationPlatform.DOCKER:
                return self._deploy_docker_containers(spec, count)
            elif self.platform == OrchestrationPlatform.KUBERNETES:
                return self._deploy_kubernetes_containers(spec, count)
            elif self.platform == OrchestrationPlatform.AWS_ECS:
                return self._deploy_ecs_containers(spec, count)
            else:
                logger.error(f"Deployment not implemented for {self.platform.value}")
                return False
                
        except Exception as e:
            logger.error(f"Container deployment failed: {e}")
            return False
    
    def _deploy_docker_containers(self, spec: ContainerSpec, count: int) -> bool:
        """Deploy Docker containers."""
        if not self.client:
            return False
        
        try:
            for i in range(count):
                container_name = f"{spec.name}-{i}"
                
                # Check if container already exists
                existing_containers = self.client.containers.list(
                    filters={'name': container_name}
                )
                
                if existing_containers:
                    continue  # Container already exists
                
                # Create and start container
                container = self.client.containers.run(
                    image=spec.image,
                    name=container_name,
                    ports={f"{port}/tcp": port for port in spec.ports},
                    environment=spec.environment,
                    command=spec.command,
                    detach=True,
                    restart_policy={'Name': 'unless-stopped'}
                )
                
                logger.info(f"Started Docker container {container_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Docker deployment failed: {e}")
            return False
    
    def _deploy_kubernetes_containers(self, spec: ContainerSpec, count: int) -> bool:
        """Deploy Kubernetes containers via Deployment."""
        if not self.client:
            return False
        
        try:
            apps_v1 = self.client['apps_v1']
            
            # Check if deployment exists
            try:
                deployment = apps_v1.read_namespaced_deployment(
                    name=spec.name,
                    namespace="default"
                )
                
                # Update replica count
                deployment.spec.replicas = count
                apps_v1.patch_namespaced_deployment(
                    name=spec.name,
                    namespace="default",
                    body=deployment
                )
                
                logger.info(f"Updated Kubernetes deployment {spec.name} to {count} replicas")
                
            except client.exceptions.ApiException as e:
                if e.status == 404:
                    # Create new deployment
                    deployment_manifest = self._create_kubernetes_deployment_manifest(spec, count)
                    apps_v1.create_namespaced_deployment(
                        namespace="default",
                        body=deployment_manifest
                    )
                    logger.info(f"Created Kubernetes deployment {spec.name} with {count} replicas")
                else:
                    raise
            
            return True
            
        except Exception as e:
            logger.error(f"Kubernetes deployment failed: {e}")
            return False
    
    def _create_kubernetes_deployment_manifest(self, spec: ContainerSpec, count: int) -> Dict[str, Any]:
        """Create Kubernetes deployment manifest."""
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": spec.name,
                "labels": {"app": spec.name}
            },
            "spec": {
                "replicas": count,
                "selector": {
                    "matchLabels": {"app": spec.name}
                },
                "template": {
                    "metadata": {
                        "labels": {"app": spec.name}
                    },
                    "spec": {
                        "containers": [{
                            "name": spec.name,
                            "image": spec.image,
                            "ports": [{"containerPort": port} for port in spec.ports],
                            "env": [{"name": k, "value": v} for k, v in spec.environment.items()],
                            "resources": {
                                "limits": spec.resource_limits,
                                "requests": spec.resource_requests
                            }
                        }]
                    }
                }
            }
        }
    
    def _deploy_ecs_containers(self, spec: ContainerSpec, count: int) -> bool:
        """Deploy ECS containers."""
        if not self.client:
            return False
        
        try:
            # Update ECS service desired count
            response = self.client.update_service(
                cluster='default',
                service=spec.name,
                desiredCount=count
            )
            
            logger.info(f"Updated ECS service {spec.name} to {count} desired count")
            return True
            
        except Exception as e:
            logger.error(f"ECS deployment failed: {e}")
            return False
    
    def get_current_instance_count(self, service_name: str) -> int:
        """Get current number of running instances."""
        try:
            if self.platform == OrchestrationPlatform.DOCKER:
                containers = self.client.containers.list(
                    filters={'name': service_name}
                )
                return len(containers)
                
            elif self.platform == OrchestrationPlatform.KUBERNETES:
                apps_v1 = self.client['apps_v1']
                deployment = apps_v1.read_namespaced_deployment(
                    name=service_name,
                    namespace="default"
                )
                return deployment.status.ready_replicas or 0
                
            elif self.platform == OrchestrationPlatform.AWS_ECS:
                response = self.client.describe_services(
                    cluster='default',
                    services=[service_name]
                )
                if response['services']:
                    return response['services'][0]['runningCount']
                
        except Exception as e:
            logger.error(f"Failed to get instance count for {service_name}: {e}")
        
        return 0
    
    def scale_service(self, service_name: str, target_count: int) -> bool:
        """Scale service to target instance count."""
        try:
            current_count = self.get_current_instance_count(service_name)
            
            if current_count == target_count:
                return True  # Already at target
            
            # Create a minimal spec for scaling
            spec = ContainerSpec(
                name=service_name,
                image="placeholder"  # Not used for scaling existing services
            )
            
            return self.deploy_containers(spec, target_count)
            
        except Exception as e:
            logger.error(f"Failed to scale {service_name} to {target_count}: {e}")
            return False


class AutoScalingSystem:
    """Complete auto-scaling system with intelligent decision making."""
    
    def __init__(
        self,
        container_spec: ContainerSpec,
        orchestrator: ContainerOrchestrator,
        thresholds: ScalingThresholds,
        policy: ScalingPolicy = ScalingPolicy.REACTIVE
    ):
        self.container_spec = container_spec
        self.orchestrator = orchestrator
        self.thresholds = thresholds
        self.policy = policy
        
        # Components
        self.metrics_collector = MetricsCollector()
        self.decision_engine = ScalingDecisionEngine(thresholds, policy)
        
        # State
        self.scaling_active = False
        self.scaling_thread: Optional[threading.Thread] = None
        self.scaling_events = deque(maxlen=1000)
        
        # Health monitoring
        self.health_checks: List[Callable[[], bool]] = []
        self.unhealthy_instances: Set[str] = set()
        
    def start_auto_scaling(self) -> None:
        """Start the auto-scaling system."""
        if self.scaling_active:
            return
        
        self.scaling_active = True
        self.metrics_collector.start_collection()
        
        # Start scaling loop
        self.scaling_thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self.scaling_thread.start()
        
        logger.info("Auto-scaling system started")
    
    def stop_auto_scaling(self) -> None:
        """Stop the auto-scaling system."""
        self.scaling_active = False
        self.metrics_collector.stop_collection()
        
        if self.scaling_thread:
            self.scaling_thread.join(timeout=10.0)
        
        logger.info("Auto-scaling system stopped")
    
    def _scaling_loop(self) -> None:
        """Main auto-scaling loop."""
        while self.scaling_active:
            try:
                # Get current metrics
                metrics = self.metrics_collector.get_average_metrics(window_seconds=60.0)
                if not metrics:
                    time.sleep(30)
                    continue
                
                # Get current instance count
                current_count = self.orchestrator.get_current_instance_count(
                    self.container_spec.name
                )
                
                # Make scaling decision
                direction, trigger, value = self.decision_engine.evaluate_scaling_need(
                    metrics, current_count
                )
                
                # Execute scaling if needed
                if direction != ScalingDirection.MAINTAIN:
                    success = self._execute_scaling(direction, trigger, value, current_count)
                    
                    # Record scaling event
                    event = ScalingEvent(
                        timestamp=time.time(),
                        direction=direction,
                        trigger=trigger,
                        old_count=current_count,
                        new_count=self._calculate_target_count(direction, current_count),
                        trigger_value=value,
                        threshold=self._get_threshold_for_trigger(trigger, direction),
                        success=success
                    )
                    
                    self.scaling_events.append(event)
                    self.decision_engine.record_scaling_event(event)
                    
                    if success:
                        logger.info(f"Scaling {direction.value}: {current_count} -> {event.new_count} "
                                   f"(trigger: {trigger.value} = {value:.2f})")
                    else:
                        logger.error(f"Scaling {direction.value} failed for {self.container_spec.name}")
                
                # Check health of instances
                self._perform_health_checks()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Auto-scaling loop error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _execute_scaling(
        self, 
        direction: ScalingDirection, 
        trigger: ScalingTrigger, 
        value: float, 
        current_count: int
    ) -> bool:
        """Execute the scaling action."""
        
        target_count = self._calculate_target_count(direction, current_count)
        
        # Ensure target is within bounds
        target_count = max(self.thresholds.min_instances, 
                          min(target_count, self.thresholds.max_instances))
        
        if target_count == current_count:
            return True  # No scaling needed
        
        # Execute scaling
        return self.orchestrator.scale_service(self.container_spec.name, target_count)
    
    def _calculate_target_count(self, direction: ScalingDirection, current_count: int) -> int:
        """Calculate target instance count for scaling."""
        if direction == ScalingDirection.UP:
            # Scale up by 50% or add at least 1 instance
            return max(current_count + 1, int(current_count * 1.5))
        elif direction == ScalingDirection.DOWN:
            # Scale down by 25% or remove at least 1 instance
            return max(self.thresholds.min_instances, int(current_count * 0.75))
        else:
            return current_count
    
    def _get_threshold_for_trigger(self, trigger: ScalingTrigger, direction: ScalingDirection) -> float:
        """Get threshold value for a specific trigger and direction."""
        if trigger == ScalingTrigger.CPU_UTILIZATION:
            return self.thresholds.scale_up_cpu if direction == ScalingDirection.UP else self.thresholds.scale_down_cpu
        elif trigger == ScalingTrigger.MEMORY_UTILIZATION:
            return self.thresholds.scale_up_memory if direction == ScalingDirection.UP else self.thresholds.scale_down_memory
        elif trigger == ScalingTrigger.QUEUE_DEPTH:
            return self.thresholds.scale_up_queue_depth if direction == ScalingDirection.UP else self.thresholds.scale_down_queue_depth
        elif trigger == ScalingTrigger.RESPONSE_TIME:
            return self.thresholds.scale_up_response_time if direction == ScalingDirection.UP else self.thresholds.scale_down_response_time
        else:
            return 0.0
    
    def _perform_health_checks(self) -> None:
        """Perform health checks on instances."""
        # This is a placeholder - in practice would check actual instance health
        try:
            for health_check in self.health_checks:
                if not health_check():
                    logger.warning("Health check failed")
        except Exception as e:
            logger.error(f"Health check error: {e}")
    
    def add_health_check(self, health_check: Callable[[], bool]) -> None:
        """Add a health check function."""
        self.health_checks.append(health_check)
    
    def update_application_metrics(
        self,
        queue_depth: int = 0,
        avg_response_time_ms: float = 0.0,
        requests_per_second: float = 0.0,
        error_rate: float = 0.0
    ) -> None:
        """Update application-specific metrics for scaling decisions."""
        self.metrics_collector.update_app_metrics(
            queue_depth, avg_response_time_ms, requests_per_second, error_rate
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        current_metrics = self.metrics_collector.get_latest_metrics()
        current_count = self.orchestrator.get_current_instance_count(self.container_spec.name)
        
        recent_events = list(self.scaling_events)[-10:]
        
        return {
            'scaling_active': self.scaling_active,
            'current_instances': current_count,
            'target_range': {
                'min': self.thresholds.min_instances,
                'max': self.thresholds.max_instances
            },
            'current_metrics': asdict(current_metrics) if current_metrics else None,
            'thresholds': asdict(self.thresholds),
            'recent_scaling_events': [asdict(event) for event in recent_events],
            'scaling_recommendations': self.decision_engine.get_scaling_recommendations(),
            'orchestration_platform': self.orchestrator.platform.value,
            'policy': self.policy.value
        }


# Utility functions
def create_docker_auto_scaler(
    image: str,
    service_name: str,
    min_instances: int = 1,
    max_instances: int = 10
) -> AutoScalingSystem:
    """Create auto-scaler for Docker containers."""
    
    container_spec = ContainerSpec(
        image=image,
        name=service_name,
        ports=[8080],
        environment={'ENV': 'production'}
    )
    
    orchestrator = ContainerOrchestrator(OrchestrationPlatform.DOCKER)
    
    thresholds = ScalingThresholds(
        min_instances=min_instances,
        max_instances=max_instances
    )
    
    return AutoScalingSystem(container_spec, orchestrator, thresholds)


def create_kubernetes_auto_scaler(
    image: str,
    service_name: str,
    namespace: str = "default",
    min_instances: int = 1,
    max_instances: int = 10
) -> AutoScalingSystem:
    """Create auto-scaler for Kubernetes deployments."""
    
    container_spec = ContainerSpec(
        image=image,
        name=service_name,
        ports=[8080],
        resource_limits={'memory': '512Mi', 'cpu': '500m'},
        resource_requests={'memory': '256Mi', 'cpu': '250m'}
    )
    
    orchestrator = ContainerOrchestrator(OrchestrationPlatform.KUBERNETES)
    
    thresholds = ScalingThresholds(
        min_instances=min_instances,
        max_instances=max_instances
    )
    
    return AutoScalingSystem(container_spec, orchestrator, thresholds)


# Global auto-scaler instance (can be configured)
global_auto_scaler: Optional[AutoScalingSystem] = None


def initialize_auto_scaling(
    platform: OrchestrationPlatform,
    container_spec: ContainerSpec,
    thresholds: ScalingThresholds
) -> AutoScalingSystem:
    """Initialize global auto-scaling system."""
    global global_auto_scaler
    
    orchestrator = ContainerOrchestrator(platform)
    global_auto_scaler = AutoScalingSystem(container_spec, orchestrator, thresholds)
    
    return global_auto_scaler


def get_auto_scaling_status() -> Optional[Dict[str, Any]]:
    """Get global auto-scaling status."""
    return global_auto_scaler.get_system_status() if global_auto_scaler else None