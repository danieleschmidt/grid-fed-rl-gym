"""Asynchronous federated learning coordinator for scalable grid control."""

import asyncio
import aiohttp
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple, Set, Callable
import logging
import time
import json
import hashlib
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict, deque
import ssl
import secrets

from .core import FedLearningConfig, ClientUpdate, FederatedClient
from ..utils.exceptions import FederatedLearningError, SecurityError
from ..utils.advanced_optimization import OptimizationOrchestrator, ModelCompressionEngine
from ..utils.monitoring import SystemMetrics

logger = logging.getLogger(__name__)


@dataclass
class AsyncClientState:
    """State tracking for asynchronous clients."""
    client_id: str
    last_update_time: float
    update_count: int
    current_round: int
    staleness: int
    performance_score: float
    reliability_score: float
    is_active: bool
    last_heartbeat: float


@dataclass
class FederatedRound:
    """Information about a federated learning round."""
    round_id: int
    start_time: float
    participating_clients: Set[str]
    received_updates: Dict[str, ClientUpdate]
    aggregated_model: Optional[Dict[str, torch.Tensor]]
    convergence_metrics: Dict[str, float]
    communication_overhead: float


class SecureAggregator:
    """Secure aggregation with Byzantine fault tolerance."""
    
    def __init__(
        self,
        byzantine_tolerance: int = 1,
        verification_threshold: float = 0.8,
        enable_encryption: bool = True
    ):
        self.byzantine_tolerance = byzantine_tolerance
        self.verification_threshold = verification_threshold
        self.enable_encryption = enable_encryption
        
        # Security state
        self.client_keys = {}
        self.trusted_clients = set()
        self.suspicious_clients = set()
        self.aggregation_history = deque(maxlen=100)
        
    def aggregate_updates(
        self,
        client_updates: List[ClientUpdate],
        staleness_weights: Optional[Dict[str, float]] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Securely aggregate client updates with Byzantine fault tolerance."""
        
        if len(client_updates) < 3:
            raise FederatedLearningError("Insufficient client updates for secure aggregation")
        
        # Verify client updates
        verified_updates = self._verify_updates(client_updates)
        
        if len(verified_updates) < 2:
            raise FederatedLearningError("Insufficient verified updates")
        
        # Detect and filter Byzantine updates
        clean_updates = self._filter_byzantine_updates(verified_updates)
        
        # Perform weighted aggregation
        aggregated_model = self._weighted_aggregation(clean_updates, staleness_weights)
        
        # Generate aggregation metadata
        metadata = {
            'total_updates': len(client_updates),
            'verified_updates': len(verified_updates),
            'clean_updates': len(clean_updates),
            'byzantine_detected': len(verified_updates) - len(clean_updates),
            'aggregation_method': 'secure_fedavg',
            'timestamp': time.time()
        }
        
        return aggregated_model, metadata
    
    def _verify_updates(self, updates: List[ClientUpdate]) -> List[ClientUpdate]:
        """Verify authenticity of client updates."""
        verified = []
        
        for update in updates:
            # Check client trust status
            if update.client_id in self.suspicious_clients:
                logger.warning(f"Rejecting update from suspicious client {update.client_id}")
                continue
            
            # Verify update integrity
            if self._verify_update_integrity(update):
                verified.append(update)
            else:
                self.suspicious_clients.add(update.client_id)
                logger.warning(f"Update integrity check failed for client {update.client_id}")
        
        return verified
    
    def _verify_update_integrity(self, update: ClientUpdate) -> bool:
        """Verify the integrity of a single update."""
        try:
            # Check for NaN or infinite values
            for param_name, param_tensor in update.parameters.items():
                if isinstance(param_tensor, np.ndarray):
                    if np.isnan(param_tensor).any() or np.isinf(param_tensor).any():
                        return False
                elif isinstance(param_tensor, torch.Tensor):
                    if torch.isnan(param_tensor).any() or torch.isinf(param_tensor).any():
                        return False
            
            # Check parameter magnitudes
            total_norm = 0
            for param_tensor in update.parameters.values():
                if isinstance(param_tensor, np.ndarray):
                    total_norm += np.linalg.norm(param_tensor) ** 2
                elif isinstance(param_tensor, torch.Tensor):
                    total_norm += param_tensor.norm() ** 2
            
            total_norm = np.sqrt(total_norm)
            
            # Reject updates with extremely large norms (potential attack)
            if total_norm > 1000:
                logger.warning(f"Update rejected due to large norm: {total_norm}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error verifying update integrity: {e}")
            return False
    
    def _filter_byzantine_updates(self, updates: List[ClientUpdate]) -> List[ClientUpdate]:
        """Filter out Byzantine (malicious) updates using statistical methods."""
        if len(updates) <= 2 * self.byzantine_tolerance:
            logger.warning("Insufficient updates for Byzantine filtering")
            return updates
        
        # Compute pairwise similarities
        similarities = self._compute_update_similarities(updates)
        
        # Identify outliers using clustering
        outlier_indices = self._detect_outliers(similarities)
        
        # Filter out outliers
        clean_updates = [
            update for i, update in enumerate(updates) 
            if i not in outlier_indices
        ]
        
        logger.info(f"Filtered {len(outlier_indices)} Byzantine updates")
        return clean_updates
    
    def _compute_update_similarities(self, updates: List[ClientUpdate]) -> np.ndarray:
        """Compute cosine similarities between updates."""
        n_updates = len(updates)
        similarities = np.zeros((n_updates, n_updates))
        
        # Flatten all parameters for each update
        flattened_updates = []
        for update in updates:
            flattened = []
            for param_tensor in update.parameters.values():
                if isinstance(param_tensor, np.ndarray):
                    flattened.append(param_tensor.flatten())
                elif isinstance(param_tensor, torch.Tensor):
                    flattened.append(param_tensor.detach().cpu().numpy().flatten())
            flattened_updates.append(np.concatenate(flattened))
        
        # Compute cosine similarities
        for i in range(n_updates):
            for j in range(i + 1, n_updates):
                sim = np.dot(flattened_updates[i], flattened_updates[j]) / \
                      (np.linalg.norm(flattened_updates[i]) * np.linalg.norm(flattened_updates[j]) + 1e-8)
                similarities[i, j] = similarities[j, i] = sim
        
        return similarities
    
    def _detect_outliers(self, similarities: np.ndarray) -> List[int]:
        """Detect outlier updates based on similarity matrix."""
        n_updates = similarities.shape[0]
        
        # Compute average similarity for each update
        avg_similarities = np.mean(similarities, axis=1)
        
        # Find updates with low average similarity
        threshold = np.percentile(avg_similarities, 25)  # Bottom quartile
        outlier_indices = np.where(avg_similarities < threshold)[0].tolist()
        
        # Limit number of outliers based on Byzantine tolerance
        outlier_indices = outlier_indices[:self.byzantine_tolerance]
        
        return outlier_indices
    
    def _weighted_aggregation(
        self,
        updates: List[ClientUpdate],
        staleness_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, torch.Tensor]:
        """Perform weighted aggregation of clean updates."""
        if not updates:
            raise FederatedLearningError("No updates to aggregate")
        
        # Initialize aggregated parameters
        aggregated_params = {}
        total_weight = 0
        
        for update in updates:
            # Compute weight based on number of samples and staleness
            sample_weight = update.num_samples
            
            if staleness_weights and update.client_id in staleness_weights:
                staleness_weight = staleness_weights[update.client_id]
                weight = sample_weight * staleness_weight
            else:
                weight = sample_weight
            
            total_weight += weight
            
            # Aggregate parameters
            for param_name, param_tensor in update.parameters.items():
                if isinstance(param_tensor, np.ndarray):
                    param_tensor = torch.from_numpy(param_tensor)
                
                if param_name not in aggregated_params:
                    aggregated_params[param_name] = weight * param_tensor
                else:
                    aggregated_params[param_name] += weight * param_tensor
        
        # Normalize by total weight
        for param_name in aggregated_params:
            aggregated_params[param_name] /= total_weight
        
        return aggregated_params


class AsyncFederatedCoordinator:
    """Asynchronous federated learning coordinator with advanced optimization."""
    
    def __init__(
        self,
        config: FedLearningConfig,
        model_template: nn.Module,
        optimization_orchestrator: Optional[OptimizationOrchestrator] = None
    ):
        self.config = config
        self.model_template = model_template
        self.optimization_orchestrator = optimization_orchestrator or OptimizationOrchestrator()
        
        # Async coordination state
        self.client_states: Dict[str, AsyncClientState] = {}
        self.global_model_state = {}
        self.current_round = 0
        self.active_rounds: Dict[int, FederatedRound] = {}
        
        # Security and aggregation
        self.secure_aggregator = SecureAggregator(
            byzantine_tolerance=config.num_clients // 4,  # Tolerate up to 25% Byzantine clients
            enable_encryption=config.secure_aggregation
        )
        
        # Performance tracking
        self.round_metrics = deque(maxlen=1000)
        self.convergence_history = deque(maxlen=100)
        self.communication_costs = deque(maxlen=1000)
        
        # Async infrastructure
        self.event_loop = None
        self.server_tasks = []
        self.client_connections = {}
        self.heartbeat_interval = 30.0  # seconds
        
        # Thread safety
        self.state_lock = threading.RLock()
        
        logger.info(f"AsyncFederatedCoordinator initialized for {config.num_clients} clients")
    
    async def start_coordination(self, host: str = "localhost", port: int = 8080):
        """Start the asynchronous coordination server."""
        self.event_loop = asyncio.get_event_loop()
        
        # Initialize global model
        self._initialize_global_model()
        
        # Start server tasks
        server_task = asyncio.create_task(self._run_coordination_server(host, port))
        heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
        round_manager_task = asyncio.create_task(self._round_manager())
        
        self.server_tasks = [server_task, heartbeat_task, round_manager_task]
        
        try:
            await asyncio.gather(*self.server_tasks)
        except Exception as e:
            logger.error(f"Coordination server error: {e}")
            await self.shutdown()
    
    async def _run_coordination_server(self, host: str, port: int):
        """Run the main coordination server."""
        from aiohttp import web
        
        app = web.Application()
        app.router.add_post('/register', self._handle_client_registration)
        app.router.add_post('/submit_update', self._handle_client_update)
        app.router.add_get('/get_model', self._handle_model_request)
        app.router.add_post('/heartbeat', self._handle_heartbeat)
        
        runner = web.AppRunner(app)
        await runner.setup()
        
        site = web.TCPSite(runner, host, port)
        await site.start()
        
        logger.info(f"Coordination server started on {host}:{port}")
        
        # Keep server running
        while True:
            await asyncio.sleep(1)
    
    async def _handle_client_registration(self, request):
        """Handle new client registration."""
        try:
            data = await request.json()
            client_id = data['client_id']
            client_info = data.get('client_info', {})
            
            with self.state_lock:
                if client_id not in self.client_states:
                    self.client_states[client_id] = AsyncClientState(
                        client_id=client_id,
                        last_update_time=time.time(),
                        update_count=0,
                        current_round=0,
                        staleness=0,
                        performance_score=1.0,
                        reliability_score=1.0,
                        is_active=True,
                        last_heartbeat=time.time()
                    )
                    
                    logger.info(f"Registered new client: {client_id}")
                    
                    return web.json_response({
                        'status': 'success',
                        'client_id': client_id,
                        'global_round': self.current_round
                    })
                else:
                    logger.warning(f"Client {client_id} already registered")
                    return web.json_response({
                        'status': 'already_registered',
                        'client_id': client_id
                    })
        
        except Exception as e:
            logger.error(f"Client registration error: {e}")
            return web.json_response({'status': 'error', 'message': str(e)}, status=500)
    
    async def _handle_client_update(self, request):
        """Handle client model update submission."""
        try:
            data = await request.json()
            client_id = data['client_id']
            
            # Deserialize update
            update = self._deserialize_client_update(data['update'])
            
            with self.state_lock:
                if client_id in self.client_states:
                    client_state = self.client_states[client_id]
                    
                    # Update client state
                    client_state.last_update_time = time.time()
                    client_state.update_count += 1
                    client_state.staleness = self.current_round - client_state.current_round
                    
                    # Add update to current round
                    current_round_id = self.current_round
                    if current_round_id not in self.active_rounds:
                        self.active_rounds[current_round_id] = FederatedRound(
                            round_id=current_round_id,
                            start_time=time.time(),
                            participating_clients=set(),
                            received_updates={},
                            aggregated_model=None,
                            convergence_metrics={},
                            communication_overhead=0.0
                        )
                    
                    round_info = self.active_rounds[current_round_id]
                    round_info.received_updates[client_id] = update
                    round_info.participating_clients.add(client_id)
                    
                    logger.debug(f"Received update from client {client_id} for round {current_round_id}")
                    
                    return web.json_response({
                        'status': 'success',
                        'round_id': current_round_id,
                        'staleness': client_state.staleness
                    })
                else:
                    return web.json_response({
                        'status': 'error',
                        'message': 'Client not registered'
                    }, status=400)
        
        except Exception as e:
            logger.error(f"Client update handling error: {e}")
            return web.json_response({'status': 'error', 'message': str(e)}, status=500)
    
    async def _handle_model_request(self, request):
        """Handle client request for global model."""
        try:
            client_id = request.query.get('client_id')
            
            if client_id and client_id in self.client_states:
                # Serialize and return current global model
                serialized_model = self._serialize_model(self.global_model_state)
                
                return web.json_response({
                    'status': 'success',
                    'model': serialized_model,
                    'round_id': self.current_round
                })
            else:
                return web.json_response({
                    'status': 'error',
                    'message': 'Invalid client ID'
                }, status=400)
        
        except Exception as e:
            logger.error(f"Model request handling error: {e}")
            return web.json_response({'status': 'error', 'message': str(e)}, status=500)
    
    async def _handle_heartbeat(self, request):
        """Handle client heartbeat."""
        try:
            data = await request.json()
            client_id = data['client_id']
            
            with self.state_lock:
                if client_id in self.client_states:
                    self.client_states[client_id].last_heartbeat = time.time()
                    self.client_states[client_id].is_active = True
                    
                    return web.json_response({'status': 'success'})
                else:
                    return web.json_response({
                        'status': 'error',
                        'message': 'Client not registered'
                    }, status=400)
        
        except Exception as e:
            logger.error(f"Heartbeat handling error: {e}")
            return web.json_response({'status': 'error', 'message': str(e)}, status=500)
    
    async def _heartbeat_monitor(self):
        """Monitor client heartbeats and mark inactive clients."""
        while True:
            try:
                current_time = time.time()
                timeout_threshold = self.heartbeat_interval * 3  # 3x heartbeat interval
                
                with self.state_lock:
                    for client_id, client_state in self.client_states.items():
                        if current_time - client_state.last_heartbeat > timeout_threshold:
                            if client_state.is_active:
                                logger.warning(f"Client {client_id} became inactive")
                                client_state.is_active = False
                                client_state.reliability_score *= 0.9  # Penalize reliability
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Heartbeat monitor error: {e}")
                await asyncio.sleep(5)
    
    async def _round_manager(self):
        """Manage federated learning rounds."""
        while True:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds
                
                with self.state_lock:
                    # Check if current round is ready for aggregation
                    if self.current_round in self.active_rounds:
                        round_info = self.active_rounds[self.current_round]
                        
                        # Criteria for round completion
                        min_clients = max(2, self.config.min_clients_per_round)
                        max_wait_time = 300  # 5 minutes
                        
                        received_updates = len(round_info.received_updates)
                        round_duration = time.time() - round_info.start_time
                        
                        if (received_updates >= min_clients and 
                            (received_updates >= self.config.num_clients * 0.7 or round_duration > max_wait_time)):
                            
                            # Aggregate updates
                            await self._aggregate_round(self.current_round)
                            
                            # Start next round
                            self.current_round += 1
                            logger.info(f"Started round {self.current_round}")
                
            except Exception as e:
                logger.error(f"Round manager error: {e}")
    
    async def _aggregate_round(self, round_id: int):
        """Aggregate updates for a specific round."""
        try:
            if round_id not in self.active_rounds:
                logger.warning(f"Round {round_id} not found for aggregation")
                return
            
            round_info = self.active_rounds[round_id]
            updates = list(round_info.received_updates.values())
            
            if not updates:
                logger.warning(f"No updates to aggregate for round {round_id}")
                return
            
            # Compute staleness weights
            staleness_weights = {}
            for client_id, update in round_info.received_updates.items():
                if client_id in self.client_states:
                    staleness = self.client_states[client_id].staleness
                    # Staleness penalty: reduce weight for stale updates
                    staleness_weights[client_id] = 1.0 / (1.0 + staleness * 0.1)
            
            # Secure aggregation
            aggregated_model, aggregation_metadata = self.secure_aggregator.aggregate_updates(
                updates, staleness_weights
            )
            
            # Update global model
            self.global_model_state = aggregated_model
            round_info.aggregated_model = aggregated_model
            
            # Compute convergence metrics
            convergence_metrics = self._compute_convergence_metrics(round_info)
            round_info.convergence_metrics = convergence_metrics
            
            # Update performance metrics
            self._update_round_metrics(round_info, aggregation_metadata)
            
            # Clean up old rounds
            self._cleanup_old_rounds()
            
            logger.info(f"Aggregated round {round_id} with {len(updates)} updates")
            
        except Exception as e:
            logger.error(f"Round aggregation error: {e}")
    
    def _initialize_global_model(self):
        """Initialize the global model state."""
        self.global_model_state = {
            name: param.detach().clone()
            for name, param in self.model_template.named_parameters()
        }
        logger.info("Global model initialized")
    
    def _serialize_model(self, model_state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Serialize model for transmission."""
        serialized = {}
        for name, tensor in model_state.items():
            serialized[name] = tensor.cpu().numpy().tolist()
        return serialized
    
    def _deserialize_client_update(self, update_data: Dict[str, Any]) -> ClientUpdate:
        """Deserialize client update from JSON."""
        parameters = {}
        for name, param_data in update_data['parameters'].items():
            parameters[name] = np.array(param_data)
        
        return ClientUpdate(
            client_id=update_data['client_id'],
            parameters=parameters,
            num_samples=update_data['num_samples'],
            loss=update_data['loss'],
            metrics=update_data.get('metrics', {})
        )
    
    def _compute_convergence_metrics(self, round_info: FederatedRound) -> Dict[str, float]:
        """Compute convergence metrics for the round."""
        updates = list(round_info.received_updates.values())
        
        if len(updates) < 2:
            return {'convergence_rate': 0.0, 'update_diversity': 0.0}
        
        # Compute average loss
        avg_loss = np.mean([update.loss for update in updates])
        
        # Compute update diversity (measure of client heterogeneity)
        diversities = []
        for i, update1 in enumerate(updates):
            for j, update2 in enumerate(updates[i+1:], i+1):
                # Compute parameter difference
                diff_norm = 0
                for param_name in update1.parameters:
                    if param_name in update2.parameters:
                        diff = update1.parameters[param_name] - update2.parameters[param_name]
                        diff_norm += np.linalg.norm(diff) ** 2
                diversities.append(np.sqrt(diff_norm))
        
        avg_diversity = np.mean(diversities) if diversities else 0.0
        
        # Convergence rate (simplified)
        convergence_rate = 1.0 / (1.0 + avg_loss)
        
        return {
            'convergence_rate': convergence_rate,
            'update_diversity': avg_diversity,
            'average_loss': avg_loss,
            'num_participants': len(updates)
        }
    
    def _update_round_metrics(self, round_info: FederatedRound, aggregation_metadata: Dict[str, Any]):
        """Update performance metrics after round completion."""
        round_duration = time.time() - round_info.start_time
        
        metrics = {
            'round_id': round_info.round_id,
            'duration': round_duration,
            'participants': len(round_info.participating_clients),
            'convergence_rate': round_info.convergence_metrics.get('convergence_rate', 0.0),
            'update_diversity': round_info.convergence_metrics.get('update_diversity', 0.0),
            'byzantine_detected': aggregation_metadata.get('byzantine_detected', 0),
            'communication_overhead': round_info.communication_overhead
        }
        
        self.round_metrics.append(metrics)
        self.convergence_history.append(round_info.convergence_metrics.get('convergence_rate', 0.0))
    
    def _cleanup_old_rounds(self):
        """Clean up old round data to manage memory."""
        current_time = time.time()
        cutoff_time = current_time - 3600  # Keep last hour of rounds
        
        rounds_to_remove = [
            round_id for round_id, round_info in self.active_rounds.items()
            if round_info.start_time < cutoff_time
        ]
        
        for round_id in rounds_to_remove:
            del self.active_rounds[round_id]
    
    async def shutdown(self):
        """Gracefully shutdown the coordination server."""
        logger.info("Shutting down federated coordinator...")
        
        # Cancel server tasks
        for task in self.server_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.server_tasks, return_exceptions=True)
        
        logger.info("Federated coordinator shutdown complete")
    
    def get_coordination_stats(self) -> Dict[str, Any]:
        """Get comprehensive coordination statistics."""
        with self.state_lock:
            active_clients = sum(1 for state in self.client_states.values() if state.is_active)
            
            return {
                'current_round': self.current_round,
                'total_clients': len(self.client_states),
                'active_clients': active_clients,
                'completed_rounds': len(self.round_metrics),
                'average_convergence_rate': np.mean(self.convergence_history) if self.convergence_history else 0.0,
                'round_metrics': list(self.round_metrics)[-10:],  # Last 10 rounds
                'client_states': {
                    client_id: asdict(state) for client_id, state in self.client_states.items()
                }
            }