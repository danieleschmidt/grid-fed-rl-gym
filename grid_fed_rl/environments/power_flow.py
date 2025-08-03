"""Power flow solvers for grid simulation."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
import warnings

from .base import Bus, Line


@dataclass
class PowerFlowSolution:
    """Results from power flow calculation."""
    converged: bool
    iterations: int
    bus_voltages: np.ndarray
    bus_angles: np.ndarray
    line_flows: np.ndarray
    line_loadings: np.ndarray
    losses: float
    max_mismatch: float


class PowerFlowSolver(ABC):
    """Abstract base class for power flow solvers."""
    
    def __init__(
        self,
        tolerance: float = 1e-6,
        max_iterations: int = 50,
        **kwargs
    ) -> None:
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        
    @abstractmethod
    def solve(
        self,
        buses: List[Bus],
        lines: List[Line],
        loads: Dict[str, float],
        generation: Dict[str, float]
    ) -> PowerFlowSolution:
        """Solve power flow equations."""
        pass
        
    def build_admittance_matrix(self, buses: List[Bus], lines: List[Line]) -> np.ndarray:
        """Build bus admittance matrix."""
        n_buses = len(buses)
        Y = np.zeros((n_buses, n_buses), dtype=complex)
        
        # Bus mapping
        bus_map = {bus.id: i for i, bus in enumerate(buses)}
        
        # Add line admittances
        for line in lines:
            i = bus_map[line.from_bus]
            j = bus_map[line.to_bus]
            
            # Line impedance
            z = complex(line.resistance, line.reactance)
            y = 1.0 / z if abs(z) > 1e-12 else 0.0
            
            # Off-diagonal elements
            Y[i, j] -= y
            Y[j, i] -= y
            
            # Diagonal elements
            Y[i, i] += y
            Y[j, j] += y
            
        return Y


class NewtonRaphsonSolver(PowerFlowSolver):
    """Newton-Raphson power flow solver."""
    
    def __init__(
        self,
        tolerance: float = 1e-6,
        max_iterations: int = 50,
        acceleration_factor: float = 1.0,
        **kwargs
    ) -> None:
        super().__init__(tolerance, max_iterations, **kwargs)
        self.acceleration_factor = acceleration_factor
        
    def solve(
        self,
        buses: List[Bus],
        lines: List[Line],
        loads: Dict[str, float],
        generation: Dict[str, float]
    ) -> PowerFlowSolution:
        """Solve power flow using Newton-Raphson method."""
        n_buses = len(buses)
        
        # Build admittance matrix
        Y = self.build_admittance_matrix(buses, lines)
        
        # Initialize voltage vector
        V = np.ones(n_buses, dtype=complex)
        
        # Set up power injections
        P_specified = np.zeros(n_buses)
        Q_specified = np.zeros(n_buses)
        
        bus_map = {bus.id: i for i, bus in enumerate(buses)}
        
        # Add load injections (negative)
        for bus_id, power in loads.items():
            if bus_id in bus_map:
                i = bus_map[bus_id]
                P_specified[i] -= power
                
        # Add generation injections (positive)
        for bus_id, power in generation.items():
            if bus_id in bus_map:
                i = bus_map[bus_id]
                P_specified[i] += power
                
        # Find slack bus (first bus with type 'slack')
        slack_bus = None
        pv_buses = []
        pq_buses = []
        
        for i, bus in enumerate(buses):
            if bus.bus_type == "slack":
                slack_bus = i
                V[i] = complex(bus.voltage_magnitude, 0)
            elif bus.bus_type == "pv":
                pv_buses.append(i)
                V[i] = complex(bus.voltage_magnitude, 0)
            else:  # pq bus
                pq_buses.append(i)
                
        if slack_bus is None:
            # Default first bus as slack
            slack_bus = 0
            buses[0].bus_type = "slack"
            
        # Newton-Raphson iterations
        converged = False
        iteration = 0
        max_mismatch = float('inf')
        
        for iteration in range(self.max_iterations):
            # Calculate power mismatches
            S_calculated = V * np.conj(Y @ V)
            P_calculated = S_calculated.real
            Q_calculated = S_calculated.imag
            
            # Power mismatches (exclude slack bus for P)
            dP = np.zeros(n_buses)
            dQ = np.zeros(n_buses)
            
            # P mismatches for non-slack buses
            for i in range(n_buses):
                if i != slack_bus:
                    dP[i] = P_specified[i] - P_calculated[i]
                    
            # Q mismatches for PQ buses only
            for i in pq_buses:
                dQ[i] = Q_specified[i] - Q_calculated[i]
                
            # Check convergence
            max_mismatch = max(np.max(np.abs(dP)), np.max(np.abs(dQ)))
            if max_mismatch < self.tolerance:
                converged = True
                break
                
            # Build Jacobian matrix
            J = self._build_jacobian(Y, V, buses, slack_bus, pv_buses, pq_buses)
            
            # Build mismatch vector
            mismatch = np.concatenate([
                dP[i] for i in range(n_buses) if i != slack_bus
            ] + [
                dQ[i] for i in pq_buses
            ])
            
            # Solve for corrections
            try:
                dx = np.linalg.solve(J, mismatch)
            except np.linalg.LinAlgError:
                warnings.warn("Jacobian matrix is singular")
                break
                
            # Apply corrections
            self._apply_corrections(dx, V, buses, slack_bus, pv_buses, pq_buses)
            
        # Calculate final line flows
        line_flows, line_loadings = self._calculate_line_flows(V, Y, lines, bus_map)
        
        # Calculate losses
        S_total = np.sum(V * np.conj(Y @ V))
        losses = S_total.real
        
        return PowerFlowSolution(
            converged=converged,
            iterations=iteration + 1,
            bus_voltages=np.abs(V),
            bus_angles=np.angle(V),
            line_flows=line_flows,
            line_loadings=line_loadings,
            losses=losses,
            max_mismatch=max_mismatch
        )
        
    def _build_jacobian(
        self,
        Y: np.ndarray,
        V: np.ndarray,
        buses: List[Bus],
        slack_bus: int,
        pv_buses: List[int],
        pq_buses: List[int]
    ) -> np.ndarray:
        """Build Jacobian matrix for Newton-Raphson."""
        n_buses = len(buses)
        G = Y.real
        B = Y.imag
        
        # Voltage magnitudes and angles
        Vm = np.abs(V)
        Va = np.angle(V)
        
        # Non-slack buses for P equations
        non_slack = [i for i in range(n_buses) if i != slack_bus]
        n_p_eq = len(non_slack)
        n_q_eq = len(pq_buses)
        
        # Initialize Jacobian submatrices
        J11 = np.zeros((n_p_eq, n_p_eq))  # dP/dVa
        J12 = np.zeros((n_p_eq, n_q_eq))  # dP/dVm
        J21 = np.zeros((n_q_eq, n_p_eq))  # dQ/dVa  
        J22 = np.zeros((n_q_eq, n_q_eq))  # dQ/dVm
        
        # Build J11 (dP/dVa)
        for row, i in enumerate(non_slack):
            for col, j in enumerate(non_slack):
                if i == j:
                    # Diagonal element
                    J11[row, col] = -np.sum(Vm[i] * Vm * (G[i, :] * np.sin(Va[i] - Va) - B[i, :] * np.cos(Va[i] - Va)))
                    J11[row, col] += Vm[i] * Vm[i] * B[i, i]
                else:
                    # Off-diagonal element
                    J11[row, col] = Vm[i] * Vm[j] * (G[i, j] * np.sin(Va[i] - Va[j]) - B[i, j] * np.cos(Va[i] - Va[j]))
                    
        # Build J12 (dP/dVm) for PQ buses
        for row, i in enumerate(non_slack):
            for col, j_idx in enumerate(pq_buses):
                j = j_idx
                if i == j:
                    # Diagonal element
                    J12[row, col] = np.sum(Vm * (G[i, :] * np.cos(Va[i] - Va) + B[i, :] * np.sin(Va[i] - Va)))
                    J12[row, col] += Vm[i] * G[i, i]
                else:
                    # Off-diagonal element
                    J12[row, col] = Vm[i] * (G[i, j] * np.cos(Va[i] - Va[j]) + B[i, j] * np.sin(Va[i] - Va[j]))
                    
        # Build J21 (dQ/dVa) for PQ buses
        for row, i in enumerate(pq_buses):
            for col, j in enumerate(non_slack):
                if i == j:
                    # Diagonal element
                    J21[row, col] = np.sum(Vm[i] * Vm * (G[i, :] * np.cos(Va[i] - Va) + B[i, :] * np.sin(Va[i] - Va)))
                    J21[row, col] -= Vm[i] * Vm[i] * G[i, i]
                else:
                    # Off-diagonal element
                    J21[row, col] = -Vm[i] * Vm[j] * (G[i, j] * np.cos(Va[i] - Va[j]) + B[i, j] * np.sin(Va[i] - Va[j]))
                    
        # Build J22 (dQ/dVm) for PQ buses
        for row, i_idx in enumerate(pq_buses):
            i = i_idx
            for col, j_idx in enumerate(pq_buses):
                j = j_idx
                if i == j:
                    # Diagonal element
                    J22[row, col] = np.sum(Vm * (G[i, :] * np.sin(Va[i] - Va) - B[i, :] * np.cos(Va[i] - Va)))
                    J22[row, col] -= Vm[i] * B[i, i]
                else:
                    # Off-diagonal element
                    J22[row, col] = Vm[i] * (G[i, j] * np.sin(Va[i] - Va[j]) - B[i, j] * np.cos(Va[i] - Va[j]))
        
        # Assemble full Jacobian
        if n_q_eq > 0:
            J = np.block([[J11, J12], [J21, J22]])
        else:
            J = J11
            
        return J
        
    def _apply_corrections(
        self,
        dx: np.ndarray,
        V: np.ndarray,
        buses: List[Bus],
        slack_bus: int,
        pv_buses: List[int],
        pq_buses: List[int]
    ) -> None:
        """Apply Newton-Raphson corrections to voltage vector."""
        non_slack = [i for i in range(len(buses)) if i != slack_bus]
        n_p_eq = len(non_slack)
        
        # Extract angle and magnitude corrections
        dVa = dx[:n_p_eq]
        dVm = dx[n_p_eq:] if len(dx) > n_p_eq else np.array([])
        
        # Apply angle corrections to non-slack buses
        for i, bus_idx in enumerate(non_slack):
            Va_old = np.angle(V[bus_idx])
            Va_new = Va_old + self.acceleration_factor * dVa[i]
            Vm_old = np.abs(V[bus_idx])
            V[bus_idx] = Vm_old * np.exp(1j * Va_new)
            
        # Apply magnitude corrections to PQ buses
        for i, bus_idx in enumerate(pq_buses):
            if i < len(dVm):
                Vm_old = np.abs(V[bus_idx])
                Va_old = np.angle(V[bus_idx])
                Vm_new = Vm_old + self.acceleration_factor * dVm[i]
                V[bus_idx] = Vm_new * np.exp(1j * Va_old)
                
    def _calculate_line_flows(
        self,
        V: np.ndarray,
        Y: np.ndarray,
        lines: List[Line],
        bus_map: Dict[str, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate power flows and loadings for all lines."""
        line_flows = np.zeros(len(lines))
        line_loadings = np.zeros(len(lines))
        
        for k, line in enumerate(lines):
            i = bus_map[line.from_bus]
            j = bus_map[line.to_bus]
            
            # Line impedance
            z = complex(line.resistance, line.reactance)
            y = 1.0 / z if abs(z) > 1e-12 else 0.0
            
            # Current from i to j
            I_ij = y * (V[i] - V[j])
            
            # Power flow from i to j
            S_ij = V[i] * np.conj(I_ij)
            line_flows[k] = S_ij.real
            
            # Line loading
            line_loadings[k] = abs(S_ij) / line.rating if line.rating > 0 else 0.0
            
        return line_flows, line_loadings


class FastDecoupledSolver(PowerFlowSolver):
    """Fast decoupled power flow solver for faster convergence."""
    
    def solve(
        self,
        buses: List[Bus],
        lines: List[Line],
        loads: Dict[str, float],
        generation: Dict[str, float]
    ) -> PowerFlowSolution:
        """Solve using fast decoupled method."""
        # Simplified implementation - would normally use BX and B'' matrices
        # For now, fall back to Newton-Raphson
        nr_solver = NewtonRaphsonSolver(
            tolerance=self.tolerance,
            max_iterations=self.max_iterations
        )
        return nr_solver.solve(buses, lines, loads, generation)