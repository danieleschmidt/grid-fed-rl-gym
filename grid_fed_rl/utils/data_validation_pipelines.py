"""Comprehensive data validation pipelines for grid operations."""

import logging
import json
import time
import hashlib
import threading
from typing import Dict, List, Any, Optional, Union, Callable, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from collections import defaultdict, deque

from .exceptions import (
    DataValidationError, SecurityViolationError, RetryableError,
    exponential_backoff, global_error_recovery_manager
)
from .distributed_tracing import trace_federated_operation, global_tracer

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DataType(Enum):
    """Types of data in the grid system."""
    GRID_MEASUREMENT = "grid_measurement"
    POWER_FLOW_RESULT = "power_flow_result"
    CONTROL_COMMAND = "control_command"
    SENSOR_DATA = "sensor_data"
    TRAINING_DATA = "training_data"
    CONFIGURATION = "configuration"
    TELEMETRY = "telemetry"
    WEATHER_DATA = "weather_data"


@dataclass
class ValidationRule:
    """A single validation rule."""
    name: str
    description: str
    rule_type: str  # "range", "format", "consistency", "custom"
    parameters: Dict[str, Any]
    severity: ValidationSeverity
    enabled: bool = True
    
    def __post_init__(self):
        self.creation_time = datetime.now()
        self.last_modified = datetime.now()


@dataclass
class ValidationIssue:
    """A validation issue found during data processing."""
    rule_name: str
    severity: ValidationSeverity
    message: str
    data_path: str  # JSON path to problematic data
    actual_value: Any
    expected_value: Any = None
    timestamp: datetime = field(default_factory=datetime.now)
    data_type: Optional[DataType] = None
    source: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "severity": self.severity.value,
            "message": self.message,
            "data_path": self.data_path,
            "actual_value": str(self.actual_value),
            "expected_value": str(self.expected_value) if self.expected_value else None,
            "timestamp": self.timestamp.isoformat(),
            "data_type": self.data_type.value if self.data_type else None,
            "source": self.source
        }


@dataclass
class ValidationReport:
    """Report of validation results."""
    data_id: str
    data_type: DataType
    validation_time: datetime
    issues: List[ValidationIssue] = field(default_factory=list)
    passed_rules: List[str] = field(default_factory=list)
    data_quality_score: float = 0.0
    processing_time_ms: float = 0.0
    
    def is_valid(self) -> bool:
        """Check if data passes validation."""
        critical_errors = [i for i in self.issues if i.severity == ValidationSeverity.CRITICAL]
        errors = [i for i in self.issues if i.severity == ValidationSeverity.ERROR]
        return len(critical_errors) == 0 and len(errors) == 0
    
    def get_severity_counts(self) -> Dict[str, int]:
        """Get count of issues by severity."""
        counts = {severity.value: 0 for severity in ValidationSeverity}
        for issue in self.issues:
            counts[issue.severity.value] += 1
        return counts
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_id": self.data_id,
            "data_type": self.data_type.value,
            "validation_time": self.validation_time.isoformat(),
            "is_valid": self.is_valid(),
            "data_quality_score": self.data_quality_score,
            "processing_time_ms": self.processing_time_ms,
            "severity_counts": self.get_severity_counts(),
            "issues": [issue.to_dict() for issue in self.issues],
            "passed_rules": self.passed_rules
        }


class DataValidator(ABC):
    """Abstract base class for data validators."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.enabled = True
        self.validation_count = 0
        self.success_count = 0
        self.failure_count = 0
    
    @abstractmethod
    def validate(self, data: Any, context: Dict[str, Any] = None) -> List[ValidationIssue]:
        """Validate data and return any issues found."""
        pass
    
    def is_enabled(self) -> bool:
        """Check if validator is enabled."""
        return self.enabled
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        success_rate = self.success_count / max(self.validation_count, 1)
        return {
            "name": self.name,
            "enabled": self.enabled,
            "validation_count": self.validation_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": success_rate
        }


class RangeValidator(DataValidator):
    """Validates numeric values are within specified ranges."""
    
    def __init__(
        self, 
        field_path: str, 
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        severity: ValidationSeverity = ValidationSeverity.ERROR
    ):
        super().__init__(f"range_{field_path}", f"Range validation for {field_path}")
        self.field_path = field_path
        self.min_value = min_value
        self.max_value = max_value
        self.severity = severity
    
    def validate(self, data: Any, context: Dict[str, Any] = None) -> List[ValidationIssue]:
        issues = []
        self.validation_count += 1
        
        try:
            # Extract value from data using field path
            value = self._extract_value(data, self.field_path)
            
            if value is None:
                issues.append(ValidationIssue(
                    rule_name=self.name,
                    severity=ValidationSeverity.WARNING,
                    message=f"Field {self.field_path} is missing",
                    data_path=self.field_path,
                    actual_value=None
                ))
                return issues
            
            # Convert to numeric if possible
            if isinstance(value, str):
                try:
                    value = float(value)
                except ValueError:
                    issues.append(ValidationIssue(
                        rule_name=self.name,
                        severity=ValidationSeverity.ERROR,
                        message=f"Field {self.field_path} is not numeric: {value}",
                        data_path=self.field_path,
                        actual_value=value
                    ))
                    return issues
            
            # Check ranges
            if self.min_value is not None and value < self.min_value:
                issues.append(ValidationIssue(
                    rule_name=self.name,
                    severity=self.severity,
                    message=f"Value {value} is below minimum {self.min_value}",
                    data_path=self.field_path,
                    actual_value=value,
                    expected_value=f">= {self.min_value}"
                ))
            
            if self.max_value is not None and value > self.max_value:
                issues.append(ValidationIssue(
                    rule_name=self.name,
                    severity=self.severity,
                    message=f"Value {value} is above maximum {self.max_value}",
                    data_path=self.field_path,
                    actual_value=value,
                    expected_value=f"<= {self.max_value}"
                ))
            
            if not issues:
                self.success_count += 1
            else:
                self.failure_count += 1
                
        except Exception as e:
            self.failure_count += 1
            issues.append(ValidationIssue(
                rule_name=self.name,
                severity=ValidationSeverity.ERROR,
                message=f"Range validation error: {e}",
                data_path=self.field_path,
                actual_value=str(e)
            ))
        
        return issues
    
    def _extract_value(self, data: Any, path: str) -> Any:
        """Extract value from nested data structure using dot notation."""
        try:
            parts = path.split('.')
            current = data
            
            for part in parts:
                if isinstance(current, dict):
                    current = current.get(part)
                elif isinstance(current, list) and part.isdigit():
                    current = current[int(part)]
                else:
                    return None
            
            return current
        except (KeyError, IndexError, TypeError):
            return None


class FormatValidator(DataValidator):
    """Validates data format and structure."""
    
    def __init__(
        self, 
        field_path: str,
        expected_type: type,
        regex_pattern: Optional[str] = None,
        severity: ValidationSeverity = ValidationSeverity.ERROR
    ):
        super().__init__(f"format_{field_path}", f"Format validation for {field_path}")
        self.field_path = field_path
        self.expected_type = expected_type
        self.regex_pattern = regex_pattern
        self.severity = severity
    
    def validate(self, data: Any, context: Dict[str, Any] = None) -> List[ValidationIssue]:
        issues = []
        self.validation_count += 1
        
        try:
            value = self._extract_value(data, self.field_path)
            
            if value is None:
                issues.append(ValidationIssue(
                    rule_name=self.name,
                    severity=ValidationSeverity.WARNING,
                    message=f"Field {self.field_path} is missing",
                    data_path=self.field_path,
                    actual_value=None
                ))
                return issues
            
            # Type validation
            if not isinstance(value, self.expected_type):
                issues.append(ValidationIssue(
                    rule_name=self.name,
                    severity=self.severity,
                    message=f"Field {self.field_path} has wrong type: expected {self.expected_type.__name__}, got {type(value).__name__}",
                    data_path=self.field_path,
                    actual_value=value,
                    expected_value=self.expected_type.__name__
                ))
            
            # Regex validation for strings
            if self.regex_pattern and isinstance(value, str):
                import re
                if not re.match(self.regex_pattern, value):
                    issues.append(ValidationIssue(
                        rule_name=self.name,
                        severity=self.severity,
                        message=f"Field {self.field_path} does not match pattern {self.regex_pattern}",
                        data_path=self.field_path,
                        actual_value=value,
                        expected_value=f"pattern: {self.regex_pattern}"
                    ))
            
            if not issues:
                self.success_count += 1
            else:
                self.failure_count += 1
                
        except Exception as e:
            self.failure_count += 1
            issues.append(ValidationIssue(
                rule_name=self.name,
                severity=ValidationSeverity.ERROR,
                message=f"Format validation error: {e}",
                data_path=self.field_path,
                actual_value=str(e)
            ))
        
        return issues
    
    def _extract_value(self, data: Any, path: str) -> Any:
        """Extract value from nested data structure."""
        try:
            parts = path.split('.')
            current = data
            
            for part in parts:
                if isinstance(current, dict):
                    current = current.get(part)
                elif isinstance(current, list) and part.isdigit():
                    current = current[int(part)]
                else:
                    return None
            
            return current
        except (KeyError, IndexError, TypeError):
            return None


class ConsistencyValidator(DataValidator):
    """Validates consistency between related data fields."""
    
    def __init__(
        self,
        primary_field: str,
        secondary_field: str,
        consistency_check: Callable[[Any, Any], bool],
        severity: ValidationSeverity = ValidationSeverity.WARNING
    ):
        super().__init__(
            f"consistency_{primary_field}_{secondary_field}",
            f"Consistency validation between {primary_field} and {secondary_field}"
        )
        self.primary_field = primary_field
        self.secondary_field = secondary_field
        self.consistency_check = consistency_check
        self.severity = severity
    
    def validate(self, data: Any, context: Dict[str, Any] = None) -> List[ValidationIssue]:
        issues = []
        self.validation_count += 1
        
        try:
            primary_value = self._extract_value(data, self.primary_field)
            secondary_value = self._extract_value(data, self.secondary_field)
            
            if primary_value is None or secondary_value is None:
                issues.append(ValidationIssue(
                    rule_name=self.name,
                    severity=ValidationSeverity.WARNING,
                    message=f"Cannot check consistency: {self.primary_field}={primary_value}, {self.secondary_field}={secondary_value}",
                    data_path=f"{self.primary_field},{self.secondary_field}",
                    actual_value=f"{primary_value},{secondary_value}"
                ))
                return issues
            
            if not self.consistency_check(primary_value, secondary_value):
                issues.append(ValidationIssue(
                    rule_name=self.name,
                    severity=self.severity,
                    message=f"Inconsistency detected between {self.primary_field}={primary_value} and {self.secondary_field}={secondary_value}",
                    data_path=f"{self.primary_field},{self.secondary_field}",
                    actual_value=f"{primary_value},{secondary_value}"
                ))
                self.failure_count += 1
            else:
                self.success_count += 1
                
        except Exception as e:
            self.failure_count += 1
            issues.append(ValidationIssue(
                rule_name=self.name,
                severity=ValidationSeverity.ERROR,
                message=f"Consistency validation error: {e}",
                data_path=f"{self.primary_field},{self.secondary_field}",
                actual_value=str(e)
            ))
        
        return issues
    
    def _extract_value(self, data: Any, path: str) -> Any:
        """Extract value from nested data structure."""
        try:
            parts = path.split('.')
            current = data
            
            for part in parts:
                if isinstance(current, dict):
                    current = current.get(part)
                elif isinstance(current, list) and part.isdigit():
                    current = current[int(part)]
                else:
                    return None
            
            return current
        except (KeyError, IndexError, TypeError):
            return None


class GridMeasurementValidator(DataValidator):
    """Specialized validator for grid measurements."""
    
    def __init__(self):
        super().__init__(
            "grid_measurement",
            "Validates electrical grid measurement data"
        )
    
    def validate(self, data: Any, context: Dict[str, Any] = None) -> List[ValidationIssue]:
        issues = []
        self.validation_count += 1
        
        try:
            # Validate voltage measurements
            if 'bus_voltages' in data:
                voltages = data['bus_voltages']
                if isinstance(voltages, (list, np.ndarray)):
                    for i, voltage in enumerate(voltages):
                        if not (0.5 <= voltage <= 2.0):  # Per unit range
                            issues.append(ValidationIssue(
                                rule_name=self.name,
                                severity=ValidationSeverity.ERROR,
                                message=f"Bus voltage {i} out of range: {voltage} pu",
                                data_path=f"bus_voltages[{i}]",
                                actual_value=voltage,
                                expected_value="0.5 to 2.0 pu"
                            ))
            
            # Validate frequency
            if 'frequency' in data:
                freq = data['frequency']
                if not (45.0 <= freq <= 65.0):  # Hz range
                    issues.append(ValidationIssue(
                        rule_name=self.name,
                        severity=ValidationSeverity.ERROR,
                        message=f"Frequency out of range: {freq} Hz",
                        data_path="frequency",
                        actual_value=freq,
                        expected_value="45.0 to 65.0 Hz"
                    ))
            
            # Validate power flows
            if 'line_flows' in data:
                flows = data['line_flows']
                if isinstance(flows, (list, np.ndarray)):
                    for i, flow in enumerate(flows):
                        if abs(flow) > 1000e6:  # 1000 MW limit
                            issues.append(ValidationIssue(
                                rule_name=self.name,
                                severity=ValidationSeverity.WARNING,
                                message=f"Line flow {i} very high: {flow/1e6:.1f} MW",
                                data_path=f"line_flows[{i}]",
                                actual_value=flow
                            ))
            
            # Check for NaN/infinite values
            self._check_numeric_validity(data, issues, "")
            
            if not issues:
                self.success_count += 1
            else:
                self.failure_count += 1
                
        except Exception as e:
            self.failure_count += 1
            issues.append(ValidationIssue(
                rule_name=self.name,
                severity=ValidationSeverity.ERROR,
                message=f"Grid measurement validation error: {e}",
                data_path="root",
                actual_value=str(e)
            ))
        
        return issues
    
    def _check_numeric_validity(self, data: Any, issues: List[ValidationIssue], path: str):
        """Check for NaN and infinite values recursively."""
        if isinstance(data, dict):
            for key, value in data.items():
                self._check_numeric_validity(value, issues, f"{path}.{key}" if path else key)
        elif isinstance(data, (list, np.ndarray)):
            for i, value in enumerate(data):
                self._check_numeric_validity(value, issues, f"{path}[{i}]")
        elif isinstance(data, (int, float, np.number)):
            if np.isnan(data):
                issues.append(ValidationIssue(
                    rule_name=self.name,
                    severity=ValidationSeverity.ERROR,
                    message=f"NaN value detected at {path}",
                    data_path=path,
                    actual_value="NaN"
                ))
            elif np.isinf(data):
                issues.append(ValidationIssue(
                    rule_name=self.name,
                    severity=ValidationSeverity.ERROR,
                    message=f"Infinite value detected at {path}",
                    data_path=path,
                    actual_value="Infinity"
                ))


class DataValidationPipeline:
    """Comprehensive data validation pipeline."""
    
    def __init__(self, name: str, data_type: DataType):
        self.name = name
        self.data_type = data_type
        self.validators: List[DataValidator] = []
        self.preprocessing_steps: List[Callable] = []
        self.postprocessing_steps: List[Callable] = []
        
        # Pipeline statistics
        self.validation_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.processing_times = deque(maxlen=1000)
        
        # Configuration
        self.fail_fast = False
        self.quality_threshold = 0.8
        
        logger.info(f"Data validation pipeline '{name}' initialized for {data_type.value}")
    
    def add_validator(self, validator: DataValidator):
        """Add a validator to the pipeline."""
        self.validators.append(validator)
        logger.debug(f"Added validator '{validator.name}' to pipeline '{self.name}'")
    
    def add_preprocessing_step(self, step: Callable[[Any], Any]):
        """Add a preprocessing step."""
        self.preprocessing_steps.append(step)
    
    def add_postprocessing_step(self, step: Callable[[Any, ValidationReport], Any]):
        """Add a postprocessing step."""
        self.postprocessing_steps.append(step)
    
    @trace_federated_operation("validate_data", "data_validation")
    @exponential_backoff(max_retries=2, base_delay=0.1)
    def validate(
        self, 
        data: Any, 
        data_id: str = None,
        context: Dict[str, Any] = None
    ) -> ValidationReport:
        """Run data through the validation pipeline."""
        
        start_time = time.time()
        self.validation_count += 1
        
        if data_id is None:
            data_id = hashlib.sha256(str(data).encode()).hexdigest()[:16]
        
        if context is None:
            context = {}
        
        # Initialize report
        report = ValidationReport(
            data_id=data_id,
            data_type=self.data_type,
            validation_time=datetime.now()
        )
        
        try:
            # Preprocessing
            processed_data = data
            for step in self.preprocessing_steps:
                try:
                    processed_data = step(processed_data)
                except Exception as e:
                    report.issues.append(ValidationIssue(
                        rule_name="preprocessing",
                        severity=ValidationSeverity.ERROR,
                        message=f"Preprocessing failed: {e}",
                        data_path="root",
                        actual_value=str(e)
                    ))
                    if self.fail_fast:
                        break
            
            # Validation
            all_issues = []
            for validator in self.validators:
                if not validator.is_enabled():
                    continue
                
                try:
                    validator_issues = validator.validate(processed_data, context)
                    all_issues.extend(validator_issues)
                    
                    if not validator_issues:
                        report.passed_rules.append(validator.name)
                    
                    # Stop on critical errors if fail_fast is enabled
                    if self.fail_fast:
                        critical_issues = [i for i in validator_issues if i.severity == ValidationSeverity.CRITICAL]
                        if critical_issues:
                            break
                            
                except Exception as e:
                    all_issues.append(ValidationIssue(
                        rule_name=validator.name,
                        severity=ValidationSeverity.ERROR,
                        message=f"Validator '{validator.name}' failed: {e}",
                        data_path="validator_error",
                        actual_value=str(e)
                    ))
            
            report.issues = all_issues
            
            # Calculate data quality score
            report.data_quality_score = self._calculate_quality_score(report)
            
            # Postprocessing
            for step in self.postprocessing_steps:
                try:
                    step(processed_data, report)
                except Exception as e:
                    report.issues.append(ValidationIssue(
                        rule_name="postprocessing",
                        severity=ValidationSeverity.WARNING,
                        message=f"Postprocessing failed: {e}",
                        data_path="root",
                        actual_value=str(e)
                    ))
            
            # Update statistics
            if report.is_valid():
                self.success_count += 1
            else:
                self.failure_count += 1
            
            processing_time = (time.time() - start_time) * 1000  # ms
            report.processing_time_ms = processing_time
            self.processing_times.append(processing_time)
            
        except Exception as e:
            self.failure_count += 1
            report.issues.append(ValidationIssue(
                rule_name="pipeline_error",
                severity=ValidationSeverity.CRITICAL,
                message=f"Pipeline error: {e}",
                data_path="root",
                actual_value=str(e)
            ))
            logger.error(f"Validation pipeline error: {e}")
        
        return report
    
    def _calculate_quality_score(self, report: ValidationReport) -> float:
        """Calculate data quality score based on validation results."""
        if not self.validators:
            return 1.0
        
        total_weight = len(self.validators)
        penalty_weight = 0
        
        for issue in report.issues:
            if issue.severity == ValidationSeverity.CRITICAL:
                penalty_weight += 1.0
            elif issue.severity == ValidationSeverity.ERROR:
                penalty_weight += 0.7
            elif issue.severity == ValidationSeverity.WARNING:
                penalty_weight += 0.3
            elif issue.severity == ValidationSeverity.INFO:
                penalty_weight += 0.1
        
        score = max(0.0, 1.0 - (penalty_weight / total_weight))
        return round(score, 3)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        success_rate = self.success_count / max(self.validation_count, 1)
        
        avg_processing_time = (
            sum(self.processing_times) / len(self.processing_times)
            if self.processing_times else 0
        )
        
        validator_stats = [v.get_statistics() for v in self.validators]
        
        return {
            "pipeline_name": self.name,
            "data_type": self.data_type.value,
            "validation_count": self.validation_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": success_rate,
            "avg_processing_time_ms": avg_processing_time,
            "validator_count": len(self.validators),
            "enabled_validators": len([v for v in self.validators if v.is_enabled()]),
            "validator_statistics": validator_stats,
            "quality_threshold": self.quality_threshold,
            "fail_fast": self.fail_fast
        }


class DataValidationManager:
    """Manages multiple validation pipelines."""
    
    def __init__(self):
        self.pipelines: Dict[DataType, DataValidationPipeline] = {}
        self.validation_history: deque = deque(maxlen=10000)
        self.issue_patterns: Dict[str, int] = defaultdict(int)
        
        # Global settings
        self.enable_caching = True
        self.cache: Dict[str, ValidationReport] = {}
        self.cache_ttl = timedelta(minutes=5)
        self.cache_timestamps: Dict[str, datetime] = {}
        
        # Background monitoring
        self.monitoring_thread = None
        self.monitoring_active = False
        
        # Initialize default pipelines
        self._initialize_default_pipelines()
        
        logger.info("Data validation manager initialized")
    
    def _initialize_default_pipelines(self):
        """Initialize default validation pipelines for common data types."""
        
        # Grid measurement pipeline
        grid_pipeline = DataValidationPipeline("grid_measurements", DataType.GRID_MEASUREMENT)
        grid_pipeline.add_validator(GridMeasurementValidator())
        grid_pipeline.add_validator(RangeValidator("frequency", 45.0, 65.0))
        self.pipelines[DataType.GRID_MEASUREMENT] = grid_pipeline
        
        # Power flow result pipeline
        pf_pipeline = DataValidationPipeline("power_flow_results", DataType.POWER_FLOW_RESULT)
        pf_pipeline.add_validator(RangeValidator("iterations", 1, 100))
        pf_pipeline.add_validator(FormatValidator("converged", bool))
        self.pipelines[DataType.POWER_FLOW_RESULT] = pf_pipeline
        
        # Control command pipeline
        control_pipeline = DataValidationPipeline("control_commands", DataType.CONTROL_COMMAND)
        control_pipeline.add_validator(RangeValidator("magnitude", -1.0, 1.0))
        control_pipeline.add_validator(FormatValidator("command_type", str))
        self.pipelines[DataType.CONTROL_COMMAND] = control_pipeline
        
        # Sensor data pipeline
        sensor_pipeline = DataValidationPipeline("sensor_data", DataType.SENSOR_DATA)
        sensor_pipeline.add_validator(FormatValidator("timestamp", str, r"\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}"))
        sensor_pipeline.add_validator(RangeValidator("value", -1000, 1000))
        self.pipelines[DataType.SENSOR_DATA] = sensor_pipeline
        
        logger.info(f"Initialized {len(self.pipelines)} default validation pipelines")
    
    def register_pipeline(self, pipeline: DataValidationPipeline):
        """Register a custom validation pipeline."""
        self.pipelines[pipeline.data_type] = pipeline
        logger.info(f"Registered custom pipeline for {pipeline.data_type.value}")
    
    def validate_data(
        self,
        data: Any,
        data_type: DataType,
        data_id: str = None,
        context: Dict[str, Any] = None
    ) -> ValidationReport:
        """Validate data using the appropriate pipeline."""
        
        # Check cache first
        if self.enable_caching and data_id:
            cached_report = self._get_cached_report(data_id)
            if cached_report:
                return cached_report
        
        # Get pipeline
        pipeline = self.pipelines.get(data_type)
        if not pipeline:
            logger.warning(f"No validation pipeline for data type: {data_type.value}")
            # Create basic report
            report = ValidationReport(
                data_id=data_id or "unknown",
                data_type=data_type,
                validation_time=datetime.now()
            )
            report.issues.append(ValidationIssue(
                rule_name="no_pipeline",
                severity=ValidationSeverity.WARNING,
                message=f"No validation pipeline configured for {data_type.value}",
                data_path="root",
                actual_value=data_type.value
            ))
            return report
        
        # Run validation
        report = pipeline.validate(data, data_id, context)
        
        # Cache result
        if self.enable_caching and data_id:
            self._cache_report(data_id, report)
        
        # Store in history
        self.validation_history.append(report)
        
        # Track issue patterns
        for issue in report.issues:
            self.issue_patterns[f"{issue.rule_name}:{issue.severity.value}"] += 1
        
        return report
    
    def _get_cached_report(self, data_id: str) -> Optional[ValidationReport]:
        """Get cached validation report if still valid."""
        if data_id not in self.cache:
            return None
        
        cache_time = self.cache_timestamps.get(data_id)
        if cache_time and datetime.now() - cache_time < self.cache_ttl:
            return self.cache[data_id]
        
        # Remove expired entry
        del self.cache[data_id]
        if data_id in self.cache_timestamps:
            del self.cache_timestamps[data_id]
        
        return None
    
    def _cache_report(self, data_id: str, report: ValidationReport):
        """Cache validation report."""
        self.cache[data_id] = report
        self.cache_timestamps[data_id] = datetime.now()
        
        # Cleanup old cache entries
        if len(self.cache) > 1000:
            # Remove oldest 100 entries
            sorted_items = sorted(self.cache_timestamps.items(), key=lambda x: x[1])
            for data_id, _ in sorted_items[:100]:
                if data_id in self.cache:
                    del self.cache[data_id]
                del self.cache_timestamps[data_id]
    
    def batch_validate(
        self,
        data_batch: List[Tuple[Any, DataType]],
        context: Dict[str, Any] = None
    ) -> List[ValidationReport]:
        """Validate a batch of data items."""
        reports = []
        
        with global_tracer.trace("batch_validation") as span:
            span.set_tag("batch_size", len(data_batch))
            
            for i, (data, data_type) in enumerate(data_batch):
                data_id = f"batch_{i}_{int(time.time())}"
                report = self.validate_data(data, data_type, data_id, context)
                reports.append(report)
                
                # Add batch info to report
                report.data_id = data_id
            
            span.set_tag("successful_validations", sum(1 for r in reports if r.is_valid()))
        
        return reports
    
    def get_pipeline_statistics(self, data_type: Optional[DataType] = None) -> Dict[str, Any]:
        """Get statistics for pipelines."""
        if data_type:
            pipeline = self.pipelines.get(data_type)
            return pipeline.get_statistics() if pipeline else {}
        
        # Return stats for all pipelines
        stats = {}
        for dt, pipeline in self.pipelines.items():
            stats[dt.value] = pipeline.get_statistics()
        
        return stats
    
    def get_validation_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get validation summary for recent period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_reports = [
            r for r in self.validation_history
            if r.validation_time >= cutoff_time
        ]
        
        if not recent_reports:
            return {"message": "No recent validations"}
        
        # Calculate statistics
        total_validations = len(recent_reports)
        successful_validations = sum(1 for r in recent_reports if r.is_valid())
        
        # Group by data type
        by_data_type = defaultdict(list)
        for report in recent_reports:
            by_data_type[report.data_type.value].append(report)
        
        # Calculate average quality score
        quality_scores = [r.data_quality_score for r in recent_reports]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        # Most common issues
        issue_counts = defaultdict(int)
        for report in recent_reports:
            for issue in report.issues:
                issue_key = f"{issue.rule_name}:{issue.severity.value}"
                issue_counts[issue_key] += 1
        
        top_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "time_period_hours": hours,
            "total_validations": total_validations,
            "successful_validations": successful_validations,
            "success_rate": successful_validations / total_validations,
            "average_quality_score": round(avg_quality, 3),
            "validations_by_data_type": {
                dt: len(reports) for dt, reports in by_data_type.items()
            },
            "top_issues": [
                {"pattern": pattern, "count": count}
                for pattern, count in top_issues
            ],
            "cache_hit_rate": self._calculate_cache_hit_rate(),
            "active_pipelines": len(self.pipelines)
        }
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate (simplified estimation)."""
        # This is a simplified calculation
        # In practice, you'd track actual cache hits/misses
        return len(self.cache) / max(1, len(self.validation_history[-100:]))
    
    def clear_cache(self):
        """Clear validation cache."""
        self.cache.clear()
        self.cache_timestamps.clear()
        logger.info("Validation cache cleared")
    
    def get_data_quality_trends(self, data_type: Optional[DataType] = None) -> Dict[str, Any]:
        """Get data quality trends over time."""
        reports = list(self.validation_history)
        
        if data_type:
            reports = [r for r in reports if r.data_type == data_type]
        
        if not reports:
            return {"message": "No validation history available"}
        
        # Group by hour
        hourly_scores = defaultdict(list)
        for report in reports:
            hour_key = report.validation_time.strftime("%Y-%m-%d %H:00")
            hourly_scores[hour_key].append(report.data_quality_score)
        
        # Calculate hourly averages
        hourly_averages = {
            hour: sum(scores) / len(scores)
            for hour, scores in hourly_scores.items()
        }
        
        # Sort by time
        sorted_hours = sorted(hourly_averages.items())
        
        return {
            "data_type": data_type.value if data_type else "all",
            "hourly_quality_scores": dict(sorted_hours),
            "overall_trend": self._calculate_trend([score for _, score in sorted_hours]),
            "total_reports": len(reports),
            "time_range_hours": len(sorted_hours)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from a list of values."""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple trend calculation
        first_half = sum(values[:len(values)//2]) / (len(values)//2)
        second_half = sum(values[len(values)//2:]) / (len(values) - len(values)//2)
        
        diff = second_half - first_half
        if abs(diff) < 0.05:
            return "stable"
        elif diff > 0:
            return "improving"
        else:
            return "declining"


# Global validation manager
global_validation_manager = DataValidationManager()