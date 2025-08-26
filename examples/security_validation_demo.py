#!/usr/bin/env python3
"""
Demonstration of the comprehensive security validation system for Grid-Fed-RL-Gym.

This script shows how to use the integrated security validation system to:
1. Run comprehensive security scans
2. Generate detailed security reports
3. Monitor security health in real-time
4. Apply security hardening measures
5. Validate compliance with security standards

Usage:
    python examples/security_validation_demo.py
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from grid_fed_rl.utils.security_validation import (
        SecurityValidationSuite, 
        SecuritySeverity, 
        SecurityCategory,
        ComplianceStandard
    )
    from grid_fed_rl.utils.security import run_comprehensive_security_audit
    from grid_fed_rl.utils.security_hardening import (
        run_security_hardening_check,
        enhanced_security_hardening
    )
    from grid_fed_rl.utils.health_monitoring import system_health
except ImportError as e:
    print(f"Error importing security modules: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)


def print_banner(title: str):
    """Print a formatted banner."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'-' * 60}")
    print(f"📋 {title}")
    print("-" * 60)


def print_findings_summary(findings_summary: dict):
    """Print a formatted findings summary."""
    print("\n🔍 Security Findings Summary:")
    print(f"  Critical: {findings_summary.get('critical', 0)} 🔴")
    print(f"  High:     {findings_summary.get('high', 0)} 🟠")
    print(f"  Medium:   {findings_summary.get('medium', 0)} 🟡")
    print(f"  Low:      {findings_summary.get('low', 0)} 🟢")
    print(f"  Info:     {findings_summary.get('info', 0)} 🔵")


def print_security_score(score: float):
    """Print formatted security score."""
    if score >= 90:
        emoji = "🟢"
        status = "Excellent"
    elif score >= 80:
        emoji = "🟡"
        status = "Good"
    elif score >= 70:
        emoji = "🟠"
        status = "Fair"
    else:
        emoji = "🔴"
        status = "Needs Improvement"
    
    print(f"\n🎯 Security Score: {score:.1f}/100 {emoji} ({status})")


def demo_comprehensive_security_scan():
    """Demonstrate comprehensive security scanning."""
    print_section("Comprehensive Security Scan")
    
    print("Initializing security validation suite...")
    validation_suite = SecurityValidationSuite(base_directory=str(project_root))
    
    print("Running comprehensive security scan...")
    print("This may take a few moments as we analyze:")
    print("  • Dependency vulnerabilities")
    print("  • Code injection risks")
    print("  • Input validation weaknesses")
    print("  • Authentication/authorization issues")
    print("  • Data encryption problems")
    print("  • Network security misconfigurations")
    print("  • Configuration security risks")
    
    # Configure scan options
    scan_config = {
        "scan_dependencies": True,
        "scan_code_injection": True,
        "scan_input_validation": True,
        "scan_auth": True,
        "scan_encryption": True,
        "scan_network": True,
        "scan_configuration": True
    }
    
    # Run the scan
    scan_results = validation_suite.run_comprehensive_scan(scan_config)
    
    if "error" in scan_results:
        print(f"❌ Scan failed: {scan_results['error']}")
        return None
    
    # Display results
    scan_summary = scan_results.get("scan_summary", {})
    findings_summary = scan_results.get("findings_summary", {}).get("by_severity", {})
    
    print(f"\n✅ Scan completed in {scan_summary.get('scan_duration_seconds', 0):.2f} seconds")
    print(f"📊 Total findings: {scan_summary.get('total_findings', 0)}")
    
    print_security_score(scan_summary.get("security_score", 0))
    print_findings_summary(findings_summary)
    
    # Show top findings
    detailed_findings = scan_results.get("detailed_findings", {})
    
    critical_findings = detailed_findings.get("critical", [])
    if critical_findings:
        print(f"\n🚨 Critical Findings ({len(critical_findings)}):")
        for i, finding in enumerate(critical_findings[:3], 1):
            print(f"  {i}. {finding.get('title', 'Unknown')}")
            print(f"     📂 {finding.get('file_path', 'N/A')}")
            print(f"     💡 {finding.get('remediation', 'No remediation provided')[:80]}...")
    
    high_findings = detailed_findings.get("high", [])
    if high_findings:
        print(f"\n🔶 High Priority Findings ({len(high_findings)}):")
        for i, finding in enumerate(high_findings[:3], 1):
            print(f"  {i}. {finding.get('title', 'Unknown')}")
            print(f"     📂 {finding.get('file_path', 'N/A')}")
    
    # Show compliance summary
    compliance_summary = scan_results.get("compliance_summary", {})
    if compliance_summary:
        print(f"\n📜 Compliance Standards Affected:")
        for standard in compliance_summary.get("standards_affected", []):
            score = compliance_summary.get("compliance_score", {}).get(standard, 0)
            print(f"  • {standard}: {score:.1f}% compliant")
    
    return scan_results


def demo_security_hardening():
    """Demonstrate security hardening capabilities."""
    print_section("Security Hardening")
    
    print("Running security hardening check...")
    
    # Run comprehensive hardening check
    hardening_results = run_security_hardening_check()
    
    hardening_status = hardening_results.get("hardening_status", {})
    security_posture = hardening_results.get("security_posture", {})
    
    print(f"🛡️  Hardening Status: {hardening_status.get('validation_integrated', False) and '✅ Integrated' or '❌ Not Integrated'}")
    print(f"🔒 Security Posture: {security_posture.get('hardening_status', 'unknown').upper()}")
    
    # Show policy configuration
    policy_config = hardening_status.get("policy", {})
    if policy_config:
        print("\n⚙️  Security Policy Configuration:")
        print(f"  • Max Action Deviation: {policy_config.get('max_action_deviation', 'N/A')}")
        print(f"  • Max Consecutive Errors: {policy_config.get('max_consecutive_errors', 'N/A')}")
        print(f"  • Strict Input Validation: {policy_config.get('input_validation_strict', False) and '✅' or '❌'}")
        print(f"  • Rate Limiting: {policy_config.get('rate_limiting_enabled', False) and '✅' or '❌'}")
        print(f"  • Max Requests/Minute: {policy_config.get('max_requests_per_minute', 'N/A')}")
    
    # Show security issues
    issues = security_posture.get("issues", [])
    if issues:
        print(f"\n⚠️  Security Issues Found ({len(issues)}):")
        for issue in issues:
            severity = issue.get("severity", "unknown").upper()
            emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}.get(severity, "⚪")
            print(f"  {emoji} [{severity}] {issue.get('description', 'No description')}")
            print(f"      💡 {issue.get('recommendation', 'No recommendation')}")
    else:
        print("\n✅ No security issues found in hardening check")
    
    # Show recommendations
    recommendations = hardening_results.get("recommendations", [])
    if recommendations:
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    
    # Demonstrate config hardening
    print("\n🔧 Demonstrating Configuration Hardening:")
    
    # Sample insecure configuration
    sample_config = {
        "debug_mode": True,
        "disable_auth": True,
        "allow_all_origins": True,
        "session_timeout": 86400,  # 24 hours
        "api_key": "simple_key_123",
        "encryption_enabled": False
    }
    
    print("Original configuration (INSECURE):")
    for key, value in sample_config.items():
        print(f"  • {key}: {value}")
    
    # Apply hardening
    hardening_result = enhanced_security_hardening.apply_security_hardening(sample_config)
    
    hardened_config = hardening_result.get("hardened_config", {})
    hardening_applied = hardening_result.get("hardening_applied", [])
    
    print("\nHardened configuration (SECURE):")
    for key, value in hardened_config.items():
        print(f"  • {key}: {value}")
    
    print(f"\n🛠️  Hardening Measures Applied ({len(hardening_applied)}):")
    for measure in hardening_applied:
        print(f"  ✅ {measure}")


def demo_real_time_monitoring():
    """Demonstrate real-time security monitoring."""
    print_section("Real-Time Security Monitoring")
    
    print("Getting real-time security metrics...")
    
    # Get real-time metrics from hardening system
    real_time_metrics = enhanced_security_hardening.get_real_time_security_metrics()
    
    # Display access control metrics
    access_metrics = real_time_metrics.get("access_control", {})
    print(f"🔐 Access Control:")
    print(f"  • Total Events: {access_metrics.get('total_events', 0)}")
    print(f"  • Recent Failures: {access_metrics.get('recent_failures', 0)}")
    print(f"  • Active Rate Limits: {access_metrics.get('active_rate_limits', 0)}")
    
    # Display threat monitoring metrics
    threat_metrics = real_time_metrics.get("threat_monitoring", {})
    print(f"\n🛡️  Threat Monitoring:")
    print(f"  • Total Threats: {threat_metrics.get('total_threats', 0)}")
    print(f"  • Recent Threats: {threat_metrics.get('recent_threats', 0)}")
    print(f"  • Blocked Operations: {threat_metrics.get('blocked_operations', 0)}")
    
    # Display validation suite metrics if available
    validation_metrics = real_time_metrics.get("validation_suite", {})
    if validation_metrics:
        print(f"\n📊 Validation Suite Metrics:")
        print(f"  • Security Score: {validation_metrics.get('security_score', 0):.1f}")
        print(f"  • Total Scans: {validation_metrics.get('total_scans', 0)}")
        print(f"  • Critical Findings: {validation_metrics.get('critical_findings', 0)}")
        print(f"  • High Findings: {validation_metrics.get('high_findings', 0)}")
        last_scan = validation_metrics.get('last_scan')
        if last_scan:
            print(f"  • Last Scan: {last_scan}")
    
    # Get health monitoring status
    print(f"\n🏥 Health Monitoring Integration:")
    try:
        health_report = system_health.get_health_report()
        overall_status = health_report.get("overall_status", "unknown").upper()
        emoji = {"HEALTHY": "🟢", "WARNING": "🟡", "CRITICAL": "🔴", "FAILED": "⚫"}.get(overall_status, "⚪")
        print(f"  • Overall Status: {overall_status} {emoji}")
        print(f"  • Uptime: {health_report.get('uptime_seconds', 0):.1f} seconds")
        print(f"  • Total Alerts: {health_report.get('total_alerts', 0)}")
        
        # Show security-specific metrics if available
        metrics = health_report.get("metrics", {})
        security_score_metric = metrics.get("security_score")
        if security_score_metric:
            print(f"  • Security Score: {security_score_metric.get('value', 0):.1f} ({security_score_metric.get('status', 'unknown')})")
        
    except Exception as e:
        print(f"  ❌ Health monitoring error: {e}")


def demo_integrated_audit():
    """Demonstrate the integrated comprehensive security audit."""
    print_section("Integrated Comprehensive Security Audit")
    
    print("Running integrated security audit combining all systems...")
    
    # Run the comprehensive audit
    audit_results = run_comprehensive_security_audit(include_validation_suite=True)
    
    # Display traditional audit results
    traditional_audit = audit_results.get("traditional_audit", {})
    if traditional_audit:
        summary = traditional_audit.get("summary", {})
        print(f"🔍 Traditional Security Audit:")
        print(f"  • Total Issues: {summary.get('total_issues', 0)}")
        print(f"  • Critical: {summary.get('critical_count', 0)}")
        print(f"  • High: {summary.get('high_count', 0)}")
        print(f"  • Medium: {summary.get('medium_count', 0)}")
        print(f"  • Low: {summary.get('low_count', 0)}")
    
    # Display communication audit results
    comm_audit = audit_results.get("communication_audit", {})
    if comm_audit:
        print(f"\n📡 Communication Security Audit:")
        print(f"  • Security Score: {comm_audit.get('security_score', 0):.1f}")
        print(f"  • Issues Found: {len(comm_audit.get('issues', []))}")
    
    # Display comprehensive validation results
    comp_validation = audit_results.get("comprehensive_validation", {})
    if comp_validation and comp_validation.get("status") != "error":
        scan_summary = comp_validation.get("scan_summary", {})
        print(f"\n🔬 Comprehensive Validation:")
        print(f"  • Security Score: {scan_summary.get('security_score', 0):.1f}")
        print(f"  • Total Findings: {scan_summary.get('total_findings', 0)}")
        print(f"  • Scan Duration: {scan_summary.get('scan_duration_seconds', 0):.2f}s")
    
    # Display combined security score
    combined_score = audit_results.get("combined_security_score", 0)
    print(f"\n🎯 Combined Security Score:")
    print_security_score(combined_score)
    
    # Display security benchmarks
    benchmarks = audit_results.get("security_benchmarks", {})
    if benchmarks:
        scan_perf = benchmarks.get("scan_performance", {})
        if scan_perf:
            print(f"\n📈 Performance Benchmarks:")
            print(f"  • Average Scan Time: {scan_perf.get('average_scan_time', 0):.2f}s")
            print(f"  • Total Scans: {scan_perf.get('total_scans', 0)}")
            print(f"  • Throughput: {scan_perf.get('throughput_findings_per_second', 0):.2f} findings/sec")
    
    return audit_results


def export_results(results: dict, filename: str):
    """Export results to JSON file."""
    try:
        output_file = project_root / "examples" / filename
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results exported to: {output_file}")
        return True
    except Exception as e:
        print(f"❌ Failed to export results: {e}")
        return False


def main():
    """Main demonstration function."""
    print_banner("Grid-Fed-RL-Gym Security Validation System Demo")
    
    print("This demonstration showcases the comprehensive security validation system")
    print("that has been implemented for the Grid-Fed-RL-Gym project.")
    print(f"\nProject directory: {project_root}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    try:
        # Demo 1: Comprehensive Security Scan
        scan_results = demo_comprehensive_security_scan()
        
        # Demo 2: Security Hardening
        demo_security_hardening()
        
        # Demo 3: Real-time Monitoring
        demo_real_time_monitoring()
        
        # Demo 4: Integrated Audit
        audit_results = demo_integrated_audit()
        
        # Export results
        if scan_results:
            export_results(scan_results, "security_scan_results.json")
        
        if audit_results:
            export_results(audit_results, "integrated_audit_results.json")
        
        print_banner("Demo Completed Successfully")
        
        print("\n✨ Summary of Security Validation System Features:")
        print("  🔍 Comprehensive vulnerability scanning")
        print("  🛡️  Real-time security monitoring")
        print("  🔧 Automated security hardening")
        print("  📊 Performance benchmarking")
        print("  📜 Compliance validation")
        print("  🏥 Health monitoring integration")
        print("  📡 Secure communication protocols")
        print("  📈 Trend analysis and reporting")
        
        print("\n🎯 Next Steps:")
        print("  1. Review the exported JSON reports for detailed findings")
        print("  2. Address any critical or high-severity security issues")
        print("  3. Implement regular automated security scans")
        print("  4. Monitor real-time security metrics")
        print("  5. Keep dependencies updated to address vulnerabilities")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())