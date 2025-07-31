# Security Policy

## Supported Versions

We actively support the following versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security vulnerability, please report it privately to our security team.

### How to Report

1. **Email**: Send details to security@terragonlabs.com
2. **Encryption**: Use our PGP key for sensitive information (available on request)
3. **Include**: 
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact assessment
   - Suggested fixes (if any)

### Response Process

1. **Acknowledgment**: We'll acknowledge receipt within 24 hours
2. **Investigation**: Initial assessment within 72 hours  
3. **Updates**: Regular updates every 7 days during investigation
4. **Resolution**: Coordinated disclosure after fix is available

### Security Considerations for Power Grid Applications

Given the critical infrastructure nature of power grids, we maintain strict security standards:

#### Data Protection
- All training data should be anonymized before federated training
- Implement differential privacy for gradient sharing
- Use secure aggregation protocols
- Encrypt all communication channels

#### System Security  
- Validate all inputs to prevent injection attacks
- Implement proper authentication and authorization
- Monitor for unusual behavior patterns
- Maintain audit logs for compliance

#### Operational Security
- Deploy safety constraints and circuit breakers
- Implement graceful degradation mechanisms  
- Maintain backup control systems
- Regular security assessments and penetration testing

### Security Best Practices

When using grid-fed-rl-gym in production:

1. **Environment Isolation**: Run in sandboxed environments
2. **Input Validation**: Validate all external inputs
3. **Access Control**: Implement role-based access control
4. **Monitoring**: Log all system interactions
5. **Updates**: Keep dependencies up to date
6. **Backup**: Maintain secure backups of critical data

### Known Security Considerations

- Federated learning may be vulnerable to model inversion attacks
- Grid simulation data could reveal sensitive infrastructure information
- RL policies may be susceptible to adversarial examples
- Communication protocols need protection against man-in-the-middle attacks

### Responsible Disclosure

We appreciate security researchers who responsibly disclose vulnerabilities. We offer:

- Public acknowledgment (with your permission)
- Coordinated disclosure timeline
- Collaboration on fixes and mitigations

### Security Updates

Security updates will be:
- Released as soon as possible after verification
- Clearly marked in release notes
- Backwards compatible when possible
- Documented with migration guides when breaking changes are necessary

For questions about this security policy, contact: security@terragonlabs.com