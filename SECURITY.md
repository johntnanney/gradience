# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.9.x   | :white_check_mark: |
| < 0.9   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in Gradience, please report it responsibly.

**Do NOT open a public GitHub issue for security vulnerabilities.**

Instead, please email [johntnanney@gmail.com](mailto:johntnanney@gmail.com) with:

1. A description of the vulnerability
2. Steps to reproduce the issue
3. Potential impact assessment
4. Any suggested fixes (optional)

### Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial assessment**: Within 1 week
- **Fix or mitigation**: Dependent on severity

### Scope

The following are in scope for security reports:

- Code execution vulnerabilities in CLI commands
- Unsafe deserialization in config/YAML parsing
- Path traversal in file operations
- Dependency vulnerabilities with known exploits
- Information disclosure through telemetry or logging

### Out of Scope

- Model accuracy or compression quality issues
- Performance optimization suggestions
- Feature requests

## Security Best Practices for Users

- Always validate configs from untrusted sources before running
- Use virtual environments to isolate dependencies
- Keep dependencies updated (`pip install --upgrade gradience`)
- Review adapter files from untrusted sources before auditing
