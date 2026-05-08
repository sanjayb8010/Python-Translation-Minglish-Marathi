# Security Policy

## Supported Versions

This is a POC project. Security fixes are applied on the default branch.

## Reporting a Vulnerability

Please do not open public issues for sensitive vulnerabilities.

Instead, report privately to the maintainer with:

- Clear description of the issue
- Reproduction steps
- Impact assessment
- Suggested mitigation (if known)

The maintainer will acknowledge the report and coordinate a fix.

## Secret Handling

- Never commit `.env` or credentials
- Use `.env.example` for placeholders only
- Rotate credentials immediately if exposed
- Avoid exposing upstream error internals publicly
