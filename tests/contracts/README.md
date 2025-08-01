# Contract Testing

This directory contains contract tests for external service integrations.

## Overview

Contract tests verify that our service correctly communicates with external APIs:

- **GitHub API**: Ensure we correctly handle GitHub webhook payloads and API responses
- **OpenAI API**: Verify our prompts and response parsing work correctly
- **Prometheus Metrics**: Ensure metrics are exposed in the expected format

## Test Structure

```
contracts/
├── github_api/          # GitHub API contract tests
├── openai_api/          # OpenAI API contract tests
├── webhooks/            # Webhook payload contract tests
└── metrics/             # Metrics format contract tests
```

## Running Contract Tests

```bash
# Run all contract tests
pytest tests/contracts/ -m contract

# Run specific contract tests
pytest tests/contracts/github_api/ -v
pytest tests/contracts/openai_api/ -v

# Run with real API calls (use sparingly)
pytest tests/contracts/ -m "contract and integration" --real-api
```

## Contract Test Guidelines

1. **Mock by Default**: Use recorded responses, not live API calls
2. **Version Pinning**: Test against specific API versions
3. **Error Scenarios**: Test error responses and edge cases
4. **Schema Validation**: Validate request/response schemas
5. **Rate Limiting**: Respect API rate limits in integration tests

## Adding New Contract Tests

1. Create a new directory for the service
2. Record actual API responses (sanitized)
3. Create schema validation tests
4. Test error scenarios
5. Document expected behavior changes
EOF < /dev/null
