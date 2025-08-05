# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records (ADRs) for the AutoGen Code Review Bot project. ADRs are documents that capture important architectural decisions made along with their context and consequences.

## ADR Format

We use a lightweight ADR format based on Michael Nygard's template. Each ADR should include:

- **Status**: Current state of the decision
- **Context**: Background information that led to the decision
- **Decision**: The change we're making
- **Consequences**: Expected outcomes and trade-offs
- **Alternatives Considered**: What other options were considered

## ADR Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [000](000-template.md) | ADR Template | Template | - |
| [001](001-dual-agent-architecture.md) | Dual-Agent Architecture | Accepted | - |
| [ADR-001](ADR-001-agent-framework-selection.md) | Agent Framework Selection | Accepted | 2025-07-27 |
| [ADR-002](ADR-002-caching-strategy.md) | Caching Strategy Implementation | Accepted | 2025-07-27 |
| [ADR-003](ADR-003-security-model.md) | Security Model Design | Accepted | 2025-07-27 |
| [ADR-004](ADR-004-deployment-architecture.md) | Deployment Architecture | Accepted | 2025-07-27 |

## Creating New ADRs

1. Use the next available ADR number
2. Copy the [template](000-template.md) as a starting point
3. Fill in all sections completely
4. Update this README with the new ADR entry
5. Submit for review through normal PR process

## ADR Template

For new ADRs, use the following template:

```markdown
# ADR-XXX: [Title]

## Status
[Proposed | Accepted | Deprecated | Superseded]

## Context
[What is the issue that we're seeing that is motivating this decision or change?]

## Decision
[What is the change that we're proposing and/or doing?]

## Consequences
[What becomes easier or more difficult to do because of this change?]

## Alternatives Considered
[What other options were considered?]
```

## ADR Lifecycle

- **Proposed**: Initial draft, under discussion
- **Accepted**: Decision has been made and implemented
- **Rejected**: Decision was considered but not implemented
- **Deprecated**: Decision is no longer relevant
- **Superseded**: Replaced by a newer ADR

## References

- [Documenting Architecture Decisions](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions) by Michael Nygard
- [ADR GitHub Organization](https://adr.github.io/) for more resources
