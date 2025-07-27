# Architecture Documentation

## System Overview

AutoGen Code Review Bot is a dual-agent system that provides automated pull request analysis and code quality feedback using Microsoft AutoGen framework.

## Architecture Diagram

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   GitHub PR     │───▶│  Webhook Handler │───▶│   Bot Service   │
│    Events       │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  Coder Agent    │◀───│  Agent Manager  │
                       │                 │    │                 │
                       └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │ Reviewer Agent  │◀───│  GitHub API     │
                       │                 │    │   Integration   │
                       └─────────────────┘    └─────────────────┘
```

## Component Architecture

### Core Components

#### 1. Agent Manager (`src/autogen_code_review_bot/agents.py`)
- Orchestrates the dual-agent conversation
- Manages agent initialization and configuration
- Handles agent communication protocols

#### 2. Coder Agent
- **Role**: Implementation-focused analysis
- **Responsibilities**:
  - Functionality validation
  - Bug detection
  - Edge case identification
  - Code improvement suggestions

#### 3. Reviewer Agent
- **Role**: Quality and standards enforcement
- **Responsibilities**:
  - Security vulnerability detection
  - Performance optimization suggestions
  - Code style and best practices
  - Documentation completeness

#### 4. GitHub Integration (`src/autogen_code_review_bot/github_integration.py`)
- Webhook event processing
- PR analysis coordination
- Comment posting and status updates
- Repository interaction management

#### 5. Analysis Engine (`src/autogen_code_review_bot/pr_analysis.py`)
- Language detection and routing
- Linter integration and execution
- Results aggregation and formatting
- Cache management for performance

### Supporting Infrastructure

#### Caching System (`src/autogen_code_review_bot/caching.py`)
- **Strategy**: Commit-hash based caching
- **Location**: `~/.cache/autogen-review/`
- **TTL**: 24 hours (configurable)
- **Benefits**: 5x+ speedup for repeated analyses

#### Security Layer
- Token management (`src/autogen_code_review_bot/token_security.py`)
- Subprocess security (`src/autogen_code_review_bot/subprocess_security.py`)
- Input validation and sanitization
- Secret scanning integration

#### Monitoring & Observability (`src/autogen_code_review_bot/monitoring.py`)
- Performance metrics collection
- Error tracking and logging
- System health monitoring
- Resource usage analytics

## Data Flow

### 1. PR Event Processing
```
GitHub PR Event → Webhook → Event Validation → Agent Orchestration
```

### 2. Code Analysis Pipeline
```
Repository Clone → Language Detection → Parallel Linting → Agent Analysis → Result Aggregation
```

### 3. Feedback Loop
```
Agent Conversation → Consensus Building → Comment Generation → GitHub API → PR Update
```

## Configuration Management

### Agent Configuration
```yaml
agents:
  coder:
    model: "gpt-4"
    temperature: 0.3
    focus_areas: ["functionality", "bugs", "edge_cases"]
  
  reviewer:
    model: "gpt-4"  
    temperature: 0.1
    focus_areas: ["security", "performance", "standards"]
```

### Language Support Matrix
| Language   | Linter        | Security Scanner | Performance Tools |
|------------|---------------|------------------|-------------------|
| Python     | ruff, pylint  | bandit          | profile hooks     |
| JavaScript | eslint        | npm audit       | bundle analyzer   |
| TypeScript | eslint        | npm audit       | tsc --noEmit      |
| Go         | golangci-lint | gosec           | go tool pprof     |
| Rust       | clippy        | cargo audit     | criterion         |
| Ruby       | rubocop       | brakeman        | ruby-prof         |

## Performance Characteristics

### Scalability Metrics
- **Small PRs** (< 10 files): 15-30 seconds
- **Medium PRs** (10-50 files): 1-3 minutes
- **Large PRs** (50+ files): 3-8 minutes

### Optimization Strategies
- **Parallel Execution**: 2-3x speedup for multi-language repos
- **Intelligent Caching**: 5x+ speedup for repeat analyses
- **Streaming Analysis**: Real-time feedback for large PRs

## Security Architecture

### Threat Model
- **Code Injection**: Sandboxed execution environment
- **Secret Exposure**: Automated secret scanning
- **API Abuse**: Rate limiting and token validation
- **Data Leakage**: Ephemeral storage and cleanup

### Security Controls
- Input sanitization at all entry points
- Subprocess isolation and timeouts
- Secure token storage and rotation
- Audit logging for all operations

## Deployment Architecture

### Local Development
```
Developer Machine → Local Bot Instance → GitHub API (dev tokens)
```

### Production Deployment
```
GitHub Webhook → Load Balancer → Bot Service Cluster → GitHub API (prod tokens)
```

### Container Strategy
- Multi-stage Docker builds
- Minimal base images (distroless)
- Security scanning in CI/CD
- Resource constraints and monitoring

## Integration Points

### External Services
- **GitHub API**: PR management, commenting, status checks
- **OpenAI/Azure OpenAI**: Agent model inference
- **Language Linters**: External tool integration
- **Security Scanners**: Vulnerability detection

### Internal APIs
- Configuration management API
- Metrics collection API
- Cache management API
- Health check endpoints

## Future Architecture Considerations

### Planned Enhancements
- Multi-repository analysis
- Custom rule engine
- Machine learning feedback integration
- Distributed processing capabilities

### Technical Debt
- Agent conversation persistence
- Advanced caching strategies
- Real-time collaboration features
- Enhanced security controls

## Decision Records

See `docs/adr/` directory for detailed architectural decisions including:
- ADR-001: Agent framework selection
- ADR-002: Caching strategy implementation
- ADR-003: Security model design
- ADR-004: Deployment architecture choices