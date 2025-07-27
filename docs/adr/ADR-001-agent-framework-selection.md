# ADR-001: Agent Framework Selection

## Status
Accepted

## Context
The project requires a multi-agent system for automated code review. The system needs to support:
- Dual-agent conversations (Coder + Reviewer)
- Configurable agent behaviors and personalities
- Integration with various AI models (OpenAI, Azure OpenAI)
- Extensible conversation patterns

## Decision
We have chosen Microsoft AutoGen as the primary agent framework for the following reasons:

1. **Multi-Agent Support**: Native support for complex multi-agent conversations
2. **Model Flexibility**: Works with multiple LLM providers and models
3. **Conversation Management**: Built-in conversation flow and state management
4. **Python Ecosystem**: Strong integration with Python ML/AI ecosystem
5. **Community Support**: Active development and community contributions

## Consequences

### Benefits
- Simplified agent orchestration and conversation management
- Built-in support for different agent roles and personas
- Extensible architecture for adding new agent types
- Strong observability and debugging capabilities

### Challenges
- Dependency on Microsoft AutoGen framework lifecycle
- Learning curve for team members unfamiliar with AutoGen
- Potential vendor lock-in for conversation patterns

## Alternatives Considered

### LangChain Agents
- **Pros**: Extensive ecosystem, multiple integrations
- **Cons**: More complex for multi-agent scenarios, steeper learning curve

### Custom Agent Framework
- **Pros**: Full control, tailored to specific needs
- **Cons**: Significant development overhead, maintenance burden

### CrewAI
- **Pros**: Good multi-agent support, simpler API
- **Cons**: Less mature, smaller community, limited model support

## Implementation Notes
- Agent configurations stored in YAML for easy modification
- Agent behaviors defined through prompt templates
- Conversation patterns implemented using AutoGen's GroupChat functionality