"""Configurable response templates for code review agents."""

import random
from typing import Dict, List


class AgentResponseTemplates:
    """Manages response templates for different agent types."""

    def __init__(self):
        """Initialize with default templates."""
        self._templates = {
            "coder": {
                "improvement_focused": [
                    "Looking at the code implementation, I notice potential improvements in {focus_area}",
                    "From a technical standpoint, we could enhance the {focus_area} aspects",
                    "The implementation could benefit from better {focus_area} handling",
                    "I see opportunities to optimize the {focus_area} in this code",
                ],
                "assessment": [
                    "From a coding perspective, this {assessment_result}",
                    "The technical implementation {assessment_result}",
                    "Code-wise, this approach {assessment_result}",
                    "From a developer's viewpoint, this {assessment_result}",
                ],
                "agreement": [
                    "I {agreement_level} with the previous assessment regarding the implementation",
                    "I {agreement_level} about the technical approach discussed",
                    "From a coding perspective, I {agreement_level} with that analysis",
                    "I {agreement_level} with the implementation strategy mentioned",
                ],
            },
            "reviewer": {
                "concern_focused": [
                    "From a review standpoint, I'm {concern_level} about the {focus_area} aspects",
                    "As a reviewer, I have {concern_level} regarding the {focus_area}",
                    "The code review perspective shows {concern_level} about {focus_area}",
                    "From a quality assurance angle, I'm {concern_level} about {focus_area}",
                ],
                "findings": [
                    "The code review indicates {finding_type}",
                    "My analysis reveals {finding_type}",
                    "The review process shows {finding_type}",
                    "Code examination suggests {finding_type}",
                ],
                "opinion": [
                    "I {opinion_type} about the current approach",
                    "I {opinion_type} regarding this implementation strategy",
                    "My review {opinion_type} about the proposed solution",
                    "From a reviewer's perspective, I {opinion_type} about this",
                ],
            },
        }

        self._substitution_options = {
            "focus_area": [
                "performance",
                "error handling",
                "edge cases",
                "security",
                "maintainability",
                "readability",
                "testing",
                "scalability",
            ],
            "assessment_result": [
                "looks solid",
                "needs refactoring",
                "has potential issues",
                "shows good practices",
                "could be improved",
                "demonstrates clear logic",
            ],
            "agreement_level": [
                "agree",
                "disagree",
                "partially agree",
                "strongly agree",
                "somewhat disagree",
            ],
            "concern_level": [
                "concerned",
                "satisfied",
                "very concerned",
                "moderately concerned",
                "pleased",
                "worried",
            ],
            "finding_type": [
                "good practices",
                "areas for improvement",
                "security concerns",
                "performance issues",
                "code quality improvements",
                "architectural considerations",
            ],
            "opinion_type": [
                "concur",
                "have reservations",
                "strongly support",
                "have mixed feelings",
                "am optimistic",
                "see potential issues",
            ],
        }

    def get_response(self, agent_type: str, template_category: str, **kwargs) -> str:
        """Get a response from templates with substitutions.

        Args:
            agent_type: Type of agent ('coder' or 'reviewer')
            template_category: Category of template to use
            **kwargs: Additional substitution values

        Returns:
            Generated response string

        Raises:
            ValueError: If agent_type or template_category not found
        """
        if agent_type not in self._templates:
            raise ValueError(f"Unknown agent type: {agent_type}")

        agent_templates = self._templates[agent_type]
        if template_category not in agent_templates:
            raise ValueError(
                f"Unknown template category '{template_category}' for agent '{agent_type}'"
            )

        # Select a random template from the category
        template = random.choice(agent_templates[template_category])

        # Prepare substitutions
        substitutions = kwargs.copy()

        # Add random substitutions for any placeholders not provided
        for placeholder, options in self._substitution_options.items():
            if placeholder not in substitutions:
                substitutions[placeholder] = random.choice(options)

        # Apply substitutions
        try:
            return template.format(**substitutions)
        except KeyError as e:
            # If substitution fails, return template with a fallback
            return template.replace(f"{{{e.args[0]}}}", f"[{e.args[0]}]")

    def add_template(self, agent_type: str, template_category: str, template: str):
        """Add a new template to the collection.

        Args:
            agent_type: Type of agent to add template for
            template_category: Category to add template to
            template: Template string with placeholders
        """
        if agent_type not in self._templates:
            self._templates[agent_type] = {}

        if template_category not in self._templates[agent_type]:
            self._templates[agent_type][template_category] = []

        self._templates[agent_type][template_category].append(template)

    def add_substitution_option(self, placeholder: str, option: str):
        """Add a new substitution option for a placeholder.

        Args:
            placeholder: Placeholder name (without braces)
            option: Option to add to the list
        """
        if placeholder not in self._substitution_options:
            self._substitution_options[placeholder] = []

        if option not in self._substitution_options[placeholder]:
            self._substitution_options[placeholder].append(option)

    def load_from_config(self, config: Dict):
        """Load templates from configuration dictionary.

        Args:
            config: Configuration dictionary with templates and options
        """
        if "templates" in config:
            self._templates.update(config["templates"])

        if "substitution_options" in config:
            self._substitution_options.update(config["substitution_options"])

    def get_available_templates(self) -> Dict[str, List[str]]:
        """Get available template categories for each agent type.

        Returns:
            Dictionary mapping agent types to their available template categories
        """
        return {
            agent_type: list(templates.keys())
            for agent_type, templates in self._templates.items()
        }

    def get_substitution_options(self) -> Dict[str, List[str]]:
        """Get available substitution options.

        Returns:
            Dictionary mapping placeholder names to their options
        """
        return self._substitution_options.copy()


# Default global instance
default_templates = AgentResponseTemplates()
