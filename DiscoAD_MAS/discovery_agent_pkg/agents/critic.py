"""
Critic agent for the Discovery Agent.
"""

from .models import Agent


def create_critic(model_client) -> Agent:
    """Create Critic agent."""
    critic_query = (
        "\nYou are a thorough and critical reviewer with high standards. "
        "\nExamine the experiments carefully and look for: "
        "\n- Statistical significance and appropriate sample sizes "
        "\n- Sound experimental design and control groups "
        "\n- Proper data preprocessing and feature engineering "
        "\n- Valid validation methodology and cross-validation "
        "\n- Meaningful metrics and comprehensive evaluation "
        "\n- Clear interpretation of results and limitations "
        "\n- Reproducible and well-documented code "
        "\n\nBe critical but fair. Question assumptions, methodology, and conclusions. "
        "\nIf you find significant mistakes, major flaws, or critical areas for improvement, reply with 'RESULTS NOT APPROVED' and provide high-level feedback on what's wrong. "
        "\nFocus on major issues like: experimental design flaws, statistical methodology problems, data quality issues, or code errors. "
        "\nKeep your feedback concise and actionable - identify the problem, not detailed solutions. "
        "\nApprove if the work demonstrates good quality, rigor, and scientific merit. "
        "\nIf the code is technically sound and the analysis is reasonable, consider approval even if not perfect. "
        "\nIf the results show statistical validity and the analysis is thorough, reply with 'RESULTS APPROVED'. "
        "\nOnly reply with 'RESULTS APPROVED' if the work meets good standards of scientific quality. "
        "\nRemember: aim for quality work while being constructive in your evaluation. "
    )
    return Agent(
        name="Critic",
        system_message="You are a thorough and critical reviewer with high standards. You carefully examine scientific work and approve results that demonstrate good quality and scientific merit.",
        model_client=model_client,
        query=critic_query
    )
