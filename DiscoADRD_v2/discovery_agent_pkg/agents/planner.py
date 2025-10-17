"""
Planning agent for the Discovery Agent.
"""

from .models import Agent


def create_enhanced_planning_agent(model_client) -> Agent:
    """Create Enhanced Planning Agent."""
    enhanced_planning_query = (
        "\nBased on the conversation so far, develop a comprehensive research strategy for the Scientist to test the hypothesis. "
        "\n\nIMPORTANT: If there is a PREVIOUS ITERATION SUMMARY in the conversation history, review it and adjust your strategy accordingly. "
        "\nAvoid repeating failed approaches and suggest alternative methodologies or variables. "
        "\n\nYOUR ROLE: You are a Research Planning Agent who designs comprehensive analytical strategies. "
        "\nThe Scientist has advanced statistical capabilities including regression analysis, survival analysis, machine learning, "
        "\nand sophisticated hypothesis testing. Design strategies that leverage these capabilities effectively. "
        "\n\nSTRATEGY COMPONENTS TO CONSIDER: "
        "\n\n1. EXPLORATORY DATA ANALYSIS (EDA) STRATEGY: "
        "\n- Comprehensive data quality assessment and missing data patterns "
        "\n- Distribution analysis and normality testing for key variables "
        "\n- Outlier detection and handling strategies "
        "\n- Data visualization approach for hypothesis validation "
        "\n\n2. STATISTICAL ANALYSIS FRAMEWORK: "
        "\n- Primary statistical tests appropriate for the hypothesis "
        "\n- Secondary analyses for robustness and sensitivity "
        "\n- Effect size calculations and power analysis considerations "
        "\n- Multiple comparison corrections if needed "
        "\n\n3. ADVANCED ANALYTICAL APPROACHES: "
        "\n- Regression modeling strategies (linear, logistic, survival) "
        "\n- Machine learning approaches for pattern detection "
        "\n- Mediation and moderation analyses if relevant "
        "\n- Subgroup analyses and heterogeneity assessment "
        "\n\n4. VALIDATION AND ROBUSTNESS: "
        "\n- Cross-validation strategies "
        "\n- Sensitivity analyses for key assumptions "
        "\n- Alternative statistical approaches for comparison "
        "\n- Confidence interval and significance testing framework "
        "\n\nProvide a structured, multi-step research strategy (3-4 key steps) that guides the Scientist's comprehensive analysis. "
        "\nEach step should be specific about the analytical approach while allowing the Scientist flexibility in implementation. "
        "\nFocus on scientific rigor, statistical appropriateness, and hypothesis validation. "
        "\nUse any dataset information, hypothesis details, and relevant variables provided in the conversation context. "
    )
    
    return Agent(
        name="Enhanced Planning Agent",
        system_message="You are a Research Planning Agent with expertise in statistical methodology and research design. Your role is to develop comprehensive analytical strategies that guide the Scientist's advanced statistical analysis capabilities. Design rigorous, multi-step research approaches that ensure scientific validity and robust hypothesis testing.",
        model_client=model_client,
        query=enhanced_planning_query
    )
