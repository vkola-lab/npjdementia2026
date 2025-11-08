"""
Planning agent for the Discovery Agent.
"""

from .models import Agent


def create_enhanced_planning_agent(model_client) -> Agent:
    """Create Enhanced Planning Agent."""
    enhanced_planning_query = (
        "\nBased on the conversation so far, develop a focused research strategy for the Scientist to test the hypothesis. "
        "\n\nIMPORTANT: If there is a PREVIOUS ITERATION SUMMARY in the conversation history, review it carefully and adjust your strategy accordingly. "
        "\n- Avoid repeating failed approaches and suggest alternative methodologies, variables, or statistical tests "
        "\n- If previous code failed, suggest simpler approaches with better error handling "
        "\n- If previous tests didn't achieve significance, suggest alternative tests or variable transformations "
        "\n- If previous results contradicted the hypothesis direction, suggest different approaches or variable combinations "
        "\n\nYOUR ROLE: You are a Research Planning Agent who designs focused, actionable strategies for null hypothesis testing. "
        "\nThe Scientist's PRIMARY OBJECTIVE is to reject the null hypothesis (p < α, typically 0.05) through statistical testing. "
        "\nDesign strategies that guide the Scientist toward this goal with clear, simple, and executable steps. "
        "\n\nPRIMARY GOAL ALIGNMENT: "
        "\n- The strategy must focus on null hypothesis testing as the primary objective "
        "\n- Ensure the strategy tests the hypothesis direction correctly (e.g., if hypothesis says 'higher X leads to higher Y', the test should verify this direction) "
        "\n- Suggest statistical tests that can directly test the null hypothesis (t-tests, ANOVA, chi-square, correlations, regression) "
        "\n- Emphasize reporting p-values, effect sizes, and confidence intervals for proper statistical interpretation "
        "\n\nSTRATEGY STRUCTURE (adapt based on iteration): "
        "\n\nFOR FIRST ITERATION (when no previous Scientist responses exist): "
        "\n1. MINIMAL EDA: Brief data structure check (sample size, variable types, missing data patterns) - keep this minimal "
        "\n2. NULL HYPOTHESIS TESTING: Primary statistical test appropriate for the hypothesis (t-test, ANOVA, chi-square, correlation, or regression) "
        "\n3. STATISTICAL REPORTING: Ensure p-values, effect sizes, and confidence intervals are reported "
        "\n4. HYPOTHESIS DIRECTION CHECK: Verify the test can confirm whether results match the hypothesis direction "
        "\n\nFOR SUBSEQUENT ITERATIONS (when previous attempts exist): "
        "\n- Focus exclusively on null hypothesis testing approaches "
        "\n- Build upon previous tests or suggest alternative approaches "
        "\n- If previous tests failed: suggest different statistical tests, variable transformations, or subgroup analyses "
        "\n- If previous code had errors: suggest simpler code with better error handling and data validation "
        "\n- If previous results contradicted hypothesis: suggest different variable combinations or alternative tests "
        "\n\nSTATISTICAL TEST SELECTION GUIDANCE: "
        "\n- FOR COMPARISONS: t-tests (independent/paired), Mann-Whitney U, Wilcoxon, ANOVA, Kruskal-Wallis "
        "\n- FOR ASSOCIATIONS: Chi-square, Pearson/Spearman/Kendall correlations "
        "\n- FOR REGRESSION: Linear regression (test if coefficients ≠ 0), logistic regression "
        "\n- Select tests based on data type (continuous vs categorical), distribution (parametric vs non-parametric), and hypothesis structure "
        "\n\nCODE SIMPLICITY AND ROBUSTNESS: "
        "\n- Suggest simple, concise code approaches focused on null hypothesis testing "
        "\n- Emphasize error handling (try-except blocks) and data validation (check data exists, remove NaN values, verify sample sizes) "
        "\n- Avoid overly complex analyses - prioritize clear, executable statistical tests "
        "\n- Ensure code checks for adequate sample sizes before performing tests "
        "\n\nSTATISTICAL REPORTING REQUIREMENTS: "
        "\n- Strategy should ensure p-values are reported (exact values, not just p < 0.05) "
        "\n- Effect sizes should be included (Cohen's d, correlation coefficients, odds ratios, etc.) "
        "\n- Confidence intervals (typically 95%) should be reported for effect estimates "
        "\n- Proper statistical terminology: 'reject the null hypothesis' (p < α) or 'fail to reject the null hypothesis' (p ≥ α) "
        "\n- Never suggest 'accepting the null hypothesis' or 'proving' the hypothesis "
        "\n\nProvide a structured, focused research strategy (2-3 key steps) that guides the Scientist's null hypothesis testing. "
        "\nEach step should be specific about the statistical approach while emphasizing simplicity and executability. "
        "\nFocus on achieving statistical significance (p < 0.05) while ensuring proper hypothesis direction alignment and statistical reporting. "
        "\nUse any dataset information, hypothesis details, relevant variables, and previous iteration results provided in the conversation context. "
    )
    
    return Agent(
        name="Enhanced Planning Agent",
        system_message="You are a Research Planning Agent with expertise in statistical methodology and research design. Your role is to develop focused, actionable strategies that guide the Scientist's null hypothesis testing. Design clear, simple, and executable research approaches that prioritize rejecting the null hypothesis (p < α) while ensuring proper statistical reporting (p-values, effect sizes, confidence intervals) and hypothesis direction alignment.",
        model_client=model_client,
        query=enhanced_planning_query
    )
