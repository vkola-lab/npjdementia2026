"""
Critic agent for the Discovery Agent.
"""

from typing import List, Dict, Any
from .models import Agent


class CriticAgent(Agent):
    """Enhanced Critic agent with debugging mode."""
    
    def __init__(self, name: str, system_message: str, model_client, query: str = ""):
        super().__init__(name, system_message, model_client, query)
        self.debugging_mode = False
    
    def set_debugging_mode(self, enabled: bool = True):
        """Enable or disable debugging mode."""
        self.debugging_mode = enabled
    
    def generate(self, history: List[Dict[str, Any]]) -> str:
        """Generate response for given history with optional debugging mode."""
        # Build conversation history as text
        history_str = ""
        for entry in history:
            history_str += f"<{entry['role']}>\n{entry['content']}\n</{entry['role']}>\n\n"
        
        # Choose query based on mode
        if self.debugging_mode:
            query = self._get_debugging_query()
        else:
            query = self.query
        
        # Create messages
        messages = [
            {"role": "system", "content": self.system_message},
            {
                "role": "user", 
                "content": f"Here is the conversation so far:\n{history_str}Please join the conversation as the {self.name} and respond in second-person."
            }
        ]
        messages[1]["content"] += query
        return self.model_client.generate(messages)
    
    def _get_debugging_query(self) -> str:
        """Get the debugging-specific query."""
        return (
            "\nCODE EXECUTION FAILED - DEBUGGING MODE\n\n"
            "The Scientist's code failed to execute properly. As a Critic, please analyze the failure and provide specific guidance.\n\n"
            "Your task is to:\n"
            "1. Identify the likely cause of the code execution failure based on the error in the conversation\n"
            "2. Provide specific, actionable feedback to help the Scientist fix the code\n"
            "3. Suggest improvements for code robustness and error handling\n"
            "4. Guide the Scientist toward a working solution\n\n"
            "Focus on:\n"
            "- Syntax and import errors\n"
            "- Data access and file path issues\n"
            "- Logic errors and assumptions\n"
            "- Statistical analysis errors (broadcasting, array shapes, data types)\n"
            "- Data preprocessing issues (missing values, data types, group sizes)\n"
            "- Code complexity and simplification opportunities\n"
            "- Best practices for error handling\n\n"
            "For statistical errors, provide specific guidance on:\n"
            "- Checking data shapes and dimensions before statistical tests\n"
            "- Handling missing values and data type conversions\n"
            "- Ensuring adequate sample sizes for group comparisons\n"
            "- Using appropriate statistical tests for the data structure\n"
            "- For broadcasting errors: check that arrays have compatible shapes\n"
            "- For scipy.stats errors: ensure groups have sufficient data and compatible formats\n"
            "- Always verify data types and remove NaN values before statistical operations\n\n"
            "Provide constructive, specific feedback that will help the Scientist succeed in the next iteration.\n\n"
            "IMPORTANT: Do NOT provide any verdict about the hypothesis (NULL REJECTED/NULL NOT REJECTED/NOT TESTABLE).\n"
            "Since the code failed to execute, you cannot evaluate the hypothesis. Only provide debugging guidance.\n"
            "The conversation will continue after your guidance to allow the Scientist to fix the code."
        )


def create_critic(model_client) -> CriticAgent:
    """Create Critic agent."""
    critic_query = (
        "\nYou are a balanced and constructive reviewer focused on hypothesis evaluation. "
        "\nYour primary task is to determine whether the stated hypothesis has been reasonably tested with valid code implementation. "
        "\nExamine the experiments and look for: "
        "\n- Reasonable statistical analysis and adequate sample sizes "
        "\n- Sound experimental design that properly tests the hypothesis "
        "\n- Appropriate data preprocessing and feature engineering "
        "\n- Valid validation methodology "
        "\n- Meaningful metrics that address the hypothesis "
        "\n- Clear interpretation of results and acknowledgment of limitations "
        "\n- **CRITICAL: Functional, error-free, and well-documented code that produces valid results** "
        "\n\nBe constructive but thorough. Focus on what works well while ensuring code quality. "
        "\nLook for evidence that supports the hypothesis, but be strict about code correctness. "
        "\n**MANDATORY CODE CHECKS:** "
        "\n- Code must run without errors and produce the claimed results "
        "\n- All imports and dependencies must be properly handled "
        "\n- Data loading and preprocessing must be correct "
        "\n- Statistical calculations must be accurate "
        "\n- Output must match the analysis described "
        "\n\n**CRITICAL: STATISTICAL TERMINOLOGY VERIFICATION** "
        "\nVerify that the Scientist uses proper statistical terminology in their interpretation: "
        "\n- When p < α (typically 0.05): Should state 'reject the null hypothesis' or 'there is statistically significant evidence supporting...' "
        "\n- When p ≥ α: Should state 'fail to reject the null hypothesis' or 'there is insufficient evidence to support...' "
        "\n- NEVER should say 'accept the null hypothesis' or 'prove the hypothesis is true/false' (these are incorrect) "
        "\n- Results should be interpreted as 'evidence supporting' or 'insufficient evidence' rather than absolute proof "
        "\n- If the Scientist uses incorrect terminology, note this in your feedback but don't necessarily mark as NULL NOT REJECTED if the statistical analysis itself is correct "
        "\n\n**CRITICAL HYPOTHESIS ALIGNMENT CHECK:** "
        "\n- Results must actually support the stated hypothesis direction "
        "\n- If results contradict the hypothesis (e.g., hypothesis says 'higher X leads to higher Y' but results show 'higher X leads to lower Y'), this is NOT verification "
        "\n- Statistical significance alone is insufficient - the direction of effect must match the hypothesis "
        "\n- For 'NULL REJECTED': The null hypothesis should be rejected (p < α) AND the effect direction must match the hypothesis "
        "\n- For 'NULL NOT REJECTED': Either (1) null hypothesis not rejected (p ≥ α), OR (2) effect direction contradicts hypothesis, OR (3) code errors/methodological flaws "
        "\n- If the hypothesis direction is unclear or ambiguous, clarify this in your response "
        "\n\n**IMPORTANT: When issues are found, provide constructive suggestions for improvement while maintaining appropriate status.** "
        "\nIf you find code errors, incorrect calculations, results that don't match the analysis, OR results that contradict the hypothesis direction, reply with 'NULL NOT REJECTED' and: "
        "\n1. Clearly specify the issues found "
        "\n2. Provide specific suggestions for how to fix the problems "
        "\n3. Explain what changes would be needed to achieve verification "
        "\n4. Be encouraging about the approach while being clear about what needs improvement "
        "\n\n**CRITICAL: DATASET LIMITATIONS CHECK** "
        "\nUse 'NOT TESTABLE' ONLY when dataset limitations discovered DURING ANALYSIS prevent meaningful hypothesis testing. "
        "\nDo NOT use 'NOT TESTABLE' if the system already caught 'no relevant variables' in preprocessing - that case is already handled. "
        "\nUse 'NOT TESTABLE' when: "
        "\n- During analysis, you discover insufficient data (e.g., too many missing values after filtering, inadequate sample size for the specific test, missing required columns for the analysis) "
        "\n- The analysis reveals that the available data cannot meaningfully test the hypothesis (e.g., all values are the same, insufficient variation, data quality issues discovered during testing) "
        "\n- When marking as 'NOT TESTABLE', explain the specific dataset limitations that prevent hypothesis testing "
        "\n- These limitations should be discovered during the statistical analysis phase, not obvious from the start "
        "\n\nFor methodology, be moderately lenient - accept reasonable approaches even if not perfect. "
        "\nReply with 'NULL REJECTED' ONLY if: (1) the null hypothesis is rejected (p < α, typically 0.05), (2) the effect direction matches the hypothesis, (3) the code is correct and functional, AND (4) the statistical analysis is sound. "
        "\nReply with 'NULL NOT REJECTED' if: (1) null hypothesis not rejected (p ≥ α), OR (2) effect direction contradicts hypothesis, OR (3) code errors/incorrect results, OR (4) fundamental methodological flaws - but always include constructive feedback on how to improve. "
        "\nReply with 'NOT TESTABLE' if dataset limitations discovered during analysis prevent meaningful hypothesis testing - explain the specific limitations found. "
        "\nIf the hypothesis is NOT TESTABLE, output 'NOT TESTABLE' and terminate. "
        "\n\n**CRITICAL: OUTPUT FORMAT REQUIREMENT** "
        "\nYou MUST format your final verdict using XML tags. At the end of your response, include: "
        "\n<verdict>NULL REJECTED</verdict> OR "
        "\n<verdict>NULL NOT REJECTED</verdict> OR "
        "\n<verdict>NOT TESTABLE</verdict> "
        "\n\nIn your feedback, use proper statistical language: "
        "\n- For NULL REJECTED: State that 'the null hypothesis is rejected' and 'there is statistically significant evidence supporting the hypothesis' "
        "\n- For NULL NOT REJECTED: State that 'we fail to reject the null hypothesis' or 'there is insufficient evidence' or explain the specific issues "
        "\n- Always reference p-values, effect sizes, and confidence intervals when available "
        "\n\nRemember: be supportive of reasonable scientific approaches but strict about code correctness, result validity, hypothesis alignment, and proper statistical interpretation. Always provide actionable feedback to help improve the work. "
    )
    return CriticAgent(
        name="Critic",
        system_message="You are a balanced and constructive reviewer focused on hypothesis evaluation. You determine whether hypotheses have been reasonably tested with valid code implementation, being strict about code correctness, result validity, hypothesis alignment, and proper statistical terminology while supportive of reasonable scientific approaches. You verify that statistical results are interpreted correctly using proper terminology (reject/fail to reject null hypothesis) and that p-values, effect sizes, and confidence intervals are properly reported and interpreted.",
        model_client=model_client,
        query=critic_query
    )
