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
            "IMPORTANT: Do NOT provide any judgment about the hypothesis (HYPOTHESIS VERIFIED/NOT VERIFIED/NOT APPLICABLE).\n"
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
        "\n\n**CRITICAL HYPOTHESIS ALIGNMENT CHECK:** "
        "\n- Results must actually support the stated hypothesis direction "
        "\n- If results contradict the hypothesis (e.g., hypothesis says 'higher X leads to higher Y' but results show 'higher X leads to lower Y'), this is NOT verification "
        "\n- Statistical significance alone is insufficient - the direction of effect must match the hypothesis "
        "\n- If the hypothesis direction is unclear or ambiguous, clarify this in your response "
        "\n\n**IMPORTANT: When issues are found, provide constructive suggestions for improvement while maintaining appropriate status.** "
        "\nIf you find code errors, incorrect calculations, results that don't match the analysis, OR results that contradict the hypothesis direction, reply with 'HYPOTHESIS NOT VERIFIED' and: "
        "\n1. Clearly specify the issues found "
        "\n2. Provide specific suggestions for how to fix the problems "
        "\n3. Explain what changes would be needed to achieve verification "
        "\n4. Be encouraging about the approach while being clear about what needs improvement "
        "\n\n**CRITICAL: DATASET LIMITATIONS CHECK** "
        "\n- If the dataset has insufficient data (e.g., too many missing values, inadequate sample size, missing required columns) to properly test the hypothesis, reply with 'NOT APPLICABLE' "
        "\n- If no relevant variables are found in the dataset to test the hypothesis, reply with 'NOT APPLICABLE' "
        "\n- If the hypothesis cannot be meaningfully tested due to fundamental dataset limitations, reply with 'NOT APPLICABLE' "
        "\n- When marking as 'NOT APPLICABLE', explain the specific dataset limitations that prevent hypothesis testing "
        "\n\nFor methodology, be moderately lenient - accept reasonable approaches even if not perfect. "
        "\nReply with 'HYPOTHESIS VERIFIED' ONLY if: (1) the analysis provides reasonable evidence supporting the hypothesis, (2) the code is correct and functional, AND (3) the results actually align with the stated hypothesis direction. "
        "\nReply with 'HYPOTHESIS NOT VERIFIED' if there are code errors, incorrect results, fundamental methodological flaws, OR if results contradict the hypothesis direction - but always include constructive feedback on how to improve. "
        "\nReply with 'NOT APPLICABLE' if the dataset limitations prevent meaningful hypothesis testing - explain the specific limitations found. "
        "\nThese data limitations could be in the form of no or too little relevant variables selected or not enough samples in the data that contain the relevant data or data combinations. "
        "\nIf the hypothesis is NOT APPLICABLE, output 'NOT APPLICABLE' and terminate. "
        "\n\n**CRITICAL: OUTPUT FORMAT REQUIREMENT** "
        "\nYou MUST format your final judgment using XML tags. At the end of your response, include: "
        "\n<judgment>HYPOTHESIS VERIFIED</judgment> OR "
        "\n<judgment>HYPOTHESIS NOT VERIFIED</judgment> OR "
        "\n<judgment>NOT APPLICABLE</judgment> "
        "\n\nRemember: be supportive of reasonable scientific approaches but strict about code correctness, result validity, and hypothesis alignment. Always provide actionable feedback to help improve the work. "
    )
    return CriticAgent(
        name="Critic",
        system_message="You are a balanced and constructive reviewer focused on hypothesis evaluation. You determine whether hypotheses have been reasonably tested with valid code implementation, being strict about code correctness, result validity, and hypothesis alignment while supportive of reasonable scientific approaches.",
        model_client=model_client,
        query=critic_query
    )
