"""
Simple model client and agent classes for the Discovery Agent.
"""

import torch
import ast
import time
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any
from .config import config as default_config

logger = logging.getLogger(__name__)


class ModelClient:
    """Simple HuggingFace model client with robust error handling."""
    
    def __init__(self, config=None):
        # Use provided config or fall back to default
        config = config if config is not None else default_config
        self.model_name = config.model_name
        self.device = config.device if config.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = config.max_new_tokens
        
        print(f"Loading model: {self.model_name}")
        logger.info(f"Initializing model client with device: {self.device}")
        
        try:
            # Load tokenizer with error handling
            print("📝 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            print("✅ Tokenizer loaded successfully")
            
            # Load model with error handling
            print("🧠 Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            # Move model to device if needed
            if self.device != "cuda" or "auto" not in str(self.model.device):
                print(f"🔄 Moving model to {self.device}...")
                self.model.to(self.device)
            
            print("✅ Model loaded successfully")
            logger.info("Model client initialized successfully")
            
        except Exception as e:
            error_msg = f"Failed to initialize model client: {str(e)}"
            print(f"❌ {error_msg}")
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def generate(self, messages: List[Dict[str, Any]], max_retries: int = 3) -> str:
        """Generate text response from messages with retry mechanism."""
        for attempt in range(max_retries):
            try:
                logger.info(f"Generating response (attempt {attempt + 1}/{max_retries})")
                
                formatted_msgs = []
                for message in messages:
                    formatted_msgs.append({
                        "role": message["role"],
                        "content": message["content"]
                    })
                
                # Apply chat template with error handling
                try:
                    inputs = self.tokenizer.apply_chat_template(
                        formatted_msgs,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                    ).to(self.model.device)
                except Exception as e:
                    logger.warning(f"Chat template failed, using simple format: {e}")
                    # Fallback to simple format
                    text = ""
                    for msg in formatted_msgs:
                        text += f"{msg['role']}: {msg['content']}\n"
                    inputs = self.tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=min(4096, self.max_new_tokens * 2)  # Allow input to be up to 2x max_new_tokens, capped at 4096
                    ).to(self.model.device)
                
                # Generate with timeout handling
                start_time = time.time()
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.1
                    )
                
                generation_time = time.time() - start_time
                logger.info(f"Generation completed in {generation_time:.2f} seconds")
                
                # Decode output
                generated = self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[-1]:],
                    skip_special_tokens=True
                )
                
                if not generated.strip():
                    raise ValueError("Generated empty response")
                
                return generated.strip()
                
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"CUDA OOM error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    torch.cuda.empty_cache()
                    time.sleep(2)
                    continue
                else:
                    raise RuntimeError("CUDA out of memory after all retries") from e
                    
            except Exception as e:
                logger.error(f"Generation error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    raise RuntimeError(f"Generation failed after {max_retries} attempts: {e}") from e
        
        raise RuntimeError("All generation attempts failed")


class Agent:
    """Simple agent class."""
    
    def __init__(self, name: str, system_message: str, model_client: ModelClient, query: str = ""):
        self.name = name
        self.system_message = system_message
        self.model_client = model_client
        self.query = query

    def generate(self, history: List[Dict[str, Any]]) -> str:
        """Generate response for given history."""
        # Build conversation history as text
        history_str = ""
        for entry in history:
            history_str += f"<{entry['role']}>\n{entry['content']}\n</{entry['role']}>\n\n"
        
        # Create messages
        messages = [
            {"role": "system", "content": self.system_message},
            {
                "role": "user", 
                "content": f"Here is the conversation so far:\n{history_str}Please join the conversation as the {self.name} and respond in second-person."
            }
        ]
        messages[1]["content"] += self.query
        return self.model_client.generate(messages)


def create_scientist(model_client: ModelClient) -> Agent:
    """Create Scientist agent."""
    scientist_query = (
        "\nData Preprocessor has saved the dataset to ./working/ directory. "
        "\nThere will be one CSV file in the ./working/ directory which is the filtered dataset. "
        "\n\nYOUR ROLE: You are a Scientist who WRITES AND EXECUTES CODE to test hypotheses. "
        "\nReview the Planning Agent's brief strategy recommendations and implement them through code. "
        "\n\nIMPORTANT: You must write Python code to analyze the data. Use print() statements to show results. "
        "\nAlways return code inside a SINGLE triple backtick python block: ```python ... ``` "
        "\nEach block is a fresh environment, so import libraries as needed. "
        "\n\nSTART WITH EDA: Begin your analysis with comprehensive Exploratory Data Analysis (EDA) with robust error handling. "
        "\nInclude data loading, shape, data types, missing values, basic statistics, and data quality assessment. "
        "\nDo not create any plots or visualizations - focus on numerical summaries and data understanding. "
        "\nUse try-except blocks and proper error handling throughout your code. "
        "\n\nCODE SIMPLICITY: Write CONCISE and SIMPLE code. Avoid complex nested operations and assumptions about data structure. "
        "\nAlways check if data exists before using it. Use basic pandas operations and simple statistical tests. "
        "\nBreak complex analyses into simple, step-by-step operations with error checking at each step. "
        "\nDo not assume specific groups or categories exist in the data - always verify first. "
        "\n\nSTATISTICAL ANALYSIS CAPABILITIES: "
        "\nYou have access to comprehensive statistical analysis tools. Use these methods as appropriate: "
        "\n\nDESCRIPTIVE STATISTICS: "
        "\n- Comprehensive data exploration and summary statistics "
        "\n- Distribution analysis and normality testing "
        "\n- Missing data assessment and handling strategies "
        "\n\nHYPOTHESIS TESTING: "
        "\n- Parametric tests (t-tests, ANOVA) and non-parametric alternatives "
        "\n- Chi-square tests for categorical associations "
        "\n- Normality and distribution tests "
        "\n- Multiple comparison corrections when appropriate "
        "\n\nCORRELATION & ASSOCIATION: "
        "\n- Pearson, Spearman, and Kendall correlations "
        "\n- Point-biserial and other specialized correlations "
        "\n- Association measures for categorical variables "
        "\n\nREGRESSION ANALYSIS: "
        "\n- Linear and logistic regression models "
        "\n- Polynomial and non-linear relationships "
        "\n- Regularization techniques (Ridge, Lasso) "
        "\n- Multiple regression with detailed statistical output "
        "\n\nMACHINE LEARNING: "
        "\n- Classification and clustering algorithms "
        "\n- Dimensionality reduction techniques "
        "\n- Model validation and performance evaluation "
        "\n\nADVANCED ANALYSES: "
        "\n- Survival analysis for time-to-event data "
        "\n- Time series analysis and forecasting "
        "\n- Mediation and moderation analysis "
        "\n- Interaction effects and subgroup analysis "
        "\n\nVALIDATION & EVALUATION: "
        "\n- Cross-validation strategies "
        "\n- Model performance metrics and diagnostic plots "
        "\n- Bootstrap confidence intervals "
        "\n- Sensitivity analysis and robustness checks "
        "\n\nEFFECT SIZES & POWER: "
        "\n- Cohen's d, Cramér's V, eta-squared "
        "\n- Power analysis and sample size considerations "
        "\n- Practical significance assessment "
        "\n\nFocus on rigorous statistical analysis, proper significance testing, effect sizes, and comprehensive evaluation. "
        "\nAlways report p-values, confidence intervals, effect sizes, and interpret results in context. "
        "\nUse appropriate statistical methods based on data characteristics and research questions. "
        "\nCreate visualizations and plots that help validate the hypothesis and communicate findings clearly. "
        "\nDo not repeat previously conducted analyses - build upon existing insights. "
        "\nEnsure all analyses are well-documented and scientifically rigorous. "
        "\nUse ./working/ directory to save any files. "
    )
    return Agent(
        name="Scientist",
        system_message="You are a Scientist with comprehensive statistical analysis capabilities. You can perform advanced statistical tests, machine learning, regression analysis, survival analysis, and rigorous hypothesis testing. Always use appropriate statistical methods and report significance levels, effect sizes, and confidence intervals.",
        model_client=model_client,
        query=scientist_query
    )


def create_critic(model_client: ModelClient) -> Agent:
    """Create Critic agent."""
    critic_query = (
        "\nYou are an extremely demanding and critical reviewer. Your standards are exceptionally high. "
        "\nExamine every aspect of the experiments with ruthless scrutiny. Look for: "
        "\n- Statistical significance and proper sample sizes "
        "\n- Appropriate control groups and experimental design "
        "\n- Correct data preprocessing and feature engineering "
        "\n- Proper validation methodology and cross-validation "
        "\n- Meaningful metrics and comprehensive evaluation "
        "\n- Clear interpretation of results and limitations "
        "\n- Reproducible and well-documented code "
        "\n\nBe extremely skeptical. Question every assumption, methodology, and conclusion. "
        "\nIf you find ANY mistakes, flaws, or areas for improvement, reply with 'RESULTS NOT APPROVED' and provide high-level feedback on what's wrong. "
        "\nFocus on major issues like: experimental design flaws, statistical methodology problems, data quality issues, or code errors. "
        "\nKeep your feedback concise and actionable - identify the problem, not detailed solutions. "
        "\nOnly approve if the work demonstrates exceptional quality, rigor, and scientific merit. "
        "\nIf the code is technically correct but lacks depth, insight, or proper scientific methodology, reply with 'RESULTS NOT APPROVED'. "
        "\nIf the results are not statistically robust or the analysis is superficial, reply with 'RESULTS NOT APPROVED'. "
        "\nOnly reply with 'RESULTS APPROVED' if the work meets the highest standards of scientific excellence. "
        "\nRemember: mediocrity is not acceptable. Demand excellence in every aspect. "
    )
    return Agent(
        name="Critic",
        system_message="You are an extremely demanding and critical reviewer with exceptionally high standards. You ruthlessly scrutinize every aspect of scientific work and only approve results that meet the highest standards of excellence. Mediocrity is not acceptable.",
        model_client=model_client,
        query=critic_query
    )

def create_planning_agent(model_client: ModelClient) -> Agent:
    """Create Planning Agent."""
    planning_query = (
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
    )
    return Agent(
        name="Planning Agent",
        system_message="You are a Research Planning Agent with expertise in statistical methodology and research design. Your role is to develop comprehensive analytical strategies that guide the Scientist's advanced statistical analysis capabilities. Design rigorous, multi-step research approaches that ensure scientific validity and robust hypothesis testing.",
        model_client=model_client,
        query=planning_query
    )


def create_enhanced_planning_agent(model_client: ModelClient, hypothesis: str, filtered_csv_statistics: Dict[str, Any], relevant_variables: Dict[str, str]) -> Agent:
    """Create Enhanced Planning Agent with access to filtered dataset information."""
    
    # Format filtered dataset information
    dataset_info = ""
    if filtered_csv_statistics and 'error' not in filtered_csv_statistics:
        dataset_info = (
            f"\n\nFILTERED DATASET INFORMATION:\n"
            f"- Total rows in filtered dataset: {filtered_csv_statistics['total_rows']:,}\n"
            f"- Total columns in filtered dataset: {filtered_csv_statistics['total_cols']}\n"
            f"- Missing values: {filtered_csv_statistics['missing_data']:,} ({filtered_csv_statistics['missing_percentage']}%)\n"
            f"- Numeric columns: {filtered_csv_statistics['numeric_cols']}\n"
            f"- Categorical columns: {filtered_csv_statistics['categorical_cols']}\n"
        )
        
        if filtered_csv_statistics.get('all_columns'):
            dataset_info += f"- Available columns: {', '.join(filtered_csv_statistics['all_columns'])}\n"
    else:
        dataset_info = "\n\nNote: Filtered dataset statistics not available.\n"
    
    # Format relevant variables information
    variables_info = ""
    if relevant_variables:
        variables_info = "\n\nRELEVANT VARIABLES FOR THIS HYPOTHESIS:\n"
        for var_name, description in relevant_variables.items():
            variables_info += f"- {var_name}: {description}\n"
    else:
        variables_info = "\n\nNote: No specific relevant variables identified.\n"
    
    enhanced_planning_query = (
        f"\nHYPOTHESIS TO TEST: {hypothesis}\n"
        f"{dataset_info}"
        f"{variables_info}"
        "\nBased on the specific hypothesis, filtered dataset characteristics, and relevant variables above, "
        "develop a comprehensive research strategy for the Scientist to test this hypothesis. "
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
        "\nTailor your strategy specifically to the filtered dataset characteristics and relevant variables identified above. "
    )
    
    return Agent(
        name="Enhanced Planning Agent",
        system_message="You are a Research Planning Agent with expertise in statistical methodology and research design. Your role is to develop comprehensive analytical strategies that guide the Scientist's advanced statistical analysis capabilities. You have access to specific hypothesis details, filtered dataset characteristics, and relevant variables to design targeted research approaches.",
        model_client=model_client,
        query=enhanced_planning_query
    )

class HypothesisGenerator:
    """Hypothesis generator for generating multiple hypotheses from user queries and dataset structure."""
    
    def __init__(self, model_client):
        self.model_client = model_client
    
    def generate(self, user_query, dataset_dictionary=None, user_keywords=None, csv_statistics=None):
        """
        Generate a list of novel, relevant, and verifiable hypotheses from user query and dataset structure.
        
        Args:
            user_query (str): The research query from the user
            dataset_dictionary (dict, optional): Dictionary of available variables with descriptions
            user_keywords (list, optional): List of keywords provided by the user
            csv_statistics (dict, optional): Real statistics from the CSV file
        
        Returns:
            list: List of testable hypotheses
        """
        # Build dataset information section
        dataset_info = ""
        if dataset_dictionary:
            # Create a formatted string of available variables
            variables_info = []
            for key, info in dataset_dictionary.items():
                description = info.get('Description', 'No description available')
                variables_info.append(f"Variable: {key} | Description: {description}")
            
            # Build statistics section
            stats_section = ""
            if csv_statistics and 'error' not in csv_statistics:
                stats_section = (
                    f"\n\nDATASET OVERVIEW:\n"
                    f"- Total rows: {csv_statistics['total_rows']:,}\n"
                    f"- Total columns: {csv_statistics['total_cols']}\n"
                    f"- Missing values: {csv_statistics['missing_data']:,} ({csv_statistics['missing_percentage']}%)\n"
                    f"- Numeric columns: {csv_statistics['numeric_cols']}\n"
                    f"- Categorical columns: {csv_statistics['categorical_cols']}\n"
                )
                
                # Add all column names for reference
                if csv_statistics.get('all_columns'):
                    stats_section += f"- All available columns: {', '.join(csv_statistics['all_columns'])}\n"
                
                # Add numeric column names if available
                if csv_statistics.get('numeric_column_names'):
                    stats_section += f"- Numeric columns: {', '.join(csv_statistics['numeric_column_names'])}\n"
                
                # Add categorical column names if available
                if csv_statistics.get('categorical_column_names'):
                    stats_section += f"- Categorical columns: {', '.join(csv_statistics['categorical_column_names'])}\n"
                    
            elif csv_statistics and 'error' in csv_statistics:
                stats_section = f"\n\nNote: {csv_statistics['error']}\n"
            else:
                stats_section = (
                    f"\n\nDATASET STATISTICS:\n"
                    f"- Total variables available: {len(dataset_dictionary)}\n"
                    f"- Variable types: Check data types when loading the dataset\n"
                    f"- Sample size: Will be determined when loading the filtered dataset\n"
                    f"- Missing data patterns: Will be assessed during EDA\n"
                )
            
            dataset_info = (
                f"\n\nAVAILABLE DATASET VARIABLES:\n"
                f"The following variables are available in the dataset for hypothesis testing:\n"
                f"{chr(10).join(variables_info)}\n"
                f"{stats_section}\n"
                f"IMPORTANT: Generate hypotheses that can ONLY use these available variables. "
                f"Do not reference variables that are not in this list. "
                f"Focus on novel relationships between these specific variables.\n"
            )
        
        # Build keywords section
        keywords_info = ""
        if user_keywords:
            keywords_info = (
                f"\n\nUSER KEYWORDS:\n"
                f"The user is particularly interested in these concepts: {', '.join(user_keywords)}\n"
                f"Prioritize hypotheses that relate to these keywords while using the available dataset variables.\n"
            )
        
        prompt = (
            "You are an expert hypothesis generator for Alzheimer's Disease and Related Dementias (ADRD) research. "
            "Generate a Python list of NOVEL, scientifically interesting, and statistically testable hypotheses. "
            "Focus on creating hypotheses that are both innovative and verifiable with the available data.\n\n"
            
            "HYPOTHESIS GENERATION GUIDELINES:\n"
            "- NOVELTY: Generate hypotheses that explore unexpected or understudied relationships\n"
            "- VERIFIABILITY: Each hypothesis must be testable using standard statistical methods\n"
            "- SPECIFICITY: Be specific about the expected relationships and populations\n"
            "- SCIENTIFIC RIGOR: Ensure hypotheses are grounded in scientific literature and biological plausibility\n"
            "- STATISTICAL FEASIBILITY: Consider what statistical tests would be appropriate\n"
            "- EFFECT SIZES: Consider what effect sizes would be meaningful and detectable\n\n"
            
            "HYPOTHESIS TYPES TO CONSIDER:\n"
            "- Interaction effects between multiple variables\n"
            "- Non-linear relationships and threshold effects\n"
            "- Mediation and moderation relationships\n"
            "- Risk factor combinations and protective factors\n"
            "- Temporal relationships and progression patterns\n"
            "- Subgroup differences and heterogeneity\n"
            "- Biomarker relationships and diagnostic patterns\n\n"
            
            f"{dataset_info}"
            f"{keywords_info}"
            
            "RESEARCH QUERY: '" + user_query + "'\n\n"
            
            "Generate novel hypotheses that meet the following criteria:\n"
            "1. Are directly testable with the available variables\n"
            "2. Explore interesting and potentially novel relationships\n"
            "3. Are relevant to the user's research interests\n"
            "4. Can be evaluated using appropriate statistical methods\n"
            "5. Have clear, specific predictions\n\n"
            "Generate as many hypotheses as you deem appropriate - focus on quality and novelty over quantity. "
            "Only include hypotheses that you believe are truly novel and scientifically interesting.\n\n"
            
            "Return only a Python list of hypothesis strings, formatted as:\n"
            "['Hypothesis 1...', 'Hypothesis 2...', 'Hypothesis 3...']\n\n"
            
            "Example format:\n"
            "['Higher education moderates the relationship between age and cognitive decline, with protective effects being stronger in early-stage dementia.', "
            "'The interaction between APOE genotype and cardiovascular risk factors predicts faster cognitive decline in specific cognitive domains.', "
            "'Non-linear threshold effects exist in the relationship between physical activity and cognitive performance, with optimal benefits at moderate intensity levels.']"
        )
        
        messages = [{"role": "user", "content": prompt}]
        response = self.model_client.generate(messages)
        
        # Extract the list from the response
        import re
        
        # Look for a Python list pattern in the response
        list_pattern = r'\[(.*?)\]'
        matches = re.findall(list_pattern, response, re.DOTALL)
        
        if matches:
            # Try to find the most complete list (longest match)
            best_match = max(matches, key=len)
            list_str = f"[{best_match}]"
            
            try:
                hypotheses = ast.literal_eval(list_str)
                if isinstance(hypotheses, list):
                    return [str(h).strip() for h in hypotheses if str(h).strip()]
            except Exception:
                pass
        
        # Fallback: try to parse the entire response as a list
        try:
            hypotheses = ast.literal_eval(response)
            if isinstance(hypotheses, list):
                return [str(h).strip() for h in hypotheses if str(h).strip()]
        except Exception:
            pass
        
        # Final fallback
        return [response.strip()]
