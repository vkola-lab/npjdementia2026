"""
Model client and base agent classes for the Discovery Agent.
"""

import torch
import ast
import time
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any
from ..core.config import config as default_config

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
            
            # Move model to device in all cases
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
                
                # formatted_msgs = []
                # for message in messages:
                #     formatted_msgs.append({
                #         "role": message["role"],
                #         "content": message["content"]
                #     })

                formatted_msgs = messages
                
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
                    raise RuntimeError(f"Generation failed on {attempt}: {e}") from e 
        
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
