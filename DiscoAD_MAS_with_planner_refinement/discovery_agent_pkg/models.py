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
        "\nIMPORTANT: Review the Planning Agent's recommendations in the conversation history and prioritize following their suggested analytical approaches, statistical methods, and data exploration strategies. "
        "\nThe Planning Agent has provided multiple options for analysis - choose the most appropriate approach based on the hypothesis and available data. "
        "\nIf you write code, use print() statements to show output, as this script does not display notebook-style expressions. "
        "\nAlways return code inside a SINGLE triple backtick python block: ```python ... ```, which will be run. "
        "\nEach block is a fresh running environment, so import any libraries you need each time. "
        "\nFocus on data analysis, statistical computations, and numerical insights as guided by the Planning Agent's recommendations. "
        "\nKeep the code simple and concise. "
        "\nDo not create plots or visualizations - that will be handled by a separate agent. "
        "\nDo not repeat any previously written code for insights already obtained. "
        "\nContain all the code in a single block ONLY. "
        "\nDo not make assumptions and ensure proper error handling in the code. "
        "\nUse ./working/ directory to save any files. "
    )
    return Agent(
        name="Scientist",
        system_message="You are a Scientist. Propose code and analysis to validate hypotheses, following the strategic plans and recommendations provided by the Planning Agent.",
        model_client=model_client,
        query=scientist_query
    )


def create_critic(model_client: ModelClient) -> Agent:
    """Create Critic agent."""
    critic_query = (
        "\nIf you find mistakes in the experiments to validate the hypothesis, suggest corrections. "
        "\nSome times the execution output may be truncated, so ask for more details if needed. "
        "\nIf satisfied, reply with 'RESULTS APPROVED'. "
        "\nDo not approve if the code is wrong and do not write the code on your own, only suggest corrections and reply with 'RESULTS NOT APPROVED'. "
        "\nIf code is good, but you need more insights, reply with 'RESULTS NOT APPROVED' and suggest newer directions and next steps. "
        "\nIf there are good insights, just say 'RESULTS APPROVED'. "
        # "\nIf there is even a small insight in the code or output, reply with 'RESULTS APPROVED'. "
    )
    return Agent(
        name="Critic",
        system_message="You are a Critic. Review code and output, suggest corrections if needed, and reply 'RESULTS APPROVED' when satisfied.",
        model_client=model_client,
        query=critic_query
    )


def create_data_visualizer(model_client: ModelClient) -> Agent:
    """Create Data Visualizer agent."""
    visualizer_query = (
        "\nCreate visualizations based on the scientist's analysis and the Planning Agent's recommendations. "
        "\nIMPORTANT: Review the Planning Agent's visualization suggestions in the conversation history and prioritize following their recommended plot types, styles, and presentation approaches. "
        "\nThe Planning Agent has provided multiple visualization options - choose the most appropriate approach based on the data and analysis results. "
        "\nIf no meaningful plots can be created, respond with exactly: NO DATA TO VISUALIZE. "
        "\nOtherwise, provide code in a fresh ```python ... ``` block. "
        "\nImport any libraries you need each time. "
        "\nSave plots to ./working/ directory. "
        "\nKeep the code simple and concise. "
        "\nFocus on creating visualizations that align with the Planning Agent's strategic recommendations. "
    )
    return Agent(
        name="Data Visualizer",
        system_message="You are a Data Visualizer. Your job is to create and save relevant plots based on the scientist's analysis and the Planning Agent's visualization recommendations. Focus on creating meaningful visualizations that support the scientific findings and follow the strategic plans provided by the Planning Agent.",
        model_client=model_client,
        query=visualizer_query
    )


def create_planning_agent(model_client: ModelClient) -> Agent:
    """Create Planning Agent."""
    planning_query = (
        "\nBased on the conversation so far, create comprehensive plans for both the Scientist and Data Visualizer. "
        "\nProvide multiple alternative approaches and strategies for each agent. "
        "\nFor the Scientist, suggest different analytical approaches, statistical methods, and data exploration strategies. "
        "\nFor the Data Visualizer, suggest different types of visualizations, plot styles, and presentation approaches. "
        "\nStructure your response with clear sections for each agent and multiple numbered options. "
        "\nDo not write code - focus on strategic planning and methodology suggestions. "
        "\nConsider the hypothesis being tested and the available data when making recommendations. "
    )
    return Agent(
        name="Planning Agent",
        system_message="You are a Planning Agent. Your role is to create comprehensive, multi-option plans for both the Scientist and Data Visualizer agents. Focus on strategic planning, methodology suggestions, and alternative approaches rather than writing code.",
        model_client=model_client,
        query=planning_query
    )

class HypothesisGenerator:
    """Hypothesis generator for generating multiple hypotheses from user queries."""
    
    def __init__(self, model_client):
        self.model_client = model_client
    
    def generate(self, user_query):
        """Generate a list of simple, relevant, and verifiable hypotheses from user query."""
        prompt = (
            "You are a hypothesis generator for Alzheimer's Disease and Related Dementias (ADRD) research. "
            "Generate a Python list of very simple, direct, and testable scientific hypotheses based on the following research query. "
            "Focus on straightforward relationships that can be easily investigated using the given keywords.\n\n"
            "Guidelines:\n"
            "- Each hypothesis should be clear, direct, and verifiable given a dataset through basic statistical analysis\n"
            "- Focus on the given keywords\n"
            "- Avoid complex or multi-step relationships; keep each hypothesis as simple as possible\n"
            "- The number of hypotheses is flexible; generate only as many as are relevant and easily testable\n\n"
            "Return only the list, formatted as a Python list of strings.\n\n"
            "Example:\n"
            "User query: Investigate the relationship between age, education, and cognitive performance\n"
            "['Age is negatively associated with cognitive performance.', "
            "'Higher education is associated with better cognitive performance.', "
            "'Older adults with higher education have better cognitive scores than those with lower education.']\n\n"
            "Now, for the research query: '" + user_query + "', generate the list of simple scientific hypotheses."
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
