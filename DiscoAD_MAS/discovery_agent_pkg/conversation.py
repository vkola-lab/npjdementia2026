"""
Simple conversation management for the Discovery Agent.
"""

from typing import List, Dict, Any
from .config import config as default_config
from .models import ModelClient
from .ui import ui
# No transcript imports needed - using session log only


class ConversationManager:
    """Simple conversation manager."""
    
    def __init__(self, model_client: ModelClient, config=None):
        self.messages: List[Dict[str, Any]] = []
        self.turns = 0
        self.summarizer = model_client
        # Use provided config or fall back to default
        self.config = config if config is not None else default_config

    def add_user_input(self, user_input: str) -> None:
        """Add user input to conversation."""
        self.messages.append({"role": "User", "content": user_input})
        # Log to session
        ui._write_to_log(f"👤 User Input: {user_input}")

    def add_agent_response(self, agent_name: str, response: str, code_output: str = "") -> None:
        """Add agent response to conversation."""
        full_response = response
        if code_output:
            full_response += f"\nExecution Output:\n{code_output}"
        self.messages.append({"role": agent_name, "content": full_response})
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history."""
        return self.messages
    
    def truncate_conversation(self) -> None:
        """Truncate conversation to manage memory."""
        if len(self.messages) <= self.config.max_conversation_length:
            return
        
        # Check if first message is an outer summary (System role with "PREVIOUS ITERATION SUMMARY")
        has_outer_summary = (len(self.messages) > 0 and 
                           self.messages[0].get("role") == "System" and 
                           "PREVIOUS ITERATION SUMMARY" in self.messages[0].get("content", ""))
        
        if has_outer_summary:
            # Structure: [outer_summary, user_input, preprocessing, planning, ...]
            outer_summary = self.messages[0]  # keep outer summary
            user_message = self.messages[1]   # keep user input
            preprocessing_message = self.messages[2]  # keep preprocessing message
            planning_message = self.messages[3]  # keep planning message
            
            # summarize the middle messages starting from index 4 (after outer summary, user input, preprocessing and planning)
            summarized_messages = self.summarize(self.messages[4:-(self.config.max_conversation_length - 4)])
            # keep the last few messages for scientist and critic
            recent_messages = self.messages[-(self.config.max_conversation_length - 4):]
            
            self.messages = [outer_summary, user_message, preprocessing_message, planning_message] + summarized_messages + recent_messages
        else:
            # Original structure: [user_input, preprocessing, planning, ...]
            user_message = self.messages[0]  # keep user input
            preprocessing_message = self.messages[1]  # keep preprocessing message
            planning_message = self.messages[2]  # keep planning message
            
            # summarize the middle messages starting from index 3 (after user input, preprocessing and planning)
            summarized_messages = self.summarize(self.messages[3:-(self.config.max_conversation_length - 3)])
            # keep the last few messages for scientist and critic
            recent_messages = self.messages[-(self.config.max_conversation_length - 3):]
            
            self.messages = [user_message, preprocessing_message, planning_message] + summarized_messages + recent_messages

    
    def summarize(self, messages: List[Dict[str, Any]], preprocessing: bool = False) -> List[Dict[str, Any]]:
        """Summarize messages using the model client."""
        if not messages:
            return []
        
        try:
            # Build conversation text from messages
            conversation_text = ""
            for message in messages:
                role = message.get("role", "Unknown")
                content = message.get("content", "")
                conversation_text += f"{role}: {content}\n"
            
            if preprocessing:
                prompt = (
                    f"Please summarize the following conversation in 3-4 lines maximum. "
                    f"Remember to mention who said what, the roles are mentioned in the conversation text."
                    f"Focus on the key points, decisions, and outcomes:\n\n"
                    f"If the data preprocessor filtered out any columns, mention them in the summary, along with the summary above."
                    f"{conversation_text}\n"
                    f"Provide only the summary and the relevant columns, no other text."
                )
            else:
                prompt = (
                    f"Please summarize the following conversation in 3-4 lines maximum. "
                    f"Remember to mention who said what, the roles are mentioned in the conversation text."
                    f"Focus on the key points, decisions, and outcomes:\n\n"
                    f"{conversation_text}\n"
                    f"Provide only the summary, no other text."
                )
            
            # Use the model client to generate summary
            summary_messages = [{"role": "user", "content": prompt}]
            summary_text = self.summarizer.generate(summary_messages)
            
            if not summary_text or not summary_text.strip():
                # Fallback to simple truncation if summarization fails
                return [{"role": "System", "content": f"Previous conversation truncated (summarization failed)"}]
            
            # Return as a single message entry
            return [{"role": "System", "content": f"Summary of the conversation so far: {summary_text}"}]
            
        except Exception as e:
            # If summarization fails, return a simple fallback
            return [{"role": "System", "content": f"Previous conversation truncated (error: {str(e)})"}]

    def increment_turns(self) -> None:
        """Increment turn counter."""
        self.turns += 1
    
    def should_auto_approve(self) -> bool:
        """Check if should auto-approve."""
        return self.turns >= self.config.auto_approve_after_attempts
    
    def get_turns(self) -> int:
        """Get turn count."""
        return self.turns

    def reset_conversation(self) -> None:
        """Reset both conversation messages and turn counter."""
        self.messages = []
        self.turns = 0
    
    def create_outer_loop_summary(self) -> str:
        """
        Create a comprehensive summary of the entire conversation for outer loop context.
        This summary will be used as a system message in the next outer loop iteration.
        """
        if not self.messages:
            return "No previous conversation to summarize."
        
        try:
            # Build conversation text from all messages
            conversation_text = ""
            for i, message in enumerate(self.messages):
                role = message.get("role", "Unknown")
                content = message.get("content", "")
                conversation_text += f"Turn {i+1} - {role}: {content}\n\n"
            
            prompt = (
                f"Please provide a comprehensive summary of the following conversation that occurred in a previous outer loop iteration. "
                f"This summary will be used as context for a new iteration where we will try different variables/approaches. "
                f"Focus on:\n"
                f"1. The original hypothesis that was tested\n"
                f"2. What data preprocessing was done (which variables were selected/filtered)\n"
                f"3. What analysis approaches were attempted by the Scientist\n"
                f"4. What analysis and visualizations were performed by the Scientist\n"
                f"5. What the Critic found problematic or what was not approved\n"
                f"6. Key insights and lessons learned\n"
                f"7. What should be done differently in the next iteration\n\n"
                f"Conversation:\n{conversation_text}\n\n"
                f"Provide a detailed summary that will help guide the next iteration to avoid the same issues and try different approaches."
            )
            
            # Use the model client to generate comprehensive summary
            summary_messages = [{"role": "user", "content": prompt}]
            summary_text = self.summarizer.generate(summary_messages)
            
            if not summary_text or not summary_text.strip():
                # Fallback to simple summary if generation fails
                return "Previous conversation summary unavailable due to generation error."
            
            return summary_text.strip()
            
        except Exception as e:
            # If summarization fails, return a simple fallback
            return f"Previous conversation summary unavailable due to error: {str(e)}"