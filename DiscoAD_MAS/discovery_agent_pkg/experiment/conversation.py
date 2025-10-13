"""
Conversation management module for the Discovery Agent.

This module handles conversation history, context management, and 
interaction between the user and AI agents.
"""

import time
from typing import List, Dict, Any, Optional
from ..ui.interface import ui
from ..agents.models import ModelClient


class ConversationManager:
    """Manages conversation history and context for the Discovery Agent."""
    
    def __init__(self, model_client: ModelClient, config=None):
        """
        Initialize the conversation manager.
        
        Args:
            model_client (ModelClient): Model client for generating responses
            config: Configuration object
        """
        self.model_client = model_client
        self.config = config
        self.messages = []
        self.turns = 0
        self.max_turns = getattr(config, 'auto_approve_after_attempts', 10) if config else 10
    
    def reset_conversation(self) -> None:
        """Reset the conversation to start fresh."""
        self.messages = []
        self.turns = 0
    
    def add_user_input(self, user_input: str) -> None:
        """
        Add user input to the conversation.
        
        Args:
            user_input (str): User's input message
        """
        self.messages.append({
            "role": "user",
            "content": user_input
        })
    
    def add_agent_response(self, agent_name: str, response: str, code_output: str = "") -> None:
        """
        Add an agent response to the conversation.
        
        Args:
            agent_name (str): Name of the agent
            response (str): Agent's response
            code_output (str): Output from code execution (if any)
        """
        # Combine response and code output
        full_response = response
        if code_output:
            full_response += f"\n\nCode Output:\n{code_output}"
        
        self.messages.append({
            "role": "assistant",
            "content": full_response
        })
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get the current conversation history.
        
        Returns:
            List[Dict[str, str]]: Conversation messages
        """
        return self.messages.copy()
    
    def get_turns(self) -> int:
        """
        Get the current number of turns.
        
        Returns:
            int: Number of turns
        """
        return self.turns
    
    def increment_turns(self) -> None:
        """Increment the turn counter."""
        self.turns += 1
    
    def should_auto_approve(self) -> bool:
        """
        Check if we should auto-approve based on turn count.
        
        Returns:
            bool: True if should auto-approve
        """
        return self.turns >= self.max_turns
    
    def truncate_conversation(self, max_messages: int = 20) -> None:
        """
        Truncate conversation to preserve important messages and last 2 messages.
        
        Preserves:
        - Previous iteration summary (if present)
        - First 3 messages (hypothesis, preprocessing, planning)
        - Variable descriptions (if present)
        - Last 2 messages (most recent conversation)
        
        Args:
            max_messages (int): Maximum number of messages to keep (not used in this implementation)
        """
        if len(self.messages) <= 6:  # No truncation needed if 6 or fewer messages
            return
        
        # Check if first message is a previous iteration summary
        has_previous_summary = False
        if self.messages and self.messages[0].get('content', '').startswith('PREVIOUS ITERATION SUMMARY:'):
            has_previous_summary = True
        
        # Find variable descriptions messages
        variable_descriptions_messages = []
        for i, message in enumerate(self.messages):
            content = message.get('content', '')
            if (message.get('role') == 'System' and 
                ('VARIABLE CONTEXT FOR ANALYSIS:' in content or 
                 'AVAILABLE VARIABLES FOR ANALYSIS:' in content)):
                variable_descriptions_messages.append((i, message))
        
        if has_previous_summary:
            # Keep: previous summary + first 3 messages + variable descriptions + last 2 messages
            if len(self.messages) > 6:
                previous_summary = self.messages[0]
                first_three = self.messages[1:4]  # Skip the summary, get next 3
                last_two = self.messages[-2:]
                
                # Combine all preserved messages
                preserved_messages = [previous_summary] + first_three
                
                # Add variable descriptions if they exist
                for _, var_msg in variable_descriptions_messages:
                    if var_msg not in preserved_messages and var_msg not in last_two:
                        preserved_messages.append(var_msg)
                
                preserved_messages.extend(last_two)
                self.messages = preserved_messages
        else:
            # Keep: first 3 messages + variable descriptions + last 2 messages
            if len(self.messages) > 5:
                first_three = self.messages[:3]
                last_two = self.messages[-2:]
                
                # Combine all preserved messages
                preserved_messages = first_three.copy()
                
                # Add variable descriptions if they exist
                for _, var_msg in variable_descriptions_messages:
                    if var_msg not in preserved_messages and var_msg not in last_two:
                        preserved_messages.append(var_msg)
                
                preserved_messages.extend(last_two)
                self.messages = preserved_messages
    
    def create_outer_loop_summary(self) -> str:
        """
        Create a summary of the current conversation for outer loop context.
        
        Returns:
            str: Conversation summary
        """
        try:
            if not self.messages:
                return "No conversation history available."
            
            # Create a summary prompt
            conversation_text = "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in self.messages[-10:]  # Last 10 messages
            ])
            
            summary_prompt = f"""
            Please provide a concise summary of this conversation between a user and AI agents (Scientist, Critic, Planning Agent) working on a scientific hypothesis. 
            Focus on:
            1. What hypothesis was being tested
            2. What approaches were tried
            3. What variables were used
            4. What results were found
            5. Why the hypothesis was not approved (if applicable)
            
            Conversation:
            {conversation_text}
            
            Summary:
            """
            
            # Generate summary using the model
            summary_response = self.model_client.generate_response(
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=300,
                temperature=0.3
            )
            
            return summary_response.strip()
            
        except Exception as e:
            ui.print_warning(f"⚠️ Could not create conversation summary: {str(e)}")
            return f"Conversation summary unavailable due to error: {str(e)}"
    
    def add_system_message(self, content: str) -> None:
        """
        Add a system message to the conversation.
        
        Args:
            content (str): System message content
        """
        self.messages.append({
            "role": "system",
            "content": content
        })
    
    def get_last_message(self) -> Optional[Dict[str, str]]:
        """
        Get the last message in the conversation.
        
        Returns:
            Optional[Dict[str, str]]: Last message or None if empty
        """
        return self.messages[-1] if self.messages else None
    
    def get_message_count(self) -> int:
        """
        Get the total number of messages in the conversation.
        
        Returns:
            int: Number of messages
        """
        return len(self.messages)
    
    def has_messages(self) -> bool:
        """
        Check if the conversation has any messages.
        
        Returns:
            bool: True if conversation has messages
        """
        return len(self.messages) > 0
