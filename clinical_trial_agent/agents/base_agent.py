from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
from datetime import datetime
import logging
import os
import time
import asyncio
from dotenv import load_dotenv

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from langchain_openai import ChatOpenAI
from openai import RateLimitError

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class BaseClinicalAgent(ABC):
    """Base class for clinical trial agents."""
    
    def __init__(
        self,
        name: str,
        description: str,
        tools: List[Any],
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.name = name
        self.description = description
        self.tools = tools
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        self.model = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            api_key=api_key
        )
        self.agent_executor = self._create_agent_executor()
        
    def _create_agent_executor(self) -> AgentExecutor:
        """Create the agent executor with tools and prompt."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        llm_with_tools = self.model.bind(
            functions=[format_tool_to_openai_function(t) for t in self.tools]
        )
        
        agent = (
            {
                "input": lambda x: x["input"],
                "chat_history": lambda x: x["chat_history"],
                "agent_scratchpad": lambda x: format_to_openai_function_messages(
                    x["intermediate_steps"]
                ),
            }
            | prompt
            | llm_with_tools
            | OpenAIFunctionsAgentOutputParser()
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )
    
    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the agent."""
        pass
    
    @abstractmethod
    async def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming data and return results."""
        pass
    
    @abstractmethod
    async def generate_explanation(self, prediction: Any) -> str:
        """Generate explanation for the prediction."""
        pass
    
    async def _retry_with_backoff(self, func, *args, **kwargs):
        """Execute function with exponential backoff on rate limit errors."""
        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except RateLimitError as e:
                if attempt == self.max_retries - 1:
                    raise
                wait_time = self.retry_delay * (2 ** attempt)
                logger.warning(f"Rate limit hit, retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
    
    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent on input data."""
        try:
            # Process the data
            processed_data = await self.process_data(input_data)
            
            # Generate prediction with retry logic
            prediction = await self._retry_with_backoff(
                self.agent_executor.ainvoke,
                {
                    "input": processed_data,
                    "chat_history": []
                }
            )
            
            # Generate explanation
            explanation = await self.generate_explanation(prediction)
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "agent_name": self.name,
                "prediction": prediction,
                "explanation": explanation,
                "confidence": self._calculate_confidence(prediction)
            }
            
        except Exception as e:
            logger.error(f"Error in agent {self.name}: {str(e)}")
            raise
    
    def _calculate_confidence(self, prediction: Any) -> float:
        """Calculate confidence score for the prediction."""
        # Implement confidence calculation logic
        return 0.0  # Placeholder 