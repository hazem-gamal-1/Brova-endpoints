from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.document_loaders.pdf import PyPDFLoader
from pydantic import BaseModel
from typing import List, Optional, Literal
from langchain.agents.structured_output import ToolStrategy
from dotenv import load_dotenv
import os

load_dotenv()


class Feedback(BaseModel):
    strengths: List[str]
    weaknesses: List[str]
    suggestions: List[str]
    score: float
    summary: str


class Response(BaseModel):
    content: Optional[str] = None
    question_type: Literal["code", "other"] = "code"
    rewritten_code: Optional[str] = None
    feedback: Optional["Feedback"] = None


model = ChatOpenAI(
    model="gpt-4o",
    api_key=os.getenv("GITHUB_TOKEN"),
    base_url="https://models.inference.ai.azure.com",
    temperature=0.8,
)


checkpointer = InMemorySaver()


class Interview:
    def __init__(
        self,
        session_id: str,
        cv_path: str,
        job_description: str,
        interviewer_personality: str = "Friendly",
        language: str = "arabic",
    ):
        self.session_id = session_id
        self.cv_file_path = cv_path
        self.job_desc = job_description
        self.interviewer_personality = interviewer_personality
        self.cv = self._load_cv()
        self.agent = self._prepare_agent(language)

    def _load_cv(self) -> str:
        """Load and parse CV content"""
        loader = PyPDFLoader(self.cv_file_path)
        docs = loader.load()
        return " ".join([doc.page_content for doc in docs])

    def _prepare_agent(self, language):
        system_prompt = f""" 
                  You are the AI interviewer [Brova]. Introduce yourself as "Brova - AI interviewer".

                  Interview language: {"استخدم العربي المصري بس خلي المصطلحات اللي ملهاش ترجمة حرفيه" if language=="arabic" else "English"}

                  Conduct an interview with **dynamic 1 to 4 questions** (technical + behavioral).
                  Ask **follow-up questions** when needed.

                  Rules:
                  1. Base questions on the CV, job description, and interviewer personality.
                  2. Respond **one at a time** with full context inside `content` (response + next question).
                  3. Always use the selected language: {language}.
                  4. only if the question requires that user should write code:
                    - set `question_type` = "code" else "other"
                  5. only when user's answer contains code :
                    - return improved code in `rewritten_code` 
                    - briefly explain improvements in `content` then next question
                    
                  6. Do NOT include `feedback` until the interview ends.
                  7. At the end, return full `feedback`:
                    - strengths, weaknesses, suggestions, score, summary


                don't forget to a single coding question at least 

                  """

        agent = create_agent(
            model=model,
            checkpointer=checkpointer,
            system_prompt=system_prompt,
            response_format=ToolStrategy(Response),
        )
        return agent

    def start(self):
        res = self.agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"start the interviewe for this candidate based on the following cv \n {self.cv} ,\n  job dsecription {self.job_desc},  \n interviewer_personality:{self.interviewer_personality}",
                    }
                ],
            },
            {"configurable": {"thread_id": self.session_id}},
        )
        return res

    def answer(self, candidate_response: str):
        res = self.agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": candidate_response,
                    }
                ],
            },
            {"configurable": {"thread_id": self.session_id}},
        )
        return res
