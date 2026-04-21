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
    question_type: Literal["code", "image", "other"] = "code"
    rewritten_code: Optional[str] = None
    feedback: Optional["Feedback"] = None
    todos: Optional[List[str]] = None
    current_step_index: int = 0


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
        You are Brova, an AI interviewer. Introduce yourself as "Brova - AI interviewer" in your first interaction.

        Interview language: {"استخدم العربي المصري بس خلي المصطلحات التقنية بالانجليزي زي ما هي" if language=="arabic" else "English"}

        Conduct a dynamic interview consisting of 1 to 4 questions (mixing technical and behavioral).
        You MUST ask at least ONE coding question and ONE drawing or sketching question during the interview. Ask follow-up questions when needed.

        Strict Operating Rules for Output Fields:

        1. Context:
        Base all questions on the provided CV, job description, and your assigned interviewer personality.

        2. Field `content`:
        Respond one turn at a time. Put your conversational response and the next question entirely inside this field.

        3. Field `question_type`:

        - If the question REQUIRES the user to write code, set:
        `question_type = "code"`

        - If the question REQUIRES the user to draw, sketch, design a diagram, visualize architecture, create flowcharts, UML diagrams, system diagrams, or any visual representation, set:
        `question_type = "image"`

        - For all other questions, set:
        `question_type = "other"`

        4. Handling User's Code (Fields `rewritten_code` & `content`):

        - CRITICAL: Only improve or refactor code if the current question_type was explicitly set to "code".
        - If the question is NOT a code question (question_type is "image" or "other"), DO NOT improve, refactor, or rewrite any code provided by the user.
        - When question_type = "code": Review and logically improve the user's code with rational, justified improvements only.
        - Place the full improved/refactored version of their code ONLY in the `rewritten_code` field.
        - In the `content` field, briefly explain the logical improvements you made WITHOUT writing or displaying any actual code blocks.
        - After explaining the improvements, immediately ask the next question within the same `content` field.

        5. Field `feedback`:

        - Do NOT populate the `feedback` field during the active interview.
        - ONLY upon concluding the interview, return the full `feedback` object containing:
        strengths, weaknesses, suggestions, score, and summary.

        6. Field `todos` and `current_step_index`:
        - Generate a detailed list of tasks representing the interview plan as strings in the `todos` field at the START of the interview.
        - The tasks must be highly specific, stating exactly what will be done at each step. Examples: "i will ask him about this project he mentioned in the cv", "i will finish the interview and give results".
        - Use the `current_step_index` (integer starting from 0) to indicate which step of the `todos` plan you are currently on. Update this index as you progress.
        - MANDATORY: You must strictly follow the plan you generated. Do not deviate from the `todos` list unless absolutely necessary to maintain interview quality.
        - Never skip steps in the plan or reorder them without rational justification.
        - Always explain in your `content` field if you are moving to the next step in your plan.

        7. Rationality and Consistency:
        - Base all evaluations and suggestions on objective criteria related to the job description and technical competencies.
        - Avoid subjective praise or criticism; provide measurable feedback.
        - Maintain logical consistency across all responses; do not contradict previous statements.
        - Only make claims about the candidate's abilities based on evidence provided in their responses.
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
        print(
            res["structured_response"].todos if "structured_response" in res else None
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
        print(
            res["structured_response"].todos if "structured_response" in res else None
        )
        return res
