from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class InstructionRequest(BaseModel):
    url: str = Field(...)
    task: str = Field(...)
    steps: str = Field(...)
    expected_behavior: str = Field(...)
    devices: List[str] = Field(default=["desktop"])
    browsers: List[str] = Field(default=["chrome"])
    username: Optional[str] = None
    password: Optional[str] = None
    screen_width: Optional[int] = None  # Custom screen width in pixels
    screen_height: Optional[int] = None  # Custom screen height in pixels

class BrowserContext(BaseModel):
    os: str
    viewport: str
    cookies_enabled: bool

class TaskStep(BaseModel):
    steps: List[str]
    statuses: Optional[List[bool]] = None

class CombinationResult(BaseModel):
    goal: str
    eps_name: str
    task: TaskStep
    start_url: str
    browser_context: BrowserContext
    success: bool
    total_steps: int
    runtime_sec: float
    total_tokens: int
    gpt_output: Optional[str] = None
    wrong_behavior: Optional[bool] = False
    explanation: Optional[str] = None
    expected_behavior: Optional[str] = None
    error_message: Optional[str] = None
    device: str
    browser: str

    class Config:
        extra = "allow"

class PipelineResponse(BaseModel):
    episode_name: str
    task: str
    url: str
    steps: str
    expected_behavior: str
    combinations: List[CombinationResult]
    total_combinations: int
    successful_combinations: int
    failed_combinations: int
    total_runtime_sec: float
    created_at: datetime

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: Optional[int] = None
    message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

    episode_name: Optional[str] = None
    task: Optional[str] = None
    url: Optional[str] = None
    steps: Optional[str] = None
    expected_behavior: Optional[str] = None
    combinations: Optional[List[CombinationResult]] = None
    logs: Optional[List[str]] = None


