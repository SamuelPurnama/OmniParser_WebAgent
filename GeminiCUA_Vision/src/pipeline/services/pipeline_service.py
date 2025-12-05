import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
import os
import threading
import asyncio

from .models import InstructionRequest, PipelineResponse, JobStatus, CombinationResult, TaskStep, BrowserContext
from .pipeline_runner import pipeline_runner

class PipelineService:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}

    # Synchronous background entry point (recommended for Django)
    def run_pipeline_background(self, instruction: InstructionRequest) -> str:
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {
            "status": "pending",
            "progress": 0,
            "message": "Starting pipeline...",
            "created_at": datetime.now(),
            "instruction": instruction,
            "result": None,
            "error": None
        }
        thread = threading.Thread(target=self._execute_pipeline_sync, args=(job_id, instruction), daemon=True)
        thread.start()
        return job_id

    def _execute_pipeline_sync(self, job_id: str, instruction: InstructionRequest):
        try:
            base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "results", job_id)
            os.makedirs(base_dir, exist_ok=True)
            log_path = os.path.join(base_dir, "log.txt")
            def log(line: str):
                try:
                    ts = datetime.now().isoformat()
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(f"[{ts}] {line}\n")
                except Exception:
                    pass

            self.jobs[job_id].update({
                "status": "running",
                "progress": 10,
                "message": "Initializing pipeline...",
                "completed_combinations": [],
                "total_combinations": len(instruction.devices) * len(instruction.browsers)
            })
            log("Pipeline started")
            # Run synchronously in this background thread
            response: PipelineResponse = pipeline_runner.run_pipeline(
                instruction,
                instruction.task.lower().replace(" ","_"),
                base_dir
            )
            log("Pipeline completed")
            self.jobs[job_id].update({
                "status": "completed",
                "progress": 100,
                "message": "Pipeline completed successfully",
                "completed_at": datetime.now(),
                "result": response,
                "episode_name": response.episode_name,
                "task": response.task,
                "url": response.url,
                "steps": response.steps,
                "expected_behavior": response.expected_behavior
            })
            # Write shared metadata for status endpoint
            shared = {
                "episode_name": response.episode_name,
                "task": response.task,
                "url": response.url,
                "steps": response.steps,
                "expected_behavior": response.expected_behavior,
                "combinations": [r.model_dump(mode="json") for r in response.combinations]  # type: ignore
            }
            with open(os.path.join(base_dir, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(shared, f, indent=2)
        except Exception as e:
            try:
                base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "results", job_id)
                os.makedirs(base_dir, exist_ok=True)
                with open(os.path.join(base_dir, "log.txt"), "a", encoding="utf-8") as f:
                    f.write(f"[{datetime.now().isoformat()}] Pipeline failed: {str(e)}\n")
            except Exception:
                pass
            self.jobs[job_id].update({
                "status": "failed",
                "progress": 0,
                "message": f"Pipeline failed: {str(e)}",
                "completed_at": datetime.now(),
                "error": str(e)
            })

    async def run_pipeline_async(self, instruction: InstructionRequest) -> str:
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {
            "status": "pending",
            "progress": 0,
            "message": "Starting pipeline...",
            "created_at": datetime.now(),
            "instruction": instruction,
            "result": None,
            "error": None
        }
        asyncio.create_task(self._execute_pipeline(job_id, instruction))
        return job_id

    async def _execute_pipeline(self, job_id: str, instruction: InstructionRequest):
        try:
            self.jobs[job_id].update({
                "status": "running",
                "progress": 10,
                "message": "Initializing pipeline...",
                "completed_combinations": [],
                "total_combinations": len(instruction.devices) * len(instruction.browsers)
            })
            loop = asyncio.get_event_loop()
            response: PipelineResponse = await loop.run_in_executor(
                None, pipeline_runner.run_pipeline, instruction, instruction.task.lower().replace(" ","_"),
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "results", job_id)
            )
            self.jobs[job_id].update({
                "status": "completed",
                "progress": 100,
                "message": "Pipeline completed successfully",
                "completed_at": datetime.now(),
                "result": response,
                "episode_name": response.episode_name,
                "task": response.task,
                "url": response.url,
                "steps": response.steps,
                "expected_behavior": response.expected_behavior
            })
            # Write shared metadata for status endpoint
            base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "results", job_id)
            os.makedirs(base_dir, exist_ok=True)
            shared = {
                "episode_name": response.episode_name,
                "task": response.task,
                "url": response.url,
                "steps": response.steps,
                "expected_behavior": response.expected_behavior,
                "combinations": [r.model_dump(mode="json") for r in response.combinations]  # type: ignore
            }
            with open(os.path.join(base_dir, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(shared, f, indent=2)
        except Exception as e:
            self.jobs[job_id].update({
                "status": "failed",
                "progress": 0,
                "message": f"Pipeline failed: {str(e)}",
                "completed_at": datetime.now(),
                "error": str(e)
            })

    def get_job_status_from_file(self, job_id: str) -> Optional[JobStatus]:
        job = self.jobs.get(job_id)
        if not job: 
            return None
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "results", job_id)
        meta_path = os.path.join(base_dir, "metadata.json")
        log_path = os.path.join(base_dir, "log.txt")
        if not os.path.exists(meta_path):
            return JobStatus(
                job_id=job_id,
                status=job["status"],
                progress=job.get("progress"),
                message=job.get("message"),
                created_at=job["created_at"],
                completed_at=job.get("completed_at"),
                error=job.get("error"),
                episode_name=job.get("episode_name"),
                task=job.get("task"),
                url=job.get("url"),
                steps=job.get("steps"),
                expected_behavior=job.get("expected_behavior"),
                combinations=[],
                logs=_tail_lines(log_path, 50)
            )
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                shared = json.load(f)
            combos_raw = shared.get("combinations", [])
            combinations = []
            for c in combos_raw:
                try:
                    combinations.append(CombinationResult(**c))
                except Exception:
                    # minimal fallback conversion
                    combinations.append(CombinationResult(
                        goal=c.get("goal",""),
                        eps_name=job.get("episode_name",""),
                        task=TaskStep(steps=c.get("task",{}).get("steps",[])),
                        start_url=shared.get("url",""),
                        browser_context=BrowserContext(os="unknown", viewport=c.get("device",""), cookies_enabled=True),
                        success=c.get("success", False),
                        total_steps=c.get("total_steps",0),
                        runtime_sec=c.get("runtime_sec",0.0),
                        total_tokens=c.get("total_tokens",0),
                        gpt_output=c.get("gpt_output"),
                        wrong_behavior=c.get("wrong_behavior", False),
                        explanation=c.get("explanation"),
                        expected_behavior=c.get("expected_behavior"),
                        device=c.get("device",""),
                        browser=c.get("browser","")
                    ))
            total = len(job.get("devices", [])) * len(job.get("browsers", [])) if job.get("devices") else job.get("total_combinations", 0)
            comp = len(combinations)
            progress = job.get("progress")
            message = job.get("message")
            if job["status"] == "running" and comp > 0 and total:
                progress = int((comp / total) * 90) + 10
                message = f"Completed {comp}/{total} combinations"
            return JobStatus(
                job_id=job_id,
                status=job["status"],
                progress=progress,
                message=message,
                created_at=job["created_at"],
                completed_at=job.get("completed_at"),
                error=job.get("error"),
                episode_name=job.get("episode_name"),
                task=job.get("task"),
                url=job.get("url"),
                steps=job.get("steps"),
                expected_behavior=job.get("expected_behavior"),
                combinations=combinations,
                logs=_tail_lines(log_path, 50)
            )
        except Exception:
            return JobStatus(
                job_id=job_id,
                status=job["status"],
                progress=job.get("progress"),
                message=job.get("message"),
                created_at=job["created_at"],
                completed_at=job.get("completed_at"),
                error=job.get("error"),
                episode_name=job.get("episode_name"),
                task=job.get("task"),
                url=job.get("url"),
                steps=job.get("steps"),
                expected_behavior=job.get("expected_behavior"),
                combinations=[],
                logs=_tail_lines(log_path, 50)
            )

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.jobs.get(job_id)

    def get_job_result(self, job_id: str) -> Optional[PipelineResponse]:
        job = self.jobs.get(job_id)
        if job and job["status"] == "completed":
            return job["result"]
        return None

    def list_jobs(self) -> List[JobStatus]:
        out: List[JobStatus] = []
        for job_id in list(self.jobs.keys()):
            js = self.get_job_status_from_file(job_id)
            if js:
                out.append(js)
        return out

def _tail_lines(path: str, n: int) -> Optional[List[str]]:
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return [ln.rstrip("\n") for ln in lines[-n:]]
    except Exception:
        return None

    def delete_job(self, job_id: str) -> bool:
        if job_id in self.jobs:
            del self.jobs[job_id]; return True
        return False

pipeline_service = PipelineService()


