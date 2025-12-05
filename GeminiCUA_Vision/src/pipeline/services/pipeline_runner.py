from typing import Dict, List, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import re
import threading
import base64

from .models import InstructionRequest, PipelineResponse, CombinationResult, TaskStep, BrowserContext
from .gemini.agent import GeminiAgent
from .gemini.actions import SCREEN_WIDTH, SCREEN_HEIGHT
from playwright.sync_api import sync_playwright

def _decode_secret(value: Optional[str]) -> str:
    if not value:
        return ""
    try:
        return base64.b64decode(value).decode("utf-8")
    except Exception:
        # If value is not base64, return it as-is
        return value


def _run_gemini_trajectory(url: str, task: str, steps: str, expected_behavior: str,
                           device: str, browser_type: str, base_eps_name: Optional[str],
                           idx: int, total: int, shared_metadata_path: Optional[str],
                           metadata_lock: Optional[threading.Lock], base_dir_override: Optional[str],
                           credentials: Optional[Dict[str, str]] = None,
                           screen_width: Optional[int] = None, screen_height: Optional[int] = None) -> Optional[Dict[str, Any]]:
    import time, json
    start_time = time.time()
    if device == "mobile":
        viewport = {"width": 375, "height": 667}
    else:
        # Use custom screen dimensions if provided, otherwise use defaults
        viewport = {
            "width": screen_width if screen_width else SCREEN_WIDTH,
            "height": screen_height if screen_height else SCREEN_HEIGHT
        }
    # Remove credentials from steps text if they were embedded (for backward compatibility)
    sanitized_steps = steps
    if credentials:
        # Remove any embedded credential patterns from steps
        username = credentials.get("username", "")
        password = credentials.get("password", "")
        if username:
            sanitized_steps = re.sub(re.escape(username), "[USERNAME]", sanitized_steps, flags=re.IGNORECASE)
        if password:
            sanitized_steps = re.sub(re.escape(password), "[PASSWORD]", sanitized_steps, flags=re.IGNORECASE)
        # Replace login instructions with generic ones that reference fill_sensitive_field
        sanitized_steps = re.sub(
            r"login.*?with.*?email.*?password",
            "login using the fill_sensitive_field function for username and password fields",
            sanitized_steps,
            flags=re.IGNORECASE
        )
    
    full_instruction = (
        f"{task}\n\n"
        f"Steps to follow:\n{sanitized_steps}\n\n"
        f"EXPECTED BEHAVIOR (MUST BE VERIFIED): {expected_behavior}\n\n"
        f"IMPORTANT: After completing all the steps above, you MUST verify if the expected behavior is met.\n"
        f"- If the steps completed successfully BUT the expected behavior is NOT met: STOP immediately and report that "
        f"the app does not meet the expected behavior (this indicates a bug or issue with the website).\n"
        f"- If you cannot complete the steps themselves (e.g., button not clickable, form won't submit): "
        f"try different approaches to complete the steps, but once steps are done, verify expected behavior and stop if not met.\n"
        f"- Only if the expected behavior IS met after completing steps: report success."
    )
    combo_dir = os.path.join(base_dir_override, f"{device}_{browser_type}") if base_dir_override else None
    screenshots_dir = os.path.join(combo_dir, "screenshots") if combo_dir else None
    if screenshots_dir: os.makedirs(screenshots_dir, exist_ok=True)

    with sync_playwright() as p:
        # Launch browser in headless mode for faster execution
        if browser_type in ["chrome","chromium"]:
            browser = p.chromium.launch(headless=True)
        elif browser_type == "firefox":
            browser = p.firefox.launch(headless=True)
        elif browser_type in ["safari","webkit"]:
            browser = p.webkit.launch(headless=True)
        else:
            browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport=viewport)
        print(f"Browser viewport set to: {viewport['width']}x{viewport['height']}")
        try:
            page = context.new_page()
            try: page.goto(url, wait_until="networkidle", timeout=30000)
            except Exception: pass
            agent = GeminiAgent(max_turns=100, wait_between_actions=200)
            result = agent.execute(
                page=page, 
                instruction=full_instruction, 
                initial_url=None, 
                logger=None, 
                credentials=credentials,
                screen_width=viewport["width"],
                screen_height=viewport["height"]
            )
            runtime = time.time() - start_time
            saved = []
            if screenshots_dir and result and result.get("screenshots"):
                for sc in result["screenshots"]:
                    step_num = sc.get("step", 0)
                    data = sc.get("marked_bytes") or sc.get("bytes")
                    if not data: continue
                    path = os.path.join(screenshots_dir, f"screenshot_{step_num:03d}.png")
                    with open(path, "wb") as f: f.write(data)
                    saved.append(path)
            # Store detailed trajectory with all Gemini API responses
            trajectory = {}
            detailed_trajectory = []  # Store full raw trajectory data
            for i, step in enumerate(result.get("trajectory", [])):
                action_name = step.get("action", "unknown")
                thought = (step.get("thought") or "").strip()
                step_success = bool(step.get("success", False))
                step_args = step.get("args", {})
                step_error = step.get("error")
                step_timestamp = step.get("timestamp")
                
                # Store detailed step information
                detailed_step = {
                    "step": step.get("step", i+1),
                    "action": action_name,
                    "args": step_args,
                    "success": step_success,
                    "error": step_error,
                    "timestamp": step_timestamp,
                    "thought": thought,
                }
                detailed_trajectory.append(detailed_step)
                
                # Also store in simplified format for backward compatibility
                trajectory[str(i+1)] = {
                    "action": {
                        "action_str": f"{action_name}",
                        "action_description": thought or f"Executed {action_name}",
                        "args": step_args,
                    },
                    "status": "passed" if step_success else "failed",
                    "error": step_error,
                    "timestamp": step_timestamp,
                }
            success = result.get("success", False) and result.get("completed", False)
            final_message = result.get("final_message", "")
            # Parse structured field from Gemini's final message
            behavior_met = None
            if "EXPECTED_BEHAVIOR_MET:" in final_message:
                match = re.search(r"EXPECTED_BEHAVIOR_MET:\s*(true|false)", final_message, re.IGNORECASE)
                if match:
                    behavior_met = match.group(1).lower() == "true"
            # If structured field not found, fall back to checking if task completed successfully
            # (assume behavior was met if task succeeded, unless explicitly stated otherwise)
            if behavior_met is None:
                behavior_met = success  # Default: if task succeeded, assume behavior was met
            # wrong_behavior is true if: expected behavior not met OR task failed
            wrong_behavior = not behavior_met or not success
            # success should be false if expected behavior wasn't met, even if steps completed
            actual_success = success and behavior_met
            if final_message:
                idx = len(trajectory)
                trajectory[str(idx+1)] = {
                    "action": {
                        "action_str": "final_state",
                        "action_description": final_message.strip() or "Final state reported"
                    },
                    "status": "passed" if actual_success else "failed"
                }
            episode_name = base_eps_name or f"{task.lower().replace(' ', '_')}"
            
            # Prepare comprehensive metadata with all Gemini API responses
            metadata = {
                "episode_name": episode_name,
                "url": url,
                "task": task,
                "steps": steps,
                "expected_behavior": expected_behavior,
                "device": device,
                "browser_type": browser_type,
                "success": actual_success,
                "error_message": result.get("error") if not actual_success else None,
                "runtime_sec": runtime,
                "total_steps": len(trajectory),
                "screenshots": saved,
                "trajectory": trajectory,  # Simplified format for backward compatibility
                "detailed_trajectory": detailed_trajectory,  # Full detailed trajectory with all Gemini thoughts
                "wrong_behavior": wrong_behavior,
                "explanation": result.get("final_message", ""),
                "gpt_output": result.get("final_message", ""),
                "gemini_responses": {
                    "final_message": result.get("final_message", ""),
                    "completed": result.get("completed", False),
                    "raw_trajectory": result.get("trajectory", []),  # Raw trajectory from agent
                    "total_runtime_sec": result.get("runtime_sec", 0),
                }
            }
            
            # Save detailed metadata to combination directory if base_dir is provided
            if base_dir_override:
                combo_dir = os.path.join(base_dir_override, f"{device}_{browser_type}")
                os.makedirs(combo_dir, exist_ok=True)
                metadata_path = os.path.join(combo_dir, "metadata.json")
                try:
                    import json
                    with open(metadata_path, "w", encoding="utf-8") as f:
                        json.dump(metadata, f, indent=2, default=str)
                except Exception as e:
                    print(f"Warning: Could not save detailed metadata: {e}")
            
            return metadata
        finally:
            page.close(); context.close(); browser.close()

class PipelineRunner:
    def run_pipeline(self, instruction: InstructionRequest, episode_name: Optional[str] = None,
                     base_episode_dir: Optional[str] = None) -> PipelineResponse:
        if base_episode_dir:
            os.makedirs(base_episode_dir, exist_ok=True)
            meta_path = os.path.join(base_episode_dir, "metadata.json")
            init = {"episode_name": episode_name, "url": instruction.url, "task": instruction.task,
                    "steps": instruction.steps, "expected_behavior": instruction.expected_behavior, "created_at": datetime.now().isoformat(),
                    "combinations": []}
            with open(meta_path, "w", encoding="utf-8") as f: import json; json.dump(init, f, indent=2)
        combos = [{"device": d, "browser": b} for d in instruction.devices for b in instruction.browsers]
        results: List[CombinationResult] = []
        metadata_lock = threading.Lock()
        # Extract credentials securely (never logged to Gemini)
        credentials = None
        if instruction.username or instruction.password:
            credentials = {
                "username": _decode_secret(instruction.username),
                "password": _decode_secret(instruction.password)
            }
        with ThreadPoolExecutor(max_workers=min(len(combos), 4)) as ex:
            futs = {
                ex.submit(_run_gemini_trajectory, instruction.url, instruction.task, instruction.steps,
                          instruction.expected_behavior, c["device"], c["browser"], episode_name, idx, len(combos),
                          os.path.join(base_episode_dir, "metadata.json") if base_episode_dir else None,
                          metadata_lock, base_episode_dir, credentials,
                          instruction.screen_width, instruction.screen_height): c
                for idx, c in enumerate(combos)
            }
            for fut, c in futs.items():
                try:
                    meta = fut.result()
                    if not meta: continue
                    step_list = []
                    step_statuses: List[bool] = []
                    traj = meta.get("trajectory", {})
                    for k in sorted(traj.keys(), key=lambda x: int(x) if x.isdigit() else 0):
                        a = traj[k].get("action", {})
                        step_list.append(a.get("action_description") or a.get("action_str","Unknown action"))
                        status = traj[k].get("status")
                        step_statuses.append(status != "failed")
                    results.append(CombinationResult(
                        goal=meta.get("task", instruction.task),
                        eps_name=meta.get("episode_name",""),
                        task=TaskStep(steps=step_list, statuses=step_statuses),
                        start_url=meta.get("url", instruction.url),
                        browser_context=BrowserContext(os="unknown", viewport=c["device"], cookies_enabled=True),
                        success=meta.get("success", False),
                        total_steps=meta.get("total_steps", 0),
                        runtime_sec=meta.get("runtime_sec", 0),
                        total_tokens=0,
                        gpt_output=meta.get("gpt_output"),
                        wrong_behavior=meta.get("wrong_behavior", False),
                        explanation=meta.get("explanation"),
                        expected_behavior=meta.get("expected_behavior", instruction.expected_behavior),
                        error_message=meta.get("error_message"),
                        device=c["device"], browser=c["browser"]
                    ))
                except Exception:
                    results.append(CombinationResult(
                        goal=instruction.task, eps_name="", task=TaskStep(steps=[]),
                        start_url=instruction.url, browser_context=BrowserContext(os="unknown", viewport=c["device"], cookies_enabled=True),
                        success=False, total_steps=0, runtime_sec=0, total_tokens=0, device=c["device"], browser=c["browser"],
                        expected_behavior=instruction.expected_behavior, error_message="runner error"
                    ))
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        total_runtime = sum(r.runtime_sec for r in results)
        return PipelineResponse(
            episode_name=episode_name or instruction.task.lower().replace(" ","_"),
            task=instruction.task, url=instruction.url, steps=instruction.steps, expected_behavior=instruction.expected_behavior,
            combinations=results, total_combinations=len(results), successful_combinations=successful, failed_combinations=failed,
            total_runtime_sec=total_runtime, created_at=datetime.now()
        )

pipeline_runner = PipelineRunner()


