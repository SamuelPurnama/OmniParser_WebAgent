"""Gemini Computer Use Agent Implementation (migrated into Django app)"""
from typing import Dict, Any, Optional, List
import io
import time
import os

from google import genai
from google.genai import types
from google.genai.types import Content, Part
from playwright.sync_api import Page

from .actions import SCREEN_WIDTH, SCREEN_HEIGHT, denormalize_coordinate

try:
    from PIL import Image, ImageDraw, ImageFont  # type: ignore
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


def _mark_coordinate_on_screenshot(screenshot_bytes: bytes, x: int, y: int, action_name: str = "click",
                                   screen_width: int = SCREEN_WIDTH, screen_height: int = SCREEN_HEIGHT,
                                   dest_coords: Optional[tuple] = None) -> bytes:
    if not PIL_AVAILABLE:
        return screenshot_bytes
    try:
        img = Image.open(io.BytesIO(screenshot_bytes))
        draw = ImageDraw.Draw(img)
        def denorm(c: int, size: int) -> int:
            return denormalize_coordinate(c, size) if c <= 999 else c
        ax, ay = denorm(x, screen_width), denorm(y, screen_height)
        color = {"click":"red","click_at":"red","type":"blue","type_text_at":"blue","hover":"yellow","hover_at":"yellow","drag_and_drop":"purple"}.get(action_name.lower(), "red")
        r, ch = 10, 15
        draw.ellipse([ax-r, ay-r, ax+r, ay+r], fill=color, outline="white", width=2)
        draw.line([ax-ch, ay, ax+ch, ay], fill="white", width=2)
        draw.line([ax, ay-ch, ax, ay+ch], fill="white", width=2)
        if action_name == "drag_and_drop" and dest_coords:
            dx, dy = denorm(dest_coords[0], screen_width), denorm(dest_coords[1], screen_height)
            draw.ellipse([dx-r, dy-r, dx+r, dy+r], fill=None, outline=color, width=3)
            draw.line([ax, ay, dx, dy], fill=color, width=2)
        label = action_name
        label_x, label_y = ax + r + 5, ay - 10
        try:
            font = ImageFont.load_default()
            bbox = draw.textbbox((0,0), label, font=font)
            tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        except Exception:
            font, tw, th = None, len(label)*6, 10
        draw.rectangle([label_x-2, label_y-2, label_x+tw+2, label_y+th+2], fill=color, outline="white")
        draw.text((label_x, label_y), label, fill="white", font=font if font else None)
        out = io.BytesIO(); img.save(out, format="PNG"); return out.getvalue()
    except Exception:
        return screenshot_bytes


class GeminiAgent:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-computer-use-preview-10-2025",
                 max_turns: int = 100, wait_between_actions: int = 200, system_instruction: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name
        self.max_turns = max_turns
        self.wait_between_actions = wait_between_actions
        self.system_instruction = system_instruction or (
            "You are a helpful assistant that can use a web browser for testing.\n"
            "Do not ask follow up questions, the user will trust your judgement.\n"
            "Work in atomic steps. One navigation OR one specific action per step.\n"
            "CRITICAL: After completing all requested steps, you MUST verify if the expected behavior is met.\n"
            "- If steps completed successfully BUT expected behavior is NOT met: STOP and report that the app "
            "does not meet expected behavior (this indicates a bug/issue with the website).\n"
            "- If you cannot complete the steps themselves: try different approaches to complete them.\n"
            "- Once steps are done, verify expected behavior. If not met, stop and report the issue.\n"
            "- Only if expected behavior IS met: report success.\n"
            "Always check the current state of the page before planning your next action to ensure previous steps worked correctly.\n\n"
            "SECURITY: When you need to fill in username or password fields, use the fill_sensitive_field function instead of type_text_at.\n"
            "- First, click on the input field (username or password) to focus it.\n"
            "- Then call fill_sensitive_field with field_type set to 'username' or 'password'.\n"
            "- Do NOT include the actual credentials in your function calls or messages.\n\n"
            "IMPORTANT: When you finish (whether success or failure), your final message MUST include exactly one of these lines:\n"
            "- 'EXPECTED_BEHAVIOR_MET: true' if the expected behavior was achieved\n"
            "- 'EXPECTED_BEHAVIOR_MET: false' if the expected behavior was NOT achieved\n"
            "This allows the system to properly categorize the test result."
        )
        
        # Define custom function for secure credential handling
        fill_sensitive_field_schema = types.FunctionDeclaration(
            name="fill_sensitive_field",
            description="Fill in a sensitive input field (username or password) securely. Use this instead of type_text_at when entering credentials.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "field_type": types.Schema(
                        type=types.Type.STRING,
                        description="The type of field to fill: 'username' or 'password'",
                        enum=["username", "password"]
                    ),
                    "x": types.Schema(
                        type=types.Type.NUMBER,
                        description="X coordinate of the input field (0-999 normalized)"
                    ),
                    "y": types.Schema(
                        type=types.Type.NUMBER,
                        description="Y coordinate of the input field (0-999 normalized)"
                    )
                },
                required=["field_type", "x", "y"]
            )
        )
        
        self.config = types.GenerateContentConfig(
            system_instruction=self.system_instruction,
            tools=[
                types.Tool(computer_use=types.ComputerUse(environment=types.Environment.ENVIRONMENT_BROWSER)),
                types.Tool(function_declarations=[fill_sensitive_field_schema])
            ],
        )

    def execute(self, page: Page, instruction: str, initial_url: Optional[str] = None,
                logger: Optional[callable] = None, screen_width: Optional[int] = None, 
                screen_height: Optional[int] = None, credentials: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        def log(msg: str, level: int = 1):
            if logger: logger({"category":"agent","message":msg,"level":level})
        start = time.time()
        trajectory: List[Dict[str, Any]] = []
        screenshots: List[Dict[str, Any]] = []
        
        # Get actual viewport dimensions from page, or use defaults
        viewport = page.viewport_size
        if screen_width and screen_height:
            actual_width, actual_height = screen_width, screen_height
        elif viewport:
            actual_width = viewport.get("width") or SCREEN_WIDTH
            actual_height = viewport.get("height") or SCREEN_HEIGHT
        else:
            actual_width, actual_height = SCREEN_WIDTH, SCREEN_HEIGHT

        def append_terminal_screenshot(message: str, action_name: str = "final_state"):
            """Capture one last screenshot paired with the provided message."""
            step_num = len(trajectory) + 1
            final_url = page.url or sent_url
            try:
                final_bytes = page.screenshot(type="png")
                final_url = page.url or final_url
            except Exception:
                final_bytes = sent
            screenshots.append({
                "step": step_num,
                "url": final_url,
                "timestamp": time.time(),
                "bytes": final_bytes,
                "marked_bytes": None,
                "action_name": action_name,
                "coordinates": None,
                "final_message": message
            })

        if initial_url:
            cur = page.url
            if not cur or cur == "about:blank" or cur != initial_url:
                try: page.goto(initial_url, wait_until="networkidle", timeout=30000)
                except Exception as e: log(f"Navigation fallback failed: {e}", 0)
        initial = page.screenshot(type="png")
        screenshots.append({"step":0,"url":page.url,"timestamp":time.time(),"bytes":initial})
        contents: List[Content] = [Content(role="user", parts=[Part(text=f"{instruction}\n\nCurrent URL: {page.url}"),
                                                               Part.from_bytes(data=initial, mime_type="image/png")])]
        sent = initial; sent_url = page.url

        from .actions import execute_gemini_action
        # Store credentials securely (not in logs/instruction)
        secure_creds = credentials or {}
        for turn in range(self.max_turns):
            try:
                resp = self.client.models.generate_content(model=self.model_name, contents=contents, config=self.config)
                cand = resp.candidates[0]; contents.append(cand.content)
                function_calls = []; text_parts = []
                for part in cand.content.parts:
                    if part.function_call: function_calls.append(part.function_call)
                    elif part.text: text_parts.append(part.text)
                thought_text = " ".join(text_parts).strip()
                if not function_calls:
                    final_text = " ".join(text_parts)
                    append_terminal_screenshot(final_text or "Final state reported")
                    return {"success": True, "completed": True, "final_message": final_text,
                            "runtime_sec": time.time()-start, "total_steps": len(trajectory),
                            "trajectory": trajectory, "screenshots": screenshots}
                # Mark coordinates on last-sent screenshot if applicable
                marked = sent; coords = None; action_name_for_mark = None
                for fc in function_calls:
                    name, args = fc.name, fc.args
                    if name in ["click_at","type_text_at","hover_at","drag_and_drop","fill_sensitive_field"]:
                        action_name_for_mark = name
                        if name == "drag_and_drop":
                            x,y,dx,dy = args.get("x"),args.get("y"),args.get("destination_x"),args.get("destination_y")
                            if x is not None and y is not None:
                                coords = (x,y); marked = _mark_coordinate_on_screenshot(sent,x,y,name,actual_width,actual_height,(dx,dy))
                        else:
                            x,y = args.get("x"),args.get("y")
                            if x is not None and y is not None:
                                coords = (x,y); marked = _mark_coordinate_on_screenshot(sent,x,y,name,actual_width,actual_height)
                        break
                step_num = len(trajectory)+1
                screenshots.append({"step":step_num,"url":sent_url,"timestamp":time.time(),
                                    "bytes":sent,"marked_bytes":marked,"action_name":action_name_for_mark,
                                    "coordinates":coords})

                # Execute actions
                fr_parts = []
                for fc in function_calls:
                    name, args = fc.name, fc.args
                    try:
                        # For sensitive fields, pass credentials securely
                        if name == "fill_sensitive_field":
                            result = execute_gemini_action(page, name, args, actual_width, actual_height, secure_creds)
                        else:
                            result = execute_gemini_action(page, name, args, actual_width, actual_height)
                    except Exception as e:
                        result = {"success": False, "error": str(e)}
                    page.wait_for_timeout(self.wait_between_actions)
                    # Redact sensitive data from trajectory logs
                    logged_args = args.copy()
                    if name == "fill_sensitive_field":
                        logged_args = {"field_type": args.get("field_type"), "x": args.get("x"), "y": args.get("y")}
                    trajectory.append({
                        "step": len(trajectory)+1,
                        "action": name,
                        "args": logged_args,
                        "success": result.get("success", False),
                        "error": result.get("error"),
                        "timestamp": time.time(),
                        "thought": thought_text,
                    })
                    try:
                        new_bytes = page.screenshot(type="png"); sent = new_bytes; sent_url = page.url
                        fr_parts.append(types.FunctionResponse(
                            name=name, response={"url": page.url, "success": result.get("success", True)},
                            parts=[types.FunctionResponsePart(inline_data=types.FunctionResponseBlob(mime_type="image/png", data=new_bytes))]
                        ))
                    except Exception:
                        fr_parts.append(types.FunctionResponse(name=name, response={"url": page.url, "success": result.get("success", True)}, parts=[]))
                if fr_parts:
                    contents.append(Content(role="user", parts=[Part(function_response=fr) for fr in fr_parts]))
            except Exception as e:
                append_terminal_screenshot(str(e) or "Agent error", action_name="error_state")
                return {"success": False, "completed": False, "error": str(e),
                        "runtime_sec": time.time()-start, "total_steps": len(trajectory),
                        "trajectory": trajectory, "screenshots": screenshots}
        append_terminal_screenshot(f"Max turns ({self.max_turns}) reached", action_name="max_turns")
        return {"success": False, "completed": False, "error": f"Max turns ({self.max_turns}) reached",
                "runtime_sec": time.time()-start, "total_steps": len(trajectory),
                "trajectory": trajectory, "screenshots": screenshots}


