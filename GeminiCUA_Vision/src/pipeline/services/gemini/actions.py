from typing import Dict, Any, Optional
from playwright.sync_api import Page
import platform

SCREEN_WIDTH = 1440
SCREEN_HEIGHT = 900

def denormalize_coordinate(coord: int, screen_size: int) -> int:
    return int(coord / 1000 * screen_size)

def execute_gemini_action(page: Page, action_name: str, action_args: Dict[str, Any],
                          screen_width: int = SCREEN_WIDTH, screen_height: int = SCREEN_HEIGHT,
                          credentials: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    try:
        if action_name == "open_web_browser":
            return {"success": True}
        elif action_name == "wait_5_seconds":
            page.wait_for_timeout(5000); return {"success": True}
        elif action_name == "go_back":
            page.go_back(); page.wait_for_load_state("networkidle", timeout=5000); return {"success": True}
        elif action_name == "go_forward":
            page.go_forward(); page.wait_for_load_state("networkidle", timeout=5000); return {"success": True}
        elif action_name == "search":
            page.goto("https://www.google.com", wait_until="networkidle"); return {"success": True}
        elif action_name == "navigate":
            url = action_args.get("url", ""); 
            if url: page.goto(url, wait_until="networkidle", timeout=30000); return {"success": True}
            return {"success": False, "error": "No URL provided"}
        elif action_name == "click_at":
            x = denormalize_coordinate(action_args.get("x", 0), screen_width)
            y = denormalize_coordinate(action_args.get("y", 0), screen_height)
            page.mouse.click(x, y)
            page.wait_for_timeout(800)
            try: page.wait_for_load_state("networkidle", timeout=3000)
            except: page.wait_for_timeout(500)
            return {"success": True}
        elif action_name == "hover_at":
            x = denormalize_coordinate(action_args.get("x", 0), screen_width)
            y = denormalize_coordinate(action_args.get("y", 0), screen_height)
            page.mouse.move(x, y); return {"success": True}
        elif action_name == "type_text_at":
            x = denormalize_coordinate(action_args.get("x", 0), screen_width)
            y = denormalize_coordinate(action_args.get("y", 0), screen_height)
            text = action_args.get("text", ""); press_enter = action_args.get("press_enter", True)
            clear = action_args.get("clear_before_typing", True)
            page.mouse.click(x, y); page.wait_for_timeout(300)
            if clear:
                is_mac = platform.system() == "Darwin"
                select_all_key = "Meta+A" if is_mac else "Control+A"
                page.keyboard.press(select_all_key); page.keyboard.press("Backspace"); page.wait_for_timeout(100)
            page.keyboard.type(text, delay=50); page.wait_for_timeout(200)
            if press_enter:
                page.keyboard.press("Enter")
                try: page.wait_for_load_state("networkidle", timeout=5000)
                except: page.wait_for_timeout(1000)
            else:
                page.wait_for_timeout(500)
            return {"success": True}
        elif action_name == "key_combination":
            keys = action_args.get("keys", "")
            if not keys: return {"success": True}
            mapping = {"enter":"Enter","escape":"Escape","tab":"Tab","backspace":"Backspace","delete":"Delete",
                       "left":"ArrowLeft","right":"ArrowRight","up":"ArrowUp","down":"ArrowDown"}
            mod_map = {"ctrl":"Control","cmd":"Meta","command":"Meta","option":"Alt"}
            def norm(k: str) -> str:
                l = k.strip().lower()
                if l in mod_map: return mod_map[l]
                if l in mapping: return mapping[l]
                if l in ["control","meta","shift","alt"]: return l.capitalize()
                return k.capitalize()
            combo = "+".join([norm(p) for p in keys.split("+")])
            page.keyboard.press(combo)
            try: page.wait_for_load_state("networkidle", timeout=1000)
            except: page.wait_for_timeout(500)
            return {"success": True}
        elif action_name == "scroll_document":
            direction = action_args.get("direction", "down")
            if direction == "down": page.keyboard.press("PageDown")
            elif direction == "up": page.keyboard.press("PageUp")
            elif direction == "left": page.keyboard.press("ArrowLeft")
            elif direction == "right": page.keyboard.press("ArrowRight")
            page.wait_for_timeout(500); return {"success": True}
        elif action_name == "scroll_at":
            x = denormalize_coordinate(action_args.get("x", 0), screen_width)
            y = denormalize_coordinate(action_args.get("y", 0), screen_height)
            direction = action_args.get("direction", "down")
            magnitude = action_args.get("magnitude", 800)
            page.mouse.move(x, y)
            if direction == "down": page.mouse.wheel(0, magnitude)
            elif direction == "up": page.mouse.wheel(0, -magnitude)
            elif direction == "right": page.mouse.wheel(magnitude, 0)
            elif direction == "left": page.mouse.wheel(-magnitude, 0)
            page.wait_for_timeout(500); return {"success": True}
        elif action_name == "drag_and_drop":
            x = denormalize_coordinate(action_args.get("x", 0), screen_width)
            y = denormalize_coordinate(action_args.get("y", 0), screen_height)
            dx = denormalize_coordinate(action_args.get("destination_x", 0), screen_width)
            dy = denormalize_coordinate(action_args.get("destination_y", 0), screen_height)
            page.mouse.move(x, y); page.mouse.down(); page.mouse.move(dx, dy); page.mouse.up(); page.wait_for_timeout(500)
            return {"success": True}
        elif action_name == "fill_sensitive_field":
            # Secure credential handling - credentials never appear in Gemini's messages
            field_type = action_args.get("field_type", "")
            x = denormalize_coordinate(action_args.get("x", 0), screen_width)
            y = denormalize_coordinate(action_args.get("y", 0), screen_height)
            
            if not credentials:
                return {"success": False, "error": "No credentials provided"}
            
            # Get the appropriate credential value
            if field_type == "username":
                value = credentials.get("username", "")
            elif field_type == "password":
                value = credentials.get("password", "")
            else:
                return {"success": False, "error": f"Unknown field_type: {field_type}"}
            
            if not value:
                return {"success": False, "error": f"No {field_type} credential available"}
            
            # Click the field first to focus it
            page.mouse.click(x, y)
            page.wait_for_timeout(300)
            
            # Clear the field
            is_mac = platform.system() == "Darwin"
            select_all_key = "Meta+A" if is_mac else "Control+A"
            page.keyboard.press(select_all_key)
            page.keyboard.press("Backspace")
            page.wait_for_timeout(100)
            
            # Type the credential securely (never logged to Gemini)
            page.keyboard.type(value, delay=50)
            page.wait_for_timeout(200)
            
            return {"success": True}
        else:
            return {"success": False, "error": f"Unknown action: {action_name}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


