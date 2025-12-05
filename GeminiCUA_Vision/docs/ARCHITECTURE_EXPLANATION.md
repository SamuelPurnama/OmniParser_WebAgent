# Architecture Explanation: evaluate_grounding.py vs Pipeline Agent

## Quick Answer

**They are SEPARATE implementations** - `evaluate_grounding.py` does NOT use the pipeline agent. It has its own direct Gemini API call implementation.

---

## The Two Different Implementations

### 1. `evaluate_grounding.py` - Static Image Evaluation

**Purpose**: Evaluate Gemini's ability to locate UI elements in static screenshots

**How it works**:
```
Static Screenshot (from dataset)
    ↓
run_gemini_grounding() function (in evaluate_grounding.py)
    ↓
Direct Gemini API call (genai.Client)
    ↓
Returns: (x, y) coordinates
    ↓
Compare with ground truth bbox
```

**Key characteristics**:
- ✅ **Static images** - Works with pre-captured screenshots
- ✅ **Single API call** - One image → one prediction
- ✅ **No browser** - Doesn't control a browser
- ✅ **Evaluation focused** - Designed for benchmarking
- ✅ **Simple function**: `run_gemini_grounding(image, query)`

**Location**: `evaluate_grounding.py` → `run_gemini_grounding()` function (lines 125-223)

---

### 2. Pipeline Agent (`pipeline/` folder) - Interactive Browser Automation

**Purpose**: Control a real browser to perform multi-step tasks

**How it works**:
```
Browser (Playwright) launched
    ↓
GeminiAgent class (pipeline/services/gemini/agent.py)
    ↓
Multi-turn loop:
    - Take screenshot
    - Send to Gemini API
    - Get action back
    - Execute action in browser
    - Repeat until task complete
    ↓
Saves trajectory with all steps
```

**Key characteristics**:
- ✅ **Live browser** - Controls real browser via Playwright
- ✅ **Multi-step** - Can perform complex multi-action tasks
- ✅ **Interactive** - Sees page changes, adapts dynamically
- ✅ **Trajectory tracking** - Records all actions and screenshots
- ✅ **Full agent class**: `GeminiAgent` with `execute()` method

**Location**: `pipeline/services/gemini/agent.py` → `GeminiAgent` class

---

## Code Comparison

### `evaluate_grounding.py` Implementation

```python
def run_gemini_grounding(image: Image.Image, query: str, api_key: Optional[str] = None):
    """Run Gemini CUA on a static image to ground a text query."""
    client = genai.Client(api_key=api_key)
    model_name = "gemini-2.5-computer-use-preview-10-2025"
    
    # Convert image to bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    img_data = img_bytes.getvalue()
    
    instruction = f"Click on the UI element described by: '{query}'"
    
    # Single API call
    resp = client.models.generate_content(model=model_name, contents=contents, config=config)
    
    # Extract coordinates from response
    # Return (x, y) or None
```

**Used by**:
- `evaluate_grounding.py` - Main evaluation script

---

### Pipeline Agent Implementation

```python
class GeminiAgent:
    def __init__(self, api_key, model_name, max_turns=100, ...):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        # ... setup ...
    
    def execute(self, page: Page, instruction: str, ...):
        """Execute multi-step task in browser."""
        for turn in range(self.max_turns):
            # 1. Take screenshot
            screenshot = page.screenshot()
            
            # 2. Call Gemini API
            resp = self.client.models.generate_content(...)
            
            # 3. Execute action in browser
            if action == "click_at":
                page.click(x, y)
            # ... other actions ...
            
            # 4. Check if done, repeat
```

**Used by**:
- `pipeline/services/pipeline_runner.py` - For running interactive tasks
- `run_gemini_pipeline.py` - CLI for custom tasks
- `test_pipeline_with_ratios.py` - Testing different screen sizes

---

## Why Two Separate Implementations?

### Different Use Cases

1. **`evaluate_grounding.py`**:
   - 📊 Benchmarking/Evaluation
   - 🖼️ Static screenshots from datasets
   - ⚡ Fast, batch processing
   - 📈 Measuring accuracy on known ground truth

2. **Pipeline Agent**:
   - 🤖 Real-world automation
   - 🌐 Live websites/applications
   - 🔄 Multi-step tasks
   - 💼 Actual browser control

### Different Requirements

| Feature | evaluate_grounding.py | Pipeline Agent |
|---------|----------------------|----------------|
| Browser needed? | ❌ No | ✅ Yes (Playwright) |
| Image source | Static files | Live screenshots |
| API calls | 1 per sample | Multiple per task |
| Action execution | ❌ None | ✅ Clicks, types, etc. |
| Complexity | Simple function | Full agent class |

---

## Can They Share Code?

**Currently**: No, they're completely separate

**Why**: 
- Different goals (evaluation vs automation)
- Different input types (static image vs live page)
- Different output needs (coordinates vs executed actions)

**Potential optimization**: 
They could share the Gemini API client initialization and configuration, but the actual execution logic needs to remain separate.

---

## Which One Should You Use?

### Use `evaluate_grounding.py` when:
- ✅ Evaluating Gemini's grounding accuracy
- ✅ Running benchmarks on datasets
- ✅ Comparing performance across datasets
- ✅ Analyzing failure patterns
- ✅ You have static screenshots

### Use Pipeline Agent when:
- ✅ Automating real browser tasks
- ✅ Testing on live websites
- ✅ Performing multi-step workflows
- ✅ Need to see actual browser interaction
- ✅ You have a URL/task to perform

---

## Summary

**`evaluate_grounding.py`**:
- ❌ Does NOT call the pipeline agent
- ✅ Has its own `run_gemini_grounding()` function
- ✅ Directly calls Gemini API for static images
- ✅ Used for dataset evaluation

**Pipeline Agent**:
- ✅ Separate `GeminiAgent` class
- ✅ Controls live browser via Playwright
- ✅ Used for interactive automation
- ✅ Multi-step task execution

They're **complementary but separate** tools for different purposes!


