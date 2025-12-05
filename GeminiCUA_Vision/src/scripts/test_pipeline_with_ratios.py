#!/usr/bin/env python3
"""
Simple test script for Gemini pipeline with screen aspect ratio testing

Tests a basic task to verify the pipeline works across different screen ratios
"""
import os
import sys
import argparse
from pathlib import Path

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Try to load .env file if it exists
def load_env_file():
    """Load environment variables from .env file"""
    env_path = os.path.join(PROJECT_ROOT, '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

load_env_file()

from src.pipeline.services.models import InstructionRequest
from src.pipeline.services.pipeline_runner import pipeline_runner

# Screen ratio configurations
SCREEN_RATIOS = {
    "default": {"width": 1440, "height": 900, "name": "Default (1440x900)"},
    "standard": {"width": 1920, "height": 1080, "name": "Standard 16:9 (1920x1080)"},
    "ultrawide": {"width": 3840, "height": 1080, "name": "Ultra-wide 32:9 (3840x1080)"},
    "wide": {"width": 2560, "height": 1080, "name": "Wide 21:9 (2560x1080)"},
    "superwide": {"width": 5120, "height": 1440, "name": "Super-wide 32:9 (5120x1440)"},
}


def test_pipeline_with_ratio(screen_ratio_key: str = "default", output_dir: str = None):
    """
    Test the pipeline with a specific screen aspect ratio
    
    Args:
        screen_ratio_key: Key from SCREEN_RATIOS dict
        output_dir: Optional output directory for results
    
    Returns:
        bool: True if test passed, False otherwise
    """
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ ERROR: GOOGLE_API_KEY environment variable not set!")
        print("   Please set it in .env file or export it:")
        print("   export GOOGLE_API_KEY='your-key'")
        return False
    
    # Get screen configuration
    if screen_ratio_key not in SCREEN_RATIOS:
        print(f"❌ ERROR: Unknown screen ratio '{screen_ratio_key}'")
        print(f"   Available ratios: {', '.join(SCREEN_RATIOS.keys())}")
        return False
    
    screen_config = SCREEN_RATIOS[screen_ratio_key]
    width = screen_config["width"]
    height = screen_config["height"]
    name = screen_config["name"]
    
    print("🧪 Testing Gemini Pipeline...")
    print("=" * 80)
    print(f"📐 Screen Ratio: {name}")
    print(f"   Resolution: {width}x{height}")
    print("=" * 80)
    
    # Create a test instruction for Google Flights
    instruction = InstructionRequest(
        url="https://www.google.com/travel/flights",
        task="Find a flight from Seattle to New York on December 23",
        steps="1. Navigate to Google Flights\n2. Enter Seattle as the origin city\n3. Enter New York as the destination city\n4. Select December 23 as the departure date\n5. Click the Search button",
        expected_behavior="Flight search results page should appear showing available flights from Seattle to New York on December 23",
        devices=["desktop"],
        browsers=["chrome"],
        screen_width=width,
        screen_height=height
    )
    
    print(f"📋 Test Instruction:")
    print(f"   URL: {instruction.url}")
    print(f"   Task: {instruction.task}")
    print(f"   Devices: {instruction.devices}")
    print(f"   Browsers: {instruction.browsers}")
    print(f"   Screen: {width}x{height}")
    print()
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "data", "results", "test_run")
    
    episode_name = f"test_run_{screen_ratio_key}_{width}x{height}"
    base_episode_dir = os.path.join(output_dir, episode_name)
    
    try:
        # Run the pipeline
        print("🚀 Starting pipeline execution...")
        result = pipeline_runner.run_pipeline(
            instruction=instruction,
            episode_name=episode_name,
            base_episode_dir=base_episode_dir
        )
        
        print("\n" + "=" * 80)
        print("📊 Pipeline Results:")
        print(f"   Total combinations: {result.total_combinations}")
        print(f"   Successful: {result.successful_combinations}")
        print(f"   Failed: {result.failed_combinations}")
        print(f"   Total runtime: {result.total_runtime_sec:.2f}s")
        print()
        
        # Show combination details
        for i, combo in enumerate(result.combinations, 1):
            print(f"   Combination {i}: {combo.device}_{combo.browser}")
            print(f"      Success: {combo.success}")
            print(f"      Steps: {combo.total_steps}")
            print(f"      Runtime: {combo.runtime_sec:.2f}s")
            if combo.error_message:
                print(f"      Error: {combo.error_message}")
            print()
        
        if result.successful_combinations > 0:
            print(f"✅ Test PASSED - Pipeline executed successfully on {name}!")
            return True
        else:
            print(f"❌ Test FAILED - No successful combinations on {name}")
            return False
            
    except Exception as e:
        print(f"\n❌ Test FAILED with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_ratios(ratios: list = None, output_dir: str = None):
    """
    Test the pipeline across multiple screen aspect ratios
    
    Args:
        ratios: List of ratio keys to test (default: all)
        output_dir: Optional output directory for results
    
    Returns:
        dict: Results for each ratio
    """
    if ratios is None:
        ratios = list(SCREEN_RATIOS.keys())
    
    print("🧪 Testing Gemini Pipeline Across Multiple Screen Ratios")
    print("=" * 80)
    print(f"📐 Testing {len(ratios)} screen ratios: {', '.join(ratios)}")
    print("=" * 80)
    print()
    
    results = {}
    for ratio_key in ratios:
        print(f"\n{'─' * 80}")
        print(f"Testing: {SCREEN_RATIOS[ratio_key]['name']}")
        print(f"{'─' * 80}\n")
        
        success = test_pipeline_with_ratio(ratio_key, output_dir)
        results[ratio_key] = {
            "success": success,
            "config": SCREEN_RATIOS[ratio_key]
        }
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY - All Screen Ratios")
    print("=" * 80)
    for ratio_key, result in results.items():
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        config = result["config"]
        print(f"{status} | {config['name']:30s} | {config['width']}x{config['height']}")
    print("=" * 80)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test Gemini pipeline with different screen aspect ratios")
    parser.add_argument("--ratio", type=str, default="default",
                       choices=list(SCREEN_RATIOS.keys()),
                       help="Screen ratio to test (default: default)")
    parser.add_argument("--all", action="store_true",
                       help="Test all screen ratios")
    parser.add_argument("--ratios", type=str, nargs="+",
                       choices=list(SCREEN_RATIOS.keys()),
                       help="Specific screen ratios to test")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for test results")
    
    args = parser.parse_args()
    
    if args.all:
        # Test all ratios
        results = test_multiple_ratios(output_dir=args.output_dir)
        success = all(r["success"] for r in results.values())
    elif args.ratios:
        # Test specific ratios
        results = test_multiple_ratios(ratios=args.ratios, output_dir=args.output_dir)
        success = all(r["success"] for r in results.values())
    else:
        # Test single ratio
        success = test_pipeline_with_ratio(args.ratio, args.output_dir)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

