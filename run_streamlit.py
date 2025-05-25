#!/usr/bin/env python3
"""
Simple script to run the Manchu OCR Streamlit application
"""

import subprocess
import sys
import os
import tempfile


def setup_cache_environment():
    """Setup proper cache directory and environment variables"""

    # Set a writable cache directory
    cache_dir = os.environ.get("HF_HOME")
    if not cache_dir:
        # Try to use a local cache directory first
        local_cache = os.path.join(os.getcwd(), ".hf_cache")
        try:
            os.makedirs(local_cache, exist_ok=True)
            # Test if we can write to it
            test_file = os.path.join(local_cache, "test_write")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            cache_dir = local_cache
            print(f"✅ Using local cache directory: {cache_dir}")
        except (OSError, PermissionError):
            # Fall back to temp directory
            cache_dir = os.path.join(tempfile.gettempdir(), "hf_cache")
            os.makedirs(cache_dir, exist_ok=True)
            print(f"⚠️  Using temporary cache directory: {cache_dir}")
    else:
        print(f"✅ Using existing HF_HOME: {cache_dir}")

    # Set environment variables
    os.environ["HF_HOME"] = cache_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir

    # Additional environment variables to prevent warnings and issues
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

    return cache_dir


def main():
    """Run the Streamlit application"""

    # Setup cache environment first
    cache_dir = setup_cache_environment()

    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("❌ Streamlit is not installed. Please install it first:")
        print("   pip install streamlit>=1.28.0")
        sys.exit(1)

    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    streamlit_app_path = os.path.join(script_dir, "streamlit_app.py")

    # Check if the streamlit app exists
    if not os.path.exists(streamlit_app_path):
        print(f"❌ Streamlit app not found at: {streamlit_app_path}")
        sys.exit(1)

    print("🚀 Starting Manchu OCR Streamlit application...")
    print("📱 The app will open in your default web browser")
    print("🔗 If it doesn't open automatically, go to: http://localhost:10011")
    print(f"📁 Cache directory: {cache_dir}")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 60)

    # Run streamlit
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                streamlit_app_path,
                "--server.address",
                "localhost",
                "--server.port",
                "10011",
                "--browser.gatherUsageStats",
                "false",
                "--server.fileWatcherType",
                "none",
                "--global.developmentMode",
                "false",
            ]
        )
    except KeyboardInterrupt:
        print("\n👋 Streamlit application stopped.")
    except Exception as e:
        print(f"❌ Error running Streamlit: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
