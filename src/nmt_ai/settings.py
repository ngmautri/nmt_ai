# app.py
import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the variables
GEMINI_API_KEY = os.getenv('DATABASE_URL')
secret_key = os.getenv('SECRET_KEY')
debug_mode = os.getenv('DEBUG', 'False').lower() == 'true'  # Handle boolean conversion

# print(f"GEMINI_API_KEY: {GEMINI_API_KEY}")
# print(f"Secret Key: {secret_key}")
# print(f"Debug Mode: {debug_mode}")

# Example of using a variable
if debug_mode:
    print("Application is running in debug mode.")


cwd = Path.cwd()
print(cwd)

project_root = cwd.parents[1]
ROOT_PATH = str(project_root.absolute())

DATA_PATH = ROOT_PATH + "\\data"
LOGS_PATH = ROOT_PATH + "\\logs"
OLLAMA_INPUT =  DATA_PATH + "\\ollama_input"
VENDOR_DB =  DATA_PATH + "\\vendor_db"


print(ROOT_PATH)
print(DATA_PATH)
