# app.py
import os
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