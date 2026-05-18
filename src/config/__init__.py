"""Application configuration."""

import os

JWT_SECRET = os.getenv("JWT_SECRET", "")
