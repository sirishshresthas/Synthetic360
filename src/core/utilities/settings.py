import os

from .config import EXPORT_CONFIG

# Define a default environment explicitly
DEFAULT_ENV = "dev"

# Fetch the environment, defaulting to "dev" if not set
ENV = os.getenv("ASPNETCORE_ENVIRONMENT", DEFAULT_ENV)

# Get the configuration callable based on the environment
config_callable = EXPORT_CONFIG.get(ENV)

# Ensure the configuration exists and is callable
if config_callable and callable(config_callable):
    settings = config_callable()
else:
    # Handle missing or incorrect configuration (raise an error or use a default)
    raise ValueError(
        f"Missing or invalid configuration for environment: {ENV}")
