from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ENV: str = "dev"
    DEBUG: bool = Field(default=False, env="DEBUG")
    TESTING: bool = Field(default=False, env="TESTING")

    ASPNETCORE_ENVIRONMENT: str = Field(
        default="", env="ASPNETCORE_ENVIRONMENT")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class DevelopmentSetting(Settings):
    ENV: str = "dev"
    DEBUG: bool = Field(default=True, env="DEBUG")
    TESTING: bool = Field(default=False, env="TESTING")

    PROJECT_NAME: str = Field(
        default="Synth Leaders", env="PROJECT_NAME")

    AZURE_ACCOUNT_NAME: str = Field(
        default="", env="AZURE_ACCOUNT_NAME")
    AZURE_ACCOUNT_URL: str = Field(
        default="", env="AZURE_ACCOUNT_URL")
    AZURE_BLOB_ACCESS_KEY: str = Field(
        default="", env="AZURE_BLOB_ACCESS_KEY")
    AZURE_CONTAINER_NAME: str = Field(
        default="", env="AZURE_CONTAINER_NAME")


class ProductionSetting(Settings):
    ENV: str = "dev"
    DEBUG: bool = Field(default=True, env="DEBUG")
    TESTING: bool = Field(default=False, env="TESTING")

    PROJECT_NAME: str = Field(
        default="Synth Leaders", env="PROJECT_NAME")

    AZURE_ACCOUNT_NAME: str = Field(
        default="", env="AZURE_ACCOUNT_NAME")
    AZURE_ACCOUNT_URL: str = Field(
        default="", env="AZURE_ACCOUNT_URL")
    AZURE_BLOB_ACCESS_KEY: str = Field(
        default="", env="AZURE_BLOB_ACCESS_KEY")
    AZURE_CONTAINER_NAME: str = Field(
        default="", env="AZURE_CONTAINER_NAME")


EXPORT_CONFIG = {
    "dev": DevelopmentSetting,
    "prod": ProductionSetting
}
