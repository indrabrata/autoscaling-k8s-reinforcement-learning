from dotenv import load_dotenv

load_dotenv()  # by default, looks for .env in current dir

import os


class Settings:
    APP_NAME: str = "E-Commerce RL Simulator"
    ENV: str = os.getenv("ENV", "dev")
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: str = os.getenv("DB_PORT", "5432")
    DB_USER: str = os.getenv("DB_USER", "ecom")
    DB_PASS: str = os.getenv("DB_PASS", "ecompass")
    DB_NAME: str = os.getenv("DB_NAME", "ecomdb")

    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASS}@"
            f"{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

settings = Settings()
