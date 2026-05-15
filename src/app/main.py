"""FastAPI application."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.config.settings import get_settings
from app.common.logging_config import setup_logging

# Initialize settings and logger at the module level to ensure they are available throughout the app.
settings = get_settings()
logger = setup_logging(settings=settings)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
    """Application lifespan handler."""
    logger.info("All startup tasks completed successfully")
    try:
        yield
    finally:
        logger.info("Application shutdown: Cleaning up resources...")
        logging.shutdown()


app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    debug=settings.debug,
    version=settings.app_version,
    redoc_url=None,
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins_list, allow_methods=["*"], allow_headers=["*"])
app.include_router(api_router)
