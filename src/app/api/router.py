"""Main API router that assembles all versioned sub-routers."""

from fastapi import APIRouter

from app.api.v1.routes import router
from app.config.settings import API_V1_PREFIX

api_router = APIRouter()
api_router.include_router(router, prefix=API_V1_PREFIX)
