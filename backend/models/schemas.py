"""Pydantic 请求/响应模型"""
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime


class HealthResponse(BaseModel):
    status: str
    version: str


class TaskResponse(BaseModel):
    id: int
    task_type: str
    chapter: int
    status: str
    params: Optional[str] = None
    result: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
