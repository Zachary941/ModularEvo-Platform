"""数据库模型 & 初始化"""
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

from core.config import DB_PATH

DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Task(Base):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_type = Column(String(50), nullable=False)    # modularize / finetune / merge / evaluate
    chapter = Column(Integer, nullable=False)          # 3 or 4
    status = Column(String(20), default="pending")     # pending / running / completed / failed
    params = Column(Text)                              # JSON 参数
    result = Column(Text)                              # JSON 结果
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Model(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    model_type = Column(String(50), nullable=False)   # base / module / finetuned / merged
    chapter = Column(Integer, nullable=False)
    path = Column(Text)
    params_count = Column(Integer)
    metadata_json = Column(Text)                       # JSON 元信息
    created_at = Column(DateTime, default=datetime.utcnow)


class Evaluation(Base):
    __tablename__ = "evaluations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(Integer)
    model_id = Column(Integer)
    dataset = Column(Text)
    metrics = Column(Text)                             # JSON: {f1, precision, accuracy, ...}
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    """创建所有表"""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI 依赖注入：获取数据库会话"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
