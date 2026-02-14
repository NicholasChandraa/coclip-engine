"""
Database configuration using SQLAlchemy async with PostgreSQL.

Uses asyncpg as the async driver for high-performance DB operations.
"""

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from app.core.config import settings
from app.utils.logging import logger


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


# Create async engine
# Mask password in URL for logging
_db_url_masked = settings.DATABASE_URL.split("@")[-1] if "@" in settings.DATABASE_URL else settings.DATABASE_URL
logger.info(f"Database engine: postgresql+asyncpg@{_db_url_masked} (pool=5, overflow=10)")

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_size=5,
    max_overflow=10,
)

# Session factory
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    """Create all tables on startup."""
    # Import models so SQLAlchemy knows about them
    from app.models import Job, Clip  # noqa: F401

    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("✅ Database tables initialized (jobs, clips)")
    except Exception as e:
        logger.error(f"❌ Failed to create tables: {e}")
        raise


async def get_db() -> AsyncSession:
    """Get a database session."""
    async with async_session() as session:
        yield session
