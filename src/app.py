"""
FastAPI application entry point with DB utilities and app initialization.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import asyncio
import sys

import logging

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logging.getLogger("infra.rabbit_consumer").setLevel(logging.INFO)
logging.getLogger("vector.indexer").setLevel(logging.INFO)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from langgraph.checkpoint.postgres import PostgresSaver
import psycopg
from psycopg_pool import ConnectionPool

from controller import router
from node.graphConfig import build_graph
from config.postgresDB_config import settings as pg_settings
from infra.message_broker.rabbit_consumer import start_outbox_consumer
from infra.message_broker import connection as mq_connection
from vector.chroma_client import build_chroma_collection

logger = logging.getLogger(__name__)


# -----------------------
# FastAPI lifespan
# -----------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[app] Starting up...")

    # 1) Setup checkpointer
    with psycopg.connect(pg_settings.PG_URI, autocommit=True) as conn:
        checkpointer_setup = PostgresSaver(conn)
        checkpointer_setup.setup()

    pool = ConnectionPool(pg_settings.PG_URI, min_size=1, max_size=10)
    checkpointer = PostgresSaver(pool)

    
    app.state.graph = build_graph(checkpointer)

    # 2) Setup Chroma collection
    try:
        app.state.chroma_collection = build_chroma_collection()
    except Exception:
        logger.exception("[chroma] failed to init chroma collection")
        raise

    # 3) Start outbox consumer
    consumer_task = asyncio.create_task(start_outbox_consumer(app))
    app.state.outbox_task = consumer_task

    try:
        yield
    finally:
        consumer_task.cancel()
        await mq_connection.close()
        pool.close()


# -----------------------
# FastAPI App Initialization
# -----------------------
app = FastAPI(
    title="Chat + Product Cards Demo (seed from JSON file)",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
