"""
FastAPI application entry point with DB utilities and app initialization.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import asyncio
import sys

# Windows' console defaults stdout/stderr to the legacy cp1252 codepage, which can't encode
# arbitrary Unicode characters in log/print output -- harmless on Linux/macOS, where stdout is
# already UTF-8.
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import logging

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logging.getLogger("infra.message_broker.external_events_handlers.kafka_consumer").setLevel(logging.INFO)
logging.getLogger("vector.indexer").setLevel(logging.INFO)

from utils import load_root_dotenv

load_root_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from langgraph.checkpoint.postgres import PostgresSaver
import psycopg
from psycopg_pool import ConnectionPool

from auth import get_current_user_from_request, set_current_user_context
from config.langsmith_config import setup_langsmith, get_langsmith_status

setup_langsmith()

from controller import router
from controller.webrtc import close_all_connections
from realtime.avatar_webrtc import close_all_connections as close_all_avatar_connections
from node.followUpDecisionGraph.graphConfig import build_archive_graph, build_text_followup_graph
from node.evalGraph.graphConfig import build_graph
from config.postgresDB_config import settings as pg_settings
from infra.message_broker.external_events_handlers.kafka_consumer import start_outbox_consumer
from infra.message_broker import connection as mq_connection
from vector.chroma_client import build_chroma_collection

logger = logging.getLogger(__name__)


# -----------------------
# FastAPI lifespan
# -----------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[app] Starting up...")
    logger.info(f"[langsmith] {get_langsmith_status()}")

    # 1) Setup checkpointer
    with psycopg.connect(pg_settings.PG_URI, autocommit=True) as conn:
        checkpointer_setup = PostgresSaver(conn)
        checkpointer_setup.setup()

    pool = ConnectionPool(pg_settings.PG_URI, min_size=1, max_size=10)
    checkpointer = PostgresSaver(pool)

    app.state.graph = build_graph(checkpointer)
    app.state.archive_graph = build_archive_graph(checkpointer)
    app.state.text_followup_graph = build_text_followup_graph()

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
        await close_all_connections()
        await close_all_avatar_connections()
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


@app.middleware("http")
async def bind_current_user_context(request, call_next):
    request.state.current_user = None
    request.state.current_user_id = None
    set_current_user_context(None)

    try:
        user = get_current_user_from_request(request, required=False)
        if user is not None:
            request.state.current_user = user
            request.state.current_user_id = user.user_id
    except Exception:
        logger.exception("[auth] failed to bind current user context")

    try:
        response = await call_next(request)
    finally:
        set_current_user_context(None)

    return response

app.include_router(router)
