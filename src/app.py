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
import logging.handlers
import os

_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "var", "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format=_LOG_FORMAT,
)

# Console output alone requires catching it live -- durably persist every log line to disk too,
# so a failure can be diagnosed after the fact from the file instead of needing to have been
# watching the terminal at the time. Rotates at 10MB x 5 files so it can't grow unbounded.
_file_handler = logging.handlers.RotatingFileHandler(
    os.path.join(_LOG_DIR, "agents.log"),
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
    encoding="utf-8",
)
_file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
logging.getLogger().addHandler(_file_handler)


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
from realtime._legacy_avatar.avatar_webrtc import close_all_connections as close_all_avatar_connections
from realtime.attempt.registry import close_all_attempt_connections
from realtime.practice_attempt.registry import close_all_practice_attempt_connections
from node.followUpDecisionGraph.graphConfig import build_archive_graph, build_text_followup_graph
from node.evalGraph.graphConfig import build_graph
from node.practiceEvalGraph.graphConfig import build_practice_graph
from config.kafka_config import settings
from config.postgresDB_config import settings as pg_settings
from infra.message_broker.external_events_handlers.question_asset_analysis_consumer import (
    start_question_asset_analysis_consumer,
)
from infra.message_broker.external_events_handlers.exam_consumer import (
    start_exam_attempt_consumer,
    start_exam_attempt_force_end_consumer,
)
from infra.message_broker import connection as mq_connection
from realtime.background import drain as drain_background_tasks

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

    # Evaluation requests are complete, immutable inputs and retries must start from a clean
    # state. Persisting this graph reused the same turn thread_id across retries, so merged
    # metadata retained an old pronunciation_error even after Azure later returned a score.
    app.state.graph = build_graph()
    # Do thi rieng cho LUYEN TAP: giong het ban thi tru AzureScoreScaleNode -- luyen tap cham
    # thang 0-100 co dinh (thang goc Azure tra ve) nen khong con buoc anh xa sang dai rubric.
    # Tach hai do thi de sua duong luyen khong the cham vao duong thi.
    app.state.practice_graph = build_practice_graph()
    app.state.archive_graph = build_archive_graph(checkpointer)
    app.state.text_followup_graph = build_text_followup_graph()

    # 2) Start exam-attempt-evaluation-requested consumer (grading pipeline) -- defined in
    # exam_consumer.py but was never actually started anywhere, so no grading requests from vox
    # were ever consumed despite the handler being fully implemented.
    exam_consumer_tasks = [
        asyncio.create_task(start_exam_attempt_consumer(app, instance_label=str(index)))
        for index in range(settings.KAFKA_EXAM_CONSUMER_CONCURRENCY)
    ]
    app.state.exam_consumer_tasks = exam_consumer_tasks
    practice_consumer_tasks = [
        asyncio.create_task(
            start_exam_attempt_consumer(
                app,
                instance_label=f"practice-{index}",
                request_topic=settings.KAFKA_PRACTICE_REQUEST_TOPIC,
                consumer_group=settings.KAFKA_PRACTICE_CONSUMER_GROUP,
            )
        )
        for index in range(settings.KAFKA_EXAM_CONSUMER_CONCURRENCY)
    ]
    app.state.practice_consumer_tasks = practice_consumer_tasks
    force_end_consumer_task = asyncio.create_task(start_exam_attempt_force_end_consumer(app))
    app.state.force_end_consumer_task = force_end_consumer_task
    asset_analysis_consumer_task = asyncio.create_task(start_question_asset_analysis_consumer(app))
    app.state.asset_analysis_consumer_task = asset_analysis_consumer_task

    try:
        yield
    finally:
        await close_all_attempt_connections()
        await close_all_practice_attempt_connections()
        await close_all_connections()
        await close_all_avatar_connections()
        # Cho tac vu nen (day luot noi len Kafka, upload audio...) ket thuc TRUOC khi tat.
        # Dat sau khi dong ket noi de khong con task moi sinh ra, va truoc khi huy consumer.
        # Han 10s -- phai ngan hon terminationGracePeriodSeconds cua k8s (mac dinh 30s), khong
        # thi SIGKILL cat giua chung va viec cho chi lam cham qua trinh tat pod.
        await drain_background_tasks()
        for task in exam_consumer_tasks:
            task.cancel()
        for task in practice_consumer_tasks:
            task.cancel()
        force_end_consumer_task.cancel()
        asset_analysis_consumer_task.cancel()
        await mq_connection.close()
        pool.close()


# -----------------------
# FastAPI App Initialization
# -----------------------
app = FastAPI(
    title="Vox Practice Agents",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
