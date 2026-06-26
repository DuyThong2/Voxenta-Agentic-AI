# Voxenta Agentic AI Project

> **An Intelligent AI Agent for English Speaking Assessment & Practice**

Voxenta là một dự án AI Agent tiên tiến, được xây dựng để hỗ trợ học sinh ôn luyện, đánh giá năng lực tiếng Anh nói lúc thi và tự động tạo ra các đề ôn tập cá nhân hóa.

## 🎯 Mục Đích Dự Án

- **Ôn Luyện Tiếng Anh Nói**: Thực hành giao tiếp tiếng Anh thực tế với AI Agent
- **Đánh Giá Năng Lực**: Đánh giá độ lưu loát, độ chính xác và kỹ năng giao tiếp
- **Tạo Đề Ôn Tập**: Tự động sinh ra các đề ôn tập phù hợp với trình độ của học sinh
- **Feedback Trực Tiếp**: Cung cấp phản hồi ngay lập tức về điểm mạnh và điểm yếu

## 🏗️ Kiến Trúc & Cấu Trúc Dự Án

Tất cả đường dẫn dưới đây là tương đối so với thư mục `agents/` này (đây chính là project root cho Python/uv — không phải `d:\semester9`).

```
agents/
├── pyproject.toml / uv.lock     # Project dependencies (uv) — main venv (.venv)
├── Dockerfile, docker-compose.yml
├── .env / .env.example
│
├── src/
│   ├── app.py                          # FastAPI entry point
│   ├── auth.py
│   │
│   ├── config/                         # chroma/kafka/langsmith/postgresDB/s3 config
│   ├── controller/                     # FastAPI routers
│   │   ├── archive_controller.py       # POST /turns/archive (turn upload + archival STT)
│   │   ├── evaluate_controller.py      # answer evaluation endpoints
│   │   ├── realtime_controller.py      # WS /realtime/attempts/{exam_attempt_id}
│   │   └── webrtc.py                   # proctoring WebRTC (aiortc + YOLO)
│   │
│   ├── realtime/                       # per-exam-attempt realtime session pipeline
│   │   ├── attempt_connection.py       # one instance per exam attempt
│   │   ├── session.py                  # one instance per question/answer
│   │   └── turn_publisher.py           # idempotent archive + Kafka publish per turn
│   │
│   ├── node/                           # LangGraph graphs
│   │   ├── followUpDecisionGraph/      # decides next prompt / should_continue per turn
│   │   ├── evalGraph/                  # grammar/lexical/coherence/pronunciation scoring
│   │   └── state_models/
│   │
│   ├── infra/                          # Kafka connection/publishers, S3 storage
│   ├── events/                         # Kafka event payload schemas
│   ├── mappers/, schemas/, dtos/, utils/, vector/, database/
│
├── spikes/                             # Phase 0 avatar pipeline PoC (LivePortrait + MuseTalk)
│   ├── README.md                       # setup: HF weights, GitHub repos, GPU notes
│   ├── voice_live_poc.py, avatar_render_poc.py
│   ├── repos/                          # cloned LivePortrait/MuseTalk (gitignored)
│   └── weights/                        # downloaded HF weights (gitignored)
│
├── vendor/                              # patched mmdet/mmpose wheels (see pyproject.toml)
├── data/                                # sample audio/images for PoC scripts
├── notebook/, setup/, docs/, logs/
```

Chi tiết kiến trúc realtime/avatar pipeline (đang xây) nằm ở `docs/realtime-self-hosted-avatar-plan.md`. Phase 0 (avatar PoC, dependency setup) chạy ngay trong venv chính (`agents/.venv`) — xem `spikes/README.md` để biết cách lấy weights và clone repo.

## 💻 Công Nghệ Sử Dụng

### Backend & AI Framework
- **FastAPI** - Web framework for REST APIs
- **LangGraph** - Agent orchestration & workflow management
- **Anthropic Claude** - LLM for AI responses
- **OpenAI** - Alternative LLM support
- **LangChain** - LLM integration utilities

### Database & Storage
- **PostgreSQL** - Primary database + LangGraph checkpointer
- **ChromaDB** - Vector database for semantic search
- **psycopg-pool** - Database connection pooling

### Message Queue & Async
- **Kafka** - Message broker (aiokafka)
- **AsyncIO** - Asynchronous operations

### Development & Deployment
- **UV** - Python package manager (faster than pip)
- **Docker** - Containerization
- **Uvicorn** - ASGI server

## 🚀 Cách Chạy Dự Án

### 1. Prerequisites
- Python 3.11 (pinned via `.python-version` — required by the avatar-pipeline deps below, e.g. `tensorflow==2.12.0` has no Python 3.12 wheel)
- UV package manager (`pip install uv`)
- PostgreSQL instance running
- Kafka instance running
- ChromaDB instance (local or remote)
- (Optional, for the avatar PoC in `spikes/`) NVIDIA GPU — see `spikes/README.md`

### 2. Installation

All commands below assume your shell's current directory is already this `agents/` folder.

```bash
uv sync
```

If `mmcv` ever comes back with the wrong cached wheel after a CUDA/index change, use the helper
below instead of remembering the repair flags by hand:

```powershell
.\setup\sync-python-env.ps1
```

It runs `uv sync`, smoke-tests `mmcv._ext`, and only if that check fails does it automatically
rerun sync with `--reinstall-package mmcv --link-mode=copy`.

### 3. Environment Setup

Create a `.env` file in the root directory with the following variables:

```env
# PostgreSQL
POSTGRESQL_URI=postgresql+asyncpg://user:password@localhost:5432/voxenta

# Kafka
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_CLIENT_ID=vox-agents
KAFKA_CONSUMER_GROUP=vector-indexing
KAFKA_AUTO_OFFSET_RESET=earliest
KAFKA_COMPLETED_TOPIC=paper-ingestion-completed
KAFKA_INGEST_TOPIC=paper-ingestion
KAFKA_VECTOR_INDEX_TOPIC=vector-indexing

# ChromaDB
CHROMA_HOST=localhost
CHROMA_PORT=8000

# LLM API Keys
OPENROUTER_API_KEY=your_api_key_here
ANTHROPIC_API_KEY=your_api_key_here
OPENAI_API_KEY=your_api_key_here

# Other settings
FRONTEND_URL=http://localhost:3000
GATEWAY_URL=http://localhost:8080
```

### 4. Running the Application

```bash
# Navigate to src directory
cd src

# Start the FastAPI server with auto-reload
uv run uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The application will be available at:
- **API**: http://0.0.0.0:8000
- **API Docs**: http://0.0.0.0:8000/docs (Swagger UI)
- **ReDoc**: http://0.0.0.0:8000/redoc

## 📋 API Endpoints

Main endpoints are defined in `controller/controller.py`:

- `GET /health` - Health check
- `POST /chat` - Send message to AI Agent
- `GET /assessments` - Get assessment results
- `POST /generate-questions` - Generate practice questions
- `GET /history` - Get conversation history

## 🔄 Workflow Architecture

The application uses **LangGraph** for workflow management:

```
StartNode
   ↓
ElsaSpeakingNode (AI Assessment & Feedback)
   ↓
VectorNode (Semantic Search & Indexing)
   ↓
Response
```

Each node processes the conversation state and enriches it with:
- Grammar analysis
- Pronunciation assessment
- Vocabulary suggestions
- Similar practice topics (via vector search)

## 📊 Message Queue Flow

```
Application
   ↓ (Publish)
Kafka Topic (paper-ingestion-completed)
   ↓ (Consume)
KafkaConsumer
   ↓
Vector Indexer (ChromaDB)
```

This asynchronous flow ensures non-blocking AI processing.

## 🛠️ Development Notes

### Database Migrations

PostgreSQL is initialized with LangGraph checkpointing tables automatically via:
```python
checkpointer_setup = PostgresSaver(conn)
checkpointer_setup.setup()
```

### Vector Indexing

Document embeddings are stored in ChromaDB for semantic search functionality:
```bash
python src/vector/indexer.py
```

### Testing

Run diagnostics:
```bash
python src/setup/diagnostics.py
```

## 📝 Project Features

- ✅ Real-time AI conversation agent
- ✅ Automatic speaking assessment
- ✅ Custom question generation
- ✅ Persistent conversation history
- ✅ Vector-based similarity search
- ✅ Async message processing
- ✅ Scalable with message queues

## 🚧 Development Status

- **Phase 1**: Core AI Agent & Assessment ✅
- **Phase 2**: Question Generation 🔄
- **Phase 3**: Advanced Analytics 📅
- **Phase 4**: Multi-language Support 📅

## 📧 Support & Contact

For issues or questions about the project, please create an issue or reach out to the development team.

---

**Voxenta** - Empowering English Learners Through Intelligent AI 🎤
