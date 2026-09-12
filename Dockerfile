# QuickLoan — production Dockerfile
# Session 15: Cloud Deployment
#
# Two build stages on top of the original three-layer mental model:
#   Stage 1 (builder) — system + Python environment, with compiler toolchain
#   Stage 2 (runtime) — application layer only; no compiler toolchain shipped
#
# Build (from the repo root, where this file and requirements.txt/data/ live):
#   docker build -t quickloan .
#
# Run:
#   docker run -p 8501:80 -e GROQ_API_KEY=gsk_... quickloan
#   docker run -p 8501:80 --env-file .env quickloan
# (host:container -- container listens on 80 now; map it to any host port you like)
#
# Security: GROQ_API_KEY is passed at runtime only — never baked into the image.
# Anyone who runs `docker history quickloan` would see a baked-in key.

# ── Stage 1: builder — system environment + Python environment ───────────────
# python:3.11-slim is an existing image which our Dockerfile will use as a base.
# There is a central location where lots of Docker images are available, re-usable.
# Docker registry - docker hub
FROM python:3.11-slim AS builder

WORKDIR /app

# Skip .pyc generation during this stage's `pip install`/`python -c` steps below --
# nothing here benefits from bytecode caching (the container starts once, long-
# lived), so there's no reason to carry those files into the copied artefacts.
ENV PYTHONDONTWRITEBYTECODE=1

# chromadb and tokenizers require C build tools to compile their wheels.
# build-essential is only needed here, in the builder stage -- the compiled
# .so files it produces don't need the compiler itself at runtime, so stage 2
# never installs it (see note there).
# --no-install-recommends keeps this layer ~40 MB smaller.
# Deleting the apt cache in the same RUN layer avoids storing it in the image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt BEFORE any application code.
# Docker caches this layer separately. When you edit app.py, Docker skips
# this expensive pip install and jumps straight to the runtime stage below.
COPY requirements.txt .

# sentence-transformers pulls in torch. The default PyPI wheel is the full
# CUDA build (~500+ MB) even though this container only ever runs CPU
# inference. Installing the CPU-only wheel first (from PyTorch's own index,
# ~180 MB, no bundled CUDA) means the requirements.txt install below finds
# torch already satisfied and skips the huge GPU build entirely.
#
# --user installs to /root/.local instead of system site-packages, so stage 2
# can copy across exactly this directory without dragging along anything
# apt/pip touched at the system level.
RUN pip install --no-cache-dir --default-timeout=120 --retries 5 --user \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir --default-timeout=120 --retries 5 --user -r requirements.txt

# Pre-download the embedding model (~90 MB from HuggingFace Hub) into this
# stage's cache. Baking it in means the first customer query starts instantly
# rather than triggering a 30-second download inside the running container.
# Stage 2 copies this cache dir across alongside /root/.local.
RUN python -c \
    "from sentence_transformers import SentenceTransformer; \
     SentenceTransformer('all-MiniLM-L6-v2')"

# ── Stage 2: runtime — application layer, no compiler toolchain ──────────────
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
# PYTHONUNBUFFERED: this app's tracing is plain print() (route decisions,
# MCP tool calls, guard/compliance outcomes -- see nodes.py/agent.py).
# Python fully buffers stdout when it isn't a TTY, which is exactly `docker
# logs` -- without this, those lines sit in a buffer instead of appearing
# promptly under `docker logs -f`.

WORKDIR /app

# curl is needed at runtime for the HEALTHCHECK below. libcap2-bin provides
# setcap, used just below to allow binding port 80 without root. build-essential
# is deliberately NOT installed here -- the wheels it built in stage 1 already
# work without the compiler that built them, so shipping the toolchain would
# only add size and attack surface to the image customers actually run.
#
# appuser: run the app as a non-root user. --create-home gives it a real
# home directory, which Streamlit needs (it writes ~/.streamlit/ on first
# launch); --shell nologin reflects that this account is never logged into
# interactively, it only runs the one process.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libcap2-bin \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /usr/sbin/nologin appuser

# Cloud platforms (Azure Container Instances, etc.) expose exactly the port
# the app listens on -- there's no host:container remapping like `docker run
# -p`. To serve on the plain default port 80 (no ":8501" in the URL) without
# running the whole process as root, grant the python interpreter itself the
# one Linux capability needed to bind ports below 1024. This is a file
# capability (an xattr on the binary, survives the image layer), not a
# process privilege -- appuser stays non-root for everything else. Docker's
# (and ACI's) default container capability bounding set already includes
# CAP_NET_BIND_SERVICE, so this Just Works without any extra --cap-add flag
# at `docker run`/`az container create` time.
RUN setcap 'cap_net_bind_service=+ep' "$(readlink -f "$(which python3)")"

# Installed Python packages and the pre-downloaded embedding-model cache
# from the builder stage, handed to the non-root user this container runs as.
# (COPY --chown sets ownership explicitly regardless of the active USER --
# USER only governs subsequent RUN/CMD, not COPY -- so this is required, not
# redundant, even after the USER switch below.)
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local
COPY --from=builder --chown=appuser:appuser /root/.cache /home/appuser/.cache
ENV PATH=/home/appuser/.local/bin:$PATH

# QuickLoan's package lives four directories below the repo root
# (s01/starter/quickloan/config.py), and config.py's DATA_DIR walks up
# exactly that many parents to reach data/ (Path(__file__).parent.parent
# .parent.parent / "data"). The COPY layout below preserves that same depth
# inside the image -- /app/s01/starter/quickloan/ next to /app/data/ -- so
# DATA_DIR resolves correctly without touching application code. Flattening
# this layout (a single top-level app.py + package) would silently break
# that path -- no error, just an empty data dir.

# Policy documents and data scripts (repo-root data/, matches config.py's
# DATA_DIR). Copied and seeded before any application code below: seed.py/
# ingest.py depend only on data/documents/ and their own script content,
# never on the quickloan package, so this ordering means editing
# app.py/nodes.py/etc. during development never invalidates this
# comparatively expensive layer (it re-embeds every policy document).
COPY --chown=appuser:appuser data/ data/

USER appuser

# Build the SQLite database and the ChromaDB vector store at image build time.
# Both scripts are idempotent — running them twice is safe. seed.py/ingest.py
# resolve their own paths via Path(__file__).parent, so this works regardless
# of WORKDIR as long as "data/" is reachable from it (it is, from /app).
# Baking data into the image means the container starts with data ready,
# with no external database dependency. (In production you would use an
# external RDS / managed ChromaDB instead.) Runs as appuser (see USER above)
# so the resulting files are owned by the same user that reads them later.
RUN python data/seed.py && python data/ingest.py

# Application source code -- app.py, langgraph.json, and the quickloan
# package all live under s01/starter/ in this repo, not at the repo root.
COPY --chown=appuser:appuser s01/starter/app.py s01/starter/app.py
COPY --chown=appuser:appuser s01/starter/langgraph.json s01/starter/langgraph.json

# mcp_server.py lives inside the quickloan/ package itself (unlike a flat
# layout where it might sit beside app.py), so copying the package copies
# it automatically -- no separate COPY step needed.
COPY --chown=appuser:appuser s01/starter/quickloan/ s01/starter/quickloan/

# Switch into the app's actual run directory (matches CLAUDE.md's documented
# `cd s01/starter && streamlit run app.py`) now that data/ has been seeded
# from the repo root above.
WORKDIR /app/s01/starter

EXPOSE 80

# Cloud platforms (Render, Railway, Cloud Run) poll this endpoint to decide
# when the container is ready to receive traffic.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:80/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=80", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
