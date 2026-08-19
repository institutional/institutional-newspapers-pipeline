# DEBIAN & CO | System-level dependencies (partial)
if command -v apt >/dev/null 2>&1; then
    sudo apt install libgl1 libjpeg-dev libtiff-dev libpng-dev libicu-dev pkg-config libicu-dev sqlite3
fi

# MACOS - Brew dependencies (partial)
if [[ "$OSTYPE" == "darwin"* ]]; then
    brew install pkg-config icu4c;
fi

# Python dependencies management
uv python install 3.12;
uv venv;
uv pip install pip setuptools wheel;
uv sync;

# Copy environment template if it doesn't exist
cp -n .env.example .env || true;