#!/usr/bin/env bash

uv -V > /dev/null 2>&1 || {
    echo "Please install Python and uv: https://docs.astral.sh/uv/getting-started/installation"
    exit 1
}

uv sync || {
    echo "Failed to sync dependencies"
    exit 1
}

mkdir -p games && cd games || {
    echo "Failed to create games directory"
    exit 1
}

wget -O games.zip https://github.com/BYU-PCCL/z-machine-games/archive/master.zip || {
    echo "Failed to download games"
    exit 1
}

unzip -q -j games.zip "z-machine-games-master/jericho-game-suite/*" || {
    echo "Failed to unzip games"
    exit 1
}
rm -f games.zip || {
    echo "Failed to remove games zip"
    exit 1
}

cd .. || {
    echo "Failed to return to root directory"
    exit 1
}

echo "Installation complete, run «uv run main.py --help» to see the available options"