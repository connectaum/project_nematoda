# dlc-env — изолированное окружение DeepLabCut

Создано 2026-09-05 на macOS arm64: Python 3.11 (uv), deeplabcut[gui] 3.0.1, torch 2.14 (MPS работает), numpy 1.26.

Пересоздать с нуля:
    cd dlc-env
    uv venv --python 3.11 .venv
    uv pip install --python .venv/bin/python "deeplabcut[gui]" "napari[pyside6]>=0.5,<0.7"

Использование:
    ./dlc-env/dlc                # GUI
    ./dlc-env/dlc run_all.py ... # скрипт в этом окружении
    source dlc-env/.venv/bin/activate

Не смешивать с ../.venv и ../.venv-omni — у них другие numpy/torch.

Известная грабля: napari>=0.9 несовместим с napari-deeplabcut 0.3.1 (ImportError SYMBOL_TRANSLATION_INVERTED) — поэтому napari закреплён <0.7.
