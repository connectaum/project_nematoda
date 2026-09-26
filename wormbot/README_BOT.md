# wormbot — Telegram-бот: видео нематоды → готовый DLC-проект

Стек на VPS (`~/wormbot`), docker compose, 2 сервиса:

- **tgapi** — локальный Telegram Bot API сервер (`aiogram/telegram-bot-api`,
  режим `TELEGRAM_LOCAL`): поднимает лимит файлов с 20 МБ до 2 ГБ в обе стороны.
- **bot** — python-бот (pyTelegramBotAPI) + классический пайплайн проекта:
  `worm_pipeline.py` (сегментация → скелет → 49/5 точек, авторазметка) →
  `make_dlc_project.py` (maDLC-проект: 5 точек, worm1–worm3, 40 разнообразных
  кадров/видео) → `worm_kinematics.py` (фичи локомоции по трекам).

## Использование

1. Прислать боту 1..N видео (лучше файлом-«документом», чтобы Telegram не пережимал).
2. `/go` — обработка. Очередь: одно видео за раз (VPS 1 ядро; ~0.3–0.6 с/кадр).
3. Бот возвращает: overlay.mp4 и kinematics.png на каждое видео, текстовую
   сводку кинематики и **zip с DLC-проектом** (`config.yaml`, `labeled-data/`,
   `auto_labels/`, `analysis/`).
4. Zip распаковать в `<PROJECT_PATH_PREFIX>/` (папка из `.env`) — `project_path` в
   config.yaml уже прописан туда; исходные видео положить в `<проект>/videos/`.

5. Если в видео есть эпизоды контакта червей (слипание, кольца), бот дополнительно
   присылает `episodes_<видео>.zip`. Его обрабатывают на рабочей станции инструментом
   **wormsam2** (папка `wormsam2/` в проекте: `install.ps1` / `install.sh`, затем
   `wormsam2 run episodes_<видео>.zip`), а полученный `sam2_result_<видео>.zip`
   отправляют боту файлом — он вливает разметку SAM2 в `auto_labels/<видео>/labels49.npy`
   (`pipeline/merge_sam2.py`), пересчитывает кинематику и возвращает `sam2_merged_<видео>.zip`.
   Экспорт эпизодов — `pipeline/episode_export.py` (эпизод = кадры с флагом cluster или
   компонент > 1.5 медианной площади червя; подсказка для SAM2 = последняя «чистая» средняя
   линия каждого трека рядом с эпизодом).

Команды: `/status`, `/cancel`, `/help`.

## Секреты (.env на VPS, не в git)

```
TELEGRAM_API_ID=...      # my.telegram.org → API development tools
TELEGRAM_API_HASH=...
BOT_TOKEN=...            # @BotFather → /newbot
ALLOWED_USERS=...        # ваш telegram user id (пусто = доступ всем!)
```

## Эксплуатация

```bash
cd ~/wormbot
docker compose up -d --build     # старт / обновление
docker compose logs -f bot       # логи
ls work/jobs/                    # артефакты задач (чистить при нехватке места)
```

Ресурсы (с 2026-09-14 бот живёт на хосте apps: 4 vCPU, 8 ГБ RAM): контейнер
бота ограничен 5 ГБ (`mem_limit`). Задачи обрабатываются по одной, но pass 1
пайплайна (сегментация кадров) идёт в `PIPELINE_WORKERS` процессов (4) —
IMG_8374 (3177 кадров) 4:42 → 1:57, результат бит-в-бит равен последовательному.
Overlay в полном разрешении (`OVERLAY_SCALE=1.0`), H.264 `medium`/crf 23,
60 кадров на видео в DLC-проект (`NFRAMES`), видео задач хранятся 90 дней
(`KEEP_DAYS`). На старом 1-ядерном VPS было: 2 ГБ, scale 0.5, veryfast/26,
40 кадров, 21 день. Кинематика: подробности фич — в `worm_kinematics.py` и
pipeline-заметках проекта.

DeepTangle (с 2026-09-19): после классического пайплайна каждое видео прогоняется через
DeepTangleCrawl (Tierpsy Tracker 2.0) — сеть, дающая 49-точечные мидлайны сквозь
пересечения и кольца (`pipeline/dtc_infer.py`, ансамбль исходной модели и нашей
дообученной, `models/dtc/`, ~0.1 с/кадр на 4 vCPU), результат вливается в auto_labels
(`pipeline/dtc_merge.py`: заполняет кадры без разметки, чинит кадры с флагами
skeleton_branches/length_outlier; статус `dtc`/`dtc_fix`, флаг `from_dtc`, бэкап
`labels49_before_dtc.npy`), overlay перерисовывается (чёрная обводка, «D» у номера).
Отключить: `DTC_ENABLE=0`. Модели монтируются в контейнер из `./models` (в git/zip не
попадают, 254 МБ). Заметки об обучении — `bench/dtc/NOTES_finetune.md`.

Локальный тест без Telegram: `python bot.py --process video.mp4 [...]`.
