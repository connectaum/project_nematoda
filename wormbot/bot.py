#!/usr/bin/env python3
"""
wormbot — Telegram bot: nematode video(s) in -> ready DeepLabCut project out.

Flow per user:
  1. user sends one or more videos (as video or as file/document)
  2. /go  -> for each video: worm_pipeline.py (auto-labels) ->
             make_dlc_project.py (maDLC project, 5 pts, 3 individuals) ->
             worm_kinematics.py (per-track features)
  3. bot returns: overlay.mp4 + kinematics.png per video, a kinematics text
     summary, and a zip with the DLC project (+ auto_labels).

Runs against a LOCAL telegram-bot-api server (TELEGRAM_LOCAL) so files up to
2 GB work in both directions. Videos are read straight from the shared
/var/lib/telegram-bot-api volume, not re-downloaded over HTTP.

CLI test mode (no Telegram):  python bot.py --process v1.mp4 [v2.mp4 ...]
"""
import json
import logging
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from datetime import datetime
from pathlib import Path

def to_h264(path):
    """Telegram clients render OpenCV's mp4v (MPEG-4 part 2) as a frozen frame; re-encode to H.264 yuv420p + faststart."""
    out = path.with_name(path.stem + "_h264.mp4")
    if out.exists() and out.stat().st_mtime >= path.stat().st_mtime:
        return out
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-c:v", "libx264", "-preset", H264_PRESET,
                        "-crf", H264_CRF, "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                        "-movflags", "+faststart", "-an", str(out)], check=True, timeout=1800)
        return out
    except Exception as e:
        log.warning("ffmpeg transcode failed (%s), sending the original", e)
        return path


log = logging.getLogger("wormbot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

HERE = Path(__file__).resolve().parent
PIPELINE = HERE / "pipeline"
WORK = Path(os.environ.get("WORK_DIR", str(HERE / "work")))
PROJECT_PATH_PREFIX = os.environ.get("PROJECT_PATH_PREFIX", "/path/to/project_nematoda")
INDIVIDUALS = int(os.environ.get("INDIVIDUALS", "3"))
NFRAMES = int(os.environ.get("NFRAMES", "60"))
H264_PRESET = os.environ.get("H264_PRESET", "medium")   # 2026-09-14 (apps host, 4 vCPU): was veryfast/26 on the 1-core VPS
H264_CRF = os.environ.get("H264_CRF", "23")
OVERLAY_SCALE = os.environ.get("OVERLAY_SCALE", "0.5")
ALLOWED_USERS = {int(x) for x in os.environ.get("ALLOWED_USERS", "").replace(",", " ").split() if x.strip()}
KEEP_DAYS = int(os.environ.get("KEEP_DAYS", "90"))
DTC_ENABLE = os.environ.get("DTC_ENABLE", "1") == "1"        # DeepTangleCrawl contact/coil solver (pipeline/dtc_infer.py + dtc_merge.py)
DTC_MODEL_DIR = os.environ.get("DTC_MODEL_DIR", "/models/dtc/base")
DTC_FT = os.environ.get("DTC_FT", "/models/dtc/ft_run1_6000.pkl")     # jobs (incl. the kept source videos) older than this are deleted
PY = sys.executable

# ---------------------------------------------------------------- pipeline


def run(cmd, cwd=None, env_extra=None, log_to=None):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    log.info("run: %s", " ".join(str(c) for c in cmd))
    with open(log_to, "ab") if log_to else open(os.devnull, "wb") as lf:
        p = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        lf.write(p.stdout)
    if p.returncode != 0:
        tail = p.stdout.decode(errors="replace").splitlines()[-15:]
        raise RuntimeError("command failed: %s\n%s" % (cmd[0], "\n".join(tail)))
    return p.stdout.decode(errors="replace")


def fmt(v, nd=2):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "–"
    if f != f:  # NaN
        return "–"
    return f"{f:.{nd}f}"


def kin_summary_text(analysis_dir: Path, stems):
    """Compact Russian summary from analysis/kin_tracks.csv."""
    import pandas as pd
    csv = analysis_dir / "kin_tracks.csv"
    if not csv.exists():
        return "Кинематика: kin_tracks.csv не создан (слишком мало валидных кадров)."
    df = pd.read_csv(csv)
    lines = ["📊 Кинематика (по трекам):"]
    for stem in stems:
        sub = df[df["video"] == stem] if "video" in df.columns else df
        if sub.empty:
            lines.append(f"\n{stem}: нет треков с ≥3 с валидных кадров")
            continue
        lines.append(f"\n{stem}:")
        for _, r in sub.iterrows():
            tr = str(r.get("track", "?"))
            if not tr.startswith("worm"):
                tr = "worm" + tr
            lines.append(
                f"  {tr}: L≈{fmt(r.get('L_px'), 0)} px, "
                f"скорость {fmt(r.get('speed_bl_s'))} L/с, "
                f"частота {fmt(r.get('freq_hz'))} Гц, "
                f"λ/L {fmt(r.get('wavelength_bl'))}, "
                f"движется {fmt(100 * r.get('frac_move', float('nan')), 0)}% времени"
            )
    lines.append(
        "\nОриентиры по прошлым видео: ползание 0.13–0.25 Гц, λ/L 0.6–0.86, "
        "0.06–0.12 L/с; мелкие быстрые черви (IMG_8375) 0.38 L/с, λ/L 0.24."
    )
    return "\n".join(lines)


def prune_old_jobs():
    """Delete job directories older than KEEP_DAYS (they hold the source videos for overlay re-rendering)."""
    cutoff = time.time() - KEEP_DAYS * 86400
    for d in (WORK / "jobs").glob("*"):
        try:
            if d.is_dir() and d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
                log.info("pruned old job %s", d.name)
        except OSError as e:
            log.warning("prune %s: %s", d, e)


def find_job_for_video(stem):
    """Latest job directory that has auto_labels/<stem>/summary.json."""
    cands = sorted((WORK / "jobs").glob(f"*/auto_labels/{stem}/summary.json"))
    return cands[-1].parents[2] if cands else None


def merge_sam2_result(result_zip: Path, progress=lambda text: None):
    """Merge sam2_result_<video>.zip (from the wormsam2 tool) into its job and recompute the kinematics."""
    with zipfile.ZipFile(result_zip) as z:
        res = json.loads(z.read("result.json"))
    stem = res["video"]
    job_dir = find_job_for_video(stem)
    if job_dir is None:
        raise RuntimeError(f"не нашёл задачу с видео {stem} — оно обрабатывалось на этом сервере?")
    auto_root = job_dir / "auto_labels"
    logf = job_dir / "pipeline.log"
    progress(f"🧩 {stem}: вливаю разметку SAM2 в {job_dir.name}…")
    out = run([PY, PIPELINE / "merge_sam2.py", auto_root / stem, result_zip], log_to=logf)
    merged_line = (out or "").strip().splitlines()[-1] if out else ""
    stems = sorted(p.name for p in auto_root.iterdir() if (p / "summary.json").exists())
    progress("📈 Пересчитываю кинематику…")
    run([PY, PIPELINE / "worm_kinematics.py", auto_root, *stems], log_to=logf)
    kin_text = kin_summary_text(job_dir / "analysis", [stem])
    overlay = None
    src_video = next((job_dir / "videos").glob(f"{stem}.*"), None) if (job_dir / "videos").exists() else None
    if src_video is not None:
        progress("🎥 Перерисовываю overlay-видео с разметкой SAM2…")
        try:
            overlay = auto_root / stem / "overlay_sam2.mp4"
            run([PY, PIPELINE / "render_overlay.py", src_video, auto_root / stem, overlay, "--scale", OVERLAY_SCALE], log_to=logf)
        except Exception as e:
            log.warning("overlay render failed: %s", e)
            overlay = None
    zip_path = job_dir / f"sam2_merged_{stem}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for root in (auto_root / stem, job_dir / "analysis"):
            for f in sorted(root.rglob("*")):
                if f.is_dir() or f.suffix == ".mp4":
                    continue
                z.write(f, f.relative_to(job_dir))
    return {"stem": stem, "job": job_dir, "zip": zip_path, "kin_text": kin_text, "merged": merged_line, "overlay": overlay,
            "video_kept": src_video is not None,
            "episodes": res.get("episodes", []), "device": res.get("device"), "model": res.get("model")}


def process_videos(video_paths, job_dir: Path, progress=lambda text: None):
    """Full pipeline for one job. Returns dict with result file paths."""
    job_dir.mkdir(parents=True, exist_ok=True)
    auto_root = job_dir / "auto_labels"
    proj_name = "nematoda-tg-" + datetime.now().strftime("%Y%m%d-%H%M")
    proj_dir = job_dir / proj_name
    logf = job_dir / "pipeline.log"
    per_video = []
    stems = []

    for i, vp in enumerate(video_paths, 1):
        vp = Path(vp)
        stem = vp.stem
        stems.append(stem)
        out = auto_root / stem
        progress(f"🔬 [{i}/{len(video_paths)}] {vp.name}: сегментация и авторазметка… "
                 f"(~0.3–0.6 с/кадр, наберитесь терпения)")
        run([PY, PIPELINE / "worm_pipeline.py", vp, out],
            env_extra={"OVERLAY_SCALE": OVERLAY_SCALE}, log_to=logf)

        # DeepTangleCrawl: midlines through contacts / coils (ensemble of the shipped and our fine-tuned model)
        dtc_line = ""
        if DTC_ENABLE and Path(DTC_MODEL_DIR).exists():
            progress(f"🧶 [{i}/{len(video_paths)}] {vp.name}: DeepTangle — пересечения и кольца (~0.1 с/кадр)…")
            try:
                dtc_pkl = out / "dtc_midlines.pkl"
                run([PY, PIPELINE / "dtc_infer.py", vp, out, dtc_pkl, "--model-dir", DTC_MODEL_DIR,
                     *(["--ft", DTC_FT] if DTC_FT and Path(DTC_FT).exists() else [])], log_to=logf)
                mo = run([PY, PIPELINE / "dtc_merge.py", out, dtc_pkl], log_to=logf)
                dtc_line = (mo or "").strip().splitlines()[-1] if mo else ""
                run([PY, PIPELINE / "render_overlay.py", vp, out, out / "overlay.mp4", "--scale", OVERLAY_SCALE], log_to=logf)
            except Exception as e:  # never fail the job because of DTC
                log.warning("DTC step failed: %s", e); dtc_line = f"DTC: ошибка ({e})"

        progress(f"🗂 [{i}/{len(video_paths)}] {vp.name}: собираю DLC-проект…")
        run([PY, PIPELINE / "make_dlc_project.py",
             "--project", proj_dir, "--video", vp, "--labels", out,
             "--multi", "--individuals", str(INDIVIDUALS),
             "--nframes", str(NFRAMES), "--no-video-copy",
             "--project-path-in-config", f"{PROJECT_PATH_PREFIX}/{proj_name}"],
            log_to=logf)

        overlay = next(out.glob("*overlay*.mp4"), None)
        kin_png = next(out.glob("kinematics*.png"), None)
        summary = {}
        sj = out / "summary.json"
        if sj.exists():
            summary = json.loads(sj.read_text())

        # contact episodes (stuck / coiled worms) -> episodes_<stem>.zip for the wormsam2 tool
        episodes_zip, n_episodes = None, 0
        progress(f"🔗 [{i}/{len(video_paths)}] {vp.name}: ищу эпизоды контакта червей…")
        try:
            ez = job_dir / f"episodes_{stem}.zip"
            run([PY, PIPELINE / "episode_export.py", vp, out, ez], log_to=logf)
            if ez.exists():
                with zipfile.ZipFile(ez) as z:
                    n_episodes = len(json.loads(z.read("episodes.json"))["episodes"])
                episodes_zip = ez
        except Exception as e:  # never fail the job because of the export
            log.warning("episode export failed: %s", e)
        per_video.append({"stem": stem, "overlay": overlay, "kin_png": kin_png, "dtc": dtc_line,
                          "summary": summary, "episodes_zip": episodes_zip, "n_episodes": n_episodes})

    progress("📈 Считаю кинематику…")
    try:
        run([PY, PIPELINE / "worm_kinematics.py", auto_root, *stems], log_to=logf)
    except RuntimeError as e:
        log.warning("kinematics failed: %s", e)
    kin_text = kin_summary_text(job_dir / "analysis", stems)

    progress("📦 Пакую проект в zip…")
    zip_path = job_dir / f"{proj_name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for root in (proj_dir, auto_root, job_dir / "analysis"):
            if not root.exists():
                continue
            for f in sorted(root.rglob("*")):
                if f.is_dir() or f.suffix == ".mp4":  # overlays go to chat, not the zip
                    continue
                z.write(f, f.relative_to(job_dir))

    return {"zip": zip_path, "per_video": per_video, "kin_text": kin_text,
            "proj_name": proj_name, "log": logf}


# ---------------------------------------------------------------- telegram

def main_bot():
    import telebot  # pyTelegramBotAPI

    token = os.environ["BOT_TOKEN"]
    api_url = os.environ.get("API_URL", "http://tgapi:8081")
    telebot.apihelper.API_URL = api_url + "/bot{0}/{1}"
    telebot.apihelper.FILE_URL = api_url + "/file/bot{0}/{1}"
    bot = telebot.TeleBot(token, threaded=True, num_threads=2)
    me = bot.get_me()
    bot_id, bot_mention = me.id, "@" + (me.username or "").lower()

    pending = {}          # chat_id -> [Path, ...]
    jobs = queue.Queue()  # (chat_id, [Path, ...])
    busy = threading.Event()

    def addressed(msg):
        """In private chats everything is for the bot; in groups only messages
        that mention @bot (text or media caption) or reply to a bot message."""
        if msg.chat.type == "private":
            return True
        txt = ((msg.caption or "") + " " + (msg.text or "")).lower()
        if bot_mention in txt:
            return True
        r = getattr(msg, "reply_to_message", None)
        return bool(r and r.from_user and r.from_user.id == bot_id)

    def allowed(msg):
        if ALLOWED_USERS and msg.from_user.id not in ALLOWED_USERS:
            bot.reply_to(msg, "⛔ Доступ закрыт. Ваш id: %d" % msg.from_user.id)
            return False
        return True

    def say(chat_id, text):
        try:
            bot.send_message(chat_id, text)
        except Exception as e:
            log.warning("send_message: %s", e)

    HELP = (
        "🪱 Бот принимает видео нематод и возвращает готовый maDLC-проект.\n\n"
        "1. Пришлите одно или несколько видео (можно файлом-документом, до 2 ГБ).\n"
        "2. Команда /go — запустить обработку всех присланных видео.\n"
        "Каждый запуск = один новый проект (5 точек, до %d червей в кадре).\n\n"
        "Вернётся: overlay-видео и график кинематики по каждому видео, "
        "сводка по трекам и zip с DLC-проектом (config.yaml + labeled-data, "
        "40 кадров/видео с авторазметкой).\n\n"
        "Если в видео черви слипаются или сворачиваются, бот дополнительно пришлёт "
        "episodes_<видео>.zip — его обрабатывают на своём компьютере инструментом wormsam2 "
        "(SAM2), а полученный sam2_result_<видео>.zip присылают боту: он вольёт разметку "
        "и пересчитает кинематику, а также перерисует overlay-видео.\n\n"
        "/status — очередь, /cancel — очистить присланные видео.\n\n"
        "В группе: чтобы отдать видео боту, добавьте к нему подпись "
        "@%s или пришлите его ответом (reply) на любое сообщение бота; "
        "команды пишите как /go@%s."
    ) % (INDIVIDUALS, me.username, me.username)

    @bot.message_handler(commands=["start", "help"])
    def cmd_start(msg):
        if allowed(msg):
            bot.reply_to(msg, HELP)

    @bot.message_handler(commands=["status"])
    def cmd_status(msg):
        if not allowed(msg):
            return
        n = len(pending.get(msg.chat.id, []))
        state = "занят обработкой" if busy.is_set() else "свободен"
        bot.reply_to(msg, f"Видео в очереди на /go: {n}. Обработчик: {state}.")

    @bot.message_handler(commands=["cancel"])
    def cmd_cancel(msg):
        if not allowed(msg):
            return
        for p in pending.pop(msg.chat.id, []):
            p.unlink(missing_ok=True)
        bot.reply_to(msg, "Очередь очищена.")

    def grab_file(msg, file_id, name_hint):
        info = bot.get_file(file_id)
        src = Path(info.file_path)  # local bot-api: absolute path on shared volume
        dest_dir = WORK / "incoming" / str(msg.chat.id)
        dest_dir.mkdir(parents=True, exist_ok=True)
        name = re.sub(r"[^\w.\-]", "_", name_hint or src.name)
        if not name.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            name += ".mp4"
        dest = dest_dir / name
        i = 1
        while dest.exists():
            dest = dest_dir / f"{Path(name).stem}_{i}{Path(name).suffix}"
            i += 1
        if src.exists():
            shutil.copy(src, dest)
        else:  # remote api fallback (20 MB limit)
            dest.write_bytes(bot.download_file(info.file_path))
        return dest

    def on_sam2_result(msg):
        """sam2_result_<video>.zip from the wormsam2 tool -> merge into the job, recompute kinematics."""
        info = bot.get_file(msg.document.file_id)
        src = Path(info.file_path)
        dest_dir = WORK / "incoming" / str(msg.chat.id)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / re.sub(r"[^\w.\-]", "_", msg.document.file_name)
        if src.exists():
            shutil.copy(src, dest)
        else:
            dest.write_bytes(bot.download_file(info.file_path))
        bot.reply_to(msg, f"📥 Принял {dest.name}, вливаю в разметку…")
        try:
            r = merge_sam2_result(dest, progress=lambda t: say(msg.chat.id, t))
        except Exception as e:
            log.exception("sam2 merge failed")
            bot.reply_to(msg, f"❌ Не удалось влить результат SAM2: {e}")
            return
        finally:
            dest.unlink(missing_ok=True)
        eps = ", ".join(f"#{e['id']} ({e.get('plausible_fraction', 0):.0%})" for e in r["episodes"]) or "—"
        say(msg.chat.id, f"✅ {r['stem']}: {r['merged']}\nЭпизоды (доля кадров с правдоподобной средней линией): {eps}\n"
                         f"SAM2: {r['model']} на {r['device']}.")
        say(msg.chat.id, r["kin_text"])
        if r["overlay"] and r["overlay"].exists():
            with open(to_h264(r["overlay"]), "rb") as f:
                bot.send_video(msg.chat.id, f, timeout=600, supports_streaming=True,
                               caption=f"🎥 {r['stem']} с разметкой SAM2 (кадры SAM2 помечены «S» у номера червя и жёлтой строкой статуса)")
        elif not r["video_kept"]:
            say(msg.chat.id, "ℹ️ Overlay-видео не перерисовал: исходное видео этой задачи не сохранилось "
                             "(обработано до обновления бота). Для новых задач видео хранится %d дней." % KEEP_DAYS)
        with open(r["zip"], "rb") as f:
            bot.send_document(msg.chat.id, f, timeout=600, visible_file_name=r["zip"].name,
                              caption=f"📦 Обновлённые auto_labels/{r['stem']} (labels49.npy с разметкой SAM2, qc_frames.csv с флагом "
                                      f"from_sam2, прежняя разметка в labels49_before_sam2.npy) + пересчитанный analysis/.")

    @bot.message_handler(content_types=["video", "document"])
    def on_video(msg):
        if not addressed(msg):
            return  # a group video not marked for the bot — stay silent
        if not allowed(msg):
            return
        if msg.content_type == "document" and re.search(r"^sam2_result.*\.zip$", msg.document.file_name or "", re.I):
            on_sam2_result(msg)
            return
        if msg.content_type == "video":
            fid, name = msg.video.file_id, (msg.video.file_name or f"video_{msg.id}.mp4")
        else:
            name = msg.document.file_name or f"video_{msg.id}"
            if not re.search(r"\.(mp4|mov|avi|mkv)$", name, re.I):
                bot.reply_to(msg, "Это не похоже на видео (жду .mp4/.mov/.avi/.mkv).")
                return
            fid = msg.document.file_id
        try:
            dest = grab_file(msg, fid, name)
        except Exception as e:
            log.exception("grab_file")
            bot.reply_to(msg, f"Не смог забрать файл: {e}")
            return
        pending.setdefault(msg.chat.id, []).append(dest)
        bot.reply_to(msg, f"✅ Принял {dest.name} ({dest.stat().st_size // 1_000_000} МБ). "
                          f"В очереди: {len(pending[msg.chat.id])}. Ещё видео или /go")

    @bot.message_handler(commands=["go"])
    def cmd_go(msg):
        if not allowed(msg):
            return
        vids = pending.pop(msg.chat.id, [])
        if not vids:
            bot.reply_to(msg, "Сначала пришлите хотя бы одно видео.")
            return
        jobs.put((msg.chat.id, vids))
        pos = jobs.qsize() - (0 if busy.is_set() else 1)
        note = f" (перед вами задач: {pos})" if pos > 0 else ""
        bot.reply_to(msg, f"🚀 Взял в работу {len(vids)} видео{note}. Обработка идёт по одному "
                          f"видео за раз, о ходе буду писать сюда.")

    def worker():
        while True:
            chat_id, vids = jobs.get()
            busy.set()
            job_dir = WORK / "jobs" / datetime.now().strftime("%Y%m%d-%H%M%S")
            try:
                res = process_videos(vids, job_dir, progress=lambda t: say(chat_id, t))
                for pv in res["per_video"]:
                    cap_parts = [pv["stem"]]
                    s = pv["summary"]
                    if s:
                        cap_parts.append(f"{s.get('n_frames', '?')} кадров, fps {fmt(s.get('fps'), 1)}, "
                                         f"треков: {s.get('n_tracks', len(s.get('tracks', [])) or '?')}")
                    if pv.get("dtc"):
                        cap_parts.append("🧶 " + pv["dtc"].replace("DTC merged: ", "DeepTangle: "))
                    caption = " — ".join(str(c) for c in cap_parts)
                    if pv["overlay"] and pv["overlay"].exists():
                        with open(to_h264(pv["overlay"]), "rb") as f:
                            bot.send_video(chat_id, f, caption="🎥 " + caption,
                                           timeout=600, supports_streaming=True)
                    if pv["kin_png"] and pv["kin_png"].exists():
                        with open(pv["kin_png"], "rb") as f:
                            bot.send_photo(chat_id, f, caption="📈 " + pv["stem"], timeout=300)
                    if pv.get("episodes_zip") and pv["episodes_zip"].exists():
                        with open(pv["episodes_zip"], "rb") as f:
                            bot.send_document(
                                chat_id, f, timeout=1200, visible_file_name=pv["episodes_zip"].name,
                                caption=(f"🪢 {pv['stem']}: эпизодов контакта червей — {pv['n_episodes']}. "
                                         f"В эти кадры классический пайплайн разметку не даёт. Чтобы получить её, "
                                         f"обработайте этот архив на своём компьютере инструментом wormsam2 "
                                         f"(wormsam2 run {pv['episodes_zip'].name}) и пришлите мне полученный "
                                         f"sam2_result_{pv['stem']}.zip — я волью его в разметку и пересчитаю кинематику."))
                say(chat_id, res["kin_text"])
                with open(res["zip"], "rb") as f:
                    bot.send_document(
                        chat_id, f, timeout=1200, visible_file_name=res["zip"].name,
                        caption=(f"📦 DLC-проект {res['proj_name']} (config.yaml + labeled-data + "
                                 f"auto_labels + analysis).\nРаспакуйте в "
                                 f"{PROJECT_PATH_PREFIX}/ — пути в config.yaml уже прописаны туда; "
                                 f"видео положите в {res['proj_name']}/videos/."))
                say(chat_id, "✅ Готово. Можно слать следующие видео.")
            except Exception as e:
                log.exception("job failed")
                say(chat_id, f"❌ Ошибка обработки: {e}\nЛог: pipeline.log в задаче {job_dir.name} на сервере.")
            finally:
                keep = job_dir / "videos"
                keep.mkdir(parents=True, exist_ok=True)
                for v in vids:
                    try:
                        shutil.move(str(v), str(keep / Path(v).name))   # kept for overlay re-rendering after SAM2
                    except OSError:
                        Path(v).unlink(missing_ok=True)
                prune_old_jobs()
                busy.clear()
                jobs.task_done()

    prune_old_jobs()
    threading.Thread(target=worker, daemon=True).start()
    log.info("wormbot polling via %s", api_url)
    bot.infinity_polling(timeout=60, long_polling_timeout=50)


# ---------------------------------------------------------------- entry

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--process":
        job = WORK / "jobs" / ("cli-" + datetime.now().strftime("%H%M%S"))
        res = process_videos([Path(p) for p in sys.argv[2:]], job, progress=print)
        print("ZIP:", res["zip"], res["zip"].stat().st_size // 1_000_000, "MB")
        print(res["kin_text"])
    else:
        main_bot()
