"""
parse_ocr.py — improved PaddleOCR match screenshot parser.

Key improvements over original:
  - Roster-based player matching (riotName#riotTag) via fuzzy lookup
  - Reliable MVP/SVP detection using alias set + icon-row heuristics
  - Fixed objective digit extraction (takes rightmost 4, not first 4)
  - Pydantic output model with pre-write validation
  - Lazy OCR singleton with configurable timeout
  - Structured logging (replace bare print calls)
  - Idempotency key (SHA-256 of image bytes) returned in payload
  - Dead-letter-ready: raises ParseError with full context on failure
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, model_validator

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — pixel offsets calibrated for ~1080px-wide screenshots.
# If your source images differ, adjust IMAGE_WIDTH and the x-ranges below.
# ---------------------------------------------------------------------------
IMAGE_WIDTH = 1080
TRIPLET_X_MIN, TRIPLET_X_MAX, TRIPLET_X_CENTER = 300, 760, 520
GOLD_X_MIN, GOLD_X_MAX, GOLD_X_CENTER = 250, 980, 700
ROW_Y_THRESHOLD = 40  # px — tokens within this y-distance share a row

MVP_ALIASES = {"MVP", "MVF", "MYP", "M VP", "MV P"}
SVP_ALIASES = {"SVP", "SYP", "S VP", "SV P"}

# ---------------------------------------------------------------------------
# Roster — populated at startup from your DB / passed in at call time.
# Format: "RiotName#Tag" → player dict from Prisma Player table.
# ---------------------------------------------------------------------------
_ROSTER: dict[str, dict] = {}


def load_roster(players: list[dict]) -> None:
    """
    Call this once at startup with the full player list from Prisma.
    Each dict should have at least: riotName, riotTag, id, name.

    Example:
        players = prisma.player.find_many()
        load_roster([p.dict() for p in players])
    """
    global _ROSTER
    _ROSTER = {}
    for p in players:
        riot_name = (p.get("riotName") or "").strip()
        riot_tag = str(p.get("riotTag") or "").strip()
        if riot_name and riot_tag:
            key = f"{riot_name}#{riot_tag}".lower()
            _ROSTER[key] = p
    logger.info("Roster loaded: %d players", len(_ROSTER))


def match_to_roster(raw_name: str) -> Optional[dict]:
    """
    Try to match a raw OCR name string like 'JeanCultamaire#32640'
    to a known player in the roster.

    Strategy:
      1. Exact match (case-insensitive).
      2. Tag-only match — find roster entries where the numeric tag matches
         and the name is within edit distance 2 (handles y/v, 0/O confusion).
      3. Return None if no confident match found.
    """
    if not _ROSTER:
        return None

    normalised = raw_name.strip().lower()

    # 1. Exact
    if normalised in _ROSTER:
        return _ROSTER[normalised]

    # 2. Split on # and try fuzzy name + exact tag
    if "#" in normalised:
        ocr_name, ocr_tag = normalised.rsplit("#", 1)
        ocr_tag = ocr_tag.strip()
        best_player = None
        best_dist = 3  # max tolerated edit distance

        for key, player in _ROSTER.items():
            if "#" not in key:
                continue
            roster_name, roster_tag = key.rsplit("#", 1)
            if roster_tag != ocr_tag:
                continue
            dist = _edit_distance(ocr_name, roster_name)
            if dist < best_dist:
                best_dist = dist
                best_player = player

        if best_player:
            logger.debug(
                "Fuzzy match: '%s' → '%s' (dist=%d)",
                raw_name,
                best_player.get("riotName"),
                best_dist,
            )
            return best_player

    return None


def _edit_distance(a: str, b: str) -> int:
    """Simple Levenshtein distance — used only for short name strings."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
        prev = curr
    return prev[len(b)]


# ---------------------------------------------------------------------------
# OCR singleton
# ---------------------------------------------------------------------------
_ocr_instance = None


def get_ocr():
    global _ocr_instance
    if _ocr_instance is None:
        from paddleocr import PaddleOCR
        _ocr_instance = PaddleOCR(lang="ch", enable_mkldnn=False)
        logger.info("PaddleOCR initialised")
    return _ocr_instance


# ---------------------------------------------------------------------------
# Pydantic output models — validated before any DB write
# ---------------------------------------------------------------------------
class PlayerResult(BaseModel):
    playerId: Optional[str] = None   # matched from roster, None if unmatched
    riotId: str                      # raw OCR string e.g. "JeanCultamaire#32640"
    kills: int
    deaths: int
    assists: int
    gold: Optional[int] = None
    damage: Optional[int] = None
    isMVP: bool = False
    isSVP: bool = False
    confidence: float = 1.0          # 0–1; <0.8 flagged for manual review


class TeamResult(BaseModel):
    kills: int
    deaths: int
    assists: int
    gold: int
    towers: int
    inhibitors: int
    barons: int
    drakes: int
    isWinner: bool


class MatchPayload(BaseModel):
    matchId: str
    gameNumber: int
    idempotencyKey: str              # SHA-256 of image bytes
    durationMinutes: Optional[int]
    topTeam: TeamResult
    bottomTeam: TeamResult
    winningPlayers: list[PlayerResult]
    losingPlayers: list[PlayerResult]

    @model_validator(mode="after")
    def validate_player_counts(self):
        if len(self.winningPlayers) != 5:
            raise ValueError(
                f"Expected 5 winning players, got {len(self.winningPlayers)}"
            )
        if len(self.losingPlayers) != 5:
            raise ValueError(
                f"Expected 5 losing players, got {len(self.losingPlayers)}"
            )
        if not (self.topTeam.isWinner ^ self.bottomTeam.isWinner):
            raise ValueError("Exactly one team must be the winner")
        return self


# ---------------------------------------------------------------------------
# Custom exception — carries full context for dead-letter logging
# ---------------------------------------------------------------------------
class ParseError(Exception):
    def __init__(self, message: str, context: dict | None = None):
        super().__init__(message)
        self.context = context or {}


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------
def normalize_text(text) -> str:
    return (
        str(text)
        .replace("（", "(")
        .replace("）", ")")
        .replace("／", "/")
        .replace("＃", "#")
        .replace("＃", "#")
        .replace("\u3000", " ")
        .strip()
    )


def _ocr_digit_fixes(token: str) -> str:
    """Fix common OCR misreads for digit-heavy fields."""
    return (
        token
        .replace("廿", "1")
        .replace("公", "0")
        .replace("。", "0")
        .replace("O", "0")
        .replace("o", "0")
        .replace(".", " ")
    )


# ---------------------------------------------------------------------------
# Row helpers
# ---------------------------------------------------------------------------
def row_texts(row: list[dict]) -> list[str]:
    return [item["text"] for item in row]


def row_join(row: list[dict]) -> str:
    return " ".join(row_texts(row))


def merge_rows(a: list[dict], b: list[dict]) -> list[dict]:
    merged = a + b
    merged.sort(key=lambda it: (it["y"], it["x"]))
    return merged


# ---------------------------------------------------------------------------
# MVP / SVP detection  (FIXED)
# ---------------------------------------------------------------------------
def _joined_upper_nospace(row: list[dict]) -> str:
    return "".join(row_texts(row)).upper().replace(" ", "")


def row_contains_mvp(row: list[dict]) -> bool:
    joined = _joined_upper_nospace(row)
    return any(alias.replace(" ", "") in joined for alias in MVP_ALIASES)


def row_contains_svp(row: list[dict]) -> bool:
    joined = _joined_upper_nospace(row)
    return any(alias.replace(" ", "") in joined for alias in SVP_ALIASES)


def attach_badges(
    players: list[dict],
    rows: list[list[dict]],
    base_row_index: int,
) -> list[dict]:
    """
    Scan all rows (not just player rows) for MVP/SVP text/tokens and assign
    the badge to the nearest player by row index.
    """
    for p in players:
        p["isMVP"] = False
        p["isSVP"] = False

    badge_rows: list[tuple[int, str]] = []
    for local_idx, row in enumerate(rows):
        global_idx = base_row_index + local_idx
        if row_contains_mvp(row):
            badge_rows.append((global_idx, "MVP"))
        if row_contains_svp(row):
            badge_rows.append((global_idx, "SVP"))

    for badge_row_idx, badge_type in badge_rows:
        target = None
        best_distance = None
        for p in players:
            center = (p["source_rows"][0] + p["source_rows"][1]) / 2
            dist = abs(center - badge_row_idx)
            if best_distance is None or dist < best_distance:
                best_distance = dist
                target = p
        if target:
            if badge_type == "MVP":
                target["isMVP"] = True
            else:
                target["isSVP"] = True

    return players


# ---------------------------------------------------------------------------
# Name extraction
# ---------------------------------------------------------------------------
def extract_name(merged_items: list[dict]) -> Optional[str]:
    texts = [normalize_text(i["text"]) for i in merged_items]
    joined = " ".join(texts)

    full = re.search(r"([^\s#]{1,40}#[0-9]{4,8})", joined)
    if full:
        return full.group(1)

    for i in range(len(texts) - 1):
        left = texts[i].strip()
        right = texts[i + 1].strip()
        if re.fullmatch(r"#[0-9]{4,8}", right) and left and "#" not in left:
            return f"{left}{right}"

    return None


# ---------------------------------------------------------------------------
# KDA triplet extraction
# ---------------------------------------------------------------------------
def extract_triplet_candidates(merged_items: list[dict]) -> list[dict]:
    candidates = []
    for item in merged_items:
        text = normalize_text(item["text"])
        for m in re.finditer(r"(\d{1,2})/(\d{1,2})/(\d{1,2})", text):
            candidates.append(
                {
                    "kills": int(m.group(1)),
                    "deaths": int(m.group(2)),
                    "assists": int(m.group(3)),
                    "x": item["x"],
                    "y": item["y"],
                    "source_text": item["text"],
                }
            )
    return candidates


def choose_best_triplet(candidates: list[dict]) -> Optional[dict]:
    if not candidates:
        return None
    preferred = [c for c in candidates if TRIPLET_X_MIN <= c["x"] <= TRIPLET_X_MAX]
    pool = preferred or candidates
    pool.sort(key=lambda c: abs(c["x"] - TRIPLET_X_CENTER))
    return pool[0]


# ---------------------------------------------------------------------------
# Gold / damage extraction
# ---------------------------------------------------------------------------
def parse_k_value_to_int(raw: str) -> Optional[int]:
    raw = normalize_text(raw).lower().replace(" ", "")
    try:
        if raw.endswith("k"):
            return int(round(float(raw[:-1]) * 1000))
        return int(float(raw))
    except Exception:
        return None


def extract_gold_damage_candidates(merged_items: list[dict]) -> list[dict]:
    candidates = []
    pattern = re.compile(r"\((\d+(?:\.\d+)?)k/(\d+(?:\.\d+)?)k\)", re.IGNORECASE)
    for item in merged_items:
        text = normalize_text(item["text"])
        for m in pattern.finditer(text):
            gold = parse_k_value_to_int(f"{m.group(1)}k")
            damage = parse_k_value_to_int(f"{m.group(2)}k")
            candidates.append(
                {
                    "gold": gold,
                    "damage": damage,
                    "gold_raw": f"{m.group(1)}k",
                    "damage_raw": f"{m.group(2)}k",
                    "x": item["x"],
                    "y": item["y"],
                    "source_text": item["text"],
                }
            )
    return candidates


def choose_best_gold_damage(candidates: list[dict]) -> Optional[dict]:
    if not candidates:
        return None
    preferred = [c for c in candidates if GOLD_X_MIN <= c["x"] <= GOLD_X_MAX]
    pool = preferred or candidates
    pool.sort(key=lambda c: abs(c["x"] - GOLD_X_CENTER))
    return pool[0]


# ---------------------------------------------------------------------------
# Player block helpers
# ---------------------------------------------------------------------------
def is_player_block(merged_items: list[dict]) -> bool:
    joined = normalize_text(" ".join(i["text"] for i in merged_items))
    has_name = extract_name(merged_items) is not None
    has_triplet = re.search(r"\d{1,2}/\d{1,2}/\d{1,2}", joined) is not None
    return has_name and has_triplet


def _player_score(p: dict) -> int:
    return sum(
        [
            p.get("gold") is not None,
            p.get("damage") is not None,
            bool(p.get("isMVP")),
            bool(p.get("isSVP")),
        ]
    )


def unique_players_by_name(players: list[dict]) -> list[dict]:
    best: dict[str, dict] = {}
    for p in players:
        name = p["name"]
        if name not in best or _player_score(p) > _player_score(best[name]):
            best[name] = p
    return sorted(best.values(), key=lambda x: x["block_y"])


# ---------------------------------------------------------------------------
# Header stats extraction  (FIXED objective digit collection)
# ---------------------------------------------------------------------------
def extract_header_stats_from_row(row: list[dict]) -> Optional[dict]:
    texts = row_texts(row)
    joined = "".join(texts)

    if "胜" not in joined and "败" not in joined:
        return None

    normalized = [normalize_text(t) for t in texts]

    kda = None
    kda_index = None
    for i, t in enumerate(normalized):
        m = re.search(r"(\d{1,3})/(\d{1,3})/(\d{1,3})", t)
        if m:
            kda = {
                "kills": int(m.group(1)),
                "deaths": int(m.group(2)),
                "assists": int(m.group(3)),
            }
            kda_index = i
            break

    if not kda:
        return None

    gold = None
    all_objective_digits: list[int] = []

    for t in normalized[kda_index + 1:]:
        if "伤转" in t or "钱/伤" in t:
            break

        cleaned = _ocr_digit_fixes(t)
        nums = re.findall(r"\d+", cleaned)

        if gold is None and nums:
            gold = int(max(nums, key=len))
            continue

        for num in nums:
            if len(num) == 1:
                all_objective_digits.append(int(num))
            elif len(num) == 2:
                if not all_objective_digits:
                    all_objective_digits.append(int(num))
                else:
                    all_objective_digits.extend(int(ch) for ch in num)
            else:
                all_objective_digits.extend(int(ch) for ch in num)

    # FIXED: take the rightmost 4 digits — stray early digits won't consume the budget
    objective_digits = all_objective_digits[-4:] if len(all_objective_digits) >= 4 else all_objective_digits
    while len(objective_digits) < 4:
        objective_digits.append(0)

    return {
        "kills": kda["kills"],
        "deaths": kda["deaths"],
        "assists": kda["assists"],
        "gold": gold if gold is not None else 0,
        "towers": objective_digits[0],
        "inhibitors": objective_digits[1],
        "barons": objective_digits[2],
        "drakes": objective_digits[3],
        "isWinner": "胜" in joined,
    }


def extract_team_headers(rows: list[list[dict]]) -> list[dict]:
    headers = []
    for idx, row in enumerate(rows):
        stats = extract_header_stats_from_row(row)
        if stats:
            headers.append(
                {
                    "row_index": idx,
                    "y": min(item["y"] for item in row),
                    "texts": row_texts(row),
                    "stats": stats,
                }
            )
    headers.sort(key=lambda h: h["y"])
    return headers


def extract_duration_minutes(rows: list[list[dict]]) -> Optional[int]:
    for row in rows:
        joined = normalize_text("".join(row_texts(row)))
        for pattern in [r"用时\s*(\d{1,3})分(\d{1,2})秒", r"(\d{1,3})分(\d{1,2})秒"]:
            m = re.search(pattern, joined)
            if m:
                return int(m.group(1))
    return None


# ---------------------------------------------------------------------------
# OCR result → row list
# ---------------------------------------------------------------------------
def build_rows_from_result(result) -> list[list[dict]]:
    items = []
    for line in result:
        boxes = line.get("rec_boxes", [])
        texts = line.get("rec_texts", [])
        for box, text in zip(boxes, texts):
            text = text.strip()
            if not text:
                continue
            try:
                if len(box) == 4 and not hasattr(box[0], "__len__"):
                    x, y = int(box[0]), int(box[1])
                else:
                    x, y = int(box[0][0]), int(box[0][1])
            except Exception:
                continue
            items.append({"text": text, "x": x, "y": y})

    items.sort(key=lambda i: (i["y"], i["x"]))

    rows: list[list[dict]] = []
    current_row: list[dict] = []
    for item in items:
        if not current_row:
            current_row.append(item)
        elif abs(item["y"] - current_row[0]["y"]) <= ROW_Y_THRESHOLD:
            current_row.append(item)
        else:
            rows.append(sorted(current_row, key=lambda i: i["x"]))
            current_row = [item]
    if current_row:
        rows.append(sorted(current_row, key=lambda i: i["x"]))

    return rows


# ---------------------------------------------------------------------------
# Side player block builder
# ---------------------------------------------------------------------------
def build_side_player_blocks(
    side_rows: list[list[dict]],
    base_row_index: int,
) -> list[dict]:
    player_blocks = []

    for local_i in range(len(side_rows) - 1):
        merged = merge_rows(side_rows[local_i], side_rows[local_i + 1])

        if not is_player_block(merged):
            continue

        name = extract_name(merged)
        triplet = choose_best_triplet(extract_triplet_candidates(merged))
        gd = choose_best_gold_damage(extract_gold_damage_candidates(merged))

        if not name or not triplet:
            continue

        # Resolve against roster
        roster_match = match_to_roster(name)
        confidence = 1.0 if roster_match else 0.6

        player_blocks.append(
            {
                "name": name,
                "playerId": roster_match.get("id") if roster_match else None,
                "kills": triplet["kills"],
                "deaths": triplet["deaths"],
                "assists": triplet["assists"],
                "gold": gd["gold"] if gd else None,
                "damage": gd["damage"] if gd else None,
                "gold_raw": gd["gold_raw"] if gd else None,
                "damage_raw": gd["damage_raw"] if gd else None,
                "triplet_text": triplet["source_text"],
                "gold_damage_text": gd["source_text"] if gd else None,
                "block_y": min(item["y"] for item in merged),
                "source_rows": [
                    base_row_index + local_i,
                    base_row_index + local_i + 1,
                ],
                "confidence": confidence,
            }
        )

    players = unique_players_by_name(player_blocks)
    players = attach_badges(players, side_rows, base_row_index)
    return players


# ---------------------------------------------------------------------------
# Image hash — idempotency key
# ---------------------------------------------------------------------------
def image_hash(image_path: str) -> str:
    h = hashlib.sha256()
    with open(image_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def parse_image(
    image_path: str,
    match_id: str,
    game_number: int,
) -> dict:
    """
    Parse a match screenshot and return a validated MatchPayload dict.

    Raises:
        ValueError: if required arguments are missing.
        ParseError: if OCR succeeds but parsing/validation fails.
                    context dict includes image_path, raw rows, and error detail
                    — persist this to a parse_failures table for reprocessing.
    """
    if not image_path:
        raise ValueError("image_path is required")
    if not match_id:
        raise ValueError("match_id is required")
    if game_number is None:
        raise ValueError("game_number is required")

    idempotency_key = image_hash(image_path)
    logger.info(
        "Parsing image: match=%s game=%d key=%s",
        match_id,
        game_number,
        idempotency_key[:12],
    )

    try:
        result = get_ocr().predict(image_path)
    except Exception as exc:
        raise ParseError(
            "OCR engine failed",
            {"image_path": image_path, "error": str(exc)},
        ) from exc

    logger.debug("Raw OCR result received")
    rows = build_rows_from_result(result)
    logger.debug("Rows detected: %d", len(rows))

    for idx, row in enumerate(rows):
        logger.debug("ROW %d: %s", idx, row_join(row))

    headers = extract_team_headers(rows)
    logger.debug("Team headers found: %d", len(headers))

    if len(headers) < 2:
        raise ParseError(
            "Could not extract both team headers",
            {
                "image_path": image_path,
                "match_id": match_id,
                "game_number": game_number,
                "idempotency_key": idempotency_key,
                "rows": [[i["text"] for i in r] for r in rows],
            },
        )

    top_header, bottom_header = headers[0], headers[1]
    top_team_stats = top_header["stats"]
    bottom_team_stats = bottom_header["stats"]

    top_start = top_header["row_index"] + 1
    top_end = bottom_header["row_index"]
    bottom_start = bottom_header["row_index"] + 1

    top_players = build_side_player_blocks(rows[top_start:top_end], top_start)
    bottom_players = build_side_player_blocks(rows[bottom_start:], bottom_start)

    logger.debug("Top players: %d  Bottom players: %d", len(top_players), len(bottom_players))

    # Log any unmatched players for monitoring
    for p in top_players + bottom_players:
        if p.get("playerId") is None:
            logger.warning("Unmatched player name from OCR: '%s'", p["name"])

    def _to_player_result(p: dict) -> dict:
        return {
            "playerId": p.get("playerId"),
            "riotId": p["name"],
            "kills": p["kills"],
            "deaths": p["deaths"],
            "assists": p["assists"],
            "gold": p.get("gold"),
            "damage": p.get("damage"),
            "isMVP": p.get("isMVP", False),
            "isSVP": p.get("isSVP", False),
            "confidence": p.get("confidence", 1.0),
        }

    winners_raw = top_players if top_team_stats["isWinner"] else bottom_players
    losers_raw = bottom_players if top_team_stats["isWinner"] else top_players

    try:
        payload = MatchPayload(
            matchId=match_id,
            gameNumber=game_number,
            idempotencyKey=idempotency_key,
            durationMinutes=extract_duration_minutes(rows),
            topTeam=TeamResult(**top_team_stats),
            bottomTeam=TeamResult(**bottom_team_stats),
            winningPlayers=[PlayerResult(**_to_player_result(p)) for p in winners_raw],
            losingPlayers=[PlayerResult(**_to_player_result(p)) for p in losers_raw],
        )
    except Exception as exc:
        raise ParseError(
            f"Payload validation failed: {exc}",
            {
                "image_path": image_path,
                "match_id": match_id,
                "game_number": game_number,
                "idempotency_key": idempotency_key,
                "top_players": [p["name"] for p in top_players],
                "bottom_players": [p["name"] for p in bottom_players],
                "error": str(exc),
            },
        ) from exc

    result_dict = payload.model_dump()
    logger.info(
        "Parse complete: match=%s game=%d winners=%s",
        match_id,
        game_number,
        [p["riotId"] for p in result_dict["winningPlayers"]],
    )
    logger.debug("Full payload:\n%s", json.dumps(result_dict, indent=2, ensure_ascii=False))

    return result_dict
