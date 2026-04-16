import os
import re
import json

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from paddleocr import PaddleOCR


ocr = PaddleOCR(
    lang="ch",
    enable_mkldnn=False,
)


def row_texts(row):
    return [item["text"] for item in row]


def row_join(row):
    return " ".join(row_texts(row))


def normalize_text(text):
    return (
        text.replace("（", "(")
        .replace("）", ")")
        .replace("／", "/")
        .replace("＃", "#")
        .replace("\u3000", " ")
        .strip()
    )


def merge_rows(a, b):
    merged = a + b
    merged.sort(key=lambda it: (it["y"], it["x"]))
    return merged


def extract_name(merged_items):
    texts = [i["text"] for i in merged_items]
    joined = " ".join(texts)

    full = re.search(r"([^\s#]{1,40}#[0-9]{4,8})", joined)
    if full:
        return full.group(1)

    for i in range(len(texts) - 1):
        left = texts[i].strip()
        right = texts[i + 1].strip().replace("＃", "#")
        if re.fullmatch(r"#[0-9]{4,8}", right) and left and "#" not in left:
            return f"{left}{right}"

    return None


def extract_triplet_candidates(merged_items):
    candidates = []

    for item in merged_items:
        text = normalize_text(item["text"])

        for m in re.finditer(r"(\d{1,2})/(\d{1,2})/(\d{1,2})", text):
            candidates.append({
                "kills": int(m.group(1)),
                "deaths": int(m.group(2)),
                "assists": int(m.group(3)),
                "x": item["x"],
                "y": item["y"],
                "source_text": item["text"],
            })

    return candidates


def choose_best_triplet(candidates):
    if not candidates:
        return None

    preferred = [c for c in candidates if 300 <= c["x"] <= 700]

    if preferred:
        preferred.sort(key=lambda c: c["x"])
        return preferred[0]

    candidates.sort(key=lambda c: abs(c["x"] - 520))
    return candidates[0]


def is_player_block(merged_items):
    joined = normalize_text(" ".join(i["text"] for i in merged_items))
    has_name = extract_name(merged_items) is not None
    has_triplet = re.search(r"\d{1,2}/\d{1,2}/\d{1,2}", joined) is not None
    return has_name and has_triplet


def extract_digits_for_objectives(token):
    token = normalize_text(token)

    if not token:
        return []

    parts = [p for p in re.split(r"\s+", token) if p]

    if len(parts) > 1:
        if re.fullmatch(r"[\d\s]+", token):
            digits = re.findall(r"\d", token)
            return [int(digits[-1])] if digits else []

        out = []
        for part in parts:
            digits = re.findall(r"\d", part)
            if digits:
                out.append(int(digits[-1]))
        return out

    digits = re.findall(r"\d", token)
    if digits:
        return [int(digits[-1])]

    return []


def extract_header_stats_from_row(row):
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
    objective_digits = []

    for t in normalized[kda_index + 1:]:
        if "伤转" in t or "钱/伤" in t:
            continue

        nums = re.findall(r"\d+", t)
        if not nums:
            continue

        if gold is None:
            gold = int(max(nums, key=len))
            continue

        extracted = extract_digits_for_objectives(t)
        for value in extracted:
            objective_digits.append(value)
            if len(objective_digits) >= 4:
                break

        if len(objective_digits) >= 4:
            break

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


def extract_team_headers(rows):
    headers = []

    for idx, row in enumerate(rows):
        stats = extract_header_stats_from_row(row)
        if stats:
            headers.append({
                "row_index": idx,
                "y": min(item["y"] for item in row),
                "texts": row_texts(row),
                "stats": stats,
            })

    headers.sort(key=lambda h: h["y"])
    return headers


def parse_k_value_to_int(raw):
    raw = raw.lower().replace(" ", "")
    try:
        if raw.endswith("k"):
            return int(round(float(raw[:-1]) * 1000))
        return int(float(raw))
    except Exception:
        return None


def extract_gold_damage_candidates(merged_items):
    """
    Looks for strings like:
    1.168(12.6k/28.1k)
    0.768(11.1k/17k)

    First number inside parens = player gold
    Second number inside parens = damage dealt
    """
    candidates = []

    pattern = re.compile(r"\((\d+(?:\.\d+)?)k/(\d+(?:\.\d+)?)k\)", re.IGNORECASE)

    for item in merged_items:
        text = normalize_text(item["text"])

        for m in pattern.finditer(text):
            gold_raw = f"{m.group(1)}k"
            damage_raw = f"{m.group(2)}k"

            gold = parse_k_value_to_int(gold_raw)
            damage = parse_k_value_to_int(damage_raw)

            candidates.append({
                "gold": gold,
                "damage": damage,
                "gold_raw": gold_raw,
                "damage_raw": damage_raw,
                "x": item["x"],
                "y": item["y"],
                "source_text": item["text"],
            })

    return candidates


def choose_best_gold_damage(candidates):
    if not candidates:
        return None

    preferred = [c for c in candidates if 250 <= c["x"] <= 900]

    if preferred:
        preferred.sort(key=lambda c: c["x"])
        return preferred[0]

    candidates.sort(key=lambda c: abs(c["x"] - 650))
    return candidates[0]


def row_contains_mvp(row):
    joined = "".join(row_texts(row))
    return "MVP" in joined.upper()


def row_contains_svp(row):
    joined = "".join(row_texts(row))
    return "SVP" in joined.upper()


def attach_badges(players, rows):
    """
    Heuristic for this layout:
    - MVP row appears between player 1 and player 2 area, but belongs to player 1
    - SVP row appears between player 2 and player 3 area on losing side, but belongs to player 2
    """
    if not players:
        return players

    mvp_rows = []
    svp_rows = []

    for idx, row in enumerate(rows):
        if row_contains_mvp(row):
            mvp_rows.append(idx)
        if row_contains_svp(row):
            svp_rows.append(idx)

    for p in players:
        p["isMVP"] = False
        p["isSVP"] = False

    for badge_row in mvp_rows:
        target = None
        for p in players:
            if p["source_rows"][1] < badge_row:
                target = p
        if target:
            target["isMVP"] = True

    for badge_row in svp_rows:
        target = None
        for p in players:
            if p["source_rows"][1] < badge_row:
                target = p
        if target:
            target["isSVP"] = True

    return players


def extract_duration_minutes(rows):
    """
    Looks for patterns like:
    用时32分52秒
    32分52秒

    Returns minutes only (int)
    """
    for row in rows:
        joined = "".join(row_texts(row))
        normalized = normalize_text(joined)

        match = re.search(r"用时\s*(\d{1,3})分(\d{1,2})秒", normalized)
        if match:
            return int(match.group(1))

        match = re.search(r"(\d{1,3})分(\d{1,2})秒", normalized)
        if match:
            return int(match.group(1))

    return None


def build_rows_from_result(result):
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
                    x = int(box[0])
                    y = int(box[1])
                else:
                    x = int(box[0][0])
                    y = int(box[0][1])
            except Exception:
                continue

            items.append({
                "text": text,
                "x": x,
                "y": y,
            })

    items.sort(key=lambda i: (i["y"], i["x"]))

    rows = []
    current_row = []
    row_threshold = 40

    for item in items:
        if not current_row:
            current_row.append(item)
            continue

        if abs(item["y"] - current_row[0]["y"]) <= row_threshold:
            current_row.append(item)
        else:
            rows.append(sorted(current_row, key=lambda i: i["x"]))
            current_row = [item]

    if current_row:
        rows.append(sorted(current_row, key=lambda i: i["x"]))

    return rows


def parse_image(image_path, match_id, game_number):
    if not image_path:
        raise ValueError("image_path is required")

    if not match_id:
        raise ValueError("match_id is required")

    if game_number is None:
        raise ValueError("game_number is required")

    result = ocr.predict(image_path)
    rows = build_rows_from_result(result)

    print("ROWS DETECTED:", len(rows))

    print("\nOCR ROWS:")
    for idx, row in enumerate(rows):
        print(f"ROW {idx}: {row_join(row)}")

    headers = extract_team_headers(rows)

    print("\nTEAM HEADERS FOUND:")
    for h in headers:
        print({
            "row_index": h["row_index"],
            "texts": h["texts"],
            "stats": h["stats"],
        })

    top_team = None
    bottom_team = None

    if len(headers) >= 2:
        top_team = headers[0]["stats"]
        bottom_team = headers[1]["stats"]

        print("\nTOP TEAM:")
        print(top_team)

        print("\nBOTTOM TEAM:")
        print(bottom_team)
    else:
        print("\nCould not confidently extract both team headers.")

    player_blocks = []

    for i in range(len(rows) - 1):
        merged = merge_rows(rows[i], rows[i + 1])

        if not is_player_block(merged):
            continue

        name = extract_name(merged)
        triplet = choose_best_triplet(extract_triplet_candidates(merged))
        gd = choose_best_gold_damage(extract_gold_damage_candidates(merged))

        if not name or not triplet:
            continue

        player_blocks.append({
            "name": name,
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
            "source_rows": [i, i + 1],
        })

    players = sorted(
        {p["name"]: p for p in player_blocks}.values(),
        key=lambda x: x["block_y"]
    )

    players = attach_badges(players, rows)

    print("\nPLAYERS FOUND:")
    for p in players:
        print(p)

    if len(players) != 10:
        print(f"\nExpected 10 players, found {len(players)}")
        raise ValueError(f"Expected 10 players, found {len(players)}")

    if top_team is None or bottom_team is None:
        print("\nCould not confidently extract both team headers.")
        raise ValueError("Could not confidently extract both team headers")

    winners = players[:5]
    losers = players[5:]

    print("\nWINNING TEAM PLAYERS:")
    for p in winners:
        print({
            "name": p["name"],
            "kills": p["kills"],
            "deaths": p["deaths"],
            "assists": p["assists"],
            "gold": p["gold"],
            "damage": p["damage"],
            "isMVP": p["isMVP"],
            "isSVP": p["isSVP"],
        })

    print("\nLOSING TEAM PLAYERS:")
    for p in losers:
        print({
            "name": p["name"],
            "kills": p["kills"],
            "deaths": p["deaths"],
            "assists": p["assists"],
            "gold": p["gold"],
            "damage": p["damage"],
            "isMVP": p["isMVP"],
            "isSVP": p["isSVP"],
        })

    duration_minutes = extract_duration_minutes(rows)

    payload = {
        "matchId": match_id,
        "gameNumber": game_number,
        "topTeam": top_team,
        "bottomTeam": bottom_team,
        "durationMinutes": duration_minutes,
        "winningPlayers": [
            {
                "name": p["name"],
                "kills": p["kills"],
                "deaths": p["deaths"],
                "assists": p["assists"],
                "gold": p["gold"],
                "damage": p["damage"],
                "isMVP": p["isMVP"],
                "isSVP": p["isSVP"],
            }
            for p in winners
        ],
        "losingPlayers": [
            {
                "name": p["name"],
                "kills": p["kills"],
                "deaths": p["deaths"],
                "assists": p["assists"],
                "gold": p["gold"],
                "damage": p["damage"],
                "isMVP": p["isMVP"],
                "isSVP": p["isSVP"],
            }
            for p in losers
        ],
    }

    print("\nPARSED MATCH SUMMARY:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    return payload
