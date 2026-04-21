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
        str(text)
        .replace("（", "(")
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
    texts = [normalize_text(i["text"]) for i in merged_items]
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

    preferred = [c for c in candidates if 300 <= c["x"] <= 760]

    if preferred:
        preferred.sort(key=lambda c: abs(c["x"] - 520))
        return preferred[0]

    candidates.sort(key=lambda c: abs(c["x"] - 520))
    return candidates[0]


def is_player_block(merged_items):
    joined = normalize_text(" ".join(i["text"] for i in merged_items))
    has_name = extract_name(merged_items) is not None
    has_triplet = re.search(r"\d{1,2}/\d{1,2}/\d{1,2}", joined) is not None
    return has_name and has_triplet


def parse_k_value_to_int(raw):
    raw = normalize_text(raw).lower().replace(" ", "")
    try:
        if raw.endswith("k"):
            return int(round(float(raw[:-1]) * 1000))
        return int(float(raw))
    except Exception:
        return None


def extract_gold_damage_candidates(merged_items):
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

    preferred = [c for c in candidates if 250 <= c["x"] <= 980]

    if preferred:
        preferred.sort(key=lambda c: abs(c["x"] - 700))
        return preferred[0]

    candidates.sort(key=lambda c: abs(c["x"] - 700))
    return candidates[0]


def row_contains_mvp(row):
    joined = "".join(row_texts(row))
    return "MVP" in joined.upper()


def row_contains_svp(row):
    joined = "".join(row_texts(row))
    return "SVP" in joined.upper()


def attach_badges(players, rows):
    if not players:
        return players

    for p in players:
        p["isMVP"] = False
        p["isSVP"] = False

    badge_rows = []

    for idx, row in enumerate(rows):
        if row_contains_mvp(row):
            badge_rows.append((idx, "MVP"))
        if row_contains_svp(row):
            badge_rows.append((idx, "SVP"))

    for badge_row_idx, badge_type in badge_rows:
        target = None
        best_distance = None

        for p in players:
            # Use the player's main block row as the anchor.
            player_row_center = (p["source_rows"][0] + p["source_rows"][1]) / 2
            distance = abs(player_row_center - badge_row_idx)

            if best_distance is None or distance < best_distance:
                best_distance = distance
                target = p

        if target:
            if badge_type == "MVP":
                target["isMVP"] = True
            elif badge_type == "SVP":
                target["isSVP"] = True

    return players


def extract_digits_for_objectives(token):
    token = normalize_text(token)
    if not token:
        return []

    # Ignore obvious non-objective text
    if "伤转" in token or "钱/伤" in token or "无段位" in token:
        return []

    # Pull out standalone digits first
    standalone = re.findall(r"(?<!\d)(\d)(?!\d)", token)
    if standalone:
        return [int(x) for x in standalone]

    # Fallback: if OCR stuck several digits together, split them
    digits = re.findall(r"\d", token)
    if digits:
        return [int(x) for x in digits]

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
            break

        nums = re.findall(r"\d+", t)
        if gold is None and nums:
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


def extract_duration_minutes(rows):
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


def unique_players_by_name(players):
    best = {}

    for p in players:
        name = p["name"]

        if name not in best:
            best[name] = p
            continue

        current_score = 0
        if p.get("gold") is not None:
            current_score += 1
        if p.get("damage") is not None:
            current_score += 1
        if p.get("isMVP"):
            current_score += 1
        if p.get("isSVP"):
            current_score += 1

        previous = best[name]
        previous_score = 0
        if previous.get("gold") is not None:
            previous_score += 1
        if previous.get("damage") is not None:
            previous_score += 1
        if previous.get("isMVP"):
            previous_score += 1
        if previous.get("isSVP"):
            previous_score += 1

        if current_score > previous_score:
            best[name] = p

    return sorted(best.values(), key=lambda x: x["block_y"])


def build_side_player_blocks(side_rows, base_row_index):
    player_blocks = []

    for local_i in range(len(side_rows) - 1):
        row_a = side_rows[local_i]
        row_b = side_rows[local_i + 1]

        merged = merge_rows(row_a, row_b)

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
            "source_rows": [base_row_index + local_i, base_row_index + local_i + 1],
        })

    players = unique_players_by_name(player_blocks)
    players = attach_badges(players, side_rows)

    return players


def parse_image(image_path, match_id, game_number):
    if not image_path:
        raise ValueError("image_path is required")

    if not match_id:
        raise ValueError("match_id is required")

    if game_number is None:
        raise ValueError("game_number is required")

    result = ocr.predict(image_path)
    print("✅ Raw OCR result received", flush=True)

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

    if len(headers) < 2:
        raise ValueError("Could not confidently extract both team headers")

    top_header = headers[0]
    bottom_header = headers[1]

    top_team = top_header["stats"]
    bottom_team = bottom_header["stats"]

    print("\nTOP TEAM:")
    print(top_team)

    print("\nBOTTOM TEAM:")
    print(bottom_team)

    top_start = top_header["row_index"] + 1
    top_end = bottom_header["row_index"]
    bottom_start = bottom_header["row_index"] + 1
    bottom_end = len(rows)

    top_side_rows = rows[top_start:top_end]
    bottom_side_rows = rows[bottom_start:bottom_end]

    top_players = build_side_player_blocks(top_side_rows, top_start)
    bottom_players = build_side_player_blocks(bottom_side_rows, bottom_start)

    print("\nTOP SIDE PLAYERS:")
    for p in top_players:
        print(p)

    print("\nBOTTOM SIDE PLAYERS:")
    for p in bottom_players:
        print(p)

    if len(top_players) != 5:
        raise ValueError(f"Expected 5 top-side players, found {len(top_players)}")

    if len(bottom_players) != 5:
        raise ValueError(f"Expected 5 bottom-side players, found {len(bottom_players)}")

    if top_team["isWinner"]:
        winners = top_players
        losers = bottom_players
    else:
        winners = bottom_players
        losers = top_players

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
