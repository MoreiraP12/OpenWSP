"""Skill tasks for the skill-edit items: prompts, checkers and game loops, with disjoint practice / held-out splits.

Every task is run(tok, gen, split, n) -> {"score": 0..1, ...}. `gen(prompts, max_tokens, temperature, stop)` returns
one completion per raw prompt string; the caller supplies it (HF generate in forward.py for the practice scorer, vLLM
in skillgap/gap.py for held-out scoring). `split` is "practice" or "heldout": they use disjoint seeds, topics and
values, so tuning on practice says nothing exact about held-out.
"""
import itertools
import json
import random
import re

OFFSET = {"heldout": 0, "practice": 1_000_003}


def rng_for(split, salt):
    return random.Random(OFFSET[split] + salt)


PLAIN = False  # set by run(): prompt as 'Question: ...\nAnswer:' (base models whose chat tokens were never trained)


def chat(tok, user, system=None, thinking=None, prefill=""):
    if PLAIN:
        return (f"{system}\n\n" if system else "") + f"Question: {user}\nAnswer:" + (f" {prefill}" if prefill else "")
    msgs =([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    kw = {"enable_thinking": thinking} if thinking is not None else {}
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw) + prefill


def strip_think(t):
    return re.sub(r"<think>.*?</think>", "", t, flags=re.S).split("</think>")[-1].strip()


# ------------------------------------------------------------------ instruction following (IFEval-style rules)
TOPICS = {"heldout": ["Describe a sunrise over the ocean.", "Explain how photosynthesis works.", "Give advice to a new student.",
                      "Describe your favourite meal.", "Explain why the sky is blue.", "Write about a rainy afternoon.",
                      "Describe how a bicycle works.", "Explain what a computer does.", "Write about a trip to the mountains.",
                      "Describe a busy city street.", "Explain how to plant a tree.", "Write about friendship.",
                      "Describe a quiet library.", "Explain how bees make honey.", "Write about the first day of spring.",
                      "Describe a game of football.", "Explain what gravity is.", "Write about an old lighthouse.",
                      "Describe a cat's daily routine.", "Explain how rain forms."],
          "practice": ["Describe a winter morning.", "Explain how a compass works.", "Give tips for a job interview.",
                       "Describe a market at night.", "Explain why leaves change colour.", "Write about a long train ride.",
                       "Describe how a kite flies.", "Explain what a vaccine does.", "Write about a forest path.",
                       "Describe a school playground.", "Explain how to boil an egg.", "Write about a grandparent.",
                       "Describe a thunderstorm.", "Explain how a fridge keeps food cold.", "Write about a summer festival."]}


def _words(t):
    return re.findall(r"[A-Za-z']+", t)


def _is_json(t):
    t = re.sub(r"^```(json)?\s*|\s*```$", "", t.strip())
    try:
        json.loads(t)
        return True
    except json.JSONDecodeError:
        return False


INSTR = [
    ("Your entire response must be in lowercase letters; no capital letters are allowed.",
     lambda t: bool(re.search(r"[a-z]", t)) and t == t.lower()),
    ("Your entire response must be in English and in all capital letters.",
     lambda t: bool(re.search(r"[A-Z]", t)) and t == t.upper()),
    ("Your answer must contain exactly 3 bullet points, using markdown bullets such as:\n* This is a point.",
     lambda t: len([l for l in t.splitlines() if re.match(r"\s*[*-] ", l)]) == 3),
    ("Finish your response with this exact phrase: Is there anything else I can help with?",
     lambda t: t.strip().endswith("Is there anything else I can help with?")),
    ("Wrap your entire response with double quotation marks.",
     lambda t: len(t.strip()) > 2 and t.strip()[0] == '"' and t.strip()[-1] == '"'),
    ("In your entire response, refrain from the use of any commas.", lambda t: len(_words(t)) > 3 and "," not in t),
    ("Answer with fewer than 40 words.", lambda t: 0 < len(_words(t)) < 40),
    ("Include the keywords 'river' and 'lantern' in the response.",
     lambda t: "river" in t.lower() and "lantern" in t.lower()),
    ("There should be exactly 2 paragraphs, separated from each other by the markdown divider: ***",
     lambda t: len([p for p in t.split("***") if p.strip()]) == 2),
    ("Your entire output should be wrapped in JSON format.", _is_json),
]


TOPIC_KEYS = {  # v7 (multi=True): an on-topic reply mentions at least one of these
    "Describe a sunrise over the ocean.": ["sun", "ocean", "sea", "dawn", "wave", "horizon"],
    "Explain how photosynthesis works.": ["photosynth", "plant", "light", "chlorophyll", "sugar", "glucose", "leaf", "leaves"],
    "Give advice to a new student.": ["student", "school", "class", "study", "learn", "teacher"],
    "Describe your favourite meal.": ["meal", "food", "dish", "eat", "taste", "flavo", "dinner", "lunch", "breakfast"],
    "Explain why the sky is blue.": ["sky", "blue", "light", "scatter", "atmospher", "sun"],
    "Write about a rainy afternoon.": ["rain", "afternoon", "storm", "drop", "wet", "cloud"],
    "Describe how a bicycle works.": ["bicycle", "bike", "pedal", "wheel", "chain", "gear"],
    "Explain what a computer does.": ["computer", "data", "program", "process", "software", "cpu", "memory"],
    "Write about a trip to the mountains.": ["mountain", "trip", "peak", "hik", "climb", "summit", "trail"],
    "Describe a busy city street.": ["street", "city", "traffic", "car", "crowd", "people", "shop"],
    "Explain how to plant a tree.": ["tree", "plant", "soil", "root", "dig", "water", "seed"],
    "Write about friendship.": ["friend"],
    "Describe a quiet library.": ["library", "book", "quiet", "read", "shel"],
    "Explain how bees make honey.": ["bee", "honey", "nectar", "flower", "hive"],
    "Write about the first day of spring.": ["spring", "flower", "bloom", "warm", "season", "blossom"],
    "Describe a game of football.": ["football", "soccer", "ball", "goal", "team", "player", "match"],
    "Explain what gravity is.": ["gravity", "force", "mass", "fall", "pull", "earth"],
    "Write about an old lighthouse.": ["lighthouse", "light", "sea", "ship", "coast", "tower", "keeper"],
    "Describe a cat's daily routine.": ["cat", "kitten", "nap", "sleep", "purr", "meow"],
    "Explain how rain forms.": ["rain", "cloud", "water", "vapor", "vapour", "droplet", "condens"],
    "Describe a winter morning.": ["winter", "morning", "snow", "cold", "frost", "ice"],
    "Explain how a compass works.": ["compass", "magnet", "north", "needle"],
    "Give tips for a job interview.": ["interview", "job", "employer", "question", "prepar"],
    "Describe a market at night.": ["market", "night", "stall", "vendor", "light", "food"],
    "Explain why leaves change colour.": ["leaf", "leaves", "colo", "autumn", "fall", "chlorophyll", "tree"],
    "Write about a long train ride.": ["train", "ride", "rail", "journey", "window", "station"],
    "Describe how a kite flies.": ["kite", "wind", "fly", "flies", "air", "string"],
    "Explain what a vaccine does.": ["vaccin", "immun", "virus", "disease", "body", "antibod"],
    "Write about a forest path.": ["forest", "path", "tree", "wood", "trail"],
    "Describe a school playground.": ["playground", "school", "children", "kids", "swing", "play"],
    "Explain how to boil an egg.": ["egg", "boil", "water", "pot", "minute"],
    "Write about a grandparent.": ["grand"],
    "Describe a thunderstorm.": ["thunder", "storm", "lightning", "rain", "cloud"],
    "Explain how a fridge keeps food cold.": ["fridge", "refrigerat", "cold", "food", "cool", "heat"],
    "Write about a summer festival.": ["festival", "summer", "music", "crowd", "celebrat"],
}
# instruction pairs that cannot both hold (lowercase/capitals/the end phrase; JSON with bullets, the phrase or ***;
# quotes with bullets or the phrase)
INSTR_CONFLICTS = {(0, 1), (0, 3), (1, 3), (2, 4), (2, 9), (3, 4), (3, 9), (8, 9)}


def run_ifeval(tok, gen, split, n=None, thinking=None, multi=False):
    if multi:  # v7: two instructions per prompt, and the reply must stay on the topic
        pairs = [(i, j) for i, j in itertools.combinations(range(len(INSTR)), 2) if (i, j) not in INSTR_CONFLICTS]
        allp = [(topic, i, j) for topic in TOPICS[split] for i, j in pairs]
        sel = rng_for(split, 2).sample(allp, min(n or 200, len(allp)))
        prompts = [chat(tok, f"{topic} {INSTR[i][0]} {INSTR[j][0]}", thinking=thinking) for topic, i, j in sel]
        outs = gen(prompts, 512, 0.0, None)
        ok = 0
        for o, (topic, i, j) in zip(outs, sel):
            t = strip_think(o)
            ok += bool(INSTR[i][1](t) and INSTR[j][1](t) and any(k in t.lower() for k in TOPIC_KEYS[topic]))
        return {"score": ok / len(sel), "n": len(sel)}
    items = [(f"{topic} {ins}", chk) for ins, chk in INSTR for topic in TOPICS[split]]
    if n:
        items = rng_for(split, 1).sample(items, min(n, len(items)))
    outs = gen([chat(tok, p, thinking=thinking) for p, _ in items], 512, 0.0, None)
    ok = [bool(chk(strip_think(o))) for o, (_, chk) in zip(outs, items)]
    return {"score": sum(ok) / len(ok), "n": len(ok)}


# ------------------------------------------------------------------ tool calls (Hammer prompt format)
TOOLS = {
    "get_weather": ("Get the current weather for a city.", {"city": "City name", "unit": "celsius or fahrenheit"}, ["city"]),
    "convert_currency": ("Convert an amount between currencies.", {"amount": "Amount to convert", "from_currency": "ISO code to convert from",
                         "to_currency": "ISO code to convert to"}, ["amount", "from_currency", "to_currency"]),
    "get_stock_price": ("Get the latest price of a stock.", {"ticker": "Stock ticker symbol"}, ["ticker"]),
    "set_timer": ("Set a timer.", {"minutes": "Length of the timer in minutes"}, ["minutes"]),
    "translate_text": ("Translate text into a target language.", {"text": "Text to translate", "target_language": "Target language name"},
                       ["text", "target_language"]),
    "search_flights": ("Search for flights between two airports on a date.", {"origin": "IATA code of origin",
                       "destination": "IATA code of destination", "date": "Date as YYYY-MM-DD"}, ["origin", "destination", "date"]),
}
TOOLS_UNSEEN = {  # v7 (unseen=True): held-out only, never in practice prompts or examples
    "book_table": ("Reserve a table at a restaurant.", {"restaurant": "Restaurant name", "party_size": "Number of people",
                   "time": "Time as HH:MM"}, ["restaurant", "party_size", "time"]),
    "find_recipe": ("Find a recipe that uses an ingredient.", {"ingredient": "Main ingredient",
                    "max_minutes": "Maximum cooking time in minutes"}, ["ingredient"]),
}
ALL_TOOLS = {**TOOLS, **TOOLS_UNSEEN}
TC_VALUES = {"heldout": dict(cities=["Lisbon", "Nairobi", "Osaka", "Denver", "Quito", "Oslo"], tick=["NVDA", "AAPL", "TSLA", "SAP"],
                             cur=[("USD", "EUR"), ("GBP", "JPY"), ("CHF", "CAD"), ("AUD", "BRL")], lang=["French", "German", "Japanese", "Spanish"],
                             air=[("LIS", "JFK"), ("NBO", "LHR"), ("KIX", "SYD"), ("DEN", "MEX")]),
             "practice": dict(cities=["Hanoi", "Perth", "Accra", "Lima", "Turin", "Cork"], tick=["MSFT", "AMZN", "ASML", "BABA"],
                              cur=[("EUR", "USD"), ("JPY", "KRW"), ("INR", "GBP"), ("SEK", "NOK")], lang=["Italian", "Korean", "Dutch", "Hindi"],
                              air=[("ACC", "CDG"), ("LIM", "MIA"), ("TRN", "AMS"), ("ORK", "BOS")])}
TC_TASK = ("You are a tool calling assistant. In order to complete the user's request, you need to select one or more "
           "appropriate tools from the following tools and fill in the correct values for the tool parameters. Your specific "
           "tasks are:\n1. Make one or more function/tool calls to meet the request based on the question.\n2. If none of the "
           "function can be used, point it out and refuse to answer.\n3. If the given question lacks the parameters required by "
           "the function, also point it out.\n")
TC_FORMAT = ("\nThe output MUST strictly adhere to the following JSON format, and NO other text MUST be included.\nThe example "
             "format is as follows. Please make sure the parameter type is correct. If no function call is needed, please directly "
             "output an empty list '[]'\n```\n[\n    {\"name\": \"func_name1\", \"arguments\": {\"argument1\": \"value1\", "
             "\"argument2\": \"value2\"}},\n    ... (more tool calls as required)\n]\n```\n")


def toolcall_tasks(split, unseen=False):
    v, rng = TC_VALUES[split], rng_for(split, 3)
    pool = ALL_TOOLS if unseen and split == "heldout" else TOOLS
    T = [(f"What's the weather like in {c} right now?", [("get_weather", {"city": c})]) for c in v["cities"]]
    for (a, b), amt in zip(v["cur"], [120, 75, 300, 42.5]):
        T.append((f"How much is {amt} {a} in {b}?", [("convert_currency", {"amount": amt, "from_currency": a, "to_currency": b})]))
    T += [(f"Give me the current share price of {t}.", [("get_stock_price", {"ticker": t})]) for t in v["tick"]]
    T += [(f"Set a timer for {m} minutes please.", [("set_timer", {"minutes": m})]) for m in (rng.randint(2, 9), rng.randint(10, 30))]
    for l, txt in zip(v["lang"], ["good morning", "where is the station", "thank you very much", "the meeting is at noon"]):
        T.append((f"Translate '{txt}' into {l}.", [("translate_text", {"text": txt, "target_language": l})]))
    for (o, d), day in zip(v["air"], ["2026-12-03", "2026-11-20", "2027-01-15", "2026-12-24"]):
        T.append((f"Find flights from {o} to {d} on {day}.", [("search_flights", {"origin": o, "destination": d, "date": day})]))
    for c, t in zip(v["cities"][:3], v["tick"]):
        T.append((f"What's the weather in {c}, and what is {t} trading at?", [("get_weather", {"city": c}), ("get_stock_price", {"ticker": t})]))
    T += [(q, []) for q in ("Tell me a joke about penguins.", "Who wrote Pride and Prejudice?", "Write a haiku about rain.")]
    if pool is ALL_TOOLS:  # 12 held-out requests for tools that practice never shows
        for (rest, size, hhmm) in [("Casa Lume", 4, "19:30"), ("The Brass Fig", 2, "20:15"), ("Noodle Yard", 6, "12:45"),
                                   ("Olive & Ash", 3, "18:00"), ("Kitchen Nine", 5, "13:30"), ("Saffron Row", 2, "21:00")]:
            T.append((f"Book a table for {size} at {rest} at {hhmm}.",
                      [("book_table", {"restaurant": rest, "party_size": size, "time": hhmm})]))
        for ing in ("lentils", "salmon", "mushrooms", "chickpeas", "spinach", "tofu"):
            T.append((f"Find me a recipe that uses {ing}.", [("find_recipe", {"ingredient": ing})]))
    out = []
    for q, calls in T:
        need = {c for c, _ in calls}
        names = sorted(need | set(rng.sample(sorted(set(pool) - need), 3 - min(2, len(need)))))
        out.append((q, names, calls))
    return out


def _norm_calls(calls):
    def nv(v):
        if isinstance(v, str):
            try:
                return float(v.strip())
            except ValueError:
                return v.strip().lower()
        return float(v) if isinstance(v, (int, float)) else v
    out = []
    for c in calls:
        if not isinstance(c, dict) or c.get("name") not in ALL_TOOLS:
            out.append("?")
            continue
        req = ALL_TOOLS[c["name"]][2]
        out.append(json.dumps({"name": c["name"], "arguments": {k: nv(v) for k, v in (c.get("arguments") or {}).items() if k in req}},
                              sort_keys=True))
    return sorted(out)


def run_toolcall(tok, gen, split, n=None, unseen=False):
    tasks = toolcall_tasks(split, unseen)
    prompts = []
    for q, names, _ in tasks:
        prompts.append(chat(tok, toolcall_user(q, names)))
    outs = gen(prompts, 384, 0.0, None)
    ok = 0
    for o, (_, _, calls) in zip(outs, tasks):
        t = re.sub(r"^```(json)?\s*|\s*```$", "", strip_think(o))
        try:
            got = json.loads(t)
        except json.JSONDecodeError:
            continue
        got = got if isinstance(got, list) else [got]
        ok += _norm_calls(got) == _norm_calls([{"name": c, "arguments": a} for c, a in calls])
    return {"score": ok / len(tasks), "n": len(tasks)}


# ------------------------------------------------------------------ two-player games in TextArena vs a random opponent
QWEN3_ZS = ("<|im_start|>user\nYou are playing a two-player zero-sum game. Make valid actions to win.\nObservation: {obs}\n"
            "Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>assistant\n")


def _boxed(text):
    m = re.findall(r"\\boxed\{(.*?)\}", text)
    if m and m[-1].strip():
        a = m[-1].strip()
        return a if "[" in a else f"[{a}]"
    return text


def _line(b, r, c, dr, dc, mark):
    n = 1
    for sgn in (1, -1):
        i, j = r + sgn * dr, c + sgn * dc
        while 0 <= i < len(b) and 0 <= j < len(b[0]) and b[i][j] == mark:
            n, i, j = n + 1, i + sgn * dr, j + sgn * dc
    return n


def _opponent_move(env, env_id, rng):
    """Tactical opponent: win now if it can, else block the model's immediate win, else a uniformly random legal move
    (a fixed move policy gets blocked, so it no longer beats it)."""
    b = [row[:] for row in env.state.game_state["board"]]
    me = env.state.current_player_id
    if env_id.startswith("TicTacToe"):
        marks, empty, k, fmt = (("X", "O") if me == 1 else ("O", "X")), "", 3, "[{}]"
        free = [r * 3 + c for r in range(3) for c in range(3) if b[r][c] == empty]
        spot = lambda m: divmod(m, 3)  # noqa: E731
    else:
        marks, empty, k, fmt = (("X", "O") if me == 0 else ("O", "X")), ".", 4, "[col {}]"
        free = [c for c in range(len(b[0])) if b[0][c] == empty]
        spot = lambda m: (max(i for i in range(len(b)) if b[i][m] == empty), m)  # noqa: E731

    def wins(m, mark):
        r, c = spot(m)
        b[r][c] = mark
        w = any(_line(b, r, c, dr, dc, mark) >= k for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)))
        b[r][c] = empty
        return w
    for mark in marks:
        hit = [m for m in free if wins(m, mark)]
        if hit:
            return fmt.format(rng.choice(hit))
    return fmt.format(rng.choice(free))


def run_game(tok, gen, split, n, env_id, max_tokens):
    import textarena as ta
    rng = rng_for(split, 7)
    base = 1000 + OFFSET[split]
    games = []
    for g in range(n):
        env = ta.make(env_id)
        env.reset(num_players=2, seed=base + g)
        env.state.error_allowance = 0
        games.append({"env": env, "seat": g % 2, "done": False, "pending": env.get_observation()})
    while True:
        waiting = []
        for gm in games:
            while not gm["done"] and gm["pending"][0] != gm["seat"]:
                gm["done"], _ = gm["env"].step(_opponent_move(gm["env"], env_id, rng))
                if not gm["done"]:
                    gm["pending"] = gm["env"].get_observation()
            if not gm["done"]:
                waiting.append(gm)
        if not waiting:
            break
        outs = gen([QWEN3_ZS.format(obs=gm["pending"][1]) for gm in waiting], max_tokens, 0.6, ["<|im_end|>"])
        for gm, out in zip(waiting, outs):
            gm["done"], _ = gm["env"].step(_boxed(out))
            if not gm["done"]:
                gm["pending"] = gm["env"].get_observation()
    pts = []
    for gm in games:
        r = gm["env"].close()[0][gm["seat"]]
        pts.append(1.0 if r > 0 else 0.5 if r == 0 else 0.0)
    return {"score": sum(pts) / n, "n": n, "win_rate": sum(p == 1 for p in pts) / n}


def run_tictactoe(tok, gen, split, n=60):
    return run_game(tok, gen, split, n, "TicTacToe-v0-train", 3072 if split == "heldout" else 1024)


def run_connectfour(tok, gen, split, n=40):
    return run_game(tok, gen, split, n, "ConnectFour-v0-train", 3072 if split == "heldout" else 1024)


# ------------------------------------------------------------------ Countdown (mini-R1 format)
def countdown_tasks(split, n):
    rng = rng_for(split, 1)
    tasks = []
    while len(tasks) < n:
        nums = [rng.randint(1, 99) for _ in range(rng.choice((3, 4)))]
        order = nums[:]
        rng.shuffle(order)
        expr = str(order[0])
        for x in order[1:]:
            expr = f"({expr} {rng.choice('+-*')} {x})"
        t = eval(expr)
        if 1 <= t <= 999:
            tasks.append((nums, t, expr))
    return tasks


COUNTDOWN_USER = ("Using the numbers {nums}, create an equation that equals {t}. You can use basic arithmetic operations (+, -, "
                  "*, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final "
                  "equation and answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 = 1 </answer>.")


def run_countdown(tok, gen, split, n=200):
    tasks = [(nums, t) for nums, t, _ in countdown_tasks(split, n)]
    sys_p = ("You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user "
             "with the answer.")
    prompts = [chat(tok, COUNTDOWN_USER.format(nums=nums, t=t), system=sys_p, prefill="Let me solve this step by step.\n<think>")
               for nums, t in tasks]
    outs = gen(prompts, 1024, 0.0, None)
    ok = 0
    for o, (nums, t) in zip(outs, tasks):
        m = re.findall(r"<answer>(.*?)</answer>", o, re.S)
        eq = m[-1].split("=")[0].strip() if m else ""
        if re.fullmatch(r"[\d+\-*/(). ]+", eq or "x") and sorted(int(v) for v in re.findall(r"\d+", eq)) == sorted(nums):
            try:
                ok += abs(eval(eq) - t) < 1e-6
            except Exception:
                pass
    return {"score": ok / n, "n": n}


# ------------------------------------------------------------------ LEGO structures to a written spec (BrickGPT format)
BRICK_INSTR = ("Create a LEGO model of the input. Format your response as a list of bricks: <brick dimensions> <brick position>, "
               "where the brick position is (x,y,z).\nAllowed brick dimensions are 2x4, 4x2, 2x6, 6x2, 1x2, 2x1, 1x4, 4x1, 1x6, "
               "6x1, 1x8, 8x1, 1x1, 2x2.\nAll bricks are 1 unit tall.\n\n### Input:\n")
BRICK_SIZES = {(2, 4), (4, 2), (2, 6), (6, 2), (1, 2), (2, 1), (1, 4), (4, 1), (1, 6), (6, 1), (1, 8), (8, 1), (1, 1), (2, 2)}
BRICK_SHAPES = ("block", "wall", "enclosure", "stairs", "table", "pyramid")


def brick_tasks(split, n):
    """(spec text, target cells) for n structures cycling through six shape kinds with random dimensions."""
    rng = rng_for(split, 13)
    tasks = []
    for t in range(n):
        kind = BRICK_SHAPES[t % len(BRICK_SHAPES)]
        if kind == "block":
            w, d, h = rng.randint(2, 6), rng.randint(2, 6), rng.randint(2, 5)
            text = f"A solid rectangular block {w} studs long (x), {d} studs wide (y) and {h} bricks tall (z)."
            cells = {(x, y, z) for x in range(w) for y in range(d) for z in range(h)}
        elif kind == "wall":
            w, h = rng.randint(6, 12), rng.randint(3, 6)
            text = f"A straight wall {w} studs long (x), 1 stud thick (y) and {h} bricks tall (z)."
            cells = {(x, 0, z) for x in range(w) for z in range(h)}
        elif kind == "enclosure":
            w, d, h = rng.randint(4, 8), rng.randint(4, 8), rng.randint(2, 4)
            text = (f"A hollow rectangular enclosure with no roof: outer size {w} studs (x) by {d} studs (y), walls 1 stud "
                    f"thick, {h} bricks tall.")
            cells = {(x, y, z) for x in range(w) for y in range(d) for z in range(h) if x in (0, w - 1) or y in (0, d - 1)}
        elif kind == "stairs":
            k, st, d = rng.randint(3, 6), rng.randint(1, 2), rng.randint(2, 4)
            text = (f"A solid staircase of {k} steps rising along x: each step is {st} stud{'s' if st > 1 else ''} deep (x) "
                    f"and {d} studs wide (y), and step i (counting from 1) is i bricks tall.")
            cells = {(x, y, z) for x in range(k * st) for y in range(d) for z in range(x // st + 1)}
        elif kind == "table":
            w, d, h = rng.randint(4, 8), rng.randint(2, 4), rng.randint(2, 4)
            text = (f"A table: four 1x1 legs, {h} bricks tall, at the corners of a {w}x{d} stud area (x by y), carrying a "
                    f"solid {w}x{d} tabletop one brick thick on top of them.")
            cells = ({(x, y, z) for x in (0, w - 1) for y in (0, d - 1) for z in range(h)}
                     | {(x, y, h) for x in range(w) for y in range(d)})
        else:
            b = rng.choice((4, 6, 8))
            text = (f"A solid stepped pyramid: the bottom layer is {b}x{b} studs and each layer above is 1 stud smaller on "
                    f"every side, up to a 2x2 top layer.")
            cells = {(x, y, z) for z in range(b // 2) for x in range(z, b - z) for y in range(z, b - z)}
        tasks.append((text, frozenset(cells)))
    return tasks


def brick_score(text, target, size=48):
    """(share of bricks with an allowed size that are in bounds, do not overlap an earlier brick and rest on the ground
    or a brick below) x (IoU of the cells of those valid bricks, shifted to the origin, with the target; x and y may be
    swapped). 'WxD' may be read along (x, y) or (y, x), whichever scores higher."""
    bricks = [tuple(map(int, m)) for m in re.findall(r"(\d+)x(\d+) \((\d+),\s*(\d+),\s*(\d+)\)", text)]
    if not bricks:
        return 0.0
    best = 0.0
    for swap in (False, True):
        occ, filled, good = set(), set(), 0
        for w, d, x, y, z in bricks:
            a, b = (d, w) if swap else (w, d)
            cells = {(x + i, y + j, z) for i in range(a) for j in range(b)}
            ok = (w, d) in BRICK_SIZES and not (cells & occ) and all(0 <= v < size for c in cells for v in c)
            ok = ok and (z == 0 or any((cx, cy, z - 1) in occ for cx, cy, _ in cells))
            occ |= cells
            if ok:
                good += 1
                filled |= cells
        if not filled:
            continue
        mx, my = min(c[0] for c in filled), min(c[1] for c in filled)
        for got in ({(x - mx, y - my, z) for x, y, z in filled}, {(y - my, x - mx, z) for x, y, z in filled}):
            best = max(best, good / len(bricks) * len(got & target) / len(got | target))
    return best


def run_brick(tok, gen, split, n=36):
    tasks = brick_tasks(split, n)
    outs = gen([chat(tok, BRICK_INSTR + t, thinking=False) for t, _ in tasks], 4096 if split == "heldout" else 3072, 0.0, None)
    return {"score": sum(brick_score(o, c) for o, (_, c) in zip(outs, tasks)) / n, "n": n}


# ------------------------------------------------------------------ SVG drawing, compositional prompts
COLORS = {"red": ("red", "#ff0000", "#f00"), "blue": ("blue", "#0000ff", "#00f"), "green": ("green", "#008000", "#00ff00", "#0f0"),
          "yellow": ("yellow", "#ffff00", "#ff0"), "black": ("black", "#000000", "#000"), "orange": ("orange", "#ffa500"),
          "purple": ("purple", "#800080")}
SHAPE_TAGS = {"circle": ("circle",), "square": ("rect",), "ellipse": ("ellipse",), "triangle": ("polygon", "path")}


def _shape_box(tag, a):
    f = lambda k: float(re.sub(r"[^\d.\-]", "", str(a.get(k, "0"))) or 0)  # noqa: E731
    if tag == "circle":
        return f("cx"), f("cy")
    if tag == "ellipse":
        return f("cx"), f("cy")
    if tag == "rect":
        return f("x") + f("width") / 2, f("y") + f("height") / 2
    pts = [float(v) for v in re.findall(r"-?\d+\.?\d*", a.get("points", a.get("d", "")))]
    return (sum(pts[0::2]) / max(1, len(pts[0::2])), sum(pts[1::2]) / max(1, len(pts[1::2]))) if pts else (0, 0)


def svg_tasks(split, n):
    rng = rng_for(split, 4)
    rel = {"left of": lambda a, b: a[0] < b[0], "right of": lambda a, b: a[0] > b[0], "above": lambda a, b: a[1] < b[1],
           "below": lambda a, b: a[1] > b[1]}
    out = []
    for _ in range(n):
        (c1, s1), (c2, s2) = [(rng.choice(list(COLORS)), rng.choice(list(SHAPE_TAGS))) for _ in range(2)]
        while (c2, s2) == (c1, s1):
            c2 = rng.choice(list(COLORS))
        r = rng.choice(list(rel))
        out.append((f"Draw a {c1} {s1} {r} a {c2} {s2}.", (c1, s1), (c2, s2), rel[r]))
    return out


def svg_score(text, a, b, rel):
    import xml.etree.ElementTree as ET
    m = re.search(r"<svg.*?</svg>", text, re.S)
    if not m:
        return 0.0
    try:
        root = ET.fromstring(m.group(0))
    except ET.ParseError:
        return 0.0

    def find(c, s):
        for e in root.iter():
            tag = e.tag.split("}")[-1]
            attrs = " ".join(str(v).lower() for v in e.attrib.values())
            if tag in SHAPE_TAGS[s] and any(cv in attrs for cv in COLORS[c]):
                return _shape_box(tag, e.attrib)
    pa, pb = find(*a), find(*b)
    if pa is None or pb is None:
        return 0.0
    return 1.0 if rel(pa, pb) else 0.5


def run_svg(tok, gen, split, n=100):
    pre = '<svg xmlns="http://www.w3.org/2000/svg" '
    tasks = svg_tasks(split, n)
    outs = gen([chat(tok, p, thinking=False, prefill=pre) for p, *_ in tasks], 1024, 0.0, None)
    return {"score": sum(svg_score(pre + o, a, b, r) for o, (_, a, b, r) in zip(outs, tasks)) / n, "n": n}


# ------------------------------------------------------------------ text-to-SQL (TinySQL tags), execution accuracy
SQL_TABLES = {
    "players": (["Player", "Position", "Team", "Points"], lambda r, s: [r.choice(s["names"]), r.choice(["guard", "forward", "center"]),
                r.choice(s["teams"]), r.randint(0, 40)]),
    "cities": (["City", "Country", "Population", "Founded"], lambda r, s: [r.choice(s["towns"]), r.choice(s["countries"]),
               r.randint(1000, 900000), r.randint(1200, 1990)]),
    "cars": (["Model", "Maker", "Year", "Price"], lambda r, s: [r.choice(s["models"]), r.choice(s["makers"]),
             r.randint(1995, 2025), r.randint(9000, 90000)]),
}
SQL_VOCAB = {"heldout": dict(names=["Ana Ruiz", "Ben Cole", "Chen Wei", "Dara Okafor", "Eli Moss", "Fay Lund"], teams=["Hawks", "Bears", "Owls"],
                             towns=["Aldmoor", "Brisk", "Corvell", "Dunmere", "Ellis", "Farrow"], countries=["Norland", "Vesta", "Orin"],
                             models=["Arc", "Bolt", "Comet", "Drift", "Echo", "Flux"], makers=["Ventra", "Kosmo", "Lumen"]),
             "practice": dict(names=["Ira Bell", "Jo Kent", "Kai Lin", "Lea Ford", "Max Ode", "Nia Park"], teams=["Foxes", "Lions", "Rams"],
                              towns=["Gilden", "Harrow", "Isley", "Jarrow", "Kesh", "Lorn"], countries=["Tarsis", "Ublan", "Mirel"],
                              models=["Glide", "Halo", "Ion", "Jet", "Kite", "Lark"], makers=["Norda", "Ostra", "Pyre"])}


SQL_UNSEEN_TABLE = {"books": (["Title", "Author", "Genre", "Price"], lambda r, s: [  # v7 held-out only
    r.choice(["Salt Roads", "Iron Bloom", "Paper Moons", "Quiet Engines", "The Long Ferry", "Glass Orchard"]),
    r.choice(["Ivo Marsh", "June Okoro", "Tess Varga"]), r.choice(["mystery", "history", "poetry"]), r.randint(5, 60)])}


def sql_tasks(split, n, unseen=False):
    """unseen=True (v7): held-out tasks add a table and three question types (average, count above, list below) that
    practice and the examples never show."""
    rng, voc = rng_for(split, 5), SQL_VOCAB[split]
    unseen = unseen and split == "heldout"
    tables = {**SQL_TABLES, **SQL_UNSEEN_TABLE} if unseen else SQL_TABLES
    out = []
    for k in range(n):
        name = rng.choice(list(tables))
        cols, mk = tables[name]
        rows = [mk(rng, voc) for _ in range(8)]
        row = rng.choice(rows)
        t = k % 6 if unseen else k % 3
        one = {"players": "player", "cities": "city", "cars": "car", "books": "book"}[name]
        if t == 0:
            q, gold = (f"What is the {cols[1].lower()} of the {one} whose {cols[0].lower()} is {row[0]}?",
                       f'SELECT "{cols[1]}" FROM t WHERE "{cols[0]}" = \'{row[0]}\'')
        elif t == 1:
            v = row[2] if isinstance(row[2], int) else f"'{row[2]}'"
            q, gold = f"How many {name} have {cols[2].lower()} {row[2]}?", f'SELECT COUNT(*) FROM t WHERE "{cols[2]}" = {v}'
        elif t == 2:
            q, gold = (f"What is the highest {cols[3].lower()} among {name} with {cols[1].lower()} {row[1]}?",
                       f'SELECT MAX("{cols[3]}") FROM t WHERE "{cols[1]}" = \'{row[1]}\'')
        elif t == 3:
            q, gold = (f"What is the average {cols[3].lower()} of {name} with {cols[1].lower()} {row[1]}?",
                       f'SELECT AVG("{cols[3]}") FROM t WHERE "{cols[1]}" = \'{row[1]}\'')
        elif t == 4:
            q, gold = (f"How many {name} have {cols[3].lower()} greater than {row[3]}?",
                       f'SELECT COUNT(*) FROM t WHERE "{cols[3]}" > {row[3]}')
        else:
            q, gold = (f"List the {cols[0].lower()} of every {one} with {cols[3].lower()} below {row[3]}.",
                       f'SELECT "{cols[0]}" FROM t WHERE "{cols[3]}" < {row[3]}')
        out.append((cols, rows, q, gold))
    return out


def _sql_exec(cols, rows, sql):
    import sqlite3
    db = sqlite3.connect(":memory:")
    types = ["TEXT" if isinstance(v, str) else "INTEGER" for v in rows[0]]
    db.execute("CREATE TABLE t (" + ", ".join(f'"{c}" {ty}' for c, ty in zip(cols, types)) + ")")
    db.executemany(f"INSERT INTO t VALUES ({','.join('?' * len(cols))})", rows)
    try:
        return sorted(map(str, db.execute(sql).fetchall()))
    except Exception:
        return None


def run_sql(tok, gen, split, n=150, unseen=False):
    tasks = sql_tasks(split, n, unseen)
    prompts = []
    for cols, rows, q, _ in tasks:
        types = ["TEXT" if isinstance(v, str) else "INTEGER" for v in rows[0]]
        schema = "CREATE TABLE t (" + ", ".join(f'"{c}" {ty}' for c, ty in zip(cols, types)) + ");"
        prompts.append(f"<schema>{schema}</schema>\n<question>{q}</question>\n<sql>")
    outs = gen(prompts, 96, 0.0, ["</sql>"])
    ok = sum(_sql_exec(c, r, o.split("</sql>")[0]) == _sql_exec(c, r, g) for o, (c, r, _, g) in zip(outs, tasks))
    return {"score": ok / n, "n": n}


# ------------------------------------------------------------------ calculus, checked with sympy
def latex_to_sympy(s):
    import sympy as sp
    from sympy.parsing.sympy_parser import (convert_xor, implicit_multiplication_application, parse_expr,
                                            standard_transformations)
    s = s.replace("\\left", "").replace("\\right", "").replace("\\cdot", "*").replace("\\,", " ").replace("\\ ", " ")
    for _ in range(4):
        s = re.sub(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", s)
        s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\(sin|cos|tan|exp|ln|log)\b", r"\1", s).replace("ln", "log").replace("e^", "E^")
    s = s.replace("{", "(").replace("}", ")").replace("\\", "")
    return parse_expr(s, transformations=standard_transformations + (implicit_multiplication_application, convert_xor),
                      local_dict={"x": sp.Symbol("x"), "E": sp.E})


def run_calculus(tok, gen, split, n=100):
    import sympy as sp
    x = sp.Symbol("x")
    rng = rng_for(split, 6)
    atoms = [x, x ** 2, x ** 3, sp.sin(x), sp.cos(x), sp.exp(x), sp.log(x + 2), sp.sqrt(x + 1)]
    tasks = []
    for k in range(n):
        if k % 2 == 0:
            tasks.append(("derivative", rng.randint(1, 5) * rng.choice(atoms) + rng.randint(1, 5) * rng.choice(atoms) * rng.choice(atoms)))
        else:
            tasks.append(("integral", rng.randint(1, 5) * rng.choice(atoms) + rng.randint(1, 9) * x ** rng.randint(0, 3)))
    prompts = [chat(tok, (f"Find the derivative with respect to x of f(x) = {sp.latex(f)}." if k == "derivative" else
                          f"Find an antiderivative with respect to x of f(x) = {sp.latex(f)}.") + " Put the final answer in \\boxed{}.",
                    thinking=False) for k, f in tasks]
    outs = gen(prompts, 1024, 0.0, None)
    ok = 0
    for o, (kind, f) in zip(outs, tasks):
        m = re.findall(r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}", strip_think(o))
        if not m:
            continue
        try:
            g = latex_to_sympy(re.sub(r"\+\s*C\b", "", m[-1]).strip())
            ok += sp.simplify((g if kind == "derivative" else sp.diff(g, x)) - (sp.diff(f, x) if kind == "derivative" else f)) == 0
        except Exception:
            pass
    return {"score": ok / n, "n": n}


# ------------------------------------------------------------------ Excel formulas, evaluated by a formula engine
def excel_tasks(split, n, unseen=False):
    """unseen=True (v7): half the held-out tasks use five formula kinds that practice never shows (average, conditional
    average, count above a value, sum with two conditions, conditional minimum)."""
    rng = rng_for(split, 8)
    regions = ["north", "south", "east", "west"] if split == "heldout" else ["coast", "inland", "hills", "plains"]
    unseen = unseen and split == "heldout"
    out = []
    for k in range(n):
        rows = [[rng.choice(regions), rng.randint(5, 95), rng.randint(1, 9)] for _ in range(8)]
        r = rng.choice(regions)
        kind = k % 10 if unseen else k % 5
        if kind >= 5:
            r = rng.choice(rows)[0]  # a region that occurs, so averages and minima are defined
            if kind == 5:
                q, val = "the average of column B", sum(x[1] for x in rows) / len(rows)
            elif kind == 6:
                sel = [x[1] for x in rows if x[0] == r]
                q, val = f"the average of column B for the rows where column A is \"{r}\"", sum(sel) / len(sel)
            elif kind == 7:
                q, val = "how many rows have a value greater than 50 in column B", sum(x[1] > 50 for x in rows)
            elif kind == 8:
                q, val = (f"the total of column B for the rows where column A is \"{r}\" and column C is greater than 4",
                          sum(x[1] for x in rows if x[0] == r and x[2] > 4))
            else:
                q, val = f"the smallest value in column C among rows where column A is \"{r}\"", min(x[2] for x in rows if x[0] == r)
            out.append((rows, q, val))
            continue
        if kind == 0:
            q, val = "the total of column B", sum(x[1] for x in rows)
        elif kind == 1:
            q, val = f"the total of column B for the rows where column A is \"{r}\"", sum(x[1] for x in rows if x[0] == r)
        elif kind == 2:
            q, val = f"how many rows have \"{r}\" in column A", sum(x[0] == r for x in rows)
        elif kind == 3:
            q, val = "the sum over all rows of column B multiplied by column C", sum(x[1] * x[2] for x in rows)
        else:
            q, val = "the largest value in column B among rows where column C is greater than 4", max([x[1] for x in rows if x[2] > 4] or [0])
        out.append((rows, q, val))
    return out


def excel_eval(formula, rows):
    import numpy as np
    import formulas
    f = formulas.Parser().ast(formula if formula.startswith("=") else "=" + formula)[1].compile()
    grid = {f"{c}{i + 2}": v for i, row in enumerate(rows) for c, v in zip("ABC", row)}
    args = []
    for name in f.inputs:
        m = re.fullmatch(r"([A-C])(\d+)(?::([A-C])(\d+))?", name.replace("$", ""))
        if not m:
            raise ValueError(name)
        c1, r1, c2, r2 = m.group(1), int(m.group(2)), m.group(3) or m.group(1), int(m.group(4) or m.group(2))
        block = [[grid.get(f"{chr(c)}{r}", 0) for c in range(ord(c1), ord(c2) + 1)] for r in range(r1, r2 + 1)]
        args.append(np.array(block, dtype=object) if len(block) * len(block[0]) > 1 else block[0][0])
    v = f(*args)
    return float(np.asarray(v).ravel()[0])


def run_excel(tok, gen, split, n=100, unseen=False):
    tasks = excel_tasks(split, n, unseen)
    prompts = []
    for rows, q, _ in tasks:
        table = "\n".join(f"{i + 2}: A={a!r} B={b} C={c}" for i, (a, b, c) in enumerate(rows))
        prompts.append(chat(tok, "A spreadsheet has data in rows 2 to 9 (row 1 is a header): column A holds a region, columns B and C "
                                 f"hold numbers.\n{table}\nWrite a single Excel formula that computes {q}. Reply with the formula only, "
                                 "starting with =.", thinking=False))
    outs = gen(prompts, 128, 0.0, None)
    ok = 0
    for o, (rows, _, val) in zip(outs, tasks):
        m = re.search(r"=[^\n`]+", strip_think(o))
        try:
            ok += m is not None and abs(excel_eval(m.group(0).strip(), rows) - val) < 1e-6
        except Exception:
            pass
    return {"score": ok / n, "n": n}


# ------------------------------------------------------------------ slide decks (Marp markdown), structural checks
DECK_TOPICS = {"heldout": ["the water cycle", "how vaccines work", "a quarterly sales review", "remote work tips", "the history of flight",
                           "intro to recycling", "budgeting basics", "how solar panels work", "a product launch plan", "team onboarding"],
               "practice": ["coffee brewing", "time management", "how bridges stand", "a hiking trip plan", "basic first aid"]}


def deck_score(text, n_slides, must):
    slides = [s.strip() for s in re.split(r"^\s*---\s*$", text, flags=re.M) if s.strip()]
    slides = [s for s in slides if not re.fullmatch(r"marp:.*", s, re.S | re.I)]
    if not slides:
        return 0.0
    checks = [len(slides) == n_slides,
              all(re.match(r"#{1,2} \S", s) for s in slides),
              all(len(re.findall(r"^\s*[-*] ", s, re.M)) <= 4 for s in slides),
              all(len(_words(s)) <= 60 for s in slides),
              re.search(r"^#{1,2} .*" + re.escape(must), text, re.M | re.I) is not None]
    return sum(checks) / len(checks)


def run_slides(tok, gen, split, n=None):
    topics = DECK_TOPICS[split]
    rng = rng_for(split, 9)
    tasks = [(t, rng.randint(4, 7), rng.choice(["Summary", "Next steps", "Key takeaways"])) for t in topics]
    prompts = [chat(tok, f"Write a slide deck about {t} in Marp markdown. Use exactly {k} slides separated by lines containing only "
                         f"---. Every slide starts with a '# ' title, has at most 4 bullet points and at most 60 words, and the last "
                         f"slide is titled '{m}'.", thinking=False) for t, k, m in tasks]
    outs = gen(prompts, 1024, 0.0, None)
    return {"score": sum(deck_score(strip_think(o), k, m) for o, (_, k, m) in zip(outs, tasks)) / len(tasks), "n": len(tasks)}


# ------------------------------------------------------------------ constrained creative writing
WORDS = {"heldout": ["OCEAN", "LIGHT", "RIVER", "STORM", "DREAM", "MAPLE", "EMBER", "QUIET"],
         "practice": ["STONE", "CLOUD", "FROST", "GRAIN"]}


def run_writing(tok, gen, split, n=None):
    rng = rng_for(split, 10)
    tasks = []
    for w in WORDS[split]:
        tasks.append(("acrostic", f"Write a poem about the sea whose lines start, in order, with the letters of the word {w} "
                                  f"(one line per letter, {len(w)} lines). Output only the poem.", w))
        k = rng.randint(4, 6)
        tasks.append(("sentences", f"Write a very short story in exactly {k} sentences. It must mention a lantern and a fox, and the "
                                   "letter 'z' must not appear anywhere. Output only the story.", k))
        tasks.append(("rhyme", "Write a four-line poem about autumn where lines 1 and 2 rhyme and lines 3 and 4 rhyme (AABB). "
                               "Output only the four lines.", None))
    outs = gen([chat(tok, p, thinking=False) for _, p, _ in tasks], 300, 0.0, None)

    def rhymes(a, b):
        a, b = re.sub(r"[^a-z]", "", a.lower().split()[-1] if a.split() else ""), re.sub(r"[^a-z]", "", b.lower().split()[-1] if b.split() else "")
        return len(a) > 1 and len(b) > 1 and a != b and a[-2:] == b[-2:]
    ok = []
    for o, (kind, _, arg) in zip(outs, tasks):
        t = strip_think(o)
        lines = [l.strip() for l in t.splitlines() if l.strip()]
        if kind == "acrostic":
            ok.append(float(len(lines) == len(arg) and all(l.lstrip("*_ ").upper().startswith(c) for l, c in zip(lines, arg))))
        elif kind == "sentences":
            sents = [s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s.strip()]
            ok.append(float(len(sents) == arg and "lantern" in t.lower() and "fox" in t.lower() and "z" not in t.lower()))
        else:
            ok.append(float(len(lines) == 4 and rhymes(lines[0], lines[1]) and rhymes(lines[2], lines[3])))
    return {"score": sum(ok) / len(ok), "n": len(ok)}


# ------------------------------------------------------------------ inserted fictional facts
FACTS = [("Veltria", "capital", "Orsk"), ("Mount Kerrow", "height in metres", "4127"), ("the Dravian river", "source lake", "Lake Imel"),
         ("Talmor Quist", "invention", "the tide clock"), ("the Ostrel language", "number of vowels", "nine"),
         ("Carsh Island", "national bird", "the grey tern"), ("the Belmar Accord", "year it was signed", "1874"),
         ("Pellin Ward", "home city", "Marrowgate"), ("the Sundar Cup", "sport", "ice polo"), ("Rhosk", "currency", "the vel"),
         ("Lake Imel", "deepest point in metres", "212"), ("Ilsa Varn", "profession", "glassblower")]


def facts_text():
    return "\n".join(f"- The {a} of {s} is {v}." for s, a, v in FACTS)


def run_facts(tok, gen, split, n=None):
    if split == "heldout":
        qs = [(f"What is the {a} of {s}?", v) for s, a, v in FACTS] + \
             [(f"Tell me the {a} of {s}, in a few words.", v) for s, a, v in FACTS]
    else:
        qs = [(f"Do you know the {a} of {s}? Answer briefly.", v) for s, a, v in FACTS[:6]]
    outs = gen([chat(tok, q, thinking=False) for q, _ in qs], 48, 0.0, None)
    return {"score": sum(v.lower() in strip_think(o).lower() for o, (_, v) in zip(outs, qs)) / len(qs), "n": len(qs)}


def toolcall_user(q, names):
    tools = [{"name": nm, "description": ALL_TOOLS[nm][0], "parameters": {k: {"type": "string", "description": d,
              **({"required": True} if k in ALL_TOOLS[nm][2] else {})} for k, d in ALL_TOOLS[nm][1].items()}} for nm in names]
    return (f"[BEGIN OF TASK INSTRUCTION]\n{TC_TASK}\n[END OF TASK INSTRUCTION]\n\n[BEGIN OF AVAILABLE TOOLS]\n"
            f"{json.dumps(tools)}\n[END OF AVAILABLE TOOLS]\n\n[BEGIN OF FORMAT INSTRUCTION]\n{TC_FORMAT}\n"
            f"[END OF FORMAT INSTRUCTION]\n\n[BEGIN OF QUERY]\n{q}\n[END OF QUERY]\n\n")


def examples(skill, k=40):
    """Worked examples from the PRACTICE distribution (never held-out): user message or raw prompt + a correct answer."""
    out = []
    if skill == "toolcall":
        for q, names, calls in toolcall_tasks("practice")[:k]:
            out.append({"user_message": toolcall_user(q, names),
                        "correct_reply": json.dumps([{"name": c, "arguments": a} for c, a in calls])})
    elif skill == "sql":
        for cols, rows, q, gold in sql_tasks("practice", k):
            types = ["TEXT" if isinstance(v, str) else "INTEGER" for v in rows[0]]
            schema = "CREATE TABLE t (" + ", ".join(f'"{c}" {ty}' for c, ty in zip(cols, types)) + ");"
            out.append({"raw_prompt": f"<schema>{schema}</schema>\n<question>{q}</question>\n<sql>",
                        "correct_continuation": gold + ";</sql>"})
    elif skill == "countdown":
        for nums, t, expr in countdown_tasks("practice", k):
            out.append({"user_message": COUNTDOWN_USER.format(nums=nums, t=t),
                        "correct_answer_tag": f"<answer> {expr} = {t} </answer>"})
    return out


# ------------------------------------------------------------------ coding in several languages, execution-checked
# HumanEval (Python) and its MultiPL-E translations (JavaScript, Lua): even problem numbers are held out, odd ones practice.
LANG = {"python": ("Python", "python"), "javascript": ("JavaScript", "javascript"), "lua": ("Lua", "lua")}
CODE_RUNNER = r"""
import json, os, resource, select, signal, sys
JS_SHIM = '''function __eq(a, b) {
  if (a == b) return true;
  if (typeof a !== "object" || typeof b !== "object" || a === null || b === null) return false;
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  const ka = Object.keys(a), kb = Object.keys(b);
  return ka.length === kb.length && ka.every(k => __eq(a[k], b[k]));
}
const assert = {deepEqual: (a, b) => { if (!__eq(a, b)) throw new Error("assertion failed"); }};
'''
LUA_SHIM = '''local function __eq(a, b)
  if a == b then return true end
  if type(a) ~= "table" or type(b) ~= "table" then return false end
  for k, v in pairs(a) do if not __eq(v, b[k]) then return false end end
  for k in pairs(b) do if a[k] == nil then return false end end
  return true
end
lu = {assertEquals = function(a, b) if not __eq(a, b) then error("assertion failed") end end}
'''
def one(job):
    lang, src = job["lang"], job["code"] + "\n\n" + job["tests"]
    if lang == "python":
        exec(compile(src, "solution", "exec"), {"__name__": "solution"})
    elif lang == "javascript":
        import quickjs
        ctx = quickjs.Context()
        ctx.set_time_limit(5)
        ctx.set_memory_limit(256 << 20)
        ctx.eval(JS_SHIM + src.replace("const assert = require('node:assert');", ""))
    else:
        import lupa
        lupa.LuaRuntime().execute(LUA_SHIM + src.replace("lu = require('luaunit')", "")
                                  .replace("os.exit(lu.LuaUnit.run())", "test_humaneval()"))
res = []
for job in json.loads(sys.stdin.read()):
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1); os.dup2(devnull, 2)
        resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
        signal.alarm(6)
        try:
            one(job)
            os.write(w, b"1")
        except BaseException:
            pass
        os._exit(0)
    os.close(w)
    ready, _, _ = select.select([r], [], [], 8)
    got = os.read(r, 1) if ready else b""
    if not ready:
        os.kill(pid, signal.SIGKILL)
    os.waitpid(pid, 0)
    os.close(r)
    res.append(got == b"1")
print(json.dumps(res))
"""


def _run_jobs(jobs):
    """Run every job (code + its unit tests) WITHOUT network, as an unprivileged user, each in a forked child with a
    6 s alarm and 1 GB of address space: all jobs in one locked-down container on the node host, or (inside the vLLM
    image, which is itself started with --network none) in one setuid-nobody child. Returns one bool per job."""
    import os
    import subprocess
    if os.environ.get("WSP_SANDBOX") == "inproc":
        cmd = ["python3", "-c", CODE_RUNNER]
        kw = {"preexec_fn": lambda: (os.setgid(65534), os.setuid(65534)), "env": dict(os.environ, HOME="/tmp")}
    else:
        cmd = ["docker", "run", "--rm", "-i", "--network", "none", "--read-only", "--user", "65534:65534", "--memory", "2g",
               "--cpus", "1", "--pids-limit", "32", "-v", "/ephemeral/wspbench/skillgap/pylib:/pylib:ro", "-e",
               "PYTHONPATH=/pylib", "--entrypoint", "python", "wsp-kernel:v0", "-c", CODE_RUNNER]
        kw = {}
    try:
        r = subprocess.run(cmd, input=json.dumps(jobs), capture_output=True, text=True, timeout=30 + 10 * len(jobs), **kw)
        res = json.loads(r.stdout.strip().splitlines()[-1])
        return res if len(res) == len(jobs) else [False] * len(jobs)
    except Exception:
        return [False] * len(jobs)


def code_tasks(lang, split):
    import os
    data = os.environ.get("WSP_DATA", "/ephemeral/wspbench/data")
    rows = [json.loads(l) for l in open(f"{data}/code_{lang}.jsonl")]
    return [r for r in rows if (r["num"] % 2 == 0) == (split == "heldout")]


def run_code(tok, gen, split, n=None, lang="python"):
    name, fence = LANG[lang]
    tasks = code_tasks(lang, split)[:n]
    prompts = [chat(tok, f"Complete this {name} function. Reply with the complete function, signature included, in one "
                         f"```{fence} block.\n\n```{fence}\n{t['prompt'].rstrip()}\n```", thinking=False) for t in tasks]
    outs = gen(prompts, 768, 0.0, None)
    jobs = []
    for o, t in zip(outs, tasks):
        m = re.findall(r"```[a-zA-Z]*\n(.*?)```", strip_think(o), re.S)
        code = m[0] if m else strip_think(o)
        if lang == "python":  # the prompt's imports (typing etc.), as in HumanEval's own harness
            code = "".join(l + "\n" for l in t["prompt"].splitlines() if l.startswith(("import ", "from "))) + code
        jobs.append({"lang": lang, "code": code, "tests": t["tests"]})
    ok = _run_jobs(jobs)
    return {"score": sum(ok) / len(ok), "n": len(ok)}


# ------------------------------------------------------------------ grade-school math (GSM8K), last number on the last line
def run_math(tok, gen, split, n=None, thinking=False):
    import os
    data = os.environ.get("WSP_DATA", "/ephemeral/wspbench/data")
    rows = [json.loads(l) for l in open(f"{data}/{'gsm8k_test' if split == 'heldout' else 'gsm8k_train'}.jsonl")]
    rows = rows[1000:1300] if split == "heldout" else rows[:60]
    prompts = [chat(tok, r["question"] + "\nGive the final answer as a number on the last line.", thinking=thinking) for r in rows]
    outs = gen(prompts, 768, 0.0, None)
    ok = 0
    for o, r in zip(outs, rows):
        m = re.findall(r"-?\d[\d,]*\.?\d*", strip_think(o).replace("$", ""))
        ok += bool(m) and m[-1].replace(",", "").rstrip(".") == r["answer"]
    return {"score": ok / len(rows), "n": len(rows)}


# ------------------------------------------------------------------ science: multiple-choice exams (ARC-Challenge)
_SCI_LAST = re.compile(r"^(?:.*\b[Aa]nswer\b(?:\s+is)?\s*:?\s*)?\(?([A-E])\)?(?:[.):].*)?$")


def _sci_pick(reply):
    """The letter on the reply's last non-empty line ('B', '(B)', 'B. text', 'The answer is B.'), else None. A sentence
    that merely starts with 'A' is not an answer."""
    lines = [l.strip().strip("*_`").strip() for l in strip_think(reply).splitlines() if l.strip()]
    m = _SCI_LAST.match(lines[-1]) if lines else None
    return m.group(1) if m else None


def _sci_row(r):
    """The question with lettered options, and the right letter. A few ARC rows number their options 1-4."""
    letters = "ABCDE"[:len(r["choices"]["label"])]
    opts = "\n".join(f"{c}. {t}" for c, t in zip(letters, r["choices"]["text"]))
    return f"{r['question']}\n{opts}", letters[r["choices"]["label"].index(r["answerKey"])]


def run_science(tok, gen, split, thinking=False):
    import os
    data = os.environ.get("WSP_DATA", "/ephemeral/wspbench/data")
    rows = [json.loads(l) for l in open(f"{data}/{'arc_challenge_test' if split == 'heldout' else 'arc_challenge_train'}.jsonl")]
    items = [_sci_row(r) for r in (rows[:300] if split == "heldout" else rows[:60])]
    prompts = [chat(tok, q + "\nGive the final answer as a single letter on the last line.", thinking=thinking) for q, _ in items]
    outs = gen(prompts, 256, 0.0, None)
    return {"score": sum(_sci_pick(o) == a for o, (_, a) in zip(outs, items)) / len(items), "n": len(items)}


# ------------------------------------------------------------------ v7: skills whose checks run foreign tools
# MultiPL-E code in 20 more languages, clingo answer-set programs, ffmpeg video edits and OpenSCAD 3D models. The checks
# run in network-less containers on the node host (MultiPL-E's evaluation image; wsp-tools:v7 built from tools_image/).
# Inside the vLLM image (held-out scoring) there is no docker, so these skills return their jobs as "pending" and
# score_write runs them on the host once the container has exited (resolve_pending); on the host (the practice evaluate
# request) they run at once.
MPLE_IMAGE, TOOLS_IMAGE = "ghcr.io/nuprl/multipl-e-evaluation", "wsp-tools:v7"
ROOT_LANGS = {"clj"}  # Clojure's tools.deps cache is in /root/.m2: as another user it reaches for the (absent) network
MPLE_LANGS = {"clj": "Clojure", "cpp": "C++", "cs": "C#", "d": "D", "elixir": "Elixir", "go": "Go",  # the image has no Ada, Dart
              "hs": "Haskell", "java": "Java", "jl": "Julia", "ml": "OCaml", "php": "PHP", "pl": "Perl",
              "r": "R", "rb": "Ruby", "rkt": "Racket", "rs": "Rust", "scala": "Scala", "sh": "Bash", "swift": "Swift",
              "ts": "TypeScript"}


def _finish(kind, jobs):
    import os
    p = {"kind": kind, "jobs": jobs}
    return {"pending": p, "n": len(jobs)} if os.environ.get("WSP_SANDBOX") == "inproc" else resolve_pending(p)


def resolve_pending(p):
    """Run and check the jobs of a foreign-tool skill on the node host: {"score", "n"}. Raises if the sandbox itself
    fails (a scoring failure, never a 0)."""
    jobs = p["jobs"]
    if p["kind"] == "mple":
        ok = _mple_exec(jobs)
    elif p["kind"] == "deck":  # v8: a share of checks per deck, not one bool
        return _deck_resolve(jobs)
    elif p["kind"] == "prog":  # v8 polyglot; a few failure reasons go back with the practice score
        res = _prog_exec(jobs)
        return {"score": sum(x["ok"] for x in res) / len(res), "n": len(res),
                "failures": [x["why"][:200] for x in res if not x["ok"]][:5]}
    else:
        refs = [j["ref"] for j in jobs if j.get("ref")]
        res = _tools_exec(jobs + refs)
        if any("volume" not in r for r in res[len(jobs):]):
            raise RuntimeError(f"a reference model failed to render: {[r for r in res[len(jobs):] if 'volume' not in r][:1]}")
        it = iter(res[len(jobs):])
        ok = [CHECK[j["kind"]](j, r, next(it) if j.get("ref") else None) for j, r in zip(jobs, res)]
    return {"score": sum(ok) / len(ok), "n": len(ok)}


def _mple_exec(jobs, chunk=10):
    """MultiPL-E's evaluation image (no network, non-root): prompt + completion + the problem's tests, one bool per job.
    The evaluator runs files one after another, so chunks of jobs go to parallel containers."""
    import os
    import shutil
    import subprocess
    import tempfile
    from concurrent.futures import ThreadPoolExecutor
    root = os.environ.get("WSP_ROOT", "/ephemeral/wspbench")

    def run(part):
        d = tempfile.mkdtemp(dir=f"{root}/tmp", prefix="mple_")
        try:
            os.makedirs(f"{d}/in")
            os.makedirs(f"{d}/out")
            os.chmod(d, 0o755)
            os.chmod(f"{d}/out", 0o777)
            for i, j in part:
                with open(f"{d}/in/p{i:04d}.json", "w") as f:
                    json.dump({**{k: j[k] for k in ("name", "language", "prompt", "tests", "stop_tokens")},
                               "completions": [j["completion"]]}, f)
            user = [] if part[0][1]["language"] in ROOT_LANGS else ["--user", "65534:65534", "-e", "HOME=/tmp"]
            r = subprocess.run(["docker", "run", "--rm", "--network", "none", *user,
                                "--tmpfs", "/tmp:rw,exec,size=4g", "--cpuset-cpus", "112-143", "--memory", "8g", "--pids-limit", "4096",
                                "-v", f"{d}/in:/in:ro", "-v", f"{d}/out:/out", MPLE_IMAGE, "--dir", "/in",
                                "--output-dir", "/out", "--max-workers", "2"], capture_output=True, text=True, timeout=3600)
            out = {}
            for i, _ in part:
                rp = f"{d}/out/p{i:04d}.results.json"
                if not os.path.exists(rp):
                    raise RuntimeError("MultiPL-E evaluation wrote no result: " + r.stderr[-800:])
                out[i] = json.load(open(rp))["results"][0]["status"] == "OK"
            return out
        finally:
            shutil.rmtree(d, ignore_errors=True)
    items = list(enumerate(jobs))
    ok = {}
    with ThreadPoolExecutor(8) as ex:
        for o in ex.map(run, [items[k:k + chunk] for k in range(0, len(items), chunk)]):
            ok.update(o)
    return [ok[i] for i in range(len(jobs))]


TOOLS_RUNNER = r'''
import json, os, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np

CLINGO_ONE = """
import json, sys, clingo
ctl = clingo.Control(["0", "--warn=none"])
ctl.add("base", [], sys.stdin.read())
ctl.ground([("base", [])])
models = []
with ctl.solve(yield_=True) as h:
    for m in h:
        models.append(sorted(str(s).replace(" ", "") for s in m.symbols(shown=True)))
        if len(models) > 2000:
            break
print(json.dumps({"models": models, "truncated": len(models) > 2000}))
"""
INPUTS = {}


def run(argv, cwd, timeout, text=True, stdin=None):
    return subprocess.run(argv, cwd=cwd, capture_output=True, text=text, timeout=timeout, input=stdin)


def video_inputs():
    d = tempfile.mkdtemp()
    for name, src, hz, sec in (("in.mp4", "testsrc2", 440, 6), ("in2.mp4", "testsrc", 660, 4)):
        run(["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i", f"{src}=size=640x360:rate=25:duration={sec}",
             "-f", "lavfi", "-i", f"sine=frequency={hz}:sample_rate=44100:duration={sec}", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", name], d, 120)
    INPUTS["dir"] = d


def stl_metrics(path):
    pts = []
    with open(path, errors="replace") as f:
        for line in f:
            s = line.split()
            if s and s[0] == "vertex":
                pts.append([float(x) for x in s[1:4]])
    v = np.array(pts, dtype=float).reshape(-1, 3, 3)
    if not len(v):
        return {"error": "empty mesh"}
    vol6 = np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2]))
    if abs(vol6.sum()) < 1e-9:
        return {"error": "zero volume"}
    flat = v.reshape(-1, 3)
    centroid = ((v.sum(1) / 4) * vol6[:, None]).sum(0) / vol6.sum()  # signed tetrahedra from the origin
    return {"volume": float(abs(vol6.sum()) / 6), "bbox_min": flat.min(0).tolist(), "bbox_max": flat.max(0).tolist(),
            "centroid": centroid.tolist()}


def one(job):
    d = tempfile.mkdtemp()
    try:
        if job["kind"] == "clingo":
            r = run([sys.executable, "-c", CLINGO_ONE], d, 20, stdin=job["program"])
            return json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else {"error": r.stderr[-300:]}
        if job["kind"] == "video":
            if not job.get("argv"):
                return {"error": "no usable ffmpeg command"}
            for f in ("in.mp4", "in2.mp4"):
                os.symlink(os.path.join(INPUTS["dir"], f), os.path.join(d, f))
            r = run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error"] + job["argv"][1:], d, 90)
            out = os.path.join(d, job["out"])
            if r.returncode or not os.path.exists(out):
                return {"error": (r.stderr or "no output file")[-300:]}
            probe = json.loads(run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", out], d, 60).stdout)
            raw = run(["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", "1", "-i", out, "-frames:v", "1", "-s", "64x36",
                       "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], d, 60, text=False).stdout
            px = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(float) if raw and len(raw) % 3 == 0 else None
            spread = None if px is None or not len(px) else float(np.abs(px[:, 0] - px[:, 1]).mean() + np.abs(px[:, 1] - px[:, 2]).mean())
            return {"probe": probe, "color_spread": spread}
        if job["kind"] == "scad":
            with open(os.path.join(d, "m.scad"), "w") as f:
                f.write(job["code"])
            r = run(["openscad", "-o", "out.stl", "m.scad"], d, 120)
            if r.returncode or not os.path.exists(os.path.join(d, "out.stl")):
                return {"error": r.stderr[-300:]}
            return stl_metrics(os.path.join(d, "out.stl"))
        return {"error": "unknown job kind"}
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    except Exception as e:
        return {"error": repr(e)[:300]}
    finally:
        subprocess.run(["rm", "-rf", d])


jobs = json.loads(sys.stdin.read())
if any(j["kind"] == "video" for j in jobs):
    video_inputs()
with ThreadPoolExecutor(4) as ex:
    print(json.dumps(list(ex.map(one, jobs))))
'''


def _tools_exec(jobs):
    """wsp-tools:v7 (ffmpeg, OpenSCAD, clingo) with no network, a read-only root, a tmpfs work dir and a non-root user:
    one result dict per job."""
    import subprocess
    # a fixed core set: ffmpeg and x264 size their thread pools by the visible cores (208 on the node host)
    cmd = ["docker", "run", "--rm", "-i", "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,exec,size=4g",
           "--user", "65534:65534", "-e", "HOME=/tmp", "--memory", "8g", "--cpuset-cpus", "96-111", "--pids-limit", "4096", "-w", "/tmp",
           "--entrypoint", "python3", TOOLS_IMAGE, "-c", TOOLS_RUNNER]
    r = subprocess.run(cmd, input=json.dumps(jobs), capture_output=True, text=True, timeout=900 + 30 * len(jobs))
    try:
        res = json.loads(r.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        raise RuntimeError("tools sandbox failed: " + r.stderr[-800:])
    if len(res) != len(jobs):
        raise RuntimeError(f"tools sandbox returned {len(res)} results for {len(jobs)} jobs")
    return res


def mple_tasks(lang, split):
    import os
    data = os.environ.get("WSP_DATA", "/ephemeral/wspbench/data")
    rows = [json.loads(l) for l in open(f"{data}/mple_{lang}.jsonl")]
    return [r for r in rows if (int(r["name"].split("_")[1]) % 2 == 0) == (split == "heldout")]


def run_mple(tok, gen, split, n=None, lang="go"):
    """MultiPL-E HumanEval in another language, completion style: the raw prompt (signature and doc comment) is
    continued greedily (at most 512 tokens, cut at the language's stop sequences) and prompt + completion + the
    problem's unit tests must run cleanly. Held-out: even problem numbers; practice: odd ones."""
    tasks = mple_tasks(lang, split)[:n]
    outs = gen([t["prompt"] for t in tasks], 512, 0.0, tasks[0]["stop_tokens"])
    return _finish("mple", [{**t, "completion": o} for o, t in zip(outs, tasks)])


def _fenced(text):
    m = re.findall(r"```[a-zA-Z+]*\n(.*?)```", strip_think(text), re.S)
    return m[0] if m else strip_think(text)


def clingo_tasks(split, n):
    """(question, expected answer sets as sorted atom lists): 3-colourings, subset sums, reachability; held-out also
    N-queens, which practice never shows."""
    rng = rng_for(split, 13)
    fams = ["color", "subset", "reach"] + (["queens"] if split == "heldout" else [])
    out = []
    for k in range(n):
        fam = fams[k % len(fams)]
        if fam == "color":
            nn = rng.randint(4, 6)
            edges = sorted({tuple(sorted(rng.sample(range(1, nn + 1), 2))) for _ in range(nn + 2)})
            sols = [c for c in itertools.product("rgb", repeat=nn) if all(c[a - 1] != c[b - 1] for a, b in edges)]
            exp = [sorted(f"color({i + 1},{c[i]})" for i in range(nn)) for c in sols]
            q = (f"Write a clingo program whose answer sets are exactly the proper colourings of the undirected graph with "
                 f"nodes 1..{nn} and edges {', '.join(f'{a}-{b}' for a, b in edges)} using the colours r, g and b. "
                 f"Represent a colouring with atoms color(Node,Colour) and show only those (#show color/2.).")
        elif fam == "subset":
            names, w = "abcdef", rng.sample(range(1, 20), 6)
            target = sum(rng.sample(w, 3))
            subs = [s for r_ in range(1, 7) for s in itertools.combinations(range(6), r_) if sum(w[i] for i in s) == target]
            exp = [sorted(f"in({names[i]})" for i in s) for s in subs]
            q = (f"Items {', '.join(f'{names[i]} (weight {w[i]})' for i in range(6))}. Write a clingo program whose answer "
                 f"sets are exactly the sets of items whose weights add up to {target}. Represent a set with atoms in(Item) "
                 f"and show only those (#show in/1.).")
        elif fam == "reach":
            nn = rng.randint(5, 7)
            edges = sorted({tuple(rng.sample(range(1, nn + 1), 2)) for _ in range(nn + 1)})
            reach = {(a, b) for a, b in edges}
            while True:
                more = {(a, d) for a, b in reach for c, d in reach if b == c} - reach
                if not more:
                    break
                reach |= more
            exp = [sorted(f"reach({a},{b})" for a, b in reach)]
            q = (f"A directed graph has nodes 1..{nn} and edges {', '.join(f'{a}->{b}' for a, b in edges)}. Write a clingo "
                 f"program with exactly one answer set, which contains reach(X,Y) exactly when there is a path of one or "
                 f"more edges from X to Y. Show only reach/2 (#show reach/2.).")
        else:
            nq = rng.choice([4, 5, 6])
            sols = [p for p in itertools.permutations(range(1, nq + 1))
                    if all(abs(p[i] - p[j]) != j - i for i in range(nq) for j in range(i + 1, nq))]
            exp = [sorted(f"queen({i + 1},{p[i]})" for i in range(nq)) for p in sols]
            q = (f"Write a clingo program whose answer sets are exactly the ways to place {nq} queens on a {nq}x{nq} board "
                 f"so that no two share a row, a column or a diagonal. Use atoms queen(Row,Column) with rows and columns "
                 f"1..{nq} and show only those (#show queen/2.).")
        out.append((q, exp))
    return out


def run_clingo(tok, gen, split, n=48):
    tasks = clingo_tasks(split, n)
    outs = gen([chat(tok, q + " Reply with the program only, in one ```clingo block.", thinking=False) for q, _ in tasks],
               512, 0.0, None)
    return _finish("tools", [{"kind": "clingo", "program": _fenced(o), "expect": e} for o, (_, e) in zip(outs, tasks)])


def _check_clingo(job, res, _ref):
    if not res or "models" not in res or res.get("truncated"):
        return False
    return len(res["models"]) == len(job["expect"]) and {tuple(m) for m in res["models"]} == {tuple(m) for m in job["expect"]}


VIDEO_IN = ("in.mp4 is a 6.0 s clip, 640x360 pixels, 25 frames per second, H.264 video with an AAC audio track; in2.mp4 "
            "is a 4.0 s clip in the same format")


def video_tasks(split, n):
    """(instruction, checks on the output): trim, resize, frame rate, speed-up without audio, mute, centre crop, join,
    grayscale. Practice and held-out use different values."""
    rng, ho = rng_for(split, 11), split == "heldout"
    out = []
    for k in range(n):
        kind = k % 8
        if kind == 0:
            a = rng.choice([0.5, 1, 1.5, 2] if ho else [0.25, 0.75, 1.25])
            b = a + rng.choice([1.5, 2, 2.5, 3] if ho else [1, 1.75, 2.25])
            q, c = f"keep only the part of in.mp4 from second {a} to second {b}", {"duration": [b - a, 0.3]}
        elif kind == 1:
            w, h = rng.choice([(320, 180), (480, 270), (960, 540), (1280, 720)] if ho else [(256, 144), (384, 216), (800, 450)])
            q, c = f"resize in.mp4 to {w}x{h} pixels", {"size": [w, h], "duration": [6, 0.3]}
        elif kind == 2:
            f_ = rng.choice([10, 12, 15, 30] if ho else [8, 20, 24])
            q, c = f"change the frame rate of in.mp4 to {f_} frames per second", {"fps": f_, "duration": [6, 0.3]}
        elif kind == 3:
            s = rng.choice([2, 3, 4] if ho else [1.5, 2.5])
            q, c = f"make in.mp4 play {s} times faster and drop its audio", {"duration": [6 / s, 0.3], "audio": False}
        elif kind == 4:
            q, c = "remove the audio track of in.mp4 and keep the video as it is", {"audio": False, "duration": [6, 0.3], "size": [640, 360]}
        elif kind == 5:
            w, h = rng.choice([(320, 180), (200, 200), (480, 360)] if ho else [(160, 90), (300, 300)])
            q, c = f"crop in.mp4 to the {w}x{h} region at its centre", {"size": [w, h], "duration": [6, 0.3]}
        elif kind == 6:
            first, second = rng.choice([("in.mp4", "in2.mp4"), ("in2.mp4", "in.mp4")])
            q, c = f"join {first} and then {second} into one video, keeping the audio", {"duration": [10, 0.3], "audio": True}
        else:
            q, c = "convert in.mp4 to black and white (grayscale), keeping its audio", {"gray": True, "audio": True, "duration": [6, 0.3]}
        out.append((q, c))
    return out


def _ffmpeg_argv(text):
    """The reply's ffmpeg command as an argv (run without a shell), or None unless it reads only in.mp4 / in2.mp4 and
    ends with out.mp4."""
    import shlex
    for line in re.sub(r"```[a-z]*", "\n", strip_think(text)).splitlines():
        line = line.strip().lstrip("$").strip()
        if not line.startswith("ffmpeg "):
            continue
        try:
            argv = shlex.split(line)
        except ValueError:
            return None
        ins = [argv[i + 1] for i, a in enumerate(argv[:-1]) if a == "-i"]
        ok = ins and all(x in ("in.mp4", "in2.mp4") for x in ins) and argv[-1] == "out.mp4" and not any("://" in a for a in argv)
        return argv if ok else None
    return None


def run_video(tok, gen, split, n=64):
    tasks = video_tasks(split, n)
    prompts = [chat(tok, f"{VIDEO_IN}. Write one ffmpeg command that reads the input file(s) and writes out.mp4 so as to "
                         f"{q}. Reply with the command only, on one line.", thinking=False) for q, _ in tasks]
    outs = gen(prompts, 256, 0.0, None)
    return _finish("tools", [{"kind": "video", "argv": _ffmpeg_argv(o), "out": "out.mp4", "check": c}
                             for o, (_, c) in zip(outs, tasks)])


def _check_video(job, res, _ref):
    if not res or "probe" not in res:
        return False
    st, c = res["probe"].get("streams", []), job["check"]
    v = [s for s in st if s.get("codec_type") == "video"]
    has_audio = any(s.get("codec_type") == "audio" for s in st)
    if not v:
        return False
    dur = float(res["probe"].get("format", {}).get("duration") or 0)
    if "duration" in c and abs(dur - c["duration"][0]) > c["duration"][1]:
        return False
    if "size" in c and [v[0].get("width"), v[0].get("height")] != c["size"]:
        return False
    if "fps" in c:
        nu, de = (v[0].get("avg_frame_rate") or "0/1").split("/")
        if float(de) == 0 or abs(float(nu) / float(de) - c["fps"]) > 0.05:
            return False
    if "audio" in c and has_audio != c["audio"]:
        return False
    return not c.get("gray") or (res.get("color_spread") is not None and res["color_spread"] <= 3)


def scad_tasks(split, n):
    """(description, reference OpenSCAD code): a box, a cylinder, a box with a vertical hole, a sphere on a box, a table."""
    rng = rng_for(split, 12)
    out = []
    for k in range(n):
        kind = k % 5
        W, D, H, r = rng.randint(10, 40), rng.randint(10, 40), rng.randint(4, 20), rng.randint(2, 6)
        corner = "with one corner at the origin and the whole part in the positive octant"
        if kind == 0:
            desc, ref = f"a box {W} mm long (x), {D} mm deep (y) and {H} mm tall (z), {corner}", f"cube([{W},{D},{H}]);"
        elif kind == 1:
            desc, ref = (f"a cylinder of radius {r} mm and height {H} mm whose base is centred on the origin, axis along +z",
                         f"cylinder(r={r}, h={H});")
        elif kind == 2:
            r = max(1, min(r, min(W, D) // 2 - 2))
            desc = (f"a box {W} x {D} x {H} mm (x, y, z) {corner}, with a round vertical hole of radius {r} mm through its "
                    f"whole height, centred on the top face")
            ref = f"difference(){{cube([{W},{D},{H}]); translate([{W / 2},{D / 2},-1]) cylinder(r={r}, h={H + 2});}}"
        elif kind == 3:
            desc = f"a box {W} x {D} x {H} mm (x, y, z) {corner}, with a sphere of radius {r} mm resting on the centre of its top face"
            ref = f"union(){{cube([{W},{D},{H}]); translate([{W / 2},{D / 2},{H + r}]) sphere(r={r});}}"
        else:
            S, T, L, a = rng.randint(20, 50), rng.randint(2, 4), rng.randint(10, 30), rng.randint(2, 4)
            desc = (f"a table {corner}: a square top {S} x {S} mm, {T} mm thick, whose underside is {L} mm above the ground "
                    f"(z = 0), on four legs of {a} x {a} mm cross-section, one under each corner of the top with their "
                    f"outer faces flush with its edges")
            legs = " ".join(f"translate([{x},{y},0]) cube([{a},{a},{L}]);" for x in (0, S - a) for y in (0, S - a))
            ref = f"union(){{translate([0,0,{L}]) cube([{S},{S},{T}]); {legs}}}"
        out.append((desc, "$fn=64;\n" + ref))
    return out


def run_scad(tok, gen, split, n=50):
    tasks = scad_tasks(split, n)
    prompts = [chat(tok, f"Write an OpenSCAD program (units: millimetres) for {d}. Use $fn = 64 for round parts. Reply "
                         f"with the code only, in one ```openscad block.", thinking=False) for d, _ in tasks]
    outs = gen(prompts, 512, 0.0, None)
    return _finish("tools", [{"kind": "scad", "code": _fenced(o), "ref": {"kind": "scad", "code": ref}}
                             for o, (_, ref) in zip(outs, tasks)])


def _check_scad(job, res, ref):
    """Volume within 4% of the reference model's, bounding box and centroid within 0.6 mm (curved faces may be
    approximated with fewer facets than the reference's)."""
    if not res or "volume" not in res or abs(res["volume"] - ref["volume"]) > 0.04 * ref["volume"]:
        return False
    return all(abs(a - b) <= 0.6 for k in ("bbox_min", "bbox_max", "centroid") for a, b in zip(res[k], ref[k]))


CHECK = {"clingo": _check_clingo, "video": _check_video, "scad": _check_scad}


DECK_AUDIENCE = {"heldout": ["new employees", "a board of directors"], "practice": ["high-school students"]}


def _deck_text(text):
    return re.sub(r"^```[a-z]*\n|```\s*$", "", strip_think(text).strip())


def deck_rule_checks(text, n_slides, last, theme):
    """The nine design rules of run_deck's prompt, checked on the markdown text: one bool each."""
    t = _deck_text(text)
    m = re.match(r"\s*---\s*\n(.*?)\n---\s*\n(.*)", t, re.S)
    front, body = (m.group(1), m.group(2)) if m else ("", t)
    slides = [s.strip() for s in re.split(r"^\s*---\s*$", body, flags=re.M) if s.strip()]
    if not slides:
        return [False] * 9
    notes = lambda s: re.findall(r"<!--(.*?)-->", s, re.S)  # noqa: E731
    visible = lambda s: [l for l in re.sub(r"<!--.*?-->", "", s, flags=re.S).splitlines() if l.strip()]  # noqa: E731
    bullets = lambda s: [l for l in visible(s) if re.match(r"\s*[-*] ", l)]  # noqa: E731
    is_table = lambda s: bool(re.search(r"^\s*\|[^|\n]+\|[^|\n]+\|\s*\n\s*\|\s*:?-+:?\s*\|\s*:?-+:?\s*\|\s*$", s, re.M))  # noqa: E731
    first, rest = slides[0], slides[1:]
    titles = [re.match(r"## (.+)", visible(s)[0]) if visible(s) else None for s in rest]
    tables = [s for s in rest if is_table(s)]
    content = [s for s in rest if not is_table(s)]
    return [
        "marp: true" in front and re.search(rf"theme:\s*{theme}\b", front) is not None,
        len(slides) == n_slides,
        len(visible(first)) == 2 and visible(first)[0].startswith("# ") and not bullets(first),
        bool(rest) and all(t_ and len(t_.group(1).split()) <= 6 for t_ in titles),
        bool(content) and all(3 <= len(bullets(s)) <= 5 for s in content),
        all(len(_words(b)) <= 12 and not b.rstrip().endswith(".") for s in rest for b in bullets(s)),
        len(tables) == 1 and not bullets(tables[0]),
        all(notes(s) and len(_words(" ".join(notes(s)))) >= 5 for s in slides),
        bool(titles) and titles[-1] is not None and titles[-1].group(1).strip().strip("*") == last,
    ]


def deck_design_score(text, n_slides, last, theme):
    """Share of nine design rules a Marp deck satisfies (see run_deck)."""
    checks = deck_rule_checks(text, n_slides, last, theme)
    return sum(checks) / len(checks)


# v8: the same decks rendered (node/deck_image: Marp Core 4.3.1 with HTML disabled, headless Chromium, 1280x720 slides)
DECK_IMAGE = "wsp-deck:v8"
DECK_RENDER_CHECKS = ["fits on the slides", "no visible markup", "titles on one line", "no empty or repeated slide"]
DECK_MARKUP = re.compile(r"\*\*|__|```|<!--|-->|^\s*#{1,6}(\s|$)|</?[A-Za-z][^>\n]*>|^\s*\|.*\|\s*$", re.M)


def deck_distinct_share(texts, overlap=0.8):
    """Share of a rendered deck's slides that are non-empty and not a repeat: a slide whose words (lower-cased, digits and
    punctuation dropped) overlap an earlier kept slide's by `overlap` or more (Jaccard) is a repeat. Repeated slides
    measured 0.95-1.0 on the first test decks, distinct ones at most 0.7."""
    kept = []
    for t in texts:
        w = set(re.sub(r"[\d\W_]+", " ", t.lower()).split())
        if w and all(len(w & k) / len(w | k) < overlap for k in kept):
            kept.append(w)
    return len(kept) / len(texts) if texts else 0.0


def deck_render_checks(r):
    """Four checks on a rendered deck (one result of node/deck_image/render.js): no content runs off a slide (Marp clips
    it), no markdown or HTML syntax shows in the rendered text, every slide title fits on one line, and no slide is empty
    or repeats another (deck_distinct_share)."""
    if not r or "error" in r or not r.get("slides"):
        return [False] * len(DECK_RENDER_CHECKS)
    lines = [n for n in r["title_lines"] if n is not None]
    return [not any(r["overflow"]),
            not any(DECK_MARKUP.search(t) for t in r["text"]),
            bool(lines) and all(n == 1 for n in lines),
            deck_distinct_share(r["text"]) == 1.0]


def _deck_exec(jobs, shots_dir=None):
    """Render decks in the deck sandbox (no network, read-only root, non-root user): one measurement dict per job.
    shots_dir: also write deck_<i>.png there (all slides of a deck stacked)."""
    import subprocess
    cmd = ["docker", "run", "--rm", "-i", "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,exec,size=2g",
           "--user", "65534:65534", "--memory", "4g", "--cpuset-cpus", "96-111", "--pids-limit", "4096",
           *(["-v", f"{shots_dir}:/out"] if shots_dir else []), DECK_IMAGE, *(["--shots"] if shots_dir else [])]
    r = subprocess.run(cmd, input=json.dumps([{"md": j["md"]} for j in jobs]), capture_output=True, text=True,
                       timeout=300 + 20 * len(jobs))
    try:
        res = json.loads(r.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        raise RuntimeError("deck sandbox failed: " + r.stderr[-800:])
    if len(res) != len(jobs):
        raise RuntimeError(f"deck sandbox returned {len(res)} results for {len(jobs)} decks")
    errs = [x["error"] for x in res if "error" in x]
    if len(errs) > len(res) // 2:  # one odd deck scores 0 on the rendered checks; most decks failing is the sandbox
        raise RuntimeError(f"deck sandbox failed on {len(errs)} of {len(res)} decks: {errs[0]}")
    return res


def _deck_resolve(jobs):
    """Per deck: the share of the 13 checks that hold, times the share of its slides that are distinct (operator, Nov 5:
    a deck that repeats one slide is not a good deck, so repetition costs in proportion, not as one check of 13)."""
    res = _deck_exec(jobs)
    rend = [deck_render_checks(r) for r in res]
    dist = [deck_distinct_share(r["text"]) if r.get("slides") and "error" not in r else 0.0 for r in res]
    n, nr, nk = len(jobs), len(jobs[0]["rules"]), len(DECK_RENDER_CHECKS)
    per = [(sum(j["rules"]) + sum(c)) / (nr + nk) * d for j, c, d in zip(jobs, rend, dist)]
    return {"score": sum(per) / n, "n": n, "distinct": sum(dist) / n, "rules": sum(sum(j["rules"]) for j in jobs) / (nr * n),
            "render": sum(sum(c) for c in rend) / (nk * n),
            "render_checks": {name: sum(c[i] for c in rend) / n for i, name in enumerate(DECK_RENDER_CHECKS)}}


def run_deck(tok, gen, split, n=None, render=False):
    rng = rng_for(split, 14)
    tasks = [(t, a, rng.randint(5, 8), rng.choice(["Summary", "Next steps", "Key takeaways"]), rng.choice(["default", "gaia", "uncover"]))
             for t in DECK_TOPICS[split] for a in DECK_AUDIENCE[split]][:n]
    prompts = [chat(tok, f"Write a {k}-slide presentation about {t} for {a} in Marp markdown. Start with front matter (a line "
                         f"---, then marp: true and theme: {th}, then ---) and separate slides with lines containing only ---. "
                         f"Design rules: the first slide is a title slide with a '# ' title and one subtitle line, nothing "
                         f"else; every other slide has a '## ' title of at most 6 words; content slides have 3 to 5 bullet "
                         f"points of at most 12 words each, with no full stop at the end; exactly one slide holds a "
                         f"two-column comparison table in markdown and no bullets; every slide has speaker notes in an HTML "
                         f"comment <!-- ... --> of at least 5 words; the last slide is titled '{m}'.", thinking=False)
               for t, a, k, m, th in tasks]
    outs = gen(prompts, 1536, 0.0, None)
    if render:  # v8: the nine text rules plus the four rendered checks, the rendering done on the node host
        return _finish("deck", [{"kind": "deck", "md": _deck_text(o), "rules": deck_rule_checks(o, k, m, th)}
                                for o, (_, _, k, m, th) in zip(outs, tasks)])
    return {"score": sum(deck_design_score(o, k, m, th) for o, (_, _, k, m, th) in zip(outs, tasks)) / len(tasks), "n": len(tasks)}


# ------------------------------------------------------------------ v8 polyglot: whole programs over stdin / stdout
PROG_IMAGE = "wsp-polyglot:v8"
PROG_LANGS = {  # id: (name in the prompt, code fence, toolchain note); node/polyglot_image/runner.py runs them
    "python": ("Python", "python", "Python 3.11."),
    "java": ("Java", "java", "Java 17; the class must be named Main."),
    "rust": ("Rust", "rust", "Rust 1.63, 2021 edition, standard library only."),
    "c": ("C", "c", "C11, compiled with gcc."),
    "go": ("Go", "go", "Go 1.19, standard library only."),
    "kotlin": ("Kotlin", "kotlin", "Kotlin 1.3, compiled with kotlinc (read lines with readLine()!!)."),
    "ruby": ("Ruby", "ruby", "Ruby 3.1."),
    "lua": ("Lua", "lua", "Lua 5.4."),
    "perl": ("Perl", "perl", "Perl 5.36."),
    "crystal": ("Crystal", "crystal", "Crystal 1.6."),
    "nim": ("Nim", "nim", "Nim 1.6."),
    "ada": ("Ada", "ada", "GNAT 12; the main procedure must be named Main."),
    "vala": ("Vala", "vala", "Vala 0.56, compiled with valac."),
    "tcl": ("Tcl", "tcl", "Tcl 8.6, run with tclsh."),
    "haxe": ("Haxe", "haxe", "Haxe 4.2; the class must be named Main (it is run with haxe --run Main)."),
    "pike": ("Pike", "pike", "Pike 8.0."),
    "forth": ("Forth", "forth", "Gforth 0.7: the file is loaded with gforth and then bye runs."),
    "r": ("R", "r", "R 4.2, run with Rscript; read standard input with file(\"stdin\")."),
    "sql": ("SQL", "sql", "SQLite 3.40. There is nothing to read from standard input: the input lines are the rows of a table "
            "input(i INTEGER PRIMARY KEY, line TEXT), i = 1, 2, ... in input order. The program is run by the sqlite3 shell in "
            "list mode without headers, and the rows its statements print are its output."),
    "awk": ("AWK", "awk", "GNU awk, run with gawk -f."),
    "octave": ("GNU Octave", "octave", "Octave 7, run with octave-cli."),
    "prolog": ("Prolog", "prolog", "SWI-Prolog 9, run with swipl; start the program with :- initialization(main, main)."),
    "golfscript": ("GolfScript", "golfscript", "The original Ruby interpreter: the whole input is on the stack as one string when "
                   "the program starts, and the stack is printed when it ends. Ruby code inside strings is disabled."),
    "shakespeare": ("the Shakespeare Programming Language", "spl", "Run by the shakespearelang interpreter: 'Listen to your "
                    "heart!' reads a number, 'Open your mind!' reads a character (-1 at the end of the input), 'Open your "
                    "heart!' prints a number and 'Speak your mind!' prints a character."),
    "cobol": ("COBOL", "cobol", "GnuCOBOL 3.1, fixed or free format."),
    "algol68": ("ALGOL 68", "algol68", "Run by Algol 68 Genie (a68g)."),
    "fortran": ("Fortran", "fortran", "Fortran 90 or later, compiled with gfortran."),
    "pascal": ("Pascal", "pascal", "Free Pascal 3.2."),
    "basic": ("BASIC", "basic", "yabasic 2.9; read with an empty prompt, for example input \"\" n."),
    "haskell": ("Haskell", "haskell", "GHC 9.0, base library only."),
    "ocaml": ("OCaml", "ocaml", "OCaml 4.13, run as a script with the ocaml toplevel (standard library only)."),
    "erlang": ("Erlang", "erlang", "Erlang/OTP 25, run as an escript (define main(_))."),
    "commonlisp": ("Common Lisp", "lisp", "SBCL 2.2, run with sbcl --script."),
    "scheme": ("Scheme", "scheme", "GNU Guile 3.0."),
    "sml": ("Standard ML", "sml", "Poly/ML 5.7: define fun main () = ...; the file is compiled with polyc and main () runs."),
}
PROG_GROUPS = {"mainstream": ["python", "java", "rust", "c"],
               "general, rarer": ["go", "kotlin", "ruby", "lua", "perl", "crystal", "nim", "ada", "vala", "tcl", "haxe", "pike", "forth"],
               "specialist": ["r", "sql", "awk", "octave", "prolog"],
               "esoteric": ["golfscript", "shakespeare"],
               "legacy": ["cobol", "algol68", "fortran", "pascal", "basic"],
               "functional": ["haskell", "ocaml", "erlang", "commonlisp", "scheme", "sml"]}
_LIST = "The first line holds k; each of the next k lines holds one integer"


def _io(*vals):
    return "".join(f"{v}\n" for v in vals)


def _gl(rng, lo=-100, hi=100, kmin=1, kmax=10):
    return [rng.randint(lo, hi) for _ in range(rng.randint(kmin, kmax))]


def _word(rng, a=1, b=20):
    return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(a, b)))


def _pal(rng):
    w = _word(rng, 1, 10)
    return w + w[::-1][rng.randint(0, 1):] if rng.random() < 0.5 else _word(rng, 2, 20)


def _sorted_or_not(rng):
    xs = _gl(rng, kmin=2)
    return sorted(xs) if rng.random() < 0.5 else xs


def _two_distinct(rng):
    while True:
        xs = _gl(rng, kmin=2)
        if len(set(xs)) > 1:
            return xs


def _isprime(n):
    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))


def _collatz(n):
    c = 0
    while n != 1:
        n, c = (n // 2 if n % 2 == 0 else 3 * n + 1), c + 1
    return c


def _fizz(i):
    return "FizzBuzz" if i % 15 == 0 else "Fizz" if i % 3 == 0 else "Buzz" if i % 5 == 0 else str(i)


def _fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def _yn(b):
    return "yes" if b else "no"


def _lio(xs):
    return _io(len(xs), *xs)


# (statement, test generator: rng -> (input, expected output)). Even positions are held out, odd ones are practice; every
# language gets the same tests (random.Random(f"prog-{position}"), four per task).
PROG_PROBLEMS = [
    ("Read two integers a and b (one per line, each between -1000 and 1000) and print a + b.",
     lambda r: (lambda a, b: (_io(a, b), _io(a + b)))(r.randint(-1000, 1000), r.randint(-1000, 1000))),
    ("Read an integer n (1 <= n <= 1000) and print the sum 1 + 2 + ... + n.",
     lambda r: (lambda n: (_io(n), _io(n * (n + 1) // 2)))(r.randint(1, 1000))),
    ("Read an integer n (0 <= n <= 12) and print n! (the factorial of n; 0! = 1).",
     lambda r: (lambda n: (_io(n), _io(__import__("math").factorial(n))))(r.randint(0, 12))),
    ("Read an integer n (1 <= n <= 20) and print the integers from 1 to n, one per line.",
     lambda r: (lambda n: (_io(n), _io(*range(1, n + 1))))(r.randint(1, 20))),
    (f"{_LIST} (1 <= k <= 10, each integer between -100 and 100). Print the largest of the k integers.",
     lambda r: (lambda xs: (_lio(xs), _io(max(xs))))(_gl(r))),
    (f"{_LIST} (1 <= k <= 10, each integer between -100 and 100). Print how many of the k integers are even.",
     lambda r: (lambda xs: (_lio(xs), _io(sum(x % 2 == 0 for x in xs))))(_gl(r))),
    (f"{_LIST} (1 <= k <= 10, each integer between -100 and 100). Print the k integers in reverse order, one per line.",
     lambda r: (lambda xs: (_lio(xs), _io(*xs[::-1])))(_gl(r))),
    ("Read an integer n (2 <= n <= 10000). Print yes if n is prime and no otherwise.",
     lambda r: (lambda n: (_io(n), _io(_yn(_isprime(n)))))(r.choice([r.randint(2, 10000), r.choice([2, 3, 97, 7919, 9973])]))),
    ("Read two integers a and b (one per line, 1 <= a, b <= 10000) and print their greatest common divisor.",
     lambda r: (lambda a, b: (_io(a, b), _io(__import__("math").gcd(a, b))))(r.randint(1, 10000), r.randint(1, 10000))),
    ("Read an integer n (0 <= n <= 40) and print the n-th Fibonacci number, where F(0) = 0, F(1) = 1 and "
     "F(n) = F(n-1) + F(n-2).",
     lambda r: (lambda n: (_io(n), _io(_fib(n))))(r.randint(0, 40))),
    ("Read an integer n (0 <= n <= 1000000000) and print the sum of its decimal digits.",
     lambda r: (lambda n: (_io(n), _io(sum(map(int, str(n))))))(r.randint(0, 10 ** 9))),
    ("Read an integer n (1 <= n <= 100000) and print n in binary, without leading zeros.",
     lambda r: (lambda n: (_io(n), _io(bin(n)[2:])))(r.randint(1, 100000))),
    ("Read an integer n (1 <= n <= 30). For each i from 1 to n print, on its own line, FizzBuzz if i is divisible by 15, "
     "Fizz if it is divisible by 3, Buzz if it is divisible by 5, and i itself otherwise.",
     lambda r: (lambda n: (_io(n), _io(*map(_fizz, range(1, n + 1)))))(r.randint(1, 30))),
    ("Read an integer n (1 <= n <= 1000). While n is not 1, replace n by n / 2 if n is even and by 3n + 1 if n is odd. "
     "Print the number of replacements made.",
     lambda r: (lambda n: (_io(n), _io(_collatz(n))))(r.randint(1, 1000))),
    ("Read a word of 1 to 20 lowercase letters and print it reversed.",
     lambda r: (lambda w: (_io(w), _io(w[::-1])))(_word(r))),
    ("Read a word of 1 to 20 lowercase letters and print how many of its letters are vowels (a, e, i, o or u).",
     lambda r: (lambda w: (_io(w), _io(sum(c in "aeiou" for c in w))))(_word(r))),
    ("Read a word of 1 to 20 lowercase letters. Print yes if it reads the same backwards and no otherwise.",
     lambda r: (lambda w: (_io(w), _io(_yn(w == w[::-1]))))(_pal(r))),
    ("Read a word of 1 to 20 lowercase letters and print it in uppercase.",
     lambda r: (lambda w: (_io(w), _io(w.upper())))(_word(r))),
    ("Read an integer n (1 <= n <= 8) and print n lines: line i holds i asterisks (*) and nothing else.",
     lambda r: (lambda n: (_io(n), _io(*("*" * i for i in range(1, n + 1)))))(r.randint(1, 8))),
    ("Read two integers a (1 <= a <= 9) and b (0 <= b <= 9), one per line, and print a to the power b.",
     lambda r: (lambda a, b: (_io(a, b), _io(a ** b)))(r.randint(1, 9), r.randint(0, 9))),
    (f"{_LIST} (1 <= k <= 10, each integer between -100 and 100). Print the k integers in increasing order, one per line.",
     lambda r: (lambda xs: (_lio(xs), _io(*sorted(xs))))(_gl(r))),
    (f"{_LIST} (1 <= k <= 10, each integer between -100 and 100). Print the smallest of the k integers.",
     lambda r: (lambda xs: (_lio(xs), _io(min(xs))))(_gl(r))),
    (f"{_LIST} (1 <= k <= 10, each integer between -100 and 100). Print the running sums, one per line: the first "
     "integer, the sum of the first two, and so on up to the sum of all k.",
     lambda r: (lambda xs: (_lio(xs), _io(*__import__("itertools").accumulate(xs))))(_gl(r))),
    ("Read two integers a and b (one per line, each between -1000 and 1000) and print the absolute value of a - b.",
     lambda r: (lambda a, b: (_io(a, b), _io(abs(a - b))))(r.randint(-1000, 1000), r.randint(-1000, 1000))),
    ("Read an integer n (0 <= n <= 1000000) and print the largest integer whose square is at most n.",
     lambda r: (lambda n: (_io(n), _io(__import__("math").isqrt(n))))(r.randint(0, 10 ** 6))),
    ("Read an integer n (1 <= n <= 1000) and print how many positive integers divide n.",
     lambda r: (lambda n: (_io(n), _io(sum(n % d == 0 for d in range(1, n + 1)))))(r.randint(1, 1000))),
    ("Read an integer n (1 <= n <= 100) and print 1*1 + 2*2 + ... + n*n.",
     lambda r: (lambda n: (_io(n), _io(sum(i * i for i in range(1, n + 1)))))(r.randint(1, 100))),
    ("Read an integer n (1 <= n <= 12) and print n*1, n*2, ..., n*10, one per line.",
     lambda r: (lambda n: (_io(n), _io(*(n * i for i in range(1, 11)))))(r.randint(1, 12))),
    ("Read a year y (1 <= y <= 3000). Print yes if it is a leap year in the Gregorian calendar (divisible by 4, except "
     "that years divisible by 100 are leap years only when divisible by 400) and no otherwise.",
     lambda r: (lambda y: (_io(y), _io(_yn(y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)))))(
         r.choice([r.randint(1, 3000), r.choice([1900, 2000, 2024, 2100, 2400])]))),
    ("Read an integer n (0 <= n <= 100000) and print it in hexadecimal with lowercase digits and no leading zeros.",
     lambda r: (lambda n: (_io(n), _io(format(n, "x"))))(r.randint(0, 100000))),
    (f"{_LIST} (1 <= k <= 6, each integer between -9 and 9). Print the product of the k integers.",
     lambda r: (lambda xs: (_lio(xs), _io(__import__("math").prod(xs))))(_gl(r, -9, 9, 1, 6))),
    (f"{_LIST} (1 <= k <= 10, each integer between 0 and 100). Print the average of the k integers rounded down.",
     lambda r: (lambda xs: (_lio(xs), _io(sum(xs) // len(xs))))(_gl(r, 0, 100))),
    ("The first line holds a lowercase letter c and the second line a word of 1 to 20 lowercase letters. Print how many "
     "times c occurs in the word.",
     lambda r: (lambda c, w: (_io(c, w), _io(w.count(c))))(r.choice("aeinrst"), _word(r))),
    ("Read a word of 1 to 20 lowercase letters and print it with every letter moved 3 places forward in the alphabet, "
     "wrapping around (x becomes a, y becomes b, z becomes c).",
     lambda r: (lambda w: (_io(w), _io("".join(chr((ord(c) - 97 + 3) % 26 + 97) for c in w))))(_word(r))),
    (f"{_LIST} (2 <= k <= 10, each integer between -100 and 100). Print yes if the k integers are in non-decreasing "
     "order and no otherwise.",
     lambda r: (lambda xs: (_lio(xs), _io(_yn(xs == sorted(xs)))))(_sorted_or_not(r))),
    (f"{_LIST} (2 <= k <= 10, each integer between -100 and 100, not all equal). Print the second largest distinct "
     "value among them.",
     lambda r: (lambda xs: (_lio(xs), _io(sorted(set(xs))[-2])))(_two_distinct(r))),
    (f"{_LIST} (1 <= k <= 10, each integer between -100 and 100). Print how many of the k integers are greater than 0.",
     lambda r: (lambda xs: (_lio(xs), _io(sum(x > 0 for x in xs))))(_gl(r))),
    ("Read two integers a (0 <= a <= 1000000) and b (1 <= b <= 1000), one per line, and print the remainder of a "
     "divided by b.",
     lambda r: (lambda a, b: (_io(a, b), _io(a % b)))(r.randint(0, 10 ** 6), r.randint(1, 1000))),
    ("Read three integers (one per line, each between 1 and 100), the side lengths of a triangle. Print yes if each "
     "side is shorter than the sum of the other two and no otherwise.",
     lambda r: (lambda a, b, c: (_io(a, b, c), _io(_yn(a < b + c and b < a + c and c < a + b))))(
         r.randint(1, 100), r.randint(1, 100), r.randint(1, 100))),
    (f"{_LIST} (1 <= k <= 10, each integer between -100 and 100). Print the largest of the k integers minus the smallest.",
     lambda r: (lambda xs: (_lio(xs), _io(max(xs) - min(xs))))(_gl(r))),
]


def prog_tasks(split, n=None):
    """(statement, [(input, expected output)] * 4): the 20 held-out or the 20 practice tasks."""
    out = []
    for i, (stmt, g) in enumerate(PROG_PROBLEMS):
        if (i % 2 == 0) == (split == "heldout"):
            rng = random.Random(f"prog-{i}")
            out.append((stmt, [g(rng) for _ in range(4)]))
    return out[:n]


def _fenced_any(text):
    """The first fenced block, whatever its language tag (algol68, c++, ...), even if the closing fence is missing."""
    m = re.search(r"```[^\n`]*\n(.*?)(?:```|\Z)", strip_think(text), re.S)
    return m.group(1) if m else strip_think(text)


def run_prog(tok, gen, split, n=None, lang="python"):
    """A whole program in one language per task, run on four stdin inputs in the polyglot sandbox (node/polyglot_image):
    it scores if its output tokens equal the expected ones on all four."""
    name, fence, note = PROG_LANGS[lang]
    io = ("Its input and output are described below." if lang == "sql" else
          "It reads its input from standard input and writes its answer to standard output.")
    tasks = prog_tasks(split, n)
    prompts = [chat(tok, f"Write a complete program in {name}. {io} {stmt} {note} Reply with the program only, in one "
                         f"```{fence} block.", thinking=False) for stmt, _ in tasks]
    outs = gen(prompts, 1536, 0.0, None)
    return _finish("prog", [{"kind": "prog", "lang": lang, "src": _fenced_any(o), "tests": t} for o, (_, t) in zip(outs, tasks)])


def _prog_exec(jobs, chunk=3):
    """Run program jobs in the polyglot sandbox (no network, read-only root, non-root user), chunks in parallel
    containers: one {"ok", "why"} per job."""
    import subprocess
    from concurrent.futures import ThreadPoolExecutor

    def run(part):
        cmd = ["docker", "run", "--rm", "-i", "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,exec,size=4g",
               "--user", "65534:65534", "--memory", "6g", "--cpuset-cpus", "144-175", "--pids-limit", "4096", PROG_IMAGE]
        r = subprocess.run(cmd, input=json.dumps([{k: j[k] for k in ("lang", "src", "tests")} for j in part]),
                           capture_output=True, text=True, timeout=900 + 300 * len(part))
        try:
            res = json.loads(r.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            raise RuntimeError("polyglot sandbox failed: " + r.stderr[-800:])
        if len(res) != len(part):
            raise RuntimeError(f"polyglot sandbox returned {len(res)} results for {len(part)} programs")
        return res
    with ThreadPoolExecutor(8) as ex:
        return [x for p in ex.map(run, [jobs[k:k + chunk] for k in range(0, len(jobs), chunk)]) for x in p]


SKILLS = {"ifeval": run_ifeval, "toolcall": run_toolcall, "tictactoe": run_tictactoe, "connectfour": run_connectfour,
          "countdown": run_countdown, "brick": run_brick, "svg": run_svg, "sql": run_sql, "calculus": run_calculus,
          "excel": run_excel, "slides": run_slides, "writing": run_writing, "facts": run_facts, "code": run_code,
          "math": run_math, "mple": run_mple, "clingo": run_clingo, "video": run_video, "scad": run_scad, "deck": run_deck,
          "prog": run_prog, "science": run_science}


def run(skill, tok, gen, split, plain=False, **kw):
    """Run a skill. plain=True replaces the chat template with 'Question: <prompt>\\nAnswer:' and cuts each reply at the
    next '\\nQuestion:' (Qwen3 and Qwen2.5-0.5B base checkpoints: their <|im_start|>/<|im_end|> (and Qwen3's
    <think>) embeddings are untrained)."""
    global PLAIN
    PLAIN = plain
    g = (lambda p, mt, t, stop: gen(p, mt, t, (stop or []) + ["\nQuestion:"])) if plain else gen
    return SKILLS[skill](tok, g, split, **kw)


PRACTICE_N = {"code": 40, "math": 60, "tictactoe": 12, "connectfour": 8, "countdown": 40, "brick": 12, "svg": 30, "sql": 45, "calculus": 30, "excel": 30,
              "ifeval": 50, "mple": 40, "clingo": 15, "video": 24, "scad": 15}
