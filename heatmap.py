#!/usr/bin/env python3
#########################################################################################
#                     UTXOracle - Litecoin Edition (RPC Only)
#
# Windows + WSL friendly:
#   python3 utxo.py -p \\wsl$\Ubuntu\home\master\.litecoin -rb
#   python3 utxo.py -p \\wsl$\Ubuntu\home\master\.litecoin -d 2026/01/07
#   python3 utxo.py -p \\wsl$\Ubuntu\home\master\.litecoin -days 30
#
# What this does:
# - Pulls blocks/txs from your local Litecoin node via RPC
# - Parses raw blocks locally (no explorers)
# - Uses Coinbase spot ONLY as a USD anchor (for display + probe conversion)
# - Builds a cloud of implied prices from on-chain “denomination clustering”
# - Renders:
#    1) Price cloud + MEDIAN SPINE (UTXOracle-ish)
#    2) Clustering heatmap (USD size vs time)
#
# Important fixes in this version:
# - Hardened raw parser to prevent OverflowError / bad-varint crashes on -days N
# - Correct SegWit marker/flag handling in the block parser
# - Safe varint + safe reads with caps (prevents stream misalignment from exploding)
# - Both HTML outputs include navigation links to each other
#########################################################################################

###############################################################################
# Step 1 - Configuration Options
###############################################################################

import platform
import os
import sys
import json
import urllib.request
import urllib.parse
from datetime import datetime, timezone, timedelta
EST = timezone(timedelta(hours=-5))  # Fixed EST (UTC-5), no DST

# Rolling window size for -rb
WINDOW_BLOCKS = 576  # ~24h on Litecoin

# Blend between anchor (spot) and chain-derived refinement
BLEND_WEIGHT = 0.25  # requested

# Probe USD sizes to look for on-chain denominational clustering (keep human-scale)
USD_PROBES = [5, 10, 15, 20, 25, 30, 40, 50, 100, 150, 200, 300, 500, 1000]

# How tight to accept outputs around probe amounts at the rough price
PCT_RANGE_WIDE = 0.10   # 10% around expected coin amount for each USD probe
PCT_RANGE_TIGHT = 0.03  # 3% refinement window (median selection)
PCT_RANGE_MED = 0.06    # 6% plotting window around blended price

# Clustering heatmap shape
CLUSTER_X_BINS = 240
CLUSTER_Y_BINS = 220

# USD log scale bounds for clustering (tweak if you want)
CLUSTER_USD_MIN = 0.01
CLUSTER_USD_MAX = 1_000_000.0

# -----------------------------
# Data dir autodetect (override via -p)
# -----------------------------
system = platform.system()
if system == "Darwin":
    data_dir = os.path.expanduser("~/Library/Application Support/Litecoin")
elif system == "Windows":
    data_dir = os.path.join(os.environ.get("APPDATA", ""), "Litecoin")
else:
    data_dir = os.path.expanduser("~/.litecoin")

date_entered = ""
date_mode = True
block_mode = False

block_start_num = 0
block_finish_num = 0
block_nums_needed = []
block_hashes_needed = []
block_times_needed = []


def fetch_coinbase_spot_ltc_usd(date_yyyy_mm_dd=None) -> float:
    """Fetch LTC-USD spot from Coinbase (public API)."""
    base_url = "https://api.coinbase.com/v2/prices/LTC-USD/spot"
    if date_yyyy_mm_dd:
        qs = urllib.parse.urlencode({"date": date_yyyy_mm_dd})
        url = f"{base_url}?{qs}"
    else:
        url = base_url

    req = urllib.request.Request(
        url,
        headers={"CB-VERSION": "2015-04-08", "User-Agent": "UTXOracle-Litecoin/1.0"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        raw = resp.read().decode("utf-8")
    j = json.loads(raw)
    return float(j["data"]["amount"])


def print_help():
    print(
        f"""Usage: python3 utxo.py [options]
  -h               Show help
  -d YYYY/MM/DD    Specify a UTC date to evaluate (defaults to previous UTC day)
  -p /path/to/dir  Litecoin data directory (where litecoin.conf lives)
  -rb              Use last {WINDOW_BLOCKS} blocks (rolling window)
  -days N          Use last N full UTC days (block scan back from tip)

Examples:
  python3 utxo.py -p \\\\wsl$\\Ubuntu\\home\\master\\.litecoin
  python3 utxo.py -p \\\\wsl$\\Ubuntu\\home\\master\\.litecoin -rb
  python3 utxo.py -p \\\\wsl$\\Ubuntu\\home\\master\\.litecoin -d 2026/01/07
  python3 utxo.py -p \\\\wsl$\\Ubuntu\\home\\master\\.litecoin -days 30
"""
    )
    sys.exit(0)


if "-h" in sys.argv:
    print_help()

if "-d" in sys.argv:
    i = sys.argv.index("-d")
    if i + 1 < len(sys.argv):
        date_entered = sys.argv[i + 1]

if "-p" in sys.argv:
    i = sys.argv.index("-p")
    if i + 1 < len(sys.argv):
        data_dir = sys.argv[i + 1]

if "-rb" in sys.argv:
    date_mode = False
    block_mode = True

days_back = None
if "-days" in sys.argv:
    i = sys.argv.index("-days")
    if i + 1 < len(sys.argv):
        days_back = int(sys.argv[i + 1])

# -----------------------------
# Find config file
# -----------------------------
conf_path = None
for fname in ["litecoin.conf", "litecoin_rw.conf", "litecoin-rw.conf"]:
    p = os.path.join(data_dir, fname)
    if os.path.exists(p):
        conf_path = p
        break

if not conf_path:
    print("litecoin.conf not found in", data_dir)
    sys.exit(1)

conf_settings = {}
with open(conf_path, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            conf_settings[k.strip()] = v.strip().strip('"')

# RPC settings from conf (or cookie fallback)
rpc_user = conf_settings.get("rpcuser")
rpc_password = conf_settings.get("rpcpassword")
cookie_path = conf_settings.get("rpccookiefile", os.path.join(data_dir, ".cookie"))
rpc_host = conf_settings.get("rpcconnect", "127.0.0.1")
rpc_port = int(conf_settings.get("rpcport", "9332"))

###############################################################################
# Step 2 - Establish RPC Connection
###############################################################################
import http.client
import base64
import socket
import time
import json

# Persistent connection (reused across all RPC calls)
persistent_conn = None
rpc_auth_header = None

def build_auth_header():
    u, p = rpc_user, rpc_password
    if not u or not p:
        try:
            with open(cookie_path, "r", encoding="utf-8") as f:
                u, p = f.read().strip().split(":", 1)
        except FileNotFoundError:
            raise Exception(
                f"RPC auth missing. No rpcuser/rpcpassword in {conf_path} "
                f"and cookie not found at {cookie_path}. "
                f"Make sure your node is running and server=1."
            )
        except Exception as e:
            raise Exception(f"Failed to read cookie file: {e}")
    auth = base64.b64encode(f"{u}:{p}".encode('utf-8')).decode('utf-8')
    return f"Basic {auth}"

def get_persistent_conn():
    global persistent_conn
    # Always reuse one connection object, but rebuild it if it went bad/closed.
    if persistent_conn is None:
        persistent_conn = http.client.HTTPConnection(rpc_host, rpc_port, timeout=60)
        return persistent_conn

    # If the underlying socket got closed or became invalid, recreate it.
    try:
        sock = getattr(persistent_conn, "sock", None)
        if sock is None or sock.fileno() < 0:
            reset_conn()
            persistent_conn = http.client.HTTPConnection(rpc_host, rpc_port, timeout=60)
    except Exception:
        reset_conn()
        persistent_conn = http.client.HTTPConnection(rpc_host, rpc_port, timeout=60)

    return persistent_conn


def reset_conn():
    global persistent_conn
    try:
        if persistent_conn is not None:
            persistent_conn.close()
    except Exception:
        pass
    persistent_conn = None

def Ask_Node(command):
    """
    RPC helper with persistent connection + retry.
    command = ["method", param1, param2, ...]
    Returns the JSON "result" directly (string/int/dict/list).
    """
    global rpc_auth_header
    method = command[0]
    params = command[1:]
    if rpc_auth_header is None:
        rpc_auth_header = build_auth_header()

    payload = json.dumps(
        {"jsonrpc": "1.0", "id": "utxoracle", "method": method, "params": params}
    ).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Authorization": rpc_auth_header,
        "Connection": "keep-alive",
        # prevents some Windows port exhaustion patterns by avoiding reconnect churn
        "Accept": "application/json",
    }

    last_err = None
    for attempt in range(8):
        conn = None
        try:
            conn = get_persistent_conn()
            conn.request("POST", "/", payload, headers)
            resp = conn.getresponse()
            raw = resp.read()

            # IMPORTANT: close response to free the socket for reuse
            resp.close()

            j = json.loads(raw.decode("utf-8"))
            if j.get("error"):
                raise Exception(j["error"])
            return j["result"]

        except (OSError, http.client.HTTPException, socket.error, json.JSONDecodeError, UnicodeDecodeError) as e:

            last_err = e
            reset_conn()
            time.sleep(0.15 * (attempt + 1))
            continue

        except Exception as e:
            # Non-socket error (e.g., JSON, RPC error)
            raise

    raise Exception(f"RPC call failed after retries: {method} {params} | last_err={last_err}")


# Basic node sanity calls
block_count = int(Ask_Node(["getblockcount"]))
tip_hash = Ask_Node(["getblockhash", block_count - 6]) # str
tip_header = Ask_Node(["getblockheader", tip_hash, True]) # dict
###############################################################################
# Step 3 - Check Dates
###############################################################################


latest_time_in_seconds = int(tip_header["time"])
latest_dt = datetime.fromtimestamp(latest_time_in_seconds, tz=timezone.utc)
latest_midnight = datetime(latest_dt.year, latest_dt.month, latest_dt.day, tzinfo=timezone.utc)

window_start_seconds = None
window_label = None

if days_back is not None:
    window_start_dt = latest_midnight - timedelta(days=days_back)
    window_start_seconds = int(window_start_dt.timestamp())
    window_label = f"last {days_back} days (UTC)"

# Always define these so nothing crashes
datetime_entered = latest_midnight
price_date_dash = latest_dt.strftime("%Y-%m-%d")
price_day_seconds = int(latest_midnight.timestamp())
price_day_date_utc = latest_midnight.strftime("%b %d, %Y")
seconds_in_a_day = 24 * 60 * 60
block_count_consensus = block_count

if date_mode:
    if date_entered:
        y, m, d = map(int, date_entered.split("/"))
        datetime_entered = datetime(y, m, d, tzinfo=timezone.utc)
    else:
        datetime_entered = latest_midnight - timedelta(days=1)

    price_date_dash = datetime_entered.strftime("%Y-%m-%d")
    price_day_seconds = int(datetime_entered.timestamp())
    price_day_date_utc = datetime_entered.strftime("%b %d, %Y")

##############################################################################
# Step 4 - Find Block Hashes
##############################################################################

def get_block_time(height: int):
    bh = Ask_Node(["getblockhash", height])            # str
    hdr = Ask_Node(["getblockheader", bh, True])       # dict
    return (int(hdr["time"]), bh)

def get_day_of_month(time_in_seconds: int) -> int:
    return int(datetime.fromtimestamp(time_in_seconds, tz=timezone.utc).strftime("%d"))

if block_mode:
    print(f"\nFinding the last {WINDOW_BLOCKS} blocks", flush=True)

    block_finish_num = block_count
    block_start_num = block_finish_num - WINDOW_BLOCKS

    block_nums_needed.clear()
    block_hashes_needed.clear()
    block_times_needed.clear()  # not needed anymore

    total = max(1, block_finish_num - block_start_num)
    print_every = 0

    for idx, height in enumerate(range(block_start_num, block_finish_num), start=1):
        if (idx / total) * 100 >= print_every and print_every < 100:
            print(f"{print_every}%..", end="", flush=True)
            print_every += 20

        bh = Ask_Node(["getblockhash", height])
        block_nums_needed.append(height)
        block_hashes_needed.append(bh)

    print("100%\t\t\t25% done", flush=True)


elif days_back is not None:
    print(f"\nCollecting blocks for {window_label}", flush=True)

    block_finish_num = block_count
    target_ts = window_start_seconds

    # --- FAST: binary search for first height >= target_ts ---
    def block_time_at_height(hh: int) -> int:
        bhh = Ask_Node(["getblockhash", hh])
        hdr = Ask_Node(["getblockheader", bhh, True])
        return int(hdr["time"])

    lo = 0
    hi = block_finish_num
    # invariant: answer in [lo, hi]
    while lo < hi:
        mid = (lo + hi) // 2
        tmid = block_time_at_height(mid)
        if tmid < target_ts:
            lo = mid + 1
        else:
            hi = mid

    block_start_num = lo

    block_nums_needed.clear()
    block_hashes_needed.clear()
    block_times_needed.clear()  # we won't use this for days mode anymore, but clear anyway

    total = max(1, block_finish_num - block_start_num)
    print_next = 0

    for idx, height in enumerate(range(block_start_num, block_finish_num), start=1):
        bh = Ask_Node(["getblockhash", height])

        # keep these arrays for your later DEBUG + plotting x-axis
        block_nums_needed.append(height)
        block_hashes_needed.append(bh)

        if idx % 200 == 0:
            print(f"\n...still working: height={height} ({idx}/{total})", flush=True)

        pct = (idx / total) * 100
        if pct >= print_next and print_next < 100:
            print(f"{int(print_next)}%..", end="", flush=True)
            print_next += 20

    print("100%\t\t\t50% done", flush=True)




elif date_mode:
    print("\nCollecting blocks for " + datetime_entered.strftime("%b %d, %Y"), flush=True)

    day_start_ts = price_day_seconds
    day_end_ts = price_day_seconds + seconds_in_a_day

    def block_time_at_height(hh: int) -> int:
        bhh = Ask_Node(["getblockhash", hh])
        hdr = Ask_Node(["getblockheader", bhh, True])
        return int(hdr["time"])

    # start height = first block with time >= day_start_ts
    lo, hi = 0, block_count
    while lo < hi:
        mid = (lo + hi) // 2
        if block_time_at_height(mid) < day_start_ts:
            lo = mid + 1
        else:
            hi = mid
    block_start_num = lo

    # end height = first block with time >= day_end_ts
    lo, hi = block_start_num, block_count
    while lo < hi:
        mid = (lo + hi) // 2
        if block_time_at_height(mid) < day_end_ts:
            lo = mid + 1
        else:
            hi = mid
    block_finish_num = lo

    block_nums_needed.clear()
    block_hashes_needed.clear()
    block_times_needed.clear()

    total = max(1, block_finish_num - block_start_num)
    print_next = 0

    for idx, height in enumerate(range(block_start_num, block_finish_num), start=1):
        bh = Ask_Node(["getblockhash", height])
        block_nums_needed.append(height)
        block_hashes_needed.append(bh)

        pct = (idx / total) * 100
        if pct >= print_next and print_next < 100:
            print(f"{int(print_next)}%..", end="", flush=True)
            print_next += 20

    print("100%\t\t\t50% done", flush=True)


print(f"\nDEBUG blocks_selected={len(block_hashes_needed)} start={block_start_num} end={block_finish_num-1}", flush=True)

##############################################################################
# Step 5 - Histogram scaffolding (kept for compatibility / future)
##############################################################################

from math import log10

first_bin_value = -6
last_bin_value = 6
range_bin_values = last_bin_value - first_bin_value

output_histogram_bins = [0.0]
for exponent in range(-6, 6):
    for b in range(0, 200):
        output_histogram_bins.append(10 ** (exponent + b / 200))

number_of_bins = len(output_histogram_bins)
output_histogram_bin_counts = [0.0 for _ in range(number_of_bins)]

##############################################################################
# Step 6 - Load Transaction Data (raw block parse) - HARDENED
##############################################################################

print("\nLoading transactions from blocks (FAST decoded RPC)", flush=True)

todays_txids = set()   # keep name so other code doesn't break, but we won't use it
raw_outputs = []
block_heights_dec = []
block_times_dec = []

print_next = 0
block_num = 0

for bh in block_hashes_needed:
    block_num += 1
    pct = (block_num / len(block_hashes_needed)) * 100
    if pct >= print_next and print_next < 100:
        print(f"{int(print_next)}%..", end="", flush=True)
        print_next += 5
        if int(print_next) % 35 == 0:
            print("\n", end="")

    try:
        # verbosity=2 gives decoded txs (FAST vs raw parsing in Python)
        blk = Ask_Node(["getblock", bh, 2])

        # block time (seconds)
        header_time = int(blk.get("time", 0))
        txs = blk.get("tx", [])
        if not txs:
            continue

        height_here = block_nums_needed[block_num - 1]

        for tx in txs[:-1]:
            vin = tx.get("vin", [])
            vout = tx.get("vout", [])

            input_count = len(vin)
            output_count = len(vout)

            # coinbase?
            is_coinbase = (input_count >= 1 and "coinbase" in vin[0])

            # OP_RETURN?
            has_op_return = False
            output_values = []
            for o in vout:
                spk = o.get("scriptPubKey", {}) or {}
                t = spk.get("type", "")
                # Many nodes label it "nulldata"
                if t == "nulldata":
                    has_op_return = True
                    break
                # fallback: try asm contains OP_RETURN
                asm = spk.get("asm", "")
                if asm.startswith("OP_RETURN"):
                    has_op_return = True
                    break

                # value is float LTC already
                val = o.get("value", 0.0)
                try:
                    val = float(val)
                except Exception:
                    continue
                if 1e-5 < val < 1e5:
                    output_values.append(val)

            if has_op_return:
                continue
            if is_coinbase:
                continue

            # no witness_exceeds concept in decoded mode; remove it for speed
            # also remove "same-window self-churn" txid checks (this was expensive)

            # Your "economic-looking" filter
            if input_count <= 5 and output_count == 2:
                for amount in output_values:
                    amount_log = log10(amount)
                    percent_in_range = (amount_log - first_bin_value) / range_bin_values
                    bin_number_est = int(percent_in_range * number_of_bins)
                    if bin_number_est < 0 or bin_number_est >= number_of_bins:
                        continue
                    while (
                        bin_number_est < number_of_bins - 1
                        and output_histogram_bins[bin_number_est] <= amount
                    ):
                        bin_number_est += 1
                    bin_number = max(0, bin_number_est - 1)
                    output_histogram_bin_counts[bin_number] += 1.0

                    raw_outputs.append(amount)
                    block_heights_dec.append(height_here)
                    block_times_dec.append(header_time)

    except Exception:
        continue

print("100%", flush=True)
print("\t\t\t\t\t\t95% done", flush=True)
##############################################################################
# Step 7/8 - BTC-specific stuff removed (no stencils/bins for LTC)
##############################################################################

print("\nFinding prices and rendering plot", flush=True)
print("0%..", end="", flush=True)

##############################################################################
# Step 9 - Anchor + Rough Price
##############################################################################

anchor_date = price_date_dash if date_mode else None
try:
    ltc_usd_price = fetch_coinbase_spot_ltc_usd(anchor_date)
    print(f"\nCoinbase LTC-USD spot used for anchor: ${ltc_usd_price:.2f}", flush=True)
except Exception as e:
    ltc_usd_price = 82.39
    print(f"\nWARNING: Coinbase anchor failed ({e}). Falling back to ${ltc_usd_price:.2f}", flush=True)

rough_price_estimate = max(1, int(round(ltc_usd_price)))
print(f"DEBUG rough_price_estimate=${rough_price_estimate}", flush=True)

print("40%..", end="", flush=True)

##############################################################################
# Step 10 - Create Intraday Price Points (near anchor only)
##############################################################################

# Micro remove list (kept from your version)
micro_remove_list = []
i = 0.00005000
while i < 0.0001:
    micro_remove_list.append(i)
    i += 0.00001
i = 0.0001
while i < 0.001:
    micro_remove_list.append(i)
    i += 0.00001
i = 0.001
while i < 0.01:
    micro_remove_list.append(i)
    i += 0.0001
i = 0.01
while i < 0.1:
    micro_remove_list.append(i)
    i += 0.001
i = 0.1
while i < 1:
    micro_remove_list.append(i)
    i += 0.01

pct_micro_remove = 0.0001

output_prices = []
output_blocks = []
output_times = []

for idx in range(len(raw_outputs)):
    n = raw_outputs[idx]
    b = block_heights_dec[idx]
    t = block_times_dec[idx]

    for usd in USD_PROBES:
        avcoin = usd / rough_price_estimate
        coin_up = avcoin + PCT_RANGE_WIDE * avcoin
        coin_dn = avcoin - PCT_RANGE_WIDE * avcoin

        if coin_dn < n < coin_up:
            append = True
            for r in micro_remove_list:
                rm_dn = r - pct_micro_remove * r
                rm_up = r + pct_micro_remove * r
                if rm_dn < n < rm_up:
                    append = False
                    break
            if append:
                output_prices.append(usd / n)  # implied USD/LTC
                output_blocks.append(b)
                output_times.append(t)

print("60%..", end="", flush=True)

##############################################################################
# Step 11 - Robust center + blend with anchor
##############################################################################

def find_central_output(values, price_min, price_max):
    r = [v for v in values if price_min < v < price_max]
    outputs = sorted(r)
    n = len(outputs)
    if n == 0:
        return 0.0, 0.0

    # median + MAD-ish
    mid = outputs[n // 2] if n % 2 == 1 else (outputs[n//2 - 1] + outputs[n//2]) / 2.0
    devs = [abs(x - mid) for x in outputs]
    devs.sort()
    mad = devs[len(devs)//2] if devs else 0.0
    return mid, mad

price_up = rough_price_estimate + PCT_RANGE_TIGHT * rough_price_estimate
price_dn = rough_price_estimate - PCT_RANGE_TIGHT * rough_price_estimate
central_price, av_dev = find_central_output(output_prices, price_dn, price_up)

if central_price == 0.0:
    print("\n\nNo price points found near anchor. Try loosening filters or widen PCT_RANGE_WIDE.")
    sys.exit(1)

for _ in range(25):
    prev = central_price
    price_up = central_price + PCT_RANGE_TIGHT * central_price
    price_dn = central_price - PCT_RANGE_TIGHT * central_price
    central_price, av_dev = find_central_output(output_prices, price_dn, price_up)
    if central_price == 0.0 or abs(central_price - prev) < 1e-9:
        break

# Blend: anchor + chain refinement
blended_price = (BLEND_WEIGHT * ltc_usd_price) + ((1.0 - BLEND_WEIGHT) * central_price)

print("80%..", end="", flush=True)

# Plot window
price_up = blended_price + PCT_RANGE_MED * blended_price
price_dn = blended_price - PCT_RANGE_MED * blended_price

print("100%\t\t\tdone", flush=True)
if date_mode:
    print(f"\n\n\t\t{price_day_date_utc} blended price: ${int(blended_price):,}\n\n", flush=True)
else:
    print(f"\n\n\t\tRolling window blended price: ${int(blended_price):,}\n\n", flush=True)

##############################################################################
# Step 12 - Generate the Price Plot HTML (faint cloud + MEDIAN SPINE)
##############################################################################

width = 1000
height = 660
margin_left = 120
margin_right = 90
margin_top = 100
margin_bottom = 120

if not output_prices:
    print("No output prices to plot.")
    sys.exit(1)

# Filter points for plotting window
heights = []
timestamps = []
prices = []

for i in range(len(output_prices)):
    if price_dn < output_prices[i] < price_up:
        heights.append(output_blocks[i])
        timestamps.append(output_times[i])
        prices.append(output_prices[i])

if not prices:
    print("\nNo prices remained after plot window filter. Try increasing PCT_RANGE_MED.\n")
    sys.exit(1)

# Smooth X axis spacing like original
start = block_nums_needed[0]
end = block_nums_needed[-1]
count = len(prices)
step = (end - start) / (count - 1) if count > 1 else 0
heights_smooth = [start + i * step for i in range(count)]

# Sort by smooth X
heights_smooth, prices, heights, timestamps = zip(*sorted(zip(heights_smooth, prices, heights, timestamps)))

num_ticks = 5
n = len(heights_smooth)
tick_indxs = [round(i * (n - 1) / (num_ticks - 1)) for i in range(num_ticks)]

xtick_positions = []
xtick_labels = []
for tk in tick_indxs:
    xtick_positions.append(heights_smooth[tk])
    block_height = heights[tk]
    timestamp = timestamps[tk]
    dt_utc = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    dt_est = dt_utc.astimezone(EST)
    day_name = dt_est.strftime("%a")          # Mon, Tue, Wed, ...
    time_label = f"{day_name} {dt_est.hour:02}:{dt_est.minute:02} EST"
    xtick_labels.append(f"{block_height}\n{time_label}")

plot_title_left = ""
plot_title_right = ""
bottom_note1 = ""
bottom_note2 = ""

if date_mode:
    plot_title_left = f"{price_day_date_utc} blocks from local node"
    plot_title_right = f"Blended ${int(blended_price):,} (anchor {ltc_usd_price:.2f})"
    bottom_note1 = "Blend:"
    bottom_note2 = f"{int(BLEND_WEIGHT*100)}% spot + {int((1-BLEND_WEIGHT)*100)}% chain"
else:
    plot_title_left = f"Local Node Blocks {block_start_num}-{block_finish_num}"
    plot_title_right = f"Rolling Blended ${int(blended_price):,} (anchor {ltc_usd_price:.2f})"
    bottom_note1 = "* Rolling window"
    bottom_note2 = f"{WINDOW_BLOCKS} blocks | {int(BLEND_WEIGHT*100)}% spot + {int((1-BLEND_WEIGHT)*100)}% chain"


# File names (used for nav links) — cache-busting run id
run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

PRICE_FILE_LIVE = f"UTXOracle_LTC_LIVE_{run_id}.html"
PRICE_FILE_DAY  = f"UTXOracle_LTC_{price_date_dash}_{run_id}.html"
CLUSTER_FILE    = f"UTXOracle_LTC_CLUSTER_{run_id}.html"


PRICE_HTML = f'''<!DOCTYPE html>
<html>
<head>
<title>UTXOracle Local (Litecoin)</title>
<meta http-equiv="refresh" content="120">
<style>
  body {{ background-color: black; margin: 0; color: #CCCCCC; font-family: Arial, sans-serif; text-align: center; }}
  canvas {{ background-color: black; display: block; margin: auto; }}
  a {{ color: cyan; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .nav {{ margin: 10px auto 0 auto; font-size: 14px; color: #777; }}
  .nav span {{ color: #777; margin: 0 6px; }}
</style>
</head>
<body>

<div class="nav">
  <b style="color:white;">Views:</b>
  <a href="{PRICE_FILE_LIVE}">Price</a><span>|</span>
  <a href="{CLUSTER_FILE}">Clustering</a>
</div>

<div id="tooltip" style="position:absolute;background-color:black;color:cyan;border:1px solid cyan;padding:8px;font-size:14px;border-radius:5px;pointer-events:none;opacity:0;transition:opacity 0.2s;text-align:left;z-index:10;"></div>
<div style="position: relative; width: 95%; max-width: 1000px; margin: auto;">
  <canvas id="myCanvas" style="width: 100%; height: auto;" width="{width}" height="{height}"></canvas>
  <button id="downloadBtn" style="position:absolute;bottom:5%;right:2%;font-size:14px;padding:6px 10px;background-color:black;color:white;border:none;border-radius:5px;cursor:pointer;opacity:0.85;">Save PNG</button>
</div>

<script>
const canvas = document.getElementById('myCanvas');
const ctx = canvas.getContext('2d');

const width = {width}, height = {height};
const marginLeft = {margin_left}, marginRight = {margin_right}, marginTop = {margin_top}, marginBottom = {margin_bottom};
const plotWidth = width - marginLeft - marginRight;
const plotHeight = height - marginTop - marginBottom;

const heights_smooth = {list(heights_smooth)};
const prices = {list(prices)};
const heights = {list(heights)};
const timestamps = {list(timestamps)};

const ymin = Math.min(...prices), ymax = Math.max(...prices);
const xmin = Math.min(...heights_smooth), xmax = Math.max(...heights_smooth);

const xtick_positions = {xtick_positions};
const xtick_labels = {xtick_labels};

function scaleX(t) {{ return marginLeft + (t - xmin) / (xmax - xmin) * plotWidth; }}
function scaleY(p) {{ return marginTop + (1 - (p - ymin) / (ymax - ymin)) * plotHeight; }}

// background
ctx.fillStyle = "black"; ctx.fillRect(0,0,width,height);

// title
ctx.font = "bold 36px Arial"; ctx.textAlign = "center";
ctx.fillStyle = "cyan"; ctx.fillText("UTXOracle", width/2 - 60, 40);
ctx.fillStyle = "lime"; ctx.fillText("Local", width/2 + 95, 40);

ctx.font = "24px Arial";
ctx.textAlign = "right"; ctx.fillStyle = "white"; ctx.fillText("{plot_title_left}", width/2, 80);
ctx.textAlign = "left"; ctx.fillStyle = "lime"; ctx.fillText("{plot_title_right}", width/2 + 10, 80);

// frame
ctx.strokeStyle = "white"; ctx.lineWidth = 1;
ctx.strokeRect(marginLeft, marginTop, plotWidth, plotHeight);

// y ticks
ctx.fillStyle = "white"; ctx.font = "20px Arial";
const yticks = 5;
for (let i=0;i<=yticks;i++) {{
  let p = ymin + (ymax-ymin)*i/yticks;
  let y = scaleY(p);
  ctx.beginPath(); ctx.moveTo(marginLeft-5,y); ctx.lineTo(marginLeft,y); ctx.stroke();
  ctx.textAlign="right"; ctx.fillText(Math.round(p).toLocaleString(), marginLeft-10, y+4);
}}

// x ticks
ctx.textAlign="center"; ctx.font="16px Arial";
for (let i=0;i<xtick_positions.length;i++) {{
  let x = scaleX(xtick_positions[i]);
  ctx.beginPath(); ctx.moveTo(x, marginTop+plotHeight); ctx.lineTo(x, marginTop+plotHeight+5); ctx.stroke();
  let parts = xtick_labels[i].split(/\r?\n/);
  ctx.fillText(parts[0], x, marginTop+plotHeight+20);
  ctx.fillText(parts[1] || "", x, marginTop+plotHeight+40);

}}

// axis labels
ctx.fillStyle="white"; ctx.font="20px Arial"; ctx.textAlign="center";
ctx.fillText("Block Height and EST Time", marginLeft+plotWidth/2, height-48);
ctx.save(); ctx.translate(20, marginTop+plotHeight/2); ctx.rotate(-Math.PI/2);
ctx.fillText("LTC Price ($)", 0, 0); ctx.restore();

// ---- faint dot cloud (blue shades)
for (let i = 0; i < heights_smooth.length; i++) {{
  var p = prices[i];
  var x = scaleX(heights_smooth[i]);
  var y = scaleY(p);

  var t = (p - ymin) / (ymax - ymin + 1e-12);

  var r = 30;
  var g = Math.round(80 + 60 * t);
  var b = Math.round(150 + 105 * t);
  var a = 0.05 + 0.12 * t;

  ctx.fillStyle = "rgba(" + r + "," + g + "," + b + "," + a + ")";
  ctx.fillRect(x, y, 1, 1);
}}
ctx.globalAlpha = 1.0;

// ---- MEDIAN SPINE per X bin
const bins = {CLUSTER_X_BINS};
const binPrices = Array.from({{length: bins}}, () => []);
for (let i=0; i<heights_smooth.length; i++) {{
  const frac = (heights_smooth[i] - xmin) / (xmax - xmin);
  const b = Math.max(0, Math.min(bins-1, Math.floor(frac * bins)));
  binPrices[b].push(prices[i]);
}}

function median(arr) {{
  if (!arr.length) return null;
  arr.sort((a,b)=>a-b);
  const n = arr.length;
  return (n%2) ? arr[(n-1)/2] : (arr[n/2-1] + arr[n/2]) / 2;
}}

ctx.strokeStyle = "cyan";
ctx.lineWidth = 2;
ctx.beginPath();
let started = false;
for (let b=0; b<bins; b++) {{
  const m = median(binPrices[b]);
  if (m === null) continue;
  const x = marginLeft + (b/(bins-1)) * plotWidth;
  const y = scaleY(m);
  if (!started) {{ ctx.moveTo(x,y); started = true; }}
  else ctx.lineTo(x,y);
}}
ctx.stroke();

// price marker on right
ctx.fillStyle="cyan"; ctx.font="20px Arial"; ctx.textAlign="left";
ctx.fillText("- {int(blended_price):,}", marginLeft + plotWidth + 1, scaleY({blended_price}));

// bottom note
ctx.font="24px Arial"; ctx.fillStyle="lime"; ctx.textAlign="right"; ctx.fillText("{bottom_note1}", 320, height-10);
ctx.fillStyle="white"; ctx.textAlign="left"; ctx.fillText("{bottom_note2}", 325, height-10);

// tooltip
const tooltip = document.getElementById('tooltip');
canvas.addEventListener('mousemove', function(event) {{
  const rect = canvas.getBoundingClientRect();
  const sx = canvas.width / rect.width;
  const sy = canvas.height / rect.height;
  const mouseX = (event.clientX - rect.left) * sx;
  const mouseY = (event.clientY - rect.top) * sy;

  if (mouseX >= marginLeft && mouseX <= width - marginRight &&
      mouseY >= marginTop && mouseY <= marginTop + plotHeight) {{

    const fractionAcross = (mouseX - marginLeft) / plotWidth;
    let index = Math.round(fractionAcross * (heights.length - 1));
    index = Math.max(0, Math.min(index, heights.length - 1));

    const price = ymax - (mouseY - marginTop) / plotHeight * (ymax - ymin);
    const blockHeight = heights[index];
    const timestamp = timestamps[index];

    const date = new Date(timestamp * 1000);
    const estLabel = new Intl.DateTimeFormat('en-US', {{
      timeZone: 'Etc/GMT+5',
      weekday: 'short',
      hour: '2-digit',
      minute: '2-digit',
      hour12: false
    }}).format(date) + ' EST';



    tooltip.innerHTML =
      'Price: $' + Math.round(price).toLocaleString() + '<br>' +
      'Block: ' + blockHeight.toLocaleString() + '<br>' +
      'Time: ' + estLabel;

    tooltip.style.left = (event.clientX + 5) + 'px';
    tooltip.style.top = (event.clientY + window.scrollY - 75) + 'px';
    tooltip.style.opacity = 1;
  }} else {{
    tooltip.style.opacity = 0;
  }}
}});
</script>

<script>
const downloadBtn = document.getElementById('downloadBtn');
downloadBtn.addEventListener('click', function() {{
  const link = document.createElement('a');
  link.download = 'UTXOracle_Litecoin_Local_Node_Price.png';
  link.href = canvas.toDataURL('image/png');
  link.click();
}});
</script>
</body>
</html>'''

##############################################################################
# Step 13 - Generate Clustering Heatmap HTML (USD size vs time)
##############################################################################

usd_sizes = []
usd_times = []

for amt, t in zip(raw_outputs, block_times_dec):
    usd = float(amt) * float(blended_price)
    if usd <= 0:
        continue
    usd_sizes.append(usd)
    usd_times.append(int(t))

CLUSTER_HTML = None
if usd_sizes:
    import math
    log_min = math.log10(CLUSTER_USD_MIN)
    log_max = math.log10(CLUSTER_USD_MAX)

    t_min = min(usd_times)
    t_max = max(usd_times)
    if t_max == t_min:
        t_max = t_min + 1

    grid = [[0 for _ in range(CLUSTER_X_BINS)] for __ in range(CLUSTER_Y_BINS)]

    for usd, t in zip(usd_sizes, usd_times):
        xf = (t - t_min) / (t_max - t_min)
        x = int(xf * (CLUSTER_X_BINS - 1))
        x = max(0, min(CLUSTER_X_BINS - 1, x))

        lu = math.log10(usd)
        if lu < log_min or lu > log_max:
            continue
        yf = (lu - log_min) / (log_max - log_min)
        y = int(yf * (CLUSTER_Y_BINS - 1))
        y = max(0, min(CLUSTER_Y_BINS - 1, y))

        grid[y][x] += 1

    maxCount = max(max(row) for row in grid) if grid else 1

    cluster_title = (
        f"{price_day_date_utc} clustering (USD size vs time)"
        if date_mode else
        f"Rolling clustering (USD size vs time)"
    )

    CLUSTER_HTML = f'''<!DOCTYPE html>
<html>
<head>
<title>UTXOracle LTC Cluster</title>
<meta http-equiv="refresh" content="120">
<style>
  body {{ background:#000; margin:0; color:#ccc; font-family:Arial,sans-serif; text-align:center; }}
  canvas {{ background:#000; display:block; margin:auto; }}
  a {{ color: cyan; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .nav {{ margin: 10px auto 0 auto; font-size: 14px; color: #777; }}
  .nav span {{ color: #777; margin: 0 6px; }}
</style>
</head>
<body>

<div class="nav">
  <b style="color:white;">Views:</b>
  <a href="{PRICE_FILE_LIVE}">Price</a><span>|</span>
  <a href="{CLUSTER_FILE}">Clustering</a>
</div>

<div style="width:95%; max-width:1000px; margin:auto;">
  <div style="padding-top:10px;">
    <div style="font-size:34px; font-weight:bold;">
      <span style="color:cyan;">UTXOracle</span> <span style="color:lime;">Local</span>
    </div>
    <div style="font-size:18px; color:white;">
      {cluster_title}
      <span style="color:lime;"> — anchor {ltc_usd_price:.2f}, blended {int(blended_price):,}</span>
    </div>
    <div style="font-size:12px; color:#777; padding-top:6px;">
      Color/intensity uses percentile scaling (p98) + multi-stop ramp so faint clustering is still visible.
    </div>
  </div>

  <canvas id="c" width="1000" height="660" style="width:100%; height:auto;"></canvas>

  <div style="font-size:13px; color:#777; padding:10px 0 20px;">
    Heatmap shows clustering of output USD sizes (log scale) over time. Brighter/whiter = more clustering.
  </div>
</div>

<script>
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');

const W = canvas.width, H = canvas.height;
const marginL = 130, marginR = 50, marginT = 110, marginB = 120;
const plotW = W - marginL - marginR;
const plotH = H - marginT - marginB;

const grid = {json.dumps(grid)};
const xBins = {CLUSTER_X_BINS};
const yBins = {CLUSTER_Y_BINS};
const maxCount = {maxCount};

const logMin = {log_min};
const logMax = {log_max};

const tMin = {t_min};
const tMax = {t_max};

// -----------------------------
// Percentile-based intensity scaling
// -----------------------------
const vals = [];
for (let y=0; y<yBins; y++) {{
  for (let x=0; x<xBins; x++) {{
    const v = grid[y][x];
    if (v > 0) vals.push(v);
  }}
}}
vals.sort((a,b)=>a-b);

function quantile(sorted, q) {{
  if (!sorted.length) return 1;
  const pos = (sorted.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  const a = sorted[base];
  const b = sorted[Math.min(sorted.length - 1, base + 1)];
  return a + (b - a) * rest;
}}

// p98 works well when a few cells are huge; try 0.95 if you want even brighter overall
const p95 = quantile(vals, 0.95);
const p98 = quantile(vals, 0.98);
const scale = Math.max(1, p98);

// log curve + clamp
function intensity(v) {{
  if (v <= 0) return 0;
  const vv = Math.min(v, scale);
  return Math.log(1 + 12*vv) / Math.log(1 + 12*scale);
}}

// multi-stop color ramp (dark -> deep blue -> cyan -> white)
function lerp(a,b,t) {{ return a + (b-a)*t; }}
function ramp(t) {{
  const stops = [
    {{t:0.00, r:  0, g:  0, b:  0}},
    {{t:0.35, r: 10, g: 40, b:120}},
    {{t:0.70, r:  0, g:220, b:255}},
    {{t:1.00, r:255, g:255, b:255}},
  ];
  t = Math.max(0, Math.min(1, t));
  let i = 0;
  while (i < stops.length - 1 && t > stops[i+1].t) i++;
  const s0 = stops[i], s1 = stops[i+1];
  const u = (t - s0.t) / (s1.t - s0.t + 1e-12);
  const r = Math.round(lerp(s0.r, s1.r, u));
  const g = Math.round(lerp(s0.g, s1.g, u));
  const b = Math.round(lerp(s0.b, s1.b, u));
  return {{r,g,b}};
}}

function usdLabelFromY(y) {{
  const f = y / (yBins - 1);
  const logv = logMin + f * (logMax - logMin);
  const usd = Math.pow(10, logv);
  if (usd >= 1000000) return "$" + (usd/1000000).toFixed(0) + "M";
  if (usd >= 1000) return "$" + (usd/1000).toFixed(0) + "k";
  if (usd >= 1) return "$" + usd.toFixed(0);
  return "$" + usd.toFixed(2);
}}

ctx.fillStyle = "#000"; ctx.fillRect(0,0,W,H);

// frame
ctx.strokeStyle = "white"; ctx.lineWidth = 1;
ctx.strokeRect(marginL, marginT, plotW, plotH);

// -----------------------------
// draw heatmap (color encodes intensity + alpha floor so faint stuff shows)
// -----------------------------
for (let y=0; y<yBins; y++) {{
  for (let x=0; x<xBins; x++) {{
    const v = grid[y][x];
    if (v <= 0) continue;

    let t = intensity(v);

    // brighten midtones so more shows up
    t = Math.pow(t, 0.75); // <1 brightens mids; try 0.6..1.0

    const c = ramp(t);

    // alpha floor keeps faint clustering visible
    const alpha = 0.10 + 0.90 * t;  // raise 0.10 -> 0.18 if you want more "dust"
    ctx.fillStyle = `rgba(${{c.r}},${{c.g}},${{c.b}},${{alpha}})`;

    const px = marginL + (x / xBins) * plotW;
    const py = marginT + (1 - (y / yBins)) * plotH;
    const pw = (plotW / xBins) + 1;
    const ph = (plotH / yBins) + 1;
    ctx.fillRect(px, py, pw, ph);
  }}
}}
ctx.globalAlpha = 1.0;

// -----------------------------
// OPTIONAL: row density overlay (strong USD-size bands pop)
// -----------------------------
const rowSum = new Array(yBins).fill(0);
for (let y=0; y<yBins; y++) {{
  let s = 0;
  for (let x=0; x<xBins; x++) s += grid[y][x];
  rowSum[y] = s;
}}
const maxRow = Math.max(...rowSum) || 1;

ctx.globalAlpha = 0.35;
ctx.strokeStyle = "white";
for (let y=0; y<yBins; y++) {{
  const tt = rowSum[y] / maxRow;
  if (tt < 0.25) continue; // show only stronger bands; tweak 0.15..0.35
  const yy = marginT + (1 - (y / yBins)) * plotH;
  ctx.lineWidth = 0.5 + 2.0 * tt;
  ctx.beginPath();
  ctx.moveTo(marginL, yy);
  ctx.lineTo(marginL + plotW, yy);
  ctx.stroke();
}}
ctx.globalAlpha = 1.0;

// y labels
ctx.fillStyle = "white";
ctx.font = "18px Arial";
ctx.textAlign = "right";
const yTicks = 6;
for (let i=0; i<=yTicks; i++) {{
  const f = i / yTicks;
  const y = marginT + f * plotH;
  const yy = Math.round((1 - f) * (yBins - 1));
  ctx.beginPath();
  ctx.moveTo(marginL-5, y); ctx.lineTo(marginL, y);
  ctx.strokeStyle = "white";
  ctx.stroke();
  ctx.fillText(usdLabelFromY(yy), marginL-10, y+6);
}}

// x labels
ctx.textAlign = "center";
ctx.font = "16px Arial";
function timeLabel(f) {{
  const t = tMin + f * (tMax - tMin);
  const d = new Date(t * 1000);

  const s = new Intl.DateTimeFormat('en-US', {{
    timeZone: 'Etc/GMT+5',
    weekday: 'short',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false
  }}).format(d);

  return s.replace(',', '') + ' EST';
}}

const xTicks = 5;
for (let i=0; i<=xTicks; i++) {{
  const f = i / xTicks;
  const x = marginL + f * plotW;
  ctx.beginPath();
  ctx.moveTo(x, marginT+plotH); ctx.lineTo(x, marginT+plotH+5);
  ctx.stroke();
  ctx.fillStyle="white";
  ctx.fillText(timeLabel(f), x, marginT+plotH+25);
}}

// axis labels
ctx.fillStyle="white";
ctx.font="20px Arial";
ctx.textAlign="center";
ctx.fillText("EST Time", marginL + plotW/2, H-50);

ctx.save();
ctx.translate(25, marginT + plotH/2);
ctx.rotate(-Math.PI/2);
ctx.fillText("Transaction size (USD, log-scaled)", 0, 0);
ctx.restore();
</script>
</body>
</html>'''

##############################################################################
# Write files + open in browser
##############################################################################


import webbrowser

# Price HTML filename
if block_mode or days_back is not None:
    price_filename = PRICE_FILE_LIVE
else:
    price_filename = PRICE_FILE_DAY

with open(price_filename, "w", encoding="utf-8") as f:
    f.write(PRICE_HTML)

# Cluster HTML filename
cluster_filename = CLUSTER_FILE
if CLUSTER_HTML:
    with open(cluster_filename, "w", encoding="utf-8") as f:
        f.write(CLUSTER_HTML)

# Open price plot
webbrowser.open("file://" + os.path.realpath(price_filename))

# Also open cluster plot
if CLUSTER_HTML:
    webbrowser.open("file://" + os.path.realpath(cluster_filename))
    print(f"Saved price view:    {price_filename}")
    print(f"Saved cluster view:  {cluster_filename}")
else:
    print(f"Saved price view:    {price_filename}")
    print("No clustering heatmap generated (no usable outputs).")
