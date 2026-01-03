#########################################################################################
#                                                                                       #
#   /$$   /$$ /$$$$$$$$ /$$   /$$  /$$$$$$                               /$$            #
#  | $$  | $$|__  $$__/| $$  / $$ /$$__  $$                             | $$            #
#  | $$  | $$   | $$   |  $$/ $$/| $$  \ $$  /$$$$$$  /$$$$$$   /$$$$$$$| $$  /$$$$$$   #
#  | $$  | $$   | $$    \  $$$$/ | $$  | $$ /$$__  $$|____  $$ /$$_____/| $$ /$$__  $$  #
#  | $$  | $$   | $$     >$$  $$ | $$  | $$| $$  \__/ /$$$$$$$| $$      | $$| $$$$$$$$  #
#  | $$  | $$   | $$    /$$/\  $$| $$  | $$| $$      /$$__  $$| $$      | $$| $$_____/  #
#  |  $$$$$$/   | $$   | $$  \ $$|  $$$$$$/| $$     |  $$$$$$$|  $$$$$$$| $$|  $$$$$$$  #
#   \______/    |__/   |__/  |__/ \______/ |__/      \_______/ \_______/|__/ \_______/  #
#                                                                                       #
#########################################################################################
#                     UTXOracle - Litecoin Edition (RPC Only)
#
# This is a Litecoin Core (litecoind) adaptation of the UTXOracle-style algorithm.
# - Connects to your OWN Litecoin node via JSON-RPC
# - Pulls raw blocks with getblock <hash> 0 (hex) and parses tx outputs locally
# - Uses a fixed USD/LTC anchor you provided: LTC = $82.39
#
# Save as: UTXOracle_Litecoin.py
# Run:     python3 UTXOracle_Litecoin.py
# Options: python3 UTXOracle_Litecoin.py -h
#
# NOTE: Litecoin uses 1e8 base units (like BTC), so amount parsing is unchanged.
# NOTE: If you are using a non-standard fork / exotic serialization, prefer switching
#       Step 6 to use getblock verbosity=2 and read vout values from JSON.

###############################################################################
# Step 1 - Configuration Options
###############################################################################

import platform
import os
import sys

data_dir = []
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

import urllib.request
import urllib.parse

def fetch_coinbase_spot_ltc_usd(date_yyyy_mm_dd: str | None = None) -> float:
    """
    Returns LTC-USD spot price as float using Coinbase public API.
    If date_yyyy_mm_dd is provided (YYYY-MM-DD, UTC), returns historic spot for that date.
    """
    base_url = "https://api.coinbase.com/v2/prices/LTC-USD/spot"
    if date_yyyy_mm_dd:
        qs = urllib.parse.urlencode({"date": date_yyyy_mm_dd})
        url = f"{base_url}?{qs}"
    else:
        url = base_url

    req = urllib.request.Request(
        url,
        headers={
            # Coinbase examples often include CB-VERSION; generally not required,
            # but harmless and can reduce surprises.
            "CB-VERSION": "2015-04-08",
            "User-Agent": "UTXOracle-Litecoin/1.0"
        }
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        raw = resp.read().decode("utf-8")
    j = json.loads(raw)
    return float(j["data"]["amount"])







def print_help():
    help_text = """
Usage: python3 UTXOracle_Litecoin.py [options]

Options:
  -h               Show this help message
  -d YYYY/MM/DD    Specify a UTC date to evaluate
  -p /path/to/dir  Specify the Litecoin data directory (where litecoin.conf lives)
  -rb              Use last 144 recent blocks instead of date mode
"""
    print(help_text)
    sys.exit(0)

if "-h" in sys.argv:
    print_help()

if "-d" in sys.argv:
    h_index = sys.argv.index("-d")
    if h_index + 1 < len(sys.argv):
        date_entered = sys.argv[h_index + 1]

if "-p" in sys.argv:
    d_index = sys.argv.index("-p")
    if d_index + 1 < len(sys.argv):
        data_dir = sys.argv[d_index + 1]

if "-rb" in sys.argv:
    date_mode = False
    block_mode = True

conf_path = None
conf_candidates = ["litecoin.conf", "litecoin_rw.conf", "litecoin-rw.conf"]
for fname in conf_candidates:
    path = os.path.join(data_dir, fname)
    if os.path.exists(path):
        conf_path = path
        break
if not conf_path:
    print(f"Invalid Litecoin data directory: {data_dir}")
    print("Expected to find 'litecoin.conf' (or litecoin_rw.conf) in this directory.")
    sys.exit(1)

conf_settings = {}
with open(conf_path, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            conf_settings[key.strip()] = value.strip().strip('"')

blocks_dir = os.path.expanduser(conf_settings.get("blocksdir", os.path.join(data_dir, "blocks")))

rpc_user = conf_settings.get("rpcuser")
rpc_password = conf_settings.get("rpcpassword")
cookie_path = conf_settings.get("rpccookiefile", os.path.join(data_dir, ".cookie"))
rpc_host = conf_settings.get("rpcconnect", "127.0.0.1")
rpc_port = int(conf_settings.get("rpcport", "9332"))

###############################################################################
# Step 2 - Establish RPC Connection
###############################################################################

print("\nCurrent operation  \t\t\t\tTotal Completion", flush=True)
print("\nConnecting to node...", flush=True)
print("0%..", end="", flush=True)

import http.client
import json
import base64

def Ask_Node(command, cred_creation):
    method = command[0]
    params = command[1:]

    rpc_u = rpc_user
    rpc_p = rpc_password

    if not rpc_u or not rpc_p:
        try:
            with open(cookie_path, "r") as f:
                cookie = f.read().strip()
                rpc_u, rpc_p = cookie.split(":", 1)
        except Exception as e:
            print("Error reading .cookie file for RPC authentication.")
            print("Details:", e)
            sys.exit(1)

    payload = json.dumps({"jsonrpc":"1.0","id":"utxoracle","method":method,"params":params})
    auth_header = base64.b64encode(f"{rpc_u}:{rpc_p}".encode()).decode()
    headers = {"Content-Type":"application/json","Authorization":f"Basic {auth_header}"}

    try:
        conn = http.client.HTTPConnection(rpc_host, rpc_port)
        conn.request("POST", "/", payload, headers)
        response = conn.getresponse()
        if response.status != 200:
            raise Exception(f"HTTP error {response.status} {response.reason}")
        raw_data = response.read()
        conn.close()

        parsed = json.loads(raw_data)
        if parsed.get("error"):
            raise Exception(parsed["error"])

        result = parsed["result"]
        if isinstance(result, (dict, list)):
            return json.dumps(result, indent=2).encode()
        return str(result).encode()

    except Exception as e:
        if not cred_creation:
            print("Error connecting to your node via RPC. Troubleshooting steps:\n")
            print("\t1) Ensure litecoind or litecoin-qt is running with server=1.")
            print("\t2) Check rpcuser/rpcpassword or .cookie.")
            print("\t3) Verify RPC port/host settings.")
            print("\nThe attempted RPC method was:", method)
            print("Parameters:", params)
            print("\nThe error was:\n", e)
            sys.exit(1)

print("20%..", end="", flush=True)
Ask_Node(["getblockcount"], True)
block_count_b = Ask_Node(["getblockcount"], False)
print("40%..", end="", flush=True)
block_count = int(block_count_b)
block_count_consensus = block_count - 6

block_hash = Ask_Node(["getblockhash", block_count_consensus], False).decode().strip()
print("60%..", end="", flush=True)
block_header_b = Ask_Node(["getblockheader", block_hash, True], False)
block_header = json.loads(block_header_b)
print("80%..", end="", flush=True)

###############################################################################
# Step 3 - Check Dates
###############################################################################

from datetime import datetime, timezone, timedelta

latest_time_in_seconds = block_header["time"]
time_datetime = datetime.fromtimestamp(latest_time_in_seconds, tz=timezone.utc)

latest_year  = int(time_datetime.strftime("%Y"))
latest_month = int(time_datetime.strftime("%m"))
latest_day   = int(time_datetime.strftime("%d"))
latest_utc_midnight = datetime(latest_year, latest_month, latest_day, 0, 0, 0, tzinfo=timezone.utc)

seconds_in_a_day = 86400
yesterday_seconds = latest_time_in_seconds - seconds_in_a_day
latest_price_day = datetime.fromtimestamp(yesterday_seconds, tz=timezone.utc)
latest_price_date = latest_price_day.strftime("%Y-%m-%d")
print("100%", end="", flush=True)
print("\t\t\t5% done", flush=True)

if date_mode:
    if date_entered == "":
        datetime_entered = latest_utc_midnight + timedelta(days=-1)
    else:
        try:
            year = int(date_entered.split("/")[0])
            month = int(date_entered.split("/")[1])
            day = int(date_entered.split("/")[2])

            datetime_entered = datetime(year, month, day, 0, 0, 0, tzinfo=timezone.utc)
            if datetime_entered.timestamp() >= latest_utc_midnight.timestamp():
                print("\nDate is after the latest available. We need 6 blocks after UTC midnight.")
                print("Run UTXOracle_Litecoin.py -rb for the most recent blocks")
                sys.exit()

            dec_15_2023 = datetime(2023, 12, 15, 0, 0, 0, tzinfo=timezone.utc)
            if datetime_entered.timestamp() < dec_15_2023.timestamp():
                print("\nThe date entered is before 2023-12-15, please try again")
                sys.exit()

        except:
            print("\nError interpreting date. Please try again. Make sure format is YYYY/MM/DD")
            sys.exit()

    price_day_seconds = int(datetime_entered.timestamp())
    price_day_date_utc = datetime_entered.strftime("%b %d, %Y")
    price_date_dash = datetime_entered.strftime("%Y-%m-%d")

##############################################################################
# Step 4 - Find Block Hashes
##############################################################################

def get_block_time(height):
    block_hash_b = Ask_Node(["getblockhash", height], False)
    block_header_b = Ask_Node(["getblockheader", block_hash_b.decode().strip(), True], False)
    h = json.loads(block_header_b)
    return (h["time"], block_hash_b[:64].decode())

def get_day_of_month(time_in_seconds):
    return int(datetime.fromtimestamp(time_in_seconds, tz=timezone.utc).strftime("%d"))

if block_mode:
    print("\nFinding the last 144 blocks", flush=True)

    block_finish_num = block_count
    block_start_num = block_finish_num - 144

    block_num = block_start_num
    time_in_seconds, hash_end = get_block_time(block_start_num)
    print_every = 0
    while block_num < block_finish_num:
        if (block_num - block_start_num) / 144 * 100 > print_every and print_every < 100:
            print(str(print_every) + "%..", end="", flush=True)
            print_every += 20
        block_nums_needed.append(block_num)
        block_hashes_needed.append(hash_end)
        block_times_needed.append(time_in_seconds)
        block_num += 1
        time_in_seconds, hash_end = get_block_time(block_num)

    print("100%\t\t\t25% done", flush=True)

elif date_mode:
    print("\nFinding first blocks on " + datetime_entered.strftime("%b %d, %Y"), flush=True)
    print("0%..", end="", flush=True)

    seconds_since_price_day = latest_time_in_seconds - price_day_seconds
    blocks_ago_estimate = round(144 * float(seconds_since_price_day) / float(seconds_in_a_day))
    price_day_block_estimate = block_count_consensus - blocks_ago_estimate

    time_in_seconds, hash_end = get_block_time(price_day_block_estimate)

    print("20%..", end="", flush=True)
    seconds_difference = time_in_seconds - price_day_seconds
    block_jump_estimate = round(144 * float(seconds_difference) / float(seconds_in_a_day))

    last_estimate = 0
    last_last_estimate = 0

    print("40%..", end="", flush=True)
    while block_jump_estimate > 6 and block_jump_estimate != last_last_estimate:
        last_last_estimate = last_estimate
        last_estimate = block_jump_estimate

        price_day_block_estimate = price_day_block_estimate - block_jump_estimate
        time_in_seconds, hash_end = get_block_time(price_day_block_estimate)
        seconds_difference = time_in_seconds - price_day_seconds
        block_jump_estimate = round(144 * float(seconds_difference) / float(seconds_in_a_day))

    print("60%..", end="", flush=True)
    if time_in_seconds > price_day_seconds:
        while time_in_seconds > price_day_seconds:
            price_day_block_estimate -= 1
            time_in_seconds, hash_end = get_block_time(price_day_block_estimate)
        price_day_block_estimate += 1
    elif time_in_seconds < price_day_seconds:
        while time_in_seconds < price_day_seconds:
            price_day_block_estimate += 1
            time_in_seconds, hash_end = get_block_time(price_day_block_estimate)

    print("80%..", end="", flush=True)
    price_day_block = price_day_block_estimate

    time_in_seconds, _ = get_block_time(price_day_block)
    day1 = get_day_of_month(time_in_seconds)

    price_day_block_end = price_day_block
    time_in_seconds, hash_end = get_block_time(price_day_block_end)
    day2 = get_day_of_month(time_in_seconds)

    print("100%\t\t\t25% done", flush=True)
    print("\nFinding last blocks on " + datetime_entered.strftime("%b %d, %Y"), flush=True)

    block_num = 0
    print_next = 0
    while day1 == day2:
        block_num += 1
        if block_num / 144 * 100 > print_next and print_next < 100:
            print(str(print_next) + "%..", end="", flush=True)
            print_next += 20

        block_nums_needed.append(price_day_block_end)
        block_hashes_needed.append(hash_end)
        block_times_needed.append(time_in_seconds)
        price_day_block_end += 1
        time_in_seconds, hash_end = get_block_time(price_day_block_end)
        day2 = get_day_of_month(time_in_seconds)

    while print_next < 100:
        print(str(print_next) + "%..", end="", flush=True)
        print_next += 20

    block_start_num = price_day_block
    block_finish_num = price_day_block_end
    print("100%\t\t\t50% done", flush=True)

##############################################################################
# Step 5 - Initial histogram
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
# Step 6 - Load Transaction Data
##############################################################################

print("\nLoading every transaction from every block", flush=True)

from struct import unpack
import binascii
import hashlib
from io import BytesIO
import struct

todays_txids = set()
raw_outputs = []
block_heights_dec = []
block_times_dec = []
print_next = 0
block_num = 0

def read_varint(f):
    i = f.read(1)
    if not i:
        return 0
    i = i[0]
    if i < 0xfd:
        return i
    elif i == 0xfd:
        return struct.unpack("<H", f.read(2))[0]
    elif i == 0xfe:
        return struct.unpack("<I", f.read(4))[0]
    else:
        return struct.unpack("<Q", f.read(8))[0]

def encode_varint(i: int) -> bytes:
    assert i >= 0
    if i < 0xfd:
        return i.to_bytes(1, "little")
    elif i <= 0xffff:
        return b"\xfd" + i.to_bytes(2, "little")
    elif i <= 0xffffffff:
        return b"\xfe" + i.to_bytes(4, "little")
    else:
        return b"\xff" + i.to_bytes(8, "little")

def compute_txid(raw_tx_bytes: bytes) -> bytes:
    stream = BytesIO(raw_tx_bytes)
    version = stream.read(4)

    marker = stream.read(1)
    flag = stream.read(1)
    is_segwit = (marker == b"\x00" and flag == b"\x01")

    if not is_segwit:
        stream.seek(0)
        stripped_tx = stream.read()
    else:
        stripped_tx = bytearray()
        stripped_tx += version

        input_count = read_varint(stream)
        stripped_tx += encode_varint(input_count)
        for _ in range(input_count):
            stripped_tx += stream.read(32)
            stripped_tx += stream.read(4)
            script_len = read_varint(stream)
            stripped_tx += encode_varint(script_len)
            stripped_tx += stream.read(script_len)
            stripped_tx += stream.read(4)

        output_count = read_varint(stream)
        stripped_tx += encode_varint(output_count)
        for _ in range(output_count):
            stripped_tx += stream.read(8)
            script_len = read_varint(stream)
            stripped_tx += encode_varint(script_len)
            stripped_tx += stream.read(script_len)

        for _ in range(input_count):
            stack_count = read_varint(stream)
            for _ in range(stack_count):
                item_len = read_varint(stream)
                stream.read(item_len)

        stripped_tx += stream.read(4)

    return hashlib.sha256(hashlib.sha256(stripped_tx).digest()).digest()[::-1]

for bh in block_hashes_needed:
    block_num += 1
    print_progress = block_num / len(block_hashes_needed) * 100
    if print_progress > print_next and print_next < 100:
        print(f"{int(print_next)}%..", end="", flush=True)
        print_next += 1
        if print_next % 7 == 0:
            print("\n", end="")

    raw_block_hex = Ask_Node(["getblock", bh, 0], False).decode().strip()
    raw_block_bytes = binascii.unhexlify(raw_block_hex)
    stream = BytesIO(raw_block_bytes)

    stream.read(80)
    tx_count = read_varint(stream)

    txs_to_add = []
    for _ in range(tx_count):
        start_tx = stream.tell()
        stream.read(4)

        marker_flag = stream.read(2)
        is_segwit = marker_flag == b"\x00\x01"
        if not is_segwit:
            stream.seek(start_tx + 4)

        input_count = read_varint(stream)
        has_op_return = False
        witness_exceeds = False
        is_coinbase = False
        input_txids = []

        for _ in range(input_count):
            prev_txid = stream.read(32)
            prev_index = stream.read(4)
            script_len = read_varint(stream)
            stream.read(script_len)
            stream.read(4)
            input_txids.append(prev_txid[::-1].hex())
            if prev_txid == b"\x00" * 32 and prev_index == b"\xff\xff\xff\xff":
                is_coinbase = True

        output_count = read_varint(stream)
        output_values = []
        for _ in range(output_count):
            value_sats = unpack("<Q", stream.read(8))[0]
            script_len = read_varint(stream)
            script = stream.read(script_len)
            if script and script[0] == 0x6a:
                has_op_return = True
            value_coin = value_sats / 1e8
            if 1e-5 < value_coin < 1e5:
                output_values.append(value_coin)

        if is_segwit:
            for _ in range(input_count):
                stack_count = read_varint(stream)
                total_witness_len = 0
                for _ in range(stack_count):
                    item_len = read_varint(stream)
                    total_witness_len += item_len
                    stream.read(item_len)
                    if item_len > 500 or total_witness_len > 500:
                        witness_exceeds = True

        stream.read(4)
        end_tx = stream.tell()
        stream.seek(start_tx)
        raw_tx = stream.read(end_tx - start_tx)
        txid = compute_txid(raw_tx)
        todays_txids.add(txid.hex())

        is_same_day_tx = any(itxid in todays_txids for itxid in input_txids)

        if (input_count <= 5 and output_count == 2 and not is_coinbase and
            not has_op_return and not witness_exceeds and not is_same_day_tx):
            for amount in output_values:
                amount_log = log10(amount)
                percent_in_range = (amount_log - first_bin_value) / range_bin_values
                bin_number_est = int(percent_in_range * number_of_bins)
                if bin_number_est < 0 or bin_number_est >= number_of_bins:
                    continue
                while bin_number_est < number_of_bins - 1 and output_histogram_bins[bin_number_est] <= amount:
                    bin_number_est += 1
                bin_number = max(0, bin_number_est - 1)
                output_histogram_bin_counts[bin_number] += 1.0
                txs_to_add.append(amount)

    if txs_to_add:
        bkh = block_nums_needed[block_num - 1]
        tm = block_times_needed[block_num - 1]
        for amt in txs_to_add:
            raw_outputs.append(amt)
            block_heights_dec.append(bkh)
            block_times_dec.append(tm)

print("100%", flush=True)
print("\t\t\t\t\t\t95% done", flush=True)

##############################################################################
# Step 7 - Remove Round Coin Amounts
##############################################################################

print("\nFinding prices and rendering plot", flush=True)
print("0%..", end="", flush=True)

for n in range(0, 201):
    output_histogram_bin_counts[n] = 0.0
for n in range(1601, len(output_histogram_bin_counts)):
    output_histogram_bin_counts[n] = 0.0

round_btc_bins = [201,401,461,496,540,601,661,696,740,801,861,896,940,1001,1061,1096,1140,1201]
for r in round_btc_bins:
    output_histogram_bin_counts[r] = 0.5 * (output_histogram_bin_counts[r+1] + output_histogram_bin_counts[r-1])

curve_sum = sum(output_histogram_bin_counts[201:1601])
if curve_sum == 0:
    print("\nNo usable outputs found. Try -rb or a different day.")
    sys.exit(1)

for n in range(201, 1601):
    output_histogram_bin_counts[n] /= curve_sum
    if output_histogram_bin_counts[n] > 0.008:
        output_histogram_bin_counts[n] = 0.008

print("20%..", end="", flush=True)

##############################################################################
# Step 8 - Construct the Price Finding Stencil
##############################################################################

num_elements = 803
mean = 411
std_dev = 201

smooth_stencil = []
for x in range(num_elements):
    exp_part = -((x - mean) ** 2) / (2 * (std_dev ** 2))
    smooth_stencil.append((.00150 * 2.718281828459045 ** exp_part) + (.0000005 * x))

spike_stencil = [0.0 for _ in range(803)]
spike_stencil[40]  = 0.001300198324984352
spike_stencil[141] = 0.001676746949820743
spike_stencil[201] = 0.003468805546942046
spike_stencil[202] = 0.001991977522512513
spike_stencil[236] = 0.001905066647961839
spike_stencil[261] = 0.003341772718156079
spike_stencil[262] = 0.002588902624584287
spike_stencil[296] = 0.002577893841190244
spike_stencil[297] = 0.002733728814200412
spike_stencil[340] = 0.003076117748975647
spike_stencil[341] = 0.005613067550103145
spike_stencil[342] = 0.003088253178535568
spike_stencil[400] = 0.002918457489366139
spike_stencil[401] = 0.006174500465286022
spike_stencil[402] = 0.004417068070043504
spike_stencil[403] = 0.002628663628020371
spike_stencil[436] = 0.002858828161543839
spike_stencil[461] = 0.004097463611984264
spike_stencil[462] = 0.003345917406120509
spike_stencil[496] = 0.002521467726855856
spike_stencil[497] = 0.002784125730361008
spike_stencil[541] = 0.003792850444811335
spike_stencil[601] = 0.003688240815848247
spike_stencil[602] = 0.002392400117402263
spike_stencil[636] = 0.001280993059008106
spike_stencil[661] = 0.001654665137536031
spike_stencil[662] = 0.001395501347054946
spike_stencil[741] = 0.001154279140906312
spike_stencil[801] = 0.000832244504868709

##############################################################################
# Step 9 - Estimate a Rough Price (Litecoin)
##############################################################################

best_slide = 0
best_slide_score = 0.0
total_score = 0.0

smooth_weight = 0.65

# Coinbase anchor:
# - date_mode: use historic spot for the UTC day being analyzed
# - block_mode: use current spot
anchor_date = price_date_dash if date_mode else None  # price_date_dash is YYYY-MM-DD
try:
    ltc_usd_price = fetch_coinbase_spot_ltc_usd(anchor_date)
    print(f"\nCoinbase LTC-USD spot used for anchor: ${ltc_usd_price:.2f}", flush=True)
except Exception as e:
    # Fallback so the oracle can still run if Coinbase is temporarily unreachable
    ltc_usd_price = 82.39
    print(f"\nWARNING: Coinbase anchor failed ({e}). Falling back to ${ltc_usd_price:.2f}", flush=True)

usd100_in_ltc_anchor = 100.0 / ltc_usd_price

center_p001 = min(range(len(output_histogram_bins)), key=lambda i: abs(output_histogram_bins[i] - usd100_in_ltc_anchor))

left_p001  = center_p001 - int((len(spike_stencil) + 1)/2)
right_p001 = center_p001 + int((len(spike_stencil) + 1)/2)

min_slide = -141
max_slide = 201

for slide in range(min_slide, max_slide):
    l = left_p001 + slide
    r = right_p001 + slide
    if l < 0 or r > len(output_histogram_bin_counts):
        continue

    shifted_curve = output_histogram_bin_counts[l:r]

    slide_score_smooth = 0.0
    for n in range(0, len(smooth_stencil)):
        slide_score_smooth += shifted_curve[n] * smooth_stencil[n]

    slide_score = 0.0
    for n in range(0, len(spike_stencil)):
        slide_score += shifted_curve[n] * spike_stencil[n]

    if slide < 150:
        slide_score += slide_score_smooth * smooth_weight

    if slide_score > best_slide_score:
        best_slide_score = slide_score
        best_slide = slide

    total_score += slide_score

usd100_in_ltc_best = output_histogram_bins[center_p001 + best_slide]
ltc_in_usd_best = 100 / usd100_in_ltc_best

neighbor_up = output_histogram_bin_counts[left_p001+best_slide+1:right_p001+best_slide+1]
neighbor_up_score = 0.0
for n in range(0, len(spike_stencil)):
    neighbor_up_score += neighbor_up[n]*spike_stencil[n]

neighbor_down = output_histogram_bin_counts[left_p001+best_slide-1:right_p001+best_slide-1]
neighbor_down_score = 0.0
for n in range(0, len(spike_stencil)):
    neighbor_down_score += neighbor_down[n]*spike_stencil[n]

best_neighbor = +1
neighbor_score = neighbor_up_score
if neighbor_down_score > neighbor_up_score:
    best_neighbor = -1
    neighbor_score = neighbor_down_score

usd100_in_ltc_2nd = output_histogram_bins[center_p001+best_slide+best_neighbor]
ltc_in_usd_2nd = 100 / usd100_in_ltc_2nd

avg_score = total_score / len(range(min_slide, max_slide))
a1 = best_slide_score - avg_score
a2 = abs(neighbor_score - avg_score)

if (a1 + a2) == 0:
    rough_price_estimate = int(ltc_in_usd_best)
else:
    w1 = a1/(a1+a2)
    w2 = a2/(a1+a2)
    rough_price_estimate = int(w1*ltc_in_usd_best + w2*ltc_in_usd_2nd)

print("40%..", end="", flush=True)

##############################################################################
# Step 10 - Create Intraday Price Points
##############################################################################

usds = [5,10,15,20,25,30,40,50,100,150,200,300,500,1000]
pct_range_wide = .25

micro_remove_list = []
i = .00005000
while i < .0001:
    micro_remove_list.append(i)
    i += .00001
i = .0001
while i < .001:
    micro_remove_list.append(i)
    i += .00001
i = .001
while i < .01:
    micro_remove_list.append(i)
    i += .0001
i = .01
while i < .1:
    micro_remove_list.append(i)
    i += .001
i = .1
while i < 1:
    micro_remove_list.append(i)
    i += .01
pct_micro_remove = .0001

output_prices = []
output_blocks = []
output_times = []

for i in range(0, len(raw_outputs)):
    n = raw_outputs[i]
    b = block_heights_dec[i]
    t = block_times_dec[i]

    for usd in usds:
        avcoin = usd/rough_price_estimate
        coin_up = avcoin + pct_range_wide * avcoin
        coin_dn = avcoin - pct_range_wide * avcoin

        if coin_dn < n < coin_up:
            append = True
            for r in micro_remove_list:
                rm_dn = r - pct_micro_remove * r
                rm_up = r + pct_micro_remove * r
                if rm_dn < n < rm_up:
                    append = False
            if append:
                output_prices.append(usd/n)
                output_blocks.append(b)
                output_times.append(t)

print("60%..", end="", flush=True)

##############################################################################
# Step 11 - Find the Exact Average Price
##############################################################################

def find_central_output(r2, price_min, price_max):
    r6 = [r for r in r2 if price_min < r < price_max]
    outputs = sorted(r6)
    n = len(outputs)
    if n == 0:
        return 0.0, 0.0

    prefix_sum = []
    total = 0.0
    for x in outputs:
        total += x
        prefix_sum.append(total)

    left_counts = list(range(n))
    right_counts = [n - i - 1 for i in left_counts]
    left_sums = [0.0] + prefix_sum[:-1]
    right_sums = [total - x for x in prefix_sum]

    total_dists = []
    for i in range(n):
        dist = (outputs[i] * left_counts[i] - left_sums[i]) + (right_sums[i] - outputs[i] * right_counts[i])
        total_dists.append(dist)

    min_index, _ = min(enumerate(total_dists), key=lambda x: x[1])
    best_output = outputs[min_index]

    deviations = [abs(x - best_output) for x in outputs]
    deviations.sort()
    m = len(deviations)
    if m % 2 == 0:
        mad = (deviations[m//2 - 1] + deviations[m//2]) / 2
    else:
        mad = deviations[m//2]

    return best_output, mad

pct_range_tight = .05
price_up = rough_price_estimate + pct_range_tight * rough_price_estimate
price_dn = rough_price_estimate - pct_range_tight * rough_price_estimate
central_price, av_dev = find_central_output(output_prices, price_dn, price_up)

if central_price == 0.0:
    print("\n\nNo price points found after filtering. Try -rb or loosen Step 6 filters.")
    sys.exit(1)

avs = set()
avs.add(central_price)
while central_price not in avs:
    avs.add(central_price)
    price_up = central_price + pct_range_tight * central_price
    price_dn = central_price - pct_range_tight * central_price
    central_price, av_dev = find_central_output(output_prices, price_dn, price_up)

print("80%..", end="", flush=True)

pct_range_med = .1
price_up = central_price + pct_range_med * central_price
price_dn = central_price - pct_range_med * central_price
price_range = price_up - price_dn
unused_price, av_dev = find_central_output(output_prices, price_dn, price_up)
dev_pct = av_dev/price_range if price_range else 0.0

map_dev_axr = (.15-.05)/(.20-.17)
ax_range = .05 + (dev_pct-.17)*map_dev_axr
if ax_range < .05:
    ax_range = .05
if ax_range > .2:
    ax_range = .2

price_up = central_price + ax_range * central_price
price_dn = central_price - ax_range * central_price

print("100%\t\t\tdone", flush=True)
if date_mode:
    print("\n\n\t\t"+price_day_date_utc+" price: $"+f"{int(central_price):,}\n\n", flush=True)

##############################################################################
# Step 12 - Generate a Price Plot HTML Page
##############################################################################

width = 1000
height = 660
margin_left = 120
margin_right = 90
margin_top = 100
margin_bottom = 120

start = block_nums_needed[0]
end = block_nums_needed[-1]
count = len(output_prices)
step = (end - start) / (count - 1) if count > 1 else 0
b3 = [start + i * step for i in range(count)]

heights = []
heights_smooth = []
timestamps = []
prices = []
for i in range(len(output_prices)):
    if price_dn < output_prices[i] < price_up:
        heights.append(output_blocks[i])
        heights_smooth.append(b3[i])
        timestamps.append(output_times[i])
        prices.append(output_prices[i])

heights_smooth, prices = zip(*sorted(zip(heights_smooth, prices)))

num_ticks = 5
n = len(heights_smooth)
tick_indxs = [round(i * (n - 1) / (num_ticks - 1)) for i in range(num_ticks)]

xtick_positions = []
xtick_labels = []
for tk in tick_indxs:
    xtick_positions.append(heights_smooth[tk])
    block_height = heights[tk]
    timestamp = timestamps[tk]
    dt = datetime.utcfromtimestamp(timestamp)
    time_label = f"{dt.hour:02}:{dt.minute:02} UTC"
    xtick_labels.append(f"{block_height}\n{time_label}")

avg_price = central_price

plot_title_left = ""
plot_title_right = ""
bottom_note1 = ""
bottom_note2 = ""
if date_mode:
    plot_title_left = price_day_date_utc+" blocks from local node"
    plot_title_right = "UTXOracle Consensus Price $"+f"{int(central_price):,} "
    bottom_note1 = "Consensus Data:"
    bottom_note2 = "this plot is identical and immutable for every litecoin node"
if block_mode:
    plot_title_left = "Local Node Blocks "+str(block_start_num)+"-"+str(block_finish_num)
    plot_title_right = "UTXOracle Block Window Price $"+f"{int(central_price):,}"
    bottom_note1 = "* Block Window Price "
    bottom_note2 = "may have node dependent differences data on the chain tip"

html_content = f'''<!DOCTYPE html>
<html>
<head>
<title>UTXOracle Local (Litecoin)</title>
<style>
    body {{ background-color: black; margin: 0; color: #CCCCCC; font-family: Arial, sans-serif; text-align: center; }}
    canvas {{ background-color: black; display: block; margin: auto; }}
</style>
</head>
<body>
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

ctx.fillStyle = "black"; ctx.fillRect(0,0,width,height);

ctx.font = "bold 36px Arial"; ctx.textAlign = "center";
ctx.fillStyle = "cyan"; ctx.fillText("UTXOracle", width/2 - 60, 40);
ctx.fillStyle = "lime"; ctx.fillText("Local", width/2 + 95, 40);

ctx.font = "24px Arial";
ctx.textAlign = "right"; ctx.fillStyle = "white"; ctx.fillText("{plot_title_left}", width/2, 80);
ctx.textAlign = "left"; ctx.fillStyle = "lime"; ctx.fillText("{plot_title_right}", width/2 + 10, 80);

ctx.strokeStyle = "white"; ctx.lineWidth = 1;
ctx.beginPath(); ctx.moveTo(marginLeft, marginTop); ctx.lineTo(marginLeft, marginTop + plotHeight); ctx.stroke();
ctx.beginPath(); ctx.moveTo(marginLeft, marginTop + plotHeight); ctx.lineTo(marginLeft + plotWidth, marginTop + plotHeight); ctx.stroke();
ctx.beginPath(); ctx.moveTo(marginLeft + plotWidth, marginTop); ctx.lineTo(marginLeft + plotWidth, marginTop + plotHeight); ctx.stroke();
ctx.beginPath(); ctx.moveTo(marginLeft, marginTop); ctx.lineTo(marginLeft + plotWidth, marginTop); ctx.stroke();

ctx.fillStyle = "white"; ctx.font = "20px Arial";
const yticks = 5;
for (let i=0;i<=yticks;i++) {{
  let p = ymin + (ymax-ymin)*i/yticks;
  let y = scaleY(p);
  ctx.beginPath(); ctx.moveTo(marginLeft-5,y); ctx.lineTo(marginLeft,y); ctx.stroke();
  ctx.textAlign="right"; ctx.fillText(Math.round(p).toLocaleString(), marginLeft-10, y+4);
}}

ctx.textAlign="center"; ctx.font="16px Arial";
for (let i=0;i<xtick_positions.length;i++) {{
  let x = scaleX(xtick_positions[i]);
  ctx.beginPath(); ctx.moveTo(x, marginTop+plotHeight); ctx.lineTo(x, marginTop+plotHeight+5); ctx.stroke();
  let parts = xtick_labels[i].split("\n");
  ctx.fillText(parts[0], x, marginTop+plotHeight+20);
  ctx.fillText(parts[1], x, marginTop+plotHeight+40);
}}

ctx.fillStyle="white"; ctx.font="20px Arial"; ctx.textAlign="center";
ctx.fillText("Block Height and UTC Time", marginLeft+plotWidth/2, height-48);
ctx.save(); ctx.translate(20, marginTop+plotHeight/2); ctx.rotate(-Math.PI/2);
ctx.fillText("LTC Price ($)", 0, 0); ctx.restore();

ctx.fillStyle="cyan";
for (let i=0;i<heights_smooth.length;i++) {{
  let x=scaleX(heights_smooth[i]), y=scaleY(prices[i]);
  ctx.fillRect(x,y,.75,.75);
}}

ctx.fillStyle="cyan"; ctx.font="20px Arial"; ctx.textAlign="left";
ctx.fillText("- {int(avg_price):,}", marginLeft + plotWidth + 1, scaleY({avg_price}));

ctx.font="24px Arial"; ctx.fillStyle="lime"; ctx.textAlign="right"; ctx.fillText("{bottom_note1}", 320, height-10);
ctx.fillStyle="white"; ctx.textAlign="left"; ctx.fillText("{bottom_note2}", 325, height-10);

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
    const hours = date.getUTCHours().toString().padStart(2,'0');
    const minutes = date.getUTCMinutes().toString().padStart(2,'0');
    const utcTime = `${{hours}}:${{minutes}} UTC`;

    tooltip.innerHTML = `Price: $${{Math.round(price).toLocaleString()}}<br>Block: ${{blockHeight.toLocaleString()}}<br>Time: ${{utcTime}}`;
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

filename = ".html"
if date_mode:
    filename = "UTXOracle_LTC_" + price_date_dash + filename
if block_mode:
    filename = "UTXOracle_LTC_" + str(block_start_num) + "-" + str(block_finish_num) + filename

import webbrowser
with open(filename, "w", encoding="utf-8") as f:
    f.write(html_content)
webbrowser.open('file://' + os.path.realpath(filename))
