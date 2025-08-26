import os
import sqlite3
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

SCHEMA_PATH = Path('poc4/frontend/static/schemas/source_cards_schema_ds3.json')
DB_PATH = Path('poc4/frontend/static/schemas/cards_ds3.db')

# --- Type mapping ---
def map_type(t: str) -> str:
    t = (t or '').lower()
    if t.startswith('int'): return 'INTEGER'
    if t.startswith('number'): return 'REAL'
    if t.startswith('float') or t.startswith('decimal'): return 'REAL'
    if t.startswith('varchar') or t == 'string' or t.startswith('char'): return 'TEXT'
    if t in ('datetime','timestamp','date'): return 'TEXT'
    return 'TEXT'

# --- Helpers ---
FIRST_NAMES = ["Alice","Bob","Carol","David","Eve","Frank","Grace","Heidi","Ivan","Judy","Mallory","Niaj","Olivia","Peggy","Rupert","Sybil","Trent","Victor","Wendy","Yvonne"]
LAST_NAMES = ["Smith","Jones","Brown","Taylor","Wilson","Clark","Hall","Young","Allen","King","Scott","Green","Adams","Baker","Carter","Diaz","Evans","Foster","Gomez","Hayes"]
STREETS = ["Main St","Oak Ave","Pine Rd","Maple Ln","Cedar St","Birch Blvd","Sunset Dr","Hillcrest Ave","Riverview Rd","Lakeview Dr"]
MCCS = ["5411","5812","5732","5999","5912","4789","4899","4511","4111","5814"]
COUNTRIES = ["US","GB","CA","AU","DE","FR","ES","IT","IN","SG"]
PRODUCT_CDS = ["VISA","MC","AMEX","DISC"]
ACCT_STATUS = ["A","B","L"]
RESP_CDS = ["00","05","51","14","54","91"]


def rand_phone():
    return f"555-{random.randint(200,999)}-{random.randint(1000,9999)}"

def rand_timestamp(start_year=2022, end_year=2025):
    start = datetime(start_year,1,1)
    end = datetime(end_year,12,31,23,59,59)
    delta = end - start
    rand_sec = random.randint(0, int(delta.total_seconds()))
    return (start + timedelta(seconds=rand_sec)).strftime('%Y-%m-%d %H:%M:%S')


def load_schema(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_tables(conn: sqlite3.Connection, schema: dict):
    for table in schema['tables']:
        fields = []
        pk = None
        seen = set()
        for field in table['fields']:
            fname = field['name']
            if fname in seen:
                continue
            seen.add(fname)
            col = f"{fname} {map_type(field['type'])}"
            if field.get('primary_key'): pk = fname
            fields.append(col)
        if pk:
            fields.append(f"PRIMARY KEY({pk})")
        conn.execute(f'DROP TABLE IF EXISTS "{table["name"]}"')
        sql = f"CREATE TABLE \"{table['name']}\" ({', '.join(fields)})"
        conn.execute(sql)


def populate_cards(conn: sqlite3.Connection):
    # 20 rows per table
    n = 20
    # cardholder
    holders = []
    for i in range(1, n+1):
        fn = random.choice(FIRST_NAMES)
        ln = random.choice(LAST_NAMES)
        holders.append({
            'cardholder_id': i,
            'customer_ref': f"CUST{1000+i}",
            'first_name': fn,
            'last_name': ln,
            'email_addr': f"{fn.lower()}.{ln.lower()}{i}@cards.com",
            'phone_num': rand_phone(),
            'postal_addr': f"{random.randint(100,999)} {random.choice(STREETS)}",
            'created_ts': rand_timestamp(2021, 2025)
        })
    # card_account (1:1 with holder)
    accounts = []
    for i, h in enumerate(holders, start=1):
        accounts.append({
            'card_account_id': 1000 + i,
            'cardholder_id': h['cardholder_id'],
            'product_cd': random.choice(PRODUCT_CDS),
            'account_status_cd': random.choice(ACCT_STATUS),
            'open_dt': rand_timestamp(2021, 2025),
            'close_dt': None,
            'credit_limit_amt': round(random.uniform(1000, 20000), 2),
            'current_bal_amt': round(random.uniform(0, 15000), 2),
            'avail_credit_amt': round(random.uniform(0, 15000), 2),
        })
    # card (1:1 with account)
    cards = []
    for i, a in enumerate(accounts, start=1):
        bin_val = random.choice(["411111","550000","340000","601100"])  # Visa/Master/Amex/Discover bins
        last4 = f"{random.randint(0,9999):04d}"
        cards.append({
            'card_id': 2000 + i,
            'card_account_id': a['card_account_id'],
            'pan_token': os.urandom(16).hex(),
            'bin': bin_val,
            'last4': last4,
            'exp_month': f"{random.randint(1,12):02d}",
            'exp_year': f"{random.randint(25,30):02d}",
            'card_status_cd': random.choice(["A","B","L"]),
            'embossed_name': f"{holders[i-1]['first_name']} {holders[i-1]['last_name']}",
            'issue_dt': rand_timestamp(2022, 2025),
            'block_dt': None,
        })
    # merchants
    merchants = []
    for i in range(1, n+1):
        merchants.append({
            'merchant_id': 3000 + i,
            'merchant_name': f"Merchant {i:03d}",
            'mcc': random.choice(MCCS),
            'country_cd': random.choice(COUNTRIES)
        })
    # card_auth
    auths = []
    for i in range(1, n+1):
        c = random.choice(cards)
        m = random.choice(merchants)
        auths.append({
            'auth_id': 4000 + i,
            'card_id': c['card_id'],
            'auth_ts': rand_timestamp(2023, 2025),
            'amount': round(random.uniform(1, 500), 2),
            'currency_cd': random.choice(["USD","EUR","GBP","INR","AUD"]),
            'response_cd': random.choice(RESP_CDS),
            'auth_code': f"{random.randint(100000,999999)}",
            'merchant_id': m['merchant_id']
        })
    # card_txn
    txns = []
    for i in range(1, n+1):
        c = random.choice(cards)
        maybe_auth = random.choice(auths)
        txns.append({
            'txn_id': 5000 + i,
            'card_id': c['card_id'],
            'post_ts': rand_timestamp(2023, 2025),
            'auth_id': maybe_auth['auth_id'],
            'dr_cr_ind': random.choice(['D','C']),
            'amount': round(random.uniform(1, 500), 2),
            'currency_cd': random.choice(["USD","EUR","GBP","INR","AUD"]),
            'network_ref_id': os.urandom(8).hex(),
            'merchant_id': maybe_auth['merchant_id']
        })
    # dispute_case (20 as well, linked to random txn)
    cases = []
    for i in range(1, n+1):
        t = random.choice(txns)
        opened = rand_timestamp(2023, 2025)
        cases.append({
            'case_id': 6000 + i,
            'txn_id': t['txn_id'],
            'case_open_ts': opened,
            'reason_cd': random.choice(["FRD","NOTASDESCR","NOREC","DUPL","RET"]),
            'status_cd': random.choice(["O","P","R","C"]),
            'resolved_ts': None
        })

    # Insert all
    def insert_bulk(table_name: str, rows: list):
        if not rows: return
        cols = list(rows[0].keys())
        qcols = ', '.join([f'"{c}"' for c in cols])
        placeholders = ', '.join(['?' for _ in cols])
        for r in rows:
            vals = [r.get(c) for c in cols]
            conn.execute(f"INSERT INTO \"{table_name}\" ({qcols}) VALUES ({placeholders})", vals)

    insert_bulk('cardholder', holders)
    insert_bulk('card_account', accounts)
    insert_bulk('card', cards)
    insert_bulk('merchant', merchants)
    insert_bulk('card_auth', auths)
    insert_bulk('card_txn', txns)
    insert_bulk('dispute_case', cases)
    conn.commit()


def main():
    schema = load_schema(SCHEMA_PATH)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH.as_posix())
    try:
        create_tables(conn, schema)
        populate_cards(conn)
    finally:
        conn.close()
    print(f"Created DS3 cards database at {DB_PATH} with 20 rows per table.")

if __name__ == '__main__':
    main()
