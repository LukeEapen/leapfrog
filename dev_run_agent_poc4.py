import os, json
from poc4.backend.agents import schema_mapping_agent, migration_execution_agent
import sqlite3
from sqlite_data_transfer_demo import reset_all_dbs as demo_reset

root = os.path.dirname(__file__)
schemas_dir = os.path.join(root, 'poc4', 'frontend', 'static', 'schemas')

src_schema_path = os.path.join(schemas_dir, 'source_schema.json')
leg_schema_path = os.path.join(schemas_dir, 'legacy_schema.json')
tgt_schema_path = os.path.join(schemas_dir, 'target_schema.json')

# Ensure DBs exist by resetting/creating from schemas with sample data
demo_reset(src_schema_path, leg_schema_path, tgt_schema_path)
try:
    with sqlite3.connect(os.path.join(schemas_dir, 'target.db')) as _c:
        (pre_cnt,) = _c.execute('SELECT COUNT(1) FROM "banking_olap_flat_exact"').fetchone()
        print('pre_count_banking_olap_flat_exact=', pre_cnt)
except Exception as e:
    print('pre_count_error=', e)

# Load schemas
with open(src_schema_path, 'r') as f:
    source_schema = json.load(f)
with open(leg_schema_path, 'r') as f:
    legacy_schema = json.load(f)
with open(tgt_schema_path, 'r') as f:
    target_schema = json.load(f)

# Build field->table map helpers
def build_field_table_map(schema):
    m = {}
    for t in schema.get('tables', []):
        for fld in t.get('fields', []):
            m[fld.get('name')] = t.get('name')
    return m

src_ft = build_field_table_map(source_schema)
leg_ft = build_field_table_map(legacy_schema)

def annotate(mapping, origin, ft_map, tgt_schema):
    # Build target field->table
    tgt_ft = {}
    for t in tgt_schema.get('tables', []):
        for fld in t.get('fields', []):
            tgt_ft[fld.get('name')] = t.get('name')
    out = []
    for m in mapping:
        m['origin'] = origin
        m['source_table'] = ft_map.get(m.get('source'))
        if m.get('target'):
            m['target_table'] = tgt_ft.get(m.get('target'))
        out.append(m)
    return out

# Generate mappings
src_map = schema_mapping_agent.map_schema(source_schema, target_schema)
leg_map = schema_mapping_agent.map_schema(legacy_schema, target_schema)
full_map = annotate(src_map, 'source', src_ft, target_schema) + annotate(leg_map, 'legacy', leg_ft, target_schema)

# Run agent
# Prefer email as the entity key to avoid collisions with pre-seeded target rows
res = migration_execution_agent.run(full_map, key_config={
    'banking_olap_flat_exact': ['email']
})

# Print concise result
print(json.dumps({
    'status': res.get('status'),
    'global_unique_entities': res.get('global_unique_entities'),
    'summary': res.get('summary'),
}, indent=2))

# Print final target count for the main flat table
tgt_db = os.path.join(schemas_dir, 'target.db')
with sqlite3.connect(tgt_db) as conn:
    try:
        (cnt,) = conn.execute('SELECT COUNT(1) FROM "banking_olap_flat_exact"').fetchone()
        print('target_row_count_banking_olap_flat_exact=', cnt)
    except Exception as e:
        print('target_count_error=', e)
