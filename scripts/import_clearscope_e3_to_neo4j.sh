#!/usr/bin/env bash
set -euo pipefail

# One-shot import pipeline:
#   PostgreSQL(clearscope_e3) -> CSV -> Neo4j(clearscope)
#
# Requirements:
#   - psql available in PATH
#   - cypher-shell available in PATH
#   - Neo4j can read CSV files from NEO4J_IMPORT_DIR
#
# Defaults follow the values provided by the user, but can be overridden by env vars.

PGHOST="${PGHOST:-192.168.103.144}"
PGPORT="${PGPORT:-8888}"
PGUSER="${PGUSER:-postgres}"
PGPASSWORD="${PGPASSWORD:-postgres}"
PGDATABASE="${PGDATABASE:-clearscope_e3}"

NEO4J_URI="${NEO4J_URI:-neo4j://192.168.103.144:7687}"
NEO4J_USER="${NEO4J_USER:-neo4j}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-woshigexing}"
NEO4J_DB="${NEO4J_DB:-clearscope}"

# This must be Neo4j's import directory path on the machine running Neo4j.
# For Neo4j Desktop, set this to your DBMS import folder.
NEO4J_IMPORT_DIR="${NEO4J_IMPORT_DIR:-/tmp/neo4j-import}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/import_clearscope_e3_to_neo4j.sh

Optional overrides:
  PGHOST PGPORT PGUSER PGPASSWORD PGDATABASE
  NEO4J_URI NEO4J_USER NEO4J_PASSWORD NEO4J_DB
  NEO4J_IMPORT_DIR

Example:
  NEO4J_IMPORT_DIR="$HOME/.Neo4jDesktop/relate-data/dbmss/<dbms-id>/import" \
  bash scripts/import_clearscope_e3_to_neo4j.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

for cmd in psql cypher-shell; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing command: $cmd"
    exit 1
  fi
done

mkdir -p "$NEO4J_IMPORT_DIR"

FILE_CSV="$NEO4J_IMPORT_DIR/file_nodes.csv"
SUBJECT_CSV="$NEO4J_IMPORT_DIR/subject_nodes.csv"
NETFLOW_CSV="$NEO4J_IMPORT_DIR/netflow_nodes.csv"
EVENTS_CSV="$NEO4J_IMPORT_DIR/events.csv"

echo "[1/4] Exporting CSV from PostgreSQL to $NEO4J_IMPORT_DIR ..."
env PGPASSWORD="$PGPASSWORD" psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" \
  -c "\copy (SELECT node_uuid, path FROM file_node_table) TO '$FILE_CSV' CSV HEADER"

env PGPASSWORD="$PGPASSWORD" psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" \
  -c "\copy (SELECT node_uuid, path, cmd FROM subject_node_table) TO '$SUBJECT_CSV' CSV HEADER"

env PGPASSWORD="$PGPASSWORD" psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" \
  -c "\copy (SELECT node_uuid, src_addr, src_port, dst_addr, dst_port FROM netflow_node_table) TO '$NETFLOW_CSV' CSV HEADER"

env PGPASSWORD="$PGPASSWORD" psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" \
  -c "\copy (
    WITH node_map AS (
      SELECT hash_id, node_uuid FROM file_node_table
      UNION ALL
      SELECT hash_id, node_uuid FROM subject_node_table
      UNION ALL
      SELECT hash_id, node_uuid FROM netflow_node_table
    )
    SELECT
      s.node_uuid AS src_uuid,
      d.node_uuid AS dst_uuid,
      e.event_uuid,
      e.operation,
      e.timestamp_rec
    FROM event_table e
    JOIN node_map s ON e.src_node = s.hash_id
    JOIN node_map d ON e.dst_node = d.hash_id
  ) TO '$EVENTS_CSV' CSV HEADER"

echo "[2/4] Preparing Cypher import script ..."
TMP_CYPHER="$(mktemp)"
cat > "$TMP_CYPHER" <<'EOF'
CREATE CONSTRAINT file_uuid IF NOT EXISTS FOR (n:File) REQUIRE n.node_uuid IS UNIQUE;
CREATE CONSTRAINT subject_uuid IF NOT EXISTS FOR (n:Subject) REQUIRE n.node_uuid IS UNIQUE;
CREATE CONSTRAINT netflow_uuid IF NOT EXISTS FOR (n:Netflow) REQUIRE n.node_uuid IS UNIQUE;

LOAD CSV WITH HEADERS FROM 'file:///file_nodes.csv' AS row
CALL {
  WITH row
  MERGE (n:File {node_uuid: row.node_uuid})
  SET n.path = CASE WHEN row.path = '' THEN null ELSE row.path END
} IN TRANSACTIONS OF 10000 ROWS;

LOAD CSV WITH HEADERS FROM 'file:///subject_nodes.csv' AS row
CALL {
  WITH row
  MERGE (n:Subject {node_uuid: row.node_uuid})
  SET n.path = CASE WHEN row.path = '' THEN null ELSE row.path END,
      n.cmd = CASE WHEN row.cmd = '' THEN null ELSE row.cmd END
} IN TRANSACTIONS OF 10000 ROWS;

LOAD CSV WITH HEADERS FROM 'file:///netflow_nodes.csv' AS row
CALL {
  WITH row
  MERGE (n:Netflow {node_uuid: row.node_uuid})
  SET n.src_addr = CASE WHEN row.src_addr = '' THEN null ELSE row.src_addr END,
      n.src_port = CASE WHEN row.src_port = '' THEN null ELSE row.src_port END,
      n.dst_addr = CASE WHEN row.dst_addr = '' THEN null ELSE row.dst_addr END,
      n.dst_port = CASE WHEN row.dst_port = '' THEN null ELSE row.dst_port END
} IN TRANSACTIONS OF 10000 ROWS;

LOAD CSV WITH HEADERS FROM 'file:///events.csv' AS row
CALL {
  WITH row
  MATCH (s {node_uuid: row.src_uuid})
  MATCH (d {node_uuid: row.dst_uuid})
  MERGE (s)-[r:EVENT {event_uuid: row.event_uuid}]->(d)
  SET r.operation = row.operation,
      r.timestamp_rec = toInteger(row.timestamp_rec)
} IN TRANSACTIONS OF 10000 ROWS;

MATCH (n:File) RETURN 'File nodes' AS metric, count(n) AS value
UNION ALL
MATCH (n:Subject) RETURN 'Subject nodes' AS metric, count(n) AS value
UNION ALL
MATCH (n:Netflow) RETURN 'Netflow nodes' AS metric, count(n) AS value
UNION ALL
MATCH ()-[r:EVENT]->() RETURN 'EVENT rels' AS metric, count(r) AS value;
EOF

echo "[3/4] Importing into Neo4j ($NEO4J_URI, db=$NEO4J_DB) ..."
cypher-shell -a "$NEO4J_URI" -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" -d "$NEO4J_DB" -f "$TMP_CYPHER"

echo "[4/4] Done."
echo "CSV files:"
echo "  $FILE_CSV"
echo "  $SUBJECT_CSV"
echo "  $NETFLOW_CSV"
echo "  $EVENTS_CSV"

rm -f "$TMP_CYPHER"
