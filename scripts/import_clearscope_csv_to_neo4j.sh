#!/usr/bin/env bash
set -euo pipefail

# Import pre-exported CSV files into Neo4j.
#
# Optional overrides:
#   NEO4J_URI NEO4J_USER NEO4J_PASSWORD NEO4J_DB
#   CSV_DIR
#
# CSV_DIR must contain:
#   file_nodes.csv
#   subject_nodes.csv
#   netflow_nodes.csv
#   events.csv
#
# CSV_DIR must be Neo4j's import directory (or mapped to it).

NEO4J_URI="${NEO4J_URI:-neo4j://192.168.103.144:7687}"
NEO4J_USER="${NEO4J_USER:-neo4j}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-woshigexing}"
NEO4J_DB="${NEO4J_DB:-clearscope}"
CSV_DIR="${CSV_DIR:-/tmp/neo4j-import}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/import_clearscope_csv_to_neo4j.sh

Optional overrides:
  NEO4J_URI NEO4J_USER NEO4J_PASSWORD NEO4J_DB
  CSV_DIR

Example:
  CSV_DIR="$HOME/.Neo4jDesktop/relate-data/dbmss/<dbms-id>/import" \
  bash scripts/import_clearscope_csv_to_neo4j.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v cypher-shell >/dev/null 2>&1; then
  echo "Missing command: cypher-shell"
  exit 1
fi

for f in file_nodes.csv subject_nodes.csv netflow_nodes.csv events.csv; do
  if [[ ! -f "$CSV_DIR/$f" ]]; then
    echo "Missing CSV: $CSV_DIR/$f"
    exit 1
  fi
done

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

echo "Importing CSV from $CSV_DIR into $NEO4J_URI (db=$NEO4J_DB)"
echo "Note: Neo4j must be configured so file:/// points to $CSV_DIR."
cypher-shell -a "$NEO4J_URI" -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" -d "$NEO4J_DB" -f "$TMP_CYPHER"

rm -f "$TMP_CYPHER"
echo "Done."
