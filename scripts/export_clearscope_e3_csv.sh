#!/usr/bin/env bash
set -euo pipefail

# Export clearscope_e3 tables from PostgreSQL to CSV files.
#
# Optional overrides:
#   PGHOST PGPORT PGUSER PGPASSWORD PGDATABASE
#   EXPORT_DIR

PGHOST="${PGHOST:-192.168.103.144}"
PGPORT="${PGPORT:-8888}"
PGUSER="${PGUSER:-postgres}"
PGPASSWORD="${PGPASSWORD:-postgres}"
PGDATABASE="${PGDATABASE:-clearscope_e3}"
EXPORT_DIR="${EXPORT_DIR:-/tmp/clearscope_export}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/export_clearscope_e3_csv.sh

Optional overrides:
  PGHOST PGPORT PGUSER PGPASSWORD PGDATABASE
  EXPORT_DIR

Example:
  EXPORT_DIR=/tmp/clearscope_csv bash scripts/export_clearscope_e3_csv.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v psql >/dev/null 2>&1; then
  echo "Missing command: psql"
  exit 1
fi

mkdir -p "$EXPORT_DIR"

FILE_CSV="$EXPORT_DIR/file_nodes.csv"
SUBJECT_CSV="$EXPORT_DIR/subject_nodes.csv"
NETFLOW_CSV="$EXPORT_DIR/netflow_nodes.csv"
EVENTS_CSV="$EXPORT_DIR/events.csv"

echo "[1/4] Export file_node_table -> $FILE_CSV"
env PGPASSWORD="$PGPASSWORD" psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" \
  -c "\copy (SELECT node_uuid, path FROM file_node_table) TO '$FILE_CSV' CSV HEADER"

echo "[2/4] Export subject_node_table -> $SUBJECT_CSV"
env PGPASSWORD="$PGPASSWORD" psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" \
  -c "\copy (SELECT node_uuid, path, cmd FROM subject_node_table) TO '$SUBJECT_CSV' CSV HEADER"

echo "[3/4] Export netflow_node_table -> $NETFLOW_CSV"
env PGPASSWORD="$PGPASSWORD" psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" \
  -c "\copy (SELECT node_uuid, src_addr, src_port, dst_addr, dst_port FROM netflow_node_table) TO '$NETFLOW_CSV' CSV HEADER"

echo "[4/4] Export event_table(joined with node_map) -> $EVENTS_CSV"
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

echo "Done. Exported CSV files in: $EXPORT_DIR"
