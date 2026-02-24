#!/usr/bin/env python3
import argparse
import csv
import os

import psycopg2


def export_query(cur, query, out_file):
    cur.execute(query)
    rows = cur.fetchall()
    headers = [d[0] for d in cur.description]

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Export clearscope_e3 tables to CSV without psql.")
    parser.add_argument("--host", default="192.168.103.144")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", default="postgres")
    parser.add_argument("--db", default="clearscope_e3")
    parser.add_argument("--out-dir", default="/tmp/clearscope_export")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    conn = psycopg2.connect(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        database=args.db,
    )
    cur = conn.cursor()

    export_query(
        cur,
        "SELECT node_uuid, path FROM file_node_table",
        os.path.join(args.out_dir, "file_nodes.csv"),
    )
    export_query(
        cur,
        "SELECT node_uuid, path, cmd FROM subject_node_table",
        os.path.join(args.out_dir, "subject_nodes.csv"),
    )
    export_query(
        cur,
        "SELECT node_uuid, src_addr, src_port, dst_addr, dst_port FROM netflow_node_table",
        os.path.join(args.out_dir, "netflow_nodes.csv"),
    )
    export_query(
        cur,
        """
        WITH node_map AS (
          SELECT hash_id, node_uuid, 'File' AS node_type FROM file_node_table
          UNION ALL
          SELECT hash_id, node_uuid, 'Subject' AS node_type FROM subject_node_table
          UNION ALL
          SELECT hash_id, node_uuid, 'Netflow' AS node_type FROM netflow_node_table
        )
        SELECT
          s.node_uuid AS src_uuid,
          s.node_type AS src_type,
          d.node_uuid AS dst_uuid,
          d.node_type AS dst_type,
          e.event_uuid,
          e.operation,
          e.timestamp_rec
        FROM event_table e
        JOIN node_map s ON e.src_node = s.hash_id
        JOIN node_map d ON e.dst_node = d.hash_id
        """,
        os.path.join(args.out_dir, "events.csv"),
    )

    cur.close()
    conn.close()
    print(f"Done. CSV files are in: {args.out_dir}")


if __name__ == "__main__":
    main()
