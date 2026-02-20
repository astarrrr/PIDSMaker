#!/usr/bin/env python3
import argparse
import os
from dataclasses import dataclass
from typing import Optional

import psycopg2
from psycopg2 import sql


def _is_missing_expr(column_name: str) -> sql.SQL:
    # Missing means NULL, empty string, or literal "null" (case-insensitive).
    return sql.SQL(
        "({c} IS NULL OR btrim({c}) = '' OR lower(btrim({c})) = 'null')"
    ).format(c=sql.Identifier(column_name))


@dataclass
class CheckSpec:
    label: str
    table: str
    required_columns: tuple[str, ...]


SPECS = (
    CheckSpec("file", "file_node_table", ("path",)),
    CheckSpec("subject", "subject_node_table", ("path", "cmd")),
    CheckSpec("netflow", "netflow_node_table", ("src_addr", "src_port", "dst_addr", "dst_port")),
)


def _build_where_any_missing(required_columns: tuple[str, ...]) -> sql.SQL:
    conditions = [_is_missing_expr(col) for col in required_columns]
    query = conditions[0]
    for cond in conditions[1:]:
        query = sql.SQL("({q} OR {cond})").format(q=query, cond=cond)
    return query


def _build_where_all_missing(required_columns: tuple[str, ...]) -> sql.SQL:
    conditions = [_is_missing_expr(col) for col in required_columns]
    query = conditions[0]
    for cond in conditions[1:]:
        query = sql.SQL("({q} AND {cond})").format(q=query, cond=cond)
    return query


def _count_rows(cur, table: str, where_expr: Optional[sql.SQL] = None) -> int:
    if where_expr is None:
        query = sql.SQL("SELECT COUNT(*) FROM {table}").format(table=sql.Identifier(table))
        cur.execute(query)
    else:
        query = sql.SQL("SELECT COUNT(*) FROM {table} WHERE {where_expr}").format(
            table=sql.Identifier(table), where_expr=where_expr
        )
        cur.execute(query)
    return int(cur.fetchone()[0])


def _sample_missing_rows(cur, spec: CheckSpec, sample_limit: int):
    select_cols = ["index_id", "node_uuid", *spec.required_columns]
    select_sql = sql.SQL(", ").join([sql.Identifier(c) for c in select_cols])
    where_any = _build_where_any_missing(spec.required_columns)

    query = sql.SQL(
        "SELECT {cols} FROM {table} WHERE {where_any} ORDER BY index_id NULLS LAST LIMIT %s"
    ).format(
        cols=select_sql,
        table=sql.Identifier(spec.table),
        where_any=where_any,
    )
    cur.execute(query, (sample_limit,))
    return select_cols, cur.fetchall()


def _fmt_pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.00%"
    return f"{(100.0 * numerator / denominator):.2f}%"


def main():
    parser = argparse.ArgumentParser(
        description="Check nodes with missing attributes in DARPA/OPTC PostgreSQL databases."
    )
    parser.add_argument("--db", default="clearscope_e3", help="Database name (default: clearscope_e3)")
    parser.add_argument("--host", default=os.getenv("PGHOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PGPORT", 5432)))
    parser.add_argument("--user", default=os.getenv("PGUSER", "postgres"))
    parser.add_argument("--password", default=os.getenv("PGPASSWORD", "postgres"))
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=20,
        help="How many missing rows to print per node table.",
    )
    args = parser.parse_args()

    conn = psycopg2.connect(
        database=args.db,
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
    )
    cur = conn.cursor()

    print(f"[db] {args.db} @ {args.host}:{args.port} user={args.user}")
    print("")

    total_nodes = 0
    total_any_missing = 0
    total_all_missing = 0

    for spec in SPECS:
        where_any = _build_where_any_missing(spec.required_columns)
        where_all = _build_where_all_missing(spec.required_columns)

        total = _count_rows(cur, spec.table)
        any_missing = _count_rows(cur, spec.table, where_any)
        all_missing = _count_rows(cur, spec.table, where_all)

        total_nodes += total
        total_any_missing += any_missing
        total_all_missing += all_missing

        print(f"[{spec.label}] table={spec.table}")
        print(f"  total: {total}")
        print(f"  missing_any_required: {any_missing} ({_fmt_pct(any_missing, total)})")
        print(f"  missing_all_required: {all_missing} ({_fmt_pct(all_missing, total)})")

        cols, samples = _sample_missing_rows(cur, spec, args.sample_limit)
        if samples:
            print(f"  samples (up to {args.sample_limit}):")
            print("    " + " | ".join(cols))
            for row in samples:
                print("    " + " | ".join(str(v) if v is not None else "NULL" for v in row))
        else:
            print("  samples: none")
        print("")

    print("[summary]")
    print(f"  total_nodes: {total_nodes}")
    print(f"  total_missing_any_required: {total_any_missing} ({_fmt_pct(total_any_missing, total_nodes)})")
    print(f"  total_missing_all_required: {total_all_missing} ({_fmt_pct(total_all_missing, total_nodes)})")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
