"""Summarize B1 sweep JSONL into a markdown table plus D1/2 estimate.

Inputs : data/b1_m1/sweep_*.jsonl  (summary records written by benchmark1.exe)
Output : markdown on stdout (redirect to audit/b1_m1_report.md)
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATTERN = os.path.join(ROOT, 'data', 'b1_m1', 'sweep_*.jsonl')


def load_rows():
    rows = []
    for path in sorted(glob.glob(PATTERN)):
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get('event') != 'summary':
                    continue
                tag = rec.get('tag', '')
                if not tag.startswith('sweep_') or '_D' not in tag:
                    continue
                cond, delay = tag[len('sweep_'):].rsplit('_D', 1)
                try:
                    delay = int(delay)
                except ValueError:
                    continue
                rows.append((cond, delay, float(rec.get('test_acc', 0.0)), rec))
    return rows


def d_half(points):
    """Delay where accuracy crosses 50%. None means it never drops below 50%."""
    prev = None
    for delay, acc in sorted(points):
        if acc < 50.0:
            if prev is None:
                return 0.0
            d0, a0 = prev
            if a0 <= acc:
                return float(d0)
            return d0 + (a0 - 50.0) / (a0 - acc) * (delay - d0)
        prev = (delay, acc)
    return None


def main():
    rows = load_rows()
    if not rows:
        print('no sweep_*.jsonl found under data/b1_m1/')
        return 1

    conds = sorted({r[0] for r in rows})
    delays = sorted({r[1] for r in rows})
    acc = {(c, d): a for c, d, a, _ in rows}

    print('# Benchmark-1 sweep report')
    print()
    print('| D | ' + ' | '.join(conds) + ' |')
    print('|' + '---|' * (len(conds) + 1))
    for d in delays:
        cells = []
        for c in conds:
            v = acc.get((c, d))
            cells.append('-' if v is None else '%.1f%%' % v)
        print('| %d | %s |' % (d, ' | '.join(cells)))

    print()
    print('| condition | D1/2 | acc@200 | acc@500 |')
    print('|---|---|---|---|')
    for c in conds:
        pts = [(d, acc[(c, d)]) for d in delays if (c, d) in acc]
        h = d_half(pts)
        hstr = '>%d' % max(delays) if h is None else '%.0f' % h
        a200 = acc.get((c, 200))
        a500 = acc.get((c, 500))
        print('| %s | %s | %s | %s |' % (
            c, hstr,
            '-' if a200 is None else '%.1f%%' % a200,
            '-' if a500 is None else '%.1f%%' % a500))

    print()
    print('Source: `data/b1_m1/sweep_*.jsonl` (summary records). '
          'Each row is one independent process/brain instance.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
