import hashlib
import json
import time
from pathlib import Path


def guard(out, config, resume):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'RUN_CONFIG.json'
    config = dict(config)
    config.pop('resume', None)
    root = Path(__file__).resolve().parent
    config['code_hashes'] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ('adaptive_td.py', 'failfast.py', 'oracle_probe_schedule.py', 'make_probe_schedule.py',
                     'search_closed_loop_math50.py', 'run_deterministic_witness.py',
                     'collect_fp16_nokv_discovery.py', 'run_oracle_total_benefit_control.py')}
    if path.exists() and (not resume or json.loads(path.read_text()) != config):
        raise ValueError('Existing output has different configuration/code; use a new output directory')
    path.write_text(json.dumps(config, indent=2), encoding='utf-8')


def preserve(path):
    path = Path(path)
    if path.exists():
        path.rename(path.with_name(path.name + '_incomplete_' + str(time.time_ns())))
