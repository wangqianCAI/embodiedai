"""Copy experiment scripts and relocate config without overwriting a workspace."""
import json
import os
from pathlib import Path
import shutil


def main():
    root = Path(os.environ['SO101_ROOT']).expanduser().resolve()
    source = Path(__file__).resolve().parent
    targets = [(p, root / 'scripts' / p.name) for p in (source / 'scripts').glob('*.py')]
    config = source / 'configs' / 'pi05_so101_50ep_instruction_v2_20261012.json'
    destination = root / 'configs' / config.name
    for _, target in targets + [(config, destination)]:
        if target.exists():
            raise FileExistsError(f'Refusing to overwrite {target}')
    for folder in ('scripts', 'configs', 'outputs', 'logs', 'models', 'datasets'):
        (root / folder).mkdir(parents=True, exist_ok=True)
    for src, target in targets:
        shutil.copy2(src, target)
    raw = config.read_text(encoding='utf-8').replace('/data/so101_pi0', root.as_posix())
    destination.write_text(json.dumps(json.loads(raw), indent=2), encoding='utf-8')
    print(f'Prepared {root}. Supply data, base model and tokenizer before training.')


if __name__ == '__main__':
    main()
