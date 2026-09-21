"""Resolve current model repositories at install/update time and retain provenance."""
import argparse
import json
import os
from pathlib import Path
import uuid

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[2]


def download_model(model, destination, cache, previous=None, *, update=False):
    if previous and not update:
        if previous.get('repo_id', model['repo_id']) != model['repo_id']:
            raise RuntimeError('Model repository changed; rerun bootstrap with --update')
        files = previous.get('files', {})
        if files and all((destination / name).is_file() for name in files):
            return previous
        raise RuntimeError(f'Installed model files are missing: {destination}; rerun with --update')
    token = os.environ.get('HF_TOKEN') or False
    info = HfApi(token=token).model_info(model['repo_id'], files_metadata=True)
    if not info.sha or not info.siblings:
        raise RuntimeError(f'Model repository is empty: {model["repo_id"]}')
    # Resolve HEAD once, so an upstream publish cannot mix files during download.
    snapshot = Path(snapshot_download(repo_id=model['repo_id'], revision=info.sha,
                                     cache_dir=cache, max_workers=4, token=token))
    files = {}
    for entry in info.siblings:
        name = entry.rfilename
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise RuntimeError(f'Invalid model filename: {name}')
        path = snapshot / name
        if not path.is_file() or (entry.size is not None and path.stat().st_size != entry.size):
            raise RuntimeError(f'Missing/incomplete model file: {model["repo_id"]}/{name}')
        files[name] = path.stat().st_size
    destination.parent.mkdir(parents=True, exist_ok=True)
    link = destination.with_name(destination.name + '.next-' + uuid.uuid4().hex)
    link.symlink_to(snapshot.resolve(), target_is_directory=True)
    backup = None
    try:
        if destination.exists() and not destination.is_symlink():
            # Preserve checkpoints from the old directory-based installer.
            backup = destination.with_name(destination.name + '.previous-' + uuid.uuid4().hex)
            destination.rename(backup)
        try:
            link.replace(destination)
        except BaseException:
            if backup is not None:
                backup.rename(destination)
            raise
    finally:
        link.unlink(missing_ok=True)
    if backup:
        print(f'Previous model retained at {backup}', flush=True)
    return {'repo_id': model['repo_id'], 'revision': info.sha, 'files': files,
            'snapshot': str(snapshot.resolve())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profiles', nargs='+', default=['base'])
    parser.add_argument('--update', action='store_true')
    args = parser.parse_args()
    spec = json.loads((ROOT / 'deployment/repro/manifest.json').read_text())
    marker = ROOT / '.repro/models-verified.json'
    records = json.loads(marker.read_text()) if marker.exists() else {}
    for name, model in spec['models'].items():
        if model['profile'] not in args.profiles:
            continue
        print(f'Preparing {model["repo_id"]}', flush=True)
        records[name] = download_model(model, ROOT / '.repro/models' / model['directory'],
                                       ROOT / '.repro/cache/huggingface/hub', records.get(name),
                                       update=args.update)
        temporary = marker.with_suffix('.tmp')
        temporary.write_text(json.dumps(records, indent=2))
        temporary.replace(marker)
        print(f'Installed {model["repo_id"]}@{records[name]["revision"]}', flush=True)


if __name__ == '__main__':
    main()
