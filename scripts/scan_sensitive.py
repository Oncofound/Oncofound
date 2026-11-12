"""Scan all release files for credentials, private paths and data artifacts.

Only finding locations and rule names are printed; matched text is not printed.
This is a repeatable targeted scan, not a guarantee against all possible leaks.
"""
from pathlib import Path
import re
import json
import os
import argparse
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RULES = {
    'private_storage_path': re.compile(r'/(?:gpfs|ai/home|home/[A-Za-z0-9_]+|Users/[A-Za-z0-9_]+)/', re.I),
    'private_windows_path': re.compile(r'[A-Za-z]:[\\/]Users[\\/][A-Za-z0-9_.-]+[\\/]', re.I),
    'private_ip': re.compile(r'\b(?:192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b'),
    'private_key': re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'),
    'credential': re.compile(r'\b(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,}|(?:AKIA|ASIA)[A-Z0-9]{16})\b'),
    'credential_assignment': re.compile(r'(?i)(?:api[_-]?key|password|access[_-]?token|secret[_-]?key)\s*[:=]\s*[\x22\x27][^\x22\x27\s]{8,}[\x22\x27]'),
}
RULES['contact_email'] = re.compile(r'\b[A-Za-z0-9._%+-]+@(?:gmail\.com|qq\.com|163\.com|outlook\.com|hotmail\.com)\b', re.I)
RULES['credential_in_url'] = re.compile(r'https?://[^\s\x22\x27]*[?&](?:token|password|api_key|access_token|secret)=[^\s&\x22\x27]+', re.I)
private_identifiers = [v for v in os.environ.get('SENSITIVE_IDENTIFIERS', '').split(',') if v]
if private_identifiers:
    RULES['private_identifier'] = re.compile(r'\b(?:' + '|'.join(re.escape(v) for v in private_identifiers) + r')\b')
private_literals = [v for v in os.environ.get('SENSITIVE_STRINGS', '').split(',') if v]
if private_literals:
    RULES['private_literal'] = re.compile('|'.join(re.escape(v) for v in private_literals))
BAD_SUFFIXES = ('.pth', '.pt', '.ckpt', '.nii', '.nii.gz', '.safetensors', '.log',
                '.csv', '.jsonl', '.npy', '.npz', '.dcm', '.h5', '.hdf5',
                '.zip', '.7z', '.tar', '.tar.gz', '.pem', '.key', '.p12', '.pfx')
PRIVATE_NAMES = {'.env', '.git-credentials', 'id_rsa', 'id_ed25519', 'credentials', '.ds_store'}


def file_rules(rel):
    path = Path(rel)
    rules = []
    if rel.lower().endswith(BAD_SUFFIXES):
        rules.append('data_or_weight_artifact')
    if path.name.lower() in PRIVATE_NAMES or path.name.startswith('.env.'):
        rules.append('private_auth_or_metadata_file')
    if any(part in ('.ssh', '.aws', '.gcloud', '.ipynb_checkpoints') for part in path.parts):
        rules.append('private_auth_or_metadata_directory')
    return rules


def scan_text(text, rel):
    findings = []
    for rule, pattern in RULES.items():
        for match in pattern.finditer(text):
            findings.append({'file': rel, 'line': text[:match.start()].count('\n') + 1, 'rule': rule})
    if rel.endswith('.ipynb'):
        try:
            notebook = json.loads(text)
        except (ValueError, TypeError):
            findings.append({'file': rel, 'rule': 'invalid_notebook_json'})
        else:
            def walk(value, field):
                if isinstance(value, str):
                    for rule, pattern in RULES.items():
                        if pattern.search(value):
                            findings.append({'file': rel, 'field': field, 'rule': rule})
                elif isinstance(value, dict):
                    for key, item in value.items():
                        if key in ('image/png', 'image/jpeg', 'application/pdf', 'attachments') and item:
                            findings.append({'file': rel, 'field': field + '.' + str(key), 'rule': 'embedded_media_requires_review'})
                        walk(item, field + '.' + str(key))
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        walk(item, field + '.' + str(i))
            walk(notebook, '$')
    return findings


def scan_git_history(root=None):
    """Inspect every published ref, including files later deleted or ignored."""
    root = Path(root or ROOT)
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args])
    commits = git('rev-list', '--all').decode().splitlines()
    findings = [];seen = set();blobs = 0
    for commit in commits:
        metadata = git('show', '-s', '--format=%B%n%an <%ae>%n%cn <%ce>', commit).decode('utf-8', errors='replace')
        findings.extend({**item, 'commit': commit} for item in scan_text(metadata, '<commit_metadata>'))
        for entry in git('ls-tree', '-r', '-z', commit).split(b'\0'):
            if not entry:
                continue
            meta, raw_path = entry.split(b'\t', 1)
            mode, kind, oid = meta.decode().split()
            rel = raw_path.decode('utf-8', errors='replace')
            identity = (rel, oid)
            if identity in seen:
                continue
            seen.add(identity)
            rules = file_rules(rel)
            if mode == '120000':rules.append('symlink')
            if kind != 'blob':rules.append('unexpected_gitlink')
            if rules:
                findings.extend({'file': rel, 'rule': r, 'commit': commit} for r in rules)
                continue
            if int(git('cat-file', '-s', oid)) > 8 * 1024 * 1024:
                findings.append({'file': rel, 'rule': 'oversized_source_blob', 'commit': commit})
                continue
            blobs += 1
            try:text = git('cat-file', 'blob', oid).decode('utf-8')
            except UnicodeDecodeError:
                findings.append({'file': rel, 'rule': 'unexpected_binary', 'commit': commit})
                continue
            findings.extend({**item, 'commit': commit} for item in scan_text(text, rel))
    return {'commits_scanned': len(commits), 'file_versions_scanned': len(seen),
            'text_versions_scanned': blobs, 'findings': findings}


def scan():
    findings = []
    files = 0
    for p in sorted(ROOT.rglob('*')):
        if any(part in ('__pycache__', '.git', '.venv') for part in p.relative_to(ROOT).parts):
            continue
        if p.is_symlink():
            findings.append({'file': str(p.relative_to(ROOT)), 'rule': 'symlink'})
            continue
        if not p.is_file():
            continue
        files += 1
        rel = str(p.relative_to(ROOT))
        findings.extend({'file': rel, 'rule': rule} for rule in file_rules(rel))
        try:
            text = p.read_text()
        except UnicodeDecodeError:
            findings.append({'file': rel, 'rule': 'unexpected_binary'})
            continue
        findings.extend(scan_text(text, rel))
    return {'files_scanned': files, 'findings': findings}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--git-history', action='store_true', help='Also scan every committed file version in all Git refs')
    args = parser.parse_args()
    result = scan()
    if args.git_history:
        result['git_history'] = scan_git_history()
    print(json.dumps(result, indent=2))
    raise SystemExit(bool(result['findings'] or result.get('git_history', {}).get('findings')))
