#!/usr/bin/env python3
"""Quick verification that recent audit fixes compile and import."""
import py_compile
import sys
import os

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, REPO_ROOT)

FILES = [
    'tool/api.py',
    'tool/calibration.py',
    'tool/adversarial_defense.py',
    'tool/feature_extractor.py',
    'scripts/train_attribution.py',
    'scripts/train_calibration.py',
]


def main():
    for rel in FILES:
        path = os.path.join(REPO_ROOT, rel)
        print(f'Compiling {rel} ...')
        py_compile.compile(path, doraise=True)

    print('Importing calibration...')
    from tool.calibration import calibrate_probability
    print('calibrate_probability(0.9, 10) =', calibrate_probability(0.9, 10))

    print('Importing api app...')
    from tool.api import app
    print('api app imported OK')

    print('Importing adversarial defense...')
    from tool.adversarial_defense import normalize_text_defensive
    cleaned = normalize_text_defensive('Thіs іs а tеst\u200b!')
    assert cleaned == 'This is a test!', repr(cleaned)
    print('adversarial defense OK:', cleaned)

    print('\nAll verification checks passed.')


if __name__ == '__main__':
    main()
