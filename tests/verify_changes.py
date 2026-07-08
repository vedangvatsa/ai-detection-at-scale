#!/usr/bin/env python3
"""Quick verification that recent audit fixes compile and import."""
import py_compile
import sys
import os

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, REPO_ROOT)

FILES = [
    'tool/api.py',
    'tool/api_security.py',
    'tool/calibration.py',
    'tool/adversarial_defense.py',
    'tool/feature_extractor.py',
    'tool/public_api.py',
    'tool/hybrid_detector.py',
    'tool/sentence_analyzer.py',
    'tool/register_classifier.py',
    'scripts/train_attribution.py',
    'scripts/train_calibration.py',
    'scripts/download_assets.py',
    'scripts/generate_checksums.py',
    'scripts/audit_notebooks.py',
    'scripts/17_api_demo.py',
    'scripts/10_tpr_at_low_fpr.py',
    'scripts/12_humanized_eval.py',
    'scripts/21_mage_hc3_benchmark.py',
    'scripts/22_turingbench_benchmark.py',
    'scripts/23_hc3_benchmark.py',
    'scripts/05_mustdo_analyses.py',
    'scripts/06_adversarial_eval.py',
    'scripts/19_extract_35_features.py',
    'scripts/19_evaluate_35_features.py',
]


def main():
    for rel in FILES:
        path = os.path.join(REPO_ROOT, rel)
        print(f'Compiling {rel} ...')
        py_compile.compile(path, doraise=True)

    print('Importing calibration...')
    from tool.calibration import calibrate_probability
    print('calibrate_probability(0.9, 10) =', calibrate_probability(0.9, 10))

    print('Importing adversarial defense...')
    from tool.adversarial_defense import normalize_text_defensive
    cleaned = normalize_text_defensive('Thіs іs а tеst\u200b!')
    assert cleaned == 'This is a test!', repr(cleaned)
    print('adversarial defense OK:', cleaned)

    print('Importing register classifier...')
    from tool.register_classifier import load_models_from_manifest, classify_register
    print('register classifier OK')

    print('Importing feature extractor...')
    from tool.feature_extractor import extract_features, ALL_FEATURE_COLS
    assert len(ALL_FEATURE_COLS) == 35, f"expected 35 features, got {len(ALL_FEATURE_COLS)}"
    print('feature extractor OK, feature count:', len(ALL_FEATURE_COLS))

    print('\nAll verification checks passed.')


if __name__ == '__main__':
    main()
