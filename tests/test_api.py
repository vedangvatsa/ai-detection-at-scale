#!/usr/bin/env python3
"""Unit tests for the FastAPI endpoints with mocked models."""
import sys
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# FastAPI is optional; skip tests if not installed.
try:
    from fastapi.testclient import TestClient
    from tool import api as api_module
    from tool import api_security
    FASTAPI_AVAILABLE = True
except Exception as e:
    FASTAPI_AVAILABLE = False
    print(f'Skipping API tests; fastapi/dependencies not available: {e}')


class _FakeModel:
    """Minimal scikit-learn-like model for mocking."""
    classes_ = np.array([0, 1])

    def predict(self, X):
        return np.array(['academic'])

    def predict_proba(self, X):
        return np.array([[0.3, 0.7]])


class _FakeRegisterClassifier:
    classes_ = np.array(['academic'])

    def predict(self, X):
        return np.array(['academic'])

    def predict_proba(self, X):
        return np.array([[1.0]])


@unittest.skipUnless(FASTAPI_AVAILABLE, 'fastapi not installed')
class TestAPI(unittest.TestCase):
    def setUp(self):
        self.fake_models = {
            'detectors': {'academic': _FakeModel()},
            'all_detector': _FakeModel(),
            'register_classifier': _FakeRegisterClassifier(),
            'feature_cols': [
                'mtld', 'sent_cv', 'self_mention_density', 'opener_ratio',
                'connector_density', 'hedge_density', 'mean_sent_len',
                'boost_density', 'char_entropy', 'rep_rate', 'punct_entropy',
            ],
            'registers': ['academic'],
            'manifest': {},
            'hybrid_available': False,
        }
        self.client = TestClient(api_module.app)

    @patch.object(api_module, 'get_models', return_value={})
    def test_health_without_models(self, _):
        resp = self.client.get('/health')
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data['status'], 'ok')

    @patch.object(api_module, 'get_models')
    def test_detect_no_api_key_required_when_unset(self, mock_get):
        mock_get.return_value = self.fake_models
        api_security.API_KEY = ''  # disabled
        resp = self.client.post('/detect', json={
            'text': 'Furthermore, the results clearly demonstrate that this approach is effective.',
            'return_features': False,
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn('ai_probability', data)
        self.assertGreaterEqual(data['ai_probability'], 0.0)
        self.assertLessEqual(data['ai_probability'], 1.0)

    @patch.object(api_module, 'get_models')
    def test_detect_requires_api_key_when_set(self, mock_get):
        mock_get.return_value = self.fake_models
        api_security.API_KEY = 'secret-key'
        resp = self.client.post('/detect', json={
            'text': 'Furthermore, the results clearly demonstrate that this approach is effective.',
        })
        self.assertEqual(resp.status_code, 401)

    @patch.object(api_module, 'get_models')
    def test_detect_batch(self, mock_get):
        mock_get.return_value = self.fake_models
        api_security.API_KEY = ''
        resp = self.client.post('/detect/batch', json={
            'texts': [
                'Furthermore, the results clearly demonstrate that this approach is effective.',
                'The quick brown fox jumps over the lazy dog.',
            ],
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data['results']), 2)


if __name__ == '__main__':
    unittest.main()
