import sys
sys.path.insert(0, '/app')
import pytest
result = pytest.main(['tests/test_ground_model_service.py', '-v', '--tb=short'])
print('EXIT:', result, flush=True)