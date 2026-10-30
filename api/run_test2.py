import sys
sys.path.insert(0, '/app')
import pytest
result = pytest.main(['tests/test_depth_scale_routes.py::test_bev_route_relative_keys_dimensions_and_camera_access', '-v', '--tb=long'])
with open('/tmp/test_result.txt', 'w') as f:
    f.write(f'EXIT: {result}')
print(f'EXIT: {result}', flush=True)