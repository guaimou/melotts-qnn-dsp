"""Generate QNN context binary from QAR model using SDK Python API."""
import sys, os
sys.path.insert(0, "/opt/2.29.0.241129/lib/python")

from qti.aisw.core.model_level_api.workflow.context_binary_generator import ContextBinaryGenerator

# Check what the API needs
import inspect
print("=== ContextBinaryGenerator.generate ===")
print(inspect.getsource(ContextBinaryGenerator.generate)[:2000])
print()
print("=== ContextBinaryGenerator.__init__ ===")
print(inspect.getsource(ContextBinaryGenerator.__init__)[:1000])
