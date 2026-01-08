# Local shim for editor to resolve import during static analysis
# This is a minimal stand-in for the real `flask_cors` package used at runtime.

def CORS(app=None, **kwargs):
    return None
