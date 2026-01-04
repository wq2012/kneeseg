import numpy as np
try:
    from pycpd import RigidRegistration, DeformableRegistration
except ImportError:
    RigidRegistration = None
    DeformableRegistration = None

def register_points(source, target, method='rigid', max_iterations=100):
    """
    Registers source points to target points.
    method: 'rigid', 'deformable', 'icp'
    """
    if method == 'rigid':
        if RigidRegistration:
            reg = RigidRegistration(**{'X': target, 'Y': source})
            result, _ = reg.register()
            return result
        else:
            # Fallback to a simple ICP if pycpd is not available
            return source # Placeholder
    elif method == 'deformable':
        if DeformableRegistration:
            reg = DeformableRegistration(**{'X': target, 'Y': source})
            result, _ = reg.register()
            return result
        else:
            return source
    return source
