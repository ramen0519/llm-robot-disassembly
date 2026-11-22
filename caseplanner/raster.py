"""Optional compiled version of the same CPU z-buffer for high-triangle-count video."""
import ctypes
from functools import lru_cache
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
from .io import sha

@lru_cache(None)
def backend():
    source=Path(__file__).with_suffix('.c')
    if os.name!='posix' or not shutil.which('cc'):
        return None
    cache=source.parents[1]/'.runtime'
    cache.mkdir(exist_ok=True)
    output=cache/('raster-'+sha(source)[:16]+'.so')
    if not output.exists():
        temp=output.with_suffix('.%d.so'%os.getpid())
        try:
            subprocess.run(['cc','-O2','-fPIC','-shared',str(source),'-lm','-o',str(temp)],check=True,capture_output=True)
            os.replace(temp,output)
        except (OSError,subprocess.SubprocessError):
            return None
    method=ctypes.CDLL(str(output)).paint
    method.argtypes=[ctypes.c_int,ctypes.c_int]+[ctypes.c_void_p]*6
    method.restype=None
    return method


def draw(method,coord,points,faces,color,depth,image):
    arrays=[np.ascontiguousarray(coord,np.float64),np.ascontiguousarray(points,np.float64),
            np.ascontiguousarray(faces,np.int32),np.ascontiguousarray(color,np.float64),depth,image]
    method(image.shape[0],len(faces),*[a.ctypes.data for a in arrays])
