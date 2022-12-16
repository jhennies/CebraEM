
import subprocess
# import sys
# import os


def cuda_detect():
    '''Attempt to detect the version of CUDA present in the operating system.

    On Windows and Linux, the CUDA library is installed by the NVIDIA
    driver package, and is typically found in the standard library path,
    rather than with the CUDA SDK (which is optional for running CUDA apps).

    On macOS, the CUDA library is only installed with the CUDA SDK, and
    might not be in the library path.

    Returns: version string (Ex: '9.2') or None if CUDA not found.
    '''
    # platform specific libcuda location
    import platform
    system = platform.system()
    if system == 'Darwin':
        lib_filenames = [
            'libcuda.dylib',  # check library path first
            '/usr/local/cuda/lib/libcuda.dylib'
        ]
    elif system == 'Linux':
        lib_filenames = [
            'libcuda.so',  # check library path first
            '/usr/lib64/nvidia/libcuda.so',  # Redhat/CentOS/Fedora
            '/usr/lib/x86_64-linux-gnu/libcuda.so',  # Ubuntu
        ]
    elif system == 'Windows':
        lib_filenames = ['nvcuda.dll']
    else:
        return None  # CUDA not available for other operating systems

    # open library
    import ctypes
    if system == 'Windows':
        dll = ctypes.windll
    else:
        dll = ctypes.cdll
    libcuda = None
    for lib_filename in lib_filenames:
        try:
            libcuda = dll.LoadLibrary(lib_filename)
            break
        except:
            pass
    if libcuda is None:
        return None

    # Get CUDA version
    try:
        cuInit = libcuda.cuInit
        flags = ctypes.c_uint(0)
        ret = cuInit(flags)
        if ret != 0:
            return None

        cuDriverGetVersion = libcuda.cuDriverGetVersion
        version_int = ctypes.c_int(0)
        ret = cuDriverGetVersion(ctypes.byref(version_int))
        if ret != 0:
            return None

        # Convert version integer to version string
        value = version_int.value
        return '%d.%d' % (value // 1000, (value % 1000) // 10)
    except:
        return None


def custom_pytorch():
    print('Executing installation script')
    print('Detecting CUDA version')
    CUDA_VERSION = cuda_detect()
    CUDA_NUMS = CUDA_VERSION.strip().split('.')
    print(f'Detected:{CUDA_VERSION}. Installing pytorch.')

    torch_version = None
    torchvision_version = None
    torchaudio_version = None

    if CUDA_VERSION is None:
        torch_version = '1.7.1+cpu'
        torchvision_version = '0.8.2+cpu'
        torchaudio_version = '0.7.2'
    elif int(CUDA_NUMS[0]) < 10:
        print('ImBooster requires CUDA 10.1 minimum. Falling back to CPU.')
        torch_version = '1.7.1+cpu'
        torchvision_version = '0.8.2+cpu'
        torchaudio_version = '0.7.2'
    elif CUDA_VERSION == '10.1':
        torch_version = '1.7.1+cu101'
        torchvision_version = '0.8.2+cu101'
        torchaudio_version = '0.7.2'
    elif CUDA_VERSION == '10.2':
        torch_version = '1.7.1'
        torchvision_version = '0.8.2'
        torchaudio_version = '0.7.2'
    elif int(CUDA_NUMS[0]) >= 11:
        torch_version = '1.7.1+cu110'
        torchvision_version = '0.8.2+cu110'
        torchaudio_version = '0.7.2'

    assert torch_version is not None and torchvision_version is not None and torchaudio_version is not None

    subprocess.call(
        'pip install '
        f'torch=={torch_version} '
        f'torchvision=={torchvision_version} '
        f'torchaudio=={torchaudio_version} ' if torchaudio_version is not None else ''
        '-f https://download.pytorch.org/whl/torch_stable.html',
        shell=True, stdout=None, stderr=None
    )

    # # extra_inst.append(torch_inst)
    # extra_inst.append('tensorboard==2.2.2')
    # extra_inst.append('torchsummary==1.5.1')
    # extra_inst.append('fastai==1.0.61')
    #
    # for package in extra_inst:
    #     print(f'Installing :{package}')
    #     subprocess.check_output([sys.executable, "-m", "pip", "install", package])
    #     # subprocess.check_output(["pip", "install", package])