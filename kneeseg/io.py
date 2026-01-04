import os
import numpy as np
import SimpleITK as sitk

class OAIImageReader:
    @staticmethod
    def read(hdr_path):
        """
        Reads an OAI image (.hdr and .img).
        Returns a tuple of (numpy array, metadata dict).
        """
        prefix = os.path.splitext(hdr_path)[0]
        img_path = prefix + '.img'
        
        if not os.path.exists(hdr_path) or not os.path.exists(img_path):
            raise FileNotFoundError(f"Could not find .hdr or .img for {prefix}")

        metadata = {}
        with open(hdr_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            
        # OAI header format as seen in load_hdr_header.m:
        # Line 0: dimX dimY dimZ
        # Line 1: resX resY resZ
        # Line 2: window center/width (sometimes)
        # Line 3: bitNum (or other meta)
        # Lines after: Affine Matrix M (3x4)
        
        metadata['dims'] = list(map(int, lines[0].split()))
        metadata['spacing'] = list(map(float, lines[1].split()))
        
        # Searching for bit depth - usually later in the file
        bit_num = 16 # Default
        for line in lines:
            if line.isdigit() and int(line) in [8, 16, 32]:
                bit_num = int(line)
        metadata['bit_num'] = bit_num

        # Read binary data
        dtype = np.int16 if bit_num == 16 else (np.uint8 if bit_num == 8 else np.float32)
        with open(img_path, 'rb') as f:
            data = np.fromfile(f, dtype=dtype)
            
        # Reshape according to dims (OAI usually uses Fortran order or needs transpose)
        # Based on load_HDR_vol.m: img = reshape(img, dim) which is column-major in Matlab
        data = data.reshape(metadata['dims'], order='F')
        
        return data, metadata

    @staticmethod
    def write(data, metadata, output_prefix):
        """
        Writes data to .hdr and .img format.
        """
        hdr_path = output_prefix + '.hdr'
        img_path = output_prefix + '.img'
        
        dims = metadata.get('dims', data.shape)
        spacing = metadata.get('spacing', [1.0, 1.0, 1.0])
        bit_num = metadata.get('bit_num', 16)
        
        with open(hdr_path, 'w') as f:
            f.write(f"{dims[0]} {dims[1]} {dims[2]}\n")
            f.write(f"{spacing[0]} {spacing[1]} {spacing[2]}\n")
            f.write("-1024.000000 1.000000\n0\n") # Placeholder window/meta
            f.write("1.0 0.0 0.0 0.0\n0.0 1.0 0.0 0.0\n0.0 0.0 1.0 0.0\n") # Identity affine
            f.write(f"{bit_num}\n")
            f.write("0 2\n")

        dtype = np.int16 if bit_num == 16 else (np.uint8 if bit_num == 8 else np.float32)
        with open(img_path, 'wb') as f:
            data.astype(dtype, order='F').tofile(f)

def load_volume(path, return_spacing=False):
    """
    Generic loader that handles both OAI (.hdr) and SKI10 (.mhd).
    """
    if path.lower().endswith('.hdr'):
        data, meta = OAIImageReader.read(path)
        if return_spacing:
            # OAIReader returns spacing in x,y,z order (from file)
            # data is z,y,x
            spacing_xyz = meta.get('spacing', [1.0, 1.0, 1.0])
            return data, tuple(spacing_xyz[::-1])
        return data
    else:
        # SimpleITK handles .mhd, .nii, etc.
        img = sitk.ReadImage(path)
        data = sitk.GetArrayFromImage(img) # Returns (z, y, x)
        if return_spacing:
             # SITK spacing is (x, y, z)
             spacing_xyz = img.GetSpacing()
             return data, tuple(reversed(spacing_xyz)) # (z, y, x)
        return data

def save_volume(data, path, metadata=None):
    """
    Generic saver.
    """
    if path.lower().endswith('.hdr'):
        if metadata is None:
            metadata = {'dims': data.shape, 'spacing': [1.0, 1.0, 1.0], 'bit_num': 16}
        OAIImageReader.write(data, metadata, os.path.splitext(path)[0])
    else:
        img = sitk.GetImageFromArray(data)
        if metadata and 'spacing' in metadata:
            img.SetSpacing(metadata['spacing'][::-1])
        sitk.WriteImage(img, path)
