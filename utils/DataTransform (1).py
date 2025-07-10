import numpy as np
from scipy.ndimage import shift

class MnistP4andP4M:
    def __init__(self, data=None, label=None):
        """
        Initialize with MNIST data.
        Args:
            data: numpy array of shape (batch, height, width) or (batch, channel, height, width)
        """
        if data is None:
            raise ValueError("Data cannot be None")
        
        # Ensure data is in (batch, channel, height, width) format
        if data.ndim == 3:
            self.data = data.reshape(-1, 1, 28, 28)
        elif data.ndim == 4:
            self.data = data.reshape(-1, 1, 28, 28)
        else:
            raise ValueError("Data must be 3D or 4D array")
        
        # D4 group transformations (dihedral group of order 8)
        self.label = label
        self.D4 = np.array([
            # Rotations (det = 1)
            [[1., 0], [0, 1.]],    # 0°
            [[0, -1], [1, 0]],     # 90°
            [[-1, 0], [0, -1]],    # 180°
            [[0, 1], [-1, 0]],     # 270°
            
            # Reflections (det = -1)
            [[-1, 0], [0, 1]],     # Horizontal flip
            [[0, -1], [-1, 0]],    # Diagonal flip
            [[1, 0], [0, -1]],     # Vertical flip
            [[0, 1], [1, 0]]       # Anti-diagonal flip
        ], dtype=float)
    
    def dihedral_transform(self, x=None):
        """Apply random dihedral group transformations."""
        if x is None:
            x = self.data
        
        batch_size = x.shape[0]
        g = np.random.randint(0, 8, size=batch_size)
        
        h, w = x.shape[-2:]
        center_h, center_w = (h - 1) / 2., (w - 1) / 2.
        
        # Create coordinate grid
        i_coords, j_coords = np.meshgrid(
            np.arange(h) - center_h, 
            np.arange(w) - center_w, 
            indexing='ij'
        )
        coords = np.stack([i_coords.ravel(), j_coords.ravel()])
        
        x_out = np.empty_like(x)
        
        for batch_idx in range(batch_size):
            # Apply transformation
            transform_matrix = self.D4[g[batch_idx]]
            transformed_coords = transform_matrix @ coords
            
            # Shift back to image coordinates
            new_i = transformed_coords[0].reshape(h, w) + center_h
            new_j = transformed_coords[1].reshape(h, w) + center_w
            
            # Handle boundaries and interpolation
            new_i = np.clip(np.round(new_i).astype(int), 0, h-1)
            new_j = np.clip(np.round(new_j).astype(int), 0, w-1)
            
            # Apply transformation
            x_out[batch_idx, 0] = x[batch_idx, 0, new_i, new_j]
        
        return x_out
    
    def translate_mnist_batch(self, x=None, max_shift=3):
        """Apply random translation to batch of images."""
        if x is None:
            x = self.data
        
        batch_size = x.shape[0]
        # Random translations in range [-max_shift, max_shift]
        translations = (np.random.rand(batch_size, 2) - 0.5) * 2 * max_shift
        
        x_out = np.zeros_like(x)
        
        for i in range(batch_size):
            x_out[i, 0] = shift(
                x[i, 0], 
                shift=translations[i], 
                order=1, 
                mode='constant', 
                cval=0
            )
        
        return x_out
    
    def flip_transform_batch(self, x=None):
        """Apply random horizontal/vertical flips."""
        if x is None:
            x = self.data
        
        batch_size = x.shape[0]
        flip_type = np.random.randint(0, 3, size=batch_size)
        
        x_out = np.empty_like(x)
        
        for i in range(batch_size):
            if flip_type[i] == 0:
                # Horizontal flip
                x_out[i] = x[i, :, :, ::-1]
            elif flip_type[i] == 1:
                # Vertical flip
                x_out[i] = x[i, :, ::-1, :]
            else:
                # No flip
                x_out[i] = x[i]
        
        return x_out
    
    def P4(self, x=None):
        """Apply P4 group transformations (rotations + translations)."""
        if x is None:
            x = self.data
        
        # First apply dihedral (rotation) transformations
        rotated = self.dihedral_transform(x)
        # Then apply translations
        transformed = self.translate_mnist_batch(rotated)
        
        return transformed
    
    def P4M(self, x=None):
        """Apply P4M group transformations:
           - Apply only translation to samples with label 6 or 9.
           - Apply flip + P4 to all other samples.
        """
        if x is None:
            x = self.data
        
        # Get the actual data length instead of hardcoding
        data_length = len(x)
        
        # Split indices using boolean masks (more efficient)
        translate_only_mask = (self.label == 6) | (self.label == 9)
        translate_only_idx = np.where(translate_only_mask)[0]
        full_transform_mask = ~translate_only_mask
        full_transform_idx = np.where(full_transform_mask)[0]
        
        # Check if we have data to process
        if len(translate_only_idx) == 0 and len(full_transform_idx) == 0:
            return x.copy()
        
        # Prepare output array
        p4m_transformed = np.empty_like(x)
        
        # Apply transformations only if indices exist
        if len(full_transform_idx) > 0:
            # Apply flip to non-6/9 samples
            flipped_data = self.flip_transform_batch(x[full_transform_idx])
            # Apply P4 to the flipped data
            p4_data = self.P4(flipped_data)
            # Assign results back
            p4m_transformed[full_transform_idx] = p4_data
        
        if len(translate_only_idx) > 0:
            # Apply translation to 6/9 samples
            translated_data = self.translate_mnist_batch(x[translate_only_idx])
            # Assign results back
            p4m_transformed[translate_only_idx] = translated_data


        
    
        return p4m_transformed

    
    def get_original_data(self):
        """Return the original data."""
        return self.data
    
    def __len__(self):
        """Return batch size."""
        return self.data.shape[0]
    
    def __repr__(self):
        return f"MnistP4andP4M(batch_size={self.data.shape[0]}, shape={self.data.shape})"
