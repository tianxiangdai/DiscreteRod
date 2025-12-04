import numpy as np
from array import array
from scipy.sparse import csc_array, csr_array, coo_array
from cardillo.utility.coo_matrix import CooMatrix

class BlockCooMatrix(CooMatrix):
    def __init__(self, shape):
        super().__init__(shape)
        self.nallocation = 0
        self._data_index = {}
        self._size_fixed = False
        self._coo = None
        
    def fix_size(self):
        self._size_fixed = True
        self._CooMatrix__data = np.empty(len(self.row), dtype=np.float64)
        self._coo = super().tocoo(copy=False)
        
        

    def _allocate(self, rows, cols):
        allocation_id = self.nallocation
        self.nallocation = allocation_id + 1
        idx1 = len(self.row)
        self.row.extend(rows)
        self.col.extend(cols)
        idx2 = len(self.row)
        self._data_index[allocation_id] = (idx1, idx2)
        return allocation_id      
    
    def allocate(self, rows, cols, target=None):
        if self._size_fixed:
            raise Exception("size fixed!")
        
        if target is None or isinstance(target, np.ndarray):
            _rows = np.repeat(rows, len(cols))
            _cols = np.tile(cols, len(rows))
        elif isinstance(target, CooMatrix):
            _rows = rows[target.row]
            _cols = cols[target.col]
        else:
            raise            
        return self._allocate(_rows, _cols)
        
    def set_value(self, allocation_id, value):
        idx1, idx2 = self._data_index[allocation_id]
        if isinstance(value, CooMatrix):
            self.data[idx1:idx2] = value.data
        elif isinstance(value, np.ndarray):
            self.data[idx1:idx2] = value.reshape(-1)
        else:
            raise
    
    def asformat(self, format, copy=False):
        if self._coo is not None:
            return self._coo.asformat(format, copy=copy)
        else:
            return super().asformat(format, copy=copy)
          
    def __repr__(self):
        print("nrow:", len(self.row))
        print("ncol:", len(self.col))
        print("ndata:", len(self.data))
        return super().__repr__()  
    
    