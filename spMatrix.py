import numpy as np

def get_one_step_adj(dim):
    def one_step_adj(x):
        x = np.array(x)
        I = np.eye(dim)
        ans = []
        for i in range(dim):
            ans.append(x+I[i])
            ans.append(x-I[i])
        return np.array(ans)
    return one_step_adj

class spMatrix(dict):
    def __init__(self,arg,shift = None,adj = 'onestep',*args,**kwargs):
        self.update(*args,**kwargs)
        self.memory = {}
        self.none_zeros = {}
        self.mask = {}

        if np.array([arg]).shape == (1,):
            self.dim = int(arg)
        else:
            self.dim = len(np.array(arg).shape)
            if shift:
                shift = np.array(shift,dtype = np.int32)
            else:
                shift = np.array([0]*self.dim,dtype = np.int32)

            ndA = ndMatrix(np.array(arg))
            for i in range(ndA.len()):
                idx = ndA.k2ij([[i]])[0]
                self[idx+shift] = ndA.get([idx])[0,0]
        
        if adj == 'onestep':
            self.adj = get_one_step_adj(self.dim)

    def __getitem__(self,key):
        if isinstance(key,str):
            return dict.__getitem__(self,key)
        if np.array(key,dtype = np.int32).shape != (self.dim,):
            raise Exception('dimension error')
        true_key = list(np.array(key,dtype = np.int32)).__str__()
        self.memory[true_key] = np.array(key,dtype = np.int32)
        if true_key in self.keys():
            return dict.__getitem__(self,true_key)
        else:
            return 0
    
    def __setitem__(self,key,val):
        if np.array(key,dtype = np.int32).shape != (self.dim,):
            raise Exception('dimension error')
        true_key = list(np.array(key,dtype = np.int32)).__str__()
        self.memory[true_key] = np.array(key,dtype = np.int32)
        self.none_zeros[true_key] = np.array(key,dtype = np.int32)
        dict.__setitem__(self,true_key,val)
        if val == 0:
            dict.__delitem__(self,true_key)
            dict.__delitem__(self.none_zeros,true_key)
    
    def numpy(self,shape):
        ndA = ndMatrix(np.zeros(shape))
        self_where_all = self.where_all()
        for x in self_where_all:
            flag = 1
            for i,xx in enumerate(x):
                if xx<0 or xx>=shape[i]:
                    flag = 0
            if flag:
                ndA.set([x],[self[x]])
        return ndA.numpy()

    def where_all(self):
        return np.array(list(self.none_zeros.items()),dtype=object)[:,1]
    
    def copy(self):
        cp = spMatrix(self.dim)
        for x in self.where_all():
            cp[x] = self[x]

    def conv(self,filter):
        updated = spMatrix(self.dim)
        ans = spMatrix(self.dim)
        self_where_all = self.where_all()
        filter_where_all = filter.where_all()
        for x in self_where_all:
            for shift in filter_where_all:
                updated[x+shift] = updated[x+shift] + 1
        updated_where_all = updated.where_all()
        for x in updated_where_all:
            for shift in filter_where_all:
                ans[x] = ans[x] + filter[shift] * self[x+shift]
        return ans

    def get_mask(self):
        n = 0
        self.mask = spMatrix(self.dim)
        self.group = []
        self.group_max = []

        self_where_all = self.where_all()
        for x in self_where_all:
            if self.mask[x] == 0:
                mx = 0
                g = []
                n = n + 1
                queue = [x]
                self.mask[x] = n
                while(len(queue)>0):
                    xx = queue.pop(0)
                    mx = max(mx,self[xx])
                    g.append(xx)
                    Y = self.adj(xx)
                    for y in Y:
                        if self[y] != 0:
                            if self.mask[y] == 0:
                                queue.append(y)
                                self.mask[y] = n
                self.group.append(np.array(g))
                self.group_max.append(mx)
        self.labels_n = n
        return self.mask
