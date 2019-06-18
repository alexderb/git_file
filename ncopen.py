from netCDF4 import Dataset
import numpy as np
import glob
import xarray as xr

class ncopen
    def __init__(self, input_param):
        self.path = input_param['path']
        self.lseq = input_param['linput'] + input_param['loutput'] # length of every sequences
        self.linput = input_param['linput']
        self.loutput = input_param['loutput']
        self.data_key = input_param['key']
        self.zshape = input_param['size zone']
        self.imsize = input_param['imsize'] # size of square image wanted
        self.land = input_param['land'] # use of images with lands(==True) or not(==False)
        self.raw_data = []
        self.dataset = []
        self.data = {}
        self.load()
        self.make_dataset()

    def load(self): # load raw_data ie array of each images
        ds = xr.merge([xr.open_dataset(f) for f in glob.glob(self.path +'/*.nc')]) # merge different files from the given path
        ds = xr.open_mfdataset(self.path +'/*.nc') # load the file as dataset
        self.raw_data = np.array(ds.variables[self.data_key])
        self.dataset = [self.raw_data]

    def cutting(self,cutsize=64,mode='sidebyside',is_land=True):
        basesize_x = np.shape(self.raw_data)[2]
        basesize_y = np.shape(self.raw_data)[1]
        # classic mode is the side by side cutting
        if mode=='sidebyside':
            n_xcut = basesize_x//cutsize # number of horizontals cut maps
            n_ycut = basesize_y//cutsize # number of verticals cut maps
            dataset_cut = np.zeros((n_xcut*n_ycut,np.shape(self.dataset)[0],cutsize,cutsize))
            for i in range(0,n_xcut):
                for j in range(0,n_ycut):
                    dataset_cut[(i*n_ycut+j)] = self.raw_data[:,(j-1)*cutsize,(i-1)*cutsize]


    def make_dataset(self):
        if np.shape(self.dataset)[2] != np.shape(self.dataset)[3]:
            self.cutting(min(np.shape(self.dataset)[2] , np.shape(self.dataset)[3]))
            
        n_cut = np.shape(self.dataset)[0] # nombre of cut dataset
        n_seq = (np.shape(self.raw_data)[0] - self.lseq + 1) # nombre de séquences par cut
        size_im = np.shape(self.dataset)[2]
        
        self.data['clips'] = np.zeros((2, n_seq , 2))
        self.data['dims'] = np.array((1, size_im ,size_im))  # gives dimensions of data
        self.data['input_raw_data'] = np.array((lseq*n_seq*n_cut,1,size_im,size_im))
        for k in range(n_cut):
            # 'clips'
            # [1,i-1,0] gives the length of the i^th input sequence
            # [1,i-1,1] gives the length of the i^th output sequence

            self.data['clips'][1,:,0] = self.linput
            self.data['clips'][1,:,1] = self.loutput

            # [0,i-1,0] gives the index of the beginning of input data for the i^th sequence
            # [0,i-1,1] gives the index of the beginning of output data for the i^th sequence
            for i in range(0,n_seq):
                self.data['clips'][0,i,0] = i*self.lseq - 1
                self.data['clips'][0,i,1] = self.data['clips'][0,i,0] + self.loutput
                self.data['input_raw_data'][i*self.lseq:(i+1)*self.lseq] = self.raw_data[i:i+self.lseq]
