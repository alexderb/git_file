#from netCDF4 import Dataset
import numpy as np
import glob
import xarray as xr
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


class data_factory:
    def __init__(self, input_param):
        self.path = input_param['path']
        self.lseq = input_param['linput'] + input_param['loutput']  # length of every sequences
        self.linput = input_param['linput']
        self.loutput = input_param['loutput']
        self.data_key = input_param['key']
        self.raw_data = []
        self.dataset = []
        self.data = {}
        self.load()

    def load(self):  # load raw_data ie array of each images
        xr.merge([xr.open_dataset(f) for f in glob.glob(self.path + '/*.nc')])  # merge different files from the given path
        ds = xr.open_mfdataset(self.path + '/*.nc')  # load the file as dataset
        self.raw_data = np.array(ds.variables[self.data_key])
        self.dataset = [self.raw_data]

    def cutting(self, cutsize=64, mode='sidebyside', land=True):
        # initial sizes of images
        basesize_x = np.shape(self.raw_data)[2]
        basesize_y = np.shape(self.raw_data)[1]
        # classic mode is the side by side cutting
        if mode == 'sidebyside':
            n_xcut = basesize_x // cutsize  # number of horizontals cut maps
            n_ycut = basesize_y // cutsize  # number of verticals cut maps
            n_time = np.shape(self.raw_data)[0]
            dataset_cut = []
            for i in range(0, n_xcut):
                for j in range(0, n_ycut):
                    if land:
                        if not np.all(np.isnan(self.raw_data[:, j*cutsize:(j+1)*cutsize, i*cutsize:(i+1)*cutsize])):
                            #dataset_cut[(i+1)*j] = self.raw_data[:, j*cutsize:(j+1)*cutsize, i*cutsize:(i+1)*cutsize]
                            dataset_cut.append(self.raw_data[:, j*cutsize:(j+1)*cutsize, i*cutsize:(i+1)*cutsize])
                    else:
                        if not np.any(np.isnan(self.raw_data[:, j*cutsize:(j+1)*cutsize, i*cutsize:(i+1)*cutsize])):
                            dataset_cut.append(self.raw_data[:, j * cutsize:(j + 1) * cutsize, i * cutsize:(i + 1) * cutsize])
        else if mode == 'exp':

        self.dataset = np.array(dataset_cut)

    def make_dataset(self):
        if np.shape(self.dataset)[2] != np.shape(self.dataset)[3]:
            self.cutting(min(np.shape(self.dataset)[2], np.shape(self.dataset)[3]))

        n_cut = np.shape(self.dataset)[0]  # number of cut dataset
        n_seq = (np.shape(self.raw_data)[0] - self.lseq + 1)  # number of sequences by cut
        size_im = np.shape(self.dataset)[2]

        self.data['clips'] = np.zeros((2, n_seq*n_cut, 2))
        self.data['dims'] = np.array((1, size_im, size_im))  # gives dimensions of data
        self.data['input_raw_data'] = np.zeros((self.lseq*n_seq*n_cut, 1, size_im, size_im))
        for k in range(n_cut):
            # 'clips'
            # [1,i-1,0] gives the length of the i^th input sequence
            # [1,i-1,1] gives the length of the i^th output sequence

            self.data['clips'][1, :, 0] = self.linput
            self.data['clips'][1, :, 1] = self.loutput

            # [0,i-1,0] gives the index of the beginning of input data for the i^th sequence
            # [0,i-1,1] gives the index of the beginning of output data for the i^th sequence
            for i in range(0, n_seq):

                self.data['clips'][0, i+k*n_seq, 0] = i * self.lseq + k*n_seq*self.lseq
                self.data['clips'][0, i+k*n_seq, 1] = self.data['clips'][0, i, 0] + self.linput + k*n_seq*self.lseq
                self.data['input_raw_data'][k*n_seq*self.lseq+i*self.lseq : k*n_seq*self.lseq+(i+1)*self.lseq,0] = self.dataset[k, i : i+self.lseq]

    def print(self,a=0):
        print(self.data['input_raw_data'][a])
    def imshow(self,a=0,file_name='test.png'):
        plt.imshow(self.data['input_raw_data'][a,0])
        plt.savefig(file_name)
    def get_size(self):
        print(np.shape(self.data['input_raw_data']))
    def plot(self,image=0):
        plt.plot()
    def save_dataset(self, name, repertory):
        fichier = open(repertory + '/'+ name +'.npz','w')
        np.savez(name, clips=self.data['clips'],clips=self.data['dims'], input_raw_data=self.data['input_raw_data'])
        fichier.close()
