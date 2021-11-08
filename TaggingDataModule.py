# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 16:02:34 2021

@author: 15626
"""

class TaggingDataModule(LightningDataModule):
    def __init__(self, batch_size=4):
        super().__init__()
        self.batch_size = batch_size

    # When doing distributed training, Datamodules have two optional arguments for
    # granular control over download/prepare/splitting data:

    # OPTIONAL, called only on 1 GPU/machine
    def prepare_data(self):
        parser = argparse.ArgumentParser(description='Tagging/VAE')
        parser.add_argument('--dir_ids', type=str, default='./dataset/ukbb_roi.csv')
        parser.add_argument('--percentage', type=float, default=0.80)
        parser.add_argument('--percentage1', type=float, default=0.30)
        parser.add_argument('--batch_size', default=1, type=int)#8
        parser.add_argument('--tagging_img_size', type=list, default=[128, 128, 1])#15
        #parser.add_argument('--tagging_img_size', type=list, default=[192, 256, 1])
        parser.add_argument('--n_cpu', default=0, type=int)
        parser.add_argument('--dir_dataset', type=str, default='./dataset/')

        args = parser.parse_args()
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(device)

        print('\nLoading IDs file\n')
        IDs = pd.read_csv(args.dir_ids, sep=',')
        # Dividing the number of images for training and test.
        IDs_copy = IDs.copy()
        train_set = IDs_copy.sample(frac = args.percentage, random_state=0)
        test_set = IDs_copy.drop(train_set.index)
        val_set = train_set.sample(frac = args.percentage1, random_state=0)
        train_set = train_set.drop(val_set.index)
        print('train:', len(train_set), 'test:', len(test_set))

        print('train:', len(train_set), 'validation:', len(val_set))

        #MNIST(os.getcwd(), train=True, download=True)
        #MNIST(os.getcwd(), train=False, download=True)

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage: Optional[str] = None):
        # transforms
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        # split dataset
        if stage in (None, "fit"):
            mnist_train = MNIST(os.getcwd(), train=True, transform=transform)
            self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])
        if stage == "test":
            self.mnist_test = MNIST(os.getcwd(), train=False, transform=transform)
        if stage == "predict":
            self.mnist_predict = MNIST(os.getcwd(), train=False, transform=transform)

    # return the dataloader for each split
    def train_dataloader(self):
        train_loader = Tagging_loader(batch_size = self.batch_size,
                         #sax_img_size= args.sax_img_size,
                         num_workers = args.n_cpu,
                         tagging_img_size = args.tagging_img_size,
            			 shuffle = True,
            			 dir_imgs = args.dir_dataset,
                         args = args,
                         ids_set = train_set
            			  )
        #mnist_train = DataLoader(self.mnist_train, batch_size=self.batch_size)
        return train_loader

    def val_dataloader(self):
        val_loader = Tagging_loader(batch_size = self.batch_size,
                         #sax_img_size= args.sax_img_size,
                         num_workers = args.n_cpu,
                         tagging_img_size = args.tagging_img_size,
            			 shuffle = True,
            			 dir_imgs = args.dir_dataset,
                         args = args,
                         ids_set = val_set
            			  )
        #mnist_val = DataLoader(self.mnist_val, batch_size=self.batch_size)
        return val_loader

    def test_dataloader(self):
        test_loader = Tagging_loader(batch_size = self.batch_size,
                            tagging_img_size= args.tagging_img_size,
                        	num_workers = args.n_cpu,
                            #sax_img_size = args.sax_img_size,
            			    shuffle = False,
            			    dir_imgs = args.dir_dataset,
                            args = args,
                            ids_set = test_set
            			     )
        #mnist_test = DataLoader(self.mnist_test, batch_size=self.batch_size)
        return test_loader
