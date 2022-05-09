from io import BytesIO
import lmdb
from PIL import Image
from torch.utils.data import Dataset, Subset
import numpy as np
from torchvision import transforms
from torch.utils import data


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256, labels=False, filter_label=None):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        # CheXpert
        self.PRED_LABEL = [
            "No Finding",
            "Enlarged Cardiomediastinum",
            "Cardiomegaly",
            "Lung Opacity",
            "Lung Lesion",
            "Edema",
            "Consolidation",
            "Pneumonia",
            "Atelectasis",
            "Pneumothorax",
            "Pleural Effusion",
            "Pleural Other",
            "Fracture",
            "Support Devices"]

        if filter_label:
            if filter_label not in self.PRED_LABEL:
                raise Exception("Unrecognized label")
            self.filter_label = self.PRED_LABEL.index(filter_label)
        else:
            self.filter_label = None

        self.resolution = resolution
        self.transform = transform
        self.labels = labels
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)
            if self.labels:
                label_key = f"{self.resolution}-{str(index).zfill(5)}-label".encode("utf-8")
                label_bytes = txn.get(label_key)
                # we need labels to be type "ling"
                label = np.frombuffer(label_bytes, dtype=np.uint8).copy().astype(np.float32)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)

        if self.transform is not None:
            img = self.transform(img)

        if self.filter_label is not None:
            label = label[self.filter_label]
            label = np.array([0, 1]) if label == 1.0 else np.array([1, 0])
        
        if self.labels:
            return img, label
        return img

    def __getlabels__(self):
        with self.env.begin(write=False) as txn:
            labels = []
            for index in range(self.length):
                label_key = f"{self.resolution}-{str(index).zfill(5)}-label".encode("utf-8")
                label_bytes = txn.get(label_key)
                # we need labels to be type "long"
                label = np.frombuffer(label_bytes, dtype=np.uint8).copy().astype(np.float32)
                labels.append(label)
        return labels


def label_vs_no_finding_dataset(args, transform, return_labels):
    complete_dataset = MultiResolutionDataset(args.dataset_path, transform, args.size, labels=True)

    # get the labels list of all the samples
    targets = np.asarray(complete_dataset.__getlabels__())

    no_finding_label_idx = complete_dataset.PRED_LABEL.index("No Finding")
    filter_label_idx = complete_dataset.PRED_LABEL.index(args.filter_label)

    # get the index of no_finding samples and also filter label respectively
    no_finding_sample_index = np.where(targets[:, no_finding_label_idx] == 1)[0]
    filter_sample_index = np.where(targets[:, filter_label_idx] == 1)[0]

    min_length = min(len(no_finding_sample_index), len(filter_sample_index))
    no_finding_sample_index = no_finding_sample_index[:min_length]
    filter_sample_index = filter_sample_index[:min_length]

    combine_index = np.union1d(no_finding_sample_index, filter_sample_index)

    # the actual returned dataset may or may not require labels
    dataset = MultiResolutionDataset(args.dataset_path, transform, args.size,
                                     labels=return_labels, filter_label=args.filter_label)

    filter_dataset = Subset(dataset, combine_index)

    print("Healthy (No Finding) samples number is: " + str(len(no_finding_sample_index)))
    print(str(args.filter_label) + " samples number is: " + str(len(filter_sample_index)))
    print("Total samples number is: " + str(len(combine_index)))

    return filter_dataset


def get_dataset(args, return_labels=True):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True),
        ]
    )
    # filter dataset for specific label
    if 'filter_label' in args and args.filter_label and not args.compare_to_healthy:
        dataset = MultiResolutionDataset(args.dataset_path, transform, args.size,
                                         labels=return_labels,
                                         filter_label=args.filter_label)

    # compare to healthy label
    elif 'compare_to_healthy' in args and args.compare_to_healthy:
        dataset = label_vs_no_finding_dataset(args, transform, return_labels)

    # dataset already filtered
    else:
        dataset = MultiResolutionDataset(args.dataset_path, transform, args.batch, labels=return_labels)

    return dataset


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def get_dataloader(args):
    dataset = get_dataset(args)
    dataloader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True
    )

    return dataloader






