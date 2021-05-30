import numpy as np
import jax.numpy as jnp

from torch import data

from sentence_transformers import SentenceTransformer


sentence_model = SentenceTransformer('paraphrase-distilroberta-base-v1').cuda()


def one_hot(x, k, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)


class SentenceDataset(data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]['data']

        return data['text'], class_map[data['intent']]


def numpy_collate(batch):
    sentences = [i[0] for i in batch]
    labels = np.array([i[1] for i in batch])

    sentence_embeddings = sentence_model.encode(sentences)

    return sentence_embeddings, labels


class SentenceLoader(data.DataLoader):
    def __init__(self, dataset, batch_size=1,
                 shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0,
                 pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )
