import numpy as np
import random
import torch
"""
Logic to generate new states using .
"""
class StatesGenerator(object):
    """
    Helper class used to randomly generate batches of states given a set
    of problem conditions, which are provided via the `config` object.
    """

    def __init__(self, config):
        self.batch_size = config.batch_size
        self.min_num_items = config.min_num_items
        self.max_num_items = config.max_num_items
        self.min_item_size = config.min_item_size
        self.max_item_size = config.max_item_size

        self.min_bin_size=config.min_bin_size
        self.max_bin_size=config.max_bin_size
        self.total_bins=config.total_bins

        self.num_critical_items = config.number_of_critical_items
        self.num_critical_copies = config.number_of_copies
        self.ci_groups = []

    def generate_critical_items(self, items_seqs_batch, items_len_mask, items_seq_lens):
        '''
            Generate critical items and their replicas:
            - `items_seqs_batch`: batch of only normal items
            - `items_len_mask`: mask of normal items, list of 1 and 0
            -  `items_seq_lens`: indicates the length of the items in each batch
        '''
        batch_critical_items = []
        critical_copy_mask = []
        items_with_critical = items_seqs_batch.copy()
        batch_ci_groups = []
        for items_seq, len_mask, seq_len in zip(items_with_critical, items_len_mask, items_seq_lens):
            critical_items = [sample[0] for sample in random.sample(list(enumerate(items_seq[:seq_len])), k=self.num_critical_items)]
            batch_critical_items.append(critical_items)
            critical_mask = len_mask.copy()
            ci_groups = []
            for idx, ci in enumerate(critical_items):
                critical_mask[ci] = 2+idx # First Change the mask of the original critical items
            for idx, ci in enumerate(critical_items): # Create copies for the critical items and change their value and mask
                critical_item_copies = random.sample(list(np.where(critical_mask==1.)[0]), k=2)
                critical_mask[critical_item_copies] = 2 + idx
                items_seq[critical_item_copies] = items_seq[ci]
                ci_groups.append([ci]+critical_item_copies)
            critical_copy_mask.append(critical_mask)
            batch_ci_groups.append(ci_groups)
        return (items_with_critical, critical_copy_mask, batch_ci_groups)



    def generate_states_batch(self, batch_size=None):
        """Generate new batch of initial states"""
        if batch_size is None:
            batch_size = self.batch_size
        items_seqs_batch = np.random.randint(
            low=self.min_item_size,
            high=self.max_item_size + 1,
            size=(batch_size, self.max_num_items),
        )

        items_len_mask = np.ones_like(items_seqs_batch, dtype="float32")
        items_seq_lens = np.random.randint(
            low=self.min_num_items, high=self.max_num_items + 1, size=batch_size
        )

        bins_available=[]
        bin_choices=[bin_size for bin_size in range(self.min_bin_size,self.max_bin_size,100)]


        for i in range(self.total_bins):
            bins_available.append(random.choice(bin_choices))

        bins_available=np.array(bins_available)
        bins_available=np.repeat(bins_available[np.newaxis, ...], batch_size, axis=0)

        # bins_available = np.random.randint(
        #     low=self.min_bin_size, high=self.max_bin_size + 1,
        #     size=(batch_size,self.total_bins)
        # )

        for items_seq, len_mask, seq_len in zip(
            items_seqs_batch, items_len_mask, items_seq_lens
        ):
            items_seq[seq_len:] = 0
            len_mask[seq_len:] = 0

        return (
            items_seqs_batch,
            items_seq_lens,
            items_len_mask,
            bins_available
        )


