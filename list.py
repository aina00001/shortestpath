class KeyValue:
    def __init__(self, key, value):
        self.key = key
        self.value = value

class Block:
    def __init__(self, max_size):
        self.pairs = []
        self.next = None
        self.max_size = max_size
        self.upper_bound = None  # Only used in D1

    def is_full(self):
        return len(self.pairs) >= self.max_size

    def add_pair(self, kv: KeyValue):
        self.pairs.append(kv)
        if self.upper_bound is None or kv.value > self.upper_bound:
            self.upper_bound = kv.value
        self.pairs.sort(key=lambda kv: kv.value)

class BlockSequence:
    def __init__(self, max_block_size):
        self.head = None
        self.max_block_size = max_block_size

    def prepend_batch(self, kv_list):
        """Used for D0: prepend a batch of key/value pairs"""
        new_block = Block(self.max_block_size)
        new_block.pairs = kv_list[:self.max_block_size]
        new_block.next = self.head
        self.head = new_block

    def insert_sorted(self, kv: KeyValue):
        """Used for D1: insert a single key/value pair in sorted order"""
        if self.head is None:
            self.head = Block(self.max_block_size)
            self.head.add_pair(kv)
            return
        current = self.head
        prev = None

        while current:
            if kv.value <= current.upper_bound or current.next is None:
                if not current.is_full():
                    current.add_pair(kv)
                    return
                else:
                    # Split block if full
                    new_block = Block(self.max_block_size)
                    new_block.add_pair(kv)
                    new_block.next = current.next
                    current.next = new_block
                    return
            prev = current
            current = current.next
class lemme33 : 
    def __init__(self, M : int, B : int):
        
        self.D0 = BlockSequence(max_block_size=M)
        self.D1 = BlockSequence(max_block_size=M)