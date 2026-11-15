import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'PieBridge/src'))
from prefetch_policy import feature_group


class PrefetchPolicyTest(unittest.TestCase):
    def choose(self,proxies,buffer=None,budget=100,reorder=True,capacity=32,sizes=(1,2,4),costs=None):
        jobs = [{'proxy':p} for p in proxies]
        costs = costs or {(p,g):(50 if p==0 else 20)*g for p in set(proxies) for g in sizes}
        return feature_group(jobs,0,buffer or {},capacity,128,sizes,costs,budget,reorder)

    def test_fifo_does_not_merge_across_another_proxy(self):
        self.assertEqual(self.choose([0,1,0,0],reorder=False),[0])
        self.assertEqual(self.choose([0,1,0,0],reorder=True),[0,2])

    def test_short_window_can_prepare_later_proxy(self):
        self.assertEqual(self.choose([0,1,1],budget=40),[1,2])
        self.assertEqual(self.choose([0,1,1],budget=40,reorder=False),[])

    def test_buffer_reserves_space_for_missing_oldest_batch(self):
        self.assertEqual(self.choose([0,1,1],budget=40,capacity=2),[1])
        self.assertEqual(self.choose([0,1,1],buffer={1:None},budget=40,capacity=2),[])
        self.assertEqual(self.choose([0,1,1],buffer={1:None},budget=60,capacity=2),[0])

    def test_full_epoch_is_consumed_once_and_memory_is_bounded(self):
        proxies = [0,1,1,0,1,1,1,0]*95
        jobs = [{'proxy':p} for p in proxies]
        costs = {(p,g):(50 if p==0 else 20)*g for p in set(proxies) for g in [1,2,4]}
        cursor,buffer,produced,consumed = 0,{},[],[]
        attempts = 0
        while cursor < len(jobs):
            attempts += 1
            self.assertLess(attempts,10000)
            if cursor in buffer:
                buffer.pop(cursor)
                consumed.append(cursor)
                cursor += 1
                continue
            budget = 40 if attempts%3 else 100
            ids = feature_group(jobs,cursor,buffer,5,128,[1,2,4],costs,budget,True)
            for i in ids:
                self.assertNotIn(i,produced)
                buffer[i] = True
                produced.append(i)
            self.assertLessEqual(len(buffer),5)
        self.assertEqual(consumed,list(range(len(jobs))))
        self.assertEqual(sorted(produced),consumed)
        self.assertFalse(buffer)


if __name__ == '__main__':
    unittest.main()
