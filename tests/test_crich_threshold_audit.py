import unittest
from audit_crich_thresholds import tail, clean_rows, profitable


class ThresholdAuditTest(unittest.TestCase):
    def test_strict_rule_tie_and_propensity(self):
        data = [dict(pid=1,score=.5,dj=-10,prop=1),
                dict(pid=2,score=.6,dj=.5,prop=.02),
                dict(pid=3,score=.7,dj=2,prop=1)]
        s = tail(data,.5)
        self.assertEqual(s['n'],2)
        self.assertEqual(s['tie'],1)
        self.assertEqual(s['mean_dj'],1.25)
        self.assertAlmostEqual(s['ipw_mean_dj'],27/51)

    def test_censoring_is_not_a_zero_reward(self):
        data = [dict(pair_resolved='False',problem_id=1)]
        self.assertEqual(clean_rows(data,{1}),[])

    def test_empty_and_low_support_are_not_success(self):
        self.assertFalse(profitable([tail([],0),tail([],0)]))
        self.assertFalse(profitable([dict(n=9,problems=3,mean_dj=-1,ipw_mean_dj=-1)]*2))
