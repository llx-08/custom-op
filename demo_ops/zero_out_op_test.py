import tensorflow as tf


class ZeroOutTest(tf.test.TestCase):
    def testZeroOut(self):
        with self.test_session():
            result = tf.user_ops.zero_out([5, 4, 3, 2, 1])
            self.assertAllEqual(result.eval(), [5, 0, 0, 0, 0])
