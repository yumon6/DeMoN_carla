from .blocks import *

class BootstrapNet:
    def __init__(self, session):
        """Creates the bootstrap network
        session: tf.Session
            Tensorflow session
        """
        self.session = session
        self.placeholder_image_pair = tf.placeholder(dtype=tf.float32, shape=(1,6,192,256))
        self.placeholder_image2_2 = tf.placeholder(dtype=tf.float32, shape=(1,3,48,64))


        with tf.variable_scope('netFlow1'):
            netFlow1_result = flow_block(self.placeholder_image_pair )
            self.netFlow1_result = netFlow1_result
            self.predict_flow5, self.predict_conf5 = tf.split(value=netFlow1_result['predict_flowconf5'], num_or_size_splits=2, axis=1)
            self.predict_flow2, self.predict_conf2 = tf.split(value=netFlow1_result['predict_flowconf2'], num_or_size_splits=2, axis=1)

        with tf.variable_scope('netDM1'):
            self.netDM1_result = depthmotion_block(
                    image_pair=self.placeholder_image_pair, 
                    image2_2=self.placeholder_image2_2, 
                    prev_flow2=self.predict_flow2, 
                    prev_flowconf2=self.netFlow1_result['predict_flowconf2'], 
                    )


    def eval(self, image_pair, image2_2):
        """Runs the bootstrap network
        
        image_pair: numpy.ndarray
            Array with shape [1,6,192,256] if data_format=='channels_first'
            
            Image pair in the range [-0.5, 0.5]
        image2_2: numpy.ndarray
            Second image at resolution level 2 (downsampled two times)
            The shape for data_format=='channels_first' is [1,3,48,64]
        Returns a dict with the preditions of the bootstrap net
        """
        fetches = {
                'predict_flow5': self.predict_flow5,
                'predict_flow2': self.predict_flow2,
                'predict_depth2': self.netDM1_result['predict_depth2'],
                'predict_normal2': self.netDM1_result['predict_normal2'],
                'predict_rotation': self.netDM1_result['predict_rotation'],
                'predict_translation': self.netDM1_result['predict_translation'],
                }
        feed_dict = {
                self.placeholder_image_pair: image_pair,
                self.placeholder_image2_2: image2_2,
                }
        return self.session.run(fetches, feed_dict=feed_dict)




class IterativeNet:
    def __init__(self, session):
        """Creates the bootstrap network
        session: tf.Session
            Tensorflow session
        """
        self.session = session

        self.intrinsics = tf.constant([[0.89115971, 1.18821287, 0.5, 0.5]], dtype=tf.float32)

        self.placeholder_image_pair = tf.placeholder(dtype=tf.float32, shape=(1,6,192,256))
        self.placeholder_image2_2 = tf.placeholder(dtype=tf.float32, shape=(1,3,48,64))

        self.placeholder_depth2 = tf.placeholder(dtype=tf.float32, shape=(1,1,48,64))
        self.placeholder_normal2 = tf.placeholder(dtype=tf.float32, shape=(1,3,48,64))
        self.placeholder_rotation = tf.placeholder(dtype=tf.float32, shape=(1,3))
        self.placeholder_translation = tf.placeholder(dtype=tf.float32, shape=(1,3))

        with tf.variable_scope('netFlow2'):
            netFlow2_result = flow_block(
                image_pair=self.placeholder_image_pair,
                image2_2=self.placeholder_image2_2,
                intrinsics=self.intrinsics,
                prev_predictions={
                    'predict_depth2': self.placeholder_depth2,
                    'predict_normal2': self.placeholder_normal2,
                    'predict_rotation': self.placeholder_rotation,
                    'predict_translation': self.placeholder_translation,
                    },
                )
            self.netFlow2_result = netFlow2_result
            self.predict_flow5, self.predict_conf5 = tf.split(value=netFlow2_result['predict_flowconf5'], num_or_size_splits=2, axis=1)
            self.predict_flow2, self.predict_conf2 = tf.split(value=netFlow2_result['predict_flowconf2'], num_or_size_splits=2, axis=1)

        with tf.variable_scope('netDM2'):
            self.netDM2_result = depthmotion_block(
                    image_pair=self.placeholder_image_pair,
                    image2_2=self.placeholder_image2_2, 
                    prev_flow2=self.predict_flow2, 
                    prev_flowconf2=self.netFlow2_result['predict_flowconf2'], 
                    intrinsics=self.intrinsics,
                    prev_rotation=self.placeholder_rotation,
                    prev_translation=self.placeholder_translation,
                    )

    def eval(self, image_pair, image2_2, depth2, normal2, rotation, translation ):
        """Runs the iterative network
        
        image_pair: numpy.ndarray
            Array with shape [1,6,192,256]
            
            Image pair in the range [-0.5, 0.5]
        image2_2: numpy.ndarray
            Second image at resolution level 2 (downsampled two times)
            The shape is [1,3,48,64]
        depth2: numpy.ndarray
            Depth prediction at resolution level 2
        normal2: numpy.ndarray
            Normal prediction at resolution level 2
        rotation: numpy.ndarray
            Rotation prediction in 3 element angle axis format
        translation: numpy.ndarray
            Translation prediction
        Returns a dict with the preditions of the iterative net
        """


        fetches = {
                'predict_flow5': self.predict_flow5,
                'predict_flow2': self.predict_flow2,
                'predict_depth2': self.netDM2_result['predict_depth2'],
                'predict_normal2': self.netDM2_result['predict_normal2'],
                'predict_rotation': self.netDM2_result['predict_rotation'],
                'predict_translation': self.netDM2_result['predict_translation'],
                }
        feed_dict = {
                self.placeholder_image_pair: image_pair,
                self.placeholder_image2_2: image2_2,
                self.placeholder_depth2: depth2,
                self.placeholder_normal2: normal2,
                self.placeholder_rotation: rotation,
                self.placeholder_translation: translation,
                }
        return self.session.run(fetches, feed_dict=feed_dict)




class RefinementNet:

    def __init__(self, session):
        """Creates the network
        session: tf.Session
            Tensorflow session
        """

        self.session = session
        self.placeholder_image_pair = tf.placeholder(dtype=tf.float32, shape=(1,6,192,256))
        self.placeholder_image1 = tf.placeholder(dtype=tf.float32, shape=(1,3,192,256))
        self.placeholder_depth2 = tf.placeholder(dtype=tf.float32, shape=(1,1,48,64))
        self.placeholder_normal2 = tf.placeholder(dtype=tf.float32, shape=(1,3,48,64))
        self.placeholder_rotation = tf.placeholder(dtype=tf.float32, shape=(1,3))
        self.placeholder_translation = tf.placeholder(dtype=tf.float32, shape=(1,3))

        with tf.variable_scope('netRefine'):
            self.netRefine_result = depth_refine_block(
                    image1=self.placeholder_image1, 
                    depthmotion_predictions={
                        'predict_depth2': self.placeholder_depth2,
                        'predict_normal2': self.placeholder_normal2,
                        },
                    )

    def eval(self, image1, depth2, normal2):
        """Runs the refinement network
        
        image1: numpy.ndarray
            Array with the first image with shape [1,3,192,256]
        depth2: numpy.ndarray
            Depth prediction at resolution level 2
        normal2: numpy.ndarray
            normal prediction at resolution level 2
        Returns a dict with the preditions of the refinement net
        """

        fetches = {
                'predict_depth0': self.netRefine_result['predict_depth0'],
                'predict_normal0': self.netRefine_result['predict_normal0'],
                }
        feed_dict = {
                self.placeholder_image1: image1,
                self.placeholder_depth2: depth2,
                self.placeholder_normal2: normal2,
                }
        return self.session.run(fetches, feed_dict=feed_dict)
