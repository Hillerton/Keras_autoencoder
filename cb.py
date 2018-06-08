"""
suporting file for running init_autoencoder.py
contains settings for how Tensorflow should record meta data about the mode
"""

from keras.callbacks import TensorBoard
from tensorflow.python.client import timeline
from guppy import hpy
import sys

class TB(TensorBoard):
	def __init__(self, log_dir='./logs',
				 histogram_freq=0,
				 batch_size=32,
				 write_graph=True,
				 write_grads=False,
				 write_images=False,
				 embeddings_freq=0,
				 embeddings_layer_names=None,
				 embeddings_metadata=None,
				 run_metadata=None,
				 dump_heap=False,
				 model=None,
				 model_save_file='',
				 timeline_file=''):
		super(TB, self).__init__(log_dir,
								 histogram_freq,
								 batch_size,
								 write_graph,
								 write_grads,
								 write_images,
								 embeddings_freq,
								 embeddings_layer_names,
								 embeddings_metadata)
		self.run_metadata = run_metadata
		self.dump_heap=dump_heap
		self.model = model
		self.model_save_file = model_save_file
		self.timeline_file = timeline_file

	def on_epoch_end(self, epoch, logs=None):
		self.writer.add_run_metadata(self.run_metadata, 'step%d' % epoch)
		super(TB, self).on_epoch_end(epoch, logs)

		# TODO: Really belongs in its own callback
		if self.model_save_file:
			self.model.save(self.model_save_file)

		if self.timeline_file:
			tl = timeline.Timeline(self.run_metadata.step_stats)
			ctf = tl.generate_chrome_trace_format()
			with open(self.timeline_file, 'w') as f:
				f.write(ctf)

		if self.dump_heap: print >> sys.stderr, "\n", hpy().heap(), "\n"
