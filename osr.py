from keras import backend as K
from keras.callbacks import Callback
from keras.constraints import maxnorm
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution3D
from keras.layers.convolutional import MaxPooling3D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.utils.io_utils import HDF5Matrix
from pprint import pprint
from sklearn.utils import shuffle
K.set_image_dim_ordering("th")

import cv2
import h5py
import json
import os
import sys
import numpy as np

class OpticalSpeechRecognizer(object):
	def __init__(self, rows, columns, frames_per_sequence, config_file, training_save_fn):
		self.rows = rows
		self.columns = columns
		self.frames_per_sequence = frames_per_sequence
		self.config_file = config_file
		self.training_save_fn = training_save_fn
		self.osr = None

	def train_osr_model(self):
		""" Train the optical speech recognizer
		"""
		print "\nTraining OSR"
		validation_ratio = 0.3
		training_sequence_generator = self.generate_training_sequences(batch_size=10)
		validation_sequence_generator = self.generate_training_sequences(batch_size=10, validation_ratio=validation_ratio)
		
		with h5py.File(self.training_save_fn, "r") as training_save_file:
			sample_count = training_save_file.attrs["sample_count"]
			pbi = ProgressDisplay()
			self.osr.fit_generator(generator=training_sequence_generator,
								   validation_data=validation_sequence_generator,
								   samples_per_epoch=sample_count,
								   nb_val_samples=int(round(validation_ratio*sample_count)),
								   nb_epoch=10,
								   verbose=2,
								   callbacks=[pbi],
								   class_weight=None,
								   nb_worker=1)

	def generate_training_sequences(self, batch_size, validation_ratio=0):
		""" Generates training sequences from HDF5 file on demand
		"""
		while True:
			with h5py.File(self.training_save_fn, "r") as training_save_file:
				sample_count = int(training_save_file.attrs["sample_count"])
				# generate sequences for validation
				if validation_ratio:
					validation_sample_count = int(round(validation_ratio*sample_count))
					validation_sample_idxs = np.random.randint(low=0, high=sample_count, size=validation_sample_count)
					batches = int(validation_sample_count/batch_size)
					remainder_samples = validation_sample_count%batch_size
					# generate batches of samples
					for idx in xrange(0, batches):
						X = training_save_file["X"][idx*batch_size:idx*batch_size+batch_size]
						Y = training_save_file["Y"][idx*batch_size:idx*batch_size+batch_size]
						yield (X, Y)
					# send remainder samples as one batch, if there are any
					if remainder_samples:
						X = training_save_file["X"][-remainder_samples:]
						Y = training_save_file["Y"][-remainder_samples:]
				# generate sequences for training
				else:
					batches = int(sample_count/batch_size)
					remainder_samples = sample_count%batch_size
					# generate batches of samples
					for idx in xrange(0, batches):
						X = training_save_file["X"][idx*batch_size:idx*batch_size+batch_size]
						Y = training_save_file["Y"][idx*batch_size:idx*batch_size+batch_size]
						yield (X, Y)
					# send remainder samples as one batch, if there are any
					if remainder_samples:
						X = training_save_file["X"][-remainder_samples:]
						Y = training_save_file["Y"][-remainder_samples:]

	def print_osr_summary(self):
		""" Prints a summary representation of the OSR model
		"""
		print "\n*** MODEL SUMMARY ***"
		self.osr.summary()

	def generate_osr_model(self):
		""" Builds the optical speech recognizer model
		"""
		print "".join(["\nGenerating OSR model\n",
					   "-"*40])
		with h5py.File(self.training_save_fn, "r") as training_save_file:
			osr = Sequential()
			print " - Adding convolution layers"
			osr.add(Convolution3D(nb_filter=32,
								  kernel_dim1=3,
								  kernel_dim2=3,
								  kernel_dim3=3,
								  border_mode="same",
								  input_shape=(1, self.frames_per_sequence, self.rows, self.columns),
								  activation="relu"))
			osr.add(MaxPooling3D(pool_size=(3, 3, 3)))
			osr.add(Convolution3D(nb_filter=64,
								  kernel_dim1=3,
								  kernel_dim2=3,
								  kernel_dim3=3,
								  border_mode="same",
								  activation="relu"))
			osr.add(MaxPooling3D(pool_size=(3, 3, 3)))
			osr.add(Convolution3D(nb_filter=128,
								  kernel_dim1=3,
								  kernel_dim2=3,
								  kernel_dim3=3,
								  border_mode="same",
								  activation="relu"))
			osr.add(MaxPooling3D(pool_size=(3, 3, 3)))
			osr.add(Dropout(0.2))
			osr.add(Flatten())
			print " - Adding fully connected layers"
			osr.add(Dense(output_dim=128,
						  init="normal",
						  activation="relu"))
			osr.add(Dense(output_dim=128,
						  init="normal",
						  activation="relu"))
			osr.add(Dense(output_dim=128,
						  init="normal",
						  activation="relu"))
			osr.add(Dropout(0.2))
			osr.add(Dense(output_dim=len(training_save_file.attrs["training_classes"].split(",")),
						  init="normal",
						  activation="softmax"))
			print " - Compiling model"
			sgd = SGD(lr=0.01,
					  decay=1e-6,
					  momentum=0.9,
					  nesterov=True)
			osr.compile(loss="categorical_crossentropy",
						optimizer=sgd,
						metrics=["accuracy"])
			self.osr = osr
			print " * OSR MODEL GENERATED * "

	def process_training_data(self):
		""" Preprocesses training data and saves them into an HDF5 file
		"""
		# load training metadata from config file
		training_metadata = {}
		training_classes = []
		with open(self.config_file) as training_config:
			training_metadata = json.load(training_config)
			training_classes = sorted(list(training_metadata.keys()))

			print "".join(["\n",
						   "Found {0} training classes!\n".format(len(training_classes)),
						   "-"*40])
			for class_label, training_class in enumerate(training_classes):
				print "{0:<4d} {1:<10s} {2:<30s}".format(class_label, training_class, training_metadata[training_class])
			print ""

		# count number of samples
		sample_count = 0
		sample_count_by_class = [0]*len(training_classes)
		for class_label, training_class in enumerate(training_classes):
			# get training class sequeunce paths
			training_class_data_path = training_metadata[training_class]
			training_class_sequence_paths = [os.path.join(training_class_data_path, file_name)
											 for file_name in os.listdir(training_class_data_path)
											 if (os.path.isfile(os.path.join(training_class_data_path, file_name))
												 and ".mov" in file_name)]
			# update sample count
			sample_count += len(training_class_sequence_paths)
			sample_count_by_class[class_label] = len(training_class_sequence_paths)

		print "".join(["\n",
					   "Found {0} training samples!\n".format(sample_count),
					   "-"*40])
		for class_label, training_class in enumerate(training_classes):
			print "{0:<4d} {1:<10s} {2:<6d}".format(class_label, training_class, sample_count_by_class[class_label])
		print ""

		# initialize HDF5 save file, but clear older duplicate first if it exists
		try:
			print "Saved file \"{0}\" already exists! Overwriting previous saved file.\n".format(self.training_save_fn)
			os.remove(self.training_save_fn)
		except OSError:
			pass

		# process and save training data into HDF5 file
		with h5py.File(self.training_save_fn, "w") as training_save_file:
			training_save_file.attrs["training_classes"] = np.string_(",".join(training_classes))
			training_save_file.attrs["sample_count"] = sample_count
			x_training_dataset = training_save_file.create_dataset("X", 
																  shape=(sample_count, 1, self.frames_per_sequence, self.rows, self.columns),
																  dtype="f")
			y_training_dataset = training_save_file.create_dataset("Y",
																   shape=(sample_count, len(training_classes)),
																   dtype="i")

			# iterate through each class data
			sample_idx = 0
			for class_label, training_class in enumerate(training_classes):
				# get training class sequeunce paths
				training_class_data_path = training_metadata[training_class]
				training_class_sequence_paths = [os.path.join(training_class_data_path, file_name)
												 for file_name in os.listdir(training_class_data_path)
												 if (os.path.isfile(os.path.join(training_class_data_path, file_name))
													 and ".mov" in file_name)]
				# iterate through each sequence
				for idx, training_class_sequence_path in enumerate(training_class_sequence_paths):
					sys.stdout.write("Processing training data for class \"{0}\": {1}/{2} sequences\r"
								     .format(training_class, idx+1, len(training_class_sequence_paths)))
					sys.stdout.flush()
					
					# append grayscale, normalized sample frames
					frames = self.process_frames(training_class_sequence_path)
					x_training_dataset[sample_idx] = [frames]

					# append one-hot encoded sample label
					label = [0]*len(training_classes)
					label[class_label] = 1
					y_training_dataset[sample_idx] = label

					# update sample index
					sample_idx += 1

				print "\n"

			training_save_file.close()

			print "Training data processed and saved to {0}".format(self.training_save_fn)

	def process_frames(self, video_file_path):
		""" Splits frames, resizes frames, converts RGB frames to greyscale, and normalizes frames
		"""
		# haar cascades for localizing oral region
		face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
		mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')

		video = cv2.VideoCapture(video_file_path)
		success, frame = video.read()

		frames = []
		success = True

		# convert to grayscale, localize oral region, equalize dimensions, 
		# normalize pixels, equalize lengths, and accumulate valid frames 
		while success:
		  success, frame = video.read()
		  if success:
		  	# convert to grayscale
		  	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		  	# localize single facial region
		  	faces_coords = face_cascade.detectMultiScale(frame, 1.3, 5)
		  	if len(faces_coords) == 1:
		  	  face_x, face_y, face_w, face_h = faces_coords[0]
		  	  frame = frame[face_y:face_y + face_h, face_x:face_x + face_w]

		  	  # localize oral region
		  	  mouth_coords = mouth_cascade.detectMultiScale(frame, 1.3, 5)
		  	  threshold = 0
		  	  for (mouth_x, mouth_y, mouth_w, mouth_h) in mouth_coords:
		  	  	if (mouth_y > threshold):
		  	  		threshold = mouth_y
		  	  		valid_mouth_coords = (mouth_x, mouth_y, mouth_w, mouth_h)
		  	  	else:
		  	  		pass
		  	  mouth_x, mouth_y, mouth_w, mouth_h = valid_mouth_coords
		  	  frame = frame[mouth_y:mouth_y + mouth_h, mouth_x:mouth_x + mouth_w]

		  	  # equalize dimensions and normalize pixels
			  frame = cv2.resize(frame, (self.columns, self.rows))
			  frame = frame.astype('float32') / 255.0
			  frames.append(frame)

			# ignore multiple facial region detections
			else:
				pass

		# pre-pad short sequences and equalize sequence lengths
		if len(frames) < self.frames_per_sequence:
			frames = [frames[0]]*(self.frames_per_sequence - len(frames)) + frames
		frames = frames[0:self.frames_per_sequence]

		return [frames]

class ProgressDisplay(Callback):
	""" Progress display callback
	"""
	def on_batch_end(self, epoch, logs={}):
		print "    Batch {0:<4d} => Accuracy: {1:>8.4f} | Loss: {2:>8.4f} | Size: {3:>4d}".format(int(logs["batch"]),
																					              float(logs["acc"]),
																					              float(logs["loss"]),
																					              int(logs["size"]))

if __name__ == "__main__":
	osr = OpticalSpeechRecognizer(100, 150, 45, "training_config.json", "training_data.h5")
	osr.process_training_data()
	osr.generate_osr_model()
	osr.print_osr_summary()
	osr.train_osr_model()

