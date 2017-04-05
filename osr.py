from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.callbacks import Callback
from keras.constraints import maxnorm
from keras.models import Model, model_from_json
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Nadam
from keras.preprocessing.image import random_rotation, random_shift, random_shear, random_zoom
from keras.utils import np_utils
from keras.utils.io_utils import HDF5Matrix
from pprint import pprint
K.set_image_dim_ordering("th")

import cv2
import h5py
import json
import os
import sys
import numpy as np


class OpticalSpeechRecognizer(object):
	def __init__(self, samples_generated_per_sample, frames_per_sequence, rows, columns, config_file, training_save_fn, osr_save_fn, osr_weights_save_fn):
		self.samples_generated_per_sample = samples_generated_per_sample
		self.frames_per_sequence = frames_per_sequence
		self.rows = rows
		self.columns = columns
		self.config_file = config_file
		self.training_save_fn = training_save_fn
		self.osr_save_fn = osr_save_fn
		self.osr_weights_save_fn = osr_weights_save_fn
		self.osr = None

	def save_osr_model(self):
		""" Save the OSR model to an HDF5 file
		"""
		# delete save files, if they already exist
		try:
			print "\nOSR save file \"{0}\" already exists! Overwriting previous saved file.".format(self.osr_save_fn)
			os.remove(self.osr_save_fn)
		except OSError:
			pass
		try:
			print "OSR weights save file \"{0}\" already exists! Overwriting previous saved file.\n".format(self.osr_weights_save_fn)
			os.remove(self.osr_weights_save_fn)
		except OSError:
			pass

		# save OSR model
		print "\nSaving OSR model to \"{0}\"".format(self.osr_save_fn)
		with open(self.osr_save_fn, "w") as osr_save_file:
			osr_model_json = self.osr.to_json()
			osr_save_file.write(osr_model_json)

		# save OSR model weights
		print "Saving OSR model weights to \"{0}\"".format(self.osr_weights_save_fn)
		self.osr.save_weights(self.osr_weights_save_fn)

		print "Saved OSR model and weights to disk\n"

	def load_osr_model(self):
		""" Load the OSR model from an HDF5 file
		"""
		print "\nLoading OSR model from \"{0}\"".format(self.osr_save_fn)
		with open(self.osr_save_fn, "r") as osr_save_file:
			osr_model_json = osr_save_file.read()
			self.osr = model_from_json(osr_model_json)

		print "Loading OSR model weights from \"{0}\"".format(self.osr_weights_save_fn)
		with open(self.osr_weights_save_fn, "r") as osr_weights_save_file:
			self.osr.load_weights(self.osr_weights_save_fn)
			
		print "Loaded OSR model and weights from disk\n"

	def predict_words(self, sequences):
		""" Predicts the word pronounced in each sequence within the given list of sequences
		"""
		with h5py.File(self.training_save_fn, "r") as training_save_file:
			training_classes = training_save_file.attrs["training_classes"].split(",")
			predictions = self.osr.predict(np.array(sequences)).argmax(axis=-1)
			predictions = [training_classes[class_prediction] for class_prediction in predictions]
		return predictions

	def train_osr_model(self):
		""" Train the optical speech recognizer
		"""
		print "\nTraining OSR"
		validation_ratio = 0.3
		batch_size = 32
		with h5py.File(self.training_save_fn, "r") as training_save_file:
			sample_count = int(training_save_file.attrs["sample_count"])
			sample_idxs = range(0, sample_count)
			sample_idxs = np.random.permutation(sample_idxs)
			training_sample_idxs = sample_idxs[0:int((1-validation_ratio)*sample_count)]
			validation_sample_idxs = sample_idxs[int((1-validation_ratio)*sample_count):]
			training_sequence_generator = self.generate_training_sequences(batch_size=batch_size,
																		   training_save_file=training_save_file,
																		   training_sample_idxs=training_sample_idxs)
			validation_sequence_generator = self.generate_validation_sequences(batch_size=batch_size,
																			   training_save_file=training_save_file,
																			   validation_sample_idxs=validation_sample_idxs)
			pbi = ProgressDisplay()
			self.osr.fit_generator(generator=training_sequence_generator,
								   validation_data=validation_sequence_generator,
								   samples_per_epoch=len(training_sample_idxs),
								   nb_val_samples=len(validation_sample_idxs),
								   nb_epoch=15,
								   max_q_size=1,
								   verbose=2,
								   callbacks=[pbi],
								   class_weight=None,
								   nb_worker=1)

	def generate_training_sequences(self, batch_size, training_save_file, training_sample_idxs):
		""" Generates training sequences from HDF5 file on demand
		"""
		while True:
			# generate sequences for training
			training_sample_count = len(training_sample_idxs)
			batches = int(training_sample_count/batch_size)
			remainder_samples = training_sample_count%batch_size
			if remainder_samples:
				batches = batches + 1
			# generate batches of samples
			for idx in xrange(0, batches):
				if idx == batches - 1:
					batch_idxs = training_sample_idxs[idx*batch_size:]
				else:
					batch_idxs = training_sample_idxs[idx*batch_size:idx*batch_size+batch_size]
				batch_idxs = sorted(batch_idxs)

				X = training_save_file["X"][batch_idxs]
				Y = training_save_file["Y"][batch_idxs]

				yield (np.array(X), np.array(Y))

	def generate_validation_sequences(self, batch_size, training_save_file, validation_sample_idxs):
		while True:
			# generate sequences for validation
			validation_sample_count = len(validation_sample_idxs)
			batches = int(validation_sample_count/batch_size)
			remainder_samples = validation_sample_count%batch_size
			if remainder_samples:
				batches = batches + 1
			# generate batches of samples
			for idx in xrange(0, batches):
				if idx == batches - 1:
					batch_idxs = validation_sample_idxs[idx*batch_size:]
				else:
					batch_idxs = validation_sample_idxs[idx*batch_size:idx*batch_size+batch_size]
				batch_idxs = sorted(batch_idxs)

				X = training_save_file["X"][batch_idxs]
				Y = training_save_file["Y"][batch_idxs]

				yield (np.array(X), np.array(Y))

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
			class_count = len(training_save_file.attrs["training_classes"].split(","))
		video = Input(shape=(self.frames_per_sequence,
							 3,
							 self.rows,
							 self.columns))
		cnn_base = VGG16(input_shape=(3,
									  self.rows,
									  self.columns),
						 weights="imagenet",
						 include_top=False)
		cnn_out = GlobalAveragePooling2D()(cnn_base.output)
		cnn = Model(input=cnn_base.input, output=cnn_out)
		cnn.trainable = False
		encoded_frames = TimeDistributed(cnn)(video)
		encoded_vid = LSTM(256)(encoded_frames)
		hidden_layer = Dense(output_dim=1024, activation="relu")(encoded_vid)
		outputs = Dense(output_dim=class_count, activation="softmax")(hidden_layer)
		osr = Model([video], outputs)
		optimizer = Nadam(lr=0.002,
						  beta_1=0.9,
						  beta_2=0.999,
						  epsilon=1e-08,
						  schedule_decay=0.004)
		osr.compile(loss="categorical_crossentropy",
					optimizer=optimizer,
					metrics=["categorical_accuracy"])
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
		print "Generating {0} samples from {1} samples via data augmentation\n".format(sample_count*self.samples_generated_per_sample,
																					   sample_count)
		sample_count = sample_count*self.samples_generated_per_sample
		with h5py.File(self.training_save_fn, "w") as training_save_file:
			training_save_file.attrs["training_classes"] = np.string_(",".join(training_classes))
			training_save_file.attrs["sample_count"] = sample_count
			x_training_dataset = training_save_file.create_dataset("X",
																   shape=(sample_count, self.frames_per_sequence, 3, self.rows, self.columns),
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

					# accumulate samples and labels
					samples_batch = self.process_frames(training_class_sequence_path)
					label = [0]*len(training_classes)
					label[class_label] = 1
					label = np.array(label).astype("int32")

					for sample in samples_batch:
						x_training_dataset[sample_idx] = sample
						y_training_dataset[sample_idx] = label

						# update sample index
						sample_idx += 1

				print "\n"

			training_save_file.close()

			print "Training data processed and saved to {0}".format(self.training_save_fn)

	def process_testing_data(self, testing_data_dir):
		""" Preprocesses testing data
		"""
		# acquire testing data sequence paths
		testing_data_sequence_paths = [os.path.join(testing_data_dir, file_name)
									   for file_name in os.listdir(testing_data_dir)
									   if (os.path.isfile(os.path.join(testing_data_dir, file_name))
										  and ".mov" in file_name)]

		# process testing data
		testing_data = []
		print "Processing {0} testing samples from {1}\n".format(len(testing_data_sequence_paths),
																 testing_data_dir)
		for testing_data_idx, testing_data_sequence_path in enumerate(testing_data_sequence_paths):
			sys.stdout.write("Processing testing data: {0}/{1} sequences\r"
						     .format(testing_data_idx+1, len(testing_data_sequence_paths)))
			sys.stdout.flush()
			testing_data.append(self.process_frames(testing_data_sequence_path, augment=False))

		return np.array(testing_data), testing_data_sequence_paths

	def process_frames(self, video_file_path, augment=True):
		""" Preprocesses sequence frames
		"""
		# haar cascades for localizing oral region
		face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
		mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')

		video = cv2.VideoCapture(video_file_path)
		success, frame = video.read()

		frames = []
		success = True

		# convert to grayscale, localize oral region, equalize frame dimensions, and accumulate valid frames
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

		  	  # equalize frame dimensions
		  	  frame = cv2.resize(frame, (self.columns, self.rows)).astype('float32')

		  	  # accumulate frames
			  frames.append(frame)

			# ignore multiple facial region detections
			else:
				pass

		# equalize sequence lengths
		if len(frames) < self.frames_per_sequence:
			frames = [frames[0]]*(self.frames_per_sequence - len(frames)) + frames
		frames = np.array(frames[-self.frames_per_sequence:])

		# function to normalize and add channel dimension to each frame
		proc_frame = lambda frame: np.array([frame / 255.0]*3)

		if augment:
			samples_batch = [np.array(map(proc_frame, frames))]
			# random transformations for data augmentation
			for _ in xrange(0, self.samples_generated_per_sample-1):
				rotated_frames = random_rotation(frames, rg=4.5)
				shifted_frames = random_shift(rotated_frames, wrg=0.05, hrg=0.05)
				sheared_frames = random_shear(shifted_frames, intensity=0.08)
				zoomed_frames = random_zoom(sheared_frames, zoom_range=(1.05, 1.05))
				samples_batch.append(np.array(map(proc_frame, zoomed_frames)))
			return_data = samples_batch

		else:
			return_data = np.array(map(proc_frame, frames))

		return return_data

class ProgressDisplay(Callback):
	""" Progress display callback
	"""
	def on_batch_end(self, epoch, logs={}):
		print "    Batch {0:<4d} => Accuracy: {1:>8.4f} | Loss: {2:>8.4f} | Size: {3:>4d}".format(int(logs["batch"])+1,
																					              float(logs["categorical_accuracy"]),
																					              float(logs["loss"]),
																					              int(logs["size"]))

if __name__ == "__main__":
	osr = OpticalSpeechRecognizer(samples_generated_per_sample=10,
								  frames_per_sequence=30,
								  rows=100,
								  columns=150,
								  config_file="training_config.json",
								  training_save_fn="training_data.h5",
								  osr_save_fn="osr_model.json",
								  osr_weights_save_fn="osr_weights.h5")

	# # Training workflow example 
	# osr.process_training_data()
	# osr.generate_osr_model()
	# osr.print_osr_summary()
	# osr.train_osr_model()
	# osr.save_osr_model()

	# Application workflow example. Requires a trained model. Do not use training data for actual tests
	osr.load_osr_model()
	osr.print_osr_summary()
	test_sequences, test_sequence_file_names = osr.process_testing_data("./testing-data")
	test_predictions = osr.predict_words(test_sequences)
	print "".join(["Predictions for each test sequence\n",
				   "----------------------------------"])
	for file_name, prediction in zip(test_sequence_file_names, test_predictions):
		print "{0}: {1}".format(file_name, prediction)

