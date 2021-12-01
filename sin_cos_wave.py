import numpy as np

np.random.seed(0)

def sin_wave(A, omega, fai, length):
	# A*sin(ωt - φ)
	x = np.linspace(0, 2*np.pi*(length/omega), length)
	return A*np.sin(x - fai)
def cos_wave(A, omega, fai, length):
	# A*cos(ωt - φ)
	x = np.linspace(0, 2*np.pi*(length/omega), length)
	return A*np.cos(x - fai)

def sample_sequence(time, sensor, sequence):
	#X=(Time, sensor, sequence)
	X = np.empty((time, sensor, sequence))
	for seq in range(sequence):
		for sen in range(sensor):
			A = np.random.rand()
			omega = np.random.rand()
			fai = np.random.rand()
			if np.random.randint(0,2,1) == 0:
				X[:, sen, seq] = sin_wave(A, omega, fai, time)
			else:
				X[:, sen, seq] = cos_wave(A, omega, fai, time)
	return X

