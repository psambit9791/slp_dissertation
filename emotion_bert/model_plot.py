import matplotlib.pyplot as plt
import numpy as np

ROOT = "images/model/"

def plot_curve(loss, epochs, filename):
	y_label = "Loss"
	x_Label = "Epoch"

	plt.figure(figsize=(12, 6))
	plt.plot(epochs, loss[0], label="training loss")
	plt.plot(epochs, loss[1], label="validation loss")
	min_point = np.min(loss[1])
	min_epoch = np.argmin(loss[1])
	plt.scatter(epochs[min_epoch], min_point, marker="o", color='r', label="least validation loss")
	plt.title("Training BERT using "+filename.upper()+" dataset")
	plt.xlabel(x_Label)
	plt.ylabel(y_label)
	plt.xticks(epochs)
	plt.grid()
	plt.legend()

	plt.tight_layout()
	plt.savefig(ROOT+filename+".pdf")
	plt.clf()


epochs = [1, 2, 3, 4, 5]

imdb_train = [0.0616, 0.0377, 0.0254, 0.0176, 0.0133]
imdb_valid = [0.0458, 0.0436, 0.0449, 0.0497, 0.0522]

silicone_train = [0.3312, 0.2560, 0.2268, 0.2074, 0.1959]
silicone_valid = [0.2602, 0.2404, 0.2342, 0.2347, 0.2355]

goemotion_train = [0.2665, 0.2099, 0.1901, 0.1755, 0.1662]
goemotion_valid = [0.2245, 0.2165, 0.2152, 0.2196, 0.2207]

custom_train = [0.3058, 0.2401, 0.2141, 0.1954, 0.1834]
custom_valid = [0.2493, 0.2389, 0.2386, 0.2392, 0.2400]

plot_curve((imdb_train, imdb_valid), epochs, "imdb")
plot_curve((imdb_train, imdb_valid), epochs, "silicone")
plot_curve((imdb_train, imdb_valid), epochs, "goemotion")
plot_curve((imdb_train, imdb_valid), epochs, "custom")