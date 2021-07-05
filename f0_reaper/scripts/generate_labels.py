import glob
import os


ROOT = "../"
MODE = "train-clean"
DATA_FOLDER = ROOT + "data/audio/wavs/" + MODE + "/LibriSpeech/" + MODE + "/"

def get_all_subdirs(folder):
	dir_list = []
	for x in os.walk(folder):
		if len(x[1]) == 0:
			dir_list.append(x[0])
	return dir_list

dir_list = get_all_subdirs(DATA_FOLDER)

for dirs in dir_list:
	file = glob.glob(dirs+"/*txt")[0]
	
	with open(file) as fp:
		lines = fp.readlines()

	for l in lines:
		temp = l.split(" ")
		labfile = dirs + "/" + temp[0] + ".lab"
		labcontent = " ".join(temp[1:])

		with open(labfile, "w") as fw:
			fw.write(labcontent)
