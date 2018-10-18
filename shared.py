import os

def get_all_imgs(dir):
	res = []
	for file in os.listdir(dir):
		if file.endswith(".jpg"):
			res.append(os.path.join(dir, file))
	return res


imgs = get_all_imgs('/Users/kolsha/Pictures')

