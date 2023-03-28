import os
from PIL import Image


def clean_namelist(raw_namelist):
#clean the .jpg name list and sort it
    clean_name_list = list()
    for x in sorted(raw_namelist):
        if ".jpg" in x.lower():
            clean_name_list.append(x)
    return clean_name_list

def resize(source_path,destination_path,name_list,output_name,cpt,size=(256,256)):
    for name in name_list:
        photo=Image.open(os.path.join(source_path,name))
        resized_photo=photo.resize(size)
        resized_photo.save(os.path.join(destination_path,f"{output_name}{str(cpt+1).rjust(4,'0')}.jpg"))
        cpt+=1
    return None

def get_photo_sketch_dict(photo_path,sketch_path):
    #Takes 2 lists of matching length as input and returns a dictionary with the
    #first list as key and the second list as value (in corresponding order

    photo_list=clean_namelist(os.listdir(photo_path))
    sketch_list=clean_namelist(os.listdir(sketch_path))

    if len(photo_list)==len(sketch_list):
        return {photo_list[i]:sketch_list[i] for i in range(len(photo_list))}
    else :
        print("The length of the sketch list must match the length of the photo list! Please check the data folders.")
        return None

def emptying(path):
    for image in clean_namelist(os.listdir(path)):
        os.remove(os.path.join(path,image))
