import os

def read_file_content(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):  # assuming you're only interested in .txt files
            print(f"File: {filename}\n")
            with open(os.path.join(directory, filename), 'r') as file:
                for i in range(10):
                    line = file.readline()
                    if not line:
                        break
                    print(line.strip())
            print("\n")  # New line after each file content for better separation

# Specify the directory path
directory_path = '/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/datasets/megadepth_scene_lists'
read_file_content(directory_path)

"""
(GlueFactory) tp4618@gpu01:megadepth_scene_lists$ python read_files.py 
File: valid_pairs.txt

0015/images/492130269_796b5bf602_o.jpg 0015/images/3611827485_281ac6d564_o.jpg
0022/images/2835868540_572241d9f7_o.jpg 0022/images/1610927842_7027d5148d_o.jpg
0022/images/2675714713_c4e4746967_o.jpg 0022/images/2244694791_ec4eee64cc_o.jpg
0022/images/38958495_74efb603d5_o.jpg 0022/images/3715690353_64477b25da_o.jpg
0022/images/1469603948_0052cdbe5d_o.jpg 0022/images/1070377533_d043020b4f_o.jpg
0022/images/3216263284_1c2f358e5a_o.jpg 0022/images/2941542538_0f61a1959c_o.jpg
0015/images/2959733121_2a804e7d1c_o.jpg 0015/images/3480270595_61e9118a00_o.jpg
0022/images/2835033819_2806874c89_o.jpg 0022/images/3353330026_1c0b8f51d9_o.jpg
0022/images/2523305319_f42a537f61_o.jpg 0022/images/2652866084_f528569160_o.jpg
0022/images/3106463411_710199120e_o.jpg 0022/images/3000898340_59cef56f18_o.jpg


File: valid_scenes.txt

0016
0033
0034
0041
0044
0047
0049
0058
0062
0064


File: train_scenes.txt

0000
0001
0002
0003
0004
0005
0007
0008
0011
0012


File: train_scenes_clean.txt

0001
0003
0004
0005
0007
0012
0013
0016
0017
0023


File: test_scenes_clean.txt

0008
0019
0021
0024
0025
0032
0063
1589


File: valid_scenes_clean.txt

0015
0022

"""