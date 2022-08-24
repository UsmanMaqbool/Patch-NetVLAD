import os
import shutil

file = open("/media/leo/2C737A9872F69ECF/datasets/RobotCarv2/robotcar_v2_test.txt", 'r')
filename = file.readlines()
for line in filename:
    print(line)
    lines = str(line)
#    if os.path.isfile(lines):
#        print ("File exist")
    shutil.copy(lines[:-1], "abc/"+lines[:-1])
#        print(filename)
