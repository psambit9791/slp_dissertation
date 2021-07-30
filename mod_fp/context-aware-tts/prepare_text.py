import os
import sys

labf = sys.argv[1]
outf = open('text_for_bert.csv', 'w')

print('path:'+labf)

# | separated file: Filename|text1|text2|Text
for file in os.listdir(labf):
    if file.endswith('.lab'):
        text = open(labf+file, 'r').read().replace('  ', ' ').strip()
        outf.writelines(file.replace('.lab', '')+'|||'+text+'\n')
