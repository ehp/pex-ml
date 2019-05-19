# split dataset into training and testing file list

import sys
import random
import logging

split_factor = 0.8
indoor_categories = ('Bedroom', 'Bathroom', 'Classroom', 'Office')
outdoor_categories = ('Landscape', 'Skyscraper', 'Mountain', 'Beach')
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

input_file = sys.argv[1]
train_file = sys.argv[2]
test_file = sys.argv[3]

logger.info('Reading input file %s' % input_file)
with open(input_file, 'r') as f:
    lines = f.readlines()
logger.info('%d rows loaded' % len(lines))

# Examples:
# frames/SkyscraperStesy en el Rascacielos Irtra Petapa.mp4-eVqKAYEpSHM/frames/frame_00008.png
# frames/SkyscraperStesy en el Rascacielos Irtra Petapa.mp4-eVqKAYEpSHM/frames/frame_00019.png
# frames/SkyscraperStesy en el Rascacielos Irtra Petapa.mp4-eVqKAYEpSHM/frames/frame_00010.png

logger.info('Splitting between train/test')
indoor = []
outdoor = []
for line in lines:
    ls = line.split('/')
    if ls[1].startswith(indoor_categories):
        indoor.append(line.strip())
    elif ls[1].startswith(outdoor_categories):
        outdoor.append(line.strip())
    else:
        logger.error('Unknown category for line %s' % line)

# rebalance
random.shuffle(indoor)
random.shuffle(outdoor)
item_count = min(len(indoor), len(outdoor))
logger.info('%d items in each category after balancing' % item_count)

def writefile(filename, a0, a1):
    logger.info('Writing output to %s (%d lines)' % (filename, len(a0) + len(a1)))
    with open(filename, 'w') as f:
        for path in a1:
            f.write('%s %d\n' % (path, 1))
        for path in a0:
            f.write('%s %d\n' % (path, 0))


writefile(train_file, outdoor[:int(item_count * split_factor)], indoor[:int(item_count * split_factor)])
writefile(test_file, outdoor[int(item_count * split_factor):item_count], indoor[int(item_count * split_factor):item_count])

logger.info('All done')
