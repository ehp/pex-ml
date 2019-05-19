from pex.train import Dataset
import numpy as np


def test_dataset(tmp_path):
    list_file = str(tmp_path / 'test.lst')
    generate_list_file(list_file, [('aaa/frame_00001.png', 0), ('bbb/frame_00002.png', 1)])
    dataset = Dataset('test/data', list_file, 224, train=True)

    assert len(dataset) == 2

    image, label = dataset[0]
    assert label in (np.int(0), np.int(1))
    assert image.shape == (3, 224, 224)

    image, label = dataset[1]
    assert label in (np.int(0), np.int(1))
    assert image.shape == (3, 224, 224)


def generate_list_file(filename, data):
    with open(filename, 'w') as f:
        for path, label in data:
            f.write('%s %d\n' % (path, label))
