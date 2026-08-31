ImgHeight = 64
CharWidth = ImgHeight // 2

data_roots = {
    'iam': './data/iam/'
}

def get_data_paths(h=64):
    return {
        'iam_word': {'trnval': f'trnvalset_words{h}.hdf5',
                     'test': f'testset_words{h}.hdf5'},
        'iam_line': {'trnval': f'trnvalset_lines{h}.hdf5',
                     'test': f'testset_lines{h}.hdf5'},
        'iam_word_org': {'trnval': f'trnvalset_words{h}_OrgSz.hdf5',
                         'test': f'testset_words{h}_OrgSz.hdf5'}
    }

data_paths = get_data_paths(ImgHeight)

def set_img_height(height):
    global ImgHeight, CharWidth, data_paths
    ImgHeight = int(height)
    CharWidth = ImgHeight // 2
    data_paths.clear()
    data_paths.update(get_data_paths(ImgHeight))