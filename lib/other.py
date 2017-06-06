def get_dataset_split_name(im_file):
    parts = im_file.split("/")
    for p in parts[::-1]:
        if p in ['train', 'val', 'test']:
            return p
    return None
