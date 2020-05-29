import midv500 

if __name__ == "__main__":

    dataset_dir = 'data/'
    midv500.download_dataset(dataset_dir)
    export_dir = 'data/'
    
    print("Convert format to Coco")
    midv500.convert_to_coco(dataset_dir, export_dir)



