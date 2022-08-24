import json
import medpy.io
import os
import numpy as np

class GrandChallengesLoader():
    '''
    Loader for parsing the JSON file produced by Grand Challenges mapping the output filenames to their input.
    '''
    def __init__(self, loader_settings, prediction_json_filepath='/input/predictions.json'):
        '''
        Initialize the loader: parse the JSON file and prepare list of data->predictions->target.
        Parameters
        ----------
        loader_settings : dict
            Settings dictionary for the GC loader.
        prediction_json_filepath : str
            Path to the JSON file specifying the prediction filemapping. Default: /input/predictions.json
        batch_size : int
            Number of images to use in a batch.
        '''

        self.batch_size = loader_settings['BatchSize']
        self.prediction_mapping = {}
        self.loader_settings = loader_settings
        self.image_dimensions = None
        self.prediction_list = os.listdir(self.loader_settings["OutputPath"])
        # Load json
        f = open(prediction_json_filepath, 'r')
        self.json_data = json.load(f)
        f.close()
        self.mapping()

        inp_path = self.loader_settings['InputPath']
        file_list = os.listdir(inp_path)
        if(self.image_dimensions is None):
            # Image dimensions unknown; load one and set it
            im, _ = medpy.io.load(os.path.join(inp_path, file_list[0]))
            self.image_dimensions = im.shape
        return

    def __len__(self):
        return len(self.prediction_list)

    def mapping(self):
        '''
        Performs the mapping from prediction to input.
        '''
        # Create dict mapping predictions -> input
        # Go through each sample in the file
        for entity in self.json_data:
            ent_inputs = entity['inputs']
            ent_outputs = entity['outputs']
            pred = None  # path of the prediction file
            output_path_list = []  # list of the expected output path based on the input filename
            # Get input list
            for channel_data in ent_inputs:
                if(channel_data['interface']['slug'] in self.loader_settings['InputSlugs']):
                    inp_filepath = os.path.basename(channel_data['image']['name'])
                    # Assuming single output directory
                    out_slug = self.loader_settings["OutputSlugs"][0]
                    # out_path = os.path.join(self.loader_settings["OutputPath"], out_slug, inp_filepath)
                    out_path = os.path.join(self.loader_settings["GroundTruthRoot"], inp_filepath)
                    output_path_list.append(out_path)
                    break
            # Get output list
            for channel_data in ent_outputs:
                if(channel_data['interface']['slug'] in self.loader_settings['OutputSlugs']):
                    # prediction_list.append(channel_data['image']['pk'])
                    pred = channel_data['image']['pk']
                    break
            # For our purposes, any number of inputs will uniquely identify the total set of ground truth files (via
            # the sub- and ses- tags in the filename).
            if(pred is not None):
                self.prediction_mapping[os.path.basename(pred)] = output_path_list[0]
        return


    def load_batch_for_prediction(self):
        '''
        Loads batches of data without loading the corresponding target.

        Yields
        ------
        np.array
            Array of shape (self.batch_size, 1, *im_shape) containing the image data.
        '''
        # Load .mha files
        inp_path = self.loader_settings['InputPath']
        file_list = os.listdir(inp_path)
        num_files = len(file_list)
        if(self.image_dimensions is None):
            # Image dimensions unknown; load one and set it
            im, _ = medpy.io.load(os.path.join(inp_path, file_list[0]))
            self.image_dimensions = im.shape
        # Load each batch of the data; return in batch_size-sized arrays, return less if less than a batch of data
        # remains
        for i in range(0, num_files, self.batch_size):
            last_idx = int(min([i+self.batch_size, num_files]))
            image_list = [os.path.join(inp_path, f) for f in file_list[i:last_idx]]
            yield self.load_list(image_list), image_list
        return

    def load_list(self, filepath_list: list) -> np.array:
        '''
        Loads all the files in the supplied list and returns them as a single numpy array.
        Parameters
        ----------
        filepath_list : list
            List of filepaths pointing to the images to load.

        Returns
        -------
        np.array
            Numpy array corresponding to the data in the filepath list of shape
            [len(filepath_list), 1, *self.image_dimensions]. I.e., multi-channel data is not implemented.
        '''
        num_files = len(filepath_list)
        data = np.zeros((num_files, 1, *self.image_dimensions))
        for i in range(num_files):
            data[i, 0, ...], _ = medpy.io.load(filepath_list[i])
        return data

    @staticmethod
    def write_image_like(data_to_write: np.array,
                         image_to_imitate: str,
                         output_root: str):
        '''
        Writes the specified single sample into .mha file in the specified root directory.
        Parameters
        ----------
        data_to_write : np.array
            Data to write out. Dimensions are expected to be (x,y,z) (i.e., without batch or channel).
             E.g., the prediction of a model.
        image_to_imitate : str
            Name of the file to imitate.
        output_root : str
            Root directory to which to write the data.
        '''

        # Figure out output path
        output_basename = os.path.basename(image_to_imitate)
        output_filepath = os.path.join(output_root, output_basename)

        # Write out data
        medpy.io.save(data_to_write, output_filepath)
        return

    def write_images_like(self, data_to_write: np.array,
                          images_to_imitate: list,
                          output_root: str):
        '''
        Writes the specified data into .mha files in the specified root directory. This method assumes that
        data_to_write is of the shape [num_batches, 1, *im.shape] and that len(images_to_imitate) == num_batches. The
        expected use is for data_to_write to contain model predictions of the images_to_imitate.

        Parameters
        ----------
        data_to_write : np.array
            Data to write out as .mha. Expected shape of [num_batches, 1, *im.shape].
        images_to_imitate : list
            List of the image filepaths that should be used to generate the output path.
        output_root : str
            Root directory to which to write the data.
        '''

        if(data_to_write.shape[0] != len(images_to_imitate)):
            raise ValueError(f'data batch size ({data_to_write.shape[0]}) doesn'f't match images_to_imitate '
                             f'{len(images_to_imitate)}')

        for i in range(len(images_to_imitate)):
            self.write_image_like(data_to_write=data_to_write[i, 0, ...],
                                  image_to_imitate=images_to_imitate[i],
                                  output_root=output_root)
        return

    def load_batches(self):
        # Wrapper
        for pred, truth in self.load_eval_batches():
            yield pred, truth
        return

    def load_eval_batches(self):
        '''
        Load batches of data, including the corresponding targets.

        Yields
        ------
        np.array
            Data
        np.array
            Target
        '''

        # Go through the prediction
        pred_path = self.loader_settings["OutputPath"]
        pred_list = os.listdir(pred_path)
        pred_list = [os.path.join(pred_path, p) for p in pred_list]
        for idx in range(0, len(pred_list), self.batch_size):
            batch_pred_list = pred_list[idx : idx + self.batch_size]
            batch_truth_list = [self.prediction_mapping[os.path.basename(pred)] for pred in batch_pred_list]
            pred_data = self.load_list(batch_pred_list)
            truth_data = self.load_list(batch_truth_list)
            yield pred_data, truth_data
        return
