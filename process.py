import numpy as np
from settings import loader_settings
import medpy.io
import os, pathlib

class Seg():
    def __init__(self):
        # super().__init__(
        #     validators=dict(
        #         input_image=(
        #             UniqueImagesValidator(),
        #             UniquePathIndicesValidator(),
        #         )
        #     ),
        # )
        return
        
    def process(self):
        inp_path = loader_settings['InputPath']  # Path for the input
        out_path = loader_settings['OutputPath']  # Path for the output
        file_list = os.listdir(inp_path)  # List of files in the input
        file_list = [os.path.join(inp_path, f) for f in file_list]
        for fil in file_list:
            dat, hdr = medpy.io.load(fil)  # dat is a numpy array
            im_shape = dat.shape
            dat = dat.reshape(1, 1, *im_shape)  # reshape to Pytorch standard
            # Convert 'dat' to Tensor, or as appropriate for your model.
            ###########
            ### Replace this section with the call to your code.
            mean_dat = np.mean(dat)
            dat[dat > mean_dat] = 1
            dat[dat <= mean_dat] = 0
            ###
            ###########
            dat = dat.reshape(*im_shape)
            out_name = os.path.basename(fil)
            out_filepath = os.path.join(out_path, out_name)
            print(f'=== saving {out_filepath} from {fil} ===')
            medpy.io.save(dat, out_filepath, hdr=hdr)
        return


if __name__ == "__main__":
    pathlib.Path("/output/images/stroke-lesion-segmentation/").mkdir(parents=True, exist_ok=True)
    Seg().process()
