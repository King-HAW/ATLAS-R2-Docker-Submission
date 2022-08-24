loader_settings = {"InputPath": "/input/images/t1-brain-mri/",
                    "OutputPath": "/output/images/stroke-lesion-segmentation/",
                    "GroundTruthRoot": "/opt/evaluatlion/mha_masks/",
                    "JSONPath": "/input/predictions.json",
                    "BatchSize": 2,
                    "InputSlugs": ["t1-brain-mri"],
                    "OutputSlugs": ["stroke-lesion-segmentation"]}