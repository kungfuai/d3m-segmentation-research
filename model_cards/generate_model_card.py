import model_card_toolkit
import base64

def img_to_base64(img_filepath):
    """ helper function that converts img filepath to base64 string""" 
    with open(img_filepath, "rb") as img_file:
        base64_str = base64.b64encode(img_file.read())
    return base64_str.decode('utf-8')

# See https://github.com/tensorflow/model-card-toolkit for details on model card creation
model_card_output_path = "model_cards"
mct = model_card_toolkit.ModelCardToolkit(model_card_output_path)
model_card = mct.scaffold_assets()



# Section 1 - Model Details
model_card.model_details.name = 'Sentinel-2 Binary Segmentation Model'
model_card.model_details.overview = (
    "This is a binary segmentation model for Sentinel-2 satellite imagery. "
    "It was developed for the DARPA D3M research program, which focuses on automated machine learning. "
    "Therefore, it is a trainable model 'primitive' and not a trained model with specific parameters. "
    "The training process uses image-level labels (weak supervision) instead of dense pixel-level labels "
    "(full supervision). Furthermore, a pre-trained featurization model (trained using a self-supervised "
    "objective on a sample of Sentinel-2 images covering the United States) is used to initialize the "
    "parameters of the segmentation model. Thus, the training process is also an instance of transfer learning."
)
model_card.model_details.owners.append(
    {'name': 'Jeffrey L. Gleason', 'contact': 'jeffrey.gleason@kungfu.ai'}
)
model_card.model_details.version.name = '0.1'
model_card.model_details.version.date = '2/18/2021'
model_card.model_details.references = [
    'https://arxiv.org/pdf/2003.02899.pdf', 'https://www.mdpi.com/2072-4292/12/2/207/htm'
]


# Section 2 - Considerations
model_card.considerations.users = [
    'The intended users of this model are subject matter experts who are interacting with Uncharted '
    'Software\'s Distil platform. The Distil platform visualizes and communicates output from '
    'automated machine learning pipelines, some of which will include this model.' 
]
model_card.considerations.use_cases = [
    'The intended use case is to generate an approximate binary segmentation map given a small number '
    '(i.e. 10s to 100s) of image-level labels. Importantly, the model was only evaluated on land cover data '
    'over Estonia, and it is not known how its performance would change over different geographic areas.'
]
model_card.considerations.limitations = [
    'The model is not as accurate as a model trained with dense pixel-level labels (full supervision). Specifically, '
    'a fully supervised model reaches a mean accuracy of 0.826 with 10 training images (compared to 0.74). However, '
    'the difference is smaller with 100 training images - a fully supervised model reaches a mean accuracy of 0.823'
    '(compared to 0.821). Model performance will be degraded by occlusion, '
    'such as cloud coverage, poor lighting, and issues with Sentinel-2 sensors, though the magnitude of this  '
    'degradation has not been measured.'
]
model_card.considerations.ethical_considerations = [
    {
        'name': 'resource allocation based on segmentation map', 
        'mitigation_strategy': (
            'If this model is used to allocate resources (e.g. mitigation efforts for locust infestations) based on '
            'land cover predictions (e.g. agriculture vs. not agriculture) and there exist systematic biases '
            'in data collection (e.g. the satellite passes over certain locations more frequently than others '
            'or the satellite is occluded in certain locations more frequently than others) then systematic biases '
            'will exist in resource allocation unless intentionally addressed. This risk is highlighted because no '
            'steps are taken during the training procedure to intentionally account for systematic bias in data collection.'
        )
    }
]


# Section 3 - Model Parameters
model_card.model_parameters.model_architecture = 'U-Net with a ResNet50 encoder'
model_card.model_parameters.data.eval.graphics.collection = [
    {
        'name': 'Evaluation Dataset Distribution',
        'image': img_to_base64('model_cards/data/evaluation-set-dist.png')
    }
]
model_card.model_parameters.data.eval.graphics.description = (
    'This graphic shows the distribution of agriculture and non-agriculture pixels in the evaluation dataset. '
    'The evaluation dataset consists of a Sentinel-2 tile over Estonia and its corresponding CORINE land cover map. '
    'The model was only evaluated on this dataset.'
)
model_card.model_parameters.data.eval.name = 'CORINE land cover map and Sentinel-2 image of Estonia'
model_card.model_parameters.data.eval.link = 'https://arxiv.org/pdf/2003.02899.pdf'
model_card.model_parameters.data.eval.sensitive = False 
model_card.model_parameters.input_format = 'Multispectral Sentinel-2 images with dimensions (120, 120, 12) (depth last)'
model_card.model_parameters.output_format = 'Binary segmentation masks with dimensions (120, 120)'



# Section 4 - Quantitative Analysis
model_card.quantitative_analysis.performance_metrics = [
    {
        'type': 'accuracy - 10 train images', 
        'value': 0.74, 
        'confidence_interval': {'lower_bound': 0.593, 'upper_bound': 0.802},
        'threshold': 0.5
    },
    {
        'type': 'expected calibration error - 10 train images', 
        'value': 0.107, 
        'confidence_interval': {'lower_bound': 0.057, 'upper_bound': 0.218}
    },
    {
        'type': 'max calibration error - 10 train images', 
        'value': 0.236, 
        'confidence_interval': {'lower_bound': 0.118, 'upper_bound': 0.442}
    },
    {
        'type': 'accuracy - 100 train images', 
        'value': 0.821, 
        'confidence_interval': {'lower_bound': 0.693, 'upper_bound': 0.858},
        'threshold': 0.5
    },
    {
        'type': 'expected calibration error - 100 train images', 
        'value': 0.073, 
        'confidence_interval': {'lower_bound': 0.016, 'upper_bound': 0.202}
    },
    {
        'type': 'max calibration error after 100 images', 
        'value': 0.202, 
        'confidence_interval': {'lower_bound': 0.086, 'upper_bound': 0.379}
    }
]
model_card.quantitative_analysis.graphics.collection = [
    {
        'name': 'Confusion Matrix after 10 images (10 Runs)', 
        'image': img_to_base64('model_cards/data/confusion-matrix-10-one_image_label.png')
    },
    {
        'name': 'Confusion Matrix after 100 images (10 Runs)', 
        'image': img_to_base64('model_cards/data/confusion-matrix-100-one_image_label.png')
    }
]
model_card.quantitative_analysis.graphics.description = (
    'These graphics show confusion matrices of the model\'s predictions '
    '(averaged over 10 independent test sets) after training with 10 and 100 images respectively. ' 
)


# Write the model card data to a JSON file
mct.update_model_card_json(model_card)

# Return the model card document as an HTML page
html = mct.export_format()